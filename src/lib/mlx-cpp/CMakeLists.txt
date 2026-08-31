cmake_minimum_required(VERSION 3.16)

project(mlx-cpp LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
    "Emit the exact native compile command ledger" FORCE)

# ----------------------------- Fetch MLX C++ -----------------------------

include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

# Machine-readable single source of truth for the overlays actually applied by
# this configure.  mlxcel-core's build script consumes this file after CMake
# completes; it must not rediscover or reorder the patch set independently.
set(MLXCEL_OVERLAY_LEDGER "${CMAKE_BINARY_DIR}/mlxcel-active-overlays.tsv")
file(WRITE "${MLXCEL_OVERLAY_LEDGER}" "mlxcel-overlay-ledger-v1\n")
set_property(GLOBAL PROPERTY MLXCEL_OVERLAY_ORDINAL 0)

function(mlx_record_source_overlay source_label source_path destination_label destination_path)
  foreach(_ledger_value IN ITEMS "${source_label}" "${destination_label}")
    if(_ledger_value MATCHES "[\t\r\n]")
      message(FATAL_ERROR "Overlay ledger label contains a control character")
    endif()
  endforeach()
  file(SHA256 "${source_path}" _source_sha256)
  file(SHA256 "${destination_path}" _destination_sha256)
  if(NOT _source_sha256 STREQUAL _destination_sha256)
    message(FATAL_ERROR "Overlay destination differs from source: ${destination_label}")
  endif()
  get_property(_ordinal GLOBAL PROPERTY MLXCEL_OVERLAY_ORDINAL)
  file(APPEND "${MLXCEL_OVERLAY_LEDGER}"
       "${_ordinal}\t${source_label}\t${destination_label}\t${_source_sha256}\n")
  math(EXPR _next_ordinal "${_ordinal} + 1")
  set_property(GLOBAL PROPERTY MLXCEL_OVERLAY_ORDINAL "${_next_ordinal}")
endfunction()

function(mlx_apply_source_overlays mlx_source_dir)
  # Metal patches: JIT type promotion fix + kernel utils (SIMD group matrix).
  #
  # The SDPA / steel-attention overlays (scaled_dot_product_attention.cpp,
  # scaled_dot_product_attention.metal, steel_attention.metal,
  # steel_attention_nax.metal) were dropped when the MLX pin moved to
  # v0.31.2 — that release already carries the NAX fixes we had been
  # carrying locally (`#3361 int16 overflow in SDPA NAX mask indexing`,
  # `#3422 Speed up NAX split-K`, `#3419 Segmented mm nax kernel`). Removing
  # the patches keeps our build binary-equivalent to the upstream Python
  # wheel and simplifies future upstream syncs.
  # The steel/gemm/mma.h overlay (issue #217: upstream safe-load fix
  # cherry-pick, PRs #3560/#3565) was retired when the pin moved to
  # 2026-06-11 upstream main, which carries the fix natively (issue #222).
  # quantized.cpp carries one delta, the MLXCEL_QMV_WIDE off-switch on
  # `use_qmv_wide` (issue #1187). Upstream has no equivalent knob and the
  # predicate is unchanged as of pin 9a795735, so it has to be overlaid here.
  # Refresh the file wholesale on a bump and re-apply that one hunk.
  set(_metal_patch_files
      "mlx/backend/metal/compiled.cpp"
      "mlx/backend/metal/kernels/utils.h"
      "mlx/backend/metal/quantized.cpp")
  foreach(_patch_file ${_metal_patch_files})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}")
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
      mlx_record_source_overlay(
        "src/lib/mlx-cpp/patches/${_patch_file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${_patch_file}"
        "${mlx_source_dir}/${_patch_file}")
    endif()
  endforeach()
  message(STATUS "Applied Metal kernel patches")

  # CUDA patches: custom grouped GEMM, quantized kernels, reduce ops, binary ops.
  # These are in patches/mlx/backend/cuda/ and only compiled for CUDA targets,
  # so they are safe to apply unconditionally (harmless on Metal-only builds).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda")
    file(GLOB_RECURSE _cuda_patch_files
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/patches"
         "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda/*")
    list(SORT _cuda_patch_files)
    foreach(_patch_file ${_cuda_patch_files})
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
      mlx_record_source_overlay(
        "src/lib/mlx-cpp/patches/${_patch_file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${_patch_file}"
        "${mlx_source_dir}/${_patch_file}")
    endforeach()
    message(STATUS "Applied CUDA backend patches (${_cuda_patch_files})")
  endif()

  # CUDA-only patches: bf16 type promotion and norm computation changes.
  # These change dtype promotion rules (bf16+fp32→bf16) to avoid copy_v kernel
  # overhead on CUDA, but break Metal precision for bf16 vision weights
  # (e.g., SigLIP position embeddings). Apply ONLY for CUDA builds.
  if(MLX_BUILD_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda")
    file(GLOB _cuda_only_patches
         "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda/*.cpp")
    list(SORT _cuda_only_patches)
    foreach(_patch_file ${_cuda_only_patches})
      get_filename_component(_fname "${_patch_file}" NAME)
      configure_file(
        "${_patch_file}"
        "${mlx_source_dir}/mlx/${_fname}"
        COPYONLY)
      mlx_record_source_overlay(
        "src/lib/mlx-cpp/patches-cuda/${_fname}"
        "${_patch_file}"
        "mlx/${_fname}"
        "${mlx_source_dir}/mlx/${_fname}")
    endforeach()
    message(STATUS "Applied CUDA-only dtype/precision patches")
  endif()
endfunction()

set(_mlx_cached_source_dir "${CMAKE_BINARY_DIR}/_deps/mlx-src")
set(_mlx_cached_binary_dir "${CMAKE_BINARY_DIR}/_deps/mlx-build")

# Reuse an already-populated source tree when present so offline rebuilds can
# proceed without re-running FetchContent. This is especially useful when Cargo
# changes the build hash but the developer already has a cached MLX checkout.
# NOTE: Stale-cache validation is handled by build.rs (purge_stale_mlx_cache)
# which runs *before* CMake, ensuring _deps/ is either valid or absent, and
# build.rs re-checks the fetched checkout's HEAD afterwards so a tree seeded
# out of band cannot slip through this reuse branch either.
if(EXISTS "${_mlx_cached_source_dir}/CMakeLists.txt")
  set(mlx_SOURCE_DIR "${_mlx_cached_source_dir}")
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  message(STATUS "Using cached MLX source from ${mlx_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    # SINGLE SOURCE OF TRUTH for the pinned MLX commit, issue #1047. No other
    # file stores this value. Two parsers read this exact line, both scoped to
    # this declaration by the GIT_REPOSITORY URL just above it:
    #   src/lib/mlxcel-core/build_support/mlx_pin.rs, used by mlxcel-core's
    #     build.rs for cache purging, the fetched-HEAD check and the
    #     MLXCEL_MLX_COMMIT env baked into the binary
    #   scripts/ci/mlx_pinned_commit.sh, used by release.yml
    # The awk scan in mlx_pinned_commit.sh is the stricter of the two parsers,
    # so keeping its shape satisfies both:
    #   * GIT_TAG must be the first token on its line (leading whitespace is
    #     fine; nothing else may precede it), followed by a bare or quoted
    #     40-character lowercase hex sha, e.g. `GIT_TAG 2c46b95...` or
    #     `GIT_TAG "2c46b95..."`.
    #   * That line must come after the GIT_REPOSITORY line above naming
    #     ml-explore/mlx, inside this same FetchContent_Declare(...) call.
    #   * No line containing a closing paren `)` may appear between the
    #     GIT_REPOSITORY line and the GIT_TAG line; the awk scan treats any
    #     `)` as ending the declaration, so one there would hide the tag
    #     instead of being read past.
    # Both parsers fail loudly rather than guess when the shape does not hold.
    GIT_TAG 9a795735ad9a42664e08f42361b405ed570bcf1a)

  # Use FetchContent_Populate + add_subdirectory so we can apply source
  # overlays before the MLX build system processes the files.
  FetchContent_GetProperties(mlx)
  if(NOT mlx_POPULATED)
    FetchContent_Populate(mlx)
  endif()
endif()

mlx_apply_source_overlays("${mlx_SOURCE_DIR}")

# Sanity-check: detect deprecated MLX Device API usage in patch files that
# would cause build failures after an upstream MLX version bump.
set(_deprecated_patterns
    "d\\.get_command_encoder("
    "d\\.new_command_buffer(")
foreach(_patch_file ${_metal_patch_files_global})
  # _metal_patch_files is function-local; re-glob here
endforeach()
file(GLOB_RECURSE _all_patches
     "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/metal/*.cpp")
foreach(_pf ${_all_patches})
  file(READ "${_pf}" _pf_content)
  foreach(_pat ${_deprecated_patterns})
    string(FIND "${_pf_content}" "${_pat}" _found)
    if(NOT _found EQUAL -1)
      message(WARNING "Patch file ${_pf} uses deprecated API '${_pat}' — "
                      "update to match upstream MLX at GIT_TAG commit.")
    endif()
  endforeach()
endforeach()

add_subdirectory(${mlx_SOURCE_DIR} ${mlx_BINARY_DIR})

# Machine-readable ledger of the nested FetchContent sources that this exact
# configure activated. Rust validates each physical tree against the strict
# pre-acquisition manifest; this file records selection only and never invents
# URL/revision provenance.
set(MLXCEL_NATIVE_DEPENDENCY_LEDGER
    "${CMAKE_BINARY_DIR}/mlxcel-active-native-dependencies.tsv")
file(WRITE "${MLXCEL_NATIVE_DEPENDENCY_LEDGER}"
     "mlxcel-native-dependency-ledger-v1\n")
foreach(_dependency IN ITEMS
        mlx metal_cpp dlfcn-win32 openblas json fmt gguflib cccl nvtx3 cudnn
        cutlass jaccl)
  string(TOLOWER "${_dependency}" _dependency_prefix)
  set(_source_variable "${_dependency_prefix}_SOURCE_DIR")
  if(DEFINED ${_source_variable}
     AND IS_DIRECTORY "${${_source_variable}}")
    foreach(_ledger_value IN ITEMS "${_dependency}" "${${_source_variable}}")
      if(_ledger_value MATCHES "[\t\r\n]")
        message(FATAL_ERROR
                "Native dependency ledger contains a control character")
      endif()
    endforeach()
    file(APPEND "${MLXCEL_NATIVE_DEPENDENCY_LEDGER}"
         "${_dependency}\t${${_source_variable}}\n")
  endif()
endforeach()

# ----------------------------- CUDA CCCL ---------------------------------
# Ensure MLX_CCCL_DIR is defined for CUDA JIT kernel compilation at runtime.
# MLX only sets this for test builds, but statically linked binaries need it too.
# Since upstream e9463bb the macro is consumed by cccl_dir() in dirs.cpp, which
# is compiled in the dedicated mlx_dirs OBJECT target (upstream isolates the
# dynamic defines there to avoid invalidating the compile cache), so define it
# on mlx_dirs when that target exists and fall back to mlx otherwise.
if(MLX_BUILD_CUDA)
  set(_cccl_include "${CMAKE_BINARY_DIR}/_deps/cccl-src/include")
  if(EXISTS "${_cccl_include}")
    if(TARGET mlx_dirs)
      target_compile_definitions(mlx_dirs
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    else()
      target_compile_definitions(mlx
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    endif()
  endif()

  # CUTLASS/CuTe headers for the runtime-JIT'd quantized kernels (MLX #3706/
  # #3576 moved qmm_sm80/sm90/naive and gather_gemm to NVRTC JIT compilation,
  # and those kernels include <cute/...> / <cutlass/...> at runtime). Upstream
  # only installs the headers next to installed binaries; statically linked
  # mlxcel binaries need a compiled-in fallback for dev builds. Consumed by
  # our jit_module.cpp overlay, which compiles into the mlx target.
  set(_cutlass_include "${CMAKE_BINARY_DIR}/_deps/cutlass-src/include")
  if(EXISTS "${_cutlass_include}")
    target_compile_definitions(mlx
                               PRIVATE MLX_CUTLASS_DIR="${_cutlass_include}")
  endif()
endif()

# Record the target graph that this configure actually selected.  The build
# script role-tokenizes every path before the ledger enters the immutable
# identity, so this producer deliberately records CMake's ordered target
# properties instead of guessing the graph again in Rust.
function(mlxcel_collect_build_targets directory output_variable)
  get_property(_local_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(_child_directories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  set(_all_targets ${_local_targets})
  foreach(_child IN LISTS _child_directories)
    mlxcel_collect_build_targets("${_child}" _child_targets)
    list(APPEND _all_targets ${_child_targets})
  endforeach()
  set(${output_variable} "${_all_targets}" PARENT_SCOPE)
endfunction()

set(MLXCEL_TARGET_GRAPH_LEDGER
    "${CMAKE_BINARY_DIR}/mlxcel-active-target-graph.tsv")
file(WRITE "${MLXCEL_TARGET_GRAPH_LEDGER}" "mlxcel-target-graph-v1\n")
mlxcel_collect_build_targets("${CMAKE_CURRENT_SOURCE_DIR}" _mlxcel_build_targets)
list(REMOVE_DUPLICATES _mlxcel_build_targets)
list(SORT _mlxcel_build_targets)
set(_mlxcel_target_properties
    SOURCE_DIR
    BINARY_DIR
    LINK_LIBRARIES
    INTERFACE_LINK_LIBRARIES
    INCLUDE_DIRECTORIES
    INTERFACE_INCLUDE_DIRECTORIES
    COMPILE_DEFINITIONS
    INTERFACE_COMPILE_DEFINITIONS
    COMPILE_OPTIONS
    INTERFACE_COMPILE_OPTIONS
    LINK_OPTIONS
    INTERFACE_LINK_OPTIONS
    LINK_DEPENDS)
foreach(_target IN LISTS _mlxcel_build_targets)
  get_target_property(_target_type "${_target}" TYPE)
  if(NOT _target_type)
    message(FATAL_ERROR "Target ${_target} has no CMake TYPE")
  endif()
  foreach(_ledger_value IN ITEMS "${_target}" "${_target_type}")
    if(_ledger_value MATCHES "[\t\r\n]")
      message(FATAL_ERROR "Target graph ledger contains a control character")
    endif()
  endforeach()
  file(APPEND "${MLXCEL_TARGET_GRAPH_LEDGER}"
       "${_target}\t${_target_type}\tTYPE\t0\t${_target_type}\n")
  foreach(_property IN LISTS _mlxcel_target_properties)
    get_target_property(_property_values "${_target}" "${_property}")
    if(_property_values AND NOT _property_values MATCHES "-NOTFOUND$")
      if(_property_values MATCHES "[\t\r\n]")
        message(FATAL_ERROR
                "Target graph value contains a control character")
      endif()
      # Keep CMake's complete semicolon-delimited property in one record.
      # Splitting it here would tear generator expressions such as
      # `$<$<COMPILE_LANGUAGE:CXX>:/wd4244;/wd4267>` into invalid fragments.
      file(APPEND "${MLXCEL_TARGET_GRAPH_LEDGER}"
           "${_target}\t${_target_type}\t${_property}\t0\t${_property_values}\n")
    endif()
  endforeach()
endforeach()
