// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Build script for mlx-cxx
// This builds the MLX C++ library and the cxx bridge

use cmake::Config;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
use url::Url;

const CORE_CXX_SOURCES: &[&str] = &[
    "cpp/mlx_cxx_bridge.cpp",
    // Bridge implementation split out of mlx_cxx_bridge.cpp by domain
    // (shared helpers in cpp/mlx_cxx_internal.h): fused decode Metal
    // kernels, the NemotronH full-forward path, and safetensors loading +
    // Metal 4 / turbo / paged attention launchers.
    "cpp/mlx_cxx_kernels.cpp",
    "cpp/mlx_cxx_nemotron.cpp",
    "cpp/mlx_cxx_ext.cpp",
    // Fused Sparse-V SDPA kernel launcher. Lives under
    // `src/lib/mlx-cpp/turbo/` so the MLX-upstream-commit upgrade
    // checklist ("Bumping the MLX upstream pin" in CONTRIBUTING.md)
    // treats this directory as in-scope.
    "../mlx-cpp/turbo/sparse_v_sdpa.cpp",
    // Fused Turbo4Delegated cold-V weighted-sum kernel
    // launcher. Reads the packed cold V directly so the dequantised
    // FP16 cold body never materialises in global memory; the host
    // pairs this with a hot-V matmul to produce the final SDPA output.
    "../mlx-cpp/turbo/turbo4_delegated_sdpa.cpp",
    // Fused paged-attention decode kernel launcher (epic #116 Phase 6,
    // #123). Reads scattered KV blocks out of the global pool via a block
    // table with no separate gather copy; the gather-then-SDPA path stays
    // the correctness reference and fallback.
    "../mlx-cpp/turbo/paged_attention.cpp",
    // Paged-attention decode v2 (#898): CSR page table plus cross-CTA
    // split-KV in `paged_attention_v2.cpp`, and the variable-length
    // attention-state merge kernel in `paged_attention_v2_merge.cpp`. The
    // merge half is split out because the cascade issue #903 reuses it
    // unchanged. v1 above stays intact and remains the default path.
    "../mlx-cpp/turbo/paged_attention_v2.cpp",
    "../mlx-cpp/turbo/paged_attention_v2_merge.cpp",
    // Softmax-free Gumbel-max categorical sampling kernel launcher (#900).
    // Replaces the `random::categorical` normalization pass plus the
    // `argpartition` sort on the no-filter sampling path with one
    // index-carrying max reduction over the vocabulary.
    "../mlx-cpp/turbo/sampling.cpp",
    // Sorting-free top-k / top-p / min-p sampling by dual-pivot rejection
    // (#901). Replaces the `argpartition` + `argsort` + `cumsum` filter
    // chain with a shrinking probability interval resolved by two pivots
    // per vocabulary sweep.
    "../mlx-cpp/turbo/sampling_rejection.cpp",
    // Fused residual-add + RMSNorm and fused q/k RoPE + KV-append-layout
    // decode kernel launchers (#905). Both are two/three-output custom
    // kernels: MLX arrays are immutable from the graph's perspective, so
    // the in-place residual update and the append payload are returned as
    // values rather than written through.
    "../mlx-cpp/turbo/fused_norm.cpp",
    "../mlx-cpp/turbo/fused_rope_append.cpp",
];

const CORE_CXX_INCLUDES: &[&str] = &["cpp", "../mlx-cpp/turbo"];
const CORE_CXX_OPTIONAL_FLAGS: &[&str] = &[
    "-Wno-unused-parameter",
    "-Wno-deprecated-declarations",
    // MLX v0.31.0 triggers a Clang deprecated-copy warning from bf16.h when
    // included by the generated cxx bridge, so suppress it for all profiles.
    "-Wno-deprecated-copy",
];

// Single-source-of-truth resolution and verification of the pinned MLX commit.
// Shared by path (not by dependency) with `mlxcel-mlx-pin`, which unit-tests it
// without dragging in an MLX build; see that crate's manifest for the reason.
#[path = "build_support/mlx_pin.rs"]
mod mlx_pin;

#[path = "../../../build_support/native_identity.rs"]
mod native_identity;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    enforce_strict_fresh_output(&out_dir);

    // The pinned MLX commit has exactly one home: GIT_TAG in
    // ../mlx-cpp/CMakeLists.txt, which is what CMake actually fetches. Resolve
    // it once, here, before anything compares against it. Everything below
    // (the _deps/ purge decision, the MLXCEL_MLX_COMMIT export, the post-build
    // HEAD verification and the cache marker) consumes this one value, so they
    // cannot disagree with each other or with the tag that was fetched (#1047).
    let mlx_commit = match mlx_pin::read_pinned_commit(&manifest_dir) {
        Ok(commit) => commit,
        Err(err) => panic!("mlxcel-core: {err}"),
    };

    // Expose the pinned MLX commit to the crate so the runtime can scope the
    // persistent CUDA PTX cache directory by it (see ensure_persistent_ptx_cache).
    println!("cargo:rustc-env=MLXCEL_MLX_COMMIT={mlx_commit}");

    // Build MLX using cmake
    let mlx_dst = build_mlx(&mlx_commit);
    // Verify what actually landed on disk before blessing it. CMake reuses an
    // already-populated _deps/mlx-src rather than re-running FetchContent, so a
    // checkout restored from a CI cache or seeded by hand can disagree with the
    // pin without anything upstream of here noticing.
    let fetched_mlx_commit = verify_fetched_mlx_head(&mlx_dst, &mlx_commit);
    mark_mlx_cache_valid(&out_dir, &mlx_commit);
    let mlx_include = mlx_dst.join("build/include");
    let mlx_lib = mlx_dst.join("build/lib");

    // Build the cxx bridge with optimization flags
    let mut bridge = cxx_build::bridge("src/lib.rs");
    for source in CORE_CXX_SOURCES {
        bridge.file(source);
    }
    bridge.include(&mlx_include);
    for include in CORE_CXX_INCLUDES {
        bridge.include(include);
    }
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        bridge.flag("/std:c++20");
    } else {
        bridge.flag_if_supported("-std=c++20");
    }
    for flag in CORE_CXX_OPTIONAL_FLAGS {
        bridge.flag_if_supported(flag);
    }

    // The qmv_wide off-switch (issue #1187) lives in the
    // mlx/backend/metal/quantized.cpp overlay, so its symbol exists only when
    // the Metal backend is actually compiled. `__APPLE__` is not that
    // condition: a macOS build without the `metal` feature sets
    // MLX_BUILD_METAL=OFF and would link against a symbol that was never
    // emitted. Gate on the feature that decides whether the file is built.
    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        bridge.define("MLXCEL_BRIDGE_METAL_BACKEND", None);
    }

    // Add optimization flags for release builds
    #[cfg(not(debug_assertions))]
    {
        bridge
            .flag_if_supported("-O3")
            .flag_if_supported("-DNDEBUG")
            .flag_if_supported("-ffast-math");
        // ISA baseline for the bridge C++. Defaults to the build host's ISA
        // (-march=native), which is correct for builds that run where they
        // are built (developer machines, the per-machine gb10/gh200 release
        // assets). Redistributable release builds must override this with a
        // portable baseline via MLXCEL_CXX_MARCH (e.g. "x86-64-v3" for the
        // generic Linux x86-64 asset), otherwise the binary inherits the
        // build runner's ISA (possibly AVX-512) and SIGILLs on older CPUs.
        // Set MLXCEL_CXX_MARCH=none to omit the flag entirely.
        match env::var("MLXCEL_CXX_MARCH").as_deref() {
            Err(_) => {
                bridge.flag_if_supported("-march=native");
            }
            Ok("none") => {}
            Ok(march) => {
                bridge.flag_if_supported(format!("-march={march}"));
            }
        }
        // On macOS, Clang produces LLVM bitcode with -flto, which is compatible
        // with Rust's LLVM LTO. On Linux with GCC, -flto produces GIMPLE IR
        // objects that are incompatible, causing undefined-reference linker errors.
        #[cfg(target_os = "macos")]
        bridge.flag_if_supported("-flto");
    }

    // Resolve the exact compiler plus the supported flag set from the same
    // configured cc::Build that performs the bridge compilation.
    let bridge_tool = bridge
        .try_get_compiler()
        .unwrap_or_else(|err| panic!("mlxcel-core: failed to resolve CXX compiler: {err}"));
    let bridge_compile_flags = bridge_tool
        .args()
        .iter()
        .map(|arg| strict_os_text(arg, "CXX compiler argument"))
        .collect::<Vec<_>>();
    let bridge_cxx_identity = tool_identity(bridge_tool.path());
    bridge.compile("mlx_cxx_bridge");

    // Link against MLX
    let mut rustc_link_contract = Vec::new();
    emit_link_search(&mut rustc_link_contract, &mlx_lib, "<mlx-lib>");
    emit_link_lib(&mut rustc_link_contract, "static=mlx");

    // Platform-specific system libraries
    #[cfg(target_os = "macos")]
    {
        emit_link_lib(&mut rustc_link_contract, "c++");
        emit_link_lib(&mut rustc_link_contract, "dylib=objc");
        emit_link_lib(&mut rustc_link_contract, "framework=Foundation");

        // Link clang compiler-rt for ___isPlatformVersionAtLeast
        // (required by MLX C++ @available() runtime checks)
        let output = Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
            .unwrap_or_else(|err| panic!("mlxcel-core: cannot resolve clang runtime: {err}"));
        if !output.status.success() {
            panic!("mlxcel-core: clang --print-runtime-dir failed");
        }
        let runtime_dir = String::from_utf8(output.stdout)
            .unwrap_or_else(|_| panic!("mlxcel-core: clang runtime path is not strict UTF-8"));
        let runtime_dir = runtime_dir.trim();
        if runtime_dir.is_empty() {
            panic!("mlxcel-core: clang returned an empty runtime directory");
        }
        emit_link_search(
            &mut rustc_link_contract,
            Path::new(runtime_dir),
            "<clang-runtime>",
        );
        emit_link_lib(&mut rustc_link_contract, "static=clang_rt.osx");
    }

    #[cfg(target_os = "linux")]
    {
        emit_link_lib(&mut rustc_link_contract, "stdc++");
        // CPU backend needs BLAS/LAPACK on Linux
        emit_link_lib(&mut rustc_link_contract, "dylib=openblas");
        emit_link_lib(&mut rustc_link_contract, "dylib=lapack");
    }

    // Backend-specific linking
    #[cfg(target_os = "macos")]
    {
        // Metal and Accelerate are always linked on macOS
        emit_link_lib(&mut rustc_link_contract, "framework=Metal");
        emit_link_lib(&mut rustc_link_contract, "framework=Accelerate");
    }

    #[cfg(feature = "cuda")]
    {
        link_cuda(&mut rustc_link_contract);
    }

    // Rerun if bridge files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_internal.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_kernels.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_nemotron.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_ext.cpp");
    println!("cargo:rerun-if-changed=metal/fused_attention_metal4.metal");
    println!("cargo:rerun-if-changed=../mlx-cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches-cuda");
    // Sparse-V fused-skip Metal kernel launchers.
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.metal");
    // Turbo4Delegated cold-V fused weighted-sum kernel launcher.
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/turbo4_delegated_sdpa.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/turbo4_delegated_sdpa.cpp");
    // Fused paged-attention decode kernel launcher (#123).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.metal");

    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2_merge.cpp");
    // Gumbel-max categorical sampling kernel launcher (#900).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling_rejection.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling_rejection.cpp");
    // Fused residual-add RMSNorm and fused RoPE + KV-append kernel launchers (#905).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_norm.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_norm.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_rope_append.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_rope_append.cpp");
    println!("cargo:rerun-if-env-changed=MLX_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_METAL");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_ACCELERATE");
    println!("cargo:rerun-if-env-changed=MLXCEL_CXX_MARCH");

    write_native_core_build_record(
        &manifest_dir,
        &out_dir,
        &mlx_dst,
        &fetched_mlx_commit,
        &bridge_cxx_identity,
        bridge_tool.path(),
        bridge_compile_flags,
        rustc_link_contract,
    );
}

fn enforce_strict_fresh_output(out_dir: &Path) {
    if env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() != Ok("1") {
        return;
    }
    if env::var("MLXCEL_BUILD_IDENTITY_FRESH").as_deref() != Ok("1") {
        panic!(
            "mlxcel-core: strict build identity requires MLXCEL_BUILD_IDENTITY_FRESH=1 and a fresh target directory"
        );
    }
    let marker = out_dir.join(".mlxcel-strict-build-started");
    if marker.exists() {
        panic!("mlxcel-core: strict build identity refuses a reused build-script output directory");
    }
    fs::write(&marker, b"mlxcel-strict-build-v1\n")
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot create strict build marker: {err}"));
}

/// Purge stale cached MLX build artifacts before CMake runs.
///
/// CI caches may restore `_deps/` from a previous build. Even when the git
/// source checkout is correct, stale CMake build artifacts (object files in
/// `_deps/mlx-build/`) can cause compilation to succeed using outdated `.o`
/// files because make skips recompilation when timestamps look current.
///
/// Instead of fragile git-based validation, we use a simple marker file:
/// after a successful build, `_deps/.mlx-build-commit` records the commit.
/// If the marker is missing or doesn't match, we purge the entire `_deps/`.
///
/// `expected_commit` is the value resolved from `../mlx-cpp/CMakeLists.txt` in
/// `main`, so this decision is made against the tag CMake would actually fetch.
fn purge_stale_mlx_cache(out_dir: &std::path::Path, expected_commit: &str) {
    let deps_dir = out_dir.join("build/_deps");
    if !deps_dir.exists() {
        return;
    }

    let marker = deps_dir.join(".mlx-build-commit");
    let marker_contents = std::fs::read_to_string(&marker).ok();

    match mlx_pin::cache_state(marker_contents.as_deref(), expected_commit) {
        mlx_pin::CacheState::Valid => {}
        mlx_pin::CacheState::Stale { cached } => {
            eprintln!(
                "mlxcel-core: MLX build cache stale (cached={}, expected={expected_commit}), purging _deps/",
                cached.as_deref().unwrap_or("none")
            );
            let _ = std::fs::remove_dir_all(&deps_dir);
        }
    }
}

/// Fail the build when the MLX checkout CMake used is not the pinned commit.
///
/// Runs after the CMake build returns and before the cache marker or generated
/// native build record is written, so an unavailable or mismatched actual HEAD
/// is never blessed as valid. This build fetches MLX from Git; a vendored tree
/// therefore cannot silently stand in for the pinned fetched-source authority.
fn verify_fetched_mlx_head(mlx_dst: &Path, expected_commit: &str) -> String {
    let mlx_src = active_native_dependency_paths(mlx_dst)
        .remove("mlx")
        .unwrap_or_else(|| panic!("mlxcel-core: CMake did not record active MLX source"));
    match mlx_pin::check_fetched_head(&mlx_src, expected_commit) {
        mlx_pin::HeadCheck::Match => {
            let output = git_command(&mlx_src)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap_or_else(|err| {
                    panic!(
                        "mlxcel-core: cannot read the fetched MLX HEAD for build identity: {err}"
                    )
                });
            if !output.status.success() {
                panic!(
                    "mlxcel-core: fetched MLX HEAD is unavailable for the native build identity"
                );
            }
            let found = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| panic!("mlxcel-core: fetched MLX HEAD is not strict UTF-8"))
                .trim()
                .to_owned();
            if found != expected_commit {
                panic!(
                    "mlxcel-core: fetched MLX HEAD changed after verification (found {found}, expected {expected_commit})"
                );
            }
            found
        }
        mlx_pin::HeadCheck::Unavailable { reason } => {
            panic!(
                "mlxcel-core: fetched MLX HEAD is required for the native build identity ({reason})"
            );
        }
        mlx_pin::HeadCheck::Mismatch { found } => {
            panic!(
                "mlxcel-core: {}",
                mlx_pin::head_mismatch_message(&mlx_src, expected_commit, &found)
            );
        }
    }
}

/// Write a marker after successful MLX build so future runs can validate the cache.
fn mark_mlx_cache_valid(out_dir: &std::path::Path, expected_commit: &str) {
    let marker = out_dir.join("build/_deps/.mlx-build-commit");
    let _ = std::fs::write(marker, expected_commit);
}

fn build_mlx(expected_commit: &str) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    purge_stale_mlx_cache(&out_dir, expected_commit);

    let mut config = Config::new("../mlx-cpp");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON");
    configure_preacquired_native_sources(&mut config);

    // Platform features
    // On macOS: Metal and Accelerate are always available and enabled by default.
    // Feature flags can still override (e.g. for CPU-only testing).
    // On Linux: CPU-only by default, CUDA opt-in via feature flag.
    config.define("MLX_BUILD_CUDA", "OFF");

    #[cfg(target_os = "macos")]
    {
        let build_metal = cmake_bool_from_env("MLXCEL_BUILD_METAL").unwrap_or("ON");
        let build_accelerate = cmake_bool_from_env("MLXCEL_BUILD_ACCELERATE").unwrap_or("ON");

        // Default to Metal + Accelerate on macOS, but allow CPU-only rebuilds
        // for environments where Metal device enumeration is unavailable.
        config.define("MLX_BUILD_METAL", build_metal);
        config.define("MLX_BUILD_ACCELERATE", build_accelerate);
    }

    #[cfg(not(target_os = "macos"))]
    {
        config.define("MLX_BUILD_METAL", "OFF");
        config.define("MLX_BUILD_ACCELERATE", "OFF");
    }

    #[cfg(feature = "cuda")]
    {
        config.define("MLX_BUILD_CUDA", "ON");

        // Help CMake find CUDA toolkit
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            config.define("CMAKE_CUDA_COMPILER", format!("{cuda_path}/bin/nvcc"));
        } else if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
            config.define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc");
        }

        // CUDA architecture selection. An explicitly set MLX_CUDA_ARCHITECTURES is
        // honored verbatim (escape hatch); otherwise we auto-detect via nvidia-smi
        // and fall back to Hopper's sm_90a.
        //
        // The `a` suffix is load-bearing: MLX only defines MLX_CUDA_SM90A_ENABLED
        // (which compiles the dedicated Hopper `qmm_sm90` quantized kernel) when
        // "90a" is in the arch list. MLX's own CMake appends that suffix for
        // cc >= 90, but only inside its `if(NOT DEFINED MLX_CUDA_ARCHITECTURES)`
        // branch. Because we always pass MLX_CUDA_ARCHITECTURES explicitly, that
        // branch never runs, so we apply the same rule ourselves here and in
        // detect_cuda_arch. See docs/installation.md (CUDA architecture selection).
        let cuda_arch = env::var("MLX_CUDA_ARCHITECTURES")
            .unwrap_or_else(|_| detect_cuda_arch().unwrap_or_else(|| "90a".to_string()));
        config.define("MLX_CUDA_ARCHITECTURES", &cuda_arch);
    }

    config.build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreacquiredSourceKind {
    Git,
    Archive,
}

#[derive(Debug, Clone)]
struct PreacquiredNativeSource {
    name: String,
    kind: PreacquiredSourceKind,
    canonical_url: String,
    revision: Option<String>,
    source_dir: PathBuf,
    tree_sha256: [u8; 32],
    archive_path: Option<PathBuf>,
    archive_sha256: Option<[u8; 32]>,
}

struct CapturedNativeDependencies {
    named_inputs: Vec<CapturedNamedInput>,
    tree_inputs: Vec<CapturedSource>,
}

fn configure_preacquired_native_sources(config: &mut Config) {
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_IDENTITY_STRICT");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_IDENTITY_FRESH");
    println!("cargo:rerun-if-env-changed=MLXCEL_NATIVE_SOURCE_MANIFEST");
    let strict = env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() == Ok("1");
    let Some(path) = env::var_os("MLXCEL_NATIVE_SOURCE_MANIFEST").map(PathBuf::from) else {
        if strict {
            panic!("mlxcel-core: strict build identity requires MLXCEL_NATIVE_SOURCE_MANIFEST");
        }
        return;
    };
    let sources = read_preacquired_native_manifest(&path);
    if sources.is_empty() {
        panic!("mlxcel-core: pre-acquired native source manifest is empty");
    }
    config.define("FETCHCONTENT_FULLY_DISCONNECTED", "ON");
    for source in sources {
        let variable = format!(
            "FETCHCONTENT_SOURCE_DIR_{}",
            source.name.to_ascii_uppercase().replace('_', "-")
        );
        config.define(variable, &source.source_dir);
    }
}

fn read_preacquired_native_manifest(path: &Path) -> Vec<PreacquiredNativeSource> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: cannot read native source manifest {}: {err}",
            path.display()
        )
    });
    let manifest: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|err| panic!("mlxcel-core: native source manifest is invalid: {err}"));
    if manifest.get("schema").and_then(toml::Value::as_integer) != Some(1) {
        panic!("mlxcel-core: native source manifest has an unknown schema");
    }
    let entries = manifest
        .get("dependency")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("mlxcel-core: native source manifest has no dependency array"));
    if entries.len() > 64 {
        panic!("mlxcel-core: native source manifest exceeds 64 dependencies");
    }
    let mut names = std::collections::BTreeSet::new();
    entries
        .iter()
        .map(|entry| {
            let table = entry.as_table().unwrap_or_else(|| {
                panic!("mlxcel-core: native source dependency is not a TOML table")
            });
            let name = required_manifest_text(table, "name");
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|byte| {
                        byte.is_ascii_lowercase()
                            || byte.is_ascii_digit()
                            || matches!(byte, b'-' | b'_')
                    })
                || !names.insert(name.clone())
            {
                panic!("mlxcel-core: native source dependency name is invalid or duplicate");
            }
            let kind = match required_manifest_text(table, "kind").as_str() {
                "git" => PreacquiredSourceKind::Git,
                "archive" => PreacquiredSourceKind::Archive,
                other => panic!("mlxcel-core: unsupported native source kind {other:?}"),
            };
            let canonical_url = canonical_repository_url(&required_manifest_text(
                table,
                "canonical_url",
            ));
            let source_dir = fs::canonicalize(required_manifest_text(table, "source_dir"))
                .unwrap_or_else(|err| {
                    panic!("mlxcel-core: cannot resolve pre-acquired source {name}: {err}")
                });
            let tree_sha256 = parse_sha256(
                &required_manifest_text(table, "tree_sha256"),
                "native source tree",
            );
            let observed_tree = native_identity::directory_tree_digest(&source_dir)
                .unwrap_or_else(|err| panic!("mlxcel-core: cannot encode source {name}: {err}"));
            if observed_tree != tree_sha256 {
                panic!("mlxcel-core: pre-acquired source tree digest mismatch for {name}");
            }
            let revision = table
                .get("revision")
                .and_then(toml::Value::as_str)
                .map(str::to_owned);
            let archive_path = table
                .get("archive_path")
                .and_then(toml::Value::as_str)
                .map(PathBuf::from);
            let archive_sha256 = table
                .get("archive_sha256")
                .and_then(toml::Value::as_str)
                .map(|value| parse_sha256(value, "native source archive"));
            match kind {
                PreacquiredSourceKind::Git => {
                    let revision = revision.as_deref().unwrap_or_else(|| {
                        panic!("mlxcel-core: Git source {name} has no revision")
                    });
                    if !native_identity::is_full_lower_git_sha(revision)
                        || archive_path.is_some()
                        || archive_sha256.is_some()
                    {
                        panic!("mlxcel-core: Git source {name} has invalid provenance fields");
                    }
                    let actual = git_command(&source_dir)
                        .args(["rev-parse", "HEAD"])
                        .output()
                        .unwrap_or_else(|err| {
                            panic!("mlxcel-core: cannot inspect Git source {name}: {err}")
                        });
                    if !actual.status.success()
                        || String::from_utf8(actual.stdout)
                            .ok()
                            .is_none_or(|value| value.trim() != revision)
                    {
                        panic!("mlxcel-core: Git source {name} HEAD does not match its manifest");
                    }
                }
                PreacquiredSourceKind::Archive => {
                    if revision.is_some() || archive_path.is_none() || archive_sha256.is_none() {
                        panic!(
                            "mlxcel-core: archive source {name} requires archive path/digest and no revision"
                        );
                    }
                    let archive = fs::canonicalize(archive_path.as_ref().unwrap())
                        .unwrap_or_else(|err| {
                            panic!("mlxcel-core: cannot resolve archive for {name}: {err}")
                        });
                    if file_digest(&archive) != archive_sha256.unwrap() {
                        panic!("mlxcel-core: archive digest mismatch for {name}");
                    }
                }
            }
            PreacquiredNativeSource {
                name,
                kind,
                canonical_url,
                revision,
                source_dir,
                tree_sha256,
                archive_path,
                archive_sha256,
            }
        })
        .collect()
}

fn required_manifest_text(table: &toml::map::Map<String, toml::Value>, name: &str) -> String {
    let value = table
        .get(name)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("mlxcel-core: native source dependency has no {name}"));
    if value.is_empty() || value.len() > 64 * 1024 || value.contains(['\r', '\n', '\0']) {
        panic!("mlxcel-core: native source dependency {name} is invalid");
    }
    value.to_owned()
}

fn canonical_repository_url(value: &str) -> String {
    let mut parsed = Url::parse(value)
        .unwrap_or_else(|err| panic!("mlxcel-core: native source canonical URL is invalid: {err}"));
    if !matches!(parsed.scheme(), "https" | "ssh")
        || parsed.host_str().is_none()
        || parsed.username() != ""
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || matches!(parsed.path(), "" | "/")
    {
        panic!("mlxcel-core: native source canonical URL is invalid");
    }
    let path = parsed.path().trim_end_matches('/').to_owned();
    parsed.set_path(&path);
    let canonical = parsed.to_string();
    if canonical != value {
        panic!("mlxcel-core: native source canonical URL is not canonically encoded");
    }
    canonical
}

fn capture_active_native_dependencies(
    active: &std::collections::BTreeMap<String, PathBuf>,
) -> CapturedNativeDependencies {
    let strict_manifest = env::var_os("MLXCEL_NATIVE_SOURCE_MANIFEST")
        .map(PathBuf::from)
        .map(|path| read_preacquired_native_manifest(&path));
    let strict_by_name = strict_manifest
        .unwrap_or_default()
        .into_iter()
        .map(|source| (source.name.clone(), source))
        .collect::<std::collections::BTreeMap<_, _>>();
    let strict = env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() == Ok("1");
    let mut named_inputs = Vec::new();
    let mut tree_inputs = Vec::new();
    for (name, path) in active {
        let observed_tree = native_identity::directory_tree_digest(path)
            .unwrap_or_else(|err| panic!("mlxcel-core: cannot encode dependency {name}: {err}"));
        tree_inputs.push(CapturedSource {
            label: format!("native-dependency/{name}/tree"),
            sha256: observed_tree,
        });
        if let Some(source) = strict_by_name.get(name) {
            let source_dir = fs::canonicalize(&source.source_dir).unwrap_or_else(|err| {
                panic!("mlxcel-core: cannot resolve dependency {name}: {err}")
            });
            if &source_dir != path || source.tree_sha256 != observed_tree {
                panic!(
                    "mlxcel-core: active dependency {name} differs from the pre-acquisition manifest"
                );
            }
            let kind = match source.kind {
                PreacquiredSourceKind::Git => "git",
                PreacquiredSourceKind::Archive => "archive",
            };
            named_inputs.extend([
                CapturedNamedInput {
                    name: format!("native-dependency:{name}:kind"),
                    value: kind.to_owned(),
                },
                CapturedNamedInput {
                    name: format!("native-dependency:{name}:canonical-url"),
                    value: source.canonical_url.clone(),
                },
                CapturedNamedInput {
                    name: format!("native-dependency:{name}:tree-sha256"),
                    value: hex_digest(observed_tree),
                },
            ]);
            if let Some(revision) = &source.revision {
                named_inputs.push(CapturedNamedInput {
                    name: format!("native-dependency:{name}:revision"),
                    value: revision.clone(),
                });
            }
            if let Some(digest) = source.archive_sha256 {
                named_inputs.push(CapturedNamedInput {
                    name: format!("native-dependency:{name}:archive-sha256"),
                    value: hex_digest(digest),
                });
            }
            if let Some(archive) = &source.archive_path {
                let archive_name = archive
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_else(|| {
                        panic!("mlxcel-core: dependency {name} archive name is not UTF-8")
                    });
                named_inputs.push(CapturedNamedInput {
                    name: format!("native-dependency:{name}:archive-name"),
                    value: archive_name.to_owned(),
                });
            }
        } else if strict {
            panic!(
                "mlxcel-core: active dependency {name} is absent from the strict pre-acquisition manifest"
            );
        } else {
            named_inputs.push(CapturedNamedInput {
                name: format!("native-dependency:{name}:kind"),
                value: "observed-tree-unqualified".to_owned(),
            });
            named_inputs.push(CapturedNamedInput {
                name: format!("native-dependency:{name}:tree-sha256"),
                value: hex_digest(observed_tree),
            });
        }
    }
    if strict {
        for name in strict_by_name.keys() {
            if !active.contains_key(name) {
                named_inputs.push(CapturedNamedInput {
                    name: format!("native-dependency:{name}:inactive"),
                    value: "true".to_owned(),
                });
            }
        }
    }
    named_inputs.sort_by(|left, right| left.name.cmp(&right.name));
    tree_inputs.sort_by(|left, right| left.label.cmp(&right.label));
    CapturedNativeDependencies {
        named_inputs,
        tree_inputs,
    }
}

// Used by the macOS configuration block above; dead on non-macOS targets.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cmake_bool_from_env(name: &str) -> Option<&'static str> {
    let value = env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some("ON"),
        "0" | "off" | "false" | "no" => Some("OFF"),
        _ => panic!(
            "Invalid {name} value {:?}. Expected one of: 1/0, on/off, true/false, yes/no.",
            value
        ),
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> Option<String> {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let caps = String::from_utf8_lossy(&output.stdout);
    // Parse "X.Y" compute capabilities, convert to SM number (e.g. "9.0" -> "90"),
    // and append the architecture-specific "a" suffix for cc >= 90 (e.g. "90a").
    let archs: Vec<String> = caps
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let parts: Vec<&str> = line.split('.').collect();
            if parts.len() == 2 {
                Some(sm_arch_with_suffix(&format!("{}{}", parts[0], parts[1])))
            } else {
                None
            }
        })
        .collect();
    if archs.is_empty() {
        None
    } else {
        // Deduplicate
        let mut unique = archs;
        unique.sort();
        unique.dedup();
        Some(unique.join(";"))
    }
}

/// Append CUDA's architecture-specific `a` suffix for SM >= 90, mirroring MLX's
/// own CMake logic (`MLX_CUDA_ARCHITECTURES GREATER_EQUAL 90` -> append `a`).
///
/// The `a` suffix enables architecture-specific features (e.g. Hopper wgmma/TMA)
/// the dedicated quantized kernels rely on, and it gates MLX_CUDA_SM90A_ENABLED on
/// "90a" (not "90"). SM < 90 (e.g. Ampere sm_80/sm_86) has no `a` variant and is
/// returned unchanged.
#[cfg(feature = "cuda")]
fn sm_arch_with_suffix(sm: &str) -> String {
    match sm.parse::<u32>() {
        Ok(n) if n >= 90 => format!("{sm}a"),
        _ => sm.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn link_cuda(rustc_link_contract: &mut Vec<String>) {
    // Find CUDA lib directory
    let cuda_lib = if let Ok(cuda_home) = env::var("CUDA_HOME") {
        PathBuf::from(cuda_home).join("lib64")
    } else if PathBuf::from("/usr/local/cuda/lib64").exists() {
        PathBuf::from("/usr/local/cuda/lib64")
    } else {
        panic!("Cannot find CUDA library directory. Set CUDA_HOME environment variable.");
    };

    emit_link_search(rustc_link_contract, &cuda_lib, "<cuda-lib>");

    // CUDA runtime and math libraries
    emit_link_lib(rustc_link_contract, "dylib=cudart");
    emit_link_lib(rustc_link_contract, "dylib=cublas");
    emit_link_lib(rustc_link_contract, "dylib=cublasLt");
    emit_link_lib(rustc_link_contract, "dylib=cufft");

    // CUDA driver API (cuLaunchKernel, cuModuleLoad, etc.)
    emit_link_lib(rustc_link_contract, "dylib=cuda");

    // cuDNN
    emit_link_lib(rustc_link_contract, "dylib=cudnn");

    // NVRTC for runtime compilation (JIT kernels)
    emit_link_lib(rustc_link_contract, "dylib=nvrtc");

    // CUDA stubs directory (for driver API on systems without GPU driver in lib path)
    let cuda_stubs = cuda_lib.join("stubs");
    if cuda_stubs.exists() {
        emit_link_search(rustc_link_contract, &cuda_stubs, "<cuda-stubs>");
    }
}

fn emit_link_search(contract: &mut Vec<String>, path: &Path, canonical_label: &str) {
    let path = strict_path_text(path);
    println!("cargo:rustc-link-search=native={path}");
    contract.push(format!("search-native={canonical_label}"));
}

fn emit_link_lib(contract: &mut Vec<String>, library: &str) {
    println!("cargo:rustc-link-lib={library}");
    contract.push(format!("lib={library}"));
}

#[derive(Clone)]
struct CapturedSource {
    label: String,
    sha256: [u8; 32],
}

#[derive(Clone)]
struct CapturedNamedInput {
    name: String,
    value: String,
}

struct ActivePatch {
    ordinal: usize,
    source: PathBuf,
    source_label: String,
    destination: String,
    sha256: [u8; 32],
}

struct CompileClosure {
    commands: Vec<String>,
    headers: Vec<CapturedSource>,
}

fn capture_compile_and_header_closure(
    manifest_dir: &Path,
    out_dir: &Path,
    mlx_dst: &Path,
    cache: &[CapturedNamedInput],
    active_dependencies: &std::collections::BTreeMap<String, PathBuf>,
) -> CompileClosure {
    let strict = env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() == Ok("1");
    let database_path = mlx_dst.join("build/compile_commands.json");
    if !database_path.is_file() {
        if strict {
            panic!("mlxcel-core: strict build identity requires CMake compile_commands.json");
        }
        return CompileClosure {
            commands: Vec::new(),
            headers: Vec::new(),
        };
    }
    let database: serde_json::Value =
        serde_json::from_slice(&fs::read(&database_path).unwrap_or_else(|err| {
            panic!(
                "mlxcel-core: cannot read {}: {err}",
                database_path.display()
            )
        }))
        .unwrap_or_else(|err| panic!("mlxcel-core: compile_commands.json is invalid: {err}"));
    let entries = database
        .as_array()
        .unwrap_or_else(|| panic!("mlxcel-core: compile_commands.json root is not an array"));
    if entries.len() > 100_000 {
        panic!("mlxcel-core: compile command database exceeds 100000 entries");
    }
    let roots = build_path_tokens(manifest_dir, out_dir, cache, active_dependencies);
    let mut commands = Vec::new();
    let mut depfiles = Vec::new();
    for entry in entries {
        let object = entry
            .as_object()
            .unwrap_or_else(|| panic!("mlxcel-core: compile command entry is not an object"));
        let directory = object
            .get("directory")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("mlxcel-core: compile command has no directory"));
        let arguments = if let Some(arguments) = object
            .get("arguments")
            .and_then(serde_json::Value::as_array)
        {
            arguments
                .iter()
                .map(|argument| {
                    argument
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| panic!("mlxcel-core: compile argument is not a string"))
                })
                .collect::<Vec<_>>()
        } else {
            split_command_line(
                object
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| {
                        panic!("mlxcel-core: compile command has neither arguments nor command")
                    }),
            )
        };
        let output = object
            .get("output")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
            .or_else(|| compile_output_argument(&arguments))
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    directory.join(path)
                }
            });
        let Some(output) = output else {
            if strict {
                panic!("mlxcel-core: strict compile command has no output path");
            }
            continue;
        };
        if !output.is_file() {
            // A compilation database may contain configured but unbuilt
            // targets. Only an object that physically entered this fresh build
            // can contribute to the effective closure.
            continue;
        }
        let file = object
            .get("file")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("mlxcel-core: compile command has no source file"));
        let canonical_file = native_identity::canonicalize_argument(file, &roots)
            .unwrap_or_else(|err| panic!("mlxcel-core: {err}"));
        let canonical_arguments = arguments
            .iter()
            .map(|argument| {
                native_identity::canonicalize_argument(argument, &roots)
                    .unwrap_or_else(|err| panic!("mlxcel-core: {err}"))
            })
            .collect::<Vec<_>>();
        commands.push(format!(
            "cmake-unit:{canonical_file}:{}",
            serde_json::to_string(&canonical_arguments)
                .unwrap_or_else(|err| panic!("mlxcel-core: cannot encode compile argv: {err}"))
        ));
        let depfile = compile_depfile_argument(&arguments)
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    directory.join(path)
                }
            })
            .or_else(|| Some(output.with_extension("d")))
            .filter(|path| path.is_file());
        if strict && depfile.is_none() {
            panic!("mlxcel-core: strict compile unit {canonical_file} has no materialized depfile");
        }
        if let Some(depfile) = depfile {
            depfiles.push((depfile, directory));
        }
    }
    if strict && commands.is_empty() {
        panic!("mlxcel-core: strict compile closure contains no built units");
    }
    commands.sort();
    if commands.windows(2).any(|pair| pair[0] == pair[1]) {
        panic!("mlxcel-core: compile closure contains a duplicate unit");
    }

    // cc-rs bridge compilations are outside CMake's database but emit depfiles
    // on qualification's Clang lane. Fold those into the same header union.
    for path in regular_files(out_dir)
        .into_iter()
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("d"))
    {
        if !depfiles.iter().any(|(known, _)| known == &path) {
            depfiles.push((path, manifest_dir.to_path_buf()));
        }
    }
    let mut headers = std::collections::BTreeMap::new();
    for (depfile, base) in depfiles {
        for dependency in parse_depfile(&depfile) {
            let path = resolve_depfile_path(&dependency, &base, manifest_dir, out_dir);
            let Some(path) = path else {
                if strict {
                    panic!(
                        "mlxcel-core: depfile {} references unavailable input {dependency:?}",
                        depfile.display()
                    );
                }
                continue;
            };
            if !path.is_file() {
                continue;
            }
            let label = native_identity::canonicalize_argument(
                &strict_path_text(&path).replace('\\', "/"),
                &roots,
            )
            .unwrap_or_else(|err| panic!("mlxcel-core: {err}"));
            let digest = file_digest(&path);
            if headers
                .insert(label.clone(), digest)
                .is_some_and(|prior| prior != digest)
            {
                panic!("mlxcel-core: header label {label} aliases different content");
            }
        }
    }
    if strict && headers.is_empty() {
        panic!("mlxcel-core: strict compile closure contains no depfile inputs");
    }
    CompileClosure {
        commands,
        headers: headers
            .into_iter()
            .map(|(label, sha256)| CapturedSource { label, sha256 })
            .collect(),
    }
}

fn build_path_tokens(
    manifest_dir: &Path,
    out_dir: &Path,
    cache: &[CapturedNamedInput],
    active_dependencies: &std::collections::BTreeMap<String, PathBuf>,
) -> Vec<native_identity::PathToken> {
    let workspace = manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("mlxcel-core: manifest is outside its workspace"));
    let mut candidates = vec![
        ("<workspace>".to_owned(), workspace.to_path_buf()),
        ("<core-out>".to_owned(), out_dir.to_path_buf()),
    ];
    for (name, path) in active_dependencies {
        candidates.push((format!("<native-{name}>"), path.clone()));
    }
    #[cfg(target_os = "macos")]
    {
        candidates.push(("<apple-sdk>".to_owned(), apple_sdk_path()));
        candidates.push((
            "<apple-clang-toolchain>".to_owned(),
            apple_clang_toolchain_path(),
        ));
    }
    for (cache_name, token) in [
        ("CMAKE_OSX_SYSROOT", "<apple-sdk>"),
        ("CMAKE_C_COMPILER", "<cmake-c-bin>"),
        ("CMAKE_CXX_COMPILER", "<cmake-cxx-bin>"),
        ("CMAKE_LINKER", "<cmake-link-bin>"),
        ("CMAKE_AR", "<cmake-ar-bin>"),
        ("CMAKE_RANLIB", "<cmake-ranlib-bin>"),
        ("CMAKE_MAKE_PROGRAM", "<cmake-build-tool-bin>"),
        ("CMAKE_GENERATOR_INSTANCE", "<cmake-generator-instance>"),
    ] {
        if let Some(value) = cache_value(cache, cache_name)
            && value != "<empty>"
        {
            let path = PathBuf::from(value);
            let root = if path.is_file() {
                path.parent().unwrap_or(&path).to_path_buf()
            } else {
                path
            };
            if root.exists() {
                candidates.push((token.to_owned(), root));
            }
        }
    }
    for (name, token) in [
        ("DEVELOPER_DIR", "<xcode-developer>"),
        ("SDKROOT", "<apple-sdk-env>"),
        ("CUDA_HOME", "<cuda-sdk>"),
        ("VCToolsInstallDir", "<msvc-toolchain>"),
        ("WindowsSdkDir", "<windows-sdk>"),
    ] {
        if let Some(path) = env::var_os(name).map(PathBuf::from)
            && path.exists()
        {
            candidates.push((token.to_owned(), path));
        }
    }
    if Path::new("/usr").is_dir() {
        candidates.push(("<system-usr>".to_owned(), PathBuf::from("/usr")));
    }
    if let Some(system_root) = env::var_os("SystemRoot").map(PathBuf::from)
        && system_root.exists()
    {
        candidates.push(("<windows-system>".to_owned(), system_root));
    }
    let mut tokens = Vec::new();
    let mut physical = std::collections::BTreeSet::new();
    for (token, root) in candidates {
        let mapping = native_identity::PathToken::new(&token, &root)
            .unwrap_or_else(|err| panic!("mlxcel-core: {err}"));
        let key = strict_path_text(&mapping.root).to_ascii_lowercase();
        if physical.insert(key) {
            tokens.push(mapping);
        }
    }
    tokens
}

fn split_command_line(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escape = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if escape {
            current.push(character);
            escape = false;
        } else if character == '\\' && quoted {
            escape = true;
        } else if character == '\\' && characters.peek() == Some(&'"') {
            escape = true;
        } else if character == '"' {
            quoted = !quoted;
        } else if character.is_whitespace() && !quoted {
            if !current.is_empty() {
                values.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if quoted || escape {
        panic!("mlxcel-core: malformed quoted compile command");
    }
    if !current.is_empty() {
        values.push(current);
    }
    values
}

fn compile_output_argument(arguments: &[String]) -> Option<PathBuf> {
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "-o" {
            return arguments.get(index + 1).map(PathBuf::from);
        }
        if let Some(path) = argument.strip_prefix("/Fo")
            && !path.is_empty()
        {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn compile_depfile_argument(arguments: &[String]) -> Option<PathBuf> {
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "-MF" {
            return arguments.get(index + 1).map(PathBuf::from);
        }
        if let Some(path) = argument.strip_prefix("-MF")
            && !path.is_empty()
        {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn parse_depfile(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot read {}: {err}", path.display()))
        .replace("\\\r\n", " ")
        .replace("\\\n", " ");
    let delimiter = text
        .find(": ")
        .or_else(|| text.find(":\t"))
        .unwrap_or_else(|| panic!("mlxcel-core: malformed depfile {}", path.display()));
    let mut values = Vec::new();
    let mut current = String::new();
    let mut escape = false;
    for character in text[delimiter + 2..].chars() {
        if escape {
            current.push(character);
            escape = false;
        } else if character == '\\' {
            escape = true;
        } else if character.is_whitespace() {
            if !current.is_empty() {
                values.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if escape {
        panic!("mlxcel-core: depfile ends in an escape: {}", path.display());
    }
    if !current.is_empty() {
        values.push(current);
    }
    values
}

fn resolve_depfile_path(
    value: &str,
    base: &Path,
    manifest_dir: &Path,
    out_dir: &Path,
) -> Option<PathBuf> {
    let value = PathBuf::from(value);
    if value.is_absolute() {
        return value
            .is_file()
            .then(|| fs::canonicalize(&value).ok())
            .flatten();
    }
    [base, manifest_dir, out_dir]
        .into_iter()
        .map(|root| root.join(&value))
        .find(|path| path.is_file())
        .and_then(|path| fs::canonicalize(path).ok())
}

fn write_native_core_build_record(
    manifest_dir: &Path,
    out_dir: &Path,
    mlx_dst: &Path,
    mlx_commit: &str,
    bridge_cxx_identity: &str,
    bridge_compiler_path: &Path,
    bridge_compile_flags: Vec<String>,
    rustc_link_contract: Vec<String>,
) {
    const MLX_REPOSITORY: &str = "https://github.com/ml-explore/mlx.git";

    let active_dependencies = active_native_dependency_paths(mlx_dst);
    let fetched_mlx = active_dependencies
        .get("mlx")
        .unwrap_or_else(|| panic!("mlxcel-core: active dependency ledger omitted MLX"))
        .clone();
    let fetched_pristine_tree_sha256 = native_identity::git_head_tree_digest(&fetched_mlx)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot encode pristine MLX tree: {err}"));
    let fetched_effective_tree_sha256 = native_identity::git_candidate_tree_digest(&fetched_mlx)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot encode effective MLX tree: {err}"));

    let active_patches = read_cmake_overlay_ledger(manifest_dir, mlx_dst);
    verify_applied_overlays(&fetched_mlx, &active_patches);
    let ordered_patches = active_patches
        .iter()
        .map(|patch| CapturedSource {
            label: format!(
                "{}:{}=>{}",
                patch.ordinal, patch.source_label, patch.destination
            ),
            sha256: patch.sha256,
        })
        .collect::<Vec<_>>();
    let mut overlay_sources = active_patches
        .iter()
        .map(|patch| CapturedSource {
            label: format!("effective-mlx/{}", patch.destination),
            sha256: patch.sha256,
        })
        .collect::<Vec<_>>();
    overlay_sources.sort_by(|left, right| left.label.cmp(&right.label));

    let turbo_root = manifest_dir.join("../mlx-cpp/turbo");
    let turbo_sources = CORE_CXX_SOURCES
        .iter()
        .filter(|relative| relative.starts_with("../mlx-cpp/turbo/"))
        .map(|relative| source_from_manifest_path(manifest_dir, &manifest_dir.join(relative)))
        .collect::<Vec<_>>();

    let mut cxx_bridge_sources = vec![source_from_manifest_path(
        manifest_dir,
        &manifest_dir.join("src/lib.rs"),
    )];
    cxx_bridge_sources.push(source_from_exact_path(
        &out_dir.join("cxxbridge/sources/mlxcel-core/src/lib.rs.cc"),
        "generated/cxxbridge/mlxcel-core/src/lib.rs.cc",
    ));
    cxx_bridge_sources.extend(
        CORE_CXX_SOURCES
            .iter()
            .map(|relative| source_from_manifest_path(manifest_dir, &manifest_dir.join(relative))),
    );

    let cache_path = mlx_dst.join("build/CMakeCache.txt");
    let cache = parse_cmake_cache(&cache_path);
    let dependency_sources = capture_active_native_dependencies(&active_dependencies);
    let path_tokens = build_path_tokens(manifest_dir, out_dir, &cache, &active_dependencies);
    let mut cmake_definitions = selected_cmake_inputs(&cache)
        .into_iter()
        .map(|mut input| {
            if input.value != "<empty>" {
                input.value = native_identity::canonicalize_argument(&input.value, &path_tokens)
                    .unwrap_or_else(|err| panic!("mlxcel-core: CMake input {}: {err}", input.name));
            }
            input
        })
        .collect::<Vec<_>>();
    for input in &dependency_sources.named_inputs {
        cmake_definitions.push(input.clone());
    }
    cmake_definitions.extend(read_cmake_target_graph(mlx_dst, &path_tokens));
    cmake_definitions.extend(selected_native_tool_inputs(&cache, &path_tokens));
    cmake_definitions.sort_by(|left, right| left.name.cmp(&right.name));
    if cmake_definitions
        .windows(2)
        .any(|pair| pair[0].name == pair[1].name)
    {
        panic!("mlxcel-core: duplicate effective CMake/native dependency input");
    }
    let native_features = cmake_definitions
        .iter()
        .filter(|entry| entry.name.starts_with("MLX_") && entry.value == "ON")
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();

    let c_compiler = cmake_compiler_identity(&cache, "CMAKE_C_COMPILER", bridge_compiler_path);
    let mlx_cxx_compiler =
        cmake_compiler_identity(&cache, "CMAKE_CXX_COMPILER", bridge_compiler_path);
    let cmake = tool_identity(&resolve_executable(
        env::var_os("CMAKE")
            .as_deref()
            .unwrap_or_else(|| std::ffi::OsStr::new("cmake")),
    ));
    let linker_path = required_cache_value(&cache, "CMAKE_LINKER");
    let linker = tool_identity(Path::new(linker_path));

    let mut compile_flags = Vec::new();
    for argument in bridge_compile_flags {
        compile_flags.push(format!(
            "cxx-bridge:{}",
            native_identity::canonicalize_argument(&argument, &path_tokens)
                .unwrap_or_else(|err| panic!("mlxcel-core: bridge compile flag: {err}"))
        ));
    }
    for key in selected_cache_flag_names("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS") {
        if let Some(value) = cache_value(&cache, &key)
            && !matches!(value, "" | "<empty>")
        {
            let value = native_identity::canonicalize_argument(value, &path_tokens)
                .unwrap_or_else(|err| panic!("mlxcel-core: CMake compile flag {key}: {err}"));
            compile_flags.push(format!("cmake:{key}={value}"));
        }
    }
    let compile_closure = capture_compile_and_header_closure(
        manifest_dir,
        out_dir,
        mlx_dst,
        &cache,
        &active_dependencies,
    );
    compile_flags.extend(compile_closure.commands);
    compile_flags.extend(capture_metal_build_recipe(mlx_dst, &cache, &path_tokens));

    let mut link_flags = Vec::new();
    for key in selected_cache_flag_names("CMAKE_EXE_LINKER_FLAGS", "CMAKE_STATIC_LINKER_FLAGS") {
        if let Some(value) = cache_value(&cache, &key)
            && !matches!(value, "" | "<empty>")
        {
            let value = native_identity::canonicalize_argument(value, &path_tokens)
                .unwrap_or_else(|err| panic!("mlxcel-core: CMake link flag {key}: {err}"));
            link_flags.push(format!("cmake:{key}={value}"));
        }
    }
    link_flags.extend(rustc_link_contract);

    let build_environment = effective_mlxcel_build_environment(&cache, &path_tokens);
    let sdk_inputs = selected_sdk_inputs(&cache);

    let mut header_inputs = compile_closure.headers;
    if env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() != Ok("1") {
        header_inputs.extend(
            regular_files(&manifest_dir.join("cpp"))
                .into_iter()
                .filter(|path| {
                    matches!(
                        path.extension().and_then(|value| value.to_str()),
                        Some("h" | "hpp")
                    )
                })
                .chain(regular_files(&turbo_root).into_iter().filter(|path| {
                    matches!(
                        path.extension().and_then(|value| value.to_str()),
                        Some("h" | "hpp")
                    )
                }))
                .map(|path| source_from_manifest_path(manifest_dir, &path)),
        );
    }
    header_inputs.push(CapturedSource {
        label: "generated/mlx-installed-headers".to_owned(),
        sha256: native_identity::directory_tree_digest(&mlx_dst.join("build/include"))
            .unwrap_or_else(|err| {
                panic!("mlxcel-core: cannot encode installed MLX headers: {err}")
            }),
    });
    header_inputs.push(source_from_exact_path(
        &out_dir.join("cxxbridge/include/mlxcel-core/src/lib.rs.h"),
        "generated/cxxbridge/mlxcel-core/src/lib.rs.h",
    ));
    header_inputs.push(source_from_exact_path(
        &out_dir.join("cxxbridge/include/rust/cxx.h"),
        "generated/cxxbridge/rust/cxx.h",
    ));
    header_inputs.extend(dependency_sources.tree_inputs);
    header_inputs.sort_by(|left, right| left.label.cmp(&right.label));
    header_inputs.dedup_by(|right, left| {
        if right.label != left.label {
            return false;
        }
        if right.sha256 != left.sha256 {
            panic!(
                "mlxcel-core: header/build input label {} aliases different content",
                right.label
            );
        }
        true
    });

    let mlx_library_dir = mlx_dst.join("build/lib");
    let library_inputs = selected_link_artifacts(&cache, &mlx_library_dir);
    let link_inputs = library_inputs.clone();
    let mut bundled_artifacts = [
        out_dir.join("libmlx_cxx_bridge.a"),
        out_dir.join("mlx_cxx_bridge.lib"),
    ]
    .into_iter()
    .filter(|path| path.is_file())
    .map(|path| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("mlxcel-core: bridge archive name is not UTF-8"));
        source_from_exact_path(&path, &format!("generated/cxx-bridge-artifact/{name}"))
    })
    .collect::<Vec<_>>();
    if bundled_artifacts.is_empty() {
        panic!(
            "mlxcel-core: the compiled mlx_cxx_bridge archive was not found in {}",
            out_dir.display()
        );
    }
    capture_metal_bundled_artifact(mlx_dst, &cache, &mut bundled_artifacts);
    bundled_artifacts.sort_by(|left, right| left.label.cmp(&right.label));

    let mut cargo_features = env::vars_os()
        .filter_map(|(name, _)| {
            let name = name.to_str()?;
            name.strip_prefix("CARGO_FEATURE_")
                .map(|value| value.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    cargo_features.sort();
    cargo_features.dedup();

    let target_triple = required_env("TARGET");
    let target_architecture = required_env("CARGO_CFG_TARGET_ARCH");
    let build_profile = required_env("PROFILE");
    let source = render_native_core_record(
        MLX_REPOSITORY,
        mlx_commit,
        fetched_pristine_tree_sha256,
        fetched_effective_tree_sha256,
        &ordered_patches,
        &cxx_bridge_sources,
        &overlay_sources,
        &turbo_sources,
        &c_compiler,
        bridge_cxx_identity,
        &mlx_cxx_compiler,
        &cmake,
        &linker,
        &target_triple,
        &target_architecture,
        &build_profile,
        &cargo_features,
        &native_features,
        &cmake_definitions,
        &build_environment,
        &compile_flags,
        &link_flags,
        &sdk_inputs,
        &header_inputs,
        &library_inputs,
        &link_inputs,
        &bundled_artifacts,
    );
    fs::write(out_dir.join("mlxcel_core_native_build_record.rs"), source)
        .unwrap_or_else(|err| panic!("mlxcel-core: failed to write native build record: {err}"));
}

fn read_cmake_overlay_ledger(manifest_dir: &Path, mlx_dst: &Path) -> Vec<ActivePatch> {
    let ledger_path = mlx_dst.join("build/mlxcel-active-overlays.tsv");
    let ledger = fs::read_to_string(&ledger_path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: CMake overlay ledger {} is unavailable: {err}",
            ledger_path.display()
        )
    });
    let mut lines = ledger.lines();
    if lines.next() != Some("mlxcel-overlay-ledger-v1") {
        panic!("mlxcel-core: CMake overlay ledger has an unknown schema");
    }
    let workspace_root = manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("mlxcel-core: manifest is outside its workspace"));
    let workspace_root = fs::canonicalize(workspace_root)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot resolve workspace root: {err}"));
    let mut patches = Vec::new();
    for (expected_ordinal, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 || fields.iter().any(|value| value.is_empty()) {
            panic!("mlxcel-core: malformed CMake overlay ledger row");
        }
        let ordinal = fields[0]
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("mlxcel-core: invalid overlay ordinal"));
        if ordinal != expected_ordinal {
            panic!("mlxcel-core: CMake overlay ordinals are not contiguous");
        }
        let source_label = fields[1].replace('\\', "/");
        let destination = fields[2].replace('\\', "/");
        if source_label.starts_with('/')
            || destination.starts_with('/')
            || source_label
                .split('/')
                .any(|part| part == ".." || part.is_empty())
            || destination
                .split('/')
                .any(|part| part == ".." || part.is_empty())
        {
            panic!("mlxcel-core: overlay ledger path is not canonical and relative");
        }
        let source = fs::canonicalize(workspace_root.join(&source_label)).unwrap_or_else(|err| {
            panic!("mlxcel-core: cannot resolve overlay source {source_label}: {err}")
        });
        if !source.starts_with(&workspace_root) {
            panic!("mlxcel-core: overlay source escaped the workspace");
        }
        let sha256 = parse_sha256(fields[3], "overlay digest");
        if file_digest(&source) != sha256 {
            panic!("mlxcel-core: CMake overlay source digest changed after application");
        }
        patches.push(ActivePatch {
            ordinal,
            source,
            source_label,
            destination,
            sha256,
        });
    }
    if patches.is_empty() {
        panic!("mlxcel-core: CMake produced an empty active-overlay ledger");
    }
    patches
}

fn active_native_dependency_paths(mlx_dst: &Path) -> std::collections::BTreeMap<String, PathBuf> {
    let ledger_path = mlx_dst.join("build/mlxcel-active-native-dependencies.tsv");
    let ledger = fs::read_to_string(&ledger_path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: CMake native dependency ledger {} is unavailable: {err}",
            ledger_path.display()
        )
    });
    let mut lines = ledger.lines();
    if lines.next() != Some("mlxcel-native-dependency-ledger-v1") {
        panic!("mlxcel-core: CMake native dependency ledger has an unknown schema");
    }
    let mut dependencies = std::collections::BTreeMap::new();
    for line in lines {
        let (name, physical) = line
            .split_once('\t')
            .unwrap_or_else(|| panic!("mlxcel-core: malformed native dependency ledger row"));
        if name.is_empty()
            || physical.is_empty()
            || name.bytes().any(|byte| {
                !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_'))
            })
        {
            panic!("mlxcel-core: invalid native dependency ledger row");
        }
        let path = fs::canonicalize(physical).unwrap_or_else(|err| {
            panic!("mlxcel-core: cannot resolve active dependency {name}: {err}")
        });
        if dependencies.insert(name.to_owned(), path).is_some() {
            panic!("mlxcel-core: duplicate active dependency {name}");
        }
    }
    if !dependencies.contains_key("mlx") {
        panic!("mlxcel-core: active dependency ledger omitted MLX");
    }
    dependencies
}

fn read_cmake_target_graph(
    mlx_dst: &Path,
    roots: &[native_identity::PathToken],
) -> Vec<CapturedNamedInput> {
    let ledger_path = mlx_dst.join("build/mlxcel-active-target-graph.tsv");
    let ledger = fs::read_to_string(&ledger_path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: CMake target graph ledger {} is unavailable: {err}",
            ledger_path.display()
        )
    });
    let mut lines = ledger.lines();
    if lines.next() != Some("mlxcel-target-graph-v1") {
        panic!("mlxcel-core: CMake target graph ledger has an unknown schema");
    }
    let mut inputs = Vec::new();
    let mut ordinals = std::collections::BTreeMap::<(String, String), usize>::new();
    for line in lines {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 5 || fields.iter().any(|value| value.is_empty()) {
            panic!("mlxcel-core: malformed CMake target graph row");
        }
        let target = fields[0];
        let target_type = fields[1];
        let property = fields[2];
        for (label, value) in [
            ("target", target),
            ("target type", target_type),
            ("target property", property),
        ] {
            if value.len() > 1024
                || !value.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')
                })
            {
                panic!("mlxcel-core: CMake {label} is not canonically encoded");
            }
        }
        let ordinal = fields[3]
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("mlxcel-core: invalid CMake target graph ordinal"));
        let key = (target.to_owned(), property.to_owned());
        let expected = ordinals.entry(key).or_default();
        if ordinal != *expected {
            panic!("mlxcel-core: CMake target graph ordinals are not contiguous");
        }
        *expected += 1;
        let value = native_identity::canonicalize_argument(fields[4], roots)
            .unwrap_or_else(|err| panic!("mlxcel-core: target graph: {err}"));
        inputs.push(CapturedNamedInput {
            name: format!("cmake-target:{target}:{target_type}:{property}:{ordinal}"),
            value,
        });
        if inputs.len() > 4096 {
            panic!("mlxcel-core: CMake target graph exceeds 4096 entries");
        }
    }
    if inputs.is_empty()
        || !inputs
            .iter()
            .any(|entry| entry.name.starts_with("cmake-target:mlx:"))
    {
        panic!("mlxcel-core: CMake target graph omitted the mlx target");
    }
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    if inputs.windows(2).any(|pair| pair[0].name == pair[1].name) {
        panic!("mlxcel-core: CMake target graph contains duplicate identities");
    }
    inputs
}

fn verify_applied_overlays(fetched_mlx: &Path, patches: &[ActivePatch]) {
    let mut expected = std::collections::BTreeSet::new();
    for patch in patches {
        if !expected.insert(patch.destination.clone()) {
            panic!(
                "mlxcel-core: duplicate active overlay destination {}",
                patch.destination
            );
        }
        let destination = fetched_mlx.join(&patch.destination);
        let source_bytes = fs::read(&patch.source).unwrap_or_else(|err| {
            panic!(
                "mlxcel-core: cannot read active overlay {}: {err}",
                patch.source.display()
            )
        });
        let destination_bytes = fs::read(&destination).unwrap_or_else(|err| {
            panic!(
                "mlxcel-core: active overlay destination {} is unavailable: {err}",
                destination.display()
            )
        });
        if source_bytes != destination_bytes {
            panic!(
                "mlxcel-core: active overlay was not applied exactly to {}",
                patch.destination
            );
        }
        let destination_sha256: [u8; 32] = Sha256::digest(&destination_bytes).into();
        if destination_sha256 != patch.sha256 {
            panic!(
                "mlxcel-core: active overlay digest disagrees with CMake ledger for {}",
                patch.destination
            );
        }
    }

    let mut changed = git_paths(fetched_mlx, &["diff", "--name-only", "HEAD", "--"]);
    changed.extend(git_paths(
        fetched_mlx,
        &["ls-files", "--others", "--exclude-standard"],
    ));
    changed.sort();
    changed.dedup();
    let unexpected = changed
        .into_iter()
        .filter(|path| !expected.contains(path))
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        panic!(
            "mlxcel-core: fetched MLX contains modifications outside the active overlay manifest: {}",
            unexpected.join(", ")
        );
    }
}

fn git_paths(root: &Path, args: &[&str]) -> Vec<String> {
    let output = git_command(root)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot inspect fetched MLX changes: {err}"));
    if !output.status.success() {
        panic!("mlxcel-core: cannot inspect fetched MLX overlay changes");
    }
    String::from_utf8(output.stdout)
        .unwrap_or_else(|_| panic!("mlxcel-core: fetched MLX path is not strict UTF-8"))
        .lines()
        .map(|line| line.replace('\\', "/"))
        .collect()
}

fn git_command(root: &Path) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(root);
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    ] {
        command.env_remove(name);
    }
    command
}

fn source_from_manifest_path(manifest_dir: &Path, path: &Path) -> CapturedSource {
    let workspace_root = manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("mlxcel-core: manifest is outside its workspace"));
    let workspace_root = fs::canonicalize(workspace_root)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot resolve workspace root: {err}"));
    let canonical = fs::canonicalize(path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: cannot resolve build input {}: {err}",
            path.display()
        )
    });
    let label = canonical
        .strip_prefix(&workspace_root)
        .unwrap_or_else(|_| panic!("mlxcel-core: build input escaped the workspace"))
        .to_str()
        .unwrap_or_else(|| panic!("mlxcel-core: build source path is not UTF-8"))
        .replace('\\', "/");
    source_from_exact_path(&canonical, &label)
}

fn source_from_exact_path(path: &Path, label: &str) -> CapturedSource {
    let path = fs::canonicalize(path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: cannot resolve required build input {}: {err}",
            path.display()
        )
    });
    if !path.is_file() {
        panic!(
            "mlxcel-core: required build input is missing: {}",
            path.display()
        );
    }
    CapturedSource {
        label: label.to_owned(),
        sha256: file_digest(&path),
    }
}

fn regular_files(root: &Path) -> Vec<PathBuf> {
    if !root.is_dir() {
        panic!(
            "mlxcel-core: required build-input directory is missing: {}",
            root.display()
        );
    }
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).unwrap_or_else(|err| {
            panic!(
                "mlxcel-core: cannot enumerate {}: {err}",
                directory.display()
            )
        });
        for entry in entries {
            let entry = entry.unwrap_or_else(|err| {
                panic!(
                    "mlxcel-core: cannot read an entry in {}: {err}",
                    directory.display()
                )
            });
            let path = entry.path();
            if entry.file_name() == ".git" {
                continue;
            }
            let metadata = fs::symlink_metadata(&path).unwrap_or_else(|err| {
                panic!("mlxcel-core: cannot inspect {}: {err}", path.display())
            });
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() || metadata.file_type().is_symlink() {
                files.push(path);
                if files.len() > 200_000 {
                    panic!("mlxcel-core: build-input tree exceeds 200000 files");
                }
            }
        }
    }
    files.sort_by(|left, right| strict_path_text(left).cmp(&strict_path_text(right)));
    files
}

fn file_digest(path: &Path) -> [u8; 32] {
    native_identity::file_sha256(path)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot digest build input: {err}"))
}

fn strict_path_text(path: &Path) -> String {
    native_identity::strict_path_text(path).unwrap_or_else(|err| panic!("mlxcel-core: {err}"))
}

fn strict_os_text(value: &std::ffi::OsStr, name: &str) -> String {
    value
        .to_str()
        .unwrap_or_else(|| panic!("mlxcel-core: {name} is not strict UTF-8"))
        .to_owned()
}

fn parse_cmake_cache(path: &Path) -> Vec<CapturedNamedInput> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "mlxcel-core: cannot read selected CMake cache {}: {err}",
            path.display()
        )
    });
    let mut values = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.starts_with("//"))
        .filter_map(|line| {
            let (key_and_type, value) = line.split_once('=')?;
            let (name, _) = key_and_type.split_once(':')?;
            Some(CapturedNamedInput {
                name: name.to_owned(),
                value: match value {
                    "" => "<empty>".to_owned(),
                    "<empty>" => {
                        panic!("mlxcel-core: CMake cache value uses reserved empty marker: {name}")
                    }
                    _ => value.to_owned(),
                },
            })
        })
        .collect::<Vec<_>>();
    values.sort_by(|left, right| left.name.cmp(&right.name));
    values
}

fn cache_value<'a>(cache: &'a [CapturedNamedInput], name: &str) -> Option<&'a str> {
    cache
        .binary_search_by(|entry| entry.name.as_str().cmp(name))
        .ok()
        .map(|index| cache[index].value.as_str())
}

fn required_cache_value<'a>(cache: &'a [CapturedNamedInput], name: &str) -> &'a str {
    cache_value(cache, name)
        .filter(|value| *value != "<empty>")
        .unwrap_or_else(|| panic!("mlxcel-core: selected CMake cache is missing {name}"))
}

fn selected_cmake_inputs(cache: &[CapturedNamedInput]) -> Vec<CapturedNamedInput> {
    let mut selected = cache
        .iter()
        .filter(|entry| {
            entry.name.starts_with("MLX_")
                || matches!(
                    entry.name.as_str(),
                    "CMAKE_BUILD_TYPE"
                        | "CMAKE_GENERATOR"
                        | "CMAKE_GENERATOR_INSTANCE"
                        | "CMAKE_GENERATOR_PLATFORM"
                        | "CMAKE_GENERATOR_TOOLSET"
                        | "CMAKE_SYSTEM_NAME"
                        | "CMAKE_SYSTEM_PROCESSOR"
                        | "CMAKE_OSX_ARCHITECTURES"
                        | "CMAKE_OSX_DEPLOYMENT_TARGET"
                        | "CMAKE_OSX_SYSROOT"
                )
        })
        .cloned()
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| left.name.cmp(&right.name));
    selected.dedup_by(|left, right| left.name == right.name);
    selected
}

fn selected_cache_flag_names(first: &str, second: &str) -> Vec<String> {
    let profile = required_env("PROFILE").to_ascii_uppercase();
    let profile = match profile.as_str() {
        "DEBUG" => "DEBUG",
        "RELEASE" => "RELEASE",
        other => panic!("mlxcel-core: unsupported Cargo profile for build identity: {other}"),
    };
    vec![
        first.to_owned(),
        format!("{first}_{profile}"),
        second.to_owned(),
        format!("{second}_{profile}"),
    ]
}

fn cmake_compiler_identity(
    cache: &[CapturedNamedInput],
    cache_name: &str,
    configured_bridge_compiler: &Path,
) -> String {
    let executable = cache_value(cache, cache_name)
        .filter(|value| *value != "<empty>")
        .map(PathBuf::from)
        .filter(|candidate| {
            candidate.is_absolute()
                || candidate.components().count() > 1
                || candidate
                    .file_name()
                    .zip(configured_bridge_compiler.file_name())
                    .is_none_or(|(candidate, configured)| {
                        !candidate
                            .to_string_lossy()
                            .eq_ignore_ascii_case(&configured.to_string_lossy())
                    })
        })
        .unwrap_or_else(|| configured_bridge_compiler.to_path_buf());
    tool_identity(&executable)
}

fn resolve_executable(name: &std::ffi::OsStr) -> PathBuf {
    let candidate = PathBuf::from(name);
    if candidate.is_file() {
        return fs::canonicalize(&candidate).unwrap_or(candidate);
    }
    let path = env::var_os("PATH").unwrap_or_else(|| panic!("mlxcel-core: PATH is unavailable"));
    let extensions = if cfg!(windows) {
        env::var_os("PATHEXT")
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into())
            .to_string_lossy()
            .split(';')
            .map(str::to_owned)
            .collect::<Vec<_>>()
    } else {
        vec![String::new()]
    };
    for directory in env::split_paths(&path) {
        for extension in &extensions {
            let mut file_name = name.to_os_string();
            if !extension.is_empty() && Path::new(name).extension().is_none() {
                file_name.push(extension);
            }
            let path = directory.join(file_name);
            if path.is_file() {
                return fs::canonicalize(&path).unwrap_or(path);
            }
        }
    }
    panic!(
        "mlxcel-core: required build tool {:?} cannot be resolved",
        name
    );
}

fn tool_identity(path: &Path) -> String {
    let resolved = resolve_executable(path.as_os_str());
    let file_name = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("mlxcel-core: tool name is not strict UTF-8"));
    let lower = file_name.to_ascii_lowercase();
    let probes: &[&[&str]] = if matches!(lower.as_str(), "cl.exe" | "link.exe" | "lib.exe") {
        &[&["/?"]]
    } else if lower.contains("xcodebuild") {
        &[&["-version"]]
    } else if lower == "ar" || lower == "ranlib" || lower.ends_with("-ar") {
        &[&["--version"], &["-V"], &["-h"]]
    } else {
        &[&["--version"], &["-version"], &["-v"]]
    };
    let mut observed = None;
    for arguments in probes {
        let output = Command::new(&resolved).args(*arguments).output();
        let Ok(output) = output else {
            continue;
        };
        let mut bytes = output.stdout;
        bytes.extend(output.stderr);
        let Ok(version) = native_identity::canonical_tool_output(bytes, "native build tool output")
        else {
            continue;
        };
        let archive_tool = lower == "ar" || lower == "ranlib" || lower.ends_with("-ar");
        let archive_help_probe = arguments.len() == 1 && arguments[0] == "-h";
        let accepts_nonzero_status = matches!(lower.as_str(), "cl.exe" | "link.exe" | "lib.exe")
            || (archive_tool && archive_help_probe);
        if output.status.success() || accepts_nonzero_status {
            observed = Some(version);
            break;
        }
    }
    let version = observed.unwrap_or_else(|| {
        panic!(
            "mlxcel-core: no version probe succeeded for {}",
            resolved.display()
        )
    });
    format!(
        "{}|sha256={}|{}",
        file_name,
        hex_digest(file_digest(&resolved)),
        version
    )
}

fn selected_native_tool_inputs(
    cache: &[CapturedNamedInput],
    roots: &[native_identity::PathToken],
) -> Vec<CapturedNamedInput> {
    let strict = env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() == Ok("1");
    let mut values = Vec::new();
    for (name, required) in [
        ("CMAKE_MAKE_PROGRAM", true),
        ("CMAKE_AR", true),
        ("CMAKE_RANLIB", false),
        ("CMAKE_C_COMPILER_AR", false),
        ("CMAKE_C_COMPILER_RANLIB", false),
        ("CMAKE_CXX_COMPILER_AR", false),
        ("CMAKE_CXX_COMPILER_RANLIB", false),
        ("CMAKE_C_COMPILER_LAUNCHER", false),
        ("CMAKE_CXX_COMPILER_LAUNCHER", false),
        ("CMAKE_CUDA_COMPILER_LAUNCHER", false),
    ] {
        let selected = cache_value(cache, name).filter(|value| !matches!(*value, "<empty>" | ":"));
        let value = match selected {
            Some(value) => {
                let mut parts = value.split(';');
                let executable = parts.next().unwrap();
                let mut identity = tool_identity(Path::new(executable));
                for argument in parts {
                    let argument = native_identity::canonicalize_argument(argument, roots)
                        .unwrap_or_else(|err| {
                            panic!("mlxcel-core: {name} launcher argument: {err}")
                        });
                    if argument.is_empty()
                        || argument.len() > 64 * 1024
                        || argument
                            .bytes()
                            .any(|byte| matches!(byte, 0 | b'\r' | b'\n'))
                    {
                        panic!("mlxcel-core: invalid {name} launcher argument");
                    }
                    identity.push_str("|arg=");
                    identity.push_str(&argument);
                }
                identity
            }
            None if strict && required => {
                panic!("mlxcel-core: strict native identity requires {name}")
            }
            None => "<unset-or-not-applicable>".to_owned(),
        };
        values.push(CapturedNamedInput {
            name: format!("native-tool:{name}"),
            value,
        });
    }
    values
}

fn effective_mlxcel_build_environment(
    cache: &[CapturedNamedInput],
    roots: &[native_identity::PathToken],
) -> Vec<CapturedNamedInput> {
    let mut values = env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.to_str()?;
            if !name.starts_with("MLXCEL_BUILD_") {
                return None;
            }
            let value = strict_os_text(&value, name);
            Some(CapturedNamedInput {
                name: name.to_owned(),
                value: native_identity::canonicalize_argument(&value, roots)
                    .unwrap_or_else(|err| panic!("mlxcel-core: build environment {name}: {err}")),
            })
        })
        .collect::<Vec<_>>();
    for (name, cache_name) in [
        ("MLXCEL_BUILD_ACCELERATE", "MLX_BUILD_ACCELERATE"),
        ("MLXCEL_BUILD_METAL", "MLX_BUILD_METAL"),
    ] {
        values.push(CapturedNamedInput {
            name: name.to_owned(),
            value: required_cache_value(cache, cache_name).to_owned(),
        });
    }
    let strict = env::var("MLXCEL_BUILD_IDENTITY_STRICT").as_deref() == Ok("1");
    values.push(CapturedNamedInput {
        name: "MLXCEL_BUILD_IDENTITY_STRICT".to_owned(),
        value: if strict { "ON" } else { "OFF" }.to_owned(),
    });
    if let Some(path) = env::var_os("MLXCEL_NATIVE_SOURCE_MANIFEST").map(PathBuf::from) {
        values.push(CapturedNamedInput {
            name: "MLXCEL_NATIVE_SOURCE_MANIFEST_SHA256".to_owned(),
            value: hex_digest(file_digest(&path)),
        });
    } else {
        values.push(CapturedNamedInput {
            name: "MLXCEL_NATIVE_SOURCE_MANIFEST_SHA256".to_owned(),
            value: "<unset>".to_owned(),
        });
    }
    let profile = required_env("PROFILE");
    let cxx_march = if profile == "release" {
        env::var("MLXCEL_CXX_MARCH").unwrap_or_else(|_| "native".to_owned())
    } else {
        "not-applied-debug".to_owned()
    };
    values.push(CapturedNamedInput {
        name: "MLXCEL_CXX_MARCH".to_owned(),
        value: cxx_march,
    });
    values.sort_by(|left, right| left.name.cmp(&right.name));
    values.dedup_by(|left, right| left.name == right.name);
    values
}

fn selected_sdk_inputs(_cache: &[CapturedNamedInput]) -> Vec<CapturedNamedInput> {
    let mut values = Vec::new();
    #[cfg(target_os = "macos")]
    {
        let developer = required_command_text("xcode-select", &["-p"], "Xcode developer path");
        let sdk = apple_sdk_path();
        if !Path::new(&developer).is_dir() || !sdk.is_dir() {
            panic!("mlxcel-core: Xcode developer directory or macOS SDK is unavailable");
        }
        let settings = sdk.join("SDKSettings.json");
        values.extend([
            CapturedNamedInput {
                name: "APPLE_SDK_CLOSURE".to_owned(),
                value: "available-and-verified".to_owned(),
            },
            CapturedNamedInput {
                name: "XCODE_DEVELOPER_ROLE".to_owned(),
                value: "<xcode-developer>".to_owned(),
            },
            CapturedNamedInput {
                name: "XCODE_VERSION".to_owned(),
                value: required_command_text("xcodebuild", &["-version"], "Xcode version"),
            },
            CapturedNamedInput {
                name: "XCRUN".to_owned(),
                value: tool_identity(Path::new("xcrun")),
            },
            CapturedNamedInput {
                name: "MACOS_SDK_ROLE".to_owned(),
                value: "<apple-sdk>".to_owned(),
            },
            CapturedNamedInput {
                name: "MACOS_SDK_VERSION".to_owned(),
                value: required_command_text(
                    "xcrun",
                    &["--sdk", "macosx", "--show-sdk-version"],
                    "macOS SDK version",
                ),
            },
            CapturedNamedInput {
                name: "MACOS_SDK_SETTINGS_SHA256".to_owned(),
                value: hex_digest(file_digest(&settings)),
            },
            CapturedNamedInput {
                name: "XCRUN_METAL".to_owned(),
                value: tool_identity(&PathBuf::from(required_command_text(
                    "xcrun",
                    &["--sdk", "macosx", "--find", "metal"],
                    "xcrun metal path",
                ))),
            },
            CapturedNamedInput {
                name: "XCRUN_METALLIB".to_owned(),
                value: tool_identity(&PathBuf::from(required_command_text(
                    "xcrun",
                    &["--sdk", "macosx", "--find", "metallib"],
                    "xcrun metallib path",
                ))),
            },
            CapturedNamedInput {
                name: "CLANG_RESOURCE_DIR_ROLE".to_owned(),
                value: "<clang-resource-dir>".to_owned(),
            },
            CapturedNamedInput {
                name: "CLANG_RESOURCE_DIR_TREE_SHA256".to_owned(),
                value: hex_digest(
                    native_identity::directory_tree_digest(Path::new(&required_command_text(
                        "clang",
                        &["--print-resource-dir"],
                        "Clang resource directory",
                    )))
                    .unwrap_or_else(|err| {
                        panic!("mlxcel-core: cannot encode Clang resource directory: {err}")
                    }),
                ),
            },
        ]);
        let deployment = cache_value(_cache, "CMAKE_OSX_DEPLOYMENT_TARGET")
            .filter(|value| *value != "<empty>")
            .map(str::to_owned)
            .or_else(|| env::var("MACOSX_DEPLOYMENT_TARGET").ok())
            .unwrap_or_else(|| "<compiler-default>".to_owned());
        values.push(CapturedNamedInput {
            name: "MACOSX_DEPLOYMENT_TARGET".to_owned(),
            value: deployment,
        });
    }
    #[cfg(not(target_os = "macos"))]
    values.push(CapturedNamedInput {
        name: "APPLE_SDK_CLOSURE".to_owned(),
        value: format!("not-applicable-target={}", required_env("TARGET")),
    });

    #[cfg(windows)]
    {
        values.push(CapturedNamedInput {
            name: "WINDOWS_SDK_ROLE".to_owned(),
            value: if env::var_os("WindowsSdkDir").is_some() {
                "<windows-sdk>".to_owned()
            } else {
                "<unset>".to_owned()
            },
        });
        for name in ["WindowsSDKVersion", "UCRTVersion"] {
            values.push(CapturedNamedInput {
                name: name.to_owned(),
                value: env::var(name).unwrap_or_else(|_| "<unset>".to_owned()),
            });
        }
    }

    #[cfg(feature = "cuda")]
    {
        let cuda = env::var_os("CUDA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
        let nvcc = cuda.join("bin/nvcc");
        values.extend([
            CapturedNamedInput {
                name: "CUDA_SDK_ROLE".to_owned(),
                value: "<cuda-sdk>".to_owned(),
            },
            CapturedNamedInput {
                name: "CUDA_NVCC".to_owned(),
                value: tool_identity(&nvcc),
            },
        ]);
    }
    values.sort_by(|left, right| left.name.cmp(&right.name));
    if values.windows(2).any(|pair| pair[0].name == pair[1].name) {
        panic!("mlxcel-core: duplicate SDK identity input");
    }
    values
}

#[cfg(target_os = "macos")]
fn required_command_text(program: &str, arguments: &[&str], label: &str) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot execute {label}: {err}"));
    if !output.status.success() {
        panic!("mlxcel-core: {label} command failed");
    }
    let mut bytes = output.stdout;
    bytes.extend(output.stderr);
    native_identity::canonical_tool_output(bytes, label)
        .unwrap_or_else(|err| panic!("mlxcel-core: {err}"))
}

#[cfg(target_os = "macos")]
fn apple_sdk_path() -> PathBuf {
    let value = required_command_text(
        "xcrun",
        &["--sdk", "macosx", "--show-sdk-path"],
        "macOS SDK path",
    );
    let path = PathBuf::from(value);
    fs::canonicalize(&path)
        .unwrap_or_else(|err| panic!("mlxcel-core: cannot resolve macOS SDK: {err}"))
}

#[cfg(target_os = "macos")]
fn apple_clang_toolchain_path() -> PathBuf {
    let compiler = required_command_text(
        "xcrun",
        &["--sdk", "macosx", "--find", "clang"],
        "xcrun Clang path",
    );
    let compiler = fs::canonicalize(&compiler).unwrap_or_else(|err| {
        panic!("mlxcel-core: cannot resolve xcrun Clang path {compiler}: {err}")
    });
    let toolchain = compiler
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| {
            panic!(
                "mlxcel-core: xcrun Clang path has no toolchain root: {}",
                compiler.display()
            )
        });
    if !toolchain.is_dir() {
        panic!(
            "mlxcel-core: xcrun Clang toolchain root is not a directory: {}",
            toolchain.display()
        );
    }
    toolchain
}

fn selected_link_artifacts(
    _cache: &[CapturedNamedInput],
    mlx_library_dir: &Path,
) -> Vec<CapturedSource> {
    let candidates = [
        mlx_library_dir.join("libmlx.a"),
        mlx_library_dir.join("mlx.lib"),
    ];
    let selected = candidates
        .into_iter()
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    if selected.len() != 1 {
        panic!(
            "mlxcel-core: static=mlx must resolve to exactly one archive in {}",
            mlx_library_dir.display()
        );
    }
    let mut sources = vec![source_from_exact_path(
        &selected[0],
        "generated/selected-link-input/mlx",
    )];

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
            .unwrap_or_else(|err| panic!("mlxcel-core: cannot resolve clang runtime: {err}"));
        if !output.status.success() {
            panic!("mlxcel-core: clang --print-runtime-dir failed");
        }
        let runtime = String::from_utf8(output.stdout)
            .unwrap_or_else(|_| panic!("mlxcel-core: clang runtime path is not strict UTF-8"));
        let compiler_rt = Path::new(runtime.trim()).join("libclang_rt.osx.a");
        sources.push(source_from_exact_path(
            &compiler_rt,
            "generated/selected-link-input/clang_rt.osx",
        ));
        let sdk = apple_sdk_path();
        for (relative, label) in [
            (
                "System/Library/Frameworks/Metal.framework/Metal.tbd",
                "sdk/framework/Metal.tbd",
            ),
            (
                "System/Library/Frameworks/Foundation.framework/Foundation.tbd",
                "sdk/framework/Foundation.tbd",
            ),
            (
                "System/Library/Frameworks/QuartzCore.framework/QuartzCore.tbd",
                "sdk/framework/QuartzCore.tbd",
            ),
            (
                "System/Library/Frameworks/Accelerate.framework/Accelerate.tbd",
                "sdk/framework/Accelerate.tbd",
            ),
            ("usr/lib/libc++.tbd", "sdk/library/libc++.tbd"),
            ("usr/lib/libobjc.tbd", "sdk/library/libobjc.tbd"),
        ] {
            sources.push(source_from_exact_path(&sdk.join(relative), label));
        }
    }

    sources.sort_by(|left, right| left.label.cmp(&right.label));
    sources
}

fn capture_metal_build_recipe(
    _mlx_dst: &Path,
    cache: &[CapturedNamedInput],
    _roots: &[native_identity::PathToken],
) -> Vec<String> {
    if required_cache_value(cache, "MLX_BUILD_METAL") != "ON" {
        return Vec::new();
    }
    #[cfg(not(target_os = "macos"))]
    panic!("mlxcel-core: MLX_BUILD_METAL=ON is invalid on a non-Apple target");
    #[cfg(target_os = "macos")]
    {
        let mut recipes = Vec::new();
        let mut inspected_bytes = 0_u64;
        for path in regular_files(&_mlx_dst.join("build")) {
            if !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("build.ninja" | "build.make")
            ) {
                continue;
            }
            let metadata = fs::metadata(&path).unwrap_or_else(|err| {
                panic!("mlxcel-core: cannot inspect {}: {err}", path.display())
            });
            inspected_bytes = inspected_bytes
                .checked_add(metadata.len())
                .unwrap_or_else(|| panic!("mlxcel-core: Metal recipe byte count overflow"));
            if metadata.len() > 64 * 1024 * 1024 || inspected_bytes > 128 * 1024 * 1024 {
                panic!("mlxcel-core: generated Metal recipe files exceed the inspection bound");
            }
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("mlxcel-core: cannot read {}: {err}", path.display()));
            for line in text.lines().map(str::trim) {
                let lower = line.to_ascii_lowercase();
                if !lower.contains("xcrun")
                    || (!lower.contains(" metal") && !lower.contains("metallib"))
                {
                    continue;
                }
                let canonical = native_identity::canonicalize_argument(line, _roots)
                    .unwrap_or_else(|err| panic!("mlxcel-core: Metal recipe: {err}"));
                recipes.push(format!("metal-recipe:{canonical}"));
                if recipes.len() > 4096 {
                    panic!("mlxcel-core: Metal recipe exceeds 4096 commands");
                }
            }
        }
        recipes.sort();
        recipes.dedup();
        if !recipes.iter().any(|line| line.contains(" -c "))
            || !recipes.iter().any(|line| line.contains(".metallib"))
        {
            panic!("mlxcel-core: Metal build lacks an actual compile or metallib recipe");
        }
        recipes
    }
}

fn capture_metal_bundled_artifact(
    _mlx_dst: &Path,
    cache: &[CapturedNamedInput],
    _artifacts: &mut Vec<CapturedSource>,
) {
    if required_cache_value(cache, "MLX_BUILD_METAL") != "ON" {
        return;
    }
    #[cfg(not(target_os = "macos"))]
    panic!("mlxcel-core: MLX_BUILD_METAL=ON is invalid on a non-Apple target");
    #[cfg(target_os = "macos")]
    {
        let path = cache_value(cache, "MLX_METAL_PATH")
            .filter(|value| *value != "<empty>")
            .map(PathBuf::from)
            .unwrap_or_else(|| _mlx_dst.join("build/_deps/mlx-build/mlx/backend/metal/kernels"))
            .join("mlx.metallib");
        _artifacts.push(source_from_exact_path(
            &path,
            "generated/selected-runtime-artifact/mlx.metallib",
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn render_native_core_record(
    repository: &str,
    revision: &str,
    pristine_tree: [u8; 32],
    effective_tree: [u8; 32],
    ordered_patches: &[CapturedSource],
    cxx_bridge_sources: &[CapturedSource],
    overlay_sources: &[CapturedSource],
    turbo_sources: &[CapturedSource],
    c_compiler: &str,
    cxx_bridge_compiler: &str,
    mlx_cxx_compiler: &str,
    cmake: &str,
    linker: &str,
    target: &str,
    architecture: &str,
    profile: &str,
    cargo_features: &[String],
    native_features: &[String],
    cmake_definitions: &[CapturedNamedInput],
    build_environment: &[CapturedNamedInput],
    compile_flags: &[String],
    link_flags: &[String],
    sdk_inputs: &[CapturedNamedInput],
    header_inputs: &[CapturedSource],
    library_inputs: &[CapturedSource],
    link_inputs: &[CapturedSource],
    bundled_artifacts: &[CapturedSource],
) -> String {
    format!(
        "pub(super) static GENERATED_NATIVE_CORE_BUILD_RECORD: NativeCoreBuildRecord = NativeCoreBuildRecord {{\n\
         fetched_mlx_repository: {repository:?},\n\
         fetched_mlx_revision: {revision:?},\n\
         fetched_mlx_pristine_tree_sha256: {pristine_tree:?},\n\
         fetched_mlx_effective_tree_sha256: {effective_tree:?},\n\
         ordered_patches: {},\n\
         cxx_bridge_sources: {},\n\
         overlay_sources: {},\n\
         turbo_sources: {},\n\
         c_compiler: {c_compiler:?},\n\
         cxx_bridge_compiler: {cxx_bridge_compiler:?},\n\
         mlx_cxx_compiler: {mlx_cxx_compiler:?},\n\
         cmake: {cmake:?},\n\
         linker: {linker:?},\n\
         target_triple: {target:?},\n\
         target_architecture: {architecture:?},\n\
         build_profile: {profile:?},\n\
         cargo_features: {},\n\
         native_features: {},\n\
         cmake_definitions: {},\n\
         build_environment: {},\n\
         compile_flags: {},\n\
         link_flags: {},\n\
         sdk_inputs: {},\n\
         header_inputs: {},\n\
         library_inputs: {},\n\
         link_mode: NativeCoreLinkMode::Mixed,\n\
         link_inputs: {},\n\
         bundled_artifacts: {},\n\
         }};\n",
        render_sources(ordered_patches),
        render_sources(cxx_bridge_sources),
        render_sources(overlay_sources),
        render_sources(turbo_sources),
        render_texts(cargo_features),
        render_texts(native_features),
        render_named(cmake_definitions),
        render_named(build_environment),
        render_texts(compile_flags),
        render_texts(link_flags),
        render_named(sdk_inputs),
        render_sources(header_inputs),
        render_sources(library_inputs),
        render_sources(link_inputs),
        render_sources(bundled_artifacts),
    )
}

fn render_sources(values: &[CapturedSource]) -> String {
    let entries = values
        .iter()
        .map(|value| {
            format!(
                "NativeCoreBuildSource {{ label: {:?}, sha256: {:?} }}",
                value.label, value.sha256
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("&[{entries}]")
}

fn render_named(values: &[CapturedNamedInput]) -> String {
    let entries = values
        .iter()
        .map(|value| {
            format!(
                "NativeCoreNamedBuildInput {{ name: {:?}, value: {:?} }}",
                value.name, value.value
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("&[{entries}]")
}

fn render_texts(values: &[String]) -> String {
    let entries = values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("&[{entries}]")
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("mlxcel-core: required build input {name} is missing"))
}

fn hex_digest(digest: [u8; 32]) -> String {
    native_identity::hex_digest(digest)
}

fn parse_sha256(value: &str, name: &str) -> [u8; 32] {
    native_identity::parse_sha256(value, name).unwrap_or_else(|err| panic!("mlxcel-core: {err}"))
}
