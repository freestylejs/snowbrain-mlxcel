// Root build script for the `mlxcel` binary.
//
// It always captures the root package's immutable native-build metadata. For
// default / `xla-backend` builds that remains its only action. Under the
// `xla-iree` feature (real OpenXLA execution, issue #449 Phase 3) it also emits
// the IREE *runtime* link recipe.
//
// Why here and not in `mlxcel-xla`: the C shim (`mlxcel-xla/csrc/xla_iree.c`) is
// compiled by that crate's build script and its object links via the normal
// `rustc-link-lib` path. But the runtime static libs need `--whole-archive`
// (to keep the local-task HAL driver registration) and a `--start-group`, which
// can only be expressed with `cargo:rustc-link-arg`, and a *dependency's*
// link-args do not propagate to the binary that links it. The binary's own build
// script is the one place those args reach the final link, so the recipe lives
// here. (Proven in spike/iree-ffi; see its FINDINGS.md.)
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[path = "build_support/native_identity.rs"]
#[allow(dead_code)]
mod native_identity;

fn main() {
    // The outer identity is the complete Git candidate tree, including newly
    // added root entries. Cargo's package-root directory watch is the small,
    // fail-closed way to notice those additions and deletions without a
    // custom watcher. It intentionally makes generated target/model/cache
    // changes eligible to rerun this producer; those paths remain excluded
    // from the candidate digest, and the extra change-detection cost is the
    // price of keeping the identity record current.
    println!("cargo:rerun-if-changed=.");
    println!("cargo:rerun-if-env-changed=IREE_DIST");
    println!("cargo:rerun-if-env-changed=IREE_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=IREE_MACOS_HOME");

    let mut root_link_recipe = Vec::new();

    // The feature env is set for this crate's own enabled features; `xla-iree`
    // becomes `CARGO_FEATURE_XLA_IREE`.
    if env::var_os("CARGO_FEATURE_XLA_IREE").is_none() {
        write_outer_native_build_record(&root_link_recipe);
        return;
    }

    // macOS (Apple Silicon dev path): Apple ld, not GNU ld. IREE ships no macOS
    // iree-dist, so the runtime is the source-built libiree_runtime_unified.a
    // under IREE_MACOS_HOME. Apple ld uses -force_load (not --whole-archive), is
    // multi-pass (no --start-group), and has no libgcc. force_load the unified
    // runtime so use_all_available_drivers + the VM keep all their objects; link
    // the per-driver and flatcc archives normally so only the members unified
    // lacks (the driver registration objects) are pulled, without duplicating the
    // impl objects unified already bundles. The Metal HAL driver needs the system
    // frameworks.
    #[cfg(target_os = "macos")]
    {
        let home = env::var("IREE_MACOS_HOME").unwrap_or_else(|_| {
            panic!(
                "the `xla-iree` feature is enabled on macOS but IREE_MACOS_HOME is \
                 not set (there is no prebuilt macOS iree-dist); run \
                 scripts/iree/setup-macos.sh and export the env it prints"
            )
        });
        let build_root = PathBuf::from(home).join("build");
        let runtime = build_root.join("runtime/src");
        let unified = runtime.join("iree/runtime/libiree_runtime_unified.a");
        assert!(
            unified.exists(),
            "IREE_MACOS_HOME build is missing {} (run scripts/iree/setup-macos.sh first)",
            unified.display()
        );
        // force_load ONLY the unified runtime: all of its objects must be present
        // for use_all_available_drivers + the VM, and it already bundles the
        // local-task / metal driver *impl* objects. The per-driver archives and
        // flatcc are linked normally (no -force_load) so the linker pulls only the
        // members unified lacks -- chiefly each driver's *registration* object,
        // which unified's register-all references. force_loading those archives
        // too would duplicate the impl objects unified already carries.
        emit_root_artifact_arg(&mut root_link_recipe, "-Wl,-force_load,", &unified);
        let mut extra = Vec::new();
        collect_driver_archives(&runtime.join("iree/hal/drivers"), &mut extra);
        // flatcc (vmfb FlatBuffer verify/parse) is referenced by the runtime and
        // the metal driver but lives outside the runtime tree and is not bundled
        // into the unified archive, so link it explicitly.
        let flatcc = build_root.join("build_tools/third_party/flatcc/libflatcc_parsing.a");
        assert!(
            flatcc.exists(),
            "IREE_MACOS_HOME build is missing {} (run scripts/iree/setup-macos.sh first)",
            flatcc.display()
        );
        extra.push(flatcc);
        for a in &extra {
            emit_root_artifact_arg(&mut root_link_recipe, "", a);
        }
        // The Metal HAL driver is Objective-C against the system frameworks.
        for fw in ["Metal", "Foundation", "QuartzCore"] {
            emit_root_link_arg(&mut root_link_recipe, "-framework");
            emit_root_link_arg(&mut root_link_recipe, fw);
        }
        emit_root_link_lib(&mut root_link_recipe, "c++");
        // The CUDA / Linux-dist recipes below are `cfg(not(macos))`, so nothing
        // runs after this block on macOS; no explicit early return is needed.
    }

    // The CUDA and Linux-dist recipes below are GNU-ld only and never apply on
    // macOS (handled by the arm above), so they are gated out there to keep the
    // macOS build free of dead code.
    #[cfg(not(target_os = "macos"))]
    {
        // CUDA mode (GB10): link the source-built cuda-enabled IREE runtime instead
        // of the prebuilt dist (mutually exclusive; IREE_CUDA_HOME wins). The
        // source-built `libiree_runtime_unified.a` already bundles the cuda driver
        // impl + local-task, so whole-archive only it (re-linking the separate cuda
        // impl libs would multiply-define their objects); the cuda registration
        // wrapper (driver_module.c.o, pulled by the shim's explicit register call),
        // IREE's vendored printf (the unified printf.c.o needs vsnprintf_), and flatcc
        // go in a group. The cuda driver dlopens libcuda at runtime (no -lcuda).
        if let Ok(home) = env::var("IREE_CUDA_HOME") {
            let b = PathBuf::from(home).join("build");
            let unified = b.join("runtime/src/iree/runtime/libiree_runtime_unified.a");
            assert!(
                unified.exists(),
                "IREE_CUDA_HOME build is missing {} (run the runtime build first)",
                unified.display()
            );
            for d in [
                "runtime/src/iree/runtime",
                "runtime/src/iree/hal/drivers/cuda/registration",
                "build_tools/third_party/flatcc",
                "build_tools/third_party/printf",
            ] {
                emit_root_link_search(&mut root_link_recipe, &b.join(d), d);
            }
            // Everything inside the group, in link order. Each entry is here
            // for a reason worth stating, because a dropped one fails either at
            // link time with a bare undefined symbol or at runtime with a
            // device that cannot be created.
            let mut group = vec![
                // The CUDA HAL driver's registration wrapper (driver_module.c.o),
                // pulled by the shim's explicit register call. Whole-archiving
                // only the unified runtime would leave it out and then no CUDA
                // device can be created.
                "-l:libiree_hal_drivers_cuda_registration_registration.a",
            ];
            // Older source builds leave the vendored printf implementation in
            // its own archive; newer unified runtimes carry it directly and do
            // not produce that archive. Link it only when the build generated it.
            // Pushed positionally rather than inserted at a fixed index, so
            // reordering the list above cannot silently move it out of the group.
            if b.join("build_tools/third_party/printf/libprintf_printf.a")
                .exists()
            {
                group.push("-l:libprintf_printf.a");
            }
            group.extend([
                // flatbuffer parsing for the VM bytecode module loader.
                "-l:libflatcc_parsing.a",
                // Compiler intrinsics IREE's C objects reference.
                "-lgcc",
                "-lm",
                "-lpthread",
                "-ldl",
                // libc, again and on purpose. rustc already passes `-lc`, but
                // it does so before these archives, and `rustc-link-arg` can
                // only append. `libiree_runtime_unified.a(call.c.o)` is built
                // with the stack protector and references `__stack_chk_guard`,
                // so that reference appears after the only libc on the line and
                // has nothing left to resolve against. The symbol is not in
                // libc itself (it is UND in `libc.so.6`); the definition lives
                // in `ld-linux-aarch64.so.1`, reachable through libc's
                // DT_NEEDED, and ld reports the failure as
                // "DSO missing from command line" naming the dynamic linker.
                // Repeating `-lc` after the archives puts libc where the
                // pending reference can reach it.
                //
                // Measured, not assumed: this entry alone fixes the link, and
                // `-Wl,--copy-dt-needed-entries` alone does not, because rustc
                // appends our args after its `-lc` and that flag only governs
                // inputs that follow it. See issue #1274.
                "-lc",
            ]);

            let mut link_args = vec![
                "-Wl,--whole-archive",
                "-l:libiree_runtime_unified.a",
                "-Wl,--no-whole-archive",
                "-Wl,--start-group",
            ];
            link_args.extend(group);
            link_args.push("-Wl,--end-group");
            for arg in link_args {
                emit_root_link_arg(&mut root_link_recipe, arg);
            }
            write_outer_native_build_record(&root_link_recipe);
            return;
        }

        let dist = env::var("IREE_DIST").expect(
            "the `xla-iree` feature is enabled but IREE_DIST is not set; point it at \
         the extracted iree-dist-<ver>-linux-<arch> tree (include/, lib/, bin/)",
        );
        let lib = PathBuf::from(dist).join("lib");
        assert!(
            lib.join("libiree_runtime_unified.a").exists(),
            "IREE_DIST lib dir {} is missing libiree_runtime_unified.a",
            lib.display()
        );
        emit_root_link_search(&mut root_link_recipe, &lib, "iree-dist/lib");

        // GNU ld is single-pass, left to right. The shim object (linked via
        // mlxcel-xla's rustc-link-lib) references the runtime; `--whole-archive` on
        // the unified runtime archive forces in all its objects (including the
        // local-task HAL driver registration that try_create_default_device needs),
        // so the shim's references resolve regardless of order. flatcc (vmfb
        // parsing), libgcc (aarch64 outline atomics), and libm (CPU-kernel math)
        // sit after the runtime in a group so ld re-scans cross-references.
        for arg in [
            "-Wl,--whole-archive",
            "-l:libiree_runtime_unified.a",
            "-Wl,--no-whole-archive",
            "-Wl,--start-group",
            "-l:libflatcc_runtime.a",
            "-l:libflatcc_parsing.a",
            "-lgcc",
            "-lm",
            "-lpthread",
            "-ldl",
            "-Wl,--end-group",
        ] {
            emit_root_link_arg(&mut root_link_recipe, arg);
        }
    }

    write_outer_native_build_record(&root_link_recipe);
}

/// Recursively collect the enabled HAL driver static archives under
/// `<build>/runtime/src/iree/hal/drivers` for the macOS link recipe (linked
/// normally, so only the registration members unified lacks are pulled). The
/// top-level `libiree_hal_drivers_drivers.a` (the register-all init) is already
/// bundled into `libiree_runtime_unified.a`, so it is skipped.
#[cfg(target_os = "macos")]
fn collect_driver_archives(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_driver_archives(&path, out);
        } else if path.extension().is_some_and(|e| e == "a")
            && path.file_name().and_then(|n| n.to_str()) != Some("libiree_hal_drivers_drivers.a")
        {
            out.push(path);
        }
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn emit_root_artifact_arg(recipe: &mut Vec<String>, prefix: &str, artifact: &Path) {
    let artifact_text = strict_path(artifact);
    println!("cargo:rustc-link-arg={prefix}{artifact_text}");
    let file_name = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| {
            panic!(
                "mlxcel: root link artifact name is not strict UTF-8: {}",
                artifact.display()
            )
        });
    recipe.push(format!(
        "artifact-arg={prefix}<name={file_name};sha256={}>",
        hex_digest(file_digest(artifact))
    ));
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn emit_root_link_lib(recipe: &mut Vec<String>, library: &str) {
    println!("cargo:rustc-link-lib={library}");
    recipe.push(format!("lib={library}"));
}

fn emit_root_link_arg(recipe: &mut Vec<String>, argument: &str) {
    println!("cargo:rustc-link-arg={argument}");
    recipe.push(format!("arg={argument}"));
}

fn emit_root_link_search(recipe: &mut Vec<String>, path: &Path, label: &str) {
    println!("cargo:rustc-link-search=native={}", path.display());
    recipe.push(format!("search-native=<{label}>"));
}

fn write_outer_native_build_record(root_link_recipe: &[String]) {
    const CANONICAL_REPOSITORY: &str = "https://github.com/lablup/mlxcel.git";
    const AUDITED_BASE: &str = "b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df";

    let manifest_dir = PathBuf::from(required_env("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let origin = git_output(&manifest_dir, &["remote", "get-url", "origin"]);
    let revision = git_output(&manifest_dir, &["rev-parse", "HEAD"]);
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        panic!("mlxcel: outer Git revision is not a full lowercase object ID");
    }
    let lineage = git_command(&manifest_dir)
        .args(["merge-base", "--is-ancestor", AUDITED_BASE, &revision])
        .status()
        .unwrap_or_else(|err| panic!("mlxcel: cannot verify audited upstream lineage: {err}"));
    if !lineage.success() {
        panic!(
            "mlxcel: materialized source revision is not descended from audited upstream base {AUDITED_BASE}"
        );
    }

    let outer_tree_sha256 = native_identity::git_candidate_tree_digest(&manifest_dir)
        .unwrap_or_else(|err| panic!("mlxcel: cannot encode outer candidate tree: {err}"));
    let cargo_lock_sha256 = file_digest(&manifest_dir.join("Cargo.lock"));
    let rustc_path = resolve_executable(
        env::var_os("RUSTC")
            .as_deref()
            .unwrap_or_else(|| std::ffi::OsStr::new("rustc")),
    );
    let rustc = tool_identity(&rustc_path, &["--version", "--verbose"]);
    let rustup_output = command_text("rustup", &["show", "active-toolchain"]);
    let rust_toolchain = rustup_output
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("mlxcel: rustup returned an empty active toolchain"));
    let rust_linker_path = env::var_os("RUSTC_LINKER")
        .map(|name| resolve_executable(&name))
        .unwrap_or_else(resolve_default_rust_linker);
    let rust_linker = if cfg!(windows) {
        // MSVC link.exe intentionally returns LNK1100 for its help command,
        // even though it emits the authoritative version banner and option
        // set. Validate that exact banner instead of treating the documented
        // help exit as an unavailable linker.
        tool_identity_from_help(&rust_linker_path, &["/?"], "Incremental Linker Version")
    } else {
        tool_identity(&rust_linker_path, &["--version"])
    };
    let target_triple = required_env("TARGET");
    let target_architecture = required_env("CARGO_CFG_TARGET_ARCH");
    let build_profile = required_env("PROFILE");
    let declared_default_features = declared_default_features(&manifest_dir.join("Cargo.toml"));
    let default_features_enabled = env::var_os("CARGO_FEATURE_DEFAULT").is_some();
    let mut cargo_features = env::vars_os()
        .filter_map(|(name, _)| {
            let name = name.to_str()?;
            name.strip_prefix("CARGO_FEATURE_")
                .map(|value| value.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    cargo_features.sort();
    cargo_features.dedup();

    let package_name = required_env("CARGO_PKG_NAME");
    let package_version = required_env("CARGO_PKG_VERSION");
    let generated = format!(
        "pub(super) static GENERATED_OUTER_BUILD_RECORD: GeneratedOuterBuildRecord = GeneratedOuterBuildRecord {{\n\
         canonical_source_repository: {CANONICAL_REPOSITORY:?},\n\
         materialized_source_repository: {origin:?},\n\
         outer_source_revision: {revision:?},\n\
         outer_source_tree_sha256: {outer_tree_sha256:?},\n\
         package_name: {package_name:?},\n\
         package_version: {package_version:?},\n\
         cargo_lock_sha256: {cargo_lock_sha256:?},\n\
         rust_toolchain: {rust_toolchain:?},\n\
         rustc: {rustc:?},\n\
         rust_linker: {rust_linker:?},\n\
         target_triple: {target_triple:?},\n\
         target_architecture: {target_architecture:?},\n\
         build_profile: {build_profile:?},\n\
         declared_default_features: {},\n\
         default_features_enabled: {default_features_enabled:?},\n\
         cargo_features: {},\n\
         root_link_flags: {},\n\
         }};\n",
        render_texts(&declared_default_features),
        render_texts(&cargo_features),
        render_texts(root_link_recipe),
    );
    fs::write(
        out_dir.join("mlxcel_outer_native_build_record.rs"),
        generated,
    )
    .unwrap_or_else(|err| panic!("mlxcel: failed to write outer build record: {err}"));
}

fn git_output(root: &Path, args: &[&str]) -> String {
    let output = git_command(root)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel: cannot execute git for build identity: {err}"));
    if !output.status.success() {
        panic!("mlxcel: Git metadata required for native build identity is unavailable");
    }
    String::from_utf8(output.stdout)
        .unwrap_or_else(|_| panic!("mlxcel: Git metadata is not strict UTF-8"))
        .trim()
        .to_owned()
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

fn file_digest(path: &Path) -> [u8; 32] {
    native_identity::file_sha256(path)
        .unwrap_or_else(|err| panic!("mlxcel: cannot digest build input: {err}"))
}

fn resolve_executable(name: &std::ffi::OsStr) -> PathBuf {
    let candidate = PathBuf::from(name);
    if candidate.is_file() {
        return fs::canonicalize(&candidate).unwrap_or(candidate);
    }
    let search = env::var_os("PATH").unwrap_or_else(|| panic!("mlxcel: PATH is unavailable"));
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
    for directory in env::split_paths(&search) {
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
    panic!("mlxcel: required build tool {name:?} cannot be resolved");
}

fn resolve_default_rust_linker() -> PathBuf {
    if cfg!(windows) && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let tools = PathBuf::from(required_env("VCToolsInstallDir"));
        let host = match required_env("HOST").split('-').next() {
            Some("x86_64") => "x64",
            Some("aarch64") => "arm64",
            Some("i686" | "i586") => "x86",
            Some(other) => panic!("mlxcel: unsupported MSVC host architecture {other}"),
            None => panic!("mlxcel: malformed HOST triple"),
        };
        let target = match required_env("CARGO_CFG_TARGET_ARCH").as_str() {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            "x86" => "x86",
            other => panic!("mlxcel: unsupported MSVC target architecture {other}"),
        };
        let linker = tools
            .join("bin")
            .join(format!("Host{host}"))
            .join(target)
            .join("link.exe");
        if !linker.is_file() {
            panic!(
                "mlxcel: MSVC final linker is unavailable at {}",
                linker.display()
            );
        }
        return fs::canonicalize(&linker).unwrap_or(linker);
    }
    resolve_executable(std::ffi::OsStr::new("cc"))
}

fn tool_identity(path: &Path, arguments: &[&str]) -> String {
    let output = Command::new(path)
        .args(arguments)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel: cannot execute {}: {err}", path.display()));
    if !output.status.success() {
        panic!(
            "mlxcel: build tool identity command failed for {}",
            path.display()
        );
    }
    let mut text = output.stdout;
    text.extend(output.stderr);
    let text = native_identity::canonical_tool_output(text, "build tool identity")
        .unwrap_or_else(|err| panic!("mlxcel: {err}"));
    format!(
        "{}|sha256={}|{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("mlxcel: tool name is not strict UTF-8")),
        hex_digest(file_digest(path)),
        text
    )
}

fn tool_identity_from_help(path: &Path, arguments: &[&str], required_banner: &str) -> String {
    let output = Command::new(path)
        .args(arguments)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel: cannot execute {}: {err}", path.display()));
    let mut text = output.stdout;
    text.extend(output.stderr);
    let text = native_identity::canonical_tool_output(text, "build tool help identity")
        .unwrap_or_else(|err| panic!("mlxcel: {err}"));
    if !text.contains(required_banner) {
        panic!(
            "mlxcel: build tool help output for {} lacked the required identity banner",
            path.display()
        );
    }
    format!(
        "{}|sha256={}|{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("mlxcel: tool name is not strict UTF-8")),
        hex_digest(file_digest(path)),
        text
    )
}

fn command_text(program: &str, arguments: &[&str]) -> String {
    let path = resolve_executable(std::ffi::OsStr::new(program));
    let output = Command::new(&path)
        .args(arguments)
        .output()
        .unwrap_or_else(|err| panic!("mlxcel: cannot execute {program}: {err}"));
    if !output.status.success() {
        panic!("mlxcel: {program} failed while resolving build identity");
    }
    String::from_utf8(output.stdout)
        .unwrap_or_else(|_| panic!("mlxcel: {program} output is not strict UTF-8"))
        .trim()
        .to_owned()
}

fn render_texts(values: &[String]) -> String {
    format!(
        "&[{}]",
        values
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn declared_default_features(manifest: &Path) -> Vec<String> {
    let text = fs::read_to_string(manifest)
        .unwrap_or_else(|err| panic!("mlxcel: cannot read {}: {err}", manifest.display()));
    let manifest: toml::Value =
        toml::from_str(&text).unwrap_or_else(|err| panic!("mlxcel: Cargo.toml is invalid: {err}"));
    let declaration = manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .and_then(|features| features.get("default"))
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("mlxcel: [features].default is not an array"));
    let mut values = declaration
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("mlxcel: default feature is not a string"))
                .to_owned()
        })
        .collect::<Vec<_>>();
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        panic!("mlxcel: default feature declaration contains a duplicate");
    }
    values
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("mlxcel: required build input {name} is missing"))
}

fn strict_path(path: &Path) -> String {
    path.to_str()
        .unwrap_or_else(|| panic!("mlxcel: build path is not strict UTF-8: {}", path.display()))
        .to_owned()
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
