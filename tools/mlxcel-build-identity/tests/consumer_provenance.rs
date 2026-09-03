use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use mlxcel_build_identity::{
    ConsumerFragment, ConsumerProducerConfig, produce_consumer_fragment,
    render_consumer_fragment_rust,
};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const APPROVED: &str = "https://github.com/freestylejs/snowbrain-mlxcel.git";
const LINEAGE: &str = "https://github.com/lablup/mlxcel.git";

fn run_git(root: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?} failed");
}

fn workspace_consumer_fragment(
    root: &Path,
    core_version: &str,
) -> Result<ConsumerFragment, String> {
    workspace_consumer_fragment_with_lock(root, core_version, false)
}

fn workspace_consumer_fragment_with_lock(
    root: &Path,
    core_version: &str,
    wrong_lock: bool,
) -> Result<ConsumerFragment, String> {
    let workspace = root.join("workspace");
    let runner = workspace.join("runner");
    let core = workspace.join("core");
    let helper = workspace.join("build-helper");
    fs::create_dir_all(&runner).unwrap();
    fs::create_dir_all(&core).unwrap();
    fs::create_dir_all(&helper).unwrap();
    fs::write(
        workspace.join("Cargo.toml"),
        "[workspace]\nmembers = [\"runner\", \"core\", \"build-helper\"]\nresolver = \"2\"\n",
    )
    .unwrap();
    fs::write(
        runner.join("Cargo.toml"),
        format!(
            "[package]\nname = \"runner\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ncore = {{ path = \"../core\" }}\nmlxcel = {{ git = \"{APPROVED}\", rev = \"{SHA}\", default-features = false, features = [\"metal\"] }}\n\n[build-dependencies]\nbuild_helper = {{ package = \"build-helper\", path = \"../build-helper\" }}\n"
        ),
    )
    .unwrap();
    fs::write(
        core.join("Cargo.toml"),
        format!("[package]\nname = \"core\"\nversion = \"{core_version}\"\nedition = \"2024\"\n"),
    )
    .unwrap();
    fs::write(
        helper.join("Cargo.toml"),
        "[package]\nname = \"build-helper\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(workspace.join("Cargo.lock"), "version = 4\n").unwrap();
    run_git(&workspace, &["init", "-q"]);
    run_git(&workspace, &["config", "user.email", "test@example.test"]);
    run_git(&workspace, &["config", "user.name", "Test"]);
    run_git(
        &workspace,
        &[
            "add",
            "Cargo.toml",
            "Cargo.lock",
            "runner/Cargo.toml",
            "core/Cargo.toml",
            "build-helper/Cargo.toml",
        ],
    );
    run_git(&workspace, &["commit", "-qm", "fixture"]);

    let runner_id = format!("path+file://{}#runner@0.1.0", runner.display());
    let core_id = format!("path+file://{}#core@{core_version}", core.display());
    let helper_id = format!("path+file://{}#build-helper@0.1.0", helper.display());
    let mlxcel_id = format!("git+{APPROVED}?rev={SHA}#{SHA}#mlxcel@0.6.0");
    let metadata = serde_json::json!({
        "packages": [
            {"id": runner_id, "name": "runner", "version": "0.1.0", "source": null, "checksum": null, "manifest_path": runner.join("Cargo.toml")},
            {"id": core_id, "name": "core", "version": core_version, "source": null, "checksum": null, "manifest_path": core.join("Cargo.toml")},
            {"id": helper_id, "name": "build-helper", "version": "0.1.0", "source": null, "checksum": null, "manifest_path": helper.join("Cargo.toml")},
            {"id": mlxcel_id, "name": "mlxcel", "version": "0.6.0", "source": format!("git+{APPROVED}?rev={SHA}#{SHA}"), "checksum": null, "manifest_path": root.join("cargo-home/git/checkouts/mlxcel/Cargo.toml")}
        ],
        "workspace_root": workspace,
        "workspace_members": [&runner_id, &core_id, &helper_id],
        "resolve": {
            "root": runner_id,
            "nodes": [
                {"id": runner_id, "features": [], "deps": [
                    {"name": "core", "pkg": core_id, "dep_kinds": [{"kind": null, "target": "cfg(target_os = \"macos\")"}]},
                    {"name": "build_helper", "pkg": helper_id, "dep_kinds": [{"kind": "build", "target": null}]},
                    {"name": "mlxcel", "pkg": mlxcel_id, "dep_kinds": [{"kind": null, "target": null}]}
                ]},
                {"id": core_id, "features": ["simd"], "deps": []},
                {"id": helper_id, "features": [], "deps": []},
                {"id": mlxcel_id, "features": ["metal"], "deps": []}
            ]
        }
    });
    let metadata_path = root.join("metadata.json");
    fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    let fake_cargo = root.join("cargo");
    fs::write(&fake_cargo, format!("#!/bin/sh\ncat {:?}\n", metadata_path)).unwrap();
    let mut permissions = fs::metadata(&fake_cargo).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_cargo, permissions).unwrap();

    let cargo_lock_path = if wrong_lock {
        let path = root.join("wrong-Cargo.lock");
        fs::write(&path, "version = 4\n").unwrap();
        path
    } else {
        workspace.join("Cargo.lock")
    };
    produce_consumer_fragment(&ConsumerProducerConfig {
        manifest_path: runner.join("Cargo.toml"),
        cargo_lock_path,
        dependency_alias: "mlxcel".to_owned(),
        package_name: "mlxcel".to_owned(),
        approved_repository: APPROVED.to_owned(),
        canonical_lineage_repository: LINEAGE.to_owned(),
        target_triple: "aarch64-apple-darwin".to_owned(),
        target_architecture: "aarch64".to_owned(),
        build_profile: "release".to_owned(),
        rust_toolchain: "stable-aarch64-apple-darwin".to_owned(),
        rustc_identity: "rustc fixture".to_owned(),
        final_linker_identity: "ld fixture".to_owned(),
        effective_rustflags: vec!["-Copt-level=3".to_owned()],
        emitted_link_directives: vec!["lib=framework=Metal".to_owned()],
        cargo: PathBuf::from(fake_cargo),
    })
}

#[test]
fn high_level_producer_is_relocation_stable_and_source_sensitive() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let fragment = workspace_consumer_fragment(first.path(), "0.1.0").unwrap();
    let relocated = workspace_consumer_fragment(second.path(), "0.1.0").unwrap();
    assert_eq!(fragment, relocated);
    assert_eq!(
        serde_json::to_vec(&fragment).unwrap(),
        serde_json::to_vec(&relocated).unwrap()
    );
    assert!(
        fragment
            .emitted_link_directives()
            .iter()
            .any(|value| value.contains("workspace:core/Cargo.toml"))
    );
    assert!(
        fragment
            .emitted_link_directives()
            .iter()
            .any(|value| value.contains("workspace:build-helper/Cargo.toml"))
    );
    assert!(
        fragment
            .emitted_link_directives()
            .iter()
            .any(|value| value.contains("target_os"))
    );
    assert_eq!(
        fragment
            .emitted_link_directives()
            .iter()
            .filter(|value| value.starts_with("workspace-candidate-tree-sha256="))
            .count(),
        1
    );
    let encoded = String::from_utf8(serde_json::to_vec(&fragment).unwrap()).unwrap();
    for forbidden in [
        first.path().to_string_lossy(),
        second.path().to_string_lossy(),
        std::borrow::Cow::Borrowed("file://"),
        std::borrow::Cow::Borrowed("CARGO_HOME"),
    ] {
        assert!(!encoded.contains(forbidden.as_ref()), "{forbidden}");
    }

    let third = tempfile::tempdir().unwrap();
    let mutated = workspace_consumer_fragment(third.path(), "0.1.1").unwrap();
    assert_ne!(fragment.canonical_digest(), mutated.canonical_digest());
}

#[test]
fn renderer_accepts_only_the_producer_owned_fragment() {
    let directory = tempfile::tempdir().unwrap();
    let fragment = workspace_consumer_fragment(directory.path(), "0.1.0").unwrap();
    assert!(render_consumer_fragment_rust(&fragment).is_ok());
}

#[test]
#[ignore = "requires MLXCEL_TEST_RLIB from a compiled workspace candidate"]
fn rendered_fragment_typechecks_in_an_external_consumer() {
    let mlxcel_rlib = PathBuf::from(
        std::env::var_os("MLXCEL_TEST_RLIB").expect("MLXCEL_TEST_RLIB must name libmlxcel.rlib"),
    );
    assert!(
        mlxcel_rlib.is_file(),
        "{} is not a file",
        mlxcel_rlib.display()
    );
    let dependencies = mlxcel_rlib.parent().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let fragment = workspace_consumer_fragment(directory.path(), "0.1.0").unwrap();
    let source = directory.path().join("consumer.rs");
    fs::write(&source, render_consumer_fragment_rust(&fragment).unwrap()).unwrap();

    let output = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "lib",
            "--emit",
            "metadata",
        ])
        .arg("-L")
        .arg(format!("dependency={}", dependencies.display()))
        .arg("--extern")
        .arg(format!("mlxcel={}", mlxcel_rlib.display()))
        .arg(&source)
        .arg("-o")
        .arg(directory.path().join("consumer.rmeta"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "rustc failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn high_level_producer_rejects_non_workspace_lockfile() {
    let directory = tempfile::tempdir().unwrap();
    let error = workspace_consumer_fragment_with_lock(directory.path(), "0.1.0", true).unwrap_err();
    assert!(error.contains("workspace lockfile"), "{error}");
}
