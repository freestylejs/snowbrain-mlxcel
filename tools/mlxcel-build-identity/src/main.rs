use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use mlxcel_build_identity::{
    ConsumerProducerConfig, PostLinkConfig, canonical, produce_consumer_fragment,
    produce_post_link_sidecar, render_consumer_fragment_rust,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("mlxcel-build-identity: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os();
    let _program = args.next();
    let command = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(|| "expected `consumer` or `post-link` subcommand".to_owned())?;
    match command.as_str() {
        "consumer" => run_consumer(args),
        "post-link" => run_post_link(args),
        _ => Err(format!("unknown subcommand {command:?}")),
    }
}

fn run_consumer(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<(), String> {
    let config_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "consumer requires CONFIG_JSON".to_owned())?,
    );
    let json_output = PathBuf::from(
        args.next()
            .ok_or_else(|| "consumer requires OUTPUT_JSON".to_owned())?,
    );
    let rust_output = PathBuf::from(
        args.next()
            .ok_or_else(|| "consumer requires OUTPUT_RS".to_owned())?,
    );
    if args.next().is_some() {
        return Err("consumer received unexpected trailing arguments".to_owned());
    }
    let config: ConsumerProducerConfig = serde_json::from_slice(
        &fs::read(&config_path)
            .map_err(|err| format!("cannot read {}: {err}", config_path.display()))?,
    )
    .map_err(|err| format!("consumer producer config is invalid: {err}"))?;
    let fragment = produce_consumer_fragment(&config)?;
    let json = serde_json::to_vec_pretty(&fragment)
        .map_err(|err| format!("cannot encode consumer fragment: {err}"))?;
    let rust = render_consumer_fragment_rust(&fragment)?;
    atomic_write(&json_output, &json)?;
    atomic_write(&rust_output, rust.as_bytes())
}

fn run_post_link(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<(), String> {
    let artifact = PathBuf::from(
        args.next()
            .ok_or_else(|| "post-link requires ARTIFACT".to_owned())?,
    );
    let trace = PathBuf::from(
        args.next()
            .ok_or_else(|| "post-link requires LINK_TRACE".to_owned())?,
    );
    let pre_link = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(|| "post-link requires PRE_LINK_SHA256".to_owned())?;
    let output = PathBuf::from(
        args.next()
            .ok_or_else(|| "post-link requires OUTPUT".to_owned())?,
    );
    let link_map = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err("post-link received unexpected trailing arguments".to_owned());
    }
    let sidecar = produce_post_link_sidecar(&PostLinkConfig {
        artifact,
        link_trace: trace,
        link_map,
        pre_link_identity: canonical::parse_sha256(&pre_link, "pre-link identity")?,
        artifact_kind: "runner-binary".to_owned(),
    })?;
    let bytes = serde_json::to_vec_pretty(&sidecar)
        .map_err(|err| format!("cannot encode post-link sidecar: {err}"))?;
    atomic_write(&output, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, bytes)
        .map_err(|err| format!("cannot write {}: {err}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|err| format!("cannot publish {}: {err}", path.display()))
}
