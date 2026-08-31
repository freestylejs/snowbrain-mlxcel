use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::canonical::{self, MAX_CANONICAL_ITEMS, MAX_CANONICAL_TEXT_BYTES};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerProducerConfig {
    pub manifest_path: PathBuf,
    pub cargo_lock_path: PathBuf,
    pub dependency_alias: String,
    pub package_name: String,
    pub approved_repository: String,
    pub canonical_lineage_repository: String,
    pub target_triple: String,
    pub target_architecture: String,
    pub build_profile: String,
    pub rust_toolchain: String,
    pub rustc_identity: String,
    pub final_linker_identity: String,
    pub effective_rustflags: Vec<String>,
    pub emitted_link_directives: Vec<String>,
    pub cargo: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerFragment {
    pub schema: String,
    pub canonical_lineage_repository: String,
    pub approved_repository: String,
    pub manifest_repository: String,
    pub manifest_revision: String,
    pub manifest_default_features: bool,
    pub manifest_explicit_features: Vec<String>,
    pub resolved_package_name: String,
    pub resolved_package_version: String,
    pub resolved_package_source: String,
    pub resolved_package_checksum: Option<String>,
    pub cargo_lock_sha256: String,
    pub effective_cargo_features: Vec<String>,
    pub rust_toolchain: String,
    pub rustc: String,
    pub final_linker: String,
    pub target_triple: String,
    pub target_architecture: String,
    pub build_profile: String,
    pub effective_rustflags: Vec<String>,
    pub emitted_link_directives: Vec<String>,
    pub canonical_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitCargoSource {
    pub repository: String,
    pub requested_revision: String,
    pub resolved_revision: String,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<MetadataPackage>,
    resolve: MetadataResolve,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MetadataResolve {
    root: Option<String>,
    nodes: Vec<MetadataNode>,
}

#[derive(Debug, Deserialize)]
struct MetadataNode {
    id: String,
    features: Vec<String>,
    deps: Vec<MetadataDependency>,
}

#[derive(Debug, Deserialize)]
struct MetadataDependency {
    name: String,
    pkg: String,
    #[serde(default)]
    dep_kinds: Vec<MetadataDependencyKind>,
}

#[derive(Debug, Deserialize)]
struct MetadataDependencyKind {
    kind: Option<String>,
    target: Option<String>,
}

struct ManifestEdge {
    repository: String,
    revision: String,
    default_features: bool,
    features: Vec<String>,
}

pub fn produce_consumer_fragment(
    config: &ConsumerProducerConfig,
) -> Result<ConsumerFragment, String> {
    validate_config(config)?;
    let manifest_text = fs::read_to_string(&config.manifest_path).map_err(|err| {
        format!(
            "cannot read runner manifest {}: {err}",
            config.manifest_path.display()
        )
    })?;
    let manifest: toml::Value = toml::from_str(&manifest_text)
        .map_err(|err| format!("runner Cargo.toml is invalid: {err}"))?;
    let edge = manifest_edge(&manifest, &config.dependency_alias)?;
    if edge.repository != config.approved_repository {
        return Err(format!(
            "runner dependency repository {:?} is not the explicitly approved destination {:?}",
            edge.repository, config.approved_repository
        ));
    }
    if edge.default_features {
        return Err("runner mlxcel edge must set default-features = false".to_owned());
    }

    let metadata_bytes = cargo_metadata(config)?;
    let metadata: Metadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| format!("cargo metadata output is invalid: {err}"))?;
    let root = metadata
        .resolve
        .root
        .as_deref()
        .ok_or_else(|| "cargo metadata has no runner root package".to_owned())?;
    let root_node = metadata
        .resolve
        .nodes
        .iter()
        .find(|node| node.id == root)
        .ok_or_else(|| "cargo metadata omitted the runner resolve node".to_owned())?;
    let direct = root_node
        .deps
        .iter()
        .filter(|dependency| dependency.name == config.dependency_alias)
        .collect::<Vec<_>>();
    if direct.len() != 1 {
        return Err(format!(
            "runner must resolve exactly one direct {:?} edge",
            config.dependency_alias
        ));
    }
    let package = metadata
        .packages
        .iter()
        .find(|package| package.id == direct[0].pkg)
        .ok_or_else(|| "resolved direct package is absent from cargo metadata".to_owned())?;
    if package.name != config.package_name {
        return Err(format!(
            "direct edge resolved package {:?}, expected {:?}",
            package.name, config.package_name
        ));
    }
    let source_text = package.source.as_deref().ok_or_else(|| {
        "qualification runner resolved mlxcel as an unpinned path package".to_owned()
    })?;
    let source = parse_git_cargo_source(source_text)?;
    if source.repository != edge.repository
        || source.requested_revision != edge.revision
        || source.resolved_revision != edge.revision
    {
        return Err(
            "Cargo resolved source does not exactly match the manifest git/rev edge".to_owned(),
        );
    }
    let package_node = metadata
        .resolve
        .nodes
        .iter()
        .find(|node| node.id == package.id)
        .ok_or_else(|| "cargo metadata omitted the resolved mlxcel node".to_owned())?;
    let mut effective_features = package_node.features.clone();
    normalize_unique(&mut effective_features, "effective Cargo feature")?;
    for feature in &edge.features {
        if !effective_features.contains(feature) {
            return Err(format!(
                "manifest feature {feature:?} is absent from Cargo's effective feature tuple"
            ));
        }
    }
    if effective_features
        .iter()
        .any(|feature| feature == "default")
    {
        return Err("Cargo resolved the forbidden default feature".to_owned());
    }

    let active_cargo_closure = active_cargo_closure(&metadata, root)?;

    let lock_digest = canonical::file_sha256(&config.cargo_lock_path)?;
    let manifest_digest = canonical::file_sha256(&config.manifest_path)?;
    let mut rustflags = config.effective_rustflags.clone();
    normalize_preserving_order(&mut rustflags, "effective rustflag")?;
    let mut directives = config.emitted_link_directives.clone();
    normalize_preserving_order(&mut directives, "emitted link directive")?;
    if directives.is_empty() {
        return Err("consumer producer requires the runner's ordered final-link plan".to_owned());
    }
    let mut provenance = vec![
        format!("canonical-lineage={}", config.canonical_lineage_repository),
        format!("approved-materialization={}", config.approved_repository),
        format!(
            "runner-manifest-sha256={}",
            canonical::hex_digest(manifest_digest)
        ),
        format!(
            "cargo-resolution=locked-offline;filter-platform={}",
            config.target_triple
        ),
    ];
    provenance.extend(active_cargo_closure);
    provenance.extend(directives);
    directives = provenance;
    directives.extend(rustflags.iter().map(|value| format!("rustflag={value}")));

    let mut fragment = ConsumerFragment {
        schema: "mlxcel.consumer-build.v1".to_owned(),
        canonical_lineage_repository: config.canonical_lineage_repository.clone(),
        approved_repository: config.approved_repository.clone(),
        manifest_repository: edge.repository,
        manifest_revision: edge.revision,
        manifest_default_features: false,
        manifest_explicit_features: edge.features,
        resolved_package_name: package.name.clone(),
        resolved_package_version: package.version.clone(),
        resolved_package_source: source_text.to_owned(),
        resolved_package_checksum: package.checksum.clone(),
        cargo_lock_sha256: canonical::hex_digest(lock_digest),
        effective_cargo_features: effective_features,
        rust_toolchain: bounded_text(&config.rust_toolchain, "Rust toolchain")?,
        rustc: bounded_text(&config.rustc_identity, "rustc identity")?,
        final_linker: bounded_text(&config.final_linker_identity, "final linker identity")?,
        target_triple: bounded_text(&config.target_triple, "target triple")?,
        target_architecture: bounded_text(&config.target_architecture, "target architecture")?,
        build_profile: bounded_text(&config.build_profile, "build profile")?,
        effective_rustflags: rustflags,
        emitted_link_directives: directives,
        canonical_digest: String::new(),
    };
    fragment.canonical_digest = canonical::hex_digest(digest_fragment(&fragment));
    Ok(fragment)
}

pub fn parse_git_cargo_source(value: &str) -> Result<GitCargoSource, String> {
    let raw = value
        .strip_prefix("git+")
        .ok_or_else(|| "resolved Cargo source is not git+".to_owned())?;
    let mut url = Url::parse(raw).map_err(|err| format!("invalid Cargo git source: {err}"))?;
    let resolved = url
        .fragment()
        .ok_or_else(|| "Cargo git source has no resolved #commit".to_owned())?
        .to_owned();
    if !canonical::is_full_lower_git_sha(&resolved) {
        return Err("Cargo git source fragment is not a full lowercase commit".to_owned());
    }
    let pairs = url
        .query_pairs()
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    if pairs.len() != 1 || pairs[0].0 != "rev" {
        return Err("Cargo git source must contain exactly one `rev` query".to_owned());
    }
    let requested = pairs[0].1.clone();
    if !canonical::is_full_lower_git_sha(&requested) {
        return Err("Cargo git source rev is not a full lowercase commit".to_owned());
    }
    if requested != resolved {
        return Err("Cargo git source query rev and resolved fragment disagree".to_owned());
    }
    url.set_fragment(None);
    url.set_query(None);
    Ok(GitCargoSource {
        repository: normalize_repository_url(url)?,
        requested_revision: requested,
        resolved_revision: resolved,
    })
}

pub fn render_consumer_fragment_rust(fragment: &ConsumerFragment) -> Result<String, String> {
    let digest = canonical::parse_sha256(&fragment.cargo_lock_sha256, "Cargo.lock")?;
    let features = render_string_slice(&fragment.manifest_explicit_features);
    let effective = render_string_slice(&fragment.effective_cargo_features);
    let directives = render_string_slice(&fragment.emitted_link_directives);
    Ok(format!(
        "pub fn generated_mlxcel_consumer_fragment() -> Result<mlxcel::NativeConsumerBuildFragment, mlxcel::NativeRuntimeError> {{\n\
         mlxcel::NativeConsumerBuildFragment::try_from_generated(mlxcel::NativeConsumerBuildRecord {{\n\
         manifest_repository: {:?}, manifest_revision: {:?}, manifest_default_features: false,\n\
         manifest_explicit_features: {}, resolved_package_name: {:?},\n\
         resolved_package_version: {:?}, resolved_package_source: {:?},\n\
         resolved_package_checksum: {}, cargo_lock_sha256: {:?},\n\
         effective_cargo_features: {}, rust_toolchain: {:?}, rustc: {:?},\n\
         final_linker: {:?}, target_triple: {:?}, target_architecture: {:?},\n\
         build_profile: {:?}, emitted_link_directives: {},\n\
         }})\n}}\n",
        fragment.manifest_repository,
        fragment.manifest_revision,
        features,
        fragment.resolved_package_name,
        fragment.resolved_package_version,
        fragment.resolved_package_source,
        fragment
            .resolved_package_checksum
            .as_ref()
            .map_or_else(|| "None".to_owned(), |value| format!("Some({value:?})")),
        digest,
        effective,
        fragment.rust_toolchain,
        fragment.rustc,
        fragment.final_linker,
        fragment.target_triple,
        fragment.target_architecture,
        fragment.build_profile,
        directives,
    ))
}

fn manifest_edge(manifest: &toml::Value, alias: &str) -> Result<ManifestEdge, String> {
    let dependencies = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| "runner Cargo.toml has no [dependencies] table".to_owned())?;
    let value = dependencies
        .get(alias)
        .ok_or_else(|| format!("runner has no direct dependency {alias:?}"))?;
    let table = value
        .as_table()
        .ok_or_else(|| "mlxcel dependency must use a detailed table".to_owned())?;
    let repository = table
        .get("git")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| "mlxcel dependency must specify git".to_owned())?;
    let repository = normalize_repository_url(
        Url::parse(repository).map_err(|err| format!("invalid dependency git URL: {err}"))?,
    )?;
    let revision = table
        .get("rev")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| "mlxcel dependency must specify rev".to_owned())?
        .to_owned();
    if !canonical::is_full_lower_git_sha(&revision) {
        return Err("mlxcel dependency rev is not a full lowercase commit".to_owned());
    }
    if table.contains_key("branch") || table.contains_key("tag") || table.contains_key("version") {
        return Err("mlxcel dependency combines rev with a floating selector".to_owned());
    }
    let default_features = table
        .get("default-features")
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| "mlxcel dependency must explicitly set default-features".to_owned())?;
    let mut features = table
        .get("features")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| "mlxcel dependency must explicitly list features".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "mlxcel feature is not a string".to_owned())
        })
        .collect::<Result<Vec<_>, String>>()?;
    normalize_unique(&mut features, "manifest feature")?;
    Ok(ManifestEdge {
        repository,
        revision,
        default_features,
        features,
    })
}

fn cargo_metadata(config: &ConsumerProducerConfig) -> Result<Vec<u8>, String> {
    let output = Command::new(&config.cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--offline",
            "--filter-platform",
            &config.target_triple,
            "--manifest-path",
        ])
        .arg(&config.manifest_path)
        .env("CARGO_NET_OFFLINE", "true")
        .output()
        .map_err(|err| format!("cannot run cargo metadata: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata --locked --offline failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

fn active_cargo_closure(metadata: &Metadata, root: &str) -> Result<Vec<String>, String> {
    let packages = metadata
        .packages
        .iter()
        .map(|package| (package.id.as_str(), package))
        .collect::<BTreeMap<_, _>>();
    let nodes = metadata
        .resolve
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut pending = vec![root];
    let mut reached = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !reached.insert(id) {
            continue;
        }
        if reached.len() > MAX_CANONICAL_ITEMS {
            return Err("active Cargo closure exceeds the item bound".to_owned());
        }
        let node = nodes
            .get(id)
            .ok_or_else(|| format!("cargo metadata omitted active resolve node {id:?}"))?;
        for dependency in &node.deps {
            if dependency.dep_kinds.is_empty()
                || dependency
                    .dep_kinds
                    .iter()
                    .any(|kind| kind.kind.as_deref() != Some("dev"))
            {
                pending.push(dependency.pkg.as_str());
            }
        }
    }

    let mut labels = BTreeMap::new();
    let mut records = Vec::new();
    for id in &reached {
        let package = packages
            .get(id)
            .ok_or_else(|| format!("cargo metadata omitted active package {id:?}"))?;
        let node = nodes
            .get(id)
            .ok_or_else(|| format!("cargo metadata omitted active resolve node {id:?}"))?;
        let source = match package.source.as_deref() {
            Some(source) => source,
            None if *id == root => "<runner-root>",
            None => {
                return Err(format!(
                    "active Cargo dependency {:?} is an unpinned path package",
                    package.name
                ));
            }
        };
        validate_resolved_package_source(source, package.checksum.as_deref(), *id == root)?;
        let mut features = node.features.clone();
        normalize_unique(&mut features, "active Cargo package feature")?;
        let label = serde_json::to_string(&(
            package.name.as_str(),
            package.version.as_str(),
            source,
            package.checksum.as_deref(),
        ))
        .map_err(|err| format!("cannot encode active Cargo package label: {err}"))?;
        if labels.insert(*id, label.clone()).is_some() {
            return Err("active Cargo package identifier is duplicated".to_owned());
        }
        let record = serde_json::to_string(&(label.as_str(), features))
            .map_err(|err| format!("cannot encode active Cargo package: {err}"))?;
        records.push(format!("active-cargo-package={record}"));
    }

    for id in &reached {
        let node = nodes
            .get(id)
            .ok_or_else(|| format!("cargo metadata omitted active resolve node {id:?}"))?;
        for dependency in &node.deps {
            if !reached.contains(dependency.pkg.as_str()) {
                continue;
            }
            let mut kinds = dependency
                .dep_kinds
                .iter()
                .filter(|kind| kind.kind.as_deref() != Some("dev"))
                .map(|kind| {
                    (
                        kind.kind.as_deref().unwrap_or("normal"),
                        kind.target.as_deref(),
                    )
                })
                .collect::<Vec<_>>();
            if dependency.dep_kinds.is_empty() {
                kinds.push(("normal", None));
            }
            if kinds.is_empty() {
                continue;
            }
            kinds.sort();
            kinds.dedup();
            let from = labels
                .get(id)
                .ok_or_else(|| "active Cargo edge source has no label".to_owned())?;
            let to = labels
                .get(dependency.pkg.as_str())
                .ok_or_else(|| "active Cargo edge destination has no label".to_owned())?;
            let record = serde_json::to_string(&(
                from.as_str(),
                dependency.name.as_str(),
                to.as_str(),
                kinds,
            ))
            .map_err(|err| format!("cannot encode active Cargo edge: {err}"))?;
            records.push(format!("active-cargo-edge={record}"));
        }
    }
    normalize_unique(&mut records, "active Cargo closure record")?;
    Ok(records)
}

fn validate_resolved_package_source(
    source: &str,
    checksum: Option<&str>,
    runner_root: bool,
) -> Result<(), String> {
    bounded_text(source, "active Cargo package source")?;
    if runner_root {
        if source != "<runner-root>" || checksum.is_some() {
            return Err("runner root package has contradictory source provenance".to_owned());
        }
        return Ok(());
    }
    if source.starts_with("registry+") {
        let checksum = checksum
            .ok_or_else(|| "registry package in active Cargo closure has no checksum".to_owned())?;
        if checksum.len() != 64
            || !checksum
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err("registry package checksum is not canonical SHA-256".to_owned());
        }
    } else if source.starts_with("git+") {
        if checksum.is_some() {
            return Err(
                "Git package in active Cargo closure unexpectedly has a checksum".to_owned(),
            );
        }
        let (_, revision) = source
            .rsplit_once('#')
            .ok_or_else(|| "Git package source has no resolved commit".to_owned())?;
        if !canonical::is_full_lower_git_sha(revision) {
            return Err("Git package source commit is not canonical".to_owned());
        }
    } else {
        return Err("active Cargo package has an unsupported source kind".to_owned());
    }
    Ok(())
}

fn validate_config(config: &ConsumerProducerConfig) -> Result<(), String> {
    for (label, value) in [
        ("dependency alias", config.dependency_alias.as_str()),
        ("package name", config.package_name.as_str()),
        ("approved repository", config.approved_repository.as_str()),
        (
            "canonical lineage repository",
            config.canonical_lineage_repository.as_str(),
        ),
    ] {
        bounded_text(value, label)?;
    }
    let approved = normalize_repository_url(
        Url::parse(&config.approved_repository)
            .map_err(|err| format!("invalid approved repository URL: {err}"))?,
    )?;
    if approved != config.approved_repository {
        return Err("approved repository URL is not canonical".to_owned());
    }
    let lineage = normalize_repository_url(
        Url::parse(&config.canonical_lineage_repository)
            .map_err(|err| format!("invalid canonical lineage URL: {err}"))?,
    )?;
    if lineage != config.canonical_lineage_repository {
        return Err("canonical lineage URL is not canonical".to_owned());
    }
    Ok(())
}

fn normalize_repository_url(mut value: Url) -> Result<String, String> {
    if !matches!(value.scheme(), "https" | "ssh") {
        return Err("repository URL must use https or ssh".to_owned());
    }
    if value.username() != "" || value.password().is_some() {
        return Err("repository URL must not contain credentials".to_owned());
    }
    value.set_query(None);
    value.set_fragment(None);
    let path = value.path().trim_end_matches('/').to_owned();
    value.set_path(&path);
    Ok(value.to_string())
}

fn normalize_unique(values: &mut Vec<String>, label: &str) -> Result<(), String> {
    if values.len() > MAX_CANONICAL_ITEMS {
        return Err(format!("{label} list exceeds the item bound"));
    }
    for value in values.iter() {
        bounded_text(value, label)?;
    }
    values.sort();
    let before = values.len();
    values.dedup();
    if values.len() != before {
        return Err(format!("{label} list contains a duplicate"));
    }
    Ok(())
}

fn normalize_preserving_order(values: &mut [String], label: &str) -> Result<(), String> {
    if values.len() > MAX_CANONICAL_ITEMS {
        return Err(format!("{label} list exceeds the item bound"));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        bounded_text(value, label)?;
        if !seen.insert(value.clone()) {
            return Err(format!("{label} list contains a duplicate"));
        }
    }
    Ok(())
}

fn bounded_text(value: &str, label: &str) -> Result<String, String> {
    if value.is_empty()
        || value.len() > MAX_CANONICAL_TEXT_BYTES
        || value
            .bytes()
            .any(|byte| byte == 0 || byte == b'\r' || byte == b'\n')
    {
        return Err(format!(
            "{label} is empty, oversized, or contains a control byte"
        ));
    }
    Ok(value.to_owned())
}

fn digest_fragment(fragment: &ConsumerFragment) -> [u8; 32] {
    let mut encoder = canonical::CanonicalEncoder::default();
    encoder.domain(&fragment.schema);
    for value in [
        &fragment.canonical_lineage_repository,
        &fragment.approved_repository,
        &fragment.manifest_repository,
        &fragment.manifest_revision,
        &fragment.resolved_package_name,
        &fragment.resolved_package_version,
        &fragment.resolved_package_source,
        &fragment.cargo_lock_sha256,
        &fragment.rust_toolchain,
        &fragment.rustc,
        &fragment.final_linker,
        &fragment.target_triple,
        &fragment.target_architecture,
        &fragment.build_profile,
    ] {
        encoder.text(value);
    }
    encoder.bool(fragment.manifest_default_features);
    match &fragment.resolved_package_checksum {
        Some(value) => {
            encoder.bool(true);
            encoder.text(value);
        }
        None => encoder.bool(false),
    }
    for values in [
        &fragment.manifest_explicit_features,
        &fragment.effective_cargo_features,
        &fragment.effective_rustflags,
        &fragment.emitted_link_directives,
    ] {
        encoder.u64(values.len() as u64);
        for value in values {
            encoder.text(value);
        }
    }
    encoder.finish()
}

fn render_string_slice(values: &[String]) -> String {
    format!(
        "&[{}]",
        values
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn cargo_source_requires_exact_query_and_fragment_commit() {
        let valid = format!("git+https://example.test/mlxcel.git?rev={SHA}#{SHA}");
        let parsed = parse_git_cargo_source(&valid).unwrap();
        assert_eq!(parsed.requested_revision, SHA);
        assert_eq!(parsed.resolved_revision, SHA);
        assert_eq!(parsed.repository, "https://example.test/mlxcel.git");

        for invalid in [
            format!("git+https://example.test/mlxcel.git#{SHA}"),
            format!("git+https://example.test/mlxcel.git?branch=main#{SHA}"),
            format!(
                "git+https://example.test/mlxcel.git?rev={SHA}#1123456789abcdef0123456789abcdef01234567"
            ),
            format!("git+https://example.test/mlxcel.git?rev=main#{SHA}"),
        ] {
            assert!(parse_git_cargo_source(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn manifest_parser_rejects_floating_or_default_edges() {
        let valid: toml::Value = toml::from_str(&format!(
            "[dependencies]\nmlxcel = {{ git = \"https://example.test/mlxcel.git\", rev = \"{SHA}\", default-features = false, features = [\"metal\"] }}"
        ))
        .unwrap();
        assert!(!manifest_edge(&valid, "mlxcel").unwrap().default_features);

        for text in [
            format!(
                "[dependencies]\nmlxcel = {{ git = \"https://example.test/mlxcel.git\", rev = \"{SHA}\", features = [\"metal\"] }}"
            ),
            "[dependencies]\nmlxcel = { git = \"https://example.test/mlxcel.git\", branch = \"main\", default-features = false, features = [] }".to_owned(),
        ] {
            let value: toml::Value = toml::from_str(&text).unwrap();
            assert!(manifest_edge(&value, "mlxcel").is_err());
        }
    }

    #[test]
    fn active_cargo_closure_binds_packages_features_and_non_dev_edges() {
        let git_source = format!("git+https://example.test/mlxcel.git?rev={SHA}#{SHA}");
        let metadata = Metadata {
            packages: vec![
                MetadataPackage {
                    id: "root".to_owned(),
                    name: "runner".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                },
                MetadataPackage {
                    id: "mlxcel".to_owned(),
                    name: "mlxcel".to_owned(),
                    version: "0.4.0".to_owned(),
                    source: Some(git_source),
                    checksum: None,
                },
                MetadataPackage {
                    id: "serde".to_owned(),
                    name: "serde".to_owned(),
                    version: "1.0.0".to_owned(),
                    source: Some(
                        "registry+https://github.com/rust-lang/crates.io-index".to_owned(),
                    ),
                    checksum: Some("11".repeat(32)),
                },
                MetadataPackage {
                    id: "dev-only".to_owned(),
                    name: "dev-only".to_owned(),
                    version: "1.0.0".to_owned(),
                    source: Some(
                        "registry+https://github.com/rust-lang/crates.io-index".to_owned(),
                    ),
                    checksum: Some("22".repeat(32)),
                },
            ],
            resolve: MetadataResolve {
                root: Some("root".to_owned()),
                nodes: vec![
                    MetadataNode {
                        id: "root".to_owned(),
                        features: vec![],
                        deps: vec![
                            MetadataDependency {
                                name: "mlxcel".to_owned(),
                                pkg: "mlxcel".to_owned(),
                                dep_kinds: vec![MetadataDependencyKind {
                                    kind: None,
                                    target: None,
                                }],
                            },
                            MetadataDependency {
                                name: "dev_only".to_owned(),
                                pkg: "dev-only".to_owned(),
                                dep_kinds: vec![MetadataDependencyKind {
                                    kind: Some("dev".to_owned()),
                                    target: None,
                                }],
                            },
                        ],
                    },
                    MetadataNode {
                        id: "mlxcel".to_owned(),
                        features: vec!["metal".to_owned()],
                        deps: vec![MetadataDependency {
                            name: "serde".to_owned(),
                            pkg: "serde".to_owned(),
                            dep_kinds: vec![MetadataDependencyKind {
                                kind: Some("build".to_owned()),
                                target: None,
                            }],
                        }],
                    },
                    MetadataNode {
                        id: "serde".to_owned(),
                        features: vec!["derive".to_owned()],
                        deps: vec![],
                    },
                    MetadataNode {
                        id: "dev-only".to_owned(),
                        features: vec![],
                        deps: vec![],
                    },
                ],
            },
        };

        let closure = active_cargo_closure(&metadata, "root").unwrap();
        assert!(closure.iter().any(|value| value.contains("mlxcel")));
        assert!(closure.iter().any(|value| value.contains("serde")));
        assert!(closure.iter().any(|value| value.contains("derive")));
        assert!(!closure.iter().any(|value| value.contains("dev-only")));

        let mut changed = metadata;
        changed
            .resolve
            .nodes
            .iter_mut()
            .find(|node| node.id == "serde")
            .unwrap()
            .features
            .push("rc".to_owned());
        assert_ne!(active_cargo_closure(&changed, "root").unwrap(), closure);
    }

    #[test]
    fn active_cargo_closure_rejects_unpinned_transitive_path_packages() {
        let metadata = Metadata {
            packages: vec![
                MetadataPackage {
                    id: "root".to_owned(),
                    name: "runner".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                },
                MetadataPackage {
                    id: "local".to_owned(),
                    name: "local".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                },
            ],
            resolve: MetadataResolve {
                root: Some("root".to_owned()),
                nodes: vec![
                    MetadataNode {
                        id: "root".to_owned(),
                        features: vec![],
                        deps: vec![MetadataDependency {
                            name: "local".to_owned(),
                            pkg: "local".to_owned(),
                            dep_kinds: vec![],
                        }],
                    },
                    MetadataNode {
                        id: "local".to_owned(),
                        features: vec![],
                        deps: vec![],
                    },
                ],
            },
        };
        assert!(active_cargo_closure(&metadata, "root").is_err());
    }
}
