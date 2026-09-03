use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
/// Producer-owned consumer facts. External callers can inspect or serialize
/// the fragment, but cannot hand-construct one.
///
/// ```compile_fail
/// use mlxcel_build_identity::ConsumerFragment;
/// let _ = ConsumerFragment {};
/// ```
///
/// ```compile_fail
/// use mlxcel_build_identity::ConsumerFragment;
/// let _: ConsumerFragment = serde_json::from_str("{}").unwrap();
/// ```
pub struct ConsumerFragment {
    schema: String,
    canonical_lineage_repository: String,
    approved_repository: String,
    manifest_repository: String,
    manifest_revision: String,
    manifest_default_features: bool,
    manifest_explicit_features: Vec<String>,
    resolved_package_name: String,
    resolved_package_version: String,
    resolved_package_source: String,
    resolved_package_checksum: Option<String>,
    cargo_lock_sha256: String,
    effective_cargo_features: Vec<String>,
    rust_toolchain: String,
    rustc: String,
    final_linker: String,
    target_triple: String,
    target_architecture: String,
    build_profile: String,
    effective_rustflags: Vec<String>,
    emitted_link_directives: Vec<String>,
    canonical_digest: String,
}

impl ConsumerFragment {
    pub fn canonical_lineage_repository(&self) -> &str {
        &self.canonical_lineage_repository
    }

    pub fn approved_repository(&self) -> &str {
        &self.approved_repository
    }

    pub fn manifest_repository(&self) -> &str {
        &self.manifest_repository
    }

    pub fn manifest_revision(&self) -> &str {
        &self.manifest_revision
    }

    pub fn emitted_link_directives(&self) -> &[String] {
        &self.emitted_link_directives
    }

    pub fn canonical_digest(&self) -> &str {
        &self.canonical_digest
    }
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
    workspace_root: PathBuf,
    workspace_members: Vec<String>,
    resolve: MetadataResolve,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
    manifest_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CargoLock {
    #[serde(default)]
    package: Vec<CargoLockPackage>,
}

#[derive(Debug, Deserialize)]
struct CargoLockPackage {
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
    let mut metadata = metadata;
    restore_registry_checksums(&mut metadata, &config.cargo_lock_path)?;
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

    let workspace = validate_workspace_authority(
        &metadata,
        root,
        &config.manifest_path,
        &config.cargo_lock_path,
    )?;
    let workspace_tree_digest = canonical::git_candidate_tree_digest(&workspace.root)?;
    let active_cargo_closure = active_cargo_closure(&metadata, root, &workspace)?;

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
        format!(
            "workspace-candidate-tree-sha256={}",
            canonical::hex_digest(workspace_tree_digest)
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

fn restore_registry_checksums(metadata: &mut Metadata, lock_path: &Path) -> Result<(), String> {
    let lock_text = fs::read_to_string(lock_path).map_err(|err| {
        format!(
            "cannot read Cargo.lock {} while restoring package checksums: {err}",
            lock_path.display()
        )
    })?;
    let lock: CargoLock = toml::from_str(&lock_text)
        .map_err(|err| format!("Cargo.lock is invalid while restoring package checksums: {err}"))?;
    let mut checksums = BTreeMap::new();
    for package in lock.package {
        let Some(source) = package.source else {
            continue;
        };
        let key = (package.name, package.version, source);
        if checksums.insert(key, package.checksum).is_some() {
            return Err("Cargo.lock contains duplicate package checksum records".to_owned());
        }
    }
    for package in &mut metadata.packages {
        let Some(source) = package.source.as_deref() else {
            continue;
        };
        if !source.starts_with("registry+") {
            continue;
        }
        let key = (
            package.name.clone(),
            package.version.clone(),
            source.to_owned(),
        );
        let checksum = checksums
            .get(&key)
            .and_then(Option::as_deref)
            .ok_or_else(|| {
                format!(
                    "Cargo.lock has no checksum for registry package {:?} {}",
                    package.name, package.version
                )
            })?;
        if package
            .checksum
            .as_deref()
            .is_some_and(|value| value != checksum)
        {
            return Err(format!(
                "cargo metadata checksum disagrees with Cargo.lock for registry package {:?} {}",
                package.name, package.version
            ));
        }
        package.checksum = Some(checksum.to_owned());
    }
    Ok(())
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
    validate_fragment_digest(fragment)?;
    let digest = canonical::parse_sha256(&fragment.cargo_lock_sha256, "Cargo.lock")?;
    let features = render_string_slice(&fragment.manifest_explicit_features);
    let effective = render_string_slice(&fragment.effective_cargo_features);
    let directives = render_string_slice(&fragment.emitted_link_directives);
    Ok(format!(
        "pub fn generated_mlxcel_consumer_fragment() -> Result<mlxcel::native_runtime::NativeConsumerBuildFragment, mlxcel::native_runtime::NativeRuntimeError> {{\n\
         mlxcel::native_runtime::NativeConsumerBuildFragment::try_from_generated(mlxcel::native_runtime::NativeConsumerBuildRecord {{\n\
         canonical_lineage_repository: {:?}, approved_repository: {:?},\n\
         manifest_repository: {:?}, manifest_revision: {:?}, manifest_default_features: false,\n\
         manifest_explicit_features: {}, resolved_package_name: {:?},\n\
         resolved_package_version: {:?}, resolved_package_source: {:?},\n\
         resolved_package_checksum: {}, cargo_lock_sha256: {:?},\n\
         effective_cargo_features: {}, rust_toolchain: {:?}, rustc: {:?},\n\
         final_linker: {:?}, target_triple: {:?}, target_architecture: {:?},\n\
         build_profile: {:?}, emitted_link_directives: {},\n\
         }})\n}}\n",
        fragment.canonical_lineage_repository,
        fragment.approved_repository,
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

struct WorkspaceAuthority {
    root: PathBuf,
    members: BTreeSet<String>,
}

fn validate_workspace_authority(
    metadata: &Metadata,
    root: &str,
    configured_manifest: &Path,
    configured_lock: &Path,
) -> Result<WorkspaceAuthority, String> {
    if metadata.workspace_members.len() > MAX_CANONICAL_ITEMS {
        return Err("Cargo workspace member list exceeds the item bound".to_owned());
    }
    let members = metadata
        .workspace_members
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if members.len() != metadata.workspace_members.len() {
        return Err("Cargo metadata contains duplicate workspace members".to_owned());
    }
    if !members.contains(root) {
        return Err("runner root is not a declared Cargo workspace member".to_owned());
    }
    let workspace_root = fs::canonicalize(&metadata.workspace_root).map_err(|err| {
        format!(
            "cannot canonicalize Cargo workspace root {}: {err}",
            metadata.workspace_root.display()
        )
    })?;
    let git_root = git_workspace_root(&workspace_root)?;
    if git_root != workspace_root {
        return Err("Cargo workspace root is not the authoritative Git worktree root".to_owned());
    }
    let root_package = metadata
        .packages
        .iter()
        .find(|package| package.id == root)
        .ok_or_else(|| "runner root package is absent from Cargo metadata".to_owned())?;
    let metadata_manifest = canonical_workspace_manifest(root_package, &workspace_root)?;
    let configured_manifest = fs::canonicalize(configured_manifest).map_err(|err| {
        format!(
            "cannot canonicalize configured runner manifest {}: {err}",
            configured_manifest.display()
        )
    })?;
    if metadata_manifest.0 != configured_manifest {
        return Err(
            "Cargo metadata runner manifest disagrees with producer configuration".to_owned(),
        );
    }
    let expected_lock = fs::canonicalize(workspace_root.join("Cargo.lock"))
        .map_err(|err| format!("cannot canonicalize Cargo workspace lockfile: {err}"))?;
    let configured_lock = fs::canonicalize(configured_lock).map_err(|err| {
        format!(
            "cannot canonicalize configured Cargo.lock {}: {err}",
            configured_lock.display()
        )
    })?;
    if configured_lock != expected_lock {
        return Err("configured Cargo.lock is not the Cargo workspace lockfile".to_owned());
    }
    Ok(WorkspaceAuthority {
        root: workspace_root,
        members,
    })
}

fn git_workspace_root(workspace_root: &Path) -> Result<PathBuf, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace_root)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|err| format!("cannot execute Git for Cargo workspace authority: {err}"))?;
    if !output.status.success() {
        return Err("Cargo workspace has no Git source authority".to_owned());
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| "Cargo workspace Git root is not strict UTF-8".to_owned())?;
    fs::canonicalize(text.trim())
        .map_err(|err| format!("cannot canonicalize Cargo workspace Git root: {err}"))
}

fn canonical_workspace_manifest(
    package: &MetadataPackage,
    workspace_root: &Path,
) -> Result<(PathBuf, String), String> {
    let manifest = fs::canonicalize(&package.manifest_path).map_err(|err| {
        format!(
            "cannot canonicalize workspace package {:?} manifest {}: {err}",
            package.name,
            package.manifest_path.display()
        )
    })?;
    let relative = manifest.strip_prefix(workspace_root).map_err(|_| {
        format!(
            "workspace package {:?} manifest is outside the Cargo workspace root",
            package.name
        )
    })?;
    let label = canonical::strict_relative_path(relative)?;
    Ok((manifest, format!("workspace:{label}")))
}

fn active_cargo_closure(
    metadata: &Metadata,
    root: &str,
    workspace: &WorkspaceAuthority,
) -> Result<Vec<String>, String> {
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
        let workspace_source;
        let source = match package.source.as_deref() {
            Some(source) => source,
            None if workspace.members.contains(*id) => {
                workspace_source = canonical_workspace_manifest(package, &workspace.root)?.1;
                workspace_source.as_str()
            }
            None => {
                return Err(format!(
                    "active Cargo dependency {:?} is an unlisted path package",
                    package.name
                ));
            }
        };
        validate_resolved_package_source(source, package.checksum.as_deref())?;
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

fn validate_resolved_package_source(source: &str, checksum: Option<&str>) -> Result<(), String> {
    bounded_text(source, "active Cargo package source")?;
    if source.starts_with("workspace:") {
        let label = source
            .strip_prefix("workspace:")
            .ok_or_else(|| "workspace source label is malformed".to_owned())?;
        canonical::validate_relative_label(label)?;
        if checksum.is_some() {
            return Err("workspace package unexpectedly has a checksum".to_owned());
        }
        return Ok(());
    }
    if source.starts_with("registry+") {
        let registry = source
            .strip_prefix("registry+")
            .ok_or_else(|| "registry package source is malformed".to_owned())?;
        let registry = Url::parse(registry)
            .map_err(|err| format!("registry package source URL is invalid: {err}"))?;
        if registry.scheme() != "https"
            || !registry.username().is_empty()
            || registry.password().is_some()
            || registry.query().is_some()
            || registry.fragment().is_some()
        {
            return Err("registry package source URL is not portable HTTPS authority".to_owned());
        }
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
        parse_git_cargo_source(source)?;
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
    if lineage != "https://github.com/lablup/mlxcel.git" {
        return Err("canonical lineage URL is not the fixed mlxcel upstream authority".to_owned());
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

fn validate_fragment_digest(fragment: &ConsumerFragment) -> Result<(), String> {
    if fragment.schema != "mlxcel.consumer-build.v1" {
        return Err("consumer fragment schema is not canonical".to_owned());
    }
    let expected = canonical::hex_digest(digest_fragment(fragment));
    if fragment.canonical_digest != expected {
        return Err(
            "consumer fragment canonical digest is empty or does not match its facts".to_owned(),
        );
    }
    Ok(())
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

    fn renderer_fixture() -> ConsumerFragment {
        let mut fragment = ConsumerFragment {
            schema: "mlxcel.consumer-build.v1".to_owned(),
            canonical_lineage_repository: "https://github.com/lablup/mlxcel.git".to_owned(),
            approved_repository: "https://github.com/freestylejs/snowbrain-mlxcel.git".to_owned(),
            manifest_repository: "https://github.com/freestylejs/snowbrain-mlxcel.git".to_owned(),
            manifest_revision: SHA.to_owned(),
            manifest_default_features: false,
            manifest_explicit_features: vec!["metal".to_owned()],
            resolved_package_name: "mlxcel".to_owned(),
            resolved_package_version: "0.6.0".to_owned(),
            resolved_package_source: format!(
                "git+https://github.com/freestylejs/snowbrain-mlxcel.git?rev={SHA}#{SHA}"
            ),
            resolved_package_checksum: None,
            cargo_lock_sha256: "11".repeat(32),
            effective_cargo_features: vec!["metal".to_owned()],
            rust_toolchain: "stable-aarch64-apple-darwin".to_owned(),
            rustc: "rustc fixture".to_owned(),
            final_linker: "ld fixture".to_owned(),
            target_triple: "aarch64-apple-darwin".to_owned(),
            target_architecture: "aarch64".to_owned(),
            build_profile: "release".to_owned(),
            effective_rustflags: Vec::new(),
            emitted_link_directives: vec!["lib=framework=Metal".to_owned()],
            canonical_digest: String::new(),
        };
        fragment.canonical_digest = canonical::hex_digest(digest_fragment(&fragment));
        fragment
    }

    #[test]
    fn renderer_recomputes_schema_and_digest() {
        let mut fragment = renderer_fixture();
        assert!(render_consumer_fragment_rust(&fragment).is_ok());
        fragment.final_linker = "injected linker".to_owned();
        assert!(render_consumer_fragment_rust(&fragment).is_err());
        fragment = renderer_fixture();
        fragment.canonical_digest.clear();
        assert!(render_consumer_fragment_rust(&fragment).is_err());
        fragment = renderer_fixture();
        fragment.schema = "manual.consumer.v1".to_owned();
        fragment.canonical_digest = canonical::hex_digest(digest_fragment(&fragment));
        assert!(render_consumer_fragment_rust(&fragment).is_err());
    }

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
    fn active_cargo_sources_reject_local_credentials_and_floating_git() {
        let checksum = "11".repeat(32);
        for (source, checksum) in [
            (
                format!("git+file:///cargo/cache/mlxcel?rev={SHA}#{SHA}"),
                None,
            ),
            (
                format!("git+https://user:secret@example.test/mlxcel?rev={SHA}#{SHA}"),
                None,
            ),
            (
                format!("git+https://example.test/mlxcel?branch=main#{SHA}"),
                None,
            ),
            (
                "registry+file:///cargo/cache/index".to_owned(),
                Some(checksum.as_str()),
            ),
            (
                "registry+https://user:secret@example.test/index".to_owned(),
                Some(checksum.as_str()),
            ),
        ] {
            assert!(
                validate_resolved_package_source(&source, checksum).is_err(),
                "accepted non-portable source {source:?}"
            );
        }
    }

    #[test]
    fn workspace_path_authority_rejects_outside_escape_and_missing_git() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_manifest = outside.path().join("Cargo.toml");
        fs::write(
            &outside_manifest,
            "[package]\nname='outside'\nversion='0.1.0'\n",
        )
        .unwrap();
        let package = MetadataPackage {
            id: "outside".to_owned(),
            name: "outside".to_owned(),
            version: "0.1.0".to_owned(),
            source: None,
            checksum: None,
            manifest_path: outside_manifest.clone(),
        };
        assert!(canonical_workspace_manifest(&package, workspace.path()).is_err());

        let link = workspace.path().join("escaped-Cargo.toml");
        std::os::unix::fs::symlink(&outside_manifest, &link).unwrap();
        let escaped = MetadataPackage {
            manifest_path: link,
            ..package
        };
        assert!(canonical_workspace_manifest(&escaped, workspace.path()).is_err());
        assert!(git_workspace_root(workspace.path()).is_err());
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
        let directory = tempfile::tempdir().unwrap();
        let root_manifest = directory.path().join("Cargo.toml");
        fs::write(
            &root_manifest,
            "[package]\nname='runner'\nversion='0.1.0'\n",
        )
        .unwrap();
        let workspace = WorkspaceAuthority {
            root: fs::canonicalize(directory.path()).unwrap(),
            members: BTreeSet::from(["root".to_owned()]),
        };
        let git_source = format!("git+https://example.test/mlxcel.git?rev={SHA}#{SHA}");
        let metadata = Metadata {
            packages: vec![
                MetadataPackage {
                    id: "root".to_owned(),
                    name: "runner".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                    manifest_path: root_manifest,
                },
                MetadataPackage {
                    id: "mlxcel".to_owned(),
                    name: "mlxcel".to_owned(),
                    version: "0.4.0".to_owned(),
                    source: Some(git_source),
                    checksum: None,
                    manifest_path: PathBuf::from("/cargo/mlxcel/Cargo.toml"),
                },
                MetadataPackage {
                    id: "serde".to_owned(),
                    name: "serde".to_owned(),
                    version: "1.0.0".to_owned(),
                    source: Some(
                        "registry+https://github.com/rust-lang/crates.io-index".to_owned(),
                    ),
                    checksum: Some("11".repeat(32)),
                    manifest_path: PathBuf::from("/cargo/serde/Cargo.toml"),
                },
                MetadataPackage {
                    id: "dev-only".to_owned(),
                    name: "dev-only".to_owned(),
                    version: "1.0.0".to_owned(),
                    source: Some(
                        "registry+https://github.com/rust-lang/crates.io-index".to_owned(),
                    ),
                    checksum: Some("22".repeat(32)),
                    manifest_path: PathBuf::from("/cargo/dev-only/Cargo.toml"),
                },
            ],
            workspace_root: directory.path().to_path_buf(),
            workspace_members: vec!["root".to_owned()],
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

        let closure = active_cargo_closure(&metadata, "root", &workspace).unwrap();
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
        assert_ne!(
            active_cargo_closure(&changed, "root", &workspace).unwrap(),
            closure
        );
    }

    #[test]
    fn active_cargo_closure_rejects_unpinned_transitive_path_packages() {
        let directory = tempfile::tempdir().unwrap();
        let root_manifest = directory.path().join("Cargo.toml");
        fs::write(
            &root_manifest,
            "[package]\nname='runner'\nversion='0.1.0'\n",
        )
        .unwrap();
        let workspace = WorkspaceAuthority {
            root: fs::canonicalize(directory.path()).unwrap(),
            members: BTreeSet::from(["root".to_owned()]),
        };
        let metadata = Metadata {
            packages: vec![
                MetadataPackage {
                    id: "root".to_owned(),
                    name: "runner".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                    manifest_path: root_manifest,
                },
                MetadataPackage {
                    id: "local".to_owned(),
                    name: "local".to_owned(),
                    version: "0.1.0".to_owned(),
                    source: None,
                    checksum: None,
                    manifest_path: directory.path().join("local/Cargo.toml"),
                },
            ],
            workspace_root: directory.path().to_path_buf(),
            workspace_members: vec!["root".to_owned()],
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
        assert!(active_cargo_closure(&metadata, "root", &workspace).is_err());
    }

    #[test]
    fn registry_checksums_restore_from_exact_lock_records_and_fail_closed() {
        const REGISTRY: &str = "registry+https://github.com/rust-lang/crates.io-index";
        let lock = |records: &str| {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("Cargo.lock");
            std::fs::write(&path, format!("version = 4\n\n{records}")).unwrap();
            (directory, path)
        };
        let metadata = || Metadata {
            packages: vec![MetadataPackage {
                id: "serde".to_owned(),
                name: "serde".to_owned(),
                version: "1.0.0".to_owned(),
                source: Some(REGISTRY.to_owned()),
                checksum: None,
                manifest_path: PathBuf::from("/cargo/serde/Cargo.toml"),
            }],
            workspace_root: PathBuf::from("/workspace"),
            workspace_members: vec![],
            resolve: MetadataResolve {
                root: None,
                nodes: vec![],
            },
        };
        let record = |source: &str, checksum: Option<&str>| {
            format!(
                "[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\nsource = \"{source}\"\n{}\n",
                checksum.map_or_else(String::new, |value| format!("checksum = \"{value}\""))
            )
        };
        let ones = "11".repeat(32);
        let twos = "22".repeat(32);

        let (directory, path) = lock(&record(REGISTRY, Some(ones.as_str())));
        let mut valid = metadata();
        restore_registry_checksums(&mut valid, &path).unwrap();
        assert_eq!(valid.packages[0].checksum.as_deref(), Some(ones.as_str()));
        drop(directory);

        let (directory, path) = lock(&record(REGISTRY, None));
        assert!(restore_registry_checksums(&mut metadata(), &path).is_err());
        drop(directory);

        let (directory, path) = lock(&record(REGISTRY, Some(twos.as_str())));
        let mut conflicting = metadata();
        conflicting.packages[0].checksum = Some(ones.clone());
        assert!(restore_registry_checksums(&mut conflicting, &path).is_err());
        drop(directory);

        let (directory, path) = lock(&format!(
            "{}{}",
            record(REGISTRY, Some(ones.as_str())),
            record(REGISTRY, Some(ones.as_str()))
        ));
        assert!(restore_registry_checksums(&mut metadata(), &path).is_err());
        drop(directory);

        let (directory, path) = lock(&record(
            "registry+https://other.example/index",
            Some(ones.as_str()),
        ));
        assert!(restore_registry_checksums(&mut metadata(), &path).is_err());
        drop(directory);
    }
}
