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

//! Canonical native build identity and bounded worker-profile projection.

use std::collections::HashSet;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::detokenizer::{
    IncrementalDecodeGuarantee, MAX_PENDING_DECODE_BYTES, MAX_PROVEN_LOOKBEHIND_TOKENS,
    MAX_STABLE_DELTA_BYTES,
};
use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::model_report::NativeModalitySet;
use super::operation_driver::NativeOperationDriver;
use super::request::{
    MAX_NATIVE_CONSTRAINT_BYTES, MAX_NATIVE_EMBED_INPUTS, MAX_NATIVE_PREPARED_BYTES,
    MAX_NATIVE_PREPARED_MEDIA_ITEMS, MAX_NATIVE_PROMPT_TOKENS,
};
use super::sink::{
    MAX_NATIVE_ACCEPTED_UNITS, MAX_NATIVE_EMBEDDING_DIMENSION, MAX_NATIVE_STABLE_BYTES_PER_UNIT,
};

const _: () = assert!(MAX_STABLE_DELTA_BYTES <= MAX_NATIVE_STABLE_BYTES_PER_UNIT);
const _: () = assert!(MAX_PENDING_DECODE_BYTES <= MAX_NATIVE_STABLE_BYTES_PER_UNIT);

const BUILD_IDENTITY_SCHEMA: &str = "mlxcel.native-build.v2";
const WORKER_PROFILE_SCHEMA: &str = "mlxcel.native-worker-profile.v1";
const CANONICAL_MLXCEL_REPOSITORY: &str = "https://github.com/lablup/mlxcel.git";
pub const MAX_BUILD_ITEMS: usize = 4_096;
pub const MAX_BUILD_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_BUILD_TOTAL_TEXT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuildSource {
    label: Arc<str>,
    sha256: [u8; 32],
}

impl NativeBuildSource {
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeNamedBuildInput {
    name: Arc<str>,
    value: Arc<str>,
}

impl NativeNamedBuildInput {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeResolvedCargoPackage {
    name: Arc<str>,
    version: Arc<str>,
    source: Arc<str>,
    checksum: Option<Arc<str>>,
}

impl NativeResolvedCargoPackage {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    #[must_use]
    pub fn checksum(&self) -> Option<&str> {
        self.checksum.as_deref()
    }
}

/// Package facts a dependency can truthfully report about itself. Cargo
/// resolution source/checksum belong exclusively to the consumer fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSelfBuildPackage {
    name: Arc<str>,
    version: Arc<str>,
}

impl NativeSelfBuildPackage {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Borrowed record emitted by the consuming runner's build producer.
///
/// This input exists because a dependency build script cannot observe the
/// consumer's direct dependency declaration, root lockfile, or final-link
/// recipe. Construction validates every bound before retaining a copy.
#[derive(Debug, Clone, Copy)]
pub struct NativeConsumerBuildRecord<'a> {
    pub manifest_repository: &'a str,
    pub manifest_revision: &'a str,
    pub manifest_default_features: bool,
    pub manifest_explicit_features: &'a [&'a str],
    pub resolved_package_name: &'a str,
    pub resolved_package_version: &'a str,
    pub resolved_package_source: &'a str,
    pub resolved_package_checksum: Option<&'a str>,
    pub cargo_lock_sha256: [u8; 32],
    pub effective_cargo_features: &'a [&'a str],
    pub rust_toolchain: &'a str,
    pub rustc: &'a str,
    pub final_linker: &'a str,
    pub target_triple: &'a str,
    pub target_architecture: &'a str,
    pub build_profile: &'a str,
    /// Ordered, canonical directives emitted by the consumer build producer.
    pub emitted_link_directives: &'a [&'a str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeConsumerBuildFragmentInner {
    manifest_repository: Arc<str>,
    manifest_revision: Arc<str>,
    manifest_default_features: bool,
    manifest_explicit_features: Vec<Arc<str>>,
    resolved_package: NativeResolvedCargoPackage,
    cargo_lock_sha256: [u8; 32],
    effective_cargo_features: Vec<Arc<str>>,
    rust_toolchain: Arc<str>,
    rustc: Arc<str>,
    final_linker: Arc<str>,
    target_triple: Arc<str>,
    target_architecture: Arc<str>,
    build_profile: Arc<str>,
    emitted_link_directives: Vec<Arc<str>>,
}

/// Immutable, cheaply cloned consumer-owned build authority fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeConsumerBuildFragment(Arc<NativeConsumerBuildFragmentInner>);

impl NativeConsumerBuildFragment {
    pub fn try_from_generated(
        record: NativeConsumerBuildRecord<'_>,
    ) -> Result<Self, NativeRuntimeError> {
        validate_consumer_record(&record)?;
        Ok(Self(Arc::new(NativeConsumerBuildFragmentInner {
            manifest_repository: Arc::from(record.manifest_repository),
            manifest_revision: Arc::from(record.manifest_revision),
            manifest_default_features: record.manifest_default_features,
            manifest_explicit_features: record
                .manifest_explicit_features
                .iter()
                .map(|value| Arc::from(*value))
                .collect(),
            resolved_package: NativeResolvedCargoPackage {
                name: Arc::from(record.resolved_package_name),
                version: Arc::from(record.resolved_package_version),
                source: Arc::from(record.resolved_package_source),
                checksum: record.resolved_package_checksum.map(Arc::from),
            },
            cargo_lock_sha256: record.cargo_lock_sha256,
            effective_cargo_features: record
                .effective_cargo_features
                .iter()
                .map(|value| Arc::from(*value))
                .collect(),
            rust_toolchain: Arc::from(record.rust_toolchain),
            rustc: Arc::from(record.rustc),
            final_linker: Arc::from(record.final_linker),
            target_triple: Arc::from(record.target_triple),
            target_architecture: Arc::from(record.target_architecture),
            build_profile: Arc::from(record.build_profile),
            emitted_link_directives: record
                .emitted_link_directives
                .iter()
                .map(|value| Arc::from(*value))
                .collect(),
        })))
    }

    #[must_use]
    pub fn manifest_repository(&self) -> &str {
        &self.0.manifest_repository
    }
    #[must_use]
    pub fn manifest_revision(&self) -> &str {
        &self.0.manifest_revision
    }
    #[must_use]
    pub fn manifest_default_features(&self) -> bool {
        self.0.manifest_default_features
    }
    #[must_use]
    pub fn manifest_explicit_features(&self) -> &[Arc<str>] {
        &self.0.manifest_explicit_features
    }
    #[must_use]
    pub fn resolved_package(&self) -> &NativeResolvedCargoPackage {
        &self.0.resolved_package
    }
    #[must_use]
    pub fn cargo_lock_sha256(&self) -> [u8; 32] {
        self.0.cargo_lock_sha256
    }
    #[must_use]
    pub fn effective_cargo_features(&self) -> &[Arc<str>] {
        &self.0.effective_cargo_features
    }
    #[must_use]
    pub fn rust_toolchain(&self) -> &str {
        &self.0.rust_toolchain
    }
    #[must_use]
    pub fn rustc(&self) -> &str {
        &self.0.rustc
    }
    #[must_use]
    pub fn final_linker(&self) -> &str {
        &self.0.final_linker
    }
    #[must_use]
    pub fn target_triple(&self) -> &str {
        &self.0.target_triple
    }
    #[must_use]
    pub fn target_architecture(&self) -> &str {
        &self.0.target_architecture
    }
    #[must_use]
    pub fn build_profile(&self) -> &str {
        &self.0.build_profile
    }
    #[must_use]
    pub fn emitted_link_directives(&self) -> &[Arc<str>] {
        &self.0.emitted_link_directives
    }
}

/// Borrowed post-link evidence emitted beside, never inside, the final
/// artifact. The artifact digest therefore cannot become self-referential.
#[derive(Debug, Clone, Copy)]
pub struct NativePostLinkRecord<'a> {
    pub pre_link_identity: [u8; 32],
    pub artifact_kind: &'a str,
    pub artifact_sha256: [u8; 32],
    pub final_link_trace_sha256: [u8; 32],
    pub link_map_sha256: Option<[u8; 32]>,
    pub observed_link_arguments: &'a [&'a str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativePostLinkAttestationInner {
    pre_link_identity: [u8; 32],
    artifact_kind: Arc<str>,
    artifact_sha256: [u8; 32],
    final_link_trace_sha256: [u8; 32],
    link_map_sha256: Option<[u8; 32]>,
    observed_link_arguments: Vec<Arc<str>>,
}

/// Immutable post-link sidecar authority bound to a pre-link identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePostLinkAttestation(Arc<NativePostLinkAttestationInner>);

impl NativePostLinkAttestation {
    pub fn try_from_generated(
        record: NativePostLinkRecord<'_>,
    ) -> Result<Self, NativeRuntimeError> {
        validate_post_link_record(&record)?;
        Ok(Self(Arc::new(NativePostLinkAttestationInner {
            pre_link_identity: record.pre_link_identity,
            artifact_kind: Arc::from(record.artifact_kind),
            artifact_sha256: record.artifact_sha256,
            final_link_trace_sha256: record.final_link_trace_sha256,
            link_map_sha256: record.link_map_sha256,
            observed_link_arguments: record
                .observed_link_arguments
                .iter()
                .map(|value| Arc::from(*value))
                .collect(),
        })))
    }

    #[must_use]
    pub fn pre_link_identity(&self) -> [u8; 32] {
        self.0.pre_link_identity
    }
    #[must_use]
    pub fn artifact_kind(&self) -> &str {
        &self.0.artifact_kind
    }
    #[must_use]
    pub fn artifact_sha256(&self) -> [u8; 32] {
        self.0.artifact_sha256
    }
    #[must_use]
    pub fn final_link_trace_sha256(&self) -> [u8; 32] {
        self.0.final_link_trace_sha256
    }
    #[must_use]
    pub fn link_map_sha256(&self) -> Option<[u8; 32]> {
        self.0.link_map_sha256
    }
    #[must_use]
    pub fn observed_link_arguments(&self) -> &[Arc<str>] {
        &self.0.observed_link_arguments
    }

    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        digest_post_link_attestation(self)
    }

    pub fn verify_pre_link(
        &self,
        identity: &NativeBuildIdentity,
    ) -> Result<(), NativeRuntimeError> {
        if self.pre_link_identity() != identity.digest() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::BuildIdentityMismatch,
                "post-link attestation was generated for a different pre-link identity",
            ));
        }
        Ok(())
    }
}

/// One exact native submodule, fetched repository, or vendored tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeNativeSourceKind {
    Git,
    Vendored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeNativeSource {
    kind: NativeNativeSourceKind,
    label: Arc<str>,
    canonical_repository: Arc<str>,
    revision: Option<Arc<str>>,
    pristine_tree_sha256: [u8; 32],
    effective_tree_sha256: [u8; 32],
}

impl NativeNativeSource {
    #[must_use]
    pub const fn kind(&self) -> NativeNativeSourceKind {
        self.kind
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn canonical_repository(&self) -> &str {
        &self.canonical_repository
    }

    #[must_use]
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    #[must_use]
    pub const fn tree_sha256(&self) -> [u8; 32] {
        self.effective_tree_sha256
    }

    #[must_use]
    pub const fn pristine_tree_sha256(&self) -> [u8; 32] {
        self.pristine_tree_sha256
    }

    #[must_use]
    pub const fn effective_tree_sha256(&self) -> [u8; 32] {
        self.effective_tree_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeLinkMode {
    Static,
    Dynamic,
    Mixed,
}

/// Exact inputs consumed by one native artifact build. Fields are private so
/// callers can inspect but cannot hand-author a replacement authority record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeEffectiveBuildInputs {
    canonical_source_repository: Arc<str>,
    materialized_source_repository: Arc<str>,
    outer_source_revision: Arc<str>,
    outer_source_tree_sha256: [u8; 32],
    self_package: NativeSelfBuildPackage,
    native_sources: Vec<NativeNativeSource>,
    ordered_patches: Vec<NativeBuildSource>,
    cxx_bridge_sources: Vec<NativeBuildSource>,
    overlay_sources: Vec<NativeBuildSource>,
    turbo_sources: Vec<NativeBuildSource>,
    self_cargo_lock_sha256: [u8; 32],
    rust_toolchain: Arc<str>,
    rustc: Arc<str>,
    c_compiler: Arc<str>,
    cxx_bridge_compiler: Arc<str>,
    cxx_compiler: Arc<str>,
    cmake: Arc<str>,
    linker: Arc<str>,
    rust_linker: Arc<str>,
    target_triple: Arc<str>,
    target_architecture: Arc<str>,
    build_profile: Arc<str>,
    declared_default_features: Vec<Arc<str>>,
    default_features_enabled: bool,
    cargo_features: Vec<Arc<str>>,
    native_features: Vec<Arc<str>>,
    cmake_definitions: Vec<NativeNamedBuildInput>,
    build_environment: Vec<NativeNamedBuildInput>,
    compile_flags: Vec<Arc<str>>,
    link_flags: Vec<Arc<str>>,
    sdk_inputs: Vec<NativeNamedBuildInput>,
    header_inputs: Vec<NativeBuildSource>,
    library_inputs: Vec<NativeBuildSource>,
    link_mode: NativeLinkMode,
    link_inputs: Vec<NativeBuildSource>,
    bundled_artifacts: Vec<NativeBuildSource>,
    root_link_flags: Vec<Arc<str>>,
    consumer: NativeConsumerBuildFragment,
    adapter_schema_revision: Arc<str>,
}

impl NativeEffectiveBuildInputs {
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        let mut bound = BuildBound::default();
        bound.text(
            "canonical source repository",
            &self.canonical_source_repository,
        )?;
        if self.canonical_source_repository.as_ref() != CANONICAL_MLXCEL_REPOSITORY {
            return Err(invalid_identity("mlxcel upstream lineage is not canonical"));
        }
        bound.text(
            "materialized source repository",
            &self.materialized_source_repository,
        )?;
        bound.full_git_sha("outer source revision", &self.outer_source_revision)?;
        require_nonzero_digest("outer source tree", self.outer_source_tree_sha256)?;
        require_nonzero_digest("self Cargo.lock", self.self_cargo_lock_sha256)?;
        bound.text("self package name", &self.self_package.name)?;
        bound.text("self package version", &self.self_package.version)?;
        validate_native_sources(&self.native_sources, &mut bound)?;
        validate_ordered_sources("ordered patch", &self.ordered_patches, &mut bound)?;
        validate_ordered_sources("CXX bridge source", &self.cxx_bridge_sources, &mut bound)?;
        validate_sorted_sources("overlay source", &self.overlay_sources, &mut bound)?;
        validate_ordered_sources("turbo source", &self.turbo_sources, &mut bound)?;

        for (name, value) in [
            ("Rust toolchain", &self.rust_toolchain),
            ("rustc", &self.rustc),
            ("C compiler", &self.c_compiler),
            ("CXX bridge compiler", &self.cxx_bridge_compiler),
            ("C++ compiler", &self.cxx_compiler),
            ("CMake", &self.cmake),
            ("linker", &self.linker),
            ("Rust final linker", &self.rust_linker),
            ("target triple", &self.target_triple),
            ("target architecture", &self.target_architecture),
            ("build profile", &self.build_profile),
            ("adapter schema revision", &self.adapter_schema_revision),
        ] {
            bound.text(name, value)?;
        }

        validate_sorted_texts(
            "declared default Cargo feature",
            &self.declared_default_features,
            &mut bound,
        )?;
        validate_sorted_texts("Cargo feature", &self.cargo_features, &mut bound)?;
        if self.default_features_enabled
            != self
                .cargo_features
                .iter()
                .any(|feature| feature.as_ref() == "default")
        {
            return Err(invalid_identity(
                "CARGO_FEATURE_DEFAULT state disagrees with the effective Cargo feature tuple",
            ));
        }
        validate_sorted_texts("native feature", &self.native_features, &mut bound)?;
        validate_named_inputs("CMake definition", &self.cmake_definitions, &mut bound)?;
        validate_named_inputs("build environment", &self.build_environment, &mut bound)?;
        validate_ordered_texts("compile flag", &self.compile_flags, &mut bound)?;
        validate_ordered_texts("link flag", &self.link_flags, &mut bound)?;
        validate_ordered_texts("root link flag", &self.root_link_flags, &mut bound)?;
        reject_ephemeral_flag_paths("compile flag", &self.compile_flags)?;
        reject_ephemeral_flag_paths("link flag", &self.link_flags)?;
        reject_ephemeral_flag_paths("root link flag", &self.root_link_flags)?;
        validate_named_inputs("SDK input", &self.sdk_inputs, &mut bound)?;
        validate_sorted_sources("header input", &self.header_inputs, &mut bound)?;
        validate_sorted_sources("library input", &self.library_inputs, &mut bound)?;
        validate_sorted_sources("link input", &self.link_inputs, &mut bound)?;
        validate_sorted_sources("bundled artifact", &self.bundled_artifacts, &mut bound)?;
        validate_consumer_fragment(&self.consumer, &mut bound)?;
        validate_consumer_matches_self(self)?;
        Ok(())
    }

    #[must_use]
    pub fn canonical_source_repository(&self) -> &str {
        &self.canonical_source_repository
    }

    #[must_use]
    pub fn materialized_source_repository(&self) -> &str {
        &self.materialized_source_repository
    }

    #[must_use]
    pub fn outer_source_revision(&self) -> &str {
        &self.outer_source_revision
    }

    #[must_use]
    pub const fn outer_source_tree_sha256(&self) -> [u8; 32] {
        self.outer_source_tree_sha256
    }

    #[must_use]
    pub fn self_package(&self) -> &NativeSelfBuildPackage {
        &self.self_package
    }

    #[must_use]
    pub fn resolved_direct_package(&self) -> &NativeResolvedCargoPackage {
        self.consumer.resolved_package()
    }

    #[must_use]
    pub fn native_sources(&self) -> &[NativeNativeSource] {
        &self.native_sources
    }

    #[must_use]
    pub fn ordered_patches(&self) -> &[NativeBuildSource] {
        &self.ordered_patches
    }

    #[must_use]
    pub fn cxx_bridge_sources(&self) -> &[NativeBuildSource] {
        &self.cxx_bridge_sources
    }

    #[must_use]
    pub fn overlay_sources(&self) -> &[NativeBuildSource] {
        &self.overlay_sources
    }

    #[must_use]
    pub fn turbo_sources(&self) -> &[NativeBuildSource] {
        &self.turbo_sources
    }

    #[must_use]
    pub const fn self_cargo_lock_sha256(&self) -> [u8; 32] {
        self.self_cargo_lock_sha256
    }

    #[must_use]
    pub fn cargo_lock_sha256(&self) -> [u8; 32] {
        self.consumer.cargo_lock_sha256()
    }

    #[must_use]
    pub fn rust_toolchain(&self) -> &str {
        &self.rust_toolchain
    }

    #[must_use]
    pub fn rustc(&self) -> &str {
        &self.rustc
    }

    #[must_use]
    pub fn c_compiler(&self) -> &str {
        &self.c_compiler
    }

    #[must_use]
    pub fn cxx_bridge_compiler(&self) -> &str {
        &self.cxx_bridge_compiler
    }

    #[must_use]
    pub fn cxx_compiler(&self) -> &str {
        &self.cxx_compiler
    }

    #[must_use]
    pub fn cmake(&self) -> &str {
        &self.cmake
    }

    #[must_use]
    pub fn linker(&self) -> &str {
        &self.linker
    }

    #[must_use]
    pub fn rust_linker(&self) -> &str {
        &self.rust_linker
    }

    #[must_use]
    pub fn target_triple(&self) -> &str {
        &self.target_triple
    }

    #[must_use]
    pub fn target_architecture(&self) -> &str {
        &self.target_architecture
    }

    #[must_use]
    pub fn build_profile(&self) -> &str {
        &self.build_profile
    }

    #[must_use]
    pub fn declared_default_features(&self) -> &[Arc<str>] {
        &self.declared_default_features
    }

    #[must_use]
    pub const fn default_features_enabled(&self) -> bool {
        self.default_features_enabled
    }

    #[must_use]
    pub fn cargo_features(&self) -> &[Arc<str>] {
        &self.cargo_features
    }

    #[must_use]
    pub fn native_features(&self) -> &[Arc<str>] {
        &self.native_features
    }

    #[must_use]
    pub fn cmake_definitions(&self) -> &[NativeNamedBuildInput] {
        &self.cmake_definitions
    }

    #[must_use]
    pub fn build_environment(&self) -> &[NativeNamedBuildInput] {
        &self.build_environment
    }

    #[must_use]
    pub fn compile_flags(&self) -> &[Arc<str>] {
        &self.compile_flags
    }

    #[must_use]
    pub fn link_flags(&self) -> &[Arc<str>] {
        &self.link_flags
    }

    #[must_use]
    pub fn sdk_inputs(&self) -> &[NativeNamedBuildInput] {
        &self.sdk_inputs
    }

    #[must_use]
    pub fn header_inputs(&self) -> &[NativeBuildSource] {
        &self.header_inputs
    }

    #[must_use]
    pub fn library_inputs(&self) -> &[NativeBuildSource] {
        &self.library_inputs
    }

    #[must_use]
    pub const fn link_mode(&self) -> NativeLinkMode {
        self.link_mode
    }

    #[must_use]
    pub fn link_inputs(&self) -> &[NativeBuildSource] {
        &self.link_inputs
    }

    #[must_use]
    pub fn bundled_artifacts(&self) -> &[NativeBuildSource] {
        &self.bundled_artifacts
    }

    #[must_use]
    pub fn root_link_flags(&self) -> &[Arc<str>] {
        &self.root_link_flags
    }

    #[must_use]
    pub fn consumer(&self) -> &NativeConsumerBuildFragment {
        &self.consumer
    }

    #[must_use]
    pub fn adapter_schema_revision(&self) -> &str {
        &self.adapter_schema_revision
    }
}

fn invalid_identity(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidBuildIdentity, message)
}

fn require_nonzero_digest(name: &str, digest: [u8; 32]) -> Result<(), NativeRuntimeError> {
    if digest == [0; 32] {
        return Err(invalid_identity(format!(
            "{name} requires a non-zero SHA-256 digest"
        )));
    }
    Ok(())
}

#[derive(Default)]
struct BuildBound {
    items: usize,
    text_bytes: usize,
}

impl BuildBound {
    fn text(&mut self, name: &str, value: &str) -> Result<(), NativeRuntimeError> {
        if value.is_empty() || value.len() > MAX_BUILD_TEXT_BYTES {
            return Err(invalid_identity(format!(
                "{name} is empty or exceeds {MAX_BUILD_TEXT_BYTES} bytes"
            )));
        }
        self.items = self
            .items
            .checked_add(1)
            .ok_or_else(|| invalid_identity("build input count overflowed"))?;
        self.text_bytes = self
            .text_bytes
            .checked_add(value.len())
            .ok_or_else(|| invalid_identity("build input text byte total overflowed"))?;
        if self.items > MAX_BUILD_ITEMS || self.text_bytes > MAX_BUILD_TOTAL_TEXT_BYTES {
            return Err(invalid_identity(
                "effective build record exceeds its aggregate bound",
            ));
        }
        Ok(())
    }

    fn full_git_sha(&mut self, name: &str, value: &str) -> Result<(), NativeRuntimeError> {
        self.text(name, value)?;
        if value.len() != 40
            || !value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(invalid_identity(format!(
                "{name} must be a full lowercase hexadecimal Git object ID"
            )));
        }
        Ok(())
    }
}

fn validate_consumer_record(
    record: &NativeConsumerBuildRecord<'_>,
) -> Result<(), NativeRuntimeError> {
    let mut bound = BuildBound::default();
    bound.text("consumer manifest repository", record.manifest_repository)?;
    bound.full_git_sha("consumer manifest revision", record.manifest_revision)?;
    for (name, value) in [
        ("resolved direct package name", record.resolved_package_name),
        (
            "resolved direct package version",
            record.resolved_package_version,
        ),
        (
            "resolved direct package source",
            record.resolved_package_source,
        ),
        ("consumer Rust toolchain", record.rust_toolchain),
        ("consumer rustc", record.rustc),
        ("consumer final linker", record.final_linker),
        ("consumer target triple", record.target_triple),
        ("consumer target architecture", record.target_architecture),
        ("consumer build profile", record.build_profile),
    ] {
        bound.text(name, value)?;
    }
    let parsed_source = parse_resolved_git_source(record.resolved_package_source);
    if record.resolved_package_name != "mlxcel"
        || parsed_source.is_none_or(|source| {
            source.repository.trim_end_matches(".git")
                != record.manifest_repository.trim_end_matches(".git")
                || source.requested_revision != record.manifest_revision
                || source.resolved_revision != record.manifest_revision
        })
        || record.resolved_package_checksum.is_some()
    {
        return Err(invalid_identity(
            "resolved direct mlxcel package does not match the exact Git manifest edge",
        ));
    }
    if let Some(checksum) = record.resolved_package_checksum {
        bound.text("resolved direct package checksum", checksum)?;
    }
    require_nonzero_digest("consumer Cargo.lock", record.cargo_lock_sha256)?;
    validate_borrowed_sorted_texts(
        "consumer manifest explicit feature",
        record.manifest_explicit_features,
        &mut bound,
    )?;
    validate_borrowed_sorted_texts(
        "consumer effective Cargo feature",
        record.effective_cargo_features,
        &mut bound,
    )?;
    let effective_default = record
        .effective_cargo_features
        .iter()
        .any(|feature| *feature == "default");
    if effective_default != record.manifest_default_features {
        return Err(invalid_identity(
            "consumer default-feature declaration disagrees with CARGO_FEATURE_DEFAULT",
        ));
    }
    for feature in record.manifest_explicit_features {
        if !record.effective_cargo_features.contains(feature) {
            return Err(invalid_identity(
                "consumer explicit feature is absent from the resolved effective feature tuple",
            ));
        }
    }
    if record.emitted_link_directives.is_empty() {
        return Err(invalid_identity(
            "consumer producer did not record its ordered final-link directives",
        ));
    }
    validate_borrowed_ordered_texts(
        "consumer emitted link directive",
        record.emitted_link_directives,
        &mut bound,
    )?;
    reject_ephemeral_borrowed_paths(
        "consumer emitted link directive",
        record.emitted_link_directives,
    )
}

struct ParsedResolvedGitSource<'a> {
    repository: &'a str,
    requested_revision: &'a str,
    resolved_revision: &'a str,
}

fn parse_resolved_git_source(source: &str) -> Option<ParsedResolvedGitSource<'_>> {
    let source = source.strip_prefix("git+")?;
    let (request, resolved_revision) = source.split_once('#')?;
    if resolved_revision.contains('#') || !is_full_lower_git_sha(resolved_revision) {
        return None;
    }
    let (repository, query) = request.split_once('?')?;
    if repository.is_empty() || query.is_empty() {
        return None;
    }
    let mut requested_revision = None;
    for pair in query.split('&') {
        let (name, value) = pair.split_once('=')?;
        if name == "rev" {
            if requested_revision.replace(value).is_some() || !is_full_lower_git_sha(value) {
                return None;
            }
        } else if name.is_empty() || value.is_empty() {
            return None;
        }
    }
    Some(ParsedResolvedGitSource {
        repository,
        requested_revision: requested_revision?,
        resolved_revision,
    })
}

fn is_full_lower_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate_consumer_fragment(
    fragment: &NativeConsumerBuildFragment,
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    for (name, value) in [
        (
            "consumer manifest repository",
            fragment.manifest_repository(),
        ),
        ("consumer manifest revision", fragment.manifest_revision()),
        (
            "resolved direct package name",
            fragment.resolved_package().name(),
        ),
        (
            "resolved direct package version",
            fragment.resolved_package().version(),
        ),
        (
            "resolved direct package source",
            fragment.resolved_package().source(),
        ),
        ("consumer Rust toolchain", fragment.rust_toolchain()),
        ("consumer rustc", fragment.rustc()),
        ("consumer final linker", fragment.final_linker()),
        ("consumer target triple", fragment.target_triple()),
        (
            "consumer target architecture",
            fragment.target_architecture(),
        ),
        ("consumer build profile", fragment.build_profile()),
    ] {
        bound.text(name, value)?;
    }
    if let Some(checksum) = fragment.resolved_package().checksum() {
        bound.text("resolved direct package checksum", checksum)?;
    }
    require_nonzero_digest("consumer Cargo.lock", fragment.cargo_lock_sha256())?;
    validate_sorted_texts(
        "consumer manifest explicit feature",
        fragment.manifest_explicit_features(),
        bound,
    )?;
    validate_sorted_texts(
        "consumer effective Cargo feature",
        fragment.effective_cargo_features(),
        bound,
    )?;
    validate_ordered_texts(
        "consumer emitted link directive",
        fragment.emitted_link_directives(),
        bound,
    )?;
    reject_ephemeral_flag_paths(
        "consumer emitted link directive",
        fragment.emitted_link_directives(),
    )
}

fn validate_consumer_matches_self(
    inputs: &NativeEffectiveBuildInputs,
) -> Result<(), NativeRuntimeError> {
    let consumer = &inputs.consumer;
    if inputs.materialized_source_repository.as_ref() != consumer.manifest_repository()
        || inputs.outer_source_revision.as_ref() != consumer.manifest_revision()
        || inputs.self_package.name() != consumer.resolved_package().name()
        || inputs.self_package.version() != consumer.resolved_package().version()
        || inputs.default_features_enabled != consumer.manifest_default_features()
        || inputs.cargo_features != consumer.effective_cargo_features()
        || inputs.rust_toolchain.as_ref() != consumer.rust_toolchain()
        || inputs.rustc.as_ref() != consumer.rustc()
        || inputs.target_triple.as_ref() != consumer.target_triple()
        || inputs.target_architecture.as_ref() != consumer.target_architecture()
        || inputs.build_profile.as_ref() != consumer.build_profile()
    {
        return Err(invalid_identity(
            "consumer fragment disagrees with the dependency's generated self record",
        ));
    }
    Ok(())
}

fn validate_post_link_record(record: &NativePostLinkRecord<'_>) -> Result<(), NativeRuntimeError> {
    require_nonzero_digest("pre-link identity", record.pre_link_identity)?;
    require_nonzero_digest("final artifact", record.artifact_sha256)?;
    require_nonzero_digest("final link trace", record.final_link_trace_sha256)?;
    if let Some(digest) = record.link_map_sha256 {
        require_nonzero_digest("final link map", digest)?;
    }
    let mut bound = BuildBound::default();
    bound.text("final artifact kind", record.artifact_kind)?;
    if record.observed_link_arguments.is_empty() {
        return Err(invalid_identity(
            "post-link attestation has no observed linker arguments",
        ));
    }
    validate_borrowed_ordered_texts(
        "observed final linker argument",
        record.observed_link_arguments,
        &mut bound,
    )?;
    reject_ephemeral_borrowed_paths(
        "observed final linker argument",
        record.observed_link_arguments,
    )
}

fn validate_borrowed_sorted_texts(
    name: &str,
    values: &[&str],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut previous = None;
    for value in values {
        bound.text(name, value)?;
        if previous.is_some_and(|prior| prior >= *value) {
            return Err(invalid_identity(format!(
                "{name} entries must be sorted and unique"
            )));
        }
        previous = Some(*value);
    }
    Ok(())
}

fn validate_borrowed_ordered_texts(
    name: &str,
    values: &[&str],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    for value in values {
        bound.text(name, value)?;
    }
    Ok(())
}

fn reject_ephemeral_borrowed_paths(name: &str, values: &[&str]) -> Result<(), NativeRuntimeError> {
    for value in values {
        reject_ephemeral_path(name, value)?;
    }
    Ok(())
}

fn validate_sorted_texts(
    name: &str,
    values: &[Arc<str>],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut previous: Option<&str> = None;
    for value in values {
        bound.text(name, value)?;
        if previous.is_some_and(|prior| prior >= value.as_ref()) {
            return Err(invalid_identity(format!(
                "{name} entries must be sorted and unique"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_ordered_texts(
    name: &str,
    values: &[Arc<str>],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    for value in values {
        bound.text(name, value)?;
    }
    Ok(())
}

fn reject_ephemeral_flag_paths(name: &str, values: &[Arc<str>]) -> Result<(), NativeRuntimeError> {
    for value in values {
        reject_ephemeral_path(name, value)?;
    }
    Ok(())
}

fn reject_ephemeral_path(name: &str, value: &str) -> Result<(), NativeRuntimeError> {
    let normalized = value.replace('\\', "/");
    let windows_absolute = normalized
        .as_bytes()
        .windows(3)
        .any(|window| window[0].is_ascii_alphabetic() && window[1] == b':' && window[2] == b'/');
    let unix_checkout_absolute = ["/Users/", "/home/", "/tmp/", "/private/"]
        .into_iter()
        .any(|prefix| normalized.contains(prefix));
    if windows_absolute || unix_checkout_absolute {
        return Err(invalid_identity(format!(
            "{name} contains an ephemeral absolute source/build path"
        )));
    }
    Ok(())
}

fn validate_sorted_sources(
    name: &str,
    values: &[NativeBuildSource],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut previous: Option<&str> = None;
    for source in values {
        bound.text(name, &source.label)?;
        require_nonzero_digest(name, source.sha256)?;
        if previous.is_some_and(|prior| prior >= source.label.as_ref()) {
            return Err(invalid_identity(format!(
                "{name} labels must be sorted and unique"
            )));
        }
        previous = Some(&source.label);
    }
    Ok(())
}

fn validate_ordered_sources(
    name: &str,
    values: &[NativeBuildSource],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut labels = HashSet::new();
    for source in values {
        bound.text(name, &source.label)?;
        require_nonzero_digest(name, source.sha256)?;
        if !labels.insert(source.label.as_ref()) {
            return Err(invalid_identity(format!(
                "{name} labels must be unique without changing application order"
            )));
        }
    }
    Ok(())
}

fn validate_named_inputs(
    name: &str,
    values: &[NativeNamedBuildInput],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut previous: Option<&str> = None;
    for input in values {
        bound.text(name, &input.name)?;
        bound.text(name, &input.value)?;
        if previous.is_some_and(|prior| prior >= input.name.as_ref()) {
            return Err(invalid_identity(format!(
                "{name} names must be sorted and unique"
            )));
        }
        previous = Some(&input.name);
    }
    Ok(())
}

fn validate_native_sources(
    values: &[NativeNativeSource],
    bound: &mut BuildBound,
) -> Result<(), NativeRuntimeError> {
    let mut previous: Option<&str> = None;
    for source in values {
        bound.text("native source label", &source.label)?;
        bound.text("native source repository", &source.canonical_repository)?;
        if previous.is_some_and(|prior| prior >= source.label.as_ref()) {
            return Err(invalid_identity(
                "native source labels must be sorted and unique",
            ));
        }
        previous = Some(&source.label);
        require_nonzero_digest("pristine native source tree", source.pristine_tree_sha256)?;
        require_nonzero_digest("effective native source tree", source.effective_tree_sha256)?;
        match (source.kind, &source.revision) {
            (NativeNativeSourceKind::Git, Some(revision)) => {
                bound.full_git_sha("native source revision", revision)?;
            }
            (NativeNativeSourceKind::Git, None) => {
                return Err(invalid_identity(
                    "Git native source requires its exact fetched revision",
                ));
            }
            (NativeNativeSourceKind::Vendored, None) => {}
            (NativeNativeSourceKind::Vendored, Some(_)) => {
                return Err(invalid_identity(
                    "vendored native source must not claim a Git revision",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeBuildIdentityInner {
    digest: [u8; 32],
    inputs: NativeEffectiveBuildInputs,
}

/// Cheaply cloned immutable digest plus the exact generated input record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuildIdentity(Arc<NativeBuildIdentityInner>);

impl NativeBuildIdentity {
    fn from_effective_inputs(
        inputs: NativeEffectiveBuildInputs,
    ) -> Result<Self, NativeRuntimeError> {
        inputs.validate()?;
        let digest = digest_effective_inputs(&inputs);
        Ok(Self(Arc::new(NativeBuildIdentityInner { digest, inputs })))
    }

    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        self.0.digest
    }

    #[must_use]
    pub fn inputs(&self) -> &NativeEffectiveBuildInputs {
        &self.0.inputs
    }

    pub fn verify_expected(&self, expected: [u8; 32]) -> Result<(), NativeRuntimeError> {
        if self.0.digest != expected {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::BuildIdentityMismatch,
                "effective native build identity does not match the expected artifact identity",
            ));
        }
        Ok(())
    }
}

struct GeneratedOuterBuildRecord {
    canonical_source_repository: &'static str,
    materialized_source_repository: &'static str,
    outer_source_revision: &'static str,
    outer_source_tree_sha256: [u8; 32],
    package_name: &'static str,
    package_version: &'static str,
    cargo_lock_sha256: [u8; 32],
    rust_toolchain: &'static str,
    rustc: &'static str,
    rust_linker: &'static str,
    target_triple: &'static str,
    target_architecture: &'static str,
    build_profile: &'static str,
    declared_default_features: &'static [&'static str],
    default_features_enabled: bool,
    cargo_features: &'static [&'static str],
    root_link_flags: &'static [&'static str],
}

mod embedded_outer_record {
    use super::GeneratedOuterBuildRecord;
    include!(concat!(
        env!("OUT_DIR"),
        "/mlxcel_outer_native_build_record.rs"
    ));
}

pub(crate) fn embedded_native_build_identity(
    consumer: NativeConsumerBuildFragment,
) -> Result<NativeBuildIdentity, NativeRuntimeError> {
    let outer = &embedded_outer_record::GENERATED_OUTER_BUILD_RECORD;
    let core = mlxcel_core::native_core_build_record();
    if outer.target_triple != core.target_triple
        || outer.target_architecture != core.target_architecture
        || outer.build_profile != core.build_profile
    {
        return Err(invalid_identity(
            "root and core generated build records disagree on target or profile",
        ));
    }

    let mut native_features = core
        .cargo_features
        .iter()
        .map(|value| Arc::<str>::from(format!("core-cargo:{value}")))
        .chain(
            core.native_features
                .iter()
                .map(|value| Arc::<str>::from(format!("core-native:{value}"))),
        )
        .collect::<Vec<_>>();
    native_features.sort();
    native_features.dedup();
    let link_mode = match core.link_mode {
        mlxcel_core::NativeCoreLinkMode::Static => NativeLinkMode::Static,
        mlxcel_core::NativeCoreLinkMode::Dynamic => NativeLinkMode::Dynamic,
        mlxcel_core::NativeCoreLinkMode::Mixed => NativeLinkMode::Mixed,
    };

    let inputs = NativeEffectiveBuildInputs {
        canonical_source_repository: Arc::from(outer.canonical_source_repository),
        materialized_source_repository: Arc::from(outer.materialized_source_repository),
        outer_source_revision: Arc::from(outer.outer_source_revision),
        outer_source_tree_sha256: outer.outer_source_tree_sha256,
        self_package: NativeSelfBuildPackage {
            name: Arc::from(outer.package_name),
            version: Arc::from(outer.package_version),
        },
        native_sources: vec![NativeNativeSource {
            kind: NativeNativeSourceKind::Git,
            label: Arc::from("mlx"),
            canonical_repository: Arc::from(core.fetched_mlx_repository),
            revision: Some(Arc::from(core.fetched_mlx_revision)),
            pristine_tree_sha256: core.fetched_mlx_pristine_tree_sha256,
            effective_tree_sha256: core.fetched_mlx_effective_tree_sha256,
        }],
        ordered_patches: map_core_sources(core.ordered_patches),
        cxx_bridge_sources: map_core_sources(core.cxx_bridge_sources),
        overlay_sources: map_core_sources(core.overlay_sources),
        turbo_sources: map_core_sources(core.turbo_sources),
        self_cargo_lock_sha256: outer.cargo_lock_sha256,
        rust_toolchain: Arc::from(outer.rust_toolchain),
        rustc: Arc::from(outer.rustc),
        c_compiler: Arc::from(core.c_compiler),
        cxx_bridge_compiler: Arc::from(core.cxx_bridge_compiler),
        cxx_compiler: Arc::from(core.mlx_cxx_compiler),
        cmake: Arc::from(core.cmake),
        linker: Arc::from(core.linker),
        rust_linker: Arc::from(outer.rust_linker),
        target_triple: Arc::from(outer.target_triple),
        target_architecture: Arc::from(outer.target_architecture),
        build_profile: Arc::from(outer.build_profile),
        declared_default_features: outer
            .declared_default_features
            .iter()
            .map(|value| Arc::from(*value))
            .collect(),
        default_features_enabled: outer.default_features_enabled,
        cargo_features: outer
            .cargo_features
            .iter()
            .map(|value| Arc::from(*value))
            .collect(),
        native_features,
        cmake_definitions: map_core_named(core.cmake_definitions),
        build_environment: map_core_named(core.build_environment),
        compile_flags: core
            .compile_flags
            .iter()
            .map(|value| Arc::from(*value))
            .collect(),
        link_flags: core
            .link_flags
            .iter()
            .map(|value| Arc::from(*value))
            .collect(),
        sdk_inputs: map_core_named(core.sdk_inputs),
        header_inputs: map_core_sources(core.header_inputs),
        library_inputs: map_core_sources(core.library_inputs),
        link_mode,
        link_inputs: map_core_sources(core.link_inputs),
        bundled_artifacts: map_core_sources(core.bundled_artifacts),
        root_link_flags: outer
            .root_link_flags
            .iter()
            .map(|value| Arc::from(*value))
            .collect(),
        consumer,
        adapter_schema_revision: Arc::from("mlxcel.native-runtime.v1"),
    };
    NativeBuildIdentity::from_effective_inputs(inputs)
}

fn map_core_sources(values: &[mlxcel_core::NativeCoreBuildSource]) -> Vec<NativeBuildSource> {
    values
        .iter()
        .map(|value| NativeBuildSource {
            label: Arc::from(value.label),
            sha256: value.sha256,
        })
        .collect()
}

fn map_core_named(values: &[mlxcel_core::NativeCoreNamedBuildInput]) -> Vec<NativeNamedBuildInput> {
    values
        .iter()
        .map(|value| NativeNamedBuildInput {
            name: Arc::from(value.name),
            value: Arc::from(value.value),
        })
        .collect()
}

fn digest_effective_inputs(inputs: &NativeEffectiveBuildInputs) -> [u8; 32] {
    let mut encoder = CanonicalEncoder::new();
    encoder.text(BUILD_IDENTITY_SCHEMA);
    encoder.text(&inputs.canonical_source_repository);
    encoder.text(&inputs.materialized_source_repository);
    encoder.text(&inputs.outer_source_revision);
    encoder.bytes(&inputs.outer_source_tree_sha256);
    encoder.text(&inputs.self_package.name);
    encoder.text(&inputs.self_package.version);
    encoder.native_sources(&inputs.native_sources);
    encoder.sources(&inputs.ordered_patches);
    encoder.sources(&inputs.cxx_bridge_sources);
    encoder.sources(&inputs.overlay_sources);
    encoder.sources(&inputs.turbo_sources);
    encoder.bytes(&inputs.self_cargo_lock_sha256);
    encoder.text(&inputs.rust_toolchain);
    encoder.text(&inputs.rustc);
    encoder.text(&inputs.c_compiler);
    encoder.text(&inputs.cxx_bridge_compiler);
    encoder.text(&inputs.cxx_compiler);
    encoder.text(&inputs.cmake);
    encoder.text(&inputs.linker);
    encoder.text(&inputs.rust_linker);
    encoder.text(&inputs.target_triple);
    encoder.text(&inputs.target_architecture);
    encoder.text(&inputs.build_profile);
    encoder.texts(&inputs.declared_default_features);
    encoder.boolean(inputs.default_features_enabled);
    encoder.texts(&inputs.cargo_features);
    encoder.texts(&inputs.native_features);
    encoder.named(&inputs.cmake_definitions);
    encoder.named(&inputs.build_environment);
    encoder.texts(&inputs.compile_flags);
    encoder.texts(&inputs.link_flags);
    encoder.named(&inputs.sdk_inputs);
    encoder.sources(&inputs.header_inputs);
    encoder.sources(&inputs.library_inputs);
    encoder.byte(match inputs.link_mode {
        NativeLinkMode::Static => 0,
        NativeLinkMode::Dynamic => 1,
        NativeLinkMode::Mixed => 2,
    });
    encoder.sources(&inputs.link_inputs);
    encoder.sources(&inputs.bundled_artifacts);
    encoder.texts(&inputs.root_link_flags);
    encoder.consumer(&inputs.consumer);
    encoder.text(&inputs.adapter_schema_revision);
    encoder.finish()
}

fn digest_post_link_attestation(post_link: &NativePostLinkAttestation) -> [u8; 32] {
    let mut encoder = CanonicalEncoder::new();
    encoder.text("mlxcel.native-post-link.v1");
    encoder.bytes(&post_link.pre_link_identity());
    encoder.text(post_link.artifact_kind());
    encoder.bytes(&post_link.artifact_sha256());
    encoder.bytes(&post_link.final_link_trace_sha256());
    match post_link.link_map_sha256() {
        Some(digest) => {
            encoder.byte(1);
            encoder.bytes(&digest);
        }
        None => encoder.byte(0),
    }
    encoder.texts(post_link.observed_link_arguments());
    encoder.finish()
}

struct CanonicalEncoder(Sha256);

impl CanonicalEncoder {
    fn new() -> Self {
        Self(Sha256::new())
    }

    fn length(&mut self, value: usize) {
        self.0.update((value as u64).to_le_bytes());
    }

    fn byte(&mut self, value: u8) {
        self.0.update([value]);
    }

    fn bytes(&mut self, value: &[u8]) {
        self.length(value.len());
        self.0.update(value);
    }

    fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn optional_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.text(value);
            }
            None => self.byte(0),
        }
    }

    fn boolean(&mut self, value: bool) {
        self.byte(u8::from(value));
    }

    fn texts(&mut self, values: &[Arc<str>]) {
        self.length(values.len());
        for value in values {
            self.text(value);
        }
    }

    fn sources(&mut self, values: &[NativeBuildSource]) {
        self.length(values.len());
        for source in values {
            self.text(&source.label);
            self.bytes(&source.sha256);
        }
    }

    fn named(&mut self, values: &[NativeNamedBuildInput]) {
        self.length(values.len());
        for input in values {
            self.text(&input.name);
            self.text(&input.value);
        }
    }

    fn native_sources(&mut self, values: &[NativeNativeSource]) {
        self.length(values.len());
        for source in values {
            self.byte(match source.kind {
                NativeNativeSourceKind::Git => 0,
                NativeNativeSourceKind::Vendored => 1,
            });
            self.text(&source.label);
            self.text(&source.canonical_repository);
            self.optional_text(source.revision.as_deref());
            self.bytes(&source.pristine_tree_sha256);
            self.bytes(&source.effective_tree_sha256);
        }
    }

    fn consumer(&mut self, fragment: &NativeConsumerBuildFragment) {
        self.text(fragment.manifest_repository());
        self.text(fragment.manifest_revision());
        self.boolean(fragment.manifest_default_features());
        self.texts(fragment.manifest_explicit_features());
        self.text(fragment.resolved_package().name());
        self.text(fragment.resolved_package().version());
        self.text(fragment.resolved_package().source());
        self.optional_text(fragment.resolved_package().checksum());
        self.bytes(&fragment.cargo_lock_sha256());
        self.texts(fragment.effective_cargo_features());
        self.text(fragment.rust_toolchain());
        self.text(fragment.rustc());
        self.text(fragment.final_linker());
        self.text(fragment.target_triple());
        self.text(fragment.target_architecture());
        self.text(fragment.build_profile());
        self.texts(fragment.emitted_link_directives());
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeSchedulerTopology {
    SingleSequence,
    ContinuousBatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeSequenceStateTopology {
    Independent,
    Batched,
    ModelOwned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeCacheTopology {
    Dense,
    Paged,
    ModelOwned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeDeviceClass {
    Metal,
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeOffloadPolicy {
    DeviceResident,
    HostVisible,
    BoundedHybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeModelOwnedStateMode {
    Disabled,
    Optional,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAdapterBinding {
    pub schema: Arc<str>,
    pub identity: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeDrafterTopology {
    Disabled,
    Bound {
        kind: Arc<str>,
        package_identity: [u8; 32],
        max_draft_tokens: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativePreparedMultimodalClass {
    Disabled,
    OwnedPrefill {
        modalities: NativeModalitySet,
        max_media_items: u32,
        max_payload_bytes: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeContextPolicy {
    pub max_context_tokens: u32,
    pub max_position: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeBatchPolicy {
    pub max_active_sequences: u32,
    pub max_queued_operations: u32,
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeCachePolicy {
    pub topology: NativeCacheTopology,
    pub max_resident_bytes: u64,
    pub prefix_reuse: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDevicePolicy {
    pub class: NativeDeviceClass,
    pub offload: NativeOffloadPolicy,
    pub max_device_bytes: u64,
}

/// Caller-selected bounded policy. Authority facts derived from package
/// inspection are deliberately absent and are bound by
/// [`NativeWorkerConfig::try_from_report`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWorkerPolicy {
    pub operation_driver: NativeOperationDriver,
    pub scheduler_topology: NativeSchedulerTopology,
    pub sequence_state_topology: NativeSequenceStateTopology,
    pub context: NativeContextPolicy,
    pub batch: NativeBatchPolicy,
    pub cache: NativeCachePolicy,
    pub device: NativeDevicePolicy,
    pub model_owned_state: NativeModelOwnedStateMode,
    pub adapter: Option<NativeAdapterBinding>,
    pub drafter: NativeDrafterTopology,
    pub prepared_multimodal: NativePreparedMultimodalClass,
    pub max_prompt_tokens: u32,
    pub max_generation_tokens: u32,
    pub max_embedding_inputs: u32,
    pub max_embedding_dimensions: u32,
    pub max_accepted_chunk_units: u32,
    pub max_constraint_bytes: u64,
}

/// Complete immutable native projection of one canonical resident-worker
/// profile, bound to one exact package inspection report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWorkerConfig {
    package_identity: [u8; 32],
    report_identity: [u8; 32],
    model_context_window: Option<u32>,
    embedding_input_ceiling: Option<u32>,
    decoder_guarantee: IncrementalDecodeGuarantee,
    policy: NativeWorkerPolicy,
}

impl NativeWorkerConfig {
    pub fn try_from_report(
        report: &super::model_report::NativeModelReport,
        policy: NativeWorkerPolicy,
    ) -> Result<Self, NativeRuntimeError> {
        let expected_driver = match report.operation_class() {
            super::operation_driver::MlxNativeOperationClass::Autoregressive
                if report.model_capabilities().generation =>
            {
                NativeOperationDriver::AutoregressiveGenerate
            }
            super::operation_driver::MlxNativeOperationClass::EncoderDecoder
                if report.model_capabilities().generation =>
            {
                NativeOperationDriver::EncoderDecoderGenerate
            }
            super::operation_driver::MlxNativeOperationClass::Embedding
                if report.model_capabilities().embedding =>
            {
                NativeOperationDriver::Embedding
            }
            super::operation_driver::MlxNativeOperationClass::SpeechToText => {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedOperation,
                    "speech-to-text has no managed request/admission/sink lifecycle",
                ));
            }
            _ => {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedOperation,
                    "the inspected operation class has no managed native lifecycle",
                ));
            }
        };
        if policy.operation_driver != expected_driver {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "worker operation driver disagrees with immutable package inspection",
            ));
        }
        let decoder_guarantee = report.tokenizer().incremental_decode();
        if matches!(
            expected_driver,
            NativeOperationDriver::AutoregressiveGenerate
                | NativeOperationDriver::EncoderDecoderGenerate
        ) && decoder_guarantee == IncrementalDecodeGuarantee::TerminalOnly
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "terminal-only tokenizers have no lossless per-unit streaming representation",
            ));
        }
        let managed = report.managed_capabilities();
        if matches!(
            expected_driver,
            NativeOperationDriver::AutoregressiveGenerate
                | NativeOperationDriver::EncoderDecoderGenerate
        ) && !managed.generation
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "the immutable report has no qualified managed generation capability",
            ));
        }
        if expected_driver == NativeOperationDriver::Embedding && !managed.embedding {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "the immutable report has no qualified managed embedding capability",
            ));
        }
        if expected_driver != NativeOperationDriver::Embedding
            && report.context_window().is_none_or(|window| window == 0)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "managed worker requires an inspected non-zero model context window",
            ));
        }
        if expected_driver == NativeOperationDriver::Embedding
            && report
                .embedding_input_ceiling()
                .is_none_or(|ceiling| ceiling == 0)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "managed embedding worker requires an inspected driver input ceiling",
            ));
        }
        if let NativePreparedMultimodalClass::OwnedPrefill { modalities, .. } =
            &policy.prepared_multimodal
            && (!report.model_capabilities().prepared_media
                || !report.managed_capabilities().prepared_media
                || *modalities != report.modalities())
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "prepared multimodal policy disagrees with immutable package inspection",
            ));
        }
        let config = Self {
            package_identity: report.package_identity(),
            report_identity: report.report_identity(),
            model_context_window: report.context_window(),
            embedding_input_ceiling: report.embedding_input_ceiling(),
            decoder_guarantee,
            policy,
        };
        config.validate()?;
        Ok(config)
    }

    #[must_use]
    pub const fn package_identity(&self) -> [u8; 32] {
        self.package_identity
    }

    #[must_use]
    pub const fn report_identity(&self) -> [u8; 32] {
        self.report_identity
    }

    #[must_use]
    pub const fn model_context_window(&self) -> Option<u32> {
        self.model_context_window
    }

    #[must_use]
    pub const fn embedding_input_ceiling(&self) -> Option<u32> {
        self.embedding_input_ceiling
    }

    #[must_use]
    pub const fn decoder_guarantee(&self) -> IncrementalDecodeGuarantee {
        self.decoder_guarantee
    }

    #[must_use]
    pub const fn operation_driver(&self) -> NativeOperationDriver {
        self.policy.operation_driver
    }

    #[must_use]
    pub const fn policy(&self) -> &NativeWorkerPolicy {
        &self.policy
    }

    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        let policy = &self.policy;
        let nonzero = [
            policy.context.max_context_tokens,
            policy.context.max_position,
            policy.batch.max_active_sequences,
            policy.batch.max_queued_operations,
            policy.batch.max_batch_tokens,
            policy.batch.prefill_chunk_tokens,
            policy.max_prompt_tokens,
            policy.max_generation_tokens,
            policy.max_embedding_inputs,
            policy.max_embedding_dimensions,
            policy.max_accepted_chunk_units,
        ];
        if nonzero.contains(&0)
            || policy.cache.max_resident_bytes == 0
            || policy.device.max_device_bytes == 0
            || policy.max_constraint_bytes == 0
            || self.package_identity == [0; 32]
            || self.report_identity == [0; 32]
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "native worker bounds must all be non-zero",
            ));
        }
        if policy.max_prompt_tokens as usize > MAX_NATIVE_PROMPT_TOKENS
            || policy.max_embedding_inputs as usize > MAX_NATIVE_EMBED_INPUTS
            || policy.max_embedding_dimensions as usize > MAX_NATIVE_EMBEDDING_DIMENSION
            || policy.max_accepted_chunk_units as usize > MAX_NATIVE_ACCEPTED_UNITS
            || policy.max_constraint_bytes as usize > MAX_NATIVE_CONSTRAINT_BYTES
            || policy.batch.prefill_chunk_tokens > policy.batch.max_batch_tokens
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "native worker policy exceeds a DTO ceiling or is internally inconsistent",
            ));
        }
        let generation_ceiling = policy
            .max_prompt_tokens
            .checked_add(policy.max_generation_tokens)
            .ok_or_else(|| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidRequest,
                    "worker prompt plus generation ceiling overflowed",
                )
            })?;
        if policy.context.max_context_tokens > policy.context.max_position {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "worker context policy exceeds its position span",
            ));
        }
        if matches!(
            policy.operation_driver,
            NativeOperationDriver::AutoregressiveGenerate
                | NativeOperationDriver::EncoderDecoderGenerate
        ) {
            let Some(model_context_window) = self.model_context_window else {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedCapability,
                    "worker has no inspected model context window",
                ));
            };
            if model_context_window == 0 {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedCapability,
                    "worker has a zero inspected model context window",
                ));
            }
            if policy.context.max_context_tokens > model_context_window
                || policy.context.max_position > model_context_window
            {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidRequest,
                    "worker context policy exceeds the inspected model context",
                ));
            }
            if self.decoder_guarantee == IncrementalDecodeGuarantee::TerminalOnly {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedGuarantee,
                    "terminal-only tokenizer cannot start a streaming generation worker",
                ));
            }
            if generation_ceiling > policy.context.max_context_tokens
                || generation_ceiling > policy.context.max_position
                || generation_ceiling > model_context_window
            {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidRequest,
                    "worker prompt plus generation ceiling exceeds its profile or model context",
                ));
            }
        } else if policy.operation_driver == NativeOperationDriver::Embedding {
            let Some(embedding_input_ceiling) = self.embedding_input_ceiling else {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedCapability,
                    "worker has no inspected embedding input ceiling",
                ));
            };
            if embedding_input_ceiling == 0 {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedCapability,
                    "worker has a zero inspected embedding input ceiling",
                ));
            }
            if policy.context.max_context_tokens > embedding_input_ceiling
                || policy.context.max_position > embedding_input_ceiling
                || policy.max_prompt_tokens > embedding_input_ceiling
            {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidRequest,
                    "worker embedding policy exceeds the inspected driver input ceiling",
                ));
            }
        }
        if let IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } =
            self.decoder_guarantee
            && (tokens == 0 || tokens > MAX_PROVEN_LOOKBEHIND_TOKENS)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "worker decoder lookbehind exceeds the proven native ceiling",
            ));
        }
        if policy.adapter.is_some() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "adapter binding requires a separately owned immutable resource authority",
            ));
        }
        if matches!(policy.drafter, NativeDrafterTopology::Bound { .. }) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "drafter binding requires a separately owned immutable package authority",
            ));
        }
        if let NativePreparedMultimodalClass::OwnedPrefill {
            modalities,
            max_media_items,
            max_payload_bytes,
        } = &policy.prepared_multimodal
            && ((!modalities.image && !modalities.audio && !modalities.video)
                || *max_media_items == 0
                || *max_media_items as usize > MAX_NATIVE_PREPARED_MEDIA_ITEMS
                || *max_payload_bytes == 0
                || *max_payload_bytes as usize > MAX_NATIVE_PREPARED_BYTES)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "worker prepared-multimodal class violates its bound",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<[u8; 32], NativeRuntimeError> {
        self.validate()?;
        let mut encoder = CanonicalEncoder::new();
        encoder.text(WORKER_PROFILE_SCHEMA);
        encoder.bytes(&self.package_identity);
        encoder.bytes(&self.report_identity);
        match self.model_context_window {
            Some(window) => {
                encoder.byte(1);
                encoder.bytes(&window.to_le_bytes());
            }
            None => encoder.byte(0),
        }
        match self.embedding_input_ceiling {
            Some(ceiling) => {
                encoder.byte(1);
                encoder.bytes(&ceiling.to_le_bytes());
            }
            None => encoder.byte(0),
        }
        encoder.byte(self.policy.operation_driver.operation_class() as u8);
        encoder.byte(self.policy.operation_driver as u8);
        encoder.byte(self.policy.scheduler_topology as u8);
        encoder.byte(self.policy.sequence_state_topology as u8);
        encoder.bytes(&self.policy.context.max_context_tokens.to_le_bytes());
        encoder.bytes(&self.policy.context.max_position.to_le_bytes());
        encoder.bytes(&self.policy.batch.max_active_sequences.to_le_bytes());
        encoder.bytes(&self.policy.batch.max_queued_operations.to_le_bytes());
        encoder.bytes(&self.policy.batch.max_batch_tokens.to_le_bytes());
        encoder.bytes(&self.policy.batch.prefill_chunk_tokens.to_le_bytes());
        encoder.byte(self.policy.cache.topology as u8);
        encoder.bytes(&self.policy.cache.max_resident_bytes.to_le_bytes());
        encoder.boolean(self.policy.cache.prefix_reuse);
        encoder.byte(self.policy.device.class as u8);
        encoder.byte(self.policy.device.offload as u8);
        encoder.bytes(&self.policy.device.max_device_bytes.to_le_bytes());
        match self.decoder_guarantee {
            IncrementalDecodeGuarantee::RawTokenBytes => encoder.byte(0),
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } => {
                encoder.byte(1);
                encoder.bytes(&(tokens as u64).to_le_bytes());
            }
            IncrementalDecodeGuarantee::TerminalOnly => encoder.byte(2),
        }
        encoder.byte(self.policy.model_owned_state as u8);
        match &self.policy.adapter {
            Some(adapter) => {
                encoder.byte(1);
                encoder.text(&adapter.schema);
                encoder.bytes(&adapter.identity);
            }
            None => encoder.byte(0),
        }
        match &self.policy.drafter {
            NativeDrafterTopology::Disabled => encoder.byte(0),
            NativeDrafterTopology::Bound {
                kind,
                package_identity,
                max_draft_tokens,
            } => {
                encoder.byte(1);
                encoder.text(kind);
                encoder.bytes(package_identity);
                encoder.bytes(&max_draft_tokens.to_le_bytes());
            }
        }
        match &self.policy.prepared_multimodal {
            NativePreparedMultimodalClass::Disabled => encoder.byte(0),
            NativePreparedMultimodalClass::OwnedPrefill {
                modalities,
                max_media_items,
                max_payload_bytes,
            } => {
                encoder.byte(1);
                encoder.byte(u8::from(modalities.text));
                encoder.byte(u8::from(modalities.image));
                encoder.byte(u8::from(modalities.audio));
                encoder.byte(u8::from(modalities.video));
                encoder.bytes(&max_media_items.to_le_bytes());
                encoder.bytes(&max_payload_bytes.to_le_bytes());
            }
        }
        for value in [
            self.policy.max_prompt_tokens,
            self.policy.max_generation_tokens,
            self.policy.max_embedding_inputs,
            self.policy.max_embedding_dimensions,
            self.policy.max_accepted_chunk_units,
        ] {
            encoder.bytes(&value.to_le_bytes());
        }
        encoder.bytes(&self.policy.max_constraint_bytes.to_le_bytes());
        Ok(encoder.finish())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeWorkerHealth {
    Ready,
    Draining,
    ReloadRequired,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeEngineDescriptor {
    pub worker_id: u64,
    pub worker_epoch: u64,
    pub build_identity: [u8; 32],
    pub package_identity: [u8; 32],
    pub profile_identity: [u8; 32],
    pub health: NativeWorkerHealth,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(label: &str, byte: u8) -> NativeBuildSource {
        NativeBuildSource {
            label: Arc::from(label),
            sha256: [byte; 32],
        }
    }

    fn named(name: &str, value: &str) -> NativeNamedBuildInput {
        NativeNamedBuildInput {
            name: Arc::from(name),
            value: Arc::from(value),
        }
    }

    fn generation_policy() -> NativeWorkerPolicy {
        NativeWorkerPolicy {
            operation_driver: NativeOperationDriver::AutoregressiveGenerate,
            scheduler_topology: NativeSchedulerTopology::SingleSequence,
            sequence_state_topology: NativeSequenceStateTopology::Independent,
            context: NativeContextPolicy {
                max_context_tokens: 8,
                max_position: 8,
            },
            batch: NativeBatchPolicy {
                max_active_sequences: 1,
                max_queued_operations: 1,
                max_batch_tokens: 8,
                prefill_chunk_tokens: 1,
            },
            cache: NativeCachePolicy {
                topology: NativeCacheTopology::Dense,
                max_resident_bytes: 1,
                prefix_reuse: false,
            },
            device: NativeDevicePolicy {
                class: NativeDeviceClass::Cpu,
                offload: NativeOffloadPolicy::HostVisible,
                max_device_bytes: 1,
            },
            model_owned_state: NativeModelOwnedStateMode::Disabled,
            adapter: None,
            drafter: NativeDrafterTopology::Disabled,
            prepared_multimodal: NativePreparedMultimodalClass::Disabled,
            max_prompt_tokens: 4,
            max_generation_tokens: 4,
            max_embedding_inputs: 1,
            max_embedding_dimensions: 1,
            max_accepted_chunk_units: MAX_NATIVE_ACCEPTED_UNITS as u32,
            max_constraint_bytes: 1,
        }
    }

    fn generation_config(guarantee: IncrementalDecodeGuarantee) -> NativeWorkerConfig {
        NativeWorkerConfig {
            package_identity: [1; 32],
            report_identity: [2; 32],
            model_context_window: Some(8),
            embedding_input_ceiling: Some(8),
            decoder_guarantee: guarantee,
            policy: generation_policy(),
        }
    }

    #[test]
    fn streaming_profile_rejects_terminal_only_and_binds_output_ceilings() {
        let terminal = generation_config(IncrementalDecodeGuarantee::TerminalOnly);
        assert_eq!(
            terminal.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::UnsupportedGuarantee
        );

        let bounded = generation_config(IncrementalDecodeGuarantee::ProvenBoundedLookbehind {
            tokens: MAX_PROVEN_LOOKBEHIND_TOKENS,
        });
        assert!(bounded.validate().is_ok());
        assert_eq!(MAX_PROVEN_LOOKBEHIND_TOKENS, MAX_NATIVE_ACCEPTED_UNITS);
        assert_eq!(MAX_STABLE_DELTA_BYTES, MAX_NATIVE_STABLE_BYTES_PER_UNIT);
        assert_eq!(MAX_PENDING_DECODE_BYTES, MAX_NATIVE_STABLE_BYTES_PER_UNIT);

        let too_wide = generation_config(IncrementalDecodeGuarantee::ProvenBoundedLookbehind {
            tokens: MAX_PROVEN_LOOKBEHIND_TOKENS + 1,
        });
        assert!(too_wide.validate().is_err());
    }

    #[test]
    fn generation_ceiling_is_bound_to_the_exact_report_context_window() {
        let exact = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        assert!(exact.validate().is_ok());

        let mut over = exact.clone();
        over.policy.max_generation_tokens += 1;
        assert!(over.validate().is_err());

        let mut report_mismatch = exact;
        report_mismatch.model_context_window = Some(7);
        assert!(report_mismatch.validate().is_err());

        let mut short_position = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        short_position.policy.context.max_position = 7;
        assert!(short_position.validate().is_err());

        let mut inverted_context = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        inverted_context.policy.context.max_context_tokens = 8;
        inverted_context.policy.context.max_position = 7;
        inverted_context.policy.max_prompt_tokens = 3;
        inverted_context.policy.max_generation_tokens = 4;
        assert!(inverted_context.validate().is_err());
    }

    #[test]
    fn prefill_chunk_not_full_prompt_is_bound_to_batch_tokens() {
        let mut exact = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        exact.policy.batch.prefill_chunk_tokens = exact.policy.batch.max_batch_tokens;
        assert!(exact.validate().is_ok());

        let mut over = exact.clone();
        over.policy.batch.prefill_chunk_tokens += 1;
        assert!(over.validate().is_err());

        let mut chunked = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        chunked.policy.max_prompt_tokens = 5;
        chunked.policy.max_generation_tokens = 3;
        chunked.policy.batch.max_batch_tokens = 4;
        chunked.policy.batch.prefill_chunk_tokens = 4;
        assert!(chunked.validate().is_ok());
    }

    #[test]
    fn owned_prefill_media_count_is_bound_to_the_shared_dto_ceiling() {
        let mut exact = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        exact.policy.prepared_multimodal = NativePreparedMultimodalClass::OwnedPrefill {
            modalities: NativeModalitySet {
                text: true,
                image: true,
                audio: false,
                video: false,
            },
            max_media_items: MAX_NATIVE_PREPARED_MEDIA_ITEMS as u32,
            max_payload_bytes: 1,
        };
        assert!(exact.validate().is_ok());

        let mut over = exact;
        let NativePreparedMultimodalClass::OwnedPrefill {
            max_media_items, ..
        } = &mut over.policy.prepared_multimodal
        else {
            unreachable!()
        };
        *max_media_items += 1;
        assert!(over.validate().is_err());
    }

    #[test]
    fn unowned_adapter_and_drafter_identities_are_rejected_pre_start() {
        let mut adapter = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        adapter.policy.adapter = Some(NativeAdapterBinding {
            schema: Arc::from("adapter"),
            identity: [3; 32],
        });
        assert_eq!(
            adapter.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );

        let mut drafter = generation_config(IncrementalDecodeGuarantee::RawTokenBytes);
        drafter.policy.drafter = NativeDrafterTopology::Bound {
            kind: Arc::from("mtp"),
            package_identity: [4; 32],
            max_draft_tokens: 1,
        };
        assert_eq!(
            drafter.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );
    }

    #[test]
    fn embedding_policy_is_bound_to_a_nonzero_inspected_context() {
        let mut embedding = generation_config(IncrementalDecodeGuarantee::TerminalOnly);
        embedding.policy.operation_driver = NativeOperationDriver::Embedding;
        assert!(embedding.validate().is_ok());

        embedding.embedding_input_ceiling = Some(7);
        assert!(embedding.validate().is_err());
        embedding.embedding_input_ceiling = Some(0);
        assert_eq!(
            embedding.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );
        embedding.embedding_input_ceiling = None;
        assert_eq!(
            embedding.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );
    }

    fn consumer() -> NativeConsumerBuildFragment {
        NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
            manifest_repository: CANONICAL_MLXCEL_REPOSITORY,
            manifest_revision: "b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
            manifest_default_features: false,
            manifest_explicit_features: &["accelerate", "metal"],
            resolved_package_name: "mlxcel",
            resolved_package_version: "0.6.0",
            resolved_package_source: "git+https://github.com/lablup/mlxcel.git?rev=b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df#b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
            resolved_package_checksum: None,
            cargo_lock_sha256: [13; 32],
            effective_cargo_features: &["accelerate", "metal"],
            rust_toolchain: "stable-aarch64-apple-darwin",
            rustc: "rustc 1.97.0",
            final_linker: "ld64|sha256=consumer|Apple ld",
            target_triple: "aarch64-apple-darwin",
            target_architecture: "aarch64",
            build_profile: "release",
            emitted_link_directives: &["lib=mlxcel", "framework=Metal"],
        })
        .unwrap()
    }

    fn inputs() -> NativeEffectiveBuildInputs {
        NativeEffectiveBuildInputs {
            canonical_source_repository: Arc::from("https://github.com/lablup/mlxcel.git"),
            materialized_source_repository: Arc::from("https://github.com/lablup/mlxcel.git"),
            outer_source_revision: Arc::from("b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df"),
            outer_source_tree_sha256: [1; 32],
            self_package: NativeSelfBuildPackage {
                name: Arc::from("mlxcel"),
                version: Arc::from("0.6.0"),
            },
            native_sources: vec![NativeNativeSource {
                kind: NativeNativeSourceKind::Git,
                label: Arc::from("mlx"),
                canonical_repository: Arc::from("https://github.com/ml-explore/mlx.git"),
                revision: Some(Arc::from("9a795735ad9a42664e08f42361b405ed570bcf1a")),
                pristine_tree_sha256: [2; 32],
                effective_tree_sha256: [3; 32],
            }],
            ordered_patches: vec![source("patch/002", 3), source("patch/001", 4)],
            cxx_bridge_sources: vec![source("bridge/generated.cc", 5)],
            overlay_sources: vec![source("overlay/a.cpp", 6)],
            turbo_sources: vec![source("turbo/a.cpp", 7)],
            self_cargo_lock_sha256: [8; 32],
            rust_toolchain: Arc::from("stable-aarch64-apple-darwin"),
            rustc: Arc::from("rustc 1.97.0"),
            c_compiler: Arc::from("Apple clang 18"),
            cxx_bridge_compiler: Arc::from("Apple clang++ 18 bridge"),
            cxx_compiler: Arc::from("Apple clang++ 18"),
            cmake: Arc::from("cmake 4.1"),
            linker: Arc::from("Apple ld"),
            rust_linker: Arc::from("rust final Apple ld"),
            target_triple: Arc::from("aarch64-apple-darwin"),
            target_architecture: Arc::from("aarch64"),
            build_profile: Arc::from("release"),
            declared_default_features: vec![Arc::from("surgery")],
            default_features_enabled: false,
            cargo_features: vec![Arc::from("accelerate"), Arc::from("metal")],
            native_features: vec![Arc::from("MLX_ACCELERATE"), Arc::from("MLX_METAL")],
            cmake_definitions: vec![named("MLX_BUILD_ACCELERATE", "ON")],
            build_environment: vec![named("MLXCEL_BUILD_METAL", "ON")],
            compile_flags: vec![Arc::from("-O3"), Arc::from("-std=c++20")],
            link_flags: vec![Arc::from("-framework"), Arc::from("Metal")],
            sdk_inputs: vec![named("MACOSX_DEPLOYMENT_TARGET", "15.0")],
            header_inputs: vec![source("mlx_cxx_bridge.h", 9)],
            library_inputs: vec![source("libmlx.a", 10)],
            link_mode: NativeLinkMode::Static,
            link_inputs: vec![source("libmlx.a", 11)],
            bundled_artifacts: vec![source("libmlxcel_core.a", 12)],
            root_link_flags: vec![Arc::from("lib=c++")],
            consumer: consumer(),
            adapter_schema_revision: Arc::from("mlxcel.native-runtime.v1"),
        }
    }

    fn identity(
        inputs: NativeEffectiveBuildInputs,
    ) -> Result<NativeBuildIdentity, NativeRuntimeError> {
        NativeBuildIdentity::from_effective_inputs(inputs)
    }

    #[test]
    fn every_effective_build_input_family_changes_identity() {
        let baseline = identity(inputs()).unwrap().digest();
        let mutations: Vec<Box<dyn Fn(&mut NativeEffectiveBuildInputs)>> = vec![
            Box::new(|v| v.canonical_source_repository = Arc::from("https://example.invalid")),
            Box::new(|v| {
                v.materialized_source_repository =
                    Arc::from("https://github.com/example/mlxcel.git")
            }),
            Box::new(|v| {
                v.outer_source_revision = Arc::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            }),
            Box::new(|v| v.outer_source_tree_sha256[0] ^= 1),
            Box::new(|v| v.self_package.version = Arc::from("0.6.1")),
            Box::new(|v| {
                v.native_sources[0].revision =
                    Some(Arc::from("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"))
            }),
            Box::new(|v| v.native_sources[0].pristine_tree_sha256[0] ^= 1),
            Box::new(|v| v.native_sources[0].effective_tree_sha256[0] ^= 1),
            Box::new(|v| v.ordered_patches[0].sha256[0] ^= 1),
            Box::new(|v| v.cxx_bridge_sources[0].sha256[0] ^= 1),
            Box::new(|v| v.overlay_sources[0].sha256[0] ^= 1),
            Box::new(|v| v.turbo_sources[0].sha256[0] ^= 1),
            Box::new(|v| v.self_cargo_lock_sha256[0] ^= 1),
            Box::new(|v| v.rust_toolchain = Arc::from("nightly")),
            Box::new(|v| v.rustc = Arc::from("rustc changed")),
            Box::new(|v| v.c_compiler = Arc::from("clang changed")),
            Box::new(|v| v.cxx_bridge_compiler = Arc::from("bridge clang++ changed")),
            Box::new(|v| v.cxx_compiler = Arc::from("clang++ changed")),
            Box::new(|v| v.cmake = Arc::from("cmake changed")),
            Box::new(|v| v.linker = Arc::from("ld changed")),
            Box::new(|v| v.rust_linker = Arc::from("rust linker changed")),
            Box::new(|v| v.target_triple = Arc::from("x86_64-apple-darwin")),
            Box::new(|v| v.target_architecture = Arc::from("x86_64")),
            Box::new(|v| v.build_profile = Arc::from("debug")),
            Box::new(|v| {
                v.declared_default_features
                    .push(Arc::from("zz-new-default"))
            }),
            Box::new(|v| v.default_features_enabled = true),
            Box::new(|v| v.cargo_features.push(Arc::from("surgery"))),
            Box::new(|v| v.native_features.push(Arc::from("TURBO"))),
            Box::new(|v| v.cmake_definitions[0].value = Arc::from("OFF")),
            Box::new(|v| v.build_environment[0].value = Arc::from("OFF")),
            Box::new(|v| v.compile_flags[0] = Arc::from("-O2")),
            Box::new(|v| v.link_flags[1] = Arc::from("Accelerate")),
            Box::new(|v| v.sdk_inputs[0].value = Arc::from("14.0")),
            Box::new(|v| v.header_inputs[0].sha256[0] ^= 1),
            Box::new(|v| v.library_inputs[0].sha256[0] ^= 1),
            Box::new(|v| v.link_mode = NativeLinkMode::Dynamic),
            Box::new(|v| v.link_inputs[0].sha256[0] ^= 1),
            Box::new(|v| v.bundled_artifacts[0].sha256[0] ^= 1),
            Box::new(|v| v.root_link_flags[0] = Arc::from("lib=changed")),
            Box::new(|v| Arc::make_mut(&mut v.consumer.0).cargo_lock_sha256[0] ^= 1),
            Box::new(|v| Arc::make_mut(&mut v.consumer.0).final_linker = Arc::from("ld changed")),
            Box::new(|v| {
                Arc::make_mut(&mut v.consumer.0)
                    .emitted_link_directives
                    .push(Arc::from("lib=changed"))
            }),
            Box::new(|v| v.adapter_schema_revision = Arc::from("mlxcel.native-runtime.v2")),
        ];
        for mutate in mutations {
            let mut changed = inputs();
            mutate(&mut changed);
            if let Ok(changed) = identity(changed) {
                assert_ne!(baseline, changed.digest());
            }
        }
    }

    #[test]
    fn patch_application_order_is_preserved_and_duplicates_are_rejected() {
        let baseline = identity(inputs()).unwrap();
        let mut reordered = inputs();
        reordered.ordered_patches.swap(0, 1);
        let reordered = identity(reordered).unwrap();
        assert_ne!(baseline.digest(), reordered.digest());

        let mut duplicate = inputs();
        duplicate.ordered_patches[1].label = Arc::clone(&duplicate.ordered_patches[0].label);
        assert!(identity(duplicate).is_err());
    }

    #[test]
    fn native_source_kind_has_an_explicit_canonical_discriminant() {
        let git = inputs();
        let mut vendored_encoding = git.clone();
        vendored_encoding.native_sources[0].kind = NativeNativeSourceKind::Vendored;
        assert_ne!(
            digest_effective_inputs(&git),
            digest_effective_inputs(&vendored_encoding)
        );
    }

    #[test]
    fn aggregate_build_text_bound_prevents_multi_vector_retention() {
        let mut invalid = inputs();
        let value: Arc<str> = "x".repeat(MAX_BUILD_TEXT_BYTES).into();
        invalid.compile_flags = vec![value; MAX_BUILD_TOTAL_TEXT_BYTES / MAX_BUILD_TEXT_BYTES + 1];
        assert!(identity(invalid).is_err());
    }

    #[test]
    fn expected_identity_mismatch_is_typed_and_clone_is_arc_backed() {
        let identity = identity(inputs()).unwrap();
        assert_eq!(
            identity.verify_expected([0; 32]).unwrap_err().kind(),
            NativeRuntimeErrorKind::BuildIdentityMismatch
        );
        let clone = identity.clone();
        assert!(Arc::ptr_eq(&identity.0, &clone.0));
    }

    #[test]
    fn consumer_fragment_rejects_revision_default_target_and_absolute_link_mismatch() {
        let baseline = NativeConsumerBuildRecord {
            manifest_repository: CANONICAL_MLXCEL_REPOSITORY,
            manifest_revision: "b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
            manifest_default_features: false,
            manifest_explicit_features: &["accelerate", "metal"],
            resolved_package_name: "mlxcel",
            resolved_package_version: "0.6.0",
            resolved_package_source: "git+https://github.com/lablup/mlxcel.git?rev=b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df#b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
            resolved_package_checksum: None,
            cargo_lock_sha256: [20; 32],
            effective_cargo_features: &["accelerate", "metal"],
            rust_toolchain: "stable-aarch64-apple-darwin",
            rustc: "rustc 1.97.0",
            final_linker: "ld64|sha256=consumer|Apple ld",
            target_triple: "aarch64-apple-darwin",
            target_architecture: "aarch64",
            build_profile: "release",
            emitted_link_directives: &["lib=mlxcel"],
        };
        assert!(NativeConsumerBuildFragment::try_from_generated(baseline).is_ok());
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                resolved_package_source: "git+https://github.com/lablup/mlxcel.git?rev=b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ..baseline
            })
            .is_err()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                resolved_package_source: "git+https://github.com/lablup/mlxcel.git?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
                ..baseline
            })
            .is_err()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                resolved_package_source: "git+https://github.com/lablup/mlxcel.git?branch=b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df#b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df",
                ..baseline
            })
            .is_err()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                manifest_revision: "short",
                ..baseline
            })
            .is_err()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                manifest_default_features: true,
                ..baseline
            })
            .is_err()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                target_triple: "x86_64-apple-darwin",
                target_architecture: "x86_64",
                ..baseline
            })
            .is_ok()
        );
        assert!(
            NativeConsumerBuildFragment::try_from_generated(NativeConsumerBuildRecord {
                emitted_link_directives: &["C:\\ephemeral\\object.obj"],
                ..baseline
            })
            .is_err()
        );

        let mismatches: [fn(&mut NativeConsumerBuildFragmentInner); 3] = [
            |inner: &mut NativeConsumerBuildFragmentInner| {
                inner.manifest_revision = Arc::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            },
            |inner: &mut NativeConsumerBuildFragmentInner| inner.manifest_default_features = true,
            |inner: &mut NativeConsumerBuildFragmentInner| {
                inner.target_triple = Arc::from("x86_64-apple-darwin")
            },
        ];
        for mutate in mismatches {
            let mut mismatch = inputs();
            mutate(Arc::make_mut(&mut mismatch.consumer.0));
            assert!(identity(mismatch).is_err());
        }
    }

    #[test]
    fn post_link_sidecar_binds_but_does_not_change_runtime_identity() {
        let identity = identity(inputs()).unwrap();
        let attestation = NativePostLinkAttestation::try_from_generated(NativePostLinkRecord {
            pre_link_identity: identity.digest(),
            artifact_kind: "executable",
            artifact_sha256: [21; 32],
            final_link_trace_sha256: [22; 32],
            link_map_sha256: Some([23; 32]),
            observed_link_arguments: &["<object:sha256=aa>", "-framework", "Metal"],
        })
        .unwrap();
        attestation.verify_pre_link(&identity).unwrap();
        let first = attestation.digest();
        let changed = NativePostLinkAttestation::try_from_generated(NativePostLinkRecord {
            pre_link_identity: identity.digest(),
            artifact_kind: "executable",
            artifact_sha256: [24; 32],
            final_link_trace_sha256: [22; 32],
            link_map_sha256: Some([23; 32]),
            observed_link_arguments: &["<object:sha256=aa>", "-framework", "Metal"],
        })
        .unwrap();
        assert_ne!(first, changed.digest());

        let wrong = NativePostLinkAttestation::try_from_generated(NativePostLinkRecord {
            pre_link_identity: [25; 32],
            artifact_kind: "executable",
            artifact_sha256: [21; 32],
            final_link_trace_sha256: [22; 32],
            link_map_sha256: None,
            observed_link_arguments: &["<object:sha256=aa>"],
        })
        .unwrap();
        assert_eq!(
            wrong.verify_pre_link(&identity).unwrap_err().kind(),
            NativeRuntimeErrorKind::BuildIdentityMismatch
        );
    }
}
