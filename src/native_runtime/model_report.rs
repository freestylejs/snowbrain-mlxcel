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

//! Immutable package authority and neutral model inspection.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Take, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use mlxcel_core::tensor_metadata::{TensorMetadata, TensorMetadataDtype};

use super::detokenizer::IncrementalDecodeGuarantee;
use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::operation_driver::MlxNativeOperationClass;

pub const MAX_NATIVE_PACKAGE_COMPONENTS: usize = 65_536;
pub const MAX_NATIVE_PACKAGE_LABEL_BYTES: usize = 4_096;
pub const MAX_NATIVE_PACKAGE_LABEL_TOTAL_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_INSPECTION_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_NATIVE_PACKAGE_BYTES: u64 = 16 * 1024 * 1024 * 1024 * 1024;
pub const MAX_NATIVE_REPORT_ITEMS: usize = 256;
pub const MAX_NATIVE_REPORT_TEXT_BYTES: usize = 16 * 1024;
pub const MAX_NATIVE_PACKAGE_TREE_DEPTH: usize = 64;
pub const MAX_NATIVE_PACKAGE_TREE_ENTRIES: usize = 131_072;
const PACKAGE_IDENTITY_SCHEMA: &str = "mlxcel.native-package.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativePackageLimits {
    pub max_components: usize,
    pub max_total_bytes: u64,
    pub max_inspection_bytes: u64,
}

impl Default for NativePackageLimits {
    fn default() -> Self {
        Self {
            max_components: MAX_NATIVE_PACKAGE_COMPONENTS,
            max_total_bytes: MAX_NATIVE_PACKAGE_BYTES,
            max_inspection_bytes: MAX_NATIVE_INSPECTION_BYTES,
        }
    }
}

/// One exact allowlisted component in an immutable artifact source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePackageComponent {
    label: Arc<str>,
    size_bytes: u64,
    sha256: [u8; 32],
}

impl NativePackageComponent {
    pub fn new(
        label: impl AsRef<str>,
        size_bytes: u64,
        sha256: [u8; 32],
    ) -> Result<Self, NativeRuntimeError> {
        let label = label.as_ref();
        validate_component_label(label)?;
        Ok(Self {
            label: Arc::from(label),
            size_bytes,
            sha256,
        })
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }
}

struct StoredComponent {
    declaration: NativePackageComponent,
    seal: ComponentSeal,
    safetensors_header: Option<Arc<[u8]>>,
    #[cfg(windows)]
    _no_write_guard: File,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComponentSeal {
    len: u64,
    readonly: bool,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    links: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
    #[cfg(windows)]
    file_attributes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectorySeal {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    links: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
    #[cfg(windows)]
    file_attributes: u32,
}

/// Neutral source capability for one declared package snapshot.
///
/// Acquisition validates the declared size and digest while streaming each
/// artifact into a private native-owned materialization. No acquisition path
/// or root hint crosses this seam.
pub trait NativePackageArtifactSource {
    fn open_artifact(&mut self, label: &str) -> Result<Box<dyn Read + Send>, NativeRuntimeError>;
}

/// Private staging authority retained for the package lifetime. Its path is
/// never exposed publicly and is used only by root mlxcel loaders inside a
/// verify-before/verify-after closure.
struct NativeStagingAuthority {
    _lifetime: TempDir,
    canonical_root: PathBuf,
    directory_seals: BTreeMap<PathBuf, DirectorySeal>,
}

impl NativeStagingAuthority {
    fn root(&self) -> &Path {
        &self.canonical_root
    }
}

/// Opaque package object that can reopen only exact declared components.
///
/// The staging root and acquisition path are deliberately not exposed. Every
/// reopen verifies size and SHA-256 before bytes are returned to inspection.
pub struct NativeImmutablePackage {
    // Guards in `components` must close before the TempDir staging lifetime is
    // dropped on Windows, otherwise the no-delete share contract would make
    // cleanup fail. Rust drops fields in declaration order.
    components: BTreeMap<Arc<str>, StoredComponent>,
    staging: NativeStagingAuthority,
    identity: [u8; 32],
    limits: NativePackageLimits,
}

impl NativeImmutablePackage {
    pub fn acquire(
        source: &mut dyn NativePackageArtifactSource,
        declarations: Vec<NativePackageComponent>,
        limits: NativePackageLimits,
    ) -> Result<Self, NativeRuntimeError> {
        Self::acquire_in(source, declarations, limits, None)
    }

    fn acquire_in(
        source: &mut dyn NativePackageArtifactSource,
        declarations: Vec<NativePackageComponent>,
        limits: NativePackageLimits,
        temporary_parent: Option<&Path>,
    ) -> Result<Self, NativeRuntimeError> {
        if limits.max_components == 0
            || limits.max_components > MAX_NATIVE_PACKAGE_COMPONENTS
            || limits.max_total_bytes == 0
            || limits.max_total_bytes > MAX_NATIVE_PACKAGE_BYTES
            || limits.max_inspection_bytes == 0
            || limits.max_inspection_bytes > MAX_NATIVE_INSPECTION_BYTES
        {
            return Err(invalid_package("package limits exceed the native ceiling"));
        }
        if declarations.is_empty() || declarations.len() > limits.max_components {
            return Err(invalid_package(
                "package component count is zero or exceeds its declared bound",
            ));
        }

        let mut total_bytes = 0u64;
        let mut total_label_bytes = 0usize;
        let mut canonical_labels = HashSet::new();
        for declaration in &declarations {
            validate_component_label(&declaration.label)?;
            if !canonical_labels.insert(declaration.label.to_ascii_lowercase()) {
                return Err(invalid_package(
                    "package component labels have a case-insensitive alias",
                ));
            }
            total_label_bytes = total_label_bytes
                .checked_add(declaration.label.len())
                .ok_or_else(|| invalid_package("package label byte total overflowed"))?;
            if total_label_bytes > MAX_NATIVE_PACKAGE_LABEL_TOTAL_BYTES {
                return Err(invalid_package(
                    "package component labels exceed their aggregate byte bound",
                ));
            }
            total_bytes = total_bytes
                .checked_add(declaration.size_bytes)
                .ok_or_else(|| invalid_package("package byte total overflowed"))?;
            if total_bytes > limits.max_total_bytes {
                return Err(invalid_package("package exceeds its total byte bound"));
            }
        }

        let mut temporary = tempfile::Builder::new();
        temporary.prefix("mlxcel-native-package-");
        let root = temporary_parent
            .map_or_else(
                || temporary.tempdir(),
                |parent| temporary.tempdir_in(parent),
            )
            .map_err(|error| {
                invalid_package(format!(
                    "cannot create private package materialization: {error}"
                ))
            })?;
        let canonical_root = std::fs::canonicalize(root.path()).map_err(|error| {
            invalid_package(format!(
                "cannot canonicalize private package materialization: {error}"
            ))
        })?;
        let mut components = BTreeMap::new();
        let mut safetensors_header_bytes = 0u64;
        for declaration in declarations {
            let path = canonical_root.join(declaration.label.as_ref());
            let parent = path.parent().ok_or_else(|| {
                invalid_package("package component has no materialization parent")
            })?;
            std::fs::create_dir_all(parent).map_err(|error| {
                invalid_package(format!("cannot create private package directory: {error}"))
            })?;
            let reader = source.open_artifact(&declaration.label)?;
            let header_budget = if declaration.label.ends_with(".safetensors") {
                Some(
                    limits
                        .max_inspection_bytes
                        .checked_sub(safetensors_header_bytes)
                        .ok_or_else(|| invalid_package("safetensors header budget underflowed"))?,
                )
            } else {
                None
            };
            let (file, safetensors_header) =
                materialize_component(reader, &path, &declaration, header_budget)?;
            let safetensors_header = if let Some(header) = safetensors_header {
                safetensors_header_bytes = safetensors_header_bytes
                    .checked_add(8 + header.len() as u64)
                    .ok_or_else(|| invalid_package("safetensors header total overflowed"))?;
                Some(Arc::<[u8]>::from(header))
            } else {
                None
            };
            verify_metadata(
                &file.metadata().map_err(|error| {
                    invalid_package(format!(
                        "cannot stat private package component {:?}: {error}",
                        declaration.label
                    ))
                })?,
                &declaration,
            )?;
            let seal = component_seal(&file.metadata().map_err(|error| {
                invalid_package(format!(
                    "cannot seal private package component {:?}: {error}",
                    declaration.label
                ))
            })?)?;
            let key = Arc::clone(&declaration.label);
            let stored = StoredComponent {
                declaration,
                seal,
                safetensors_header,
                #[cfg(windows)]
                _no_write_guard: file,
            };
            if components.insert(key, stored).is_some() {
                return Err(invalid_package("package component labels must be unique"));
            }
        }
        verify_declared_tree(
            &canonical_root,
            components.keys().map(AsRef::as_ref),
            limits.max_components,
        )?;
        let directory_seals = capture_directory_seals(&canonical_root)?;
        let identity = package_identity(&components);
        Ok(Self {
            staging: NativeStagingAuthority {
                _lifetime: root,
                canonical_root,
                directory_seals,
            },
            components,
            identity,
            limits,
        })
    }

    #[must_use]
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }

    #[must_use]
    pub fn component(&self, label: &str) -> Option<&NativePackageComponent> {
        self.components.get(label).map(|stored| &stored.declaration)
    }

    fn open_component(&self, label: &str) -> Result<NativeComponentReader, NativeRuntimeError> {
        validate_component_label(label)?;
        let stored = self.components.get(label).ok_or_else(|| {
            invalid_package(format!("package component {label:?} is not declared"))
        })?;
        let path = checked_component_path(self.staging.root(), &stored.declaration.label)?;
        let file = open_sealed_component(&path).map_err(|error| {
            invalid_package(format!(
                "cannot reopen declared package component {label:?}: {error}"
            ))
        })?;
        verify_metadata(
            &file.metadata().map_err(|error| {
                invalid_package(format!("cannot stat package component {label:?}: {error}"))
            })?,
            &stored.declaration,
        )?;
        verify_component_seal(&file, &stored.seal, &stored.declaration)?;
        Ok(NativeComponentReader {
            inner: file.take(stored.declaration.size_bytes),
        })
    }

    pub fn read_component(&self, label: &str) -> Result<Vec<u8>, NativeRuntimeError> {
        let declaration = self.component(label).ok_or_else(|| {
            invalid_package(format!("package component {label:?} is not declared"))
        })?;
        if declaration.size_bytes > self.limits.max_inspection_bytes {
            return Err(invalid_package(format!(
                "package component {label:?} exceeds the bounded inspection-read limit"
            )));
        }
        let capacity = usize::try_from(declaration.size_bytes)
            .map_err(|_| invalid_package("component size does not fit this host"))?;
        let mut bytes = Vec::with_capacity(capacity);
        self.open_component(label)?
            .read_to_end(&mut bytes)
            .map_err(|error| {
                invalid_package(format!("cannot read component {label:?}: {error}"))
            })?;
        if bytes.len() as u64 != declaration.size_bytes
            || <[u8; 32]>::from(Sha256::digest(&bytes)) != declaration.sha256
        {
            return Err(package_changed(format!(
                "package component {label:?} changed while it was read"
            )));
        }
        Ok(bytes)
    }

    pub(crate) fn labels(&self) -> impl Iterator<Item = &str> {
        self.components.keys().map(AsRef::as_ref)
    }

    fn preflight_inspection_reads(&self) -> Result<u64, NativeRuntimeError> {
        let mut total = 0u64;
        for stored in self
            .components
            .values()
            .filter(|stored| is_inspection_read_component(stored.declaration.label()))
        {
            let size = stored.declaration.size_bytes();
            if size > self.limits.max_inspection_bytes {
                return Err(invalid_package(format!(
                    "inspection component {:?} exceeds the per-file inspection bound",
                    stored.declaration.label()
                )));
            }
            total = total
                .checked_add(size)
                .ok_or_else(|| invalid_package("inspection component byte total overflowed"))?;
            if total > self.limits.max_inspection_bytes {
                return Err(invalid_package(
                    "inspection components exceed the aggregate inspection bound",
                ));
            }
        }
        Ok(total)
    }

    /// Invoke a root mlxcel path-based loader against the exact retained
    /// staging authority. No caller obtains the root path, and both component
    /// identities and the complete declared tree are verified around the call.
    pub(crate) fn with_verified_staging_root<T>(
        &self,
        loader_labels: &[&str],
        operation: impl FnOnce(&Path) -> Result<T, NativeRuntimeError>,
    ) -> Result<T, NativeRuntimeError> {
        self.verify_all()?;
        let mut guards = self.open_loader_guards(loader_labels)?;
        let result = operation(self.staging.root());
        let guarded_verification = verify_loader_guards(&mut guards);
        let verification = self.verify_all();
        drop(guards);
        guarded_verification?;
        verification?;
        result
    }

    fn verify_all(&self) -> Result<(), NativeRuntimeError> {
        verify_declared_tree(
            self.staging.root(),
            self.components.keys().map(AsRef::as_ref),
            self.limits.max_components,
        )?;
        verify_directory_seals(self.staging.root(), &self.staging.directory_seals)?;
        for stored in self.components.values() {
            let path = checked_component_path(self.staging.root(), &stored.declaration.label)?;
            let current = open_sealed_component(&path).map_err(|error| {
                invalid_package(format!("cannot reopen current staging entry: {error}"))
            })?;
            verify_metadata(
                &current.metadata().map_err(|error| {
                    invalid_package(format!("cannot stat current staging entry: {error}"))
                })?,
                &stored.declaration,
            )?;
            verify_component_seal(&current, &stored.seal, &stored.declaration)?;
        }
        Ok(())
    }

    fn open_loader_guards<'a>(
        &'a self,
        loader_labels: &[&str],
    ) -> Result<Vec<LoaderGuard<'a>>, NativeRuntimeError> {
        let mut unique = BTreeSet::new();
        let mut guards = Vec::with_capacity(loader_labels.len());
        for label in loader_labels {
            if !unique.insert(*label) {
                continue;
            }
            let stored = self.components.get(*label).ok_or_else(|| {
                invalid_package(format!("loader component {label:?} is not declared"))
            })?;
            let path = checked_component_path(self.staging.root(), &stored.declaration.label)?;
            let mut file = open_sealed_component(&path).map_err(|error| {
                invalid_package(format!(
                    "cannot guard private package component {:?}: {error}",
                    stored.declaration.label
                ))
            })?;
            verify_handle(&mut file, &stored.declaration)?;
            verify_component_seal(&file, &stored.seal, &stored.declaration)?;
            guards.push(LoaderGuard { stored, file });
        }
        Ok(guards)
    }
}

fn verify_loader_guards(guards: &mut [LoaderGuard<'_>]) -> Result<(), NativeRuntimeError> {
    for guard in guards {
        verify_handle(&mut guard.file, &guard.stored.declaration)?;
        verify_component_seal(&guard.file, &guard.stored.seal, &guard.stored.declaration)?;
    }
    Ok(())
}

fn is_inspection_read_component(label: &str) -> bool {
    matches!(
        label,
        "config.json"
            | "model.safetensors.index.json"
            | "modules.json"
            | "1_Pooling/config.json"
            | "processor_config.json"
            | "preprocessor_config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer.model"
            | "tiktoken.model"
            | "tokenizer_config.json"
            | "sentence_bert_config.json"
            | "special_tokens_map.json"
            | "added_tokens.json"
            | "vocab.json"
            | "merges.txt"
            | "tokenizer.jsonl"
            | "chat_template.json"
            | "moondream.py"
    ) || label.ends_with(".tiktoken")
}

fn is_tokenizer_loader_component(label: &str) -> bool {
    is_inspection_read_component(label) && label != "model.safetensors.index.json"
}

/// Private reader capped at the declared component size. Bytes from this
/// stream never cross the public seam before a complete digest check.
struct NativeComponentReader {
    inner: Take<File>,
}

struct LoaderGuard<'a> {
    stored: &'a StoredComponent,
    file: File,
}

impl Read for NativeComponentReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buffer)
    }
}

fn invalid_package(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidPackage, message)
}

fn package_changed(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::PackageChanged, message)
}

pub(super) fn validate_component_label(label: &str) -> Result<(), NativeRuntimeError> {
    if label.is_empty() || label.len() > MAX_NATIVE_PACKAGE_LABEL_BYTES || label.contains('\\') {
        return Err(invalid_package(
            "package component label is empty, too large, or not slash-normalized",
        ));
    }
    let path = Path::new(label);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid_package(
            "package component label must be a normalized relative path",
        ));
    }
    for part in label.split('/') {
        if part.ends_with(['.', ' ']) || part.contains(':') || is_windows_reserved_name(part) {
            return Err(invalid_package(
                "package component label has a Windows-ambiguous path segment",
            ));
        }
    }
    Ok(())
}

fn is_windows_reserved_name(segment: &str) -> bool {
    let basename = segment.split('.').next().unwrap_or(segment);
    let upper = basename.to_ascii_uppercase();
    matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$")
        || upper
            .strip_prefix("COM")
            .or_else(|| upper.strip_prefix("LPT"))
            .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'))
}

fn stream_materialized_component(
    mut source: Box<dyn Read + Send>,
    output: &mut File,
    declaration: &NativePackageComponent,
    safetensors_header_budget: Option<u64>,
) -> Result<Option<Vec<u8>>, NativeRuntimeError> {
    if safetensors_header_budget.is_some_and(|budget| declaration.size_bytes < 8 || budget < 8) {
        return Err(invalid_package(
            "safetensors has no complete bounded header length",
        ));
    }
    let mut remaining = declaration.size_bytes;
    let mut hasher = Sha256::new();
    let mut captured_header = Vec::new();
    let mut required_header_bytes = None::<usize>;
    let mut buffer = vec![0_u8; 1024 * 1024];
    while remaining != 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| invalid_package("component copy size does not fit this host"))?;
        let read = source.read(&mut buffer[..request]).map_err(|error| {
            package_changed(format!(
                "artifact source failed while reading {:?}: {error}",
                declaration.label
            ))
        })?;
        if read == 0 {
            return Err(package_changed(format!(
                "artifact source ended before declared size for {:?}",
                declaration.label
            )));
        }
        output.write_all(&buffer[..read]).map_err(|error| {
            invalid_package(format!(
                "cannot write private package component {:?}: {error}",
                declaration.label
            ))
        })?;
        hasher.update(&buffer[..read]);

        if let Some(header_budget) = safetensors_header_budget {
            let mut offset = 0usize;
            if required_header_bytes.is_none() && captured_header.len() < 8 {
                let take = (8 - captured_header.len()).min(read);
                captured_header.extend_from_slice(&buffer[..take]);
                offset = take;
                if captured_header.len() == 8 {
                    let header_bytes = u64::from_le_bytes(
                        captured_header[..8]
                            .try_into()
                            .expect("eight-byte safetensors prefix"),
                    );
                    let total = header_bytes
                        .checked_add(8)
                        .ok_or_else(|| invalid_package("safetensors header length overflowed"))?;
                    if header_bytes == 0 || total > header_budget || total > declaration.size_bytes
                    {
                        return Err(invalid_package(
                            "safetensors header violates the aggregate inspection bound",
                        ));
                    }
                    required_header_bytes = Some(usize::try_from(total).map_err(|_| {
                        invalid_package("safetensors header length does not fit this host")
                    })?);
                }
            }
            if let Some(required) = required_header_bytes
                && captured_header.len() < required
            {
                let take = (required - captured_header.len()).min(read - offset);
                captured_header.extend_from_slice(&buffer[offset..offset + take]);
            }
        }
        remaining -= read as u64;
    }
    let mut extra = [0_u8; 1];
    if source.read(&mut extra).map_err(|error| {
        package_changed(format!(
            "artifact source failed at terminal size check for {:?}: {error}",
            declaration.label
        ))
    })? != 0
    {
        return Err(package_changed(format!(
            "artifact source exceeds declared size for {:?}",
            declaration.label
        )));
    }
    let digest: [u8; 32] = hasher.finalize().into();
    if digest != declaration.sha256 {
        return Err(package_changed(format!(
            "artifact source digest disagrees for {:?}",
            declaration.label
        )));
    }
    output.flush().map_err(|error| {
        invalid_package(format!(
            "cannot flush private package component {:?}: {error}",
            declaration.label
        ))
    })?;
    output.sync_all().map_err(|error| {
        invalid_package(format!(
            "cannot sync private package component {:?}: {error}",
            declaration.label
        ))
    })?;
    let header = if safetensors_header_budget.is_some() {
        let required = required_header_bytes
            .ok_or_else(|| invalid_package("safetensors header is truncated"))?;
        if captured_header.len() != required {
            return Err(invalid_package("safetensors header is truncated"));
        }
        Some(captured_header.split_off(8))
    } else {
        None
    };
    Ok(header)
}

#[cfg(windows)]
fn materialize_component(
    source: Box<dyn Read + Send>,
    destination: &Path,
    declaration: &NativePackageComponent,
    safetensors_header_budget: Option<u64>,
) -> Result<(File, Option<Vec<u8>>), NativeRuntimeError> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let mut output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        // Establish the no-writer/no-delete contract on the first handle.
        // Keeping this exact handle for the package lifetime closes the
        // create/write/close/reopen mapping race.
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(destination)
        .map_err(|error| {
            invalid_package(format!(
                "cannot create private package component {:?}: {error}",
                declaration.label
            ))
        })?;
    let safetensors_header =
        stream_materialized_component(source, &mut output, declaration, safetensors_header_budget)?;
    output.seek(SeekFrom::Start(0)).map_err(|error| {
        invalid_package(format!(
            "cannot rewind private package component {:?}: {error}",
            declaration.label
        ))
    })?;
    Ok((output, safetensors_header))
}

#[cfg(not(windows))]
fn materialize_component(
    source: Box<dyn Read + Send>,
    destination: &Path,
    declaration: &NativePackageComponent,
    safetensors_header_budget: Option<u64>,
) -> Result<(File, Option<Vec<u8>>), NativeRuntimeError> {
    let mut output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| {
            invalid_package(format!(
                "cannot create private package component {:?}: {error}",
                declaration.label
            ))
        })?;
    let safetensors_header =
        stream_materialized_component(source, &mut output, declaration, safetensors_header_budget)?;
    let mut permissions = output
        .metadata()
        .map_err(|error| invalid_package(format!("cannot stat private materialization: {error}")))?
        .permissions();
    permissions.set_readonly(true);
    output.set_permissions(permissions).map_err(|error| {
        invalid_package(format!(
            "cannot seal private package component {:?}: {error}",
            declaration.label
        ))
    })?;
    output.seek(SeekFrom::Start(0)).map_err(|error| {
        invalid_package(format!(
            "cannot rewind private package component {:?}: {error}",
            declaration.label
        ))
    })?;
    Ok((output, safetensors_header))
}

#[cfg(unix)]
fn open_sealed_component(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(windows)]
fn open_sealed_component(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    OpenOptions::new()
        .read(true)
        // The retained first handle already excludes all later writer/delete
        // access. A reader must share the retained handle's own write access
        // to satisfy Windows' reciprocal sharing check; this does not let a
        // new writer past the retained handle's FILE_SHARE_READ contract.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_sealed_component(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

fn component_seal(metadata: &Metadata) -> Result<ComponentSeal, NativeRuntimeError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if metadata.nlink() != 1 {
            return Err(invalid_package(
                "private package materialization unexpectedly has a hard-link alias",
            ));
        }
        return Ok(ComponentSeal {
            len: metadata.len(),
            readonly: metadata.permissions().readonly(),
            device: metadata.dev(),
            inode: metadata.ino(),
            links: metadata.nlink(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        });
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        return Ok(ComponentSeal {
            len: metadata.len(),
            readonly: metadata.permissions().readonly(),
            creation_time: metadata.creation_time(),
            last_write_time: metadata.last_write_time(),
            file_attributes: metadata.file_attributes(),
        });
    }
    #[cfg(not(any(unix, windows)))]
    {
        Ok(ComponentSeal {
            len: metadata.len(),
            readonly: metadata.permissions().readonly(),
        })
    }
}

fn capture_directory_seals(
    root: &Path,
) -> Result<BTreeMap<PathBuf, DirectorySeal>, NativeRuntimeError> {
    let mut seals = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let metadata = std::fs::symlink_metadata(&directory).map_err(|error| {
            invalid_package(format!("cannot stat private package directory: {error}"))
        })?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(invalid_package(
                "private package directory is not a direct directory",
            ));
        }
        reject_reparse_point(&metadata)?;
        let relative = directory
            .strip_prefix(root)
            .map_err(|_| invalid_package("private package directory escaped its root"))?
            .to_path_buf();
        seals.insert(relative, directory_seal(&metadata));
        for entry in std::fs::read_dir(&directory).map_err(|error| {
            invalid_package(format!(
                "cannot enumerate private package directory: {error}"
            ))
        })? {
            let entry = entry.map_err(|error| {
                invalid_package(format!("cannot enumerate private package entry: {error}"))
            })?;
            let metadata = std::fs::symlink_metadata(entry.path()).map_err(|error| {
                invalid_package(format!("cannot stat private package entry: {error}"))
            })?;
            if metadata.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    Ok(seals)
}

fn verify_directory_seals(
    root: &Path,
    expected: &BTreeMap<PathBuf, DirectorySeal>,
) -> Result<(), NativeRuntimeError> {
    let observed = capture_directory_seals(root)?;
    if observed != *expected {
        return Err(package_changed(
            "private package directory identity changed during loader access",
        ));
    }
    Ok(())
}

fn directory_seal(metadata: &Metadata) -> DirectorySeal {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        return DirectorySeal {
            device: metadata.dev(),
            inode: metadata.ino(),
            links: metadata.nlink(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        };
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        return DirectorySeal {
            creation_time: metadata.creation_time(),
            last_write_time: metadata.last_write_time(),
            file_attributes: metadata.file_attributes(),
        };
    }
    #[cfg(not(any(unix, windows)))]
    {
        DirectorySeal {}
    }
}

fn verify_component_seal(
    file: &File,
    expected: &ComponentSeal,
    declaration: &NativePackageComponent,
) -> Result<(), NativeRuntimeError> {
    let metadata = file.metadata().map_err(|error| {
        invalid_package(format!(
            "cannot inspect component seal {:?}: {error}",
            declaration.label
        ))
    })?;
    let observed = component_seal(&metadata)?;
    if observed != *expected {
        return Err(package_changed(format!(
            "package component {:?} changed its sealed physical identity or metadata",
            declaration.label
        )));
    }
    Ok(())
}

fn checked_component_path(root: &Path, label: &str) -> Result<PathBuf, NativeRuntimeError> {
    let mut path = root.to_path_buf();
    for segment in Path::new(label).components() {
        let Component::Normal(segment) = segment else {
            return Err(invalid_package("package component label is not normalized"));
        };
        path.push(segment);
        let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
            invalid_package(format!(
                "cannot inspect package component {label:?}: {error}"
            ))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(invalid_package(format!(
                "package component {label:?} traverses a symbolic link"
            )));
        }
        reject_reparse_point(&metadata)?;
    }
    let canonical = std::fs::canonicalize(&path).map_err(|error| {
        invalid_package(format!(
            "cannot canonicalize package component {label:?}: {error}"
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(invalid_package(
            "package component escapes the immutable root",
        ));
    }
    Ok(canonical)
}

fn verify_handle(
    file: &mut File,
    declaration: &NativePackageComponent,
) -> Result<(), NativeRuntimeError> {
    let metadata = file.metadata().map_err(|error| {
        invalid_package(format!(
            "cannot stat package component {:?}: {error}",
            declaration.label
        ))
    })?;
    verify_metadata(&metadata, declaration)?;
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        invalid_package(format!(
            "cannot rewind package component {:?}: {error}",
            declaration.label
        ))
    })?;
    let mut hasher = Sha256::new();
    let copied = std::io::copy(std::io::Read::by_ref(file), &mut hasher).map_err(|error| {
        invalid_package(format!(
            "cannot hash package component {:?}: {error}",
            declaration.label
        ))
    })?;
    let digest: [u8; 32] = hasher.finalize().into();
    if copied != declaration.size_bytes || digest != declaration.sha256 {
        return Err(package_changed(format!(
            "package component {:?} does not match its immutable size/digest",
            declaration.label
        )));
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        invalid_package(format!(
            "cannot rewind verified package component {:?}: {error}",
            declaration.label
        ))
    })?;
    Ok(())
}

fn verify_metadata(
    metadata: &Metadata,
    declaration: &NativePackageComponent,
) -> Result<(), NativeRuntimeError> {
    if !metadata.is_file() || metadata.len() != declaration.size_bytes {
        return Err(package_changed(format!(
            "package component {:?} is not a regular file of the declared size",
            declaration.label
        )));
    }
    reject_reparse_point(metadata)?;
    Ok(())
}

#[cfg(windows)]
fn reject_reparse_point(metadata: &Metadata) -> Result<(), NativeRuntimeError> {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid_package(
            "immutable package cannot contain a Windows reparse point",
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn reject_reparse_point(_metadata: &Metadata) -> Result<(), NativeRuntimeError> {
    Ok(())
}

fn verify_declared_tree<'a>(
    root: &Path,
    declared: impl Iterator<Item = &'a str>,
    max_components: usize,
) -> Result<(), NativeRuntimeError> {
    let declared = declared.collect::<BTreeSet<_>>();
    let mut discovered = BTreeSet::new();
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut tree_entries = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        for entry in std::fs::read_dir(&directory).map_err(|error| {
            invalid_package(format!("cannot enumerate immutable staging root: {error}"))
        })? {
            let entry = entry.map_err(|error| {
                invalid_package(format!("cannot enumerate immutable staging entry: {error}"))
            })?;
            tree_entries = tree_entries
                .checked_add(1)
                .ok_or_else(|| invalid_package("immutable staging entry count overflowed"))?;
            if tree_entries > MAX_NATIVE_PACKAGE_TREE_ENTRIES {
                return Err(invalid_package(
                    "immutable staging tree exceeds its aggregate entry bound",
                ));
            }
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                invalid_package(format!("cannot inspect immutable staging entry: {error}"))
            })?;
            if metadata.file_type().is_symlink() {
                return Err(invalid_package(
                    "immutable staging tree contains a symbolic link",
                ));
            }
            reject_reparse_point(&metadata)?;
            if metadata.is_dir() {
                let next_depth = depth + 1;
                if next_depth > MAX_NATIVE_PACKAGE_TREE_DEPTH {
                    return Err(invalid_package(
                        "immutable staging tree exceeds its depth bound",
                    ));
                }
                pending.push((path, next_depth));
                continue;
            }
            if !metadata.is_file() {
                return Err(invalid_package(
                    "immutable staging tree contains a non-file entry",
                ));
            }
            let relative = path
                .strip_prefix(root)
                .map_err(|_| invalid_package("immutable staging entry escaped its root"))?;
            let label = relative
                .to_str()
                .ok_or_else(|| invalid_package("immutable staging entry is not strict UTF-8"))?
                .replace('\\', "/");
            validate_component_label(&label)?;
            if !discovered.insert(label) || discovered.len() > max_components {
                return Err(invalid_package(
                    "immutable staging tree has duplicate or excessive files",
                ));
            }
        }
    }
    let discovered_refs = discovered
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if discovered_refs != declared {
        return Err(invalid_package(
            "immutable staging tree differs from the declared component set",
        ));
    }
    Ok(())
}

fn package_identity(components: &BTreeMap<Arc<str>, StoredComponent>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, PACKAGE_IDENTITY_SCHEMA.as_bytes());
    hasher.update((components.len() as u64).to_le_bytes());
    for stored in components.values() {
        hash_field(&mut hasher, stored.declaration.label.as_bytes());
        hasher.update(stored.declaration.size_bytes.to_le_bytes());
        hash_field(&mut hasher, &stored.declaration.sha256);
    }
    hasher.finalize().into()
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeTokenizerFormat {
    HuggingFace,
    SentencePiece,
    Tiktoken,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTokenizerReport {
    format: NativeTokenizerFormat,
    identity: [u8; 32],
    incremental_decode: IncrementalDecodeGuarantee,
}

impl NativeTokenizerReport {
    #[must_use]
    pub const fn format(&self) -> NativeTokenizerFormat {
        self.format
    }

    #[must_use]
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }

    #[must_use]
    pub const fn incremental_decode(&self) -> IncrementalDecodeGuarantee {
        self.incremental_decode
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeModalitySet {
    pub text: bool,
    pub image: bool,
    pub audio: bool,
    pub video: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeModelCapabilities {
    pub generation: bool,
    pub embedding: bool,
    pub prepared_media: bool,
}

/// Capabilities implemented by the current managed-runtime extraction. This
/// is deliberately distinct from the broader root model ecosystem capability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeManagedCapabilities {
    pub generation: bool,
    pub embedding: bool,
    pub prepared_media: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelReport {
    package_identity: [u8; 32],
    report_identity: [u8; 32],
    model_type: Arc<str>,
    operation_class: MlxNativeOperationClass,
    architecture_supported: bool,
    tokenizer: NativeTokenizerReport,
    context_window: Option<u32>,
    embedding_input_ceiling: Option<u32>,
    modalities: NativeModalitySet,
    model_capabilities: NativeModelCapabilities,
    managed_capabilities: NativeManagedCapabilities,
    speculative_modes: Arc<[Arc<str>]>,
    warnings: Arc<[Arc<str>]>,
}

/// Package-created proof that one concrete tokenizer was loaded from the
/// exact declared artifacts behind a validated worker profile. Fields and
/// construction stay private to this module so downstream engine code cannot
/// pair an arbitrary tokenizer with a stronger streaming guarantee.
#[derive(Clone)]
pub(crate) struct NativeBoundTokenizer {
    tokenizer: Arc<crate::tokenizer::MlxcelTokenizer>,
    guarantee: IncrementalDecodeGuarantee,
    package_identity: [u8; 32],
    report_identity: [u8; 32],
    tokenizer_identity: [u8; 32],
}

impl NativeBoundTokenizer {
    /// Binds a tokenizer already loaded from the retained verified source. The
    /// guarantee is produced by the native tokenizer loader, never a caller digest.
    pub(crate) fn from_loaded(
        tokenizer: crate::tokenizer::MlxcelTokenizer,
        guarantee: IncrementalDecodeGuarantee,
        binding: &super::NativeWorkerLoadBinding,
        profile: &super::NativeWorkerConfig,
    ) -> Result<Self, NativeRuntimeError> {
        if profile.package_identity() != binding.source_identity()
            || profile.report_identity() != binding.report_identity()
            || profile
                .decoder_guarantee()
                .is_some_and(|expected| expected != guarantee)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidPackage,
                "loaded tokenizer disagrees with the immutable source/profile binding",
            ));
        }
        if matches!(
            profile.operation_driver(),
            super::NativeOperationDriver::AutoregressiveGenerate
                | super::NativeOperationDriver::EncoderDecoderGenerate
        ) && guarantee == IncrementalDecodeGuarantee::TerminalOnly
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "loaded tokenizer has no supported incremental decoder",
            ));
        }
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            guarantee,
            package_identity: binding.source_identity(),
            report_identity: binding.report_identity(),
            tokenizer_identity: binding.tokenizer().digest(),
        })
    }

    pub(crate) fn shared_tokenizer(&self) -> Arc<crate::tokenizer::MlxcelTokenizer> {
        Arc::clone(&self.tokenizer)
    }

    pub(crate) const fn guarantee(&self) -> IncrementalDecodeGuarantee {
        self.guarantee
    }

    pub(crate) const fn package_identity(&self) -> [u8; 32] {
        self.package_identity
    }

    pub(crate) const fn report_identity(&self) -> [u8; 32] {
        self.report_identity
    }

    pub(crate) const fn tokenizer_identity(&self) -> [u8; 32] {
        self.tokenizer_identity
    }

    pub(crate) fn resolve_rendered_prompt(
        &self,
        prompt: &super::NativeRenderedPrompt,
    ) -> Result<Vec<i32>, NativeRuntimeError> {
        if self.package_identity != prompt.package_identity()
            || self.report_identity != prompt.report_identity()
            || self.tokenizer_identity != prompt.tokenizer_identity()
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "rendered prompt binding disagrees with the worker tokenizer",
            ));
        }
        self.tokenizer
            .encode(prompt.rendered_utf8(), prompt.add_special_tokens())
            .map_err(|error| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Tokenizer,
                    format!("rendered prompt tokenization failed: {error}"),
                )
            })?
            .into_iter()
            .map(|id| {
                i32::try_from(id).map_err(|_| {
                    NativeRuntimeError::new(
                        NativeRuntimeErrorKind::Tokenizer,
                        "rendered prompt tokenizer produced an ID outside i32",
                    )
                })
            })
            .collect()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<crate::tokenizer::MlxcelTokenizer>,
        IncrementalDecodeGuarantee,
    ) {
        (self.tokenizer, self.guarantee)
    }
}

/// Bind the existing server worker's tokenizer to the same audited decoder
/// guarantee used by immutable-package workers. This is crate-private because
/// the server model directory is not an immutable public package capability.
pub(crate) fn bind_server_tokenizer(
    root: &std::path::Path,
) -> Result<NativeBoundTokenizer, NativeRuntimeError> {
    let guarantee = server_tokenizer_guarantee(root)?;
    let tokenizer = crate::tokenizer::load_tokenizer(root).map_err(|error| {
        NativeRuntimeError::new(
            NativeRuntimeErrorKind::Tokenizer,
            format!("server tokenizer load failed: {error}"),
        )
    })?;
    Ok(NativeBoundTokenizer {
        tokenizer: Arc::new(tokenizer),
        guarantee,
        package_identity: [0; 32],
        report_identity: [0; 32],
        tokenizer_identity: [0; 32],
    })
}

pub(crate) fn server_tokenizer_guarantee(
    root: &std::path::Path,
) -> Result<IncrementalDecodeGuarantee, NativeRuntimeError> {
    let tokenizer = crate::tokenizer::load_tokenizer(root).map_err(|error| {
        NativeRuntimeError::new(
            NativeRuntimeErrorKind::Tokenizer,
            format!("server tokenizer inspection failed: {error}"),
        )
    })?;
    match tokenizer {
        crate::tokenizer::MlxcelTokenizer::Tiktoken(_) => {
            Ok(IncrementalDecodeGuarantee::RawTokenBytes)
        }
        crate::tokenizer::MlxcelTokenizer::HuggingFace(_) => {
            let bytes = std::fs::read(root.join("tokenizer.json")).map_err(|error| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidPackage,
                    format!("server tokenizer.json could not be inspected: {error}"),
                )
            })?;
            let value = serde_json::from_slice(&bytes).map_err(|error| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidPackage,
                    format!("server tokenizer.json is invalid: {error}"),
                )
            })?;
            if is_proven_replace_byte_fallback_fuse(&value) {
                Ok(IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 })
            } else {
                Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedGuarantee,
                    "server tokenizer has no audited incremental decode guarantee",
                ))
            }
        }
        crate::tokenizer::MlxcelTokenizer::SentencePiece(_) => Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedGuarantee,
            "server sentencepiece tokenizer remains terminal-only",
        )),
    }
}

impl NativeModelReport {
    #[must_use]
    pub const fn package_identity(&self) -> [u8; 32] {
        self.package_identity
    }
    #[must_use]
    pub const fn report_identity(&self) -> [u8; 32] {
        self.report_identity
    }
    #[must_use]
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
    #[must_use]
    pub const fn operation_class(&self) -> MlxNativeOperationClass {
        self.operation_class
    }
    #[must_use]
    pub const fn architecture_supported(&self) -> bool {
        self.architecture_supported
    }
    #[must_use]
    pub const fn tokenizer(&self) -> &NativeTokenizerReport {
        &self.tokenizer
    }
    #[must_use]
    pub const fn context_window(&self) -> Option<u32> {
        self.context_window
    }
    #[must_use]
    pub const fn embedding_input_ceiling(&self) -> Option<u32> {
        self.embedding_input_ceiling
    }
    #[must_use]
    pub const fn modalities(&self) -> NativeModalitySet {
        self.modalities
    }
    #[must_use]
    pub const fn model_capabilities(&self) -> NativeModelCapabilities {
        self.model_capabilities
    }
    #[must_use]
    pub const fn managed_capabilities(&self) -> NativeManagedCapabilities {
        self.managed_capabilities
    }
    #[must_use]
    pub fn speculative_modes(&self) -> &[Arc<str>] {
        &self.speculative_modes
    }
    #[must_use]
    pub fn warnings(&self) -> &[Arc<str>] {
        &self.warnings
    }
}

impl NativeImmutablePackage {
    pub(crate) fn bind_prompt_tokenizer(
        &self,
        report: &NativeModelReport,
    ) -> Result<NativeBoundTokenizer, NativeRuntimeError> {
        if self.identity() != report.package_identity()
            || report.report_identity() != model_report_identity(report)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "prompt tokenizer binding disagrees with its immutable package report",
            ));
        }
        let primary = declared_tokenizer_primary(self)?.ok_or_else(|| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "prompt package has no declared tokenizer authority",
            )
        })?;
        let tokenizer = Arc::new(load_declared_tokenizer(self, &primary)?);
        Ok(NativeBoundTokenizer {
            tokenizer,
            guarantee: report.tokenizer().incremental_decode(),
            package_identity: self.identity(),
            report_identity: report.report_identity(),
            tokenizer_identity: report.tokenizer().identity(),
        })
    }

    pub(crate) fn bind_worker_tokenizer(
        &self,
        report: &NativeModelReport,
        config: &super::profile::NativeWorkerConfig,
    ) -> Result<NativeBoundTokenizer, NativeRuntimeError> {
        config.validate()?;
        if self.identity() != report.package_identity()
            || report.report_identity() != model_report_identity(report)
            || config.package_identity() != report.package_identity()
            || config.report_identity() != report.report_identity()
            || config.decoder_guarantee() != Some(report.tokenizer().incremental_decode())
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "worker tokenizer binding disagrees with its immutable package report",
            ));
        }
        if !matches!(
            config.operation_driver(),
            super::operation_driver::NativeOperationDriver::AutoregressiveGenerate
                | super::operation_driver::NativeOperationDriver::EncoderDecoderGenerate
        ) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "only generation workers own an incremental tokenizer decoder",
            ));
        }
        let primary = declared_tokenizer_primary(self)?.ok_or_else(|| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "worker package has no declared tokenizer authority",
            )
        })?;
        let tokenizer = Arc::new(load_declared_tokenizer(self, &primary)?);
        Ok(NativeBoundTokenizer {
            tokenizer,
            guarantee: report.tokenizer().incremental_decode(),
            package_identity: self.identity(),
            report_identity: report.report_identity(),
            tokenizer_identity: report.tokenizer().identity(),
        })
    }
}

pub(crate) fn inspect_package(
    package: &NativeImmutablePackage,
) -> Result<NativeModelReport, NativeRuntimeError> {
    let mut inspection_bytes = package.preflight_inspection_reads()?;
    let config_bytes = package.read_component("config.json")?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .map_err(|error| invalid_package(format!("config.json is invalid: {error}")))?;
    let declared_model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase);
    if declared_model_type
        .as_ref()
        .is_some_and(|model_type| model_type.len() > MAX_NATIVE_PACKAGE_LABEL_BYTES)
    {
        return Err(invalid_package(
            "model_type exceeds the native report bound",
        ));
    }

    let declared_detection_files = package.labels().collect::<HashSet<_>>();
    let loader_labels = [
        "config.json",
        "model.safetensors.index.json",
        "modules.json",
        "1_Pooling/config.json",
        "processor_config.json",
    ]
    .into_iter()
    .filter(|label| declared_detection_files.contains(label))
    .collect::<Vec<_>>();
    let detected = match package.with_verified_staging_root(&loader_labels, |root| {
        crate::models::get_model_type_from_declared(root, &declared_detection_files).map_err(
            |_error| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedOperation,
                    "root mlxcel model detection rejected the declared package artifacts",
                )
            },
        )
    }) {
        Ok(model_type) => Some(model_type),
        Err(error) if error.kind() == NativeRuntimeErrorKind::UnsupportedOperation => None,
        Err(error) => return Err(error),
    };
    let operation_class = detected
        .map(classify_model_type)
        .unwrap_or(MlxNativeOperationClass::UnsupportedByNativeRuntime);
    let model_type = declared_model_type.clone().unwrap_or_else(|| {
        if detected == Some(crate::models::ModelType::Kokoro) {
            "kokoro".to_owned()
        } else {
            "unsupported".to_owned()
        }
    });
    let drafter_descriptor = declared_model_type
        .as_deref()
        .and_then(crate::models::native_inspection::native_declared_drafter_descriptor);
    let family_descriptor = drafter_descriptor.or_else(|| {
        detected.map(crate::models::native_inspection::native_family_inspection_descriptor)
    });
    let embedding_layout = matches!(
        family_descriptor,
        Some(
            crate::models::native_inspection::NativeFamilyInspectionDescriptor::Qualified(
                crate::models::native_inspection::NativeQualifiedFamily::Bert
            )
        )
    );
    let weight_metadata = inspect_weight_index(package, &mut inspection_bytes, embedding_layout)?;
    let tokenizer = inspect_tokenizer(package)?;
    let mut architecture_supported = false;
    let mut context_window =
        crate::context_window_from_config(&config).and_then(|value| u32::try_from(value).ok());
    let mut embedding_input_ceiling = None;
    let mut modalities = NativeModalitySet::default();
    let mut model_capabilities = NativeModelCapabilities::default();
    let mut warnings = Vec::new();
    let mut speculative_modes = Vec::new();

    use crate::models::native_inspection::{
        NativeFamilyInspectionDescriptor, NativeQualifiedFamily,
    };
    match family_descriptor {
        Some(NativeFamilyInspectionDescriptor::Qualified(NativeQualifiedFamily::Bert)) => {
            let canonical = canonicalize_inspected_tensor_metadata(&weight_metadata, |key| {
                crate::models::bert::canonical_native_weight_key(key, false)
                    .map(|key| key.into_owned())
            })?;
            let pooling_config = if package.component("1_Pooling/config.json").is_some() {
                let bytes = package.read_component("1_Pooling/config.json")?;
                Some(
                    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|error| {
                        invalid_package(format!("1_Pooling/config.json is invalid: {error}"))
                    })?,
                )
            } else {
                None
            };
            let quantization = crate::embeddings::loader::quantization_params(&config)
                .map(|params| mlxcel_core::tensor_metadata::TensorQuantization {
                    group_size: params.group_size,
                    bits: params.bits,
                })
                .unwrap_or(mlxcel_core::tensor_metadata::TensorQuantization {
                    group_size: 64,
                    bits: 4,
                });
            match crate::models::bert::validate_native_tensor_metadata(
                &canonical,
                &config,
                quantization,
            )
            .and_then(|facts| {
                if let Some(pooling_config) = pooling_config.as_ref() {
                    crate::embeddings::PoolingConfig::parse(pooling_config).map_err(|error| {
                        format!("BERT pooling config is incompatible with the loader: {error}")
                    })?;
                }
                Ok(facts)
            }) {
                Ok(facts) => {
                    let detected_model = detected.ok_or_else(|| {
                        invalid_package("qualified BERT descriptor has no detected model family")
                    })?;
                    let ceiling =
                        inspect_embedding_input_ceiling(package, detected_model, &config)?;
                    let tensor_context = u32::try_from(facts.context_window).map_err(|_| {
                        invalid_package("BERT context window exceeds native report bounds")
                    })?;
                    let ceiling = ceiling
                        .map(|value| value.min(tensor_context))
                        .ok_or_else(|| {
                            invalid_package(
                                "qualified BERT package has no authenticated embedding input ceiling",
                            )
                        })?;
                    architecture_supported = true;
                    context_window = Some(ceiling);
                    embedding_input_ceiling = Some(ceiling);
                    modalities.text = true;
                    model_capabilities.embedding = true;
                }
                Err(reason) => push_report_text(
                    &mut warnings,
                    &format!("BERT native descriptor rejected the package: {reason}"),
                )?,
            }
        }
        Some(NativeFamilyInspectionDescriptor::Qualified(NativeQualifiedFamily::Gemma4Unified)) => {
            let canonical = canonicalize_inspected_tensor_metadata(&weight_metadata, |key| {
                if key.contains("rotary_emb") || key == "lm_head.weight" {
                    None
                } else {
                    Some(crate::loading::normalize_gemma4_unified_key(key))
                }
            })?;
            let processor = if package.component("processor_config.json").is_some() {
                let bytes = package.read_component("processor_config.json")?;
                Some(
                    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|error| {
                        invalid_package(format!("processor_config.json is invalid: {error}"))
                    })?,
                )
            } else {
                None
            };
            match crate::loading::validate_native_gemma4_unified_tensor_metadata(
                &canonical,
                &config,
                processor.as_ref(),
            ) {
                Ok(facts) => {
                    architecture_supported = true;
                    context_window = Some(u32::try_from(facts.context_window).map_err(|_| {
                        invalid_package("Gemma 4 Unified context window exceeds native bounds")
                    })?);
                    modalities = NativeModalitySet {
                        text: true,
                        image: facts.image,
                        audio: facts.audio,
                        video: facts.video,
                    };
                    model_capabilities.generation = true;
                    model_capabilities.prepared_media =
                        modalities.image || modalities.audio || modalities.video;
                }
                Err(reason) => push_report_text(
                    &mut warnings,
                    &format!("Gemma 4 Unified native descriptor rejected the package: {reason}"),
                )?,
            }
        }
        Some(NativeFamilyInspectionDescriptor::Qualified(
            NativeQualifiedFamily::Gemma4UnifiedAssistant,
        )) => match mlxcel_core::drafter::gemma4_assistant::validate_native_tensor_metadata(
            &weight_metadata,
            &config,
        ) {
            Ok(facts) => {
                architecture_supported = true;
                context_window = Some(u32::try_from(facts.context_window).map_err(|_| {
                    invalid_package("Gemma 4 assistant context window exceeds native bounds")
                })?);
                push_report_text(&mut speculative_modes, "gemma4-assistant-drafter")?;
                push_report_text(
                    &mut warnings,
                    "the package is a qualified drafter-only resource, not a user operation",
                )?;
            }
            Err(reason) => push_report_text(
                &mut warnings,
                &format!("Gemma 4 assistant native descriptor rejected the package: {reason}"),
            )?,
        },
        Some(NativeFamilyInspectionDescriptor::ExplicitlyDeferred(reason)) => {
            push_report_text(
                &mut warnings,
                &format!("native execution is explicitly deferred: {reason}"),
            )?;
        }
        None => push_report_text(
            &mut warnings,
            "root mlxcel did not recognize an unambiguous supported model type",
        )?,
    }
    if tokenizer.format == NativeTokenizerFormat::Unsupported {
        push_report_text(
            &mut warnings,
            "the immutable package has no locally loadable supported tokenizer",
        )?;
    }
    let managed_capabilities = NativeManagedCapabilities {
        generation: operation_class == MlxNativeOperationClass::Autoregressive
            && model_capabilities.generation,
        embedding: operation_class == MlxNativeOperationClass::Embedding
            && model_capabilities.embedding,
        // Prepared media remains deferred to the later media-driver task.
        prepared_media: false,
    };
    let mut report = NativeModelReport {
        package_identity: package.identity(),
        report_identity: [0; 32],
        model_type: Arc::from(model_type),
        operation_class,
        architecture_supported,
        tokenizer,
        context_window,
        embedding_input_ceiling,
        modalities,
        model_capabilities,
        managed_capabilities,
        speculative_modes: speculative_modes.into(),
        warnings: warnings.into(),
    };
    report.report_identity = model_report_identity(&report);
    Ok(report)
}

#[cfg(test)]
pub(crate) fn qualify_generation_report_for_orchestration_tests(
    mut report: NativeModelReport,
) -> NativeModelReport {
    report.architecture_supported = true;
    report.model_capabilities.generation = true;
    report.managed_capabilities.generation = true;
    report.report_identity = model_report_identity(&report);
    report
}

fn classify_model_type(model_type: crate::models::ModelType) -> MlxNativeOperationClass {
    use crate::models::ModelType;
    match model_type {
        ModelType::DiffusionGemma | ModelType::Llada2Moe => MlxNativeOperationClass::DiffusionText,
        ModelType::Florence2VLM => MlxNativeOperationClass::EncoderDecoder,
        ModelType::Whisper => MlxNativeOperationClass::SpeechToText,
        ModelType::Kokoro | ModelType::SequenceClassifier => MlxNativeOperationClass::TaskPipeline,
        value if crate::model_metadata::is_embedding_model_type(value) => {
            MlxNativeOperationClass::Embedding
        }
        _ => MlxNativeOperationClass::Autoregressive,
    }
}

fn inspect_embedding_input_ceiling(
    package: &NativeImmutablePackage,
    model_type: crate::models::ModelType,
    config: &serde_json::Value,
) -> Result<Option<u32>, NativeRuntimeError> {
    const HARD_CAP: u64 = crate::embeddings::limits::EMBEDDING_MAX_LENGTH_CAP as u64;
    const HF_UNSET: u64 = 1_000_000;

    let mut authenticated = false;
    let mut ceiling = HARD_CAP;
    if package.component("sentence_bert_config.json").is_some() {
        let bytes = package.read_component("sentence_bert_config.json")?;
        let sidecar: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            invalid_package(format!("sentence-BERT limits are invalid: {error}"))
        })?;
        let value = sidecar
            .get("max_seq_length")
            .and_then(serde_json::Value::as_u64)
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid_package("sentence-BERT max_seq_length is absent or invalid"))?;
        authenticated = true;
        ceiling = ceiling.min(value);
    }
    if package.component("tokenizer_config.json").is_some() {
        let bytes = package.read_component("tokenizer_config.json")?;
        let sidecar: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| invalid_package(format!("tokenizer limits are invalid: {error}")))?;
        if let Some(value) = sidecar
            .get("model_max_length")
            .and_then(serde_json::Value::as_u64)
            .filter(|&value| value > 0 && value < HF_UNSET)
        {
            authenticated = true;
            ceiling = ceiling.min(value);
        }
    }
    if crate::embeddings::limits::is_absolute_position_family(model_type)
        && let Some(value) = config
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .filter(|&value| value > 0)
    {
        authenticated = true;
        ceiling = ceiling.min(value);
    }
    if authenticated {
        u32::try_from(ceiling)
            .map(Some)
            .map_err(|_| invalid_package("embedding input ceiling does not fit native policy"))
    } else {
        Ok(None)
    }
}

fn inspect_weight_index(
    package: &NativeImmutablePackage,
    inspection_bytes: &mut u64,
    embedding_layout: bool,
) -> Result<BTreeMap<String, TensorMetadata>, NativeRuntimeError> {
    let mut expected_by_shard = BTreeMap::<Arc<str>, Vec<Arc<str>>>::new();
    if package.component("model.safetensors.index.json").is_some() {
        let bytes = package.read_component("model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| invalid_package(format!("model weight index is invalid: {error}")))?;
        let map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| invalid_package("model weight index has no object weight_map"))?;
        if map.is_empty() || map.len() > MAX_NATIVE_PACKAGE_COMPONENTS {
            return Err(invalid_package(
                "model weight index has zero or excessive entries",
            ));
        }
        let mut total = 0usize;
        for (key, shard) in map {
            total = total
                .checked_add(key.len())
                .ok_or_else(|| invalid_package("model weight key bytes overflowed"))?;
            if key.is_empty()
                || key.len() > MAX_NATIVE_PACKAGE_LABEL_BYTES
                || total > MAX_NATIVE_INSPECTION_BYTES as usize
            {
                return Err(invalid_package(
                    "model weight index key text exceeds an inspection bound",
                ));
            }
            let shard = shard
                .as_str()
                .ok_or_else(|| invalid_package("model weight index shard label is not a string"))?;
            validate_component_label(shard)?;
            if Path::new(shard).components().count() != 1
                || !shard.ends_with(".safetensors")
                || package.component(shard).is_none()
            {
                return Err(invalid_package(
                    "model weight index references a non-plain or undeclared safetensors shard",
                ));
            }
            expected_by_shard
                .entry(Arc::from(shard))
                .or_default()
                .push(Arc::from(key.as_str()));
        }
    }

    let skipped_embedding_file = |label: &str| {
        embedding_layout
            && matches!(
                label,
                "consolidated.safetensors" | "adapter_model.safetensors"
            )
    };
    let labels = if expected_by_shard.is_empty() {
        package
            .labels()
            .filter(|label| {
                label.ends_with(".safetensors") && Path::new(label).components().count() == 1
            })
            .filter(|label| !skipped_embedding_file(label))
            .map(Arc::<str>::from)
            .collect::<BTreeSet<_>>()
    } else {
        expected_by_shard
            .keys()
            .filter(|label| !skipped_embedding_file(label))
            .cloned()
            .collect::<BTreeSet<_>>()
    };
    if embedding_layout && labels.is_empty() {
        return Err(invalid_package(
            "embedding loader requires a non-skipped top-level safetensors shard",
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut total_name_bytes = 0usize;
    for label in labels {
        let shard_tensors = inspect_safetensors_header(package, &label, inspection_bytes)?;
        if let Some(expected) = expected_by_shard.get(&label) {
            for key in expected {
                if !shard_tensors.contains_key(key.as_ref()) {
                    return Err(invalid_package(format!(
                        "model weight index key {key:?} is absent from declared shard {label:?}"
                    )));
                }
            }
        }
        insert_inspected_tensor_metadata(&mut tensors, &mut total_name_bytes, shard_tensors, None)?;
    }
    if embedding_layout {
        let mut nested = package
            .labels()
            .filter_map(|label| {
                let path = Path::new(label);
                let mut parts = path.components();
                let folder = match parts.next()? {
                    Component::Normal(value) => value.to_str()?,
                    _ => return None,
                };
                let file = match parts.next()? {
                    Component::Normal(value) => value.to_str()?,
                    _ => return None,
                };
                if parts.next().is_some()
                    || folder.starts_with('.')
                    || !file.ends_with(".safetensors")
                    || matches!(
                        file,
                        "consolidated.safetensors" | "adapter_model.safetensors"
                    )
                {
                    return None;
                }
                Some((Arc::<str>::from(label), Arc::<str>::from(folder)))
            })
            .collect::<Vec<_>>();
        nested.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        for (label, folder) in nested {
            let shard_tensors = inspect_safetensors_header(package, &label, inspection_bytes)?;
            insert_inspected_tensor_metadata(
                &mut tensors,
                &mut total_name_bytes,
                shard_tensors,
                Some(&folder),
            )?;
        }
    }
    package.verify_all()?;
    Ok(tensors)
}

fn insert_inspected_tensor_metadata(
    output: &mut BTreeMap<String, TensorMetadata>,
    total_name_bytes: &mut usize,
    tensors: BTreeMap<String, TensorMetadata>,
    prefix: Option<&str>,
) -> Result<(), NativeRuntimeError> {
    for (name, metadata) in tensors {
        let name = match prefix {
            Some(prefix) => format!("{prefix}.{name}"),
            None => name,
        };
        *total_name_bytes = total_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| invalid_package("safetensors tensor-name bytes overflowed"))?;
        if output.len() >= MAX_NATIVE_PACKAGE_COMPONENTS
            || *total_name_bytes > MAX_NATIVE_INSPECTION_BYTES as usize
        {
            return Err(invalid_package(
                "safetensors tensor names exceed an inspection bound",
            ));
        }
        if output.insert(name.clone(), metadata).is_some() {
            return Err(invalid_package(format!(
                "duplicate canonical safetensors tensor name {name:?}"
            )));
        }
    }
    Ok(())
}

fn canonicalize_inspected_tensor_metadata(
    source: &BTreeMap<String, TensorMetadata>,
    mut canonical_name: impl FnMut(&str) -> Option<String>,
) -> Result<BTreeMap<String, TensorMetadata>, NativeRuntimeError> {
    let mut canonical = BTreeMap::new();
    for (name, metadata) in source {
        let Some(name) = canonical_name(name) else {
            continue;
        };
        if name.is_empty() || name.len() > MAX_NATIVE_PACKAGE_LABEL_BYTES {
            return Err(invalid_package(
                "canonical tensor name exceeds the native inspection bound",
            ));
        }
        if canonical.insert(name.clone(), metadata.clone()).is_some() {
            return Err(invalid_package(format!(
                "multiple checkpoint tensors sanitize to {name:?}"
            )));
        }
    }
    Ok(canonical)
}

fn inspect_safetensors_header(
    package: &NativeImmutablePackage,
    label: &str,
    inspection_bytes: &mut u64,
) -> Result<BTreeMap<String, TensorMetadata>, NativeRuntimeError> {
    let stored = package
        .components
        .get(label)
        .ok_or_else(|| invalid_package("safetensors component is not declared"))?;
    let header = stored.safetensors_header.as_deref().ok_or_else(|| {
        invalid_package("safetensors component has no acquisition-bound header snapshot")
    })?;
    let header_bytes = header.len() as u64;
    *inspection_bytes = inspection_bytes
        .checked_add(8)
        .and_then(|value| value.checked_add(header_bytes))
        .ok_or_else(|| invalid_package("inspection byte total overflowed"))?;
    if *inspection_bytes > package.limits.max_inspection_bytes {
        return Err(invalid_package(
            "safetensors headers exceed the aggregate inspection bound",
        ));
    }
    let metadata: safetensors::tensor::Metadata = serde_json::from_slice(header)
        .map_err(|error| invalid_package(format!("safetensors header is invalid: {error}")))?;
    let data_bytes = stored
        .declaration
        .size_bytes
        .checked_sub(8 + header_bytes)
        .ok_or_else(|| invalid_package("safetensors data length underflowed"))?;
    let data_bytes = usize::try_from(data_bytes)
        .map_err(|_| invalid_package("safetensors data length does not fit this host"))?;
    if metadata.data_len() != data_bytes {
        return Err(invalid_package(
            "safetensors metadata length disagrees with the declared data",
        ));
    }
    let tensors = metadata.tensors();
    if tensors.is_empty() {
        return Err(invalid_package("safetensors has no executable tensors"));
    }
    let mut inspected = BTreeMap::new();
    let mut total_name_bytes = 0usize;
    for (name, tensor) in tensors {
        total_name_bytes = total_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| invalid_package("safetensors tensor-name bytes overflowed"))?;
        if name.is_empty()
            || name.len() > MAX_NATIVE_PACKAGE_LABEL_BYTES
            || inspected.len() >= MAX_NATIVE_PACKAGE_COMPONENTS
            || total_name_bytes > MAX_NATIVE_INSPECTION_BYTES as usize
        {
            return Err(invalid_package(
                "safetensors tensor names exceed an inspection bound",
            ));
        }
        validate_mlx_safetensors_tensor(tensor)?;
        let metadata = TensorMetadata::new(
            tensor.shape.clone(),
            tensor_metadata_dtype(tensor.dtype).ok_or_else(|| {
                invalid_package("safetensors tensor dtype is unsupported by native inspection")
            })?,
        );
        if inspected.insert(name.to_owned(), metadata).is_some() {
            return Err(invalid_package("duplicate safetensors tensor name"));
        }
    }
    Ok(inspected)
}

fn tensor_metadata_dtype(dtype: safetensors::Dtype) -> Option<TensorMetadataDtype> {
    use safetensors::Dtype;
    Some(match dtype {
        Dtype::BOOL => TensorMetadataDtype::Bool,
        Dtype::U8 => TensorMetadataDtype::U8,
        Dtype::I8 => TensorMetadataDtype::I8,
        Dtype::U16 => TensorMetadataDtype::U16,
        Dtype::I16 => TensorMetadataDtype::I16,
        Dtype::U32 => TensorMetadataDtype::U32,
        Dtype::I32 => TensorMetadataDtype::I32,
        Dtype::U64 => TensorMetadataDtype::U64,
        Dtype::I64 => TensorMetadataDtype::I64,
        Dtype::F16 => TensorMetadataDtype::F16,
        Dtype::BF16 => TensorMetadataDtype::Bf16,
        Dtype::F32 => TensorMetadataDtype::F32,
        Dtype::C64 => TensorMetadataDtype::C64,
        Dtype::F8_E4M3 => TensorMetadataDtype::F8E4M3,
        Dtype::F8_E8M0 => TensorMetadataDtype::F8E8M0,
        _ => return None,
    })
}

fn validate_mlx_safetensors_tensor(
    tensor: &safetensors::tensor::TensorInfo,
) -> Result<(), NativeRuntimeError> {
    if tensor.shape.len() > 64 {
        return Err(invalid_package("safetensors tensor rank exceeds its bound"));
    }
    if !mlx_safetensors_dtype_supported(tensor.dtype) {
        return Err(invalid_package(
            "safetensors tensor dtype is unsupported by the pinned MLX loader",
        ));
    }
    let elements = tensor.shape.iter().try_fold(1u64, |total, &dimension| {
        if dimension == 0 || dimension > i32::MAX as usize {
            return Err(invalid_package(
                "safetensors dimension is outside the pinned MLX ShapeElem range",
            ));
        }
        total
            .checked_mul(dimension as u64)
            .ok_or_else(|| invalid_package("safetensors MLX shape product overflowed"))
    })?;
    let expected_bytes = elements
        .checked_mul(tensor.dtype.bitsize() as u64)
        .filter(|bits| bits % 8 == 0)
        .map(|bits| bits / 8)
        .ok_or_else(|| invalid_package("safetensors MLX tensor byte count is invalid"))?;
    let observed_bytes = tensor
        .data_offsets
        .1
        .checked_sub(tensor.data_offsets.0)
        .ok_or_else(|| invalid_package("safetensors tensor offsets are reversed"))?;
    if expected_bytes == 0 || observed_bytes as u64 != expected_bytes {
        return Err(invalid_package(
            "safetensors tensor shape/dtype disagrees with its MLX payload",
        ));
    }
    Ok(())
}

fn mlx_safetensors_dtype_supported(dtype: safetensors::Dtype) -> bool {
    use safetensors::Dtype;

    matches!(
        dtype,
        Dtype::BOOL
            | Dtype::U8
            | Dtype::I8
            | Dtype::U16
            | Dtype::I16
            | Dtype::U32
            | Dtype::I32
            | Dtype::U64
            | Dtype::I64
            | Dtype::F16
            | Dtype::BF16
            | Dtype::F32
            | Dtype::C64
            | Dtype::F8_E4M3
            | Dtype::F8_E8M0
    )
}

fn declared_tokenizer_primary(
    package: &NativeImmutablePackage,
) -> Result<Option<Arc<str>>, NativeRuntimeError> {
    select_declared_tokenizer_primary(&package.labels().collect::<Vec<_>>())
}

/// Exact primary selection shared by cold inspection and source-driven loading.
/// This function reads declarations only; it never opens or hashes an artifact.
pub(crate) fn select_declared_tokenizer_primary(
    labels: &[&str],
) -> Result<Option<Arc<str>>, NativeRuntimeError> {
    let primary = if labels.contains(&"tokenizer.json") {
        Some(Arc::<str>::from("tokenizer.json"))
    } else if labels.contains(&"tokenizer.model") {
        Some(Arc::<str>::from("tokenizer.model"))
    } else if labels.contains(&"tiktoken.model") {
        Some(Arc::<str>::from("tiktoken.model"))
    } else {
        let mut tiktoken = labels
            .iter()
            .copied()
            .filter(|label| label.ends_with(".tiktoken"))
            .collect::<Vec<_>>();
        tiktoken.sort_unstable();
        if tiktoken.len() > 1 {
            return Err(invalid_package(
                "multiple declared .tiktoken primary artifacts are ambiguous",
            ));
        }
        tiktoken
            .first()
            .map(|label| Arc::<str>::from(*label))
            .or_else(|| {
                labels
                    .contains(&"tokenizer.jsonl")
                    .then(|| Arc::<str>::from("tokenizer.jsonl"))
            })
            .or_else(|| {
                (labels.contains(&"vocab.json") && labels.contains(&"merges.txt"))
                    .then(|| Arc::<str>::from("vocab.json"))
            })
    };
    Ok(primary)
}

fn load_declared_tokenizer(
    package: &NativeImmutablePackage,
    primary: &str,
) -> Result<crate::tokenizer::MlxcelTokenizer, NativeRuntimeError> {
    preflight_native_tokenizer_work(package)?;
    let loader_labels = package
        .labels()
        .filter(|label| is_tokenizer_loader_component(label))
        .collect::<Vec<_>>();
    package.with_verified_staging_root(&loader_labels, |root| {
        crate::tokenizer::load_tokenizer_local_only_selected_declared(root, primary, &loader_labels)
            .map_err(|_error| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Tokenizer,
                    "root mlxcel tokenizer loader rejected declared local tokenizer artifacts",
                )
            })
    })
}

fn preflight_native_tokenizer_work(
    package: &NativeImmutablePackage,
) -> Result<(), NativeRuntimeError> {
    if package.component("tokenizer_config.json").is_some() {
        let bytes = package.read_component("tokenizer_config.json")?;
        if let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && !crate::tokenizer::native_added_token_work_is_bounded(&config)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "tokenizer added-token count or transition work exceeds the native bound",
            ));
        }
    }
    if package.component("added_tokens.json").is_some() {
        let bytes = package.read_component("added_tokens.json")?;
        if let Ok(tokens) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && tokens
                .as_object()
                .is_some_and(|tokens| tokens.len() > crate::tokenizer::MAX_NATIVE_ADDED_TOKENS)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "added_tokens.json entry count exceeds the native bound",
            ));
        }
    }
    Ok(())
}

fn inspect_tokenizer(
    package: &NativeImmutablePackage,
) -> Result<NativeTokenizerReport, NativeRuntimeError> {
    let format = if let Some(primary) = declared_tokenizer_primary(package)? {
        let tokenizer = load_declared_tokenizer(package, &primary)?;
        match tokenizer {
            crate::tokenizer::MlxcelTokenizer::HuggingFace(_) => NativeTokenizerFormat::HuggingFace,
            crate::tokenizer::MlxcelTokenizer::SentencePiece(_) => {
                NativeTokenizerFormat::SentencePiece
            }
            crate::tokenizer::MlxcelTokenizer::Tiktoken(_) => NativeTokenizerFormat::Tiktoken,
        }
    } else {
        NativeTokenizerFormat::Unsupported
    };
    let guarantee = match format {
        NativeTokenizerFormat::Tiktoken => IncrementalDecodeGuarantee::RawTokenBytes,
        NativeTokenizerFormat::HuggingFace => package
            .component("tokenizer.json")
            .map(|_| package.read_component("tokenizer.json"))
            .transpose()?
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .filter(is_proven_replace_byte_fallback_fuse)
            .map_or(IncrementalDecodeGuarantee::TerminalOnly, |_| {
                IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 }
            }),
        NativeTokenizerFormat::SentencePiece | NativeTokenizerFormat::Unsupported => {
            IncrementalDecodeGuarantee::TerminalOnly
        }
    };

    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"mlxcel.native-tokenizer.v1");
    // Package identity intentionally closes over every declared loader input,
    // including config-driven tokenizer overrides and added-token artifacts.
    hash_field(&mut hasher, &package.identity());
    for component in package
        .components
        .values()
        .map(|stored| &stored.declaration)
    {
        hash_field(&mut hasher, component.label.as_bytes());
        hasher.update(component.size_bytes.to_le_bytes());
        hash_field(&mut hasher, &component.sha256);
    }
    Ok(NativeTokenizerReport {
        format,
        identity: hasher.finalize().into(),
        incremental_decode: guarantee,
    })
}

/// The exact decoder used by the audited Gemma target/drafter resources:
/// BPE + `Replace("▁", " ")` + `ByteFallback` + `Fuse`. Each transform is
/// compositional and UTF-8 completion needs at most four byte pieces. Any
/// different or unparsed decoder remains terminal-only.
pub(crate) fn is_proven_replace_byte_fallback_fuse(tokenizer: &serde_json::Value) -> bool {
    if tokenizer
        .pointer("/model/type")
        .and_then(serde_json::Value::as_str)
        != Some("BPE")
    {
        return false;
    }
    let Some(decoders) = tokenizer
        .get("decoder")
        .filter(|decoder| {
            decoder.get("type").and_then(serde_json::Value::as_str) == Some("Sequence")
        })
        .and_then(|decoder| decoder.get("decoders"))
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if decoders.len() != 3 {
        return false;
    }
    let replacement = &decoders[0];
    let pattern_is_marker = replacement.get("pattern").is_some_and(|pattern| {
        pattern.as_str() == Some("▁")
            || pattern.get("String").and_then(serde_json::Value::as_str) == Some("▁")
    });
    replacement.get("type").and_then(serde_json::Value::as_str) == Some("Replace")
        && pattern_is_marker
        && replacement
            .get("content")
            .and_then(serde_json::Value::as_str)
            == Some(" ")
        && decoders[1].get("type").and_then(serde_json::Value::as_str) == Some("ByteFallback")
        && decoders[2].get("type").and_then(serde_json::Value::as_str) == Some("Fuse")
}

fn push_report_text(output: &mut Vec<Arc<str>>, value: &str) -> Result<(), NativeRuntimeError> {
    if value.is_empty()
        || value.len() > MAX_NATIVE_REPORT_TEXT_BYTES
        || output.len() >= MAX_NATIVE_REPORT_ITEMS
    {
        return Err(invalid_package(
            "model report text exceeds its native bound",
        ));
    }
    output.push(Arc::from(value));
    Ok(())
}

fn model_report_identity(report: &NativeModelReport) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"mlxcel.native-model-report.v1");
    hash_field(&mut hasher, &report.package_identity);
    hash_field(&mut hasher, report.model_type.as_bytes());
    hasher.update([report.operation_class as u8, report.tokenizer.format as u8]);
    hash_field(&mut hasher, &report.tokenizer.identity);
    match report.tokenizer.incremental_decode {
        IncrementalDecodeGuarantee::RawTokenBytes => hasher.update([0]),
        IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } => {
            hasher.update([1]);
            hasher.update((tokens as u64).to_le_bytes());
        }
        IncrementalDecodeGuarantee::TerminalOnly => hasher.update([2]),
    }
    match report.context_window {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_le_bytes());
        }
        None => hasher.update([0]),
    }
    match report.embedding_input_ceiling {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_le_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update([
        u8::from(report.architecture_supported),
        u8::from(report.modalities.text),
        u8::from(report.modalities.image),
        u8::from(report.modalities.audio),
        u8::from(report.modalities.video),
        u8::from(report.model_capabilities.generation),
        u8::from(report.model_capabilities.embedding),
        u8::from(report.model_capabilities.prepared_media),
        u8::from(report.managed_capabilities.generation),
        u8::from(report.managed_capabilities.embedding),
        u8::from(report.managed_capabilities.prepared_media),
    ]);
    hasher.update((report.speculative_modes.len() as u64).to_le_bytes());
    for mode in report.speculative_modes.iter() {
        hash_field(&mut hasher, mode.as_bytes());
    }
    hasher.update((report.warnings.len() as u64).to_le_bytes());
    for warning in report.warnings.iter() {
        hash_field(&mut hasher, warning.as_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use base64::Engine;
    use tempfile::TempDir;

    use super::*;
    use crate::native_runtime::{
        NativeBatchPolicy, NativeCachePolicy, NativeCacheTopology, NativeConstraintTransition,
        NativeContextPolicy, NativeContinuationDirective, NativeDeviceClass, NativeDevicePolicy,
        NativeDrafterTopology, NativeIncrementalDecoder, NativeModelOwnedStateMode,
        NativeOffloadPolicy, NativeOperationDriver, NativePreparedMultimodalClass,
        NativeSchedulerTopology, NativeSequenceStateTopology, NativeTokenizerDecoder,
        NativeWorkerConfig, NativeWorkerPolicy,
    };

    fn digest(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn write_component(root: &Path, label: &str, bytes: &[u8]) -> NativePackageComponent {
        let path = root.join(label);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        File::create(path).unwrap().write_all(bytes).unwrap();
        NativePackageComponent::new(label, bytes.len() as u64, digest(bytes)).unwrap()
    }

    struct DirectorySource {
        root: PathBuf,
    }

    impl NativePackageArtifactSource for DirectorySource {
        fn open_artifact(
            &mut self,
            label: &str,
        ) -> Result<Box<dyn Read + Send>, NativeRuntimeError> {
            let file = File::open(self.root.join(label)).map_err(|error| {
                invalid_package(format!(
                    "test artifact source cannot open {label:?}: {error}"
                ))
            })?;
            Ok(Box::new(file))
        }
    }

    fn acquire_from_directory(
        root: &Path,
        declarations: Vec<NativePackageComponent>,
        limits: NativePackageLimits,
    ) -> Result<NativeImmutablePackage, NativeRuntimeError> {
        let mut source = DirectorySource {
            root: root.to_path_buf(),
        };
        NativeImmutablePackage::acquire(&mut source, declarations, limits)
    }

    fn parse_test_sha256(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64, "qualification digest must be full SHA-256");
        let mut digest = [0_u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .expect("qualification digest must be hexadecimal");
        }
        digest
    }

    fn collect_test_files(root: &Path, directory: &Path, output: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(directory).expect("qualification directory is unreadable") {
            let entry = entry.expect("qualification directory entry is unreadable");
            let file_type = entry
                .file_type()
                .expect("qualification entry type is unreadable");
            assert!(
                !file_type.is_symlink(),
                "qualification input contains a symlink"
            );
            if file_type.is_dir() {
                collect_test_files(root, &entry.path(), output);
            } else {
                assert!(
                    file_type.is_file(),
                    "qualification input is not a regular file"
                );
                let relative = entry
                    .path()
                    .strip_prefix(root)
                    .expect("qualification file escaped its resource root")
                    .to_str()
                    .expect("qualification file label is not UTF-8")
                    .replace('\\', "/");
                assert!(
                    output.insert(relative),
                    "duplicate qualification file label"
                );
            }
        }
    }

    fn inventory_content_set(files: &[serde_json::Value]) -> String {
        let mut records = files
            .iter()
            .map(|file| {
                let path = file["path"]
                    .as_str()
                    .expect("qualification file path must be a string");
                let size = file["sizeBytes"]
                    .as_u64()
                    .expect("qualification file size must be u64");
                let sha256 = file["sha256"]
                    .as_str()
                    .expect("qualification file digest must be a string");
                (path.to_owned(), format!("{path}:{size}:{sha256}"))
            })
            .collect::<Vec<_>>();
        // The inventory producer uses StringComparer.Ordinal over paths and
        // then UTF-8 hashes LF-joined records with no trailing newline.
        records.sort_by(|left, right| left.0.encode_utf16().cmp(right.0.encode_utf16()));
        assert!(
            records.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "qualification inventory contains a duplicate file label"
        );
        let manifest = records
            .into_iter()
            .map(|(_, record)| record)
            .collect::<Vec<_>>()
            .join("\n");
        Sha256::digest(manifest.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn inventory_content_set_uses_canonical_records_not_json_order() {
        let first = serde_json::json!({
            "path": "README.md",
            "sizeBytes": 2,
            "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        });
        let second = serde_json::json!({
            "path": "a.bin",
            "sizeBytes": 1,
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        let canonical = concat!(
            "README.md:2:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
            "a.bin:1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        let expected: String = Sha256::digest(canonical.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            inventory_content_set(&[first.clone(), second.clone()]),
            expected
        );
        assert_eq!(inventory_content_set(&[second, first.clone()]), expected);

        let mut changed = first;
        changed["sizeBytes"] = serde_json::json!(3);
        assert_ne!(
            inventory_content_set(&[
                serde_json::json!({
                    "path": "a.bin",
                    "sizeBytes": 1,
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }),
                changed,
            ]),
            expected
        );
    }

    fn acquire_inventory_resource(
        inventory_root: &Path,
        inventory: &serde_json::Value,
        id: &str,
        expected_role: &str,
        expected_revision: &str,
        expected_content_set: &str,
        expected_file_count: usize,
        expected_total_bytes: u64,
    ) -> NativeImmutablePackage {
        let resource = inventory["resources"]
            .as_array()
            .expect("qualification inventory resources must be an array")
            .iter()
            .find(|resource| resource["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("qualification inventory is missing {id}"));
        assert_eq!(resource["role"].as_str(), Some(expected_role));
        assert_eq!(resource["revision"].as_str(), Some(expected_revision));
        let files = resource["files"]
            .as_array()
            .expect("qualification resource files must be an array");
        let observed_content_set = inventory_content_set(files);
        assert_eq!(
            resource["contentSetSha256"].as_str(),
            Some(observed_content_set.as_str()),
            "qualification inventory content-set field disagrees with its canonical file records"
        );
        assert_eq!(
            observed_content_set, expected_content_set,
            "qualification resource content set drifted"
        );
        assert_eq!(
            files.len(),
            expected_file_count,
            "qualification resource file count drifted"
        );

        let resource_root = inventory_root.join(id);
        assert!(
            resource_root.is_dir(),
            "qualification resource directory is missing"
        );
        let mut declared_labels = BTreeSet::new();
        let mut declarations = Vec::with_capacity(files.len());
        let mut total_bytes = 0_u64;
        for file in files {
            let label = file["path"]
                .as_str()
                .expect("qualification file path must be a string");
            assert!(
                declared_labels.insert(label.to_owned()),
                "qualification inventory contains a duplicate file label"
            );
            let size = file["sizeBytes"]
                .as_u64()
                .expect("qualification file size must be u64");
            total_bytes = total_bytes
                .checked_add(size)
                .expect("qualification resource size overflowed");
            let sha256 = parse_test_sha256(
                file["sha256"]
                    .as_str()
                    .expect("qualification file digest must be a string"),
            );
            declarations.push(
                NativePackageComponent::new(label, size, sha256)
                    .expect("qualification declaration is invalid"),
            );
        }
        assert_eq!(
            total_bytes, expected_total_bytes,
            "qualification resource byte total drifted"
        );
        let mut observed_labels = BTreeSet::new();
        collect_test_files(&resource_root, &resource_root, &mut observed_labels);
        assert_eq!(
            observed_labels, declared_labels,
            "qualification resource differs from its exact inventory"
        );

        acquire_from_directory(
            &resource_root,
            declarations,
            NativePackageLimits {
                max_components: files.len(),
                max_total_bytes: total_bytes,
                max_inspection_bytes: MAX_NATIVE_INSPECTION_BYTES,
            },
        )
        .expect("qualification resource failed immutable acquisition")
    }

    fn basic_tokenizer_json() -> Vec<u8> {
        let crate::tokenizer::MlxcelTokenizer::HuggingFace(tokenizer) =
            crate::tokenizer::MlxcelTokenizer::stub()
        else {
            unreachable!()
        };
        tokenizer.to_string(false).unwrap().into_bytes()
    }

    fn safetensors_bytes(names: &[&str]) -> Vec<u8> {
        let mut entries = serde_json::Map::new();
        let mut offset = 0u64;
        for name in names {
            entries.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [offset, offset + 4]
                }),
            );
            offset += 4;
        }
        let mut header = serde_json::to_vec(&entries).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + offset as usize, 0);
        bytes
    }

    fn safetensors_with_payload(name: &str, payload_bytes: usize) -> Vec<u8> {
        assert_eq!(payload_bytes % 4, 0);
        let mut header = serde_json::to_vec(&serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": [payload_bytes / 4],
                "data_offsets": [0, payload_bytes]
            }
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + payload_bytes, 0);
        bytes
    }

    fn safetensors_from_header(header: serde_json::Value, payload_bytes: usize) -> Vec<u8> {
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + payload_bytes, 0);
        bytes
    }

    fn safetensors_f32_shapes(tensors: Vec<(String, Vec<usize>)>) -> Vec<u8> {
        let mut entries = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in tensors {
            let elements = shape.iter().copied().product::<usize>();
            let end = offset.checked_add(elements * 4).unwrap();
            entries.insert(
                name,
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
        safetensors_from_header(serde_json::Value::Object(entries), offset)
    }

    fn tiny_bert_safetensors() -> Vec<u8> {
        const HIDDEN: usize = 8;
        const INTERMEDIATE: usize = 16;
        let mut tensors = vec![
            (
                "bert.embeddings.word_embeddings.weight".to_string(),
                vec![32, HIDDEN],
            ),
            (
                "bert.embeddings.position_embeddings.weight".to_string(),
                vec![24, HIDDEN],
            ),
            (
                "bert.embeddings.token_type_embeddings.weight".to_string(),
                vec![2, HIDDEN],
            ),
            ("bert.embeddings.LayerNorm.weight".to_string(), vec![HIDDEN]),
        ];
        for index in 0..2 {
            let base = format!("bert.encoder.layer.{index}");
            for projection in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                tensors.push((format!("{base}.{projection}.weight"), vec![HIDDEN, HIDDEN]));
            }
            tensors.push((
                format!("{base}.intermediate.dense.weight"),
                vec![INTERMEDIATE, HIDDEN],
            ));
            tensors.push((
                format!("{base}.output.dense.weight"),
                vec![HIDDEN, INTERMEDIATE],
            ));
            for norm in ["attention.output.LayerNorm", "output.LayerNorm"] {
                tensors.push((format!("{base}.{norm}.weight"), vec![HIDDEN]));
            }
        }
        safetensors_f32_shapes(tensors)
    }

    fn write_minimal_weights(root: &Path, tensor: &str) -> NativePackageComponent {
        write_component(root, "model.safetensors", &safetensors_bytes(&[tensor]))
    }

    fn pinned_sentencepiece_bytes() -> Vec<u8> {
        const EXPECTED_SIZE: usize = 253_270;
        const EXPECTED_SHA256: [u8; 32] = [
            0x68, 0xe8, 0x2c, 0xf4, 0xc9, 0xc3, 0xef, 0x0b, 0xe1, 0xf2, 0x5a, 0x7c, 0xae, 0x2c,
            0x83, 0x6f, 0xca, 0xfa, 0x7d, 0x3c, 0x83, 0x82, 0x1f, 0xb9, 0x86, 0x41, 0x1e, 0x56,
            0x77, 0x9a, 0xd1, 0xe2,
        ];
        let encoded = include_str!("../../testdata/native_runtime/sentencepiece-toy.model.b64")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("pinned SentencePiece fixture base64 is corrupt");
        assert_eq!(bytes.len(), EXPECTED_SIZE);
        assert_eq!(digest(&bytes), EXPECTED_SHA256);
        bytes
    }

    fn full_singleton_tiktoken_vocab() -> Vec<u8> {
        (0u16..=255)
            .flat_map(|rank| {
                let token = base64::engine::general_purpose::STANDARD.encode([rank as u8]);
                format!("{token} {rank}\n").into_bytes()
            })
            .collect()
    }

    fn generation_policy() -> NativeWorkerPolicy {
        NativeWorkerPolicy {
            operation_driver: NativeOperationDriver::AutoregressiveGenerate,
            scheduler_topology: NativeSchedulerTopology::SingleSequence,
            sequence_state_topology: NativeSequenceStateTopology::Independent,
            context: NativeContextPolicy {
                max_context_tokens: 64,
                max_position: 64,
            },
            batch: NativeBatchPolicy {
                max_active_sequences: 1,
                max_queued_operations: 1,
                max_batch_tokens: 64,
                prefill_chunk_tokens: 16,
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
            max_prompt_tokens: 32,
            max_generation_tokens: 32,
            max_embedding_inputs: 1,
            max_embedding_dimensions: 1,
            max_accepted_chunk_units: 4_096,
            max_constraint_bytes: 1,
        }
    }

    #[test]
    fn package_reopens_only_allowlisted_verified_components() {
        let root = TempDir::new().unwrap();
        let config = br#"{"model_type":"llama"}"#;
        let declaration = write_component(root.path(), "config.json", config);
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(package.read_component("config.json").unwrap(), config);
        assert_eq!(
            package.read_component("../secret").unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );
        assert_eq!(
            package.read_component("tokenizer.json").unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );
    }

    #[test]
    fn concurrent_readers_use_independent_verified_file_handles() {
        let root = TempDir::new().unwrap();
        let bytes = Arc::<[u8]>::from(vec![0x5a; 64 * 1024]);
        let declaration = write_component(root.path(), "weights.bin", &bytes);
        let package = Arc::new(
            acquire_from_directory(
                root.path(),
                vec![declaration],
                NativePackageLimits::default(),
            )
            .unwrap(),
        );
        let threads = (0..8)
            .map(|_| {
                let package = Arc::clone(&package);
                let expected = Arc::clone(&bytes);
                std::thread::spawn(move || {
                    for _ in 0..16 {
                        assert_eq!(package.read_component("weights.bin").unwrap(), &*expected);
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn source_mutation_after_acquisition_cannot_change_package_bytes() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        std::fs::write(root.path().join("config.json"), b"two").unwrap();
        assert_eq!(package.read_component("config.json").unwrap(), b"one");
    }

    #[test]
    fn private_materialization_is_removed_when_package_drops() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "nested/config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        let staging_root = package.staging.root().to_path_buf();
        assert!(staging_root.join("nested/config.json").is_file());
        drop(package);
        assert!(!staging_root.exists());
    }

    struct FailAfterFirstArtifact {
        first: Vec<u8>,
        opens: usize,
    }

    impl NativePackageArtifactSource for FailAfterFirstArtifact {
        fn open_artifact(
            &mut self,
            _label: &str,
        ) -> Result<Box<dyn Read + Send>, NativeRuntimeError> {
            self.opens += 1;
            if self.opens == 1 {
                Ok(Box::new(Cursor::new(self.first.clone())))
            } else {
                Err(package_changed(
                    "synthetic source failure after first artifact",
                ))
            }
        }
    }

    #[test]
    fn partial_acquisition_failure_removes_private_materialization() {
        let parent = TempDir::new().unwrap();
        let first = b"first".to_vec();
        let declarations = vec![
            NativePackageComponent::new("first.bin", first.len() as u64, digest(&first)).unwrap(),
            NativePackageComponent::new("second.bin", 6, digest(b"second")).unwrap(),
        ];
        let mut source = FailAfterFirstArtifact { first, opens: 0 };
        let error = match NativeImmutablePackage::acquire_in(
            &mut source,
            declarations,
            NativePackageLimits::default(),
            Some(parent.path()),
        ) {
            Ok(_) => panic!("partial acquisition unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::PackageChanged);
        assert!(
            std::fs::read_dir(parent.path()).unwrap().next().is_none(),
            "failed acquisition leaked its private materialization"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_materialization_excludes_writers_from_the_first_created_handle() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ProbingReader {
            bytes: Cursor<Vec<u8>>,
            parent: PathBuf,
            attempted: Arc<AtomicBool>,
            writer_denied: Arc<AtomicBool>,
        }
        impl Read for ProbingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if !self.attempted.swap(true, Ordering::SeqCst) {
                    let staging = std::fs::read_dir(&self.parent)?
                        .next()
                        .expect("acquisition created its private staging directory")?
                        .path();
                    let writer = OpenOptions::new()
                        .write(true)
                        .open(staging.join("config.json"));
                    self.writer_denied.store(writer.is_err(), Ordering::SeqCst);
                }
                self.bytes.read(buffer)
            }
        }
        struct ProbingSource {
            parent: PathBuf,
            bytes: Vec<u8>,
            attempted: Arc<AtomicBool>,
            writer_denied: Arc<AtomicBool>,
        }
        impl NativePackageArtifactSource for ProbingSource {
            fn open_artifact(
                &mut self,
                _label: &str,
            ) -> Result<Box<dyn Read + Send>, NativeRuntimeError> {
                Ok(Box::new(ProbingReader {
                    bytes: Cursor::new(self.bytes.clone()),
                    parent: self.parent.clone(),
                    attempted: Arc::clone(&self.attempted),
                    writer_denied: Arc::clone(&self.writer_denied),
                }))
            }
        }

        let parent = TempDir::new().unwrap();
        let bytes = b"sealed from creation".to_vec();
        let attempted = Arc::new(AtomicBool::new(false));
        let writer_denied = Arc::new(AtomicBool::new(false));
        let mut source = ProbingSource {
            parent: parent.path().to_path_buf(),
            bytes: bytes.clone(),
            attempted: Arc::clone(&attempted),
            writer_denied: Arc::clone(&writer_denied),
        };
        let package = NativeImmutablePackage::acquire_in(
            &mut source,
            vec![
                NativePackageComponent::new("config.json", bytes.len() as u64, digest(&bytes))
                    .unwrap(),
            ],
            NativePackageLimits::default(),
            Some(parent.path()),
        )
        .unwrap();
        assert!(attempted.load(Ordering::SeqCst));
        assert!(writer_denied.load(Ordering::SeqCst));
        assert_eq!(package.read_component("config.json").unwrap(), bytes);
    }

    #[test]
    fn transient_undeclared_loader_input_invalidates_the_result() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        let error = package
            .with_verified_staging_root(&["config.json"], |staging| {
                let injected = staging.join("tiktoken.model");
                std::fs::write(&injected, b"injected")
                    .map_err(|error| invalid_package(error.to_string()))?;
                let observed =
                    std::fs::read(&injected).map_err(|error| invalid_package(error.to_string()))?;
                std::fs::remove_file(&injected)
                    .map_err(|error| invalid_package(error.to_string()))?;
                Ok(observed)
            })
            .unwrap_err();
        assert_eq!(error.kind(), NativeRuntimeErrorKind::PackageChanged);
    }

    #[cfg(not(windows))]
    #[test]
    fn same_size_declared_rewrite_is_rejected_before_loader_use() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        let path = package.staging.root().join("config.json");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        std::fs::write(&path, b"two").unwrap();
        let error = package
            .with_verified_staging_root(
                &["config.json"],
                |_staging| -> Result<(), NativeRuntimeError> {
                    panic!("loader must not observe a rewritten declared component")
                },
            )
            .unwrap_err();
        assert_eq!(error.kind(), NativeRuntimeErrorKind::PackageChanged);
    }

    #[cfg(windows)]
    #[test]
    fn lifetime_guard_prevents_same_size_rewrite_and_timestamp_restoration() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        let path = package.staging.root().join("config.json");
        assert!(
            OpenOptions::new().write(true).open(&path).is_err(),
            "the package-lifetime no-write handle must reject a writer before bytes or timestamps can change"
        );
        assert_eq!(package.read_component("config.json").unwrap(), b"one");
    }

    #[cfg(unix)]
    #[test]
    fn loader_mutate_then_restore_cannot_return_unverified_bytes() {
        use std::os::unix::fs::PermissionsExt;

        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        let error = package
            .with_verified_staging_root(&["config.json"], |staging| {
                let path = staging.join("config.json");
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                    .map_err(|error| invalid_package(error.to_string()))?;
                std::fs::write(&path, b"two")
                    .map_err(|error| invalid_package(error.to_string()))?;
                let observed =
                    std::fs::read(&path).map_err(|error| invalid_package(error.to_string()))?;
                std::fs::write(&path, b"one")
                    .map_err(|error| invalid_package(error.to_string()))?;
                Ok(observed)
            })
            .unwrap_err();
        assert_eq!(error.kind(), NativeRuntimeErrorKind::PackageChanged);
    }

    struct SwapOnOpenSource {
        root: PathBuf,
    }

    impl NativePackageArtifactSource for SwapOnOpenSource {
        fn open_artifact(
            &mut self,
            label: &str,
        ) -> Result<Box<dyn Read + Send>, NativeRuntimeError> {
            let path = self.root.join(label);
            let file = File::open(&path).map_err(|error| invalid_package(error.to_string()))?;
            let moved = self.root.join("source-before-swap");
            std::fs::rename(&path, &moved)
                .map_err(|error| invalid_package(format!("cannot swap source path: {error}")))?;
            std::fs::write(&path, b"replacement")
                .map_err(|error| invalid_package(format!("cannot replace source path: {error}")))?;
            Ok(Box::new(file))
        }
    }

    #[test]
    fn source_path_swap_after_open_cannot_change_materialized_bytes() {
        let root = TempDir::new().unwrap();
        let original = b"declared snapshot";
        let declaration = write_component(root.path(), "config.json", original);
        let mut source = SwapOnOpenSource {
            root: root.path().to_path_buf(),
        };
        let package = NativeImmutablePackage::acquire(
            &mut source,
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(package.read_component("config.json").unwrap(), original);
        assert_eq!(
            std::fs::read(root.path().join("config.json")).unwrap(),
            b"replacement"
        );
    }

    struct MutatingReader {
        first: Cursor<Vec<u8>>,
        replacement: Cursor<Vec<u8>>,
        switched: bool,
    }

    impl Read for MutatingReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if !self.switched {
                self.switched = true;
                let amount = buffer.len().min(1);
                return self.first.read(&mut buffer[..amount]);
            }
            self.replacement.read(buffer)
        }
    }

    struct MutatingSource;

    impl NativePackageArtifactSource for MutatingSource {
        fn open_artifact(
            &mut self,
            _label: &str,
        ) -> Result<Box<dyn Read + Send>, NativeRuntimeError> {
            Ok(Box::new(MutatingReader {
                first: Cursor::new(b"declared".to_vec()),
                replacement: Cursor::new(b"hostile".to_vec()),
                switched: false,
            }))
        }
    }

    #[test]
    fn source_mutation_during_materialization_fails_digest_verification() {
        let declaration =
            NativePackageComponent::new("config.json", 8, digest(b"declared")).unwrap();
        let result = NativeImmutablePackage::acquire(
            &mut MutatingSource,
            vec![declaration],
            NativePackageLimits::default(),
        );
        let error = match result {
            Ok(_) => panic!("mutated source unexpectedly acquired"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::PackageChanged);
    }

    #[test]
    fn inspection_is_neutral_and_package_bound() {
        let root = TempDir::new().unwrap();
        let config =
            br#"{"model_type":"florence2","max_position_embeddings":2048,"vision_config":{}}"#;
        let tokenizer = basic_tokenizer_json();
        let tokenizer_config = br#"{"clean_up_tokenization_spaces":false}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(root.path(), "tokenizer.json", &tokenizer),
                write_component(root.path(), "tokenizer_config.json", tokenizer_config),
                write_minimal_weights(root.path(), "model.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = inspect_package(&package).unwrap();
        assert_eq!(report.package_identity(), package.identity());
        assert_eq!(
            report.operation_class(),
            MlxNativeOperationClass::EncoderDecoder
        );
        assert_eq!(report.context_window(), Some(2048));
        assert!(!report.architecture_supported());
        assert_eq!(report.modalities(), NativeModalitySet::default());
        assert_eq!(
            report.model_capabilities(),
            NativeModelCapabilities::default()
        );
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("explicitly deferred"))
        );
        assert_eq!(
            report.tokenizer().incremental_decode(),
            IncrementalDecodeGuarantee::TerminalOnly
        );
        assert_ne!(report.report_identity(), [0; 32]);
    }

    #[test]
    fn oversized_inspection_read_fails_closed() {
        let root = TempDir::new().unwrap();
        let bytes = b"12345";
        let declaration = write_component(root.path(), "config.json", bytes);
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits {
                max_components: 1,
                max_total_bytes: 5,
                max_inspection_bytes: 4,
            },
        )
        .unwrap();
        assert_eq!(
            package.read_component("config.json").unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );
    }

    #[test]
    fn aggregate_and_each_tokenizer_family_are_preflighted_before_parse() {
        let tokenizer = basic_tokenizer_json();
        let config = br#"{"model_type":"llama"}"#;
        let root = TempDir::new().unwrap();
        let aggregate_limit = tokenizer.len().max(config.len()) as u64;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(root.path(), "tokenizer.json", &tokenizer),
            ],
            NativePackageLimits {
                max_components: 2,
                max_total_bytes: (config.len() + tokenizer.len()) as u64,
                max_inspection_bytes: aggregate_limit,
            },
        )
        .unwrap();
        assert_eq!(
            inspect_package(&package).unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );

        for label in [
            "tokenizer.json",
            "tokenizer.model",
            "tiktoken.model",
            "tiny.tiktoken",
            "model.safetensors.index.json",
        ] {
            let root = TempDir::new().unwrap();
            let bytes = vec![b'x'; 64];
            let package = acquire_from_directory(
                root.path(),
                vec![
                    write_component(root.path(), "config.json", config),
                    write_component(root.path(), label, &bytes),
                ],
                NativePackageLimits {
                    max_components: 2,
                    max_total_bytes: 128,
                    max_inspection_bytes: 63,
                },
            )
            .unwrap();
            assert_eq!(
                inspect_package(&package).unwrap_err().kind(),
                NativeRuntimeErrorKind::InvalidPackage,
                "{label} must be bounded before parser allocation"
            );
        }
    }

    #[test]
    fn weight_index_is_bound_to_declared_shards_and_actual_tensor_headers() {
        let cases = [
            (
                "missing shard",
                br#"{"weight_map":{"claimed.weight":"missing.safetensors"}}"#.as_slice(),
                None,
            ),
            (
                "fake key",
                br#"{"weight_map":{"claimed.weight":"weights.safetensors"}}"#.as_slice(),
                Some(safetensors_bytes(&["actual.weight"])),
            ),
            (
                "declared shard without tensor",
                br#"{"weight_map":{"claimed.weight":"weights.safetensors"}}"#.as_slice(),
                Some(safetensors_bytes(&[])),
            ),
        ];
        for (name, index, shard) in cases {
            let root = TempDir::new().unwrap();
            let mut declarations = vec![write_component(
                root.path(),
                "model.safetensors.index.json",
                index,
            )];
            if let Some(shard) = shard {
                declarations.push(write_component(root.path(), "weights.safetensors", &shard));
            }
            let package =
                acquire_from_directory(root.path(), declarations, NativePackageLimits::default())
                    .unwrap();
            assert!(
                inspect_weight_index(&package, &mut 0, false).is_err(),
                "{name} must fail closed"
            );
        }

        let root = TempDir::new().unwrap();
        let nested = safetensors_bytes(&["claimed.weight"]);
        let nested_index = br#"{"weight_map":{"claimed.weight":"nested/weights.safetensors"}}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "model.safetensors.index.json", nested_index),
                write_component(root.path(), "nested/weights.safetensors", &nested),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(
            inspect_weight_index(&package, &mut 0, false)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidPackage,
            "the root loader accepts only plain index shard filenames"
        );

        let root = TempDir::new().unwrap();
        let package = acquire_from_directory(
            root.path(),
            vec![write_component(
                root.path(),
                "nested/weights.safetensors",
                &nested,
            )],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert!(
            inspect_weight_index(&package, &mut 0, false)
                .unwrap()
                .is_empty(),
            "nested unindexed weights are not a root-loader executable shard"
        );

        let root = TempDir::new().unwrap();
        let shard = safetensors_bytes(&["claimed.weight"]);
        let index = br#"{"weight_map":{"claimed.weight":"weights.safetensors"}}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "model.safetensors.index.json", index),
                write_component(root.path(), "weights.safetensors", &shard),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let sharded = inspect_weight_index(&package, &mut 0, false).unwrap();
        assert!(sharded.contains_key("claimed.weight"));

        let root = TempDir::new().unwrap();
        let single = safetensors_bytes(&["single.weight"]);
        let package = acquire_from_directory(
            root.path(),
            vec![write_component(root.path(), "model.safetensors", &single)],
            NativePackageLimits::default(),
        )
        .unwrap();
        let unsharded = inspect_weight_index(&package, &mut 0, false).unwrap();
        assert!(unsharded.contains_key("single.weight"));
    }

    #[test]
    fn safetensors_header_uses_the_pinned_parser_and_exact_data_length() {
        let cases = [
            (
                "unknown dtype",
                serde_json::json!({
                    "tensor": {"dtype":"U4", "shape":[2], "data_offsets":[0,1]}
                }),
                1,
                false,
            ),
            (
                "misaligned sub-byte tensor",
                serde_json::json!({
                    "tensor": {"dtype":"F4", "shape":[1], "data_offsets":[0,1]}
                }),
                1,
                false,
            ),
            (
                "invalid metadata value type",
                serde_json::json!({
                    "__metadata__":{"revision":7},
                    "tensor": {"dtype":"U8", "shape":[1], "data_offsets":[0,1]}
                }),
                1,
                false,
            ),
            (
                "offset outside declared data",
                serde_json::json!({
                    "tensor": {"dtype":"U8", "shape":[2], "data_offsets":[0,2]}
                }),
                1,
                false,
            ),
            (
                "Rust-valid but MLX-unsupported six-bit tensor",
                serde_json::json!({
                    "tensor": {"dtype":"F6_E2M3", "shape":[4], "data_offsets":[0,3]}
                }),
                3,
                false,
            ),
            (
                "valid complex tensor",
                serde_json::json!({
                    "tensor": {"dtype":"C64", "shape":[1], "data_offsets":[0,8]}
                }),
                8,
                true,
            ),
        ];
        for (name, header, payload_bytes, valid) in cases {
            let root = TempDir::new().unwrap();
            let bytes = safetensors_from_header(header, payload_bytes);
            let package = acquire_from_directory(
                root.path(),
                vec![write_component(root.path(), "model.safetensors", &bytes)],
                NativePackageLimits::default(),
            )
            .unwrap();
            assert_eq!(
                inspect_weight_index(&package, &mut 0, false).is_ok(),
                valid,
                "{name}"
            );
        }

        let root = TempDir::new().unwrap();
        let empty = safetensors_from_header(
            serde_json::json!({
                "tensor": {"dtype":"U8", "shape":[0], "data_offsets":[0,0]}
            }),
            0,
        );
        let package = acquire_from_directory(
            root.path(),
            vec![write_component(root.path(), "model.safetensors", &empty)],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(
            inspect_weight_index(&package, &mut 0, false)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );

        let exact_dimension = safetensors::tensor::TensorInfo {
            dtype: safetensors::Dtype::U8,
            shape: vec![i32::MAX as usize],
            data_offsets: (0, i32::MAX as usize),
        };
        assert!(validate_mlx_safetensors_tensor(&exact_dimension).is_ok());
        let over_dimension = safetensors::tensor::TensorInfo {
            dtype: safetensors::Dtype::U8,
            shape: vec![i32::MAX as usize + 1],
            data_offsets: (0, i32::MAX as usize + 1),
        };
        assert_eq!(
            validate_mlx_safetensors_tensor(&over_dimension)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidPackage
        );
    }

    #[test]
    fn sparse_large_weight_inspection_reads_only_its_bounded_header() {
        const PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
        const INSPECTION_LIMIT: u64 = 1024;

        let root = TempDir::new().unwrap();
        let weights = safetensors_with_payload("large.weight", PAYLOAD_BYTES);
        let declaration = write_component(root.path(), "model.safetensors", &weights);
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits {
                max_components: 1,
                max_total_bytes: weights.len() as u64,
                max_inspection_bytes: INSPECTION_LIMIT,
            },
        )
        .unwrap();
        let mut inspected = 0;
        let keys = inspect_weight_index(&package, &mut inspected, false).unwrap();
        assert!(keys.contains_key("large.weight"));
        assert!(inspected <= INSPECTION_LIMIT);
        assert!(inspected < weights.len() as u64);
    }

    #[test]
    fn embedding_weight_projection_matches_immediate_subfolder_loader_names() {
        let nested_only_root = TempDir::new().unwrap();
        let nested_only = safetensors_bytes(&["linear.weight"]);
        let package = acquire_from_directory(
            nested_only_root.path(),
            vec![write_component(
                nested_only_root.path(),
                "2_Dense/model.safetensors",
                &nested_only,
            )],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(
            inspect_weight_index(&package, &mut 0, true)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidPackage,
            "the embedding loader returns before scanning subfolders when no top-level shard is selected"
        );

        let skipped_root = TempDir::new().unwrap();
        let skipped = safetensors_bytes(&["adapter.weight"]);
        let package = acquire_from_directory(
            skipped_root.path(),
            vec![
                write_component(skipped_root.path(), "adapter_model.safetensors", &skipped),
                write_component(
                    skipped_root.path(),
                    "2_Dense/model.safetensors",
                    &nested_only,
                ),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(
            inspect_weight_index(&package, &mut 0, true)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidPackage,
            "skip-named top-level shards do not unlock embedding subfolder discovery"
        );

        let root = TempDir::new().unwrap();
        let base = safetensors_bytes(&["bert.embeddings.word_embeddings.weight"]);
        let dense = safetensors_bytes(&["linear.weight"]);
        let skipped = safetensors_bytes(&["adapter.weight"]);
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "model.safetensors", &base),
                write_component(root.path(), "2_Dense/model.safetensors", &dense),
                write_component(root.path(), "2_Dense/adapter_model.safetensors", &skipped),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let generic = inspect_weight_index(&package, &mut 0, false).unwrap();
        assert!(generic.contains_key("bert.embeddings.word_embeddings.weight"));
        assert!(!generic.contains_key("2_Dense.linear.weight"));

        let embedding = inspect_weight_index(&package, &mut 0, true).unwrap();
        assert!(embedding.contains_key("bert.embeddings.word_embeddings.weight"));
        assert!(embedding.contains_key("2_Dense.linear.weight"));
        assert!(!embedding.contains_key("2_Dense.adapter.weight"));
    }

    #[test]
    fn incomplete_or_deferred_reports_never_claim_executable_capability() {
        let tokenizer = basic_tokenizer_json();
        for (index, config) in [
            br#"{"model_type":"llama","max_position_embeddings":64}"#.as_slice(),
            br#"{"model_type":"bert","architectures":["BertModel"],"max_position_embeddings":64}"#
                .as_slice(),
            br#"{"model_type":"whisper","max_position_embeddings":64}"#.as_slice(),
            br#"{"istftnet":{},"max_position_embeddings":64}"#.as_slice(),
        ]
        .into_iter()
        .enumerate()
        {
            let root = TempDir::new().unwrap();
            let mut components = vec![
                write_component(root.path(), "config.json", config),
                write_component(root.path(), "tokenizer.json", &tokenizer),
            ];
            if index == 1 {
                components.push(write_minimal_weights(
                    root.path(),
                    "bert.embeddings.word_embeddings.weight",
                ));
            }
            let package =
                acquire_from_directory(root.path(), components, NativePackageLimits::default())
                    .unwrap();
            let report = inspect_package(&package).unwrap();
            assert!(!report.architecture_supported());
            assert_eq!(
                report.model_capabilities(),
                NativeModelCapabilities::default()
            );
            assert_eq!(
                report.managed_capabilities(),
                NativeManagedCapabilities::default()
            );
            assert_eq!(report.modalities(), NativeModalitySet::default());
            assert!(!report.warnings().is_empty());
        }

        let root = TempDir::new().unwrap();
        let config =
            br#"{"model_type":"bert","architectures":["BertModel"],"max_position_embeddings":64}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(root.path(), "tokenizer.json", &tokenizer),
                write_minimal_weights(root.path(), "bert.embeddings.word_embeddings.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = inspect_package(&package).unwrap();
        assert_eq!(report.operation_class(), MlxNativeOperationClass::Embedding);
        assert!(!report.architecture_supported());
        assert_eq!(
            report.model_capabilities(),
            NativeModelCapabilities::default()
        );
        assert_eq!(
            report.managed_capabilities(),
            NativeManagedCapabilities::default()
        );
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("BERT native descriptor rejected"))
        );
    }

    #[test]
    fn qualified_bert_report_requires_loader_compatible_pooling() {
        let acquire = |pooling: &[u8]| {
            let root = TempDir::new().unwrap();
            let config = br#"{
                "model_type":"bert",
                "architectures":["BertModel"],
                "vocab_size":32,
                "hidden_size":8,
                "num_hidden_layers":2,
                "num_attention_heads":2,
                "intermediate_size":16,
                "max_position_embeddings":24,
                "type_vocab_size":2,
                "hidden_act":"gelu"
            }"#;
            let tokenizer = basic_tokenizer_json();
            let weights = tiny_bert_safetensors();
            acquire_from_directory(
                root.path(),
                vec![
                    write_component(root.path(), "config.json", config),
                    write_component(root.path(), "tokenizer.json", &tokenizer),
                    write_component(root.path(), "model.safetensors", &weights),
                    write_component(root.path(), "1_Pooling/config.json", pooling),
                ],
                NativePackageLimits::default(),
            )
            .unwrap()
        };

        let compatible = acquire(br#"{"pooling_mode":"mean"}"#);
        let report = inspect_package(&compatible).unwrap();
        assert!(report.architecture_supported());
        assert_eq!(report.context_window(), Some(24));
        assert_eq!(
            report.model_capabilities(),
            NativeModelCapabilities {
                embedding: true,
                ..NativeModelCapabilities::default()
            }
        );
        assert_eq!(
            report.managed_capabilities(),
            NativeManagedCapabilities {
                embedding: true,
                ..NativeManagedCapabilities::default()
            }
        );
        let mut policy = generation_policy();
        policy.operation_driver = NativeOperationDriver::Embedding;
        policy.context.max_context_tokens = 24;
        policy.context.max_position = 24;
        policy.max_prompt_tokens = 24;
        assert_eq!(
            NativeWorkerConfig::try_from_report(&report, policy)
                .expect("a qualified inspected BERT package must mint a managed worker config")
                .operation_driver(),
            NativeOperationDriver::Embedding
        );

        let unsupported = acquire(br#"{"pooling_mode_weightedmean_tokens":true}"#);
        let report = inspect_package(&unsupported).unwrap();
        assert!(!report.architecture_supported());
        assert_eq!(
            report.model_capabilities(),
            NativeModelCapabilities::default()
        );
        assert_eq!(
            report.managed_capabilities(),
            NativeManagedCapabilities::default()
        );
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("pooling config"))
        );
    }

    #[test]
    fn embedding_input_ceiling_uses_the_driver_sidecar_contract() {
        use crate::models::ModelType;

        let root = TempDir::new().unwrap();
        let config = br#"{"model_type":"bert","max_position_embeddings":512}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(
                    root.path(),
                    "sentence_bert_config.json",
                    br#"{"max_seq_length":256}"#,
                ),
                write_component(
                    root.path(),
                    "tokenizer_config.json",
                    br#"{"model_max_length":512}"#,
                ),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let config: serde_json::Value = serde_json::from_slice(config).unwrap();
        assert_eq!(
            inspect_embedding_input_ceiling(&package, ModelType::Bert, &config).unwrap(),
            Some(256)
        );

        let root = TempDir::new().unwrap();
        let config = br#"{"model_type":"qwen3","max_position_embeddings":32768}"#;
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(
                    root.path(),
                    "tokenizer_config.json",
                    br#"{"model_max_length":32768}"#,
                ),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let config: serde_json::Value = serde_json::from_slice(config).unwrap();
        assert_eq!(
            inspect_embedding_input_ceiling(&package, ModelType::Qwen3Embedding, &config).unwrap(),
            Some(crate::embeddings::limits::EMBEDDING_MAX_LENGTH_CAP as u32)
        );

        let root = TempDir::new().unwrap();
        let package = acquire_from_directory(
            root.path(),
            vec![write_component(
                root.path(),
                "config.json",
                config.to_string().as_bytes(),
            )],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert_eq!(
            inspect_embedding_input_ceiling(&package, ModelType::Qwen3Embedding, &config).unwrap(),
            None,
            "a generic context scalar cannot invent the embedding driver ceiling"
        );
    }

    #[test]
    fn exact_gemma_decoder_ast_is_the_only_hf_streaming_proof() {
        let exact = serde_json::json!({
            "model": {"type": "BPE"},
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"},
                    {"type": "Fuse"}
                ]
            }
        });
        assert!(is_proven_replace_byte_fallback_fuse(&exact));
        let mut changed = exact.clone();
        changed["decoder"]["decoders"][0]["content"] = serde_json::json!("_");
        assert!(!is_proven_replace_byte_fallback_fuse(&changed));
        changed = exact;
        changed["decoder"]["decoders"].as_array_mut().unwrap().pop();
        assert!(!is_proven_replace_byte_fallback_fuse(&changed));
    }

    #[test]
    fn actual_hf_package_report_and_decoder_share_the_proven_gemma_contract() {
        let root = TempDir::new().unwrap();
        let crate::tokenizer::MlxcelTokenizer::HuggingFace(fixture) =
            crate::tokenizer::MlxcelTokenizer::stub_gemma_sequence_byte_fallback()
        else {
            unreachable!()
        };
        let tokenizer_json = fixture.to_string(false).unwrap().into_bytes();
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(
                    root.path(),
                    "config.json",
                    br#"{"model_type":"gemma4","max_position_embeddings":64}"#,
                ),
                write_component(root.path(), "tokenizer.json", &tokenizer_json),
                write_minimal_weights(root.path(), "model.embed_tokens.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = inspect_package(&package).unwrap();
        assert_eq!(
            report.tokenizer().incremental_decode(),
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 }
        );
        assert_eq!(
            NativeWorkerConfig::try_from_report(&report, generation_policy())
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::UnsupportedCapability,
            "a proved tokenizer must pass the guarantee gate before the Order03 managed gate"
        );

        let mut qualified_report = report.clone();
        qualified_report.managed_capabilities.generation = true;
        qualified_report.report_identity = model_report_identity(&qualified_report);
        let worker_config =
            NativeWorkerConfig::try_from_report(&qualified_report, generation_policy()).unwrap();
        let authority = package
            .bind_worker_tokenizer(&qualified_report, &worker_config)
            .unwrap();
        let mut authority_decoder =
            NativeTokenizerDecoder::from_worker_authority(authority, &[]).unwrap();
        let authority_ids = [0xEA_u8, 0xB0, 0x80]
            .map(|byte| fixture.token_to_id(&format!("<0x{byte:02X}>")).unwrap());
        let mut authority_emitted = Vec::new();
        for id in authority_ids {
            authority_emitted
                .extend_from_slice(authority_decoder.push(id as i32).unwrap().stable_bytes());
        }
        authority_emitted.extend_from_slice(authority_decoder.flush().unwrap().stable_bytes());
        assert_eq!(authority_emitted, "가".as_bytes());

        let tokenizer = package
            .with_verified_staging_root(&["config.json", "tokenizer.json"], |staging| {
                crate::tokenizer::load_tokenizer_local_only_selected(staging, "tokenizer.json")
                    .map(Arc::new)
                    .map_err(|error| invalid_package(error.to_string()))
            })
            .unwrap();
        let prompt = vec![tokenizer.token_to_id("▁Hi").unwrap()];
        let generated = [0xEA_u8, 0xB0, 0x80]
            .map(|byte| tokenizer.token_to_id(&format!("<0x{byte:02X}>")).unwrap())
            .to_vec();
        let prompt_bytes = tokenizer.decode(&prompt, false).unwrap().into_bytes();
        let mut all = prompt.clone();
        all.extend_from_slice(&generated);
        let one_shot = tokenizer.decode(&all, false).unwrap().into_bytes();
        let expected = one_shot.strip_prefix(prompt_bytes.as_slice()).unwrap();
        let prompt = prompt.iter().map(|&id| id as i32).collect::<Vec<_>>();
        let mut decoder = NativeTokenizerDecoder::from_proven_profile(
            tokenizer,
            &prompt,
            report.tokenizer().incremental_decode(),
        )
        .unwrap();
        let mut emitted = Vec::new();
        for id in generated {
            emitted.extend_from_slice(decoder.push(id as i32).unwrap().stable_bytes());
        }
        emitted.extend_from_slice(decoder.flush().unwrap().stable_bytes());
        assert_eq!(emitted, expected);
        assert_eq!(emitted, "가".as_bytes());

        let mutated_root = TempDir::new().unwrap();
        let mut mutated: serde_json::Value = serde_json::from_slice(&tokenizer_json).unwrap();
        mutated["decoder"]["decoders"][0]["content"] = serde_json::json!("_");
        let mutated = serde_json::to_vec(&mutated).unwrap();
        let mutated_package = acquire_from_directory(
            mutated_root.path(),
            vec![
                write_component(
                    mutated_root.path(),
                    "config.json",
                    br#"{"model_type":"gemma4","max_position_embeddings":64}"#,
                ),
                write_component(mutated_root.path(), "tokenizer.json", &mutated),
                write_minimal_weights(mutated_root.path(), "model.embed_tokens.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let mutated_report = inspect_package(&mutated_package).unwrap();
        assert_eq!(
            mutated_report.tokenizer().incremental_decode(),
            IncrementalDecodeGuarantee::TerminalOnly
        );
        assert_eq!(
            NativeWorkerConfig::try_from_report(&mutated_report, generation_policy())
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::UnsupportedGuarantee
        );
        let mismatch =
            match mutated_package.bind_worker_tokenizer(&qualified_report, &worker_config) {
                Ok(_) => panic!("a package/report identity mismatch minted tokenizer authority"),
                Err(error) => error,
            };
        assert_eq!(mismatch.kind(), NativeRuntimeErrorKind::InvalidRequest);
        let mut changed_report = qualified_report.clone();
        changed_report.warnings = Arc::from([Arc::<str>::from("different report identity")]);
        changed_report.report_identity = model_report_identity(&changed_report);
        let mismatch = match package.bind_worker_tokenizer(&changed_report, &worker_config) {
            Ok(_) => panic!("a report/config identity mismatch minted tokenizer authority"),
            Err(error) => error,
        };
        assert_eq!(mismatch.kind(), NativeRuntimeErrorKind::InvalidRequest);
    }

    #[test]
    fn actual_sentencepiece_package_is_terminal_only_and_rejected_for_streaming_generation() {
        let root = TempDir::new().unwrap();
        let sentencepiece = pinned_sentencepiece_bytes();
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(
                    root.path(),
                    "config.json",
                    br#"{"model_type":"llama","max_position_embeddings":64}"#,
                ),
                write_component(root.path(), "tokenizer.model", &sentencepiece),
                write_minimal_weights(root.path(), "model.embed_tokens.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = inspect_package(&package).unwrap();
        assert_eq!(
            report.tokenizer().format(),
            NativeTokenizerFormat::SentencePiece
        );
        assert_eq!(
            report.tokenizer().incremental_decode(),
            IncrementalDecodeGuarantee::TerminalOnly
        );
        assert_eq!(
            NativeWorkerConfig::try_from_report(&report, generation_policy())
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::UnsupportedGuarantee
        );

        let tokenizer = package
            .with_verified_staging_root(&["config.json", "tokenizer.model"], |staging| {
                crate::tokenizer::load_tokenizer_local_only_selected(staging, "tokenizer.model")
                    .map(Arc::new)
                    .map_err(|error| invalid_package(error.to_string()))
            })
            .unwrap();
        let ids = [8_u32, 465];
        let first = tokenizer.decode(&ids[..1], false).unwrap().into_bytes();
        let expected = tokenizer.decode(&ids, false).unwrap().into_bytes();
        assert!(std::str::from_utf8(&expected).is_ok());
        assert!(
            expected.starts_with(&first) && first.len() < expected.len(),
            "the pinned two-token SentencePiece fixture must retain a real proper-prefix choice"
        );
        let mut decoder =
            NativeTokenizerDecoder::terminal_only(Arc::clone(&tokenizer), &[]).unwrap();
        assert!(
            decoder
                .push(ids[0] as i32)
                .unwrap()
                .stable_bytes()
                .is_empty()
        );
        assert!(
            decoder
                .push(ids[1] as i32)
                .unwrap()
                .stable_bytes()
                .is_empty()
        );
        assert_eq!(decoder.flush().unwrap().stable_bytes(), expected);

        let directives = [
            NativeContinuationDirective::ConsumePrefixAndContinue {
                consumed_units: 1,
                constraint: NativeConstraintTransition::Unchanged,
            },
            NativeContinuationDirective::Stop { consumed_units: 1 },
            NativeContinuationDirective::Fail {
                consumed_units: 1,
                error: NativeRuntimeError::new(
                    NativeRuntimeErrorKind::SinkRejected,
                    "proper-prefix semantic failure",
                ),
            },
        ];
        for _directive in directives {
            let sink_calls = AtomicUsize::new(0);
            let admission = NativeWorkerConfig::try_from_report(&report, generation_policy());
            if admission.is_ok() {
                sink_calls.fetch_add(1, Ordering::Relaxed);
            }
            assert_eq!(
                admission.unwrap_err().kind(),
                NativeRuntimeErrorKind::UnsupportedGuarantee
            );
            assert_eq!(
                sink_calls.load(Ordering::Relaxed),
                0,
                "TerminalOnly must reject before every Continue/Stop/Fail sink shape"
            );
        }
        let mut oov = NativeTokenizerDecoder::terminal_only(tokenizer, &[]).unwrap();
        assert_eq!(
            oov.push(i32::MAX).unwrap_err().kind(),
            super::super::error::TokenizerErrorKind::InvalidTokenId
        );
    }

    #[test]
    fn actual_tiktoken_package_mints_only_its_raw_byte_worker_authority() {
        let root = TempDir::new().unwrap();
        let vocabulary = full_singleton_tiktoken_vocab();
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(
                    root.path(),
                    "config.json",
                    br#"{"model_type":"hunyuan_v1_dense","max_position_embeddings":64}"#,
                ),
                write_component(root.path(), "tiktoken.model", &vocabulary),
                write_component(root.path(), "tokenizer_config.json", b"{}"),
                write_minimal_weights(root.path(), "model.embed_tokens.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let mut report = inspect_package(&package).unwrap();
        assert_eq!(
            report.tokenizer().incremental_decode(),
            IncrementalDecodeGuarantee::RawTokenBytes
        );
        report.managed_capabilities.generation = true;
        report.report_identity = model_report_identity(&report);
        let config = NativeWorkerConfig::try_from_report(&report, generation_policy()).unwrap();
        let authority = package.bind_worker_tokenizer(&report, &config).unwrap();
        let mut decoder = NativeTokenizerDecoder::from_worker_authority(authority, &[]).unwrap();
        assert_eq!(decoder.push(i32::from(b'a')).unwrap().stable_bytes(), b"a");
        assert!(decoder.flush().unwrap().stable_bytes().is_empty());
    }

    #[test]
    fn unknown_model_type_is_never_defaulted_to_autoregressive() {
        let root = TempDir::new().unwrap();
        let config = br#"{"model_type":"hostile_unknown"}"#;
        let tokenizer = basic_tokenizer_json();
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", config),
                write_component(root.path(), "tokenizer.json", &tokenizer),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = inspect_package(&package).unwrap();
        assert_eq!(
            report.operation_class(),
            MlxNativeOperationClass::UnsupportedByNativeRuntime
        );
        assert!(!report.warnings().is_empty());
    }

    #[test]
    fn malformed_local_tokenizer_families_are_typed_and_hide_staging_paths() {
        for (label, bytes) in [
            ("tokenizer.json", b"not tokenizer json".as_slice()),
            ("tokenizer.model", b"not sentencepiece".as_slice()),
            ("tiktoken.model", b"not tiktoken".as_slice()),
        ] {
            let root = TempDir::new().unwrap();
            let package = acquire_from_directory(
                root.path(),
                vec![
                    write_component(root.path(), "config.json", br#"{"model_type":"llama"}"#),
                    write_component(root.path(), label, bytes),
                    write_minimal_weights(root.path(), "model.weight"),
                ],
                NativePackageLimits::default(),
            )
            .unwrap();
            let error = inspect_package(&package).unwrap_err();
            assert_eq!(error.kind(), NativeRuntimeErrorKind::Tokenizer, "{label}");
            assert!(
                !error
                    .to_string()
                    .contains(package.staging.root().to_string_lossy().as_ref()),
                "{label} leaked its staging root"
            );
            assert!(!error.to_string().contains("mlxcel-native-package-"));
        }
    }

    #[test]
    fn declared_tokenizer_config_is_strictly_validated_by_package_inspection() {
        let invalid = [
            ("invalid JSON", "{"),
            ("non-object root", "[]"),
            ("wrong BOS flag", r#"{"add_bos_token":"yes"}"#),
            ("wrong added-token map", r#"{"added_tokens_decoder":[]}"#),
            (
                "wrong added-token flag",
                r#"{"added_tokens_decoder":{"100":{"content":"<x>","special":"yes"}}}"#,
            ),
        ];
        for (name, config) in invalid {
            let root = TempDir::new().unwrap();
            let package = acquire_from_directory(
                root.path(),
                vec![
                    write_component(
                        root.path(),
                        "config.json",
                        br#"{"model_type":"llama","max_position_embeddings":64}"#,
                    ),
                    write_component(root.path(), "tokenizer.json", &basic_tokenizer_json()),
                    write_component(root.path(), "tokenizer_config.json", config.as_bytes()),
                    write_minimal_weights(root.path(), "model.embed_tokens.weight"),
                ],
                NativePackageLimits::default(),
            )
            .unwrap();
            assert_eq!(
                inspect_package(&package).unwrap_err().kind(),
                NativeRuntimeErrorKind::Tokenizer,
                "{name}"
            );
        }

        for config in [None, Some("{}")] {
            let root = TempDir::new().unwrap();
            let mut declarations = vec![
                write_component(
                    root.path(),
                    "config.json",
                    br#"{"model_type":"llama","max_position_embeddings":64}"#,
                ),
                write_component(root.path(), "tokenizer.json", &basic_tokenizer_json()),
                write_minimal_weights(root.path(), "model.embed_tokens.weight"),
            ];
            if let Some(config) = config {
                declarations.push(write_component(
                    root.path(),
                    "tokenizer_config.json",
                    config.as_bytes(),
                ));
            }
            let package =
                acquire_from_directory(root.path(), declarations, NativePackageLimits::default())
                    .unwrap();
            assert!(inspect_package(&package).is_ok());
        }
    }

    #[test]
    fn tokenizer_added_token_count_and_transition_work_are_preflight_bounded() {
        fn config(entries: usize, alternating: bool) -> Vec<u8> {
            let mut decoder = serde_json::Map::new();
            for id in 0..entries {
                decoder.insert(
                    (10_000 + id).to_string(),
                    serde_json::json!({
                        "content": format!("<native-{id}>"),
                        "special": alternating && id % 2 == 0,
                    }),
                );
            }
            serde_json::to_vec(&serde_json::json!({
                "added_tokens_decoder": decoder,
            }))
            .unwrap()
        }

        for (name, entries, alternating, expected) in [
            (
                "entry ceiling",
                crate::tokenizer::MAX_NATIVE_ADDED_TOKENS,
                false,
                Ok(()),
            ),
            (
                "entry ceiling plus one",
                crate::tokenizer::MAX_NATIVE_ADDED_TOKENS + 1,
                false,
                Err(NativeRuntimeErrorKind::ResourceLimit),
            ),
            (
                "transition ceiling",
                crate::tokenizer::MAX_NATIVE_ADDED_TOKEN_RUNS,
                true,
                Ok(()),
            ),
            (
                "transition ceiling plus one",
                crate::tokenizer::MAX_NATIVE_ADDED_TOKEN_RUNS + 1,
                true,
                Err(NativeRuntimeErrorKind::ResourceLimit),
            ),
        ] {
            let root = TempDir::new().unwrap();
            let config = config(entries, alternating);
            let package = acquire_from_directory(
                root.path(),
                vec![write_component(
                    root.path(),
                    "tokenizer_config.json",
                    &config,
                )],
                NativePackageLimits::default(),
            )
            .unwrap();
            let observed = preflight_native_tokenizer_work(&package);
            match expected {
                Ok(()) => assert!(observed.is_ok(), "{name}"),
                Err(kind) => assert_eq!(observed.unwrap_err().kind(), kind, "{name}"),
            }
        }
    }

    #[test]
    fn report_identity_distinguishes_limits_and_capability_projection() {
        let root = TempDir::new().unwrap();
        let tokenizer = basic_tokenizer_json();
        let package = acquire_from_directory(
            root.path(),
            vec![
                write_component(root.path(), "config.json", br#"{"model_type":"llama"}"#),
                write_component(root.path(), "tokenizer.json", &tokenizer),
                write_minimal_weights(root.path(), "model.weight"),
            ],
            NativePackageLimits::default(),
        )
        .unwrap();
        let mut report = inspect_package(&package).unwrap();
        report.context_window = None;
        let absent = model_report_identity(&report);
        report.context_window = Some(0);
        assert_ne!(absent, model_report_identity(&report));

        report.context_window = None;
        report.embedding_input_ceiling = None;
        let absent_embedding = model_report_identity(&report);
        report.embedding_input_ceiling = Some(0);
        assert_ne!(absent_embedding, model_report_identity(&report));

        report.embedding_input_ceiling = None;
        report.architecture_supported = false;
        let unsupported_architecture = model_report_identity(&report);
        report.architecture_supported = true;
        assert_ne!(unsupported_architecture, model_report_identity(&report));

        report.architecture_supported = false;
        report.modalities = NativeModalitySet::default();
        report.model_capabilities = NativeModelCapabilities::default();
        let no_capabilities = model_report_identity(&report);
        report.modalities.image = true;
        assert_ne!(no_capabilities, model_report_identity(&report));
        report.modalities.image = false;
        report.model_capabilities.generation = true;
        assert_ne!(no_capabilities, model_report_identity(&report));
    }

    #[test]
    #[ignore = "qualification profile: streams and verifies about 7 GiB of pinned checkpoints"]
    fn pinned_qualification_packages_match_loader_owned_descriptors() {
        assert_eq!(
            std::env::var("MLXCEL_REQUIRE_PINNED_CHECKPOINTS").as_deref(),
            Ok("1"),
            "the pinned-checkpoint qualification profile must be explicitly enabled"
        );
        const TARGET_REVISION: &str = "73bcf09092aa277861d5a191b989b666f7f32e8f";
        const TARGET_CONTENT_SET: &str =
            "0e420cec93ed42eda64dfb86ce29acd4ab1ab5ff6030d468f6deae9c5f0ea5dc";
        const ASSISTANT_REVISION: &str = "537797012376cd600cf94c16d880d0db51a98be5";
        const ASSISTANT_CONTENT_SET: &str =
            "adfd2aadb3e1e2099655214c973b543c9856d8eec689e2cea1462bc1ad026b01";
        const MINILM_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
        const MINILM_CONTENT_SET: &str =
            "345fbcf14b4d6423e2d46c6b37ff73a0a052f5df8107bd302a192ce5f77b5ec9";

        let inventory_root_text = std::env::var("MLXCEL_PINNED_INVENTORY_ROOT")
            .expect("MLXCEL_PINNED_INVENTORY_ROOT must name the prepared inventory directory");
        let inventory_root = PathBuf::from(&inventory_root_text);
        let inventory_path = inventory_root.join("resource-inventory.json");
        let inventory_bytes = std::fs::read(&inventory_path)
            .expect("the pinned qualification inventory is missing or unreadable");
        let inventory: serde_json::Value = serde_json::from_slice(&inventory_bytes)
            .expect("the pinned qualification inventory is invalid JSON");
        assert_eq!(inventory["schemaVersion"].as_u64(), Some(1));
        assert_eq!(
            inventory["mlxcelRevision"].as_str(),
            Some("b2be0d0ab36b17ba35e9ae9d96fe7f19a195d5df")
        );
        assert_eq!(
            inventory["destinationRoot"].as_str(),
            Some(inventory_root_text.as_str())
        );
        assert_eq!(
            inventory["resources"]
                .as_array()
                .expect("qualification resources must be an array")
                .len(),
            3,
            "qualification inventory contains an unexpected resource"
        );

        let target = acquire_inventory_resource(
            &inventory_root,
            &inventory,
            "gemma-4-12b-it-4bit",
            "target",
            TARGET_REVISION,
            TARGET_CONTENT_SET,
            10,
            6_773_371_278,
        );
        let target_report = inspect_package(&target).expect("target inspection failed");
        assert_eq!(target_report.model_type(), "gemma4_unified");
        assert_eq!(
            target_report.operation_class(),
            MlxNativeOperationClass::Autoregressive
        );
        assert!(target_report.architecture_supported());
        assert_eq!(target_report.context_window(), Some(262_144));
        assert_eq!(
            target_report.modalities(),
            NativeModalitySet {
                text: true,
                image: true,
                audio: true,
                video: true,
            }
        );
        assert_eq!(
            target_report.model_capabilities(),
            NativeModelCapabilities {
                generation: true,
                embedding: false,
                prepared_media: true,
            }
        );
        assert_eq!(
            target_report.managed_capabilities(),
            NativeManagedCapabilities {
                generation: true,
                ..NativeManagedCapabilities::default()
            }
        );

        let target_config: serde_json::Value =
            serde_json::from_slice(&target.read_component("config.json").unwrap()).unwrap();
        let target_processor: serde_json::Value =
            serde_json::from_slice(&target.read_component("processor_config.json").unwrap())
                .unwrap();
        let target_raw = inspect_weight_index(&target, &mut 0, false).unwrap();
        let target_metadata = canonicalize_inspected_tensor_metadata(&target_raw, |key| {
            if key.contains("rotary_emb") || key == "lm_head.weight" {
                None
            } else {
                Some(crate::loading::normalize_gemma4_unified_key(key))
            }
        })
        .unwrap();
        crate::loading::validate_native_gemma4_unified_tensor_metadata(
            &target_metadata,
            &target_config,
            Some(&target_processor),
        )
        .unwrap();
        let mut target_incomplete = target_metadata.clone();
        assert!(
            target_incomplete
                .remove("language_model.model.norm.weight")
                .is_some()
        );
        assert!(
            crate::loading::validate_native_gemma4_unified_tensor_metadata(
                &target_incomplete,
                &target_config,
                Some(&target_processor),
            )
            .is_err()
        );
        let mut target_wrong_shape = target_metadata;
        target_wrong_shape.insert(
            "language_model.model.norm.weight".to_string(),
            TensorMetadata::new(vec![1], TensorMetadataDtype::Bf16),
        );
        assert!(
            crate::loading::validate_native_gemma4_unified_tensor_metadata(
                &target_wrong_shape,
                &target_config,
                Some(&target_processor),
            )
            .is_err()
        );
        drop(target);

        let assistant = acquire_inventory_resource(
            &inventory_root,
            &inventory,
            "gemma-4-12b-it-assistant-4bit",
            "speculative_drafter",
            ASSISTANT_REVISION,
            ASSISTANT_CONTENT_SET,
            7,
            270_075_926,
        );
        let assistant_report = inspect_package(&assistant).expect("assistant inspection failed");
        assert_eq!(assistant_report.model_type(), "gemma4_unified_assistant");
        assert!(assistant_report.architecture_supported());
        assert_eq!(assistant_report.context_window(), Some(131_072));
        assert_eq!(assistant_report.modalities(), NativeModalitySet::default());
        assert_eq!(
            assistant_report.model_capabilities(),
            NativeModelCapabilities::default()
        );
        assert_eq!(
            assistant_report.managed_capabilities(),
            NativeManagedCapabilities::default()
        );
        assert_eq!(
            assistant_report
                .speculative_modes()
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<_>>(),
            vec!["gemma4-assistant-drafter"]
        );
        let assistant_config: serde_json::Value =
            serde_json::from_slice(&assistant.read_component("config.json").unwrap()).unwrap();
        let assistant_metadata = inspect_weight_index(&assistant, &mut 0, false).unwrap();
        mlxcel_core::drafter::gemma4_assistant::validate_native_tensor_metadata(
            &assistant_metadata,
            &assistant_config,
        )
        .unwrap();
        let mut assistant_incomplete = assistant_metadata.clone();
        assert!(assistant_incomplete.remove("model.norm.weight").is_some());
        assert!(
            mlxcel_core::drafter::gemma4_assistant::validate_native_tensor_metadata(
                &assistant_incomplete,
                &assistant_config,
            )
            .is_err()
        );
        let mut assistant_wrong_dtype = assistant_metadata;
        assistant_wrong_dtype.insert(
            "model.norm.weight".to_string(),
            TensorMetadata::new(vec![1024], TensorMetadataDtype::U32),
        );
        assert!(
            mlxcel_core::drafter::gemma4_assistant::validate_native_tensor_metadata(
                &assistant_wrong_dtype,
                &assistant_config,
            )
            .is_err()
        );
        drop(assistant);

        let minilm = acquire_inventory_resource(
            &inventory_root,
            &inventory,
            "all-minilm-l6-v2",
            "embedding",
            MINILM_REVISION,
            MINILM_CONTENT_SET,
            11,
            91_578_415,
        );
        let minilm_report = inspect_package(&minilm).expect("MiniLM inspection failed");
        assert_eq!(minilm_report.model_type(), "bert");
        assert_eq!(
            minilm_report.operation_class(),
            MlxNativeOperationClass::Embedding
        );
        assert!(minilm_report.architecture_supported());
        assert_eq!(minilm_report.context_window(), Some(256));
        assert_eq!(minilm_report.embedding_input_ceiling(), Some(256));
        assert_eq!(
            minilm_report.modalities(),
            NativeModalitySet {
                text: true,
                ..NativeModalitySet::default()
            }
        );
        assert_eq!(
            minilm_report.model_capabilities(),
            NativeModelCapabilities {
                embedding: true,
                ..NativeModelCapabilities::default()
            }
        );
        assert_eq!(
            minilm_report.managed_capabilities(),
            NativeManagedCapabilities {
                embedding: true,
                ..NativeManagedCapabilities::default()
            }
        );
        let minilm_config: serde_json::Value =
            serde_json::from_slice(&minilm.read_component("config.json").unwrap()).unwrap();
        let minilm_raw = inspect_weight_index(&minilm, &mut 0, true).unwrap();
        let minilm_metadata = canonicalize_inspected_tensor_metadata(&minilm_raw, |key| {
            crate::models::bert::canonical_native_weight_key(key, false).map(|key| key.into_owned())
        })
        .unwrap();
        let dense_quantization = mlxcel_core::tensor_metadata::TensorQuantization {
            group_size: 64,
            bits: 4,
        };
        crate::models::bert::validate_native_tensor_metadata(
            &minilm_metadata,
            &minilm_config,
            dense_quantization,
        )
        .unwrap();
        let mut minilm_incomplete = minilm_metadata.clone();
        assert!(
            minilm_incomplete
                .remove("embeddings.LayerNorm.weight")
                .is_some()
        );
        assert!(
            crate::models::bert::validate_native_tensor_metadata(
                &minilm_incomplete,
                &minilm_config,
                dense_quantization,
            )
            .is_err()
        );
        let mut minilm_wrong_shape = minilm_metadata;
        minilm_wrong_shape.insert(
            "embeddings.LayerNorm.weight".to_string(),
            TensorMetadata::new(vec![1], TensorMetadataDtype::F32),
        );
        assert!(
            crate::models::bert::validate_native_tensor_metadata(
                &minilm_wrong_shape,
                &minilm_config,
                dense_quantization,
            )
            .is_err()
        );
    }

    #[test]
    fn undeclared_files_and_windows_ambiguous_labels_are_rejected() {
        let root = TempDir::new().unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        std::fs::write(root.path().join("undeclared.json"), b"two").unwrap();
        let package = acquire_from_directory(
            root.path(),
            vec![declaration],
            NativePackageLimits::default(),
        )
        .unwrap();
        assert!(package.component("undeclared.json").is_none());
        for label in ["AUX.rs", "weights.bin:stream", "trailing. "] {
            assert!(NativePackageComponent::new(label, 0, digest(b"")).is_err());
        }
    }

    #[test]
    fn case_aliases_are_rejected_before_filesystem_resolution() {
        let root = TempDir::new().unwrap();
        let lower = write_component(root.path(), "config.json", b"one");
        let alias = NativePackageComponent::new("CONFIG.JSON", 3, digest(b"one")).unwrap();
        assert!(
            acquire_from_directory(
                root.path(),
                vec![lower, alias],
                NativePackageLimits::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn oversized_component_label_is_rejected_before_owned_copy() {
        let label = "x".repeat(MAX_NATIVE_PACKAGE_LABEL_BYTES + 1);
        assert!(NativePackageComponent::new(label.as_str(), 0, digest(b"")).is_err());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn non_utf8_tree_entry_is_rejected_without_lossy_aliasing() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join(OsString::from_vec(vec![0xff])), b"x").unwrap();
        let declaration = write_component(root.path(), "config.json", b"one");
        assert!(
            acquire_from_directory(
                root.path(),
                vec![declaration],
                NativePackageLimits::default(),
            )
            .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn non_utf8_source_root_is_rejected_as_a_typed_error_without_lossy_aliasing() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let root = TempDir::new().unwrap();
        let invalid_root = root.path().join(OsString::from_vec(vec![0xff]));
        let declaration = write_component(root.path(), "config.json", b"one");
        let error = match acquire_from_directory(
            &invalid_root,
            vec![declaration],
            NativePackageLimits::default(),
        ) {
            Ok(_) => panic!("an invalid macOS source root was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::InvalidPackage);
        let lossy_root = invalid_root.to_string_lossy().into_owned();
        assert!(!error.to_string().contains(&lossy_root));
        assert!(!format!("{error:?}").contains(&lossy_root));
    }
}
