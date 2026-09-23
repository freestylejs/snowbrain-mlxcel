//! Bounded access to a producer-verified immutable source and native-owned backing.
//!
//! This module does not authenticate untrusted bytes. Its trusted producer must
//! retain the actual verified source capability for the entire source lifetime.
//! Identities are opaque producer identities, never inferred from paths or
//! re-created by hashing at load time. Native static inspection remains separate.

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::model_report::{
    MAX_NATIVE_INSPECTION_BYTES, MAX_NATIVE_PACKAGE_BYTES, MAX_NATIVE_PACKAGE_COMPONENTS,
    MAX_NATIVE_PACKAGE_LABEL_TOTAL_BYTES, validate_component_label,
};

const MAX_READ_BYTES: usize = 1024 * 1024;

/// One immutable component declaration already bound by the source producer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVerifiedComponent {
    label: Arc<str>,
    byte_len: u64,
    identity: [u8; 32],
}

impl NativeVerifiedComponent {
    /// Validate a canonical data label; `identity` is not a native SHA-256 claim.
    pub fn new(label: &str, byte_len: u64, identity: [u8; 32]) -> Result<Self, NativeRuntimeError> {
        validate_component_label(label)?;
        if label.chars().any(char::is_control)
            || label
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(invalid("verified source label is not canonical"));
        }
        let extension = label.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        if [
            "bat", "cmd", "com", "dll", "dylib", "exe", "jar", "msi", "node", "ps1", "py", "pyc",
            "pyo", "sh", "so", "wasm", "zsh",
        ]
        .contains(&extension.as_str())
        {
            return Err(invalid(
                "verified source cannot declare executable components",
            ));
        }
        Ok(Self {
            label: Arc::from(label),
            byte_len,
            identity,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Per-load allocation and transfer bounds; no unbounded read is permitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeLoadLimits {
    pub max_components: usize,
    pub max_total_bytes: u64,
    pub max_metadata_bytes: usize,
    pub max_read_bytes: usize,
}

impl Default for NativeLoadLimits {
    fn default() -> Self {
        Self {
            max_components: MAX_NATIVE_PACKAGE_COMPONENTS,
            max_total_bytes: MAX_NATIVE_PACKAGE_BYTES,
            max_metadata_bytes: MAX_NATIVE_INSPECTION_BYTES as usize,
            max_read_bytes: MAX_READ_BYTES,
        }
    }
}

impl NativeLoadLimits {
    fn validate(self) -> Result<(), NativeRuntimeError> {
        if self.max_components == 0
            || self.max_components > MAX_NATIVE_PACKAGE_COMPONENTS
            || self.max_total_bytes == 0
            || self.max_total_bytes > MAX_NATIVE_PACKAGE_BYTES
            || self.max_metadata_bytes == 0
            || self.max_metadata_bytes as u64 > MAX_NATIVE_INSPECTION_BYTES
            || self.max_read_bytes == 0
            || self.max_read_bytes > MAX_READ_BYTES
        {
            return Err(resource(
                "native source limits exceed their finite ceilings",
            ));
        }
        Ok(())
    }
}

/// Observations for one load attempt, not proof or final inference usage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeLoadStatistics {
    /// Actual producer callback invocations, including failed invocations.
    pub source_read_calls: u64,
    /// Bytes reported by successful callbacks within the requested range.
    pub source_bytes: u64,
    /// Bytes accepted into bounded metadata buffers.
    pub metadata_bytes: u64,
    /// Bytes whose write into private native backing completed successfully.
    /// A failed OS write may have an unobserved partial effect; no residence is
    /// published in that case. These are not native device/page-in counters.
    pub backing_bytes: u64,
    /// Largest producer read request, in bytes.
    pub largest_read: usize,
}

/// Trusted immutable read capability implemented by the installing frontend.
///
/// It must serve only the declared immutable bytes and retain their live owner.
/// `read_at` fills the complete requested range or returns an error. It must not
/// open an original path or network/cache fallback. Cancellation is observed
/// between ranges; this trait does not interrupt an OS call already in progress.
pub trait NativeVerifiedArtifactSource: Send + Sync {
    fn read_at(
        &self,
        label: &str,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<usize, NativeRuntimeError>;
}

/// Cloneable retained source; adoption checks data declarations but reads no bytes.
#[derive(Clone)]
pub struct NativeVerifiedSource {
    identity: [u8; 32],
    components: Arc<[NativeVerifiedComponent]>,
    source: Arc<dyn NativeVerifiedArtifactSource>,
    limits: NativeLoadLimits,
}

impl NativeVerifiedSource {
    pub fn new(
        identity: [u8; 32],
        mut components: Vec<NativeVerifiedComponent>,
        source: Arc<dyn NativeVerifiedArtifactSource>,
        limits: NativeLoadLimits,
    ) -> Result<Self, NativeRuntimeError> {
        limits.validate()?;
        if identity == [0; 32] || components.is_empty() || components.len() > limits.max_components
        {
            return Err(invalid(
                "verified source identity or component count is invalid",
            ));
        }
        let mut total = 0_u64;
        let mut label_bytes = 0_usize;
        let mut aliases = Vec::new();
        aliases
            .try_reserve_exact(components.len())
            .map_err(|_| resource("cannot reserve source declaration index"))?;
        for component in &components {
            total = add(total, component.byte_len)?;
            label_bytes = label_bytes
                .checked_add(component.label.len())
                .ok_or_else(|| resource("source label total overflowed"))?;
            if total > limits.max_total_bytes || label_bytes > MAX_NATIVE_PACKAGE_LABEL_TOTAL_BYTES
            {
                return Err(resource("verified source declaration totals exceed limits"));
            }
            aliases.push(component.label.to_ascii_lowercase());
        }
        aliases.sort_unstable();
        if aliases.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                "verified source contains duplicate or aliased labels",
            ));
        }
        components.sort_unstable_by(|left, right| left.label.cmp(&right.label));
        Ok(Self {
            identity,
            components: components.into(),
            source,
            limits,
        })
    }

    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn components(&self) -> &[NativeVerifiedComponent] {
        &self.components
    }
    pub const fn limits(&self) -> NativeLoadLimits {
        self.limits
    }

    pub fn component(&self, label: &str) -> Option<&NativeVerifiedComponent> {
        self.components
            .binary_search_by(|entry| entry.label.as_ref().cmp(label))
            .ok()
            .map(|index| &self.components[index])
    }

    /// Read one declared metadata artifact with per-read and aggregate limits.
    pub fn read_metadata(
        &self,
        label: &str,
        cancelled: &AtomicBool,
        statistics: &mut NativeLoadStatistics,
    ) -> Result<Vec<u8>, NativeRuntimeError> {
        check_cancelled(cancelled)?;
        let component = self
            .component(label)
            .ok_or_else(|| invalid("source artifact is undeclared"))?;
        let total = add(statistics.metadata_bytes, component.byte_len)?;
        if total > self.limits.max_metadata_bytes as u64 {
            return Err(resource("native metadata load exceeds its aggregate bound"));
        }
        let size = usize::try_from(component.byte_len)
            .map_err(|_| resource("metadata length does not fit this host"))?;
        let mut bytes = allocate(size)?;
        for (index, destination) in bytes.chunks_mut(self.limits.max_read_bytes).enumerate() {
            check_cancelled(cancelled)?;
            let offset = (index as u64)
                .checked_mul(self.limits.max_read_bytes as u64)
                .ok_or_else(|| resource("source read offset overflowed"))?;
            self.read_exact(component, offset, destination, statistics)?;
            statistics.metadata_bytes = add(statistics.metadata_bytes, destination.len() as u64)?;
        }
        check_cancelled(cancelled)?;
        Ok(bytes)
    }

    /// Copy exact declared bytes into private read-only, unlinked native backing.
    /// This is actual load/transfer I/O: no hash or static inspection is invoked.
    pub fn materialize(
        &self,
        label: &str,
        cancelled: &AtomicBool,
        statistics: &mut NativeLoadStatistics,
    ) -> Result<NativeArtifactBacking, NativeRuntimeError> {
        check_cancelled(cancelled)?;
        let component = self
            .component(label)
            .ok_or_else(|| invalid("source artifact is undeclared"))?;
        if add(statistics.backing_bytes, component.byte_len)? > self.limits.max_total_bytes {
            return Err(resource(
                "native backing transfer exceeds its aggregate bound",
            ));
        }
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
            let mut temporary = tempfile::NamedTempFile::new()
                .map_err(|_| infrastructure("cannot create private native backing"))?;
            let mut buffer = allocate(self.limits.max_read_bytes)?;
            let mut offset = 0_u64;
            while offset < component.byte_len {
                check_cancelled(cancelled)?;
                let size = (component.byte_len - offset).min(buffer.len() as u64) as usize;
                self.read_exact(component, offset, &mut buffer[..size], statistics)?;
                temporary
                    .write_all(&buffer[..size])
                    .map_err(|_| infrastructure("private native backing write failed"))?;
                offset += size as u64;
                statistics.backing_bytes = add(statistics.backing_bytes, size as u64)?;
            }
            temporary
                .flush()
                .map_err(|_| infrastructure("private backing flush failed"))?;
            check_cancelled(cancelled)?;
            let original = temporary
                .as_file()
                .metadata()
                .map_err(|_| infrastructure("cannot inspect private native backing"))?;
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(temporary.path())
                .map_err(|_| infrastructure("cannot seal private native backing"))?;
            let observed = file
                .metadata()
                .map_err(|_| infrastructure("cannot inspect read-only native backing"))?;
            if original.dev() != observed.dev()
                || original.ino() != observed.ino()
                || observed.len() != component.byte_len
                || !observed.is_file()
            {
                return Err(invalid(
                    "private native backing was replaced during publication",
                ));
            }
            // Consumes the sole writable File and removes its path. The read-only
            // descriptor is the only published capability; permissions alone are
            // deliberately not used as immutable authority.
            temporary
                .close()
                .map_err(|_| infrastructure("cannot unlink private native backing"))?;
            if file
                .metadata()
                .map_err(|_| infrastructure("cannot verify unlinked backing"))?
                .nlink()
                != 0
            {
                return Err(invalid("native backing still has a named writable route"));
            }
            check_cancelled(cancelled)?;
            Ok(NativeArtifactBacking {
                file,
                byte_len: component.byte_len,
                max_range_bytes: self.limits.max_metadata_bytes,
                max_read_bytes: self.limits.max_read_bytes,
            })
        }
        #[cfg(not(unix))]
        {
            Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "read-only unlinked native backing is unavailable on this platform",
            ))
        }
    }

    fn read_exact(
        &self,
        component: &NativeVerifiedComponent,
        offset: u64,
        destination: &mut [u8],
        statistics: &mut NativeLoadStatistics,
    ) -> Result<(), NativeRuntimeError> {
        let end = offset
            .checked_add(destination.len() as u64)
            .ok_or_else(|| resource("source read range overflowed"))?;
        if end > component.byte_len || destination.len() > self.limits.max_read_bytes {
            return Err(invalid(
                "source read exceeds its declaration or chunk bound",
            ));
        }
        statistics.source_read_calls = add(statistics.source_read_calls, 1)?;
        statistics.largest_read = statistics.largest_read.max(destination.len());
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.source.read_at(&component.label, offset, destination)
        }))
        .map_err(|_| infrastructure("verified source read panicked"))??;
        if actual > destination.len() {
            return Err(invalid("verified source reported an oversized read"));
        }
        statistics.source_bytes = add(statistics.source_bytes, actual as u64)?;
        if actual != destination.len() {
            return Err(invalid("verified source returned a short read"));
        }
        Ok(())
    }
}

/// Native-owned, pathless read-only backing. No write handle is retained.
///
/// The Unix borrowed descriptor is for native interop, not a Core authority or
/// serialized handle. Its borrower must preserve this lifetime or duplicate the
/// descriptor into an owning native reader before returning.
pub struct NativeArtifactBacking {
    file: File,
    byte_len: u64,
    max_range_bytes: usize,
    max_read_bytes: usize,
}

impl NativeArtifactBacking {
    /// Map only this constructor's sealed backing for a native family loader.
    /// The mapping owns its OS residence after this handle is dropped.
    #[cfg(unix)]
    pub(crate) fn map_read_only(&self) -> Result<memmap2::Mmap, NativeRuntimeError> {
        // SAFETY: materialize closed the only writable handle, checked inode/length,
        // and unlinked the name before publishing this read-only File. No writer
        // or writable mapping is retained or exposed by NativeArtifactBacking.
        unsafe { memmap2::MmapOptions::new().map(&self.file) }
            .map_err(|_| infrastructure("cannot map immutable native backing"))
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Bounded convenience read; native tensor I/O uses the owned native reader.
    pub fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, NativeRuntimeError> {
        let size = end
            .checked_sub(start)
            .ok_or_else(|| invalid("backing read range is reversed"))?;
        if end > self.byte_len || size > self.max_range_bytes as u64 {
            return Err(invalid("backing read exceeds its finite range bound"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let mut bytes = allocate(size as usize)?;
            for (index, chunk) in bytes.chunks_mut(self.max_read_bytes).enumerate() {
                self.file
                    .read_exact_at(chunk, start + index as u64 * self.max_read_bytes as u64)
                    .map_err(|_| infrastructure("read-only native backing read failed"))?;
            }
            Ok(bytes)
        }
        #[cfg(not(unix))]
        {
            Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "native backing range reads are unavailable on this platform",
            ))
        }
    }
}

#[cfg(unix)]
impl std::os::fd::AsFd for NativeArtifactBacking {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        std::os::fd::AsFd::as_fd(&self.file)
    }
}

fn allocate(size: usize) -> Result<Vec<u8>, NativeRuntimeError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| resource("native load buffer allocation failed"))?;
    bytes.resize(size, 0);
    Ok(bytes)
}
fn add(left: u64, right: u64) -> Result<u64, NativeRuntimeError> {
    left.checked_add(right)
        .ok_or_else(|| resource("native load accounting overflowed"))
}
fn check_cancelled(cancelled: &AtomicBool) -> Result<(), NativeRuntimeError> {
    if cancelled.load(Ordering::Acquire) {
        Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::Cancelled,
            "native load cancelled",
        ))
    } else {
        Ok(())
    }
}
fn invalid(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidPackage, message)
}
fn resource(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::ResourceLimit, message)
}
fn infrastructure(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::Infrastructure, message)
}
