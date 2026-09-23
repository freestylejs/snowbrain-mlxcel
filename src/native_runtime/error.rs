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

//! Errors exposed by the transport-neutral native runtime boundary.

use std::fmt;
use std::sync::Arc;

pub const MAX_NATIVE_ERROR_MESSAGE_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeErrorMessageTooLong {
    pub actual_bytes: usize,
    pub maximum_bytes: usize,
}

impl fmt::Display for NativeErrorMessageTooLong {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native error message has {} bytes; maximum is {}",
            self.actual_bytes, self.maximum_bytes
        )
    }
}

impl std::error::Error for NativeErrorMessageTooLong {}

/// Checked, immutable message payload used by public error variants.
///
/// The private field prevents constructing a variant with an unbounded
/// `Arc<str>`.
///
/// ```compile_fail
/// # use mlxcel::native_runtime::NativeBoundedErrorMessage;
/// # use std::sync::Arc;
/// let _ = NativeBoundedErrorMessage { message: Arc::from("unchecked") };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBoundedErrorMessage {
    message: Arc<str>,
}

impl NativeBoundedErrorMessage {
    pub fn try_new(message: impl AsRef<str>) -> Result<Self, NativeErrorMessageTooLong> {
        let message = message.as_ref();
        if message.len() > MAX_NATIVE_ERROR_MESSAGE_BYTES {
            return Err(NativeErrorMessageTooLong {
                actual_bytes: message.len(),
                maximum_bytes: MAX_NATIVE_ERROR_MESSAGE_BYTES,
            });
        }
        Ok(Self {
            message: Arc::from(message),
        })
    }

    fn from_static(message: &'static str) -> Self {
        debug_assert!(message.len() <= MAX_NATIVE_ERROR_MESSAGE_BYTES);
        Self {
            message: Arc::from(message),
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.message
    }
}

/// Stable error categories for native-runtime admission and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeRuntimeErrorKind {
    InvalidBuildIdentity,
    BuildIdentityMismatch,
    InvalidPackage,
    PackageChanged,
    InvalidRequest,
    InvalidOutput,
    UnsupportedOperation,
    UnsupportedCapability,
    UnsupportedGuarantee,
    ResourceLimit,
    StaleHandle,
    SinkRejected,
    Tokenizer,
    Infrastructure,
    /// A bounded native load was cancelled before publication.
    Cancelled,
}

/// An owned, bounded error that does not expose MLX, server, or frontend types.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeRuntimeError {
    InvalidBuildIdentity(NativeBoundedErrorMessage),
    BuildIdentityMismatch(NativeBoundedErrorMessage),
    InvalidPackage(NativeBoundedErrorMessage),
    PackageChanged(NativeBoundedErrorMessage),
    InvalidRequest(NativeBoundedErrorMessage),
    InvalidOutput(NativeBoundedErrorMessage),
    UnsupportedOperation(NativeBoundedErrorMessage),
    UnsupportedCapability(NativeBoundedErrorMessage),
    UnsupportedGuarantee(NativeBoundedErrorMessage),
    ResourceLimit(NativeBoundedErrorMessage),
    StaleHandle(NativeBoundedErrorMessage),
    SinkRejected(NativeBoundedErrorMessage),
    Tokenizer(NativeBoundedErrorMessage),
    Infrastructure(NativeBoundedErrorMessage),
    Cancelled(NativeBoundedErrorMessage),
}

impl NativeRuntimeError {
    pub fn try_new(
        kind: NativeRuntimeErrorKind,
        message: impl AsRef<str>,
    ) -> Result<Self, NativeErrorMessageTooLong> {
        Ok(Self::from_parts(
            kind,
            NativeBoundedErrorMessage::try_new(message)?,
        ))
    }

    pub(crate) fn new(kind: NativeRuntimeErrorKind, message: impl AsRef<str>) -> Self {
        Self::try_new(kind, message).unwrap_or_else(|_| {
            Self::from_parts(
                kind,
                NativeBoundedErrorMessage::from_static(
                    "native error detail exceeded the bounded message limit",
                ),
            )
        })
    }

    fn from_parts(kind: NativeRuntimeErrorKind, message: NativeBoundedErrorMessage) -> Self {
        match kind {
            NativeRuntimeErrorKind::InvalidBuildIdentity => Self::InvalidBuildIdentity(message),
            NativeRuntimeErrorKind::BuildIdentityMismatch => Self::BuildIdentityMismatch(message),
            NativeRuntimeErrorKind::InvalidPackage => Self::InvalidPackage(message),
            NativeRuntimeErrorKind::PackageChanged => Self::PackageChanged(message),
            NativeRuntimeErrorKind::InvalidRequest => Self::InvalidRequest(message),
            NativeRuntimeErrorKind::InvalidOutput => Self::InvalidOutput(message),
            NativeRuntimeErrorKind::UnsupportedOperation => Self::UnsupportedOperation(message),
            NativeRuntimeErrorKind::UnsupportedCapability => Self::UnsupportedCapability(message),
            NativeRuntimeErrorKind::UnsupportedGuarantee => Self::UnsupportedGuarantee(message),
            NativeRuntimeErrorKind::ResourceLimit => Self::ResourceLimit(message),
            NativeRuntimeErrorKind::StaleHandle => Self::StaleHandle(message),
            NativeRuntimeErrorKind::SinkRejected => Self::SinkRejected(message),
            NativeRuntimeErrorKind::Tokenizer => Self::Tokenizer(message),
            NativeRuntimeErrorKind::Infrastructure => Self::Infrastructure(message),
            NativeRuntimeErrorKind::Cancelled => Self::Cancelled(message),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> NativeRuntimeErrorKind {
        match self {
            Self::InvalidBuildIdentity(_) => NativeRuntimeErrorKind::InvalidBuildIdentity,
            Self::BuildIdentityMismatch(_) => NativeRuntimeErrorKind::BuildIdentityMismatch,
            Self::InvalidPackage(_) => NativeRuntimeErrorKind::InvalidPackage,
            Self::PackageChanged(_) => NativeRuntimeErrorKind::PackageChanged,
            Self::InvalidRequest(_) => NativeRuntimeErrorKind::InvalidRequest,
            Self::InvalidOutput(_) => NativeRuntimeErrorKind::InvalidOutput,
            Self::UnsupportedOperation(_) => NativeRuntimeErrorKind::UnsupportedOperation,
            Self::UnsupportedCapability(_) => NativeRuntimeErrorKind::UnsupportedCapability,
            Self::UnsupportedGuarantee(_) => NativeRuntimeErrorKind::UnsupportedGuarantee,
            Self::ResourceLimit(_) => NativeRuntimeErrorKind::ResourceLimit,
            Self::StaleHandle(_) => NativeRuntimeErrorKind::StaleHandle,
            Self::SinkRejected(_) => NativeRuntimeErrorKind::SinkRejected,
            Self::Tokenizer(_) => NativeRuntimeErrorKind::Tokenizer,
            Self::Infrastructure(_) => NativeRuntimeErrorKind::Infrastructure,
            Self::Cancelled(_) => NativeRuntimeErrorKind::Cancelled,
        }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidBuildIdentity(message)
            | Self::BuildIdentityMismatch(message)
            | Self::InvalidPackage(message)
            | Self::PackageChanged(message)
            | Self::InvalidRequest(message)
            | Self::InvalidOutput(message)
            | Self::UnsupportedOperation(message)
            | Self::UnsupportedCapability(message)
            | Self::UnsupportedGuarantee(message)
            | Self::ResourceLimit(message)
            | Self::StaleHandle(message)
            | Self::SinkRejected(message)
            | Self::Tokenizer(message)
            | Self::Infrastructure(message)
            | Self::Cancelled(message) => message.as_str(),
        }
    }
}

impl fmt::Display for NativeRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for NativeRuntimeError {}

/// Stable tokenizer failure categories for incremental decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TokenizerErrorKind {
    InvalidTokenId,
    DecodeFailed,
    UnsupportedGuarantee,
    LookbehindExceeded,
    RetroactiveRewrite,
    IncompleteTerminalBytes,
    InvalidTerminalBytes,
    AlreadyFlushed,
}

/// A fallible decoder error. No failure is converted to empty output or U+FFFD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerError {
    kind: TokenizerErrorKind,
    message: Arc<str>,
}

impl TokenizerError {
    pub fn try_new(
        kind: TokenizerErrorKind,
        message: impl AsRef<str>,
    ) -> Result<Self, NativeErrorMessageTooLong> {
        let message = message.as_ref();
        if message.len() > MAX_NATIVE_ERROR_MESSAGE_BYTES {
            return Err(NativeErrorMessageTooLong {
                actual_bytes: message.len(),
                maximum_bytes: MAX_NATIVE_ERROR_MESSAGE_BYTES,
            });
        }
        Ok(Self {
            kind,
            message: Arc::from(message),
        })
    }

    pub(crate) fn new(kind: TokenizerErrorKind, message: impl AsRef<str>) -> Self {
        Self::try_new(kind, message).unwrap_or_else(|error| Self {
            kind,
            message: Arc::from(format!(
                "tokenizer error detail rejected because it exceeded the bounded message limit ({error})"
            )),
        })
    }

    #[must_use]
    pub const fn kind(&self) -> TokenizerErrorKind {
        self.kind
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_public_error_messages_are_rejected_without_truncation() {
        let maximum = "x".repeat(MAX_NATIVE_ERROR_MESSAGE_BYTES);
        assert_eq!(
            NativeRuntimeError::try_new(NativeRuntimeErrorKind::SinkRejected, &maximum)
                .unwrap()
                .message()
                .len(),
            MAX_NATIVE_ERROR_MESSAGE_BYTES
        );
        let oversized = "x".repeat(MAX_NATIVE_ERROR_MESSAGE_BYTES + 1);
        let runtime =
            NativeRuntimeError::try_new(NativeRuntimeErrorKind::InvalidRequest, oversized.as_str())
                .unwrap_err();
        assert_eq!(runtime.actual_bytes, oversized.len());
        assert_eq!(runtime.maximum_bytes, MAX_NATIVE_ERROR_MESSAGE_BYTES);

        let tokenizer =
            TokenizerError::try_new(TokenizerErrorKind::DecodeFailed, oversized.as_str())
                .unwrap_err();
        assert_eq!(tokenizer.actual_bytes, MAX_NATIVE_ERROR_MESSAGE_BYTES + 1);
    }
}
