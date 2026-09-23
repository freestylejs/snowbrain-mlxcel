//! Value-only binding of a trusted producer's immutable source to one native build.
//!
//! Producer identities stay in their own domains. These values do not authenticate
//! bytes, inspect a model, or grant request/start permission. The live source must
//! retain the actual verified capability independently of these immutable facts.

use super::{
    MlxNativeOperationClass, NativeModalitySet, NativeRuntimeError, NativeRuntimeErrorKind,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

const MAX_BINDING_TEXT: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTokenizerLoadBinding {
    digest: [u8; 32],
    implementation: Arc<str>,
}
impl NativeTokenizerLoadBinding {
    pub fn new(digest: [u8; 32], implementation: &str) -> Result<Self, NativeRuntimeError> {
        if digest == [0; 32] || !valid_text(implementation) {
            return Err(invalid(
                "native tokenizer binding is empty or exceeds its bound",
            ));
        }
        Ok(Self {
            digest,
            implementation: Arc::from(implementation),
        })
    }
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn implementation(&self) -> &str {
        &self.implementation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeWorkerLoadBinding {
    identity: [u8; 32],
    source_identity: [u8; 32],
    report_identity: [u8; 32],
    native_build_identity: [u8; 32],
    architecture: Arc<str>,
    architecture_supported: bool,
    max_context_positions: Option<u32>,
    modalities: NativeModalitySet,
    tokenizer: NativeTokenizerLoadBinding,
}
impl NativeWorkerLoadBinding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_identity: [u8; 32],
        report_identity: [u8; 32],
        native_build_identity: [u8; 32],
        architecture: &str,
        architecture_supported: bool,
        max_context_positions: Option<u32>,
        modalities: NativeModalitySet,
        tokenizer: NativeTokenizerLoadBinding,
    ) -> Result<Self, NativeRuntimeError> {
        if source_identity == [0; 32]
            || report_identity == [0; 32]
            || native_build_identity == [0; 32]
            || !valid_text(architecture)
            || max_context_positions == Some(0)
        {
            return Err(invalid(
                "native worker load binding is incomplete or exceeds its bounds",
            ));
        }
        let mut encoder = Sha256::new();
        encoder.update(b"mlxcel.verified-load-binding.v1\0");
        for digest in [
            source_identity,
            report_identity,
            native_build_identity,
            tokenizer.digest(),
        ] {
            encoder.update(digest);
        }
        for text in [architecture, tokenizer.implementation()] {
            encoder.update((text.len() as u64).to_le_bytes());
            encoder.update(text.as_bytes());
        }
        encoder.update([
            u8::from(architecture_supported),
            u8::from(modalities.text),
            u8::from(modalities.image),
            u8::from(modalities.audio),
            u8::from(modalities.video),
            u8::from(max_context_positions.is_some()),
        ]);
        encoder.update(max_context_positions.unwrap_or(0).to_le_bytes());
        Ok(Self {
            identity: encoder.finalize().into(),
            source_identity,
            report_identity,
            native_build_identity,
            architecture: Arc::from(architecture),
            architecture_supported,
            max_context_positions,
            modalities,
            tokenizer,
        })
    }
    /// Binding of bounded metadata values, never an artifact-byte proof.
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub const fn source_identity(&self) -> [u8; 32] {
        self.source_identity
    }
    pub const fn report_identity(&self) -> [u8; 32] {
        self.report_identity
    }
    pub const fn native_build_identity(&self) -> [u8; 32] {
        self.native_build_identity
    }
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
    pub const fn architecture_supported(&self) -> bool {
        self.architecture_supported
    }
    pub const fn max_context_positions(&self) -> Option<u32> {
        self.max_context_positions
    }
    pub const fn modalities(&self) -> NativeModalitySet {
        self.modalities
    }
    pub const fn tokenizer(&self) -> &NativeTokenizerLoadBinding {
        &self.tokenizer
    }

    /// Current managed-inspection families, not the broader server model catalogue.
    pub(crate) fn operation(&self) -> Result<MlxNativeOperationClass, NativeRuntimeError> {
        if !self.architecture_supported
            || !self.modalities.text
            || self.max_context_positions.is_none()
        {
            return Err(unsupported(
                "source has no inspected managed model load binding",
            ));
        }
        match self.architecture.as_ref() {
            "bert" | "xlm-roberta" => Ok(MlxNativeOperationClass::Embedding),
            "gemma4_unified" => Ok(MlxNativeOperationClass::Autoregressive),
            _ => Err(unsupported(
                "source architecture has no qualified managed loader",
            )),
        }
    }
}

fn valid_text(text: &str) -> bool {
    !text.is_empty() && text.len() <= MAX_BINDING_TEXT && !text.chars().any(char::is_control)
}
fn invalid(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidPackage, message)
}
fn unsupported(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::UnsupportedOperation, message)
}
