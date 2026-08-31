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

//! Owned, bounded request and ownership-transfer contracts.

use std::sync::Arc;

use mlxcel_core::session::{
    PreparedAdapterMode, PreparedPositions, PreparedPrefill, PreparedTensorDType,
};

use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::operation_driver::NativeOperationDriver;
use super::sink::MAX_NATIVE_EMBEDDING_DIMENSION;

pub const MAX_NATIVE_PROMPT_TOKENS: usize = 1_048_576;
pub const MAX_NATIVE_STOP_TOKEN_IDS: usize = 4_096;
pub const MAX_NATIVE_EMBED_INPUTS: usize = 1_024;
pub const MAX_NATIVE_EMBED_AGGREGATE_TOKENS: usize = 4 * MAX_NATIVE_PROMPT_TOKENS;
pub const MAX_NATIVE_EMBED_AGGREGATE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_NATIVE_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_PREPARED_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_NATIVE_PREPARED_MODALITIES: usize = 1_024;
pub const MAX_NATIVE_PREPARED_MEDIA_ITEMS: usize = 1_024;
pub const MAX_NATIVE_TENSOR_RANK: usize = 16;
pub const MAX_NATIVE_CONSTRAINT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_LABEL_BYTES: usize = 4_096;

fn invalid_request(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidRequest, message)
}

/// Model-specific prepared state represented without an MLX array or model enum.
#[derive(Clone)]
pub struct NativeOwnedPreparedInput {
    schema: Arc<str>,
    payload: Arc<[u8]>,
    logical_token_len: usize,
}

impl NativeOwnedPreparedInput {
    pub fn new(
        schema: impl AsRef<str>,
        payload: impl AsRef<[u8]>,
        logical_token_len: usize,
    ) -> Result<Self, NativeRuntimeError> {
        let schema = schema.as_ref();
        let payload = payload.as_ref();
        validate_owned_prepared_lengths(schema.len(), payload.len(), logical_token_len)?;
        Ok(Self {
            schema: Arc::from(schema),
            payload: Arc::from(payload),
            logical_token_len,
        })
    }

    #[must_use]
    pub fn schema(&self) -> &str {
        &self.schema
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    #[must_use]
    pub const fn logical_token_len(&self) -> usize {
        self.logical_token_len
    }
}

fn validate_owned_prepared_lengths(
    schema_bytes: usize,
    payload_bytes: usize,
    logical_token_len: usize,
) -> Result<(), NativeRuntimeError> {
    if schema_bytes == 0 || schema_bytes > MAX_NATIVE_LABEL_BYTES {
        return Err(invalid_request(
            "prepared-input schema is empty or too large",
        ));
    }
    if payload_bytes > MAX_NATIVE_PREPARED_BYTES {
        return Err(invalid_request(
            "prepared-input payload exceeds the native bound",
        ));
    }
    if logical_token_len == 0 || logical_token_len > MAX_NATIVE_PROMPT_TOKENS {
        return Err(invalid_request(
            "prepared-input logical token count is zero or exceeds the native bound",
        ));
    }
    Ok(())
}

/// Opaque validated prepared-prefill capability. The mutable public fields of
/// the upstream transport type cannot be retained directly across admission.
#[derive(Clone)]
pub struct NativePreparedPrefill(Arc<PreparedPrefill>);

impl NativePreparedPrefill {
    pub fn try_from_owned(prepared: PreparedPrefill) -> Result<Self, NativeRuntimeError> {
        validate_prepared_prefill(&prepared)?;
        Ok(Self(Arc::new(prepared)))
    }

    #[must_use]
    pub fn sequence_len(&self) -> usize {
        self.0.sequence_len
    }

    #[must_use]
    pub fn token_ids(&self) -> &[i32] {
        &self.0.token_ids
    }

    #[must_use]
    pub fn modality_count(&self) -> usize {
        self.0.modalities.len()
    }

    pub(crate) fn as_prepared_prefill(&self) -> &PreparedPrefill {
        &self.0
    }
}

/// Prompt input ownership accepted by a native generation admission.
pub enum NativePromptInput {
    TokenIds(Arc<[i32]>),
    Prepared(NativePreparedPrefill),
    ModelSpecific(NativeOwnedPreparedInput),
}

impl NativePromptInput {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        match self {
            Self::TokenIds(ids) => {
                if ids.is_empty() || ids.len() > MAX_NATIVE_PROMPT_TOKENS {
                    return Err(invalid_request(
                        "prompt token count is zero or exceeds the native bound",
                    ));
                }
                validate_nonnegative_token_ids("prompt", ids)?;
            }
            Self::Prepared(_) => {}
            Self::ModelSpecific(input) => {
                if input.payload.len() > MAX_NATIVE_PREPARED_BYTES {
                    return Err(invalid_request(
                        "prepared-input payload exceeds the native bound",
                    ));
                }
                if input.logical_token_len == 0
                    || input.logical_token_len > MAX_NATIVE_PROMPT_TOKENS
                {
                    return Err(invalid_request(
                        "prepared-input logical token count is zero or exceeds the native bound",
                    ));
                }
            }
        }
        Ok(())
    }

    fn logical_token_len(&self) -> usize {
        match self {
            Self::TokenIds(ids) => ids.len(),
            Self::Prepared(prepared) => prepared.sequence_len(),
            Self::ModelSpecific(prepared) => prepared.logical_token_len(),
        }
    }

    fn shallow_clone(&self) -> Self {
        match self {
            Self::TokenIds(value) => Self::TokenIds(Arc::clone(value)),
            Self::Prepared(value) => Self::Prepared(value.clone()),
            Self::ModelSpecific(value) => Self::ModelSpecific(value.clone()),
        }
    }
}

fn validate_prepared_prefill(prepared: &PreparedPrefill) -> Result<(), NativeRuntimeError> {
    if prepared.token_ids.is_empty() || prepared.token_ids.len() > MAX_NATIVE_PROMPT_TOKENS {
        return Err(invalid_request(
            "prepared-prefill token count is zero or exceeds the native bound",
        ));
    }
    if prepared.sequence_len != prepared.token_ids.len()
        || prepared.modalities.len() > MAX_NATIVE_PREPARED_MODALITIES
    {
        return Err(invalid_request(
            "prepared-prefill structure violates a native count bound",
        ));
    }
    validate_nonnegative_token_ids("prepared prompt", &prepared.token_ids)?;
    validate_owned_tensor(
        "prepared embeddings",
        &prepared.embeddings.bytes,
        prepared.embeddings.dtype,
        &prepared.embeddings.shape,
    )?;
    if prepared.embeddings.shape.len() != 3
        || prepared.embeddings.shape[0] != 1
        || prepared.embeddings.shape[1] != prepared.sequence_len
        || prepared.embeddings.shape[2] == 0
        || !prepared.embeddings.dtype.is_float()
    {
        return Err(invalid_request(
            "prepared embedding shape/dtype is inconsistent with the sequence",
        ));
    }
    validate_owned_tensor(
        "prepared attention bias",
        &prepared.attention_bias.tensor.bytes,
        prepared.attention_bias.tensor.dtype,
        &prepared.attention_bias.tensor.shape,
    )?;
    if !prepared.attention_bias.tensor.dtype.is_float()
        || prepared.attention_bias.tensor.shape.last().copied() != Some(prepared.sequence_len)
    {
        return Err(invalid_request(
            "prepared attention bias is inconsistent with the sequence",
        ));
    }
    match &prepared.positions {
        PreparedPositions::Sequential { length, .. } if *length != prepared.sequence_len => {
            return Err(invalid_request(
                "prepared sequential positions disagree with the sequence",
            ));
        }
        PreparedPositions::Explicit(tensor) => {
            validate_owned_tensor(
                "prepared positions",
                &tensor.bytes,
                tensor.dtype,
                &tensor.shape,
            )?;
            if tensor.dtype != PreparedTensorDType::Int32
                || (tensor.shape.as_slice() != [prepared.sequence_len]
                    && tensor.shape.as_slice() != [1, prepared.sequence_len])
            {
                return Err(invalid_request(
                    "prepared explicit positions have an invalid shape/dtype",
                ));
            }
        }
        PreparedPositions::Mrope3D { tensor, .. } => {
            validate_owned_tensor(
                "prepared M-RoPE positions",
                &tensor.bytes,
                tensor.dtype,
                &tensor.shape,
            )?;
            if tensor.dtype != PreparedTensorDType::Int32
                || tensor.shape.as_slice() != [3, prepared.sequence_len]
            {
                return Err(invalid_request(
                    "prepared M-RoPE positions have an invalid shape/dtype",
                ));
            }
        }
        PreparedPositions::Sequential { .. } => {}
        _ => {
            return Err(invalid_request(
                "prepared positions use an unsupported native representation",
            ));
        }
    }
    let mut total = prepared.embeddings.bytes.len();
    total = total
        .checked_add(prepared.attention_bias.tensor.bytes.len())
        .and_then(|value| {
            value.checked_add(match &prepared.positions {
                mlxcel_core::session::PreparedPositions::Sequential { .. } => 0,
                mlxcel_core::session::PreparedPositions::Explicit(tensor)
                | mlxcel_core::session::PreparedPositions::Mrope3D { tensor, .. } => {
                    tensor.bytes.len()
                }
                _ => 0,
            })
        })
        .and_then(|value| value.checked_add(prepared.token_ids.len() * std::mem::size_of::<i32>()))
        .ok_or_else(|| invalid_request("prepared-prefill byte count overflowed"))?;
    let mut modality_tokens = 0usize;
    let mut modality_items = 0usize;
    let metadata_bytes = prepared
        .modalities
        .iter()
        .try_fold(0usize, |total, modality| {
            if modality.family.is_empty()
                || modality.family.len() > MAX_NATIVE_LABEL_BYTES
                || modality.item_count == 0
                || modality.item_count > MAX_NATIVE_PREPARED_MEDIA_ITEMS
                || modality.token_count == 0
            {
                return Err(invalid_request(
                    "prepared-prefill modality metadata is empty or inconsistent",
                ));
            }
            modality_tokens = modality_tokens
                .checked_add(modality.token_count)
                .ok_or_else(|| {
                    invalid_request("prepared-prefill modality token count overflowed")
                })?;
            modality_items = modality_items
                .checked_add(modality.item_count)
                .ok_or_else(|| invalid_request("prepared-prefill media item count overflowed"))?;
            if modality_items > MAX_NATIVE_PREPARED_MEDIA_ITEMS {
                return Err(invalid_request(
                    "prepared-prefill media items exceed the aggregate native bound",
                ));
            }
            total
                .checked_add(modality.family.len())
                .and_then(|value| value.checked_add(2 * std::mem::size_of::<usize>()))
                .ok_or_else(|| invalid_request("prepared-prefill metadata byte count overflowed"))
        })?;
    if modality_tokens > prepared.sequence_len
        || (prepared.adapter_mode != PreparedAdapterMode::Language
            && prepared.modalities.is_empty())
    {
        return Err(invalid_request(
            "prepared-prefill modality/adapter facts disagree with the sequence",
        ));
    }
    total = total
        .checked_add(metadata_bytes)
        .ok_or_else(|| invalid_request("prepared-prefill byte count overflowed"))?;
    if total > MAX_NATIVE_PREPARED_BYTES {
        return Err(invalid_request(
            "prepared-prefill payload exceeds the native bound",
        ));
    }
    Ok(())
}

fn validate_owned_tensor(
    name: &str,
    bytes: &[u8],
    dtype: PreparedTensorDType,
    shape: &[usize],
) -> Result<(), NativeRuntimeError> {
    if shape.is_empty() || shape.len() > MAX_NATIVE_TENSOR_RANK || shape.contains(&0) {
        return Err(invalid_request(format!(
            "{name} rank or dimension violates the native bound"
        )));
    }
    let elements = shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| invalid_request(format!("{name} shape product overflowed")))
    })?;
    let expected = elements
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| invalid_request(format!("{name} byte count overflowed")))?;
    if bytes.len() != expected || bytes.len() > MAX_NATIVE_PREPARED_BYTES {
        return Err(invalid_request(format!(
            "{name} storage does not match shape times dtype size"
        )));
    }
    Ok(())
}

fn validate_nonnegative_token_ids(name: &str, ids: &[i32]) -> Result<(), NativeRuntimeError> {
    if ids.iter().any(|id| *id < 0) {
        return Err(invalid_request(format!(
            "{name} contains a negative token id"
        )));
    }
    Ok(())
}

/// Sampling values whose numerical interpretation remains owned by mlxcel.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeSamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl NativeSamplingConfig {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid_request(
                "temperature must be finite and non-negative",
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(invalid_request("top_p must be finite and in [0, 1]"));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(invalid_request("min_p must be finite and in [0, 1]"));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(invalid_request(
                "repetition_penalty must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Hard generation bounds enforced during admission and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeGenerationLimits {
    pub max_new_tokens: u32,
    pub max_total_tokens: u32,
}

/// Neutral scheduling hints; they do not contain HTTP or frontend state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeSchedulingHints {
    pub priority: i16,
    pub deadline_after_ms: Option<u64>,
}

/// Exact cache namespace and immutable input identity supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCacheContext {
    pub namespace: Arc<str>,
    pub input_identity: [u8; 32],
    pub allow_read: bool,
    pub allow_write: bool,
}

impl NativeCacheContext {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        if self.namespace.is_empty() || self.namespace.len() > MAX_NATIVE_LABEL_BYTES {
            return Err(invalid_request("cache namespace is empty or too large"));
        }
        Ok(())
    }
}

/// Opaque physical constraint bytes. Their tool/reasoning meaning stays outside mlxcel.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeConstraintPlan {
    None,
    Fixed {
        format: Arc<str>,
        payload: Arc<[u8]>,
    },
    Dynamic {
        initial_format: Arc<str>,
        initial_payload: Arc<[u8]>,
    },
    ValidateOnly,
}

impl NativeConstraintPlan {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        let (format, payload) = match self {
            Self::None | Self::ValidateOnly => return Ok(()),
            Self::Fixed { format, payload } => (format, payload),
            Self::Dynamic {
                initial_format,
                initial_payload,
            } => (initial_format, initial_payload),
        };
        if format.is_empty() || format.len() > MAX_NATIVE_LABEL_BYTES {
            return Err(invalid_request("constraint format is empty or too large"));
        }
        if payload.len() > MAX_NATIVE_CONSTRAINT_BYTES {
            return Err(invalid_request(
                "constraint payload exceeds the native bound",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeSpeculativeFidelity {
    ExactTargetDistribution,
    GreedyEquivalent,
    Approximate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeSpeculationPreference {
    Disabled,
    Auto,
    Required,
}

/// Request-shaped speculation policy; unsupported optimization may fall back only when allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSpeculationPolicy {
    pub preference: NativeSpeculationPreference,
    pub minimum_fidelity: NativeSpeculativeFidelity,
    pub allow_ordinary_fallback: bool,
}

/// Complete generation request. Validation is mandatory before reservation creation.
pub struct NativeGenerateRequest {
    pub request_id: u128,
    pub prompt: NativePromptInput,
    pub sampling: NativeSamplingConfig,
    pub limits: NativeGenerationLimits,
    pub eos_token_ids: Arc<[i32]>,
    pub output_suppressed_token_ids: Arc<[i32]>,
    pub scheduling: NativeSchedulingHints,
    pub cache: NativeCacheContext,
    pub constraint: NativeConstraintPlan,
    pub speculation: NativeSpeculationPolicy,
}

impl NativeGenerateRequest {
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        self.prompt.validate()?;
        self.sampling.validate()?;
        self.cache.validate()?;
        self.constraint.validate()?;
        if self.limits.max_new_tokens == 0
            || self.limits.max_total_tokens < self.limits.max_new_tokens
        {
            return Err(invalid_request(
                "generation limits require non-zero new tokens within total tokens",
            ));
        }
        if self.eos_token_ids.len() > MAX_NATIVE_STOP_TOKEN_IDS
            || self.output_suppressed_token_ids.len() > MAX_NATIVE_STOP_TOKEN_IDS
        {
            return Err(invalid_request(
                "token policy list exceeds the native bound",
            ));
        }
        validate_nonnegative_token_ids("EOS policy", &self.eos_token_ids)?;
        validate_nonnegative_token_ids(
            "output-suppression policy",
            &self.output_suppressed_token_ids,
        )?;
        let prompt_tokens = u64::try_from(self.prompt.logical_token_len())
            .map_err(|_| invalid_request("prompt token count does not fit generation limits"))?;
        let required_total = prompt_tokens
            .checked_add(u64::from(self.limits.max_new_tokens))
            .ok_or_else(|| invalid_request("prompt plus generation token count overflowed"))?;
        if required_total > u64::from(self.limits.max_total_tokens) {
            return Err(invalid_request(
                "prompt tokens plus max_new_tokens exceed max_total_tokens",
            ));
        }
        Ok(())
    }

    pub(crate) fn validated(&self) -> Result<ValidatedGenerateRequest, NativeRuntimeError> {
        self.validate()?;
        Ok(ValidatedGenerateRequest(Arc::new(self.shallow_clone())))
    }

    pub(crate) fn logical_prompt_tokens(&self) -> usize {
        self.prompt.logical_token_len()
    }

    fn shallow_clone(&self) -> Self {
        Self {
            request_id: self.request_id,
            prompt: self.prompt.shallow_clone(),
            sampling: self.sampling.clone(),
            limits: self.limits,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
            output_suppressed_token_ids: Arc::clone(&self.output_suppressed_token_ids),
            scheduling: self.scheduling,
            cache: self.cache.clone(),
            constraint: self.constraint.clone(),
            speculation: self.speculation,
        }
    }
}

/// One embedding input, retaining canonical input order in the enclosing vector.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeEmbeddingInput {
    TokenIds(Arc<[i32]>),
    Text(Arc<str>),
}

impl NativeEmbeddingInput {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        match self {
            Self::TokenIds(ids) if ids.is_empty() || ids.len() > MAX_NATIVE_PROMPT_TOKENS => Err(
                invalid_request("embedding token count is zero or exceeds the native bound"),
            ),
            Self::TokenIds(ids) => validate_nonnegative_token_ids("embedding input", ids),
            Self::Text(text) if text.is_empty() || text.len() > MAX_NATIVE_TEXT_BYTES => Err(
                invalid_request("embedding text is empty or exceeds the native bound"),
            ),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeEmbeddingLimits {
    pub max_input_tokens: u32,
    pub max_output_dimensions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeEmbeddingNormalization {
    ModelDefault,
    None,
    L2,
}

pub struct NativeEmbedRequest {
    pub request_id: u128,
    pub inputs: Arc<[NativeEmbeddingInput]>,
    pub limits: NativeEmbeddingLimits,
    pub normalization: NativeEmbeddingNormalization,
}

impl NativeEmbedRequest {
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        if self.inputs.is_empty() || self.inputs.len() > MAX_NATIVE_EMBED_INPUTS {
            return Err(invalid_request(
                "embedding input count is zero or exceeds the native bound",
            ));
        }
        if self.limits.max_input_tokens == 0
            || self.limits.max_input_tokens as usize > MAX_NATIVE_PROMPT_TOKENS
            || self.limits.max_output_dimensions == 0
        {
            return Err(invalid_request(
                "embedding limits must be non-zero and within global ceilings",
            ));
        }
        if usize::try_from(self.limits.max_output_dimensions)
            .map_or(true, |dimension| dimension > MAX_NATIVE_EMBEDDING_DIMENSION)
        {
            return Err(invalid_request(
                "embedding output dimension exceeds the native bound",
            ));
        }
        let mut aggregate_tokens = 0usize;
        let mut aggregate_bytes = 0usize;
        for input in self.inputs.iter() {
            input.validate()?;
            match input {
                NativeEmbeddingInput::TokenIds(ids) => {
                    if ids.len() > self.limits.max_input_tokens as usize {
                        return Err(invalid_request(
                            "embedding token input exceeds max_input_tokens",
                        ));
                    }
                    aggregate_tokens = aggregate_tokens
                        .checked_add(ids.len())
                        .ok_or_else(|| invalid_request("embedding token total overflowed"))?;
                    aggregate_bytes = aggregate_bytes
                        .checked_add(ids.len() * std::mem::size_of::<i32>())
                        .ok_or_else(|| invalid_request("embedding byte total overflowed"))?;
                }
                NativeEmbeddingInput::Text(text) => {
                    aggregate_bytes = aggregate_bytes
                        .checked_add(text.len())
                        .ok_or_else(|| invalid_request("embedding byte total overflowed"))?;
                }
            }
            if aggregate_tokens > MAX_NATIVE_EMBED_AGGREGATE_TOKENS
                || aggregate_bytes > MAX_NATIVE_EMBED_AGGREGATE_BYTES
            {
                return Err(invalid_request(
                    "embedding inputs exceed their aggregate native bound",
                ));
            }
        }
        Ok(())
    }

    /// Validate exact post-tokenization lengths before reservation. The entry
    /// for a token-id input must equal its already-declared length; the entry
    /// for a text input is the frozen tokenizer's actual output length.
    pub fn validate_resolved_input_tokens(
        &self,
        resolved_input_tokens: &[usize],
    ) -> Result<(), NativeRuntimeError> {
        self.validate()?;
        if resolved_input_tokens.len() != self.inputs.len() {
            return Err(invalid_request(
                "resolved embedding token counts do not match the input count",
            ));
        }
        let mut aggregate = 0usize;
        for (input, &tokens) in self.inputs.iter().zip(resolved_input_tokens) {
            if tokens == 0 {
                return Err(invalid_request("resolved embedding input has zero tokens"));
            }
            if let NativeEmbeddingInput::TokenIds(ids) = input
                && tokens != ids.len()
            {
                return Err(invalid_request(
                    "resolved token count disagrees with a token-id input",
                ));
            }
            if tokens > MAX_NATIVE_PROMPT_TOKENS || tokens > self.limits.max_input_tokens as usize {
                return Err(invalid_request(
                    "resolved embedding input exceeds max_input_tokens",
                ));
            }
            aggregate = aggregate
                .checked_add(tokens)
                .ok_or_else(|| invalid_request("resolved embedding token total overflowed"))?;
            if aggregate > MAX_NATIVE_EMBED_AGGREGATE_TOKENS {
                return Err(invalid_request(
                    "resolved embedding inputs exceed the aggregate token bound",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validated_with_input_tokens(
        &self,
        resolved_input_tokens: &[usize],
    ) -> Result<ValidatedEmbedRequest, NativeRuntimeError> {
        self.validate_resolved_input_tokens(resolved_input_tokens)?;
        Ok(ValidatedEmbedRequest(Arc::new(self.shallow_clone())))
    }

    fn shallow_clone(&self) -> Self {
        Self {
            request_id: self.request_id,
            inputs: Arc::clone(&self.inputs),
            limits: self.limits,
            normalization: self.normalization,
        }
    }
}

pub(crate) struct ValidatedGenerateRequest(Arc<NativeGenerateRequest>);

pub(crate) struct ValidatedEmbedRequest(Arc<NativeEmbedRequest>);

/// Non-cloneable reservation capability produced only by a native engine.
pub struct NativeGenerateAdmission {
    reservation_id: u128,
    driver: NativeOperationDriver,
    request: Arc<NativeGenerateRequest>,
    guard: NativeReservationGuard,
}

pub(crate) trait NativeReservationRelease: Send + Sync {
    fn release(&self, reservation_id: u128);
}

pub(crate) struct NativeReservationGuard {
    reservation_id: u128,
    release: Option<Arc<dyn NativeReservationRelease>>,
}

impl NativeReservationGuard {
    /// Disarm only after `sink.started(...)` succeeds. Before that handshake,
    /// every error path drops this guard and releases the reservation.
    pub(crate) fn commit_started(mut self) {
        self.release = None;
    }
}

impl Drop for NativeReservationGuard {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release.release(self.reservation_id);
        }
    }
}

impl NativeGenerateAdmission {
    pub(crate) fn new(
        reservation_id: u128,
        driver: NativeOperationDriver,
        request: ValidatedGenerateRequest,
        release: Arc<dyn NativeReservationRelease>,
    ) -> Self {
        Self {
            reservation_id,
            driver,
            request: request.0,
            guard: NativeReservationGuard {
                reservation_id,
                release: Some(release),
            },
        }
    }

    #[must_use]
    pub const fn reservation_id(&self) -> u128 {
        self.reservation_id
    }

    #[must_use]
    pub const fn driver(&self) -> NativeOperationDriver {
        self.driver
    }

    #[must_use]
    pub fn request(&self) -> &NativeGenerateRequest {
        &self.request
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        u128,
        NativeOperationDriver,
        Arc<NativeGenerateRequest>,
        NativeReservationGuard,
    ) {
        (self.reservation_id, self.driver, self.request, self.guard)
    }
}

/// Non-cloneable embedding reservation capability.
pub struct NativeEmbedAdmission {
    reservation_id: u128,
    request: Arc<NativeEmbedRequest>,
    guard: NativeReservationGuard,
}

impl NativeEmbedAdmission {
    pub(crate) fn new(
        reservation_id: u128,
        request: ValidatedEmbedRequest,
        release: Arc<dyn NativeReservationRelease>,
    ) -> Self {
        Self {
            reservation_id,
            request: request.0,
            guard: NativeReservationGuard {
                reservation_id,
                release: Some(release),
            },
        }
    }

    #[must_use]
    pub const fn reservation_id(&self) -> u128 {
        self.reservation_id
    }

    #[must_use]
    pub fn request(&self) -> &NativeEmbedRequest {
        &self.request
    }

    pub(crate) fn into_parts(self) -> (u128, Arc<NativeEmbedRequest>, NativeReservationGuard) {
        (self.reservation_id, self.request, self.guard)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeExecutionKind {
    Generate,
    Embed,
}

/// Non-cloneable identity returned only after the native started handshake.
pub struct NativeExecutionHandle {
    worker_epoch: u64,
    execution_id: u128,
    kind: NativeExecutionKind,
}

impl NativeExecutionHandle {
    pub(crate) const fn new(
        worker_epoch: u64,
        execution_id: u128,
        kind: NativeExecutionKind,
    ) -> Self {
        Self {
            worker_epoch,
            execution_id,
            kind,
        }
    }

    #[must_use]
    pub const fn worker_epoch(&self) -> u64 {
        self.worker_epoch
    }

    #[must_use]
    pub const fn execution_id(&self) -> u128 {
        self.execution_id
    }

    #[must_use]
    pub const fn kind(&self) -> NativeExecutionKind {
        self.kind
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeReclaimRequest {
    pub idle_cache_bytes: u64,
    pub include_compiled_state: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeReclaimResult {
    pub reclaimed_host_bytes: u64,
    pub reclaimed_device_bytes: u64,
    pub retained_active_sequences: u32,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use mlxcel_core::session::{OwnedTensor, PreparedAttentionBias, PreparedModality};

    use super::*;

    static_assertions::assert_not_impl_any!(NativeGenerateAdmission: Clone);
    static_assertions::assert_not_impl_any!(NativeEmbedAdmission: Clone);
    static_assertions::assert_not_impl_any!(NativeExecutionHandle: Clone);

    fn valid_generate_request(prompt: NativePromptInput) -> NativeGenerateRequest {
        let prompt_tokens = prompt.logical_token_len() as u32;
        NativeGenerateRequest {
            request_id: 1,
            prompt,
            sampling: NativeSamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 0,
            },
            limits: NativeGenerationLimits {
                max_new_tokens: 1,
                max_total_tokens: prompt_tokens + 1,
            },
            eos_token_ids: Arc::from([]),
            output_suppressed_token_ids: Arc::from([]),
            scheduling: NativeSchedulingHints::default(),
            cache: NativeCacheContext {
                namespace: Arc::from("test"),
                input_identity: [1; 32],
                allow_read: false,
                allow_write: false,
            },
            constraint: NativeConstraintPlan::None,
            speculation: NativeSpeculationPolicy {
                preference: NativeSpeculationPreference::Disabled,
                minimum_fidelity: NativeSpeculativeFidelity::ExactTargetDistribution,
                allow_ordinary_fallback: true,
            },
        }
    }

    #[test]
    fn invalid_sampling_and_unbounded_policy_are_rejected() {
        let request = NativeGenerateRequest {
            request_id: 1,
            prompt: NativePromptInput::TokenIds(Arc::from([])),
            sampling: NativeSamplingConfig {
                temperature: f32::NAN,
                top_p: 1.0,
                top_k: None,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 0,
            },
            limits: NativeGenerationLimits {
                max_new_tokens: 1,
                max_total_tokens: 1,
            },
            eos_token_ids: Arc::from([]),
            output_suppressed_token_ids: Arc::from([]),
            scheduling: NativeSchedulingHints::default(),
            cache: NativeCacheContext {
                namespace: Arc::from("test"),
                input_identity: [0; 32],
                allow_read: false,
                allow_write: false,
            },
            constraint: NativeConstraintPlan::None,
            speculation: NativeSpeculationPolicy {
                preference: NativeSpeculationPreference::Disabled,
                minimum_fidelity: NativeSpeculativeFidelity::ExactTargetDistribution,
                allow_ordinary_fallback: true,
            },
        };
        assert_eq!(
            request.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidRequest
        );
    }

    struct ReleaseCounter(AtomicUsize);

    impl NativeReservationRelease for ReleaseCounter {
        fn release(&self, _reservation_id: u128) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn reservation_guard_remains_armed_until_started_commit() {
        let counter = Arc::new(ReleaseCounter(AtomicUsize::new(0)));
        let guard = NativeReservationGuard {
            reservation_id: 7,
            release: Some(counter.clone()),
        };
        drop(guard);
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);

        let guard = NativeReservationGuard {
            reservation_id: 8,
            release: Some(counter.clone()),
        };
        guard.commit_started();
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn validation_precedes_reservation_and_admission_releases_exactly_once() {
        let counter = Arc::new(ReleaseCounter(AtomicUsize::new(0)));
        let invalid = valid_generate_request(NativePromptInput::TokenIds(Arc::from([-1])));
        assert!(invalid.validated().is_err());
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);

        let valid = valid_generate_request(NativePromptInput::TokenIds(Arc::from([1])));
        let validated = valid.validated().unwrap();
        let admission = NativeGenerateAdmission::new(
            9,
            NativeOperationDriver::AutoregressiveGenerate,
            validated,
            counter.clone(),
        );
        drop(admission);
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);

        let validated = valid.validated().unwrap();
        let admission = NativeGenerateAdmission::new(
            10,
            NativeOperationDriver::AutoregressiveGenerate,
            validated,
            counter.clone(),
        );
        let (_, _, _, guard) = admission.into_parts();
        guard.commit_started();
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn token_ids_and_generation_total_are_validated_at_exact_boundaries() {
        let mut request = valid_generate_request(NativePromptInput::TokenIds(Arc::from([1, 2])));
        assert!(request.validate().is_ok());
        request.limits.max_total_tokens = 2;
        assert!(request.validate().is_err());

        let negative_lists = [
            NativePromptInput::TokenIds(Arc::from([-1])),
            NativePromptInput::TokenIds(Arc::from([1])),
            NativePromptInput::TokenIds(Arc::from([1])),
        ];
        for (index, prompt) in negative_lists.into_iter().enumerate() {
            let mut request = valid_generate_request(prompt);
            match index {
                0 => {}
                1 => request.eos_token_ids = Arc::from([-1]),
                2 => request.output_suppressed_token_ids = Arc::from([-1]),
                _ => unreachable!(),
            }
            assert!(request.validate().is_err());
        }

        assert!(NativeOwnedPreparedInput::new("schema", &[], MAX_NATIVE_PROMPT_TOKENS).is_ok());
        assert!(
            NativeOwnedPreparedInput::new("schema", &[], MAX_NATIVE_PROMPT_TOKENS + 1).is_err()
        );
        assert!(NativeOwnedPreparedInput::new("schema", &[], 0).is_err());
        assert!(
            valid_generate_request(NativePromptInput::TokenIds(Arc::from([])))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn embedding_limits_cover_declared_and_resolved_token_counts() {
        let request = NativeEmbedRequest {
            request_id: 1,
            inputs: Arc::from([
                NativeEmbeddingInput::TokenIds(Arc::from([1, 2])),
                NativeEmbeddingInput::Text(Arc::from("text")),
            ]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: 2,
                max_output_dimensions: MAX_NATIVE_EMBEDDING_DIMENSION as u32,
            },
            normalization: NativeEmbeddingNormalization::ModelDefault,
        };
        assert!(request.validate_resolved_input_tokens(&[2, 2]).is_ok());
        assert!(request.validate_resolved_input_tokens(&[2, 3]).is_err());
        assert!(request.validate_resolved_input_tokens(&[1, 2]).is_err());

        let mut too_many_dimensions = NativeEmbedRequest {
            request_id: request.request_id,
            inputs: Arc::clone(&request.inputs),
            limits: request.limits,
            normalization: request.normalization,
        };
        too_many_dimensions.limits.max_output_dimensions =
            MAX_NATIVE_EMBEDDING_DIMENSION as u32 + 1;
        assert!(too_many_dimensions.validate().is_err());

        let negative = NativeEmbedRequest {
            request_id: 2,
            inputs: Arc::from([NativeEmbeddingInput::TokenIds(Arc::from([-1]))]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: 1,
                max_output_dimensions: 1,
            },
            normalization: NativeEmbeddingNormalization::None,
        };
        assert!(negative.validate().is_err());

        for input in [
            NativeEmbeddingInput::TokenIds(Arc::from([])),
            NativeEmbeddingInput::Text(Arc::from("")),
        ] {
            let empty = NativeEmbedRequest {
                request_id: 3,
                inputs: Arc::from([input]),
                limits: NativeEmbeddingLimits {
                    max_input_tokens: 1,
                    max_output_dimensions: 1,
                },
                normalization: NativeEmbeddingNormalization::None,
            };
            assert!(empty.validate().is_err());
        }
        assert!(request.validate_resolved_input_tokens(&[2, 0]).is_err());
    }

    #[test]
    fn embedding_aggregate_bytes_are_bounded_even_for_shared_inputs() {
        let text: Arc<str> = "x".repeat(MAX_NATIVE_TEXT_BYTES).into();
        let inputs = (0..5)
            .map(|_| NativeEmbeddingInput::Text(Arc::clone(&text)))
            .collect::<Vec<_>>();
        let request = NativeEmbedRequest {
            request_id: 1,
            inputs: inputs.into(),
            limits: NativeEmbeddingLimits {
                max_input_tokens: 1,
                max_output_dimensions: 1,
            },
            normalization: NativeEmbeddingNormalization::ModelDefault,
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn embedding_global_token_ceiling_is_enforced_for_declared_and_resolved_inputs() {
        let ids: Arc<[i32]> = vec![1; MAX_NATIVE_PROMPT_TOKENS].into();
        let exact_ids = NativeEmbedRequest {
            request_id: 1,
            inputs: Arc::from([NativeEmbeddingInput::TokenIds(ids)]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: MAX_NATIVE_PROMPT_TOKENS as u32,
                max_output_dimensions: 1,
            },
            normalization: NativeEmbeddingNormalization::None,
        };
        assert!(exact_ids.validate().is_ok());
        assert!(
            exact_ids
                .validate_resolved_input_tokens(&[MAX_NATIVE_PROMPT_TOKENS])
                .is_ok()
        );

        let text = NativeEmbedRequest {
            request_id: 2,
            inputs: Arc::from([NativeEmbeddingInput::Text(Arc::from("x"))]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: MAX_NATIVE_PROMPT_TOKENS as u32,
                max_output_dimensions: 1,
            },
            normalization: NativeEmbeddingNormalization::None,
        };
        assert!(
            text.validate_resolved_input_tokens(&[MAX_NATIVE_PROMPT_TOKENS])
                .is_ok()
        );
        assert!(
            text.validate_resolved_input_tokens(&[MAX_NATIVE_PROMPT_TOKENS + 1])
                .is_err()
        );

        let over_limit = NativeEmbedRequest {
            request_id: 3,
            inputs: Arc::from([NativeEmbeddingInput::Text(Arc::from("x"))]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: MAX_NATIVE_PROMPT_TOKENS as u32 + 1,
                max_output_dimensions: 1,
            },
            normalization: NativeEmbeddingNormalization::None,
        };
        assert!(over_limit.validate().is_err());
    }

    #[test]
    fn prepared_input_bounds_are_checked_before_owned_copy() {
        let schema = "x".repeat(MAX_NATIVE_LABEL_BYTES + 1);
        assert!(NativeOwnedPreparedInput::new(schema.as_str(), &[], 0).is_err());
        assert!(
            validate_owned_prepared_lengths("schema".len(), MAX_NATIVE_PREPARED_BYTES + 1, 0)
                .is_err()
        );
    }

    fn prepared_with_modalities(modalities: Vec<PreparedModality>) -> PreparedPrefill {
        PreparedPrefill {
            token_ids: vec![1],
            embeddings: OwnedTensor {
                bytes: vec![0; 4],
                dtype: PreparedTensorDType::Float32,
                shape: vec![1, 1, 1],
            },
            positions: PreparedPositions::Sequential {
                start: 0,
                length: 1,
            },
            attention_bias: PreparedAttentionBias {
                tensor: OwnedTensor {
                    bytes: vec![0; 4],
                    dtype: PreparedTensorDType::Float32,
                    shape: vec![1],
                },
                causal: true,
            },
            sequence_len: 1,
            modalities,
            adapter_mode: PreparedAdapterMode::Language,
        }
    }

    #[test]
    fn prepared_media_item_counts_are_bounded_per_entry_and_in_aggregate() {
        let unbounded = prepared_with_modalities(vec![PreparedModality {
            family: "image".to_owned(),
            item_count: usize::MAX,
            token_count: 1,
        }]);
        assert!(NativePreparedPrefill::try_from_owned(unbounded).is_err());

        let aggregate = prepared_with_modalities(vec![
            PreparedModality {
                family: "image".to_owned(),
                item_count: MAX_NATIVE_PREPARED_MEDIA_ITEMS / 2 + 1,
                token_count: 1,
            },
            PreparedModality {
                family: "audio".to_owned(),
                item_count: MAX_NATIVE_PREPARED_MEDIA_ITEMS / 2,
                token_count: 1,
            },
        ]);
        assert!(NativePreparedPrefill::try_from_owned(aggregate).is_err());
    }

    #[test]
    fn prepared_tensor_dimensions_must_be_nonzero() {
        assert!(
            validate_owned_tensor(
                "zero attention bias",
                &[],
                PreparedTensorDType::Float32,
                &[0, 1],
            )
            .is_err()
        );

        let mut prepared = prepared_with_modalities(Vec::new());
        prepared.attention_bias.tensor.bytes.clear();
        prepared.attention_bias.tensor.shape = vec![0, 1];
        assert!(NativePreparedPrefill::try_from_owned(prepared).is_err());
    }
}
