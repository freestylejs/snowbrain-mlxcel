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

//! Neutral accepted-output, physical-event, and terminal sinks.

use std::sync::Arc;

use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::request::{
    MAX_NATIVE_CONSTRAINT_BYTES, MAX_NATIVE_LABEL_BYTES, MAX_NATIVE_PROMPT_TOKENS,
    NativeEmbedRequest, NativeEmbeddingNormalization, NativeGenerateRequest,
};

pub const MAX_NATIVE_ACCEPTED_UNITS: usize = 4_096;
pub const MAX_NATIVE_STABLE_BYTES_PER_UNIT: usize = 1024 * 1024;
pub const MAX_NATIVE_STABLE_BYTES_PER_CHUNK: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_EMBEDDING_DIMENSION: usize = 1_048_576;

fn invalid_output(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidOutput, message)
}

fn sink_rejected(message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::SinkRejected, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeSequenceIdentity {
    pub request_id: u128,
    pub sequence_id: u64,
    pub worker_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeAcceptedSource {
    Ordinary,
    SpeculativeTargetAccepted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAcceptedUnit {
    pub token_id: i32,
    pub stable_bytes: Arc<[u8]>,
    pub output_step: u64,
    pub source: NativeAcceptedSource,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeUsageDelta {
    pub input_tokens_evaluated: u64,
    pub input_tokens_reused: u64,
    pub output_tokens_accepted: u64,
    pub speculative_tokens_evaluated: u64,
    pub speculative_tokens_discarded: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeResumeCapability {
    AnyPrefix,
    WholeChunkOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativePrefixFacts {
    /// Number of prompt-prefix input tokens adopted by this sequence. This is
    /// independent of the output-chunk boundary.
    pub adopted_prompt_tokens: usize,
    /// Which accepted output prefixes can be retained if the sink continues.
    pub resume: NativeResumeCapability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAcceptedChunk {
    pub sequence: NativeSequenceIdentity,
    pub units: Vec<NativeAcceptedUnit>,
    pub usage: NativeUsageDelta,
    pub prefix: NativePrefixFacts,
}

impl NativeAcceptedChunk {
    /// Validate every externally retained accepted-output bound before a sink
    /// callback. Managed engines must call this before handing the chunk off.
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        if self.units.is_empty() || self.units.len() > MAX_NATIVE_ACCEPTED_UNITS {
            return Err(invalid_output(
                "accepted chunk is empty or exceeds its unit bound",
            ));
        }
        let mut stable_total = 0usize;
        let mut previous_step = None;
        for unit in &self.units {
            if unit.token_id < 0 {
                return Err(invalid_output(
                    "accepted output contains a negative token id",
                ));
            }
            if unit.stable_bytes.len() > MAX_NATIVE_STABLE_BYTES_PER_UNIT {
                return Err(invalid_output(
                    "accepted unit exceeds its stable-byte bound",
                ));
            }
            stable_total = stable_total
                .checked_add(unit.stable_bytes.len())
                .ok_or_else(|| invalid_output("accepted stable-byte total overflowed"))?;
            if stable_total > MAX_NATIVE_STABLE_BYTES_PER_CHUNK {
                return Err(invalid_output(
                    "accepted chunk exceeds its aggregate stable-byte bound",
                ));
            }
            if previous_step.is_some_and(|previous| unit.output_step <= previous) {
                return Err(invalid_output(
                    "accepted output steps must be strictly increasing",
                ));
            }
            previous_step = Some(unit.output_step);
        }
        if self.prefix.adopted_prompt_tokens > MAX_NATIVE_PROMPT_TOKENS {
            return Err(invalid_output(
                "adopted prompt prefix exceeds the native prompt-token bound",
            ));
        }
        if self.usage.output_tokens_accepted != self.units.len() as u64 {
            return Err(invalid_output(
                "accepted usage count does not match the accepted units",
            ));
        }
        Ok(())
    }

    /// Validate accepted output against the exact admitted request and the
    /// identity delivered by the successful `started` handshake.
    pub fn validate_for(
        &self,
        request: &NativeGenerateRequest,
        started_sequence: NativeSequenceIdentity,
    ) -> Result<(), NativeRuntimeError> {
        self.validate()?;
        if self.sequence != started_sequence
            || self.sequence.request_id != request.request_id
            || started_sequence.request_id != request.request_id
        {
            return Err(invalid_output(
                "accepted output identity disagrees with its request or started sequence",
            ));
        }
        if self.prefix.adopted_prompt_tokens > request.logical_prompt_tokens() {
            return Err(invalid_output(
                "adopted prompt tokens exceed the admitted prompt",
            ));
        }
        let max_new_tokens = u64::from(request.limits.max_new_tokens);
        if self.units.len() as u64 > max_new_tokens
            || self
                .units
                .last()
                .is_some_and(|unit| unit.output_step >= max_new_tokens)
        {
            return Err(invalid_output(
                "accepted output exceeds the admitted generation ceiling",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeConstraintTransition {
    Unchanged,
    Clear,
    Replace {
        format: Arc<str>,
        payload: Arc<[u8]>,
    },
}

impl NativeConstraintTransition {
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        if let Self::Replace { format, payload } = self {
            if format.is_empty() || format.len() > MAX_NATIVE_LABEL_BYTES {
                return Err(sink_rejected(
                    "replacement constraint format is empty or too large",
                ));
            }
            if payload.len() > MAX_NATIVE_CONSTRAINT_BYTES {
                return Err(sink_rejected(
                    "replacement constraint payload exceeds the native bound",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeContinuationDirective {
    Continue {
        constraint: NativeConstraintTransition,
    },
    ConsumePrefixAndContinue {
        consumed_units: usize,
        constraint: NativeConstraintTransition,
    },
    Stop {
        consumed_units: usize,
    },
    Fail {
        consumed_units: usize,
        error: NativeRuntimeError,
    },
}

/// Validated separation between physical target acceptance and semantic
/// commitment for one sink decision.
///
/// This value can only be created by validating a directive against the exact
/// accepted chunk. The accepted suffix remains physical target work; it is not
/// draft rejection and must be discarded or rolled back before continuation or
/// terminal delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDirectiveApplication {
    physically_accepted_units: usize,
    committed_units: usize,
    accepted_suffix_discarded_units: usize,
}

impl NativeDirectiveApplication {
    #[must_use]
    pub const fn physically_accepted_units(&self) -> usize {
        self.physically_accepted_units
    }

    #[must_use]
    pub const fn committed_units(&self) -> usize {
        self.committed_units
    }

    #[must_use]
    pub const fn accepted_suffix_discarded_units(&self) -> usize {
        self.accepted_suffix_discarded_units
    }
}

impl NativeContinuationDirective {
    /// Validate a sink-returned directive against the exact chunk it consumed.
    /// An engine must reject an invalid directive before retaining new policy
    /// bytes or scheduling a dependent forward pass.
    #[must_use]
    pub fn validate_for(
        &self,
        chunk: &NativeAcceptedChunk,
    ) -> Result<NativeDirectiveApplication, NativeRuntimeError> {
        chunk.validate()?;
        let (consumed_units, continues) = match self {
            Self::Continue { constraint } => {
                constraint.validate()?;
                (chunk.units.len(), true)
            }
            Self::ConsumePrefixAndContinue {
                consumed_units,
                constraint,
            } => {
                constraint.validate()?;
                (*consumed_units, true)
            }
            Self::Stop { consumed_units } | Self::Fail { consumed_units, .. } => {
                (*consumed_units, false)
            }
        };
        if consumed_units > chunk.units.len() {
            return Err(sink_rejected(
                "consumed prefix must be within the accepted chunk",
            ));
        }
        if continues
            && chunk.prefix.resume == NativeResumeCapability::WholeChunkOnly
            && consumed_units != chunk.units.len()
        {
            return Err(sink_rejected(
                "this accepted chunk can continue only after consuming the whole chunk",
            ));
        }
        Ok(NativeDirectiveApplication {
            physically_accepted_units: chunk.units.len(),
            committed_units: consumed_units,
            accepted_suffix_discarded_units: chunk.units.len() - consumed_units,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeMemoryPressure {
    Advisory,
    AdmissionLimited,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeRecoverability {
    Request,
    WorkerReload,
    ProcessInvariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeMemoryEvent {
    Pressure(NativeMemoryPressure),
    Reclaimed {
        bytes: u64,
    },
    AllocationFailure {
        recoverability: NativeRecoverability,
    },
    WorkerUnrecoverable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativePhysicalEvent {
    Prefill {
        evaluated_tokens: u32,
        reused_tokens: u32,
    },
    Decode {
        evaluated_tokens: u32,
        accepted_tokens: u32,
    },
    Memory(NativeMemoryEvent),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeStopReason {
    Eos,
    Limit,
    Requested,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativePhysicalTerminal {
    Completed { reason: NativeStopReason },
    Cancelled,
    Failed(NativeRuntimeError),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeMemorySnapshot {
    pub host_resident_bytes: u64,
    pub host_peak_bytes: u64,
    pub device_resident_bytes: u64,
    pub device_peak_bytes: u64,
    pub active_sequences: u32,
    pub queued_operations: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeEmbeddingOutput {
    pub input_index: u32,
    pub dimension: u32,
    pub normalized: bool,
    pub vector: Arc<[f32]>,
}

impl NativeEmbeddingOutput {
    /// Validate dimension/storage consistency and the global vector bound
    /// before a managed engine passes this value to an external sink.
    pub fn validate(&self) -> Result<(), NativeRuntimeError> {
        let dimension = usize::try_from(self.dimension)
            .map_err(|_| invalid_output("embedding dimension does not fit this host"))?;
        if dimension == 0
            || dimension > MAX_NATIVE_EMBEDDING_DIMENSION
            || dimension != self.vector.len()
        {
            return Err(invalid_output(
                "embedding vector violates its dimension or size bound",
            ));
        }
        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(invalid_output(
                "embedding vector contains a non-finite value",
            ));
        }
        Ok(())
    }

    /// Validate an output against the exact admitted embedding request. An
    /// engine must use this request-bound path before invoking `vector`.
    pub fn validate_for(&self, request: &NativeEmbedRequest) -> Result<(), NativeRuntimeError> {
        self.validate()?;
        let input_index = usize::try_from(self.input_index)
            .map_err(|_| invalid_output("embedding input index does not fit this host"))?;
        if input_index >= request.inputs.len() {
            return Err(invalid_output(
                "embedding input index is outside the admitted request",
            ));
        }
        if self.dimension > request.limits.max_output_dimensions {
            return Err(invalid_output(
                "embedding dimension exceeds the admitted request ceiling",
            ));
        }
        match request.normalization {
            NativeEmbeddingNormalization::L2 if !self.normalized => {
                return Err(invalid_output(
                    "embedding output is not normalized as the admitted request requires",
                ));
            }
            NativeEmbeddingNormalization::None if self.normalized => {
                return Err(invalid_output(
                    "embedding output normalized despite an explicit none policy",
                ));
            }
            NativeEmbeddingNormalization::ModelDefault
            | NativeEmbeddingNormalization::None
            | NativeEmbeddingNormalization::L2 => {}
        }
        Ok(())
    }
}

pub trait NativeGenerateSink: Send {
    fn started(&mut self, sequence: NativeSequenceIdentity) -> Result<(), NativeRuntimeError>;

    fn accepted(&mut self, chunk: NativeAcceptedChunk) -> NativeContinuationDirective;

    fn physical(&mut self, event: NativePhysicalEvent);

    fn terminal(&mut self, terminal: NativePhysicalTerminal);
}

pub trait NativeEmbeddingSink: Send {
    fn started(&mut self) -> Result<(), NativeRuntimeError>;

    fn vector(&mut self, output: NativeEmbeddingOutput) -> Result<(), NativeRuntimeError>;

    fn terminal(&mut self, terminal: NativePhysicalTerminal);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_runtime::request::{
        NativeCacheContext, NativeConstraintPlan, NativeEmbeddingInput, NativeEmbeddingLimits,
        NativeEmbeddingNormalization, NativeGenerationLimits, NativePromptInput,
        NativeSamplingConfig, NativeSchedulingHints, NativeSpeculationPolicy,
        NativeSpeculationPreference, NativeSpeculativeFidelity,
    };

    #[allow(dead_code)]
    fn sink_contracts_imply_send<G, E>()
    where
        G: NativeGenerateSink + ?Sized,
        E: NativeEmbeddingSink + ?Sized,
    {
        fn require_send<T: Send + ?Sized>() {}
        require_send::<G>();
        require_send::<E>();
    }

    fn chunk(units: usize) -> NativeAcceptedChunk {
        NativeAcceptedChunk {
            sequence: NativeSequenceIdentity {
                request_id: 1,
                sequence_id: 2,
                worker_epoch: 3,
            },
            units: (0..units)
                .map(|index| NativeAcceptedUnit {
                    token_id: index as i32,
                    stable_bytes: Arc::from(&b"x"[..]),
                    output_step: index as u64,
                    source: NativeAcceptedSource::Ordinary,
                })
                .collect(),
            usage: NativeUsageDelta {
                output_tokens_accepted: units as u64,
                ..NativeUsageDelta::default()
            },
            prefix: NativePrefixFacts {
                adopted_prompt_tokens: 1,
                resume: NativeResumeCapability::AnyPrefix,
            },
        }
    }

    fn generate_request() -> NativeGenerateRequest {
        NativeGenerateRequest {
            request_id: 1,
            prompt: NativePromptInput::TokenIds(Arc::from([1, 2])),
            sampling: NativeSamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 1,
            },
            limits: NativeGenerationLimits {
                max_new_tokens: 2,
                max_total_tokens: 4,
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
    fn accepted_chunk_is_bound_to_request_started_identity_prompt_and_output_ceiling() {
        let request = generate_request();
        let mut accepted = chunk(2);
        assert!(accepted.validate_for(&request, accepted.sequence).is_ok());

        let mut wrong_request = generate_request();
        wrong_request.request_id = 9;
        assert!(
            accepted
                .validate_for(&wrong_request, accepted.sequence)
                .is_err()
        );

        let mut wrong_started = accepted.sequence;
        wrong_started.worker_epoch += 1;
        assert!(accepted.validate_for(&request, wrong_started).is_err());

        accepted.prefix.adopted_prompt_tokens = 3;
        assert!(accepted.validate_for(&request, accepted.sequence).is_err());
        accepted.prefix.adopted_prompt_tokens = 2;
        accepted.units[1].output_step = request.limits.max_new_tokens as u64;
        assert!(accepted.validate_for(&request, accepted.sequence).is_err());
    }

    #[test]
    fn consumed_prefix_and_replacement_are_bounded_by_the_chunk() {
        let accepted = chunk(2);
        let invalid_prefix = NativeContinuationDirective::ConsumePrefixAndContinue {
            consumed_units: 3,
            constraint: NativeConstraintTransition::Unchanged,
        };
        assert_eq!(
            invalid_prefix.validate_for(&accepted).unwrap_err().kind(),
            NativeRuntimeErrorKind::SinkRejected
        );
        for consumed_units in [0, accepted.units.len()] {
            let valid = NativeContinuationDirective::ConsumePrefixAndContinue {
                consumed_units,
                constraint: NativeConstraintTransition::Unchanged,
            };
            let application = valid.validate_for(&accepted).unwrap();
            assert_eq!(application.physically_accepted_units(), 2);
            assert_eq!(application.committed_units(), consumed_units);
            assert_eq!(
                application.accepted_suffix_discarded_units(),
                2 - consumed_units
            );
        }

        let oversized = NativeContinuationDirective::Continue {
            constraint: NativeConstraintTransition::Replace {
                format: Arc::from("grammar"),
                payload: vec![0; MAX_NATIVE_CONSTRAINT_BYTES + 1].into(),
            },
        };
        assert_eq!(
            oversized.validate_for(&accepted).unwrap_err().kind(),
            NativeRuntimeErrorKind::SinkRejected
        );
    }

    #[test]
    fn whole_chunk_only_blocks_only_proper_prefix_continuation() {
        let mut accepted = chunk(2);
        accepted.prefix.resume = NativeResumeCapability::WholeChunkOnly;

        for consumed_units in [0, 1] {
            let continuing = NativeContinuationDirective::ConsumePrefixAndContinue {
                consumed_units,
                constraint: NativeConstraintTransition::Unchanged,
            };
            assert!(continuing.validate_for(&accepted).is_err());

            let stopped = NativeContinuationDirective::Stop { consumed_units };
            let application = stopped.validate_for(&accepted).unwrap();
            assert_eq!(application.committed_units(), consumed_units);
            assert_eq!(
                application.accepted_suffix_discarded_units(),
                accepted.units.len() - consumed_units
            );
        }

        let full = NativeContinuationDirective::ConsumePrefixAndContinue {
            consumed_units: accepted.units.len(),
            constraint: NativeConstraintTransition::Unchanged,
        };
        assert!(full.validate_for(&accepted).is_ok());
        assert!(
            NativeContinuationDirective::Stop { consumed_units: 3 }
                .validate_for(&accepted)
                .is_err()
        );
    }

    #[test]
    fn fail_preserves_the_bounded_typed_error_and_prefix_accounting() {
        let accepted = chunk(2);
        let error = NativeRuntimeError::try_new(
            NativeRuntimeErrorKind::SinkRejected,
            "semantic sink rejected output",
        )
        .unwrap();
        let directive = NativeContinuationDirective::Fail {
            consumed_units: 0,
            error: error.clone(),
        };
        let application = directive.validate_for(&accepted).unwrap();
        assert_eq!(application.physically_accepted_units(), 2);
        assert_eq!(application.committed_units(), 0);
        assert_eq!(application.accepted_suffix_discarded_units(), 2);
        let NativeContinuationDirective::Fail {
            error: retained, ..
        } = directive
        else {
            unreachable!();
        };
        assert_eq!(retained, error);
    }

    #[test]
    fn accepted_and_embedding_aggregate_bounds_are_enforced() {
        let mut invalid = chunk(1);
        invalid.units[0].stable_bytes = vec![0; MAX_NATIVE_STABLE_BYTES_PER_UNIT + 1].into();
        assert_eq!(
            invalid.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );

        let mut negative = chunk(1);
        negative.units[0].token_id = -1;
        assert_eq!(
            negative.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );

        let output = NativeEmbeddingOutput {
            input_index: 0,
            dimension: 2,
            normalized: false,
            vector: Arc::from([1.0]),
        };
        assert_eq!(
            output.validate().unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );
    }

    #[test]
    fn embedding_output_is_bound_to_request_index_and_dimension() {
        let request = NativeEmbedRequest {
            request_id: 1,
            inputs: Arc::from([
                NativeEmbeddingInput::TokenIds(Arc::from([1])),
                NativeEmbeddingInput::TokenIds(Arc::from([2])),
            ]),
            limits: NativeEmbeddingLimits {
                max_input_tokens: 1,
                max_output_dimensions: 2,
            },
            normalization: NativeEmbeddingNormalization::None,
        };
        let exact = NativeEmbeddingOutput {
            input_index: 1,
            dimension: 2,
            normalized: false,
            vector: Arc::from([1.0, 2.0]),
        };
        assert!(exact.validate_for(&request).is_ok());

        let mut index_over = exact.clone();
        index_over.input_index = 2;
        assert_eq!(
            index_over.validate_for(&request).unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );

        let dimension_over = NativeEmbeddingOutput {
            input_index: 0,
            dimension: 3,
            normalized: false,
            vector: Arc::from([1.0, 2.0, 3.0]),
        };
        assert_eq!(
            dimension_over.validate_for(&request).unwrap_err().kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );

        let mut unexpected_normalization = exact.clone();
        unexpected_normalization.normalized = true;
        assert_eq!(
            unexpected_normalization
                .validate_for(&request)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );

        let l2_request = NativeEmbedRequest {
            request_id: request.request_id,
            inputs: Arc::clone(&request.inputs),
            limits: request.limits,
            normalization: NativeEmbeddingNormalization::L2,
        };
        let mut missing_normalization = exact;
        missing_normalization.normalized = false;
        assert_eq!(
            missing_normalization
                .validate_for(&l2_request)
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidOutput
        );
        missing_normalization.normalized = true;
        assert!(missing_normalization.validate_for(&l2_request).is_ok());
    }
}
