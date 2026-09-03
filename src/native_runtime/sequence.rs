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

//! Physical sequence state shared by every managed-runtime frontend.
//!
//! This is intentionally crate-private: it may own MLX/model-family state, but
//! none of those implementation types may cross the public native boundary.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::SamplerState;

use crate::vision::merge::InputEmbeddings;

use super::constraint::NativeConstraintState;
use super::request::NativeReservationRelease;
use super::{
    NativeAcceptedChunk, NativeAcceptedSource, NativeAcceptedUnit, NativeContinuationDirective,
    NativeGenerateRequest, NativeGenerateSink, NativeIncrementalDecoder, NativePhysicalAccounting,
    NativePhysicalTerminal, NativePrefixFacts, NativeResumeCapability, NativeRuntimeError,
    NativeRuntimeErrorKind, NativeSequenceIdentity, NativeStopReason, NativeUsageDelta,
};

/// Priority used by the physical prefill queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum PhysicalPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
}

/// Physical lifecycle of one sequence. Frontends observe this state but do
/// not drive transitions themselves.
#[derive(Debug)]
#[non_exhaustive]
pub enum PhysicalSequencePhase {
    Queued,
    Prefilling,
    Decoding,
    Finished(PhysicalFinishReason),
}

/// Why physical execution stopped. String-stop and repetition termination are
/// retained as physical scheduler facts; their frontend meaning is assigned
/// outside this core.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PhysicalFinishReason {
    Stop,
    StopSequence,
    Length,
    RepetitionLoop,
    Cancelled,
    Error(String),
}

impl PhysicalSequencePhase {
    pub fn transition_to(&mut self, next: Self) -> Result<(), String> {
        let valid = match (&*self, &next) {
            (Self::Queued, Self::Prefilling) | (Self::Prefilling, Self::Decoding) => true,
            (
                Self::Prefilling,
                Self::Finished(
                    PhysicalFinishReason::Stop
                    | PhysicalFinishReason::StopSequence
                    | PhysicalFinishReason::Length
                    | PhysicalFinishReason::RepetitionLoop,
                ),
            ) => true,
            (Self::Decoding, Self::Finished(_)) | (Self::Decoding, Self::Queued) => true,
            (
                _,
                Self::Finished(PhysicalFinishReason::Cancelled | PhysicalFinishReason::Error(_)),
            ) if !self.is_finished() => true,
            _ => false,
        };
        if valid {
            *self = next;
            Ok(())
        } else {
            Err(format!("invalid state transition: {self:?} -> {next:?}"))
        }
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
}

/// The single physical state authority consumed by `PrefillQueue`,
/// `ActiveBatch`, and the model/cache scheduler.
pub struct PhysicalSequence {
    pub seq_id: SequenceId,
    pub state: PhysicalSequencePhase,
    pub prompt_tokens: Vec<i32>,
    pub sampling: SamplingConfig,
    pub max_tokens: usize,
    pub eos_token_ids: Vec<i32>,
    pub priority: PhysicalPriority,
    pub vlm_embeddings: Option<InputEmbeddings>,
    pub images: Vec<Vec<u8>>,
    pub audio: Vec<Vec<u8>>,
    pub generated_tokens: Vec<i32>,
    pub(crate) token_history: Vec<i32>,
    pub(crate) sampler_state: Option<SamplerState>,
    pub(crate) merged_eos: Vec<i32>,
    pub prefill_offset: usize,
    pub prefill_start_offset: usize,
    pub already_cached_tokens: usize,
    pub cancelled: Arc<AtomicBool>,
    pub created_at: Instant,
    pub prefill_start: Option<Instant>,
    pub first_token_time: Option<Instant>,
    /// Present only for native managed generation. Constructed and retained on
    /// the scheduler worker; never placed in a command or output sink.
    pub(crate) decoder: Option<Box<dyn NativeIncrementalDecoder>>,
    /// Native output ownership. Server sequences leave this absent and retain
    /// their frontend projection in `SequenceInfo`.
    pub(crate) native: Option<NativeSequencePort>,
}

impl std::fmt::Debug for PhysicalSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalSequence")
            .field("seq_id", &self.seq_id)
            .field("state", &self.state)
            .field("prompt_tokens", &self.prompt_tokens.len())
            .field("generated_tokens", &self.generated_tokens.len())
            .field("priority", &self.priority)
            .field("has_decoder", &self.decoder.is_some())
            .field("native", &self.native.is_some())
            .finish_non_exhaustive()
    }
}

pub(crate) struct NativeSequencePort {
    request: Arc<NativeGenerateRequest>,
    identity: NativeSequenceIdentity,
    sink: Box<dyn NativeGenerateSink + Send>,
    constraint: NativeConstraintState,
    accounting: NativePhysicalAccounting,
    release: Option<(u128, Arc<dyn NativeReservationRelease>)>,
    terminal: bool,
}

#[derive(Debug)]
pub(crate) struct NativeAcceptOutcome {
    pub committed_units: usize,
    pub discarded_units: usize,
    pub continue_execution: bool,
    pub terminal: Option<NativePhysicalTerminal>,
}

impl NativeSequencePort {
    pub(crate) fn new(
        request: Arc<NativeGenerateRequest>,
        identity: NativeSequenceIdentity,
        sink: Box<dyn NativeGenerateSink + Send>,
        reservation_id: u128,
        release: Arc<dyn NativeReservationRelease>,
    ) -> Self {
        Self {
            constraint: NativeConstraintState::new(request.constraint.clone()),
            request,
            identity,
            sink,
            accounting: NativePhysicalAccounting::default(),
            release: Some((reservation_id, release)),
            terminal: false,
        }
    }

    pub(crate) fn accepted(
        &mut self,
        decoder: &mut dyn NativeIncrementalDecoder,
        token_ids: &[i32],
        source: NativeAcceptedSource,
        resume: NativeResumeCapability,
        adopted_prompt_tokens: usize,
        first_output_step: u64,
        mut usage: NativeUsageDelta,
    ) -> Result<NativeAcceptOutcome, NativeRuntimeError> {
        if self.terminal {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::StaleHandle,
                "accepted output attempted after native terminal",
            ));
        }
        if token_ids.is_empty() || token_ids.len() > super::MAX_NATIVE_ACCEPTED_UNITS {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidOutput,
                "physical accepted chunk is empty or exceeds its unit bound",
            ));
        }
        if resume == NativeResumeCapability::AnyPrefix {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "this physical driver has no suffix rollback authority for AnyPrefix resume",
            ));
        }
        let mut units = Vec::with_capacity(token_ids.len());
        for (offset, &token_id) in token_ids.iter().enumerate() {
            let delta = decoder.push(token_id).map_err(|error| {
                NativeRuntimeError::new(NativeRuntimeErrorKind::Tokenizer, error.to_string())
            })?;
            units.push(NativeAcceptedUnit {
                token_id,
                stable_bytes: Arc::from(delta.stable_bytes()),
                output_step: first_output_step.saturating_add(offset as u64),
                source,
            });
        }
        usage.output_tokens_accepted = units.len() as u64;
        let chunk = NativeAcceptedChunk {
            sequence: self.identity,
            units,
            usage,
            prefix: NativePrefixFacts {
                adopted_prompt_tokens,
                resume,
            },
        };
        chunk.validate_for(&self.request, self.identity)?;
        let directive = self.sink.accepted(chunk.clone());
        let application = directive.validate_for(&chunk)?;
        self.accounting.record_chunk(usage, application);

        let terminal = match &directive {
            NativeContinuationDirective::Continue { constraint }
            | NativeContinuationDirective::ConsumePrefixAndContinue { constraint, .. } => {
                if matches!(
                    constraint,
                    super::NativeConstraintTransition::Replace { .. }
                ) {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::UnsupportedCapability,
                        "this physical driver has no compiler for a replacement constraint",
                    ));
                }
                self.constraint.apply(constraint)?;
                None
            }
            NativeContinuationDirective::Stop { .. } => Some(NativePhysicalTerminal::Completed {
                reason: NativeStopReason::Requested,
            }),
            NativeContinuationDirective::Fail { error, .. } => {
                Some(NativePhysicalTerminal::Failed(error.clone()))
            }
        };
        Ok(NativeAcceptOutcome {
            committed_units: application.committed_units(),
            discarded_units: application.accepted_suffix_discarded_units(),
            continue_execution: terminal.is_none(),
            terminal,
        })
    }

    /// Called immediately before the next dependent model sample. Tests can
    /// compare this revision with the revision applied by the sink directive.
    pub(crate) fn mark_dependent_sample(&mut self) {
        self.constraint.mark_dependent_sample();
    }

    pub(crate) fn constraint_revisions(&self) -> (u64, u64) {
        (
            self.constraint.revision(),
            self.constraint.sampled_revision(),
        )
    }

    pub(crate) fn accounting(&self) -> NativePhysicalAccounting {
        self.accounting
    }

    pub(crate) fn physical(&mut self, event: super::NativePhysicalEvent) {
        if !self.terminal {
            self.sink.physical(event);
        }
    }

    pub(crate) const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub(crate) fn terminal(&mut self, terminal: NativePhysicalTerminal) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        self.sink.terminal(terminal);
        self.release_reservation();
    }

    /// Finalize only after the scheduler has discarded any uncommitted suffix
    /// and made the sequence ineligible for cache donation. Decoder flush is
    /// therefore checked before the sole terminal callback and reservation
    /// release. A caller-supplied failure remains the primary terminal error;
    /// all other terminal kinds become a typed decoder failure when a tail
    /// cannot be attributed to an accepted unit.
    pub(crate) fn terminal_after_decoder(
        &mut self,
        decoder: &mut dyn NativeIncrementalDecoder,
        terminal: NativePhysicalTerminal,
    ) {
        if self.terminal {
            return;
        }
        let flush = self.finish_decoder(decoder);
        let terminal = match (terminal, flush) {
            (failure @ NativePhysicalTerminal::Failed(_), _) => failure,
            (_, Err(error)) => NativePhysicalTerminal::Failed(error),
            (terminal, Ok(())) => terminal,
        };
        self.terminal(terminal);
    }

    pub(crate) fn finish_decoder(
        &mut self,
        decoder: &mut dyn NativeIncrementalDecoder,
    ) -> Result<(), NativeRuntimeError> {
        let delta = decoder.flush().map_err(|error| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::Tokenizer, error.to_string())
        })?;
        if !delta.stable_bytes().is_empty() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "native streaming decoder retained terminal bytes without a unit attribution",
            ));
        }
        Ok(())
    }

    fn release_reservation(&mut self) {
        if let Some((reservation_id, release)) = self.release.take() {
            release.release(reservation_id);
        }
    }
}

impl Drop for NativeSequencePort {
    fn drop(&mut self) {
        if !self.terminal {
            self.terminal = true;
            self.sink
                .terminal(NativePhysicalTerminal::Failed(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Infrastructure,
                    "native sequence dropped before a terminal outcome",
                )));
        }
        self.release_reservation();
    }
}

impl Drop for PhysicalSequence {
    fn drop(&mut self) {
        let (Some(decoder), Some(port)) = (self.decoder.as_deref_mut(), self.native.as_mut())
        else {
            return;
        };
        if !port.is_terminal() {
            port.terminal_after_decoder(
                decoder,
                NativePhysicalTerminal::Failed(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Infrastructure,
                    "native physical sequence dropped before scheduler finalization",
                )),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn native_sequence_does_not_retain_server_projection_authority() {
        let production = include_str!("sequence.rs")
            .split_once("#[cfg(test)]")
            .expect("test module boundary")
            .0;
        for forbidden in [
            "MlxcelServerProjection",
            "queue_server_logprobs",
            "server_matched_stop",
            "crate::server::",
        ] {
            assert!(
                !production.contains(forbidden),
                "native physical sequence retained server authority: {forbidden}"
            );
        }
    }

    #[test]
    fn server_compatibility_does_not_publish_native_sequence_module_or_ports() {
        let module = include_str!("mod.rs");
        assert!(module.contains("pub(crate) mod sequence;"));
        assert!(!module.contains("\npub mod sequence;"));

        let production = include_str!("sequence.rs")
            .split_once("#[cfg(test)]")
            .expect("test module boundary")
            .0;
        assert!(production.contains("pub(crate) decoder:"));
        assert!(production.contains("pub(crate) native:"));
        assert!(production.contains("pub(crate) struct NativeSequencePort"));
    }
    use crate::native_runtime::{
        NativeCacheContext, NativeConstraintPlan, NativeConstraintTransition,
        NativeGenerationLimits, NativePromptInput, NativeSamplingConfig, NativeSchedulingHints,
        NativeSpeculationPolicy, NativeSpeculationPreference, NativeSpeculativeFidelity,
        StableDecodeDelta, TokenizerError,
    };

    struct NonSendDecoder {
        pushes: Rc<Cell<usize>>,
        flush_bytes: Vec<u8>,
    }

    static_assertions::assert_not_impl_any!(NonSendDecoder: Send, Sync);

    impl NativeIncrementalDecoder for NonSendDecoder {
        fn push(&mut self, token_id: i32) -> Result<StableDecodeDelta, TokenizerError> {
            self.pushes.set(self.pushes.get() + 1);
            StableDecodeDelta::try_new(vec![token_id as u8], 0)
        }

        fn flush(&mut self) -> Result<StableDecodeDelta, TokenizerError> {
            StableDecodeDelta::try_new(self.flush_bytes.clone(), 0)
        }
    }

    #[derive(Default)]
    struct SinkRecord {
        chunks: Vec<NativeAcceptedChunk>,
        terminals: Vec<NativePhysicalTerminal>,
    }

    struct TestSink {
        record: Arc<Mutex<SinkRecord>>,
        directives: VecDeque<NativeContinuationDirective>,
    }

    impl NativeGenerateSink for TestSink {
        fn started(&mut self, _sequence: NativeSequenceIdentity) -> Result<(), NativeRuntimeError> {
            Ok(())
        }

        fn accepted(&mut self, chunk: NativeAcceptedChunk) -> NativeContinuationDirective {
            self.record.lock().unwrap().chunks.push(chunk);
            self.directives.pop_front().unwrap()
        }

        fn physical(&mut self, _event: super::super::NativePhysicalEvent) {}

        fn terminal(&mut self, terminal: NativePhysicalTerminal) {
            self.record.lock().unwrap().terminals.push(terminal);
        }
    }

    struct TestRelease {
        count: AtomicUsize,
        record: Arc<Mutex<SinkRecord>>,
    }

    impl NativeReservationRelease for TestRelease {
        fn release(&self, _reservation_id: u128) {
            assert_eq!(
                self.record.lock().unwrap().terminals.len(),
                1,
                "reservation released before the sole terminal callback"
            );
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn request() -> Arc<NativeGenerateRequest> {
        Arc::new(NativeGenerateRequest {
            request_id: 9,
            prompt: NativePromptInput::TokenIds(Arc::from([1, 2])),
            sampling: NativeSamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: Some(1),
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 1,
            },
            limits: NativeGenerationLimits {
                max_new_tokens: 8,
                max_total_tokens: 10,
            },
            eos_token_ids: Arc::from([2]),
            output_suppressed_token_ids: Arc::from([]),
            scheduling: NativeSchedulingHints::default(),
            cache: NativeCacheContext {
                namespace: Arc::from("test"),
                input_identity: [1; 32],
                allow_read: false,
                allow_write: false,
            },
            constraint: NativeConstraintPlan::ValidateOnly,
            speculation: NativeSpeculationPolicy {
                preference: NativeSpeculationPreference::Disabled,
                minimum_fidelity: NativeSpeculativeFidelity::ExactTargetDistribution,
                allow_ordinary_fallback: true,
            },
        })
    }

    fn make_port(
        directives: Vec<NativeContinuationDirective>,
    ) -> (NativeSequencePort, Arc<Mutex<SinkRecord>>, Arc<TestRelease>) {
        let record = Arc::new(Mutex::new(SinkRecord::default()));
        let release = Arc::new(TestRelease {
            count: AtomicUsize::new(0),
            record: Arc::clone(&record),
        });
        let release_trait: Arc<dyn NativeReservationRelease> = release.clone();
        let port = NativeSequencePort::new(
            request(),
            NativeSequenceIdentity {
                request_id: 9,
                sequence_id: 3,
                worker_epoch: 4,
            },
            Box::new(TestSink {
                record: Arc::clone(&record),
                directives: directives.into(),
            }),
            7,
            release_trait,
        );
        (port, record, release)
    }

    #[test]
    fn stable_bytes_are_forwarded_once_and_constraint_precedes_dependent_sample() {
        let transition = NativeConstraintTransition::Clear;
        let (mut port, record, release) = make_port(vec![NativeContinuationDirective::Continue {
            constraint: transition,
        }]);
        let pushes = Rc::new(Cell::new(0));
        let mut decoder = NonSendDecoder {
            pushes: Rc::clone(&pushes),
            flush_bytes: Vec::new(),
        };
        let outcome = port
            .accepted(
                &mut decoder,
                &[65],
                NativeAcceptedSource::Ordinary,
                NativeResumeCapability::WholeChunkOnly,
                0,
                0,
                NativeUsageDelta::default(),
            )
            .unwrap();
        assert_eq!(outcome.committed_units, 1);
        assert!(outcome.continue_execution);
        assert_eq!(pushes.get(), 1);
        assert_eq!(
            record.lock().unwrap().chunks[0].units[0]
                .stable_bytes
                .as_ref(),
            b"A"
        );
        assert_eq!(port.constraint_revisions(), (1, 0));
        port.mark_dependent_sample();
        assert_eq!(port.constraint_revisions(), (1, 1));
        let accounting = port.accounting();
        assert_eq!(accounting.output_tokens_target_accepted, 1);
        assert_eq!(accounting.output_units_committed, 1);
        port.terminal_after_decoder(
            &mut decoder,
            NativePhysicalTerminal::Completed {
                reason: NativeStopReason::Requested,
            },
        );
        port.terminal_after_decoder(&mut decoder, NativePhysicalTerminal::Cancelled);
        assert_eq!(record.lock().unwrap().terminals.len(), 1);
        assert_eq!(release.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn attached_scheduler_failures_emit_one_terminal_without_accepted_output() {
        for message in [
            "Speculative burst: slice failed",
            "Server busy: prefill queue full",
            "Preemption re-queue failed",
        ] {
            let (mut port, record, release) = make_port(Vec::new());
            let terminal = NativePhysicalTerminal::Failed(NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                message,
            ));
            port.terminal(terminal.clone());
            port.terminal(terminal);
            let record = record.lock().unwrap();
            assert!(record.chunks.is_empty(), "{message}");
            assert_eq!(record.terminals.len(), 1, "{message}");
            assert_eq!(release.count.load(Ordering::Relaxed), 1, "{message}");
        }
    }

    #[test]
    fn live_whole_chunk_unit_fallback_rejects_partial_continue() {
        let (mut port, record, release) = make_port(vec![
            NativeContinuationDirective::ConsumePrefixAndContinue {
                consumed_units: 0,
                constraint: NativeConstraintTransition::Unchanged,
            },
        ]);
        let mut decoder = NonSendDecoder {
            pushes: Rc::new(Cell::new(0)),
            flush_bytes: Vec::new(),
        };
        let error = port
            .accepted(
                &mut decoder,
                &[65],
                NativeAcceptedSource::Ordinary,
                NativeResumeCapability::WholeChunkOnly,
                0,
                0,
                NativeUsageDelta::default(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), NativeRuntimeErrorKind::SinkRejected);
        port.terminal_after_decoder(&mut decoder, NativePhysicalTerminal::Failed(error));
        assert_eq!(record.lock().unwrap().chunks.len(), 1);
        assert_eq!(record.lock().unwrap().terminals.len(), 1);
        assert_eq!(release.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unsupported_any_prefix_declines_before_decode_or_sink() {
        let (mut port, record, _release) = make_port(Vec::new());
        let pushes = Rc::new(Cell::new(0));
        let mut decoder = NonSendDecoder {
            pushes: Rc::clone(&pushes),
            flush_bytes: Vec::new(),
        };
        let error = port
            .accepted(
                &mut decoder,
                &[65, 66],
                NativeAcceptedSource::Ordinary,
                NativeResumeCapability::AnyPrefix,
                0,
                0,
                NativeUsageDelta::default(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), NativeRuntimeErrorKind::UnsupportedGuarantee);
        assert_eq!(pushes.get(), 0);
        assert!(record.lock().unwrap().chunks.is_empty());
    }

    #[test]
    fn stop_discards_suffix_and_terminal_flush_tail_is_typed() {
        let (mut port, record, release) = make_port(vec![NativeContinuationDirective::Stop {
            consumed_units: 0,
        }]);
        let mut decoder = NonSendDecoder {
            pushes: Rc::new(Cell::new(0)),
            flush_bytes: b"tail".to_vec(),
        };
        let outcome = port
            .accepted(
                &mut decoder,
                &[65],
                NativeAcceptedSource::Ordinary,
                NativeResumeCapability::WholeChunkOnly,
                0,
                0,
                NativeUsageDelta::default(),
            )
            .unwrap();
        assert_eq!((outcome.committed_units, outcome.discarded_units), (0, 1));
        assert!(!outcome.continue_execution);
        assert!(matches!(
            outcome.terminal,
            Some(NativePhysicalTerminal::Completed {
                reason: NativeStopReason::Requested
            })
        ));
        assert_eq!(port.accounting().accepted_suffix_discarded, 1);
        assert!(record.lock().unwrap().terminals.is_empty());
        assert_eq!(release.count.load(Ordering::Relaxed), 0);
        port.terminal_after_decoder(&mut decoder, outcome.terminal.unwrap());
        let terminals = &record.lock().unwrap().terminals;
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            &terminals[0],
            NativePhysicalTerminal::Failed(error)
                if error.kind() == NativeRuntimeErrorKind::UnsupportedGuarantee
        ));
        assert_eq!(release.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_directive_preserves_error_and_cancel_flushes_before_release() {
        let caller_error = NativeRuntimeError::new(
            NativeRuntimeErrorKind::SinkRejected,
            "caller rejected accepted unit",
        );
        let (mut port, record, release) = make_port(vec![NativeContinuationDirective::Fail {
            consumed_units: 0,
            error: caller_error.clone(),
        }]);
        let mut decoder = NonSendDecoder {
            pushes: Rc::new(Cell::new(0)),
            flush_bytes: b"unattributed".to_vec(),
        };
        let outcome = port
            .accepted(
                &mut decoder,
                &[65],
                NativeAcceptedSource::Ordinary,
                NativeResumeCapability::WholeChunkOnly,
                0,
                0,
                NativeUsageDelta::default(),
            )
            .unwrap();
        port.terminal_after_decoder(&mut decoder, outcome.terminal.unwrap());
        assert!(matches!(
            &record.lock().unwrap().terminals[0],
            NativePhysicalTerminal::Failed(error) if error == &caller_error
        ));
        assert_eq!(release.count.load(Ordering::Relaxed), 1);

        let (mut port, record, release) = make_port(Vec::new());
        decoder.flush_bytes.clear();
        port.terminal_after_decoder(&mut decoder, NativePhysicalTerminal::Cancelled);
        assert_eq!(
            record.lock().unwrap().terminals,
            vec![NativePhysicalTerminal::Cancelled]
        );
        assert_eq!(release.count.load(Ordering::Relaxed), 1);
    }
}
