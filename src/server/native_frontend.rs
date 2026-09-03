// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Crate-private projection from the managed native engine to server events.
//!
//! The public native contract remains transport neutral. This module retains
//! the server-only request sidecars and is the sole place where accepted
//! stable bytes and native terminals become [`GenerateEvent`] values.

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use mlxcel_core::sampling::TokenLogprobData;

use crate::native_runtime::{
    NativeAcceptedChunk, NativeConstraintTransition, NativeContinuationDirective,
    NativeGenerateSink, NativePhysicalEvent, NativePhysicalTerminal, NativeRuntimeError,
    NativeRuntimeErrorKind, NativeSequenceIdentity, NativeStopReason,
};
use crate::server::ServerGenerateOptions;
use crate::server::batch::stop_matcher::StopMatcher;
use crate::server::media::{MediaRequestMetadata, ResolvedVideo};
use crate::server::model_provider::model_worker::build_generation_result_with_cache;
use crate::server::model_provider::{
    GenerateEvent, ModelRequest, PrefillStats, SingleStreamQueueReservation, StopKind,
};

/// Server-only operation shape admitted by the concrete native engine.
pub(crate) struct MlxcelServerGenerateRequest {
    pub prompt: String,
    pub prompt_token_ids: Option<Vec<i32>>,
    pub options: ServerGenerateOptions,
    pub images: Vec<Vec<u8>>,
    pub audio: Vec<Vec<u8>>,
    pub videos: Vec<ResolvedVideo>,
    pub media: MediaRequestMetadata,
    pub queue_reservation: Option<SingleStreamQueueReservation>,
    pub response_tx: mpsc::Sender<GenerateEvent>,
    pub cancelled: Arc<AtomicBool>,
}

impl MlxcelServerGenerateRequest {
    pub(crate) fn into_legacy_model_request(self) -> ModelRequest {
        ModelRequest::Generate {
            prompt: self.prompt,
            prompt_token_ids: self.prompt_token_ids,
            options: self.options,
            images: self.images,
            audio: self.audio,
            videos: self.videos,
            media: self.media,
            queue_reservation: self.queue_reservation,
            response_tx: self.response_tx,
            cancelled: self.cancelled,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum MlxcelServerFrontendError {
    #[error("{0}")]
    Native(#[from] NativeRuntimeError),
    #[error("native-managed server worker is no longer running")]
    WorkerGone,
}

/// Server frontend. The concrete managed engine remains behind this mutex so
/// admission and command enqueue are serialized, while each worker-started
/// handshake waits after releasing the mutex. HTTP threads never receive the
/// scheduler channel.
pub(crate) struct MlxcelServerFrontend {
    engine: Mutex<crate::native_runtime::engine::MlxcelServerManagedEngine>,
}

impl MlxcelServerFrontend {
    pub(crate) fn from_managed_engine(
        engine: crate::native_runtime::engine::MlxcelServerManagedEngine,
    ) -> Self {
        Self {
            engine: Mutex::new(engine),
        }
    }

    pub(crate) fn from_legacy_operation_channel(command_tx: mpsc::Sender<ModelRequest>) -> Self {
        Self::from_managed_engine(
            crate::native_runtime::engine::MlxcelServerManagedEngine::legacy_operation_channel(
                command_tx,
            ),
        )
    }

    pub(crate) fn start_generate(
        &self,
        request: MlxcelServerGenerateRequest,
    ) -> Result<(), MlxcelServerFrontendError> {
        let pending = self
            .engine
            .lock()
            .map_err(|_| MlxcelServerFrontendError::WorkerGone)?
            .begin_server_generate(request)?;
        pending.wait_started().map(|_| ()).map_err(Into::into)
    }

    pub(crate) fn prompt_cache_warmup(
        &self,
        tokens: Vec<i32>,
        ctx: crate::server::PromptCacheRequestContext,
    ) -> Result<(), MlxcelServerFrontendError> {
        self.engine
            .lock()
            .map_err(|_| MlxcelServerFrontendError::WorkerGone)?
            .prompt_cache_warmup(tokens, ctx)
            .map_err(Into::into)
    }

    pub(crate) fn shutdown(&self) -> bool {
        self.engine.lock().is_ok_and(|mut engine| engine.shutdown())
    }
}

#[derive(Clone)]
pub(crate) struct MlxcelServerProjection {
    state: Arc<Mutex<ServerProjectionState>>,
}

impl MlxcelServerProjection {
    pub(crate) fn new(
        response_tx: mpsc::Sender<GenerateEvent>,
        stop_sequences: Vec<String>,
    ) -> (Self, MlxcelServerNativeSink) {
        let state = Arc::new(Mutex::new(ServerProjectionState::new(
            response_tx,
            stop_sequences,
        )));
        (
            Self {
                state: Arc::clone(&state),
            },
            MlxcelServerNativeSink { state },
        )
    }

    pub(crate) fn bind_sequence(
        &self,
        prompt_tokens: usize,
        max_tokens: usize,
        cached_tokens: usize,
        created_at: Instant,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.prompt_tokens = prompt_tokens;
            state.max_tokens = max_tokens;
            state.cached_tokens = cached_tokens;
            state.created_at = Some(created_at);
        }
    }

    pub(crate) fn queue_logprobs(&self, logprobs: Option<TokenLogprobData>) {
        if let Ok(mut state) = self.state.lock() {
            state.pending_logprobs.push_back(logprobs);
        }
    }

    pub(crate) fn matched_stop(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.stop_matcher.matched().map(str::to_owned))
    }
}

struct ServerProjectionState {
    response_tx: mpsc::Sender<GenerateEvent>,
    stop_matcher: StopMatcher,
    pending_logprobs: VecDeque<Option<TokenLogprobData>>,
    started: Option<NativeSequenceIdentity>,
    created_at: Option<Instant>,
    first_token_time: Option<Instant>,
    prompt_tokens: usize,
    max_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    emitted_text: String,
    terminal: bool,
}

impl ServerProjectionState {
    fn new(response_tx: mpsc::Sender<GenerateEvent>, stop_sequences: Vec<String>) -> Self {
        Self {
            response_tx,
            stop_matcher: StopMatcher::new(stop_sequences),
            pending_logprobs: VecDeque::new(),
            started: None,
            created_at: None,
            first_token_time: None,
            prompt_tokens: 0,
            max_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            emitted_text: String::new(),
            terminal: false,
        }
    }
}

/// Native sink used by the existing server. It is deliberately private: the
/// neutral public sink never mentions HTTP, SSE, or server events.
pub(crate) struct MlxcelServerNativeSink {
    state: Arc<Mutex<ServerProjectionState>>,
}

impl NativeGenerateSink for MlxcelServerNativeSink {
    fn started(&mut self, sequence: NativeSequenceIdentity) -> Result<(), NativeRuntimeError> {
        let mut state = self.state.lock().map_err(|_| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                "server projection state is unavailable",
            )
        })?;
        if state.started.replace(sequence).is_some() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::StaleHandle,
                "server projection received a duplicate started handshake",
            ));
        }
        Ok(())
    }

    fn accepted(&mut self, chunk: NativeAcceptedChunk) -> NativeContinuationDirective {
        let Ok(mut state) = self.state.lock() else {
            return NativeContinuationDirective::Fail {
                consumed_units: 0,
                error: NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Infrastructure,
                    "server projection state is unavailable",
                ),
            };
        };
        if state.started != Some(chunk.sequence) {
            return NativeContinuationDirective::Fail {
                consumed_units: 0,
                error: NativeRuntimeError::new(
                    NativeRuntimeErrorKind::StaleHandle,
                    "server accepted output disagrees with its started sequence",
                ),
            };
        }
        for (index, unit) in chunk.units.iter().enumerate() {
            let text = match std::str::from_utf8(&unit.stable_bytes) {
                Ok(text) => text,
                Err(_) => {
                    return NativeContinuationDirective::Fail {
                        consumed_units: index,
                        error: NativeRuntimeError::new(
                            NativeRuntimeErrorKind::Tokenizer,
                            "native stable output is not valid UTF-8",
                        ),
                    };
                }
            };
            state.completion_tokens = state.completion_tokens.saturating_add(1);
            let logprobs = state.pending_logprobs.pop_front().flatten();
            let output = state.stop_matcher.push(text);
            if !output.emit.is_empty() {
                state.emitted_text.push_str(&output.emit);
                let event = match logprobs {
                    Some(value) => GenerateEvent::TokenWithLogprobs(output.emit, value),
                    None => GenerateEvent::Token(output.emit),
                };
                let _ = state.response_tx.send(event);
            }
            if output.matched.is_some() {
                return NativeContinuationDirective::Stop {
                    consumed_units: index + 1,
                };
            }
        }
        NativeContinuationDirective::Continue {
            constraint: NativeConstraintTransition::Unchanged,
        }
    }

    fn physical(&mut self, event: NativePhysicalEvent) {
        let NativePhysicalEvent::Prefill {
            evaluated_tokens: _,
            reused_tokens,
        } = event
        else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.first_token_time.is_some() {
            return;
        }
        let now = Instant::now();
        state.first_token_time = Some(now);
        let prompt_ms = state
            .created_at
            .map(|created| now.duration_since(created).as_millis() as u64)
            .unwrap_or(0);
        let _ = state.response_tx.send(GenerateEvent::Prefill(PrefillStats {
            prompt_tokens: state.prompt_tokens,
            cached_tokens: reused_tokens as usize,
            prompt_ms,
        }));
    }

    fn terminal(&mut self, terminal: NativePhysicalTerminal) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.terminal {
            return;
        }
        state.terminal = true;
        match terminal {
            NativePhysicalTerminal::Failed(error) => {
                let _ = state
                    .response_tx
                    .send(GenerateEvent::Error(error.message().to_string()));
            }
            NativePhysicalTerminal::Cancelled if state.first_token_time.is_none() => {
                let _ = state.response_tx.send(GenerateEvent::Error(
                    "Request cancelled: client disconnected".to_string(),
                ));
            }
            NativePhysicalTerminal::Completed { reason } => {
                finish_server_projection(&mut state, reason);
            }
            NativePhysicalTerminal::Cancelled => {
                finish_server_projection(&mut state, NativeStopReason::Requested);
            }
        }
    }
}

fn finish_server_projection(state: &mut ServerProjectionState, reason: NativeStopReason) {
    if !state.stop_matcher.has_matched() {
        let held = state.stop_matcher.flush();
        if !held.is_empty() {
            state.emitted_text.push_str(&held);
            let _ = state.response_tx.send(GenerateEvent::Token(held));
        }
    }
    let start = state.created_at.unwrap_or_else(Instant::now);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let prompt_eval_ms = state
        .first_token_time
        .map(|time| time.duration_since(start).as_millis() as u64)
        .unwrap_or(elapsed_ms);
    let mut result = build_generation_result_with_cache(
        std::mem::take(&mut state.emitted_text),
        state.prompt_tokens,
        state.completion_tokens,
        elapsed_ms,
        prompt_eval_ms,
        state.max_tokens,
        state.cached_tokens,
    );
    if let Some(word) = state.stop_matcher.matched() {
        result.finish_reason = "stop".to_string();
        result.stop_kind = StopKind::Word(word.to_string());
    } else if matches!(reason, NativeStopReason::Limit) {
        result.finish_reason = "length".to_string();
        result.stop_kind = StopKind::Limit;
    }
    let _ = state.response_tx.send(GenerateEvent::Done(result));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_runtime::{
        NativeAcceptedSource, NativeAcceptedUnit, NativePrefixFacts, NativeResumeCapability,
        NativeUsageDelta,
    };
    use crate::server::GenerationResult;

    fn project_units(
        max_tokens: usize,
        stop_sequences: Vec<String>,
        pieces: &[&str],
        source: NativeAcceptedSource,
        reason: NativeStopReason,
    ) -> (NativeContinuationDirective, GenerationResult) {
        let (tx, rx) = mpsc::channel();
        let (projection, mut sink) = MlxcelServerProjection::new(tx, stop_sequences);
        projection.bind_sequence(2, max_tokens, 0, Instant::now());
        let identity = NativeSequenceIdentity {
            request_id: 1,
            sequence_id: 2,
            worker_epoch: 3,
        };
        sink.started(identity).unwrap();
        let directive = sink.accepted(NativeAcceptedChunk {
            sequence: identity,
            units: pieces
                .iter()
                .enumerate()
                .map(|(index, piece)| NativeAcceptedUnit {
                    token_id: index as i32,
                    stable_bytes: Arc::from(piece.as_bytes()),
                    output_step: index as u64,
                    source,
                })
                .collect(),
            usage: NativeUsageDelta {
                output_tokens_accepted: pieces.len() as u64,
                ..NativeUsageDelta::default()
            },
            prefix: NativePrefixFacts {
                adopted_prompt_tokens: 0,
                resume: NativeResumeCapability::WholeChunkOnly,
            },
        });
        sink.terminal(NativePhysicalTerminal::Completed { reason });
        let mut done = None;
        for event in rx.try_iter() {
            match event {
                GenerateEvent::Done(result) => done = Some(result),
                GenerateEvent::Error(error) => panic!("unexpected projection error: {error}"),
                _ => {}
            }
        }
        (directive, done.expect("terminal result"))
    }

    #[test]
    fn sink_is_the_stable_bytes_to_server_event_projection() {
        let (tx, rx) = mpsc::channel();
        let (projection, mut sink) = MlxcelServerProjection::new(tx, vec!["END".to_string()]);
        projection.bind_sequence(2, 8, 0, Instant::now());
        let identity = NativeSequenceIdentity {
            request_id: 1,
            sequence_id: 2,
            worker_epoch: 3,
        };
        sink.started(identity).unwrap();
        let chunk = NativeAcceptedChunk {
            sequence: identity,
            units: vec![NativeAcceptedUnit {
                token_id: 4,
                stable_bytes: Arc::from(&b"hello"[..]),
                output_step: 0,
                source: NativeAcceptedSource::Ordinary,
            }],
            usage: NativeUsageDelta {
                output_tokens_accepted: 1,
                ..NativeUsageDelta::default()
            },
            prefix: NativePrefixFacts {
                adopted_prompt_tokens: 0,
                resume: NativeResumeCapability::WholeChunkOnly,
            },
        };
        assert!(matches!(
            sink.accepted(chunk),
            NativeContinuationDirective::Continue { .. }
        ));
        assert!(matches!(rx.recv().unwrap(), GenerateEvent::Token(text) if text == "hello"));
    }

    #[test]
    fn terminal_usage_matches_zero_based_limit_eos_stop_and_speculative_units() {
        for max_tokens in [1, 16] {
            let pieces = vec!["x"; max_tokens];
            let (directive, result) = project_units(
                max_tokens,
                Vec::new(),
                &pieces,
                NativeAcceptedSource::Ordinary,
                NativeStopReason::Limit,
            );
            assert!(matches!(
                directive,
                NativeContinuationDirective::Continue { .. }
            ));
            assert_eq!(result.completion_tokens, max_tokens);
            assert_eq!(result.finish_reason, "length");
            assert_eq!(result.stop_kind, StopKind::Limit);
        }

        let (_, eos) = project_units(
            16,
            Vec::new(),
            &["a", "b"],
            NativeAcceptedSource::Ordinary,
            NativeStopReason::Eos,
        );
        assert_eq!(eos.completion_tokens, 2);
        assert_eq!(eos.finish_reason, "stop");
        assert_eq!(eos.stop_kind, StopKind::Eos);

        let (stop_directive, stop) = project_units(
            16,
            vec!["END".to_string()],
            &["hello", "END"],
            NativeAcceptedSource::Ordinary,
            NativeStopReason::Requested,
        );
        assert!(matches!(
            stop_directive,
            NativeContinuationDirective::Stop { consumed_units: 2 }
        ));
        assert_eq!(stop.text, "hello");
        assert_eq!(stop.completion_tokens, 2);
        assert_eq!(stop.finish_reason, "stop");
        assert_eq!(stop.stop_kind, StopKind::Word("END".to_string()));

        let (_, speculative) = project_units(
            16,
            Vec::new(),
            &["a", "b", "c"],
            NativeAcceptedSource::SpeculativeTargetAccepted,
            NativeStopReason::Eos,
        );
        assert_eq!(speculative.text, "abc");
        assert_eq!(speculative.completion_tokens, 3);
        assert_eq!(speculative.finish_reason, "stop");
    }
}
