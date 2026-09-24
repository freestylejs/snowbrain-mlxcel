//! Real managed-engine generation against the existing tiny Gemma model fixture.
use super::*;
use crate::native_runtime::builder::NativeWorkerFactory;
use std::sync::mpsc::{self, SyncSender};
use std::time::{Duration, Instant};

fn engine() -> (Box<dyn NativeManagedEngine>, Arc<Source>) {
    let identity = crate::native_runtime::profile::tests::current_native_test_identity();
    let (source, producer) = source(gemma_bytes());
    let mut load = request(
        source,
        "gemma4_unified",
        NativeOperationDriver::AutoregressiveGenerate,
    );
    let binding = &load.binding;
    load.binding = NativeWorkerLoadBinding::new(
        binding.source_identity(),
        binding.report_identity(),
        identity.digest(),
        binding.architecture(),
        true,
        Some(24),
        binding.modalities(),
        binding.tokenizer().clone(),
    )
    .unwrap();
    load.profile = NativeWorkerConfig::try_from_binding(
        &load.binding,
        policy(NativeOperationDriver::AutoregressiveGenerate),
    )
    .unwrap();
    let engine = crate::native_runtime::engine::NativeEngineFactory::new(identity)
        .start_worker(load)
        .unwrap();
    (engine, producer)
}
fn generate(id: u128, text: &str) -> NativeGenerateRequest {
    NativeGenerateRequest {
        request_id: id,
        prompt: NativePromptInput::Text(Arc::from(text)),
        sampling: NativeSamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        },
        limits: NativeGenerationLimits {
            max_new_tokens: 4,
            max_total_tokens: 24,
        },
        eos_token_ids: Arc::from([]),
        output_suppressed_token_ids: Arc::from([]),
        scheduling: NativeSchedulingHints::default(),
        cache: NativeCacheContext {
            namespace: Arc::from("native-t03-test"),
            input_identity: [0; 32],
            allow_read: false,
            allow_write: false,
        },
        constraint: NativeConstraintPlan::None,
        speculation: NativeSpeculationPolicy {
            preference: NativeSpeculationPreference::Disabled,
            minimum_fidelity: NativeSpeculativeFidelity::ExactTargetDistribution,
            allow_ordinary_fallback: false,
        },
    }
}
#[derive(Debug)]
enum Event {
    Start(NativeSequenceIdentity),
    Output(NativeAcceptedChunk),
    Terminal(NativePhysicalTerminal),
    Stopped,
}
struct Sink {
    tx: SyncSender<Event>,
    reject_start: bool,
    fail_output: bool,
}
impl NativeGenerateSink for Sink {
    fn started(&mut self, sequence: NativeSequenceIdentity) -> Result<(), NativeRuntimeError> {
        if self.reject_start {
            return Err(NativeRuntimeError::try_new(
                NativeRuntimeErrorKind::SinkRejected,
                "test rejects start",
            )
            .unwrap());
        }
        self.tx.try_send(Event::Start(sequence)).unwrap();
        Ok(())
    }
    fn accepted(&mut self, chunk: NativeAcceptedChunk) -> NativeContinuationDirective {
        chunk.validate().unwrap();
        self.tx.try_send(Event::Output(chunk)).unwrap();
        if self.fail_output {
            NativeContinuationDirective::Fail {
                consumed_units: 0,
                error: NativeRuntimeError::try_new(
                    NativeRuntimeErrorKind::SinkRejected,
                    "test fails accepted output",
                )
                .unwrap(),
            }
        } else {
            NativeContinuationDirective::Continue {
                constraint: NativeConstraintTransition::Unchanged,
            }
        }
    }
    fn physical(&mut self, _: NativePhysicalEvent) {}
    fn terminal(&mut self, value: NativePhysicalTerminal) {
        self.tx.try_send(Event::Terminal(value)).unwrap();
    }
    fn physical_stopped(&mut self) {
        self.tx.try_send(Event::Stopped).unwrap();
    }
}
fn sink(
    reject_start: bool,
    fail_output: bool,
) -> (Box<dyn NativeGenerateSink>, mpsc::Receiver<Event>) {
    let (tx, rx) = mpsc::sync_channel(16);
    (
        Box::new(Sink {
            tx,
            reject_start,
            fail_output,
        }),
        rx,
    )
}
fn drained(engine: &dyn NativeManagedEngine) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let usage = engine.memory_snapshot();
        if usage.active_sequences == 0 && usage.queued_operations == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "native physical sequence did not drain"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn admission_from_a_different_native_worker_is_rejected_before_preparation() {
    let (mut first, _) = engine();
    let (mut second, _) = engine();
    let admission = first.admit_generate(&generate(10, "hello")).unwrap();
    assert!(
        second.prepare_generate(admission).is_err(),
        "foreign admission must not reserve on another scheduler"
    );
    drained(&*first);
    drained(&*second);
}

#[test]
fn prepared_capability_from_another_worker_cannot_be_activated() {
    let (mut first, _) = engine();
    let (mut second, _) = engine();
    let admission = first.admit_generate(&generate(17, "hello")).unwrap();
    let prepared = first.prepare_generate(admission).unwrap();
    let (sink, rx) = sink(false, false);
    assert!(second.start_generate(prepared, sink).is_err());
    assert!(
        rx.try_recv().is_err(),
        "foreign activation must not call a sink"
    );
    drained(&*first);
    drained(&*second);
}

#[test]
fn invalid_plain_text_is_rejected_before_physical_preparation_without_package_reads() {
    let (mut engine, source) = engine();
    let before = source.reads.lock().unwrap().clone();
    for text in [
        "",
        "abcdefghijklmnopqrstuvwxyz",
        "<literal-tool-marker>too-long",
    ] {
        assert!(engine.admit_generate(&generate(18, text)).is_err());
        drained(&*engine);
    }
    assert_eq!(*source.reads.lock().unwrap(), before);
}

#[test]
fn prepare_holds_actual_sequence_without_dispatch_and_drop_releases_it() {
    let (mut engine, source) = engine();
    let before = source.reads.lock().unwrap().clone();
    let request = generate(11, "a 안녕");
    let admission = engine.admit_generate(&request).unwrap();
    let NativePromptInput::TokenIds(ids) = &admission.request().prompt else {
        panic!("unresolved prompt");
    };
    let expected = crate::tokenizer::MlxcelTokenizer::stub_gemma_sequence_byte_fallback()
        .encode("a 안녕", false)
        .unwrap();
    assert_eq!(
        ids.as_ref(),
        expected.iter().map(|id| *id as i32).collect::<Vec<_>>()
    );
    let id = admission.reservation_id();
    let prepared = engine.prepare_generate(admission).unwrap();
    assert_eq!(prepared.reservation_id(), id);
    assert_eq!(prepared.sequence().request_id, 11);
    let idle = engine.memory_snapshot();
    assert_eq!(idle.active_sequences, 0);
    assert_eq!(idle.queued_operations, 1);
    assert!(
        engine.admit_generate(&request).is_err(),
        "live request cannot be duplicated"
    );
    drop(prepared);
    drained(&*engine);
    let next = engine.admit_generate(&request).unwrap();
    assert_ne!(next.reservation_id(), id);
    drop(next);
    assert_eq!(
        *source.reads.lock().unwrap(),
        before,
        "warm admission must not read package artifacts"
    );
}

#[test]
fn real_generation_streams_then_reports_native_cleanup_and_reuses_the_worker() {
    let (mut engine, source) = engine();
    let reads = source.reads.lock().unwrap().clone();
    let worker = engine.descriptor().worker_epoch;
    for id in [12, 13] {
        let admission = engine.admit_generate(&generate(id, "hello")).unwrap();
        let prepared = engine.prepare_generate(admission).unwrap();
        let expected = prepared.sequence();
        let (sink, rx) = sink(false, false);
        assert!(rx.try_recv().is_err(), "no sink callback before activation");
        let handle = engine.start_generate(prepared, sink).unwrap();
        assert_eq!(handle.worker_epoch(), worker);
        assert!(
            matches!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), Event::Start(route) if route == expected)
        );
        let mut accepted = 0;
        let mut bytes = 0;
        let mut terminal = false;
        loop {
            match rx.recv_timeout(Duration::from_secs(5)).unwrap() {
                Event::Output(chunk) => {
                    assert!(!terminal);
                    accepted += chunk.usage.output_tokens_accepted;
                    bytes += chunk
                        .units
                        .iter()
                        .map(|unit| unit.stable_bytes.len())
                        .sum::<usize>();
                }
                Event::Terminal(value) => {
                    assert!(!terminal);
                    assert!(
                        matches!(value, NativePhysicalTerminal::Completed { .. }),
                        "{value:?}"
                    );
                    terminal = true;
                }
                Event::Stopped => {
                    assert!(terminal, "cleanup must follow real terminal");
                    break;
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert!(
            accepted > 0 && bytes > 0,
            "real model must produce actual accepted text"
        );
        assert!(accepted <= 4);
        drained(&*engine);
    }
    assert_eq!(*source.reads.lock().unwrap(), reads);
}

#[test]
fn rejected_start_and_native_sink_failure_release_actual_physical_state() {
    let (mut engine, _) = engine();
    let admission = engine.admit_generate(&generate(14, "hello")).unwrap();
    let prepared = engine.prepare_generate(admission).unwrap();
    let (sink, rx) = sink(true, false);
    assert!(engine.start_generate(prepared, sink).is_err());
    assert!(matches!(
        rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        Event::Stopped
    ));
    drained(&*engine);
    let admission = engine.admit_generate(&generate(15, "hello")).unwrap();
    let prepared = engine.prepare_generate(admission).unwrap();
    let (sink, rx) = self::sink(false, true);
    engine.start_generate(prepared, sink).unwrap();
    let mut failed = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(5)).unwrap() {
            Event::Terminal(value) => {
                assert!(matches!(value, NativePhysicalTerminal::Failed(_)));
                failed = true;
            }
            Event::Stopped => {
                assert!(failed);
                break;
            }
            _ => {}
        }
    }
    drained(&*engine);
}

#[test]
fn cancelled_prepared_generation_never_decodes_and_has_real_cleanup() {
    let (mut engine, _) = engine();
    let admission = engine.admit_generate(&generate(16, "hello")).unwrap();
    let prepared = engine.prepare_generate(admission).unwrap();
    let handle = NativeExecutionHandle::new(
        prepared.sequence().worker_epoch,
        prepared.reservation_id(),
        NativeExecutionKind::Generate,
    );
    engine.cancel(&handle);
    let (sink, rx) = sink(false, false);
    engine.start_generate(prepared, sink).unwrap();
    let mut cancelled = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(5)).unwrap() {
            Event::Output(_) => panic!("cancelled prepared work decoded output"),
            Event::Terminal(value) => {
                assert!(
                    matches!(value, NativePhysicalTerminal::Cancelled),
                    "{value:?}"
                );
                cancelled = true;
            }
            Event::Stopped => {
                assert!(cancelled);
                break;
            }
            _ => {}
        }
    }
    drained(&*engine);
}
