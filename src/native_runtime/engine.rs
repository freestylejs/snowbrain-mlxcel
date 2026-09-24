// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Sendable client for the existing single-threaded physical scheduler.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use super::builder::{NativeManagedEngine, NativeWorkerFactory, NativeWorkerStartRequest};
use super::operation_driver::NativeOperationDriver;
use super::profile::{NativeBuildIdentity, NativeEngineDescriptor, NativeWorkerHealth};
use super::request::{
    NativeEmbedAdmission, NativeEmbedRequest, NativeExecutionHandle, NativeExecutionKind,
    NativeGenerateAdmission, NativeGenerateRequest, NativePreparedGenerate, NativePreparedRelease,
    NativeReclaimRequest, NativeReclaimResult, NativeReservationGuard, NativeReservationRelease,
};
use super::sink::{NativeEmbeddingSink, NativeGenerateSink, NativeMemorySnapshot};
use super::{NativeRuntimeError, NativeRuntimeErrorKind, NativeWorkerConfig};
use crate::server::model_provider::ModelRequest;
use crate::server::native_frontend::{MlxcelServerFrontend, MlxcelServerGenerateRequest};

static NEXT_WORKER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WORKER_EPOCH: AtomicU64 = AtomicU64::new(1);

fn uses_dedicated_legacy_server_operation(model_type: crate::models::ModelType) -> bool {
    matches!(
        model_type,
        crate::models::ModelType::DiffusionGemma
            | crate::models::ModelType::Llada2Moe
            | crate::models::ModelType::Florence2VLM
    )
}

fn runtime_uses_device_allocator() -> bool {
    cfg!(any(feature = "metal", feature = "cuda"))
}

fn project_worker_health(alive: bool) -> NativeWorkerHealth {
    if alive {
        NativeWorkerHealth::Ready
    } else {
        super::memory::health_after_failure(
            NativeWorkerHealth::Ready,
            super::NativeRecoverability::WorkerReload,
        )
    }
}

pub(crate) fn capture_memory_snapshot(
    active_sequences: u32,
    queued_operations: u32,
) -> NativeMemorySnapshot {
    project_memory_snapshot(
        mlxcel_core::memory::snapshot(),
        active_sequences,
        queued_operations,
        runtime_uses_device_allocator(),
    )
}

fn project_memory_snapshot(
    snapshot: mlxcel_core::memory::MemorySnapshot,
    active_sequences: u32,
    queued_operations: u32,
    device_allocator: bool,
) -> NativeMemorySnapshot {
    let resident = snapshot.active_bytes.saturating_add(snapshot.cache_bytes);
    let peak = snapshot.peak_bytes.max(snapshot.active_bytes);
    let (host_resident_bytes, host_peak_bytes, device_resident_bytes, device_peak_bytes) =
        if device_allocator {
            (0, 0, resident, peak)
        } else {
            (resident, peak, 0, 0)
        };
    NativeMemorySnapshot {
        host_resident_bytes,
        host_peak_bytes,
        device_resident_bytes,
        device_peak_bytes,
        active_sequences,
        queued_operations,
    }
}

pub(crate) fn reclaim_idle_allocator(
    request: NativeReclaimRequest,
    retained_active_sequences: u32,
) -> NativeReclaimResult {
    if retained_active_sequences != 0
        || (request.idle_cache_bytes == 0 && !request.include_compiled_state)
    {
        return NativeReclaimResult {
            retained_active_sequences,
            ..NativeReclaimResult::default()
        };
    }
    let before = mlxcel_core::memory::snapshot();
    mlxcel_core::clear_memory_cache();
    let after = mlxcel_core::memory::snapshot();
    project_reclaim_result(
        before,
        after,
        retained_active_sequences,
        runtime_uses_device_allocator(),
    )
}

fn project_reclaim_result(
    before: mlxcel_core::memory::MemorySnapshot,
    after: mlxcel_core::memory::MemorySnapshot,
    retained_active_sequences: u32,
    device_allocator: bool,
) -> NativeReclaimResult {
    let reclaimed = before.cache_bytes.saturating_sub(after.cache_bytes);
    let (reclaimed_host_bytes, reclaimed_device_bytes) = if device_allocator {
        (0, reclaimed)
    } else {
        (reclaimed, 0)
    };
    NativeReclaimResult {
        reclaimed_host_bytes,
        reclaimed_device_bytes,
        retained_active_sequences,
    }
}

pub(crate) enum NativeSchedulerCommand {
    ResolveGenerate {
        request: Arc<NativeGenerateRequest>,
        reply: mpsc::SyncSender<Result<NativeGenerateResolution, NativeRuntimeError>>,
    },
    PrepareGenerate {
        reservation_id: u128,
        request: Arc<NativeGenerateRequest>,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
        guard: NativeReservationGuard,
        prepared: mpsc::SyncSender<Result<super::NativeSequenceIdentity, NativeRuntimeError>>,
    },
    StartGenerate {
        reservation_id: u128,
        sequence: super::NativeSequenceIdentity,
        sink: Box<dyn NativeGenerateSink + Send>,
        started: mpsc::SyncSender<Result<(), NativeRuntimeError>>,
    },
    ReleasePrepared {
        reservation_id: u128,
        sequence: super::NativeSequenceIdentity,
    },
    StartServerGenerate {
        reservation_id: u128,
        request_id: u128,
        request: MlxcelServerGenerateRequest,
        release: Arc<dyn NativeReservationRelease>,
        started: mpsc::SyncSender<Result<(), NativeRuntimeError>>,
    },
    MemorySnapshot(mpsc::SyncSender<NativeMemorySnapshot>),
    Reclaim(NativeReclaimRequest, mpsc::SyncSender<NativeReclaimResult>),
}

pub(crate) struct NativeGenerateResolution {
    pub prompt_token_ids: Arc<[i32]>,
    pub eos_token_ids: Arc<[i32]>,
    pub output_suppressed_token_ids: Arc<[i32]>,
}

pub(crate) struct MlxcelServerEngineStartRequest {
    pub model_path: std::path::PathBuf,
    pub adapter_path: Option<std::path::PathBuf>,
    pub loaded: Arc<AtomicBool>,
    pub chat_unavailable: Arc<AtomicBool>,
    pub worker_model_id: String,
    pub sched_config: crate::server::model_provider::model_worker::WorkerSchedulerConfig,
    pub batch_metrics: Arc<crate::server::BatchMetrics>,
    pub batch_observability: Arc<crate::server::batch::BatchObservability>,
    pub single_stream_queue_admission: Arc<AtomicBool>,
}

/// Concrete engine used by the existing server frontend.
///
/// Normal decoder-only generation uses the same reservation book, started
/// handshake, native output port, worker decoder, and terminal release as the
/// public managed engine. Dedicated diffusion and encoder-decoder workers are
/// explicitly retained as legacy operation drivers until their operation
/// contracts exist in the public seam.
pub(crate) struct MlxcelServerManagedEngine {
    worker_epoch: u64,
    capacity: usize,
    command_tx: mpsc::Sender<ModelRequest>,
    reservations: Arc<ReservationBook>,
    managed_text: bool,
    next_request_id: u128,
    shutdown_sent: bool,
}

pub(crate) fn spawn_mlxcel_server_managed_engine(
    request: MlxcelServerEngineStartRequest,
) -> (MlxcelServerFrontend, std::thread::JoinHandle<()>) {
    let worker_epoch = NEXT_WORKER_EPOCH.fetch_add(1, Ordering::Relaxed);
    let capacity = request
        .sched_config
        .max_batch_size
        .max(1)
        .saturating_add(request.sched_config.max_queue_depth.max(1));
    let managed_text = !crate::models::get_model_type(&request.model_path)
        .is_ok_and(uses_dedicated_legacy_server_operation);
    let (command_tx, command_rx) = mpsc::channel();
    let worker = crate::server::model_provider::model_worker::spawn_model_worker_with_batch_config(
        request.model_path,
        request.adapter_path,
        command_rx,
        request.loaded,
        request.chat_unavailable,
        request.worker_model_id,
        request.sched_config,
        request.batch_metrics,
        request.batch_observability,
        request.single_stream_queue_admission,
        Some(worker_epoch),
    );
    let engine = MlxcelServerManagedEngine {
        worker_epoch,
        capacity,
        command_tx,
        reservations: Arc::new(ReservationBook::default()),
        managed_text,
        next_request_id: 0,
        shutdown_sent: false,
    };
    (MlxcelServerFrontend::from_managed_engine(engine), worker)
}

/// Start the server embedding operation on the same native-runtime ownership
/// boundary used by the public embedding driver. The operation shape and HTTP
/// replies remain server-private; this function owns the physical worker
/// thread so the server does not create a second embedding engine path.
pub(crate) fn spawn_mlxcel_server_embedding_managed_engine<L>(
    thread_name: &str,
    queue_depth: usize,
    request_timeout: Duration,
    batch_size: usize,
    loader: L,
) -> anyhow::Result<(
    MlxcelServerEmbeddingManagedEngine,
    crate::server::embedding_worker::EmbeddingWorkerInfo,
    std::thread::JoinHandle<()>,
)>
where
    L: FnOnce() -> anyhow::Result<crate::embeddings::LoadedEmbeddingModel> + Send + 'static,
{
    let worker_epoch = NEXT_WORKER_EPOCH.fetch_add(1, Ordering::Relaxed);
    let (command_tx, command_rx) = mpsc::sync_channel(queue_depth.max(1));
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let handle = std::thread::Builder::new()
        .name(thread_name.to_string())
        .spawn(move || {
            let stream = mlxcel_core::streams::new_thread_local_generation_stream();
            mlxcel_core::streams::install_thread_local_default_stream(stream.as_ref());
            let loaded = match loader() {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _ = ready_tx.send(Err(format!("{error:#}")));
                    return;
                }
            };
            let engine = crate::embeddings::EmbeddingEngine::new(loaded, batch_size);
            let info = server_embedding_worker_info(&engine);
            if ready_tx.send(Ok(info)).is_err() {
                return;
            }
            run_embedding_command_loop(&engine, command_rx, stream.as_ref());
        })
        .map_err(|error| anyhow::anyhow!("failed to spawn embedding worker thread: {error}"))?;
    let info = match ready_rx.recv() {
        Ok(Ok(info)) => info,
        Ok(Err(message)) => {
            let _ = handle.join();
            return Err(anyhow::anyhow!(
                "embedding worker failed to load model: {message}"
            ));
        }
        Err(_) => {
            let _ = handle.join();
            return Err(anyhow::anyhow!(
                "embedding worker thread exited before reporting readiness"
            ));
        }
    };
    let engine = MlxcelServerEmbeddingManagedEngine {
        worker_epoch,
        capacity: queue_depth.max(1).saturating_add(1),
        command_tx,
        reservations: Arc::new(ReservationBook::default()),
        next_request_id: AtomicU64::new(0),
        request_timeout,
        info,
        shutdown_sent: AtomicBool::new(false),
    };
    Ok((engine, info, handle))
}

struct MlxcelServerGenerateAdmission {
    reservation_id: u128,
    request_id: u128,
    request: MlxcelServerGenerateRequest,
    guard: NativeReservationGuard,
}

pub(crate) enum MlxcelServerGeneratePending {
    Started(NativeExecutionHandle),
    AwaitingWorker {
        started_rx: mpsc::Receiver<Result<(), NativeRuntimeError>>,
        guard: NativeReservationGuard,
        handle: NativeExecutionHandle,
    },
}

impl MlxcelServerGeneratePending {
    pub(crate) fn wait_started(self) -> Result<NativeExecutionHandle, NativeRuntimeError> {
        match self {
            Self::Started(handle) => Ok(handle),
            Self::AwaitingWorker {
                started_rx,
                guard,
                handle,
            } => {
                started_rx
                    .recv()
                    .map_err(|_| NativeManagedEngineClient::worker_gone())??;
                guard.commit_started();
                Ok(handle)
            }
        }
    }
}

impl MlxcelServerManagedEngine {
    pub(crate) fn legacy_operation_channel(command_tx: mpsc::Sender<ModelRequest>) -> Self {
        Self {
            worker_epoch: NEXT_WORKER_EPOCH.fetch_add(1, Ordering::Relaxed),
            capacity: 1,
            command_tx,
            reservations: Arc::new(ReservationBook::default()),
            managed_text: false,
            next_request_id: 0,
            shutdown_sent: false,
        }
    }

    fn admit_server_generate(
        &mut self,
        request: MlxcelServerGenerateRequest,
    ) -> Result<MlxcelServerGenerateAdmission, NativeRuntimeError> {
        validate_server_generate_request(&request)?;
        self.next_request_id = self.next_request_id.saturating_add(1).max(1);
        let request_id = self.next_request_id;
        let (reservation_id, release) = self.reservations.reserve(request_id, self.capacity)?;
        Ok(MlxcelServerGenerateAdmission {
            reservation_id,
            request_id,
            request,
            guard: NativeReservationGuard::new(reservation_id, release),
        })
    }

    pub(crate) fn begin_server_generate(
        &mut self,
        request: MlxcelServerGenerateRequest,
    ) -> Result<MlxcelServerGeneratePending, NativeRuntimeError> {
        if !self.managed_text {
            self.command_tx
                .send(request.into_legacy_model_request())
                .map_err(|_| NativeManagedEngineClient::worker_gone())?;
            self.next_request_id = self.next_request_id.saturating_add(1).max(1);
            return Ok(MlxcelServerGeneratePending::Started(
                NativeExecutionHandle::new(
                    self.worker_epoch,
                    self.next_request_id,
                    NativeExecutionKind::Generate,
                ),
            ));
        }
        let admission = self.admit_server_generate(request)?;
        let MlxcelServerGenerateAdmission {
            reservation_id,
            request_id,
            request,
            guard,
        } = admission;
        self.reservations
            .bind_cancellation(reservation_id, Arc::clone(&request.cancelled));
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let release: Arc<dyn NativeReservationRelease> = self.reservations.clone();
        self.command_tx
            .send(ModelRequest::Native(
                NativeSchedulerCommand::StartServerGenerate {
                    reservation_id,
                    request_id,
                    request,
                    release,
                    started: started_tx,
                },
            ))
            .map_err(|_| NativeManagedEngineClient::worker_gone())?;
        Ok(MlxcelServerGeneratePending::AwaitingWorker {
            started_rx,
            guard,
            handle: NativeExecutionHandle::new(
                self.worker_epoch,
                reservation_id,
                NativeExecutionKind::Generate,
            ),
        })
    }

    #[cfg(test)]
    fn start_server_generate(
        &mut self,
        request: MlxcelServerGenerateRequest,
    ) -> Result<NativeExecutionHandle, NativeRuntimeError> {
        self.begin_server_generate(request)?.wait_started()
    }

    pub(crate) fn prompt_cache_warmup(
        &mut self,
        tokens: Vec<i32>,
        ctx: crate::server::PromptCacheRequestContext,
    ) -> Result<(), NativeRuntimeError> {
        if tokens.is_empty()
            || tokens.len() > super::MAX_NATIVE_PROMPT_TOKENS
            || tokens.iter().any(|token| *token < 0)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "prompt-cache warmup tokens violate the native bound",
            ));
        }
        self.command_tx
            .send(ModelRequest::PromptCacheWarmup { tokens, ctx })
            .map_err(|_| NativeManagedEngineClient::worker_gone())
    }

    pub(crate) fn shutdown(&mut self) -> bool {
        if self.shutdown_sent {
            return true;
        }
        self.shutdown_sent = true;
        self.command_tx.send(ModelRequest::Shutdown).is_ok()
    }
}

impl Drop for MlxcelServerManagedEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn validate_server_generate_request(
    request: &MlxcelServerGenerateRequest,
) -> Result<(), NativeRuntimeError> {
    let invalid =
        |message| NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidRequest, message);
    if request.prompt.len() > super::MAX_NATIVE_TEXT_BYTES {
        return Err(invalid("server prompt exceeds the native text bound"));
    }
    if request.options.max_tokens == 0
        || request.options.max_tokens > super::MAX_NATIVE_PROMPT_TOKENS
    {
        return Err(invalid(
            "server generation limit is zero or exceeds the native bound",
        ));
    }
    if let Some(tokens) = &request.prompt_token_ids
        && (tokens.is_empty()
            || tokens.len() > super::MAX_NATIVE_PROMPT_TOKENS
            || tokens.iter().any(|token| *token < 0))
    {
        return Err(invalid("server prompt tokens violate the native bound"));
    }
    let media_items = request
        .images
        .len()
        .checked_add(request.audio.len())
        .and_then(|count| count.checked_add(request.videos.len()))
        .ok_or_else(|| invalid("server media count overflowed"))?;
    if media_items > super::MAX_NATIVE_PREPARED_MEDIA_ITEMS {
        return Err(invalid("server media count exceeds the native bound"));
    }
    let inline_bytes = request
        .images
        .iter()
        .chain(&request.audio)
        .try_fold(0usize, |total, item| total.checked_add(item.len()))
        .ok_or_else(|| invalid("server inline media bytes overflowed"))?;
    if inline_bytes > super::MAX_NATIVE_PREPARED_BYTES {
        return Err(invalid(
            "server inline media exceeds the native prepared-input bound",
        ));
    }
    Ok(())
}

enum NativeEngineChannel {
    Generate(mpsc::Sender<ModelRequest>),
    Embed(mpsc::SyncSender<NativeEmbeddingCommand>),
}

enum NativeEmbeddingCommand {
    Resolve {
        inputs: Arc<[super::NativeEmbeddingInput]>,
        reply: mpsc::SyncSender<Result<NativeEmbedResolved, NativeRuntimeError>>,
    },
    Start {
        reservation_id: u128,
        request: Arc<NativeEmbedRequest>,
        sink: Box<dyn NativeEmbeddingSink + Send>,
        release: Arc<dyn NativeReservationRelease>,
        cancelled: Arc<AtomicBool>,
    },
    StartServer {
        reservation_id: u128,
        request: ServerEmbeddingOperation,
        respond: mpsc::Sender<
            Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError>,
        >,
        release: Arc<dyn NativeReservationRelease>,
        cancelled: Arc<AtomicBool>,
    },
    MemorySnapshot(mpsc::SyncSender<NativeMemorySnapshot>),
    Reclaim(NativeReclaimRequest, mpsc::SyncSender<NativeReclaimResult>),
    Shutdown,
}

enum ServerEmbeddingOperation {
    Texts(Vec<String>, crate::embeddings::EmbedOptions),
    Tokens(Vec<Vec<u32>>, crate::embeddings::EmbedOptions),
    Image(
        crate::embeddings::ImageInput,
        crate::embeddings::EmbedOptions,
    ),
}

pub(crate) struct MlxcelServerEmbeddingManagedEngine {
    worker_epoch: u64,
    capacity: usize,
    command_tx: mpsc::SyncSender<NativeEmbeddingCommand>,
    reservations: Arc<ReservationBook>,
    next_request_id: AtomicU64,
    request_timeout: Duration,
    info: crate::server::embedding_worker::EmbeddingWorkerInfo,
    shutdown_sent: AtomicBool,
}

pub(crate) struct MlxcelServerEmbeddingPending {
    reply: mpsc::Receiver<
        Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError>,
    >,
    reservations: Arc<ReservationBook>,
    handle: NativeExecutionHandle,
    request_timeout: Duration,
}

impl MlxcelServerEmbeddingPending {
    pub(crate) fn wait(
        self,
    ) -> Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError> {
        match self.reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = self.reservations.cancel(self.handle.execution_id());
                Err(crate::server::embedding_model::EmbeddingError::Timeout)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(crate::server::embedding_model::EmbeddingError::Internal(
                    "embedding worker thread is no longer running".to_string(),
                ))
            }
        }
    }
}

impl std::fmt::Debug for MlxcelServerEmbeddingManagedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxcelServerEmbeddingManagedEngine")
            .field("worker_epoch", &self.worker_epoch)
            .field("capacity", &self.capacity)
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl MlxcelServerEmbeddingManagedEngine {
    pub(crate) fn embed_texts(
        &self,
        texts: Vec<String>,
        options: crate::embeddings::EmbedOptions,
    ) -> Result<MlxcelServerEmbeddingPending, crate::server::embedding_model::EmbeddingError> {
        let compatible = !self.info.multi_vector
            && options.instruction.is_none()
            && options.dimensions.is_none()
            && matches!(
                options.normalize,
                None | Some(crate::embeddings::EmbdNormalize::NONE)
                    | Some(crate::embeddings::EmbdNormalize::EUCLIDEAN)
            );
        if compatible {
            let inputs = texts
                .iter()
                .cloned()
                .map(|text| super::NativeEmbeddingInput::Text(Arc::from(text)))
                .collect::<Vec<_>>();
            return self.run_native_embedding(inputs, &options);
        }
        self.run_private_embedding(ServerEmbeddingOperation::Texts(texts, options))
    }

    pub(crate) fn embed_tokens(
        &self,
        rows: Vec<Vec<u32>>,
        options: crate::embeddings::EmbedOptions,
    ) -> Result<MlxcelServerEmbeddingPending, crate::server::embedding_model::EmbeddingError> {
        let compatible = !self.info.multi_vector
            && rows.iter().all(|row| row.len() <= self.info.max_length)
            && options.instruction.is_none()
            && options.dimensions.is_none()
            && matches!(
                options.normalize,
                None | Some(crate::embeddings::EmbdNormalize::NONE)
                    | Some(crate::embeddings::EmbdNormalize::EUCLIDEAN)
            );
        if compatible {
            let inputs = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|id| i32::try_from(*id))
                        .collect::<Result<Vec<_>, _>>()
                        .map(|ids| super::NativeEmbeddingInput::TokenIds(Arc::from(ids)))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    crate::server::embedding_model::EmbeddingError::InvalidInput(
                        "embedding token id exceeds the native signed token range".to_string(),
                    )
                })?;
            return self.run_native_embedding(inputs, &options);
        }
        self.run_private_embedding(ServerEmbeddingOperation::Tokens(rows, options))
    }

    pub(crate) fn embed_image(
        &self,
        image: crate::embeddings::ImageInput,
        options: crate::embeddings::EmbedOptions,
    ) -> Result<MlxcelServerEmbeddingPending, crate::server::embedding_model::EmbeddingError> {
        self.run_private_embedding(ServerEmbeddingOperation::Image(image, options))
    }

    fn run_native_embedding(
        &self,
        inputs: Vec<super::NativeEmbeddingInput>,
        options: &crate::embeddings::EmbedOptions,
    ) -> Result<MlxcelServerEmbeddingPending, crate::server::embedding_model::EmbeddingError> {
        let request_id = self
            .next_request_id
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1) as u128;
        let normalization = match options.normalize {
            None => super::NativeEmbeddingNormalization::ModelDefault,
            Some(kind) if kind == crate::embeddings::EmbdNormalize::NONE => {
                super::NativeEmbeddingNormalization::None
            }
            Some(_) => super::NativeEmbeddingNormalization::L2,
        };
        let request = NativeEmbedRequest {
            request_id,
            inputs: Arc::from(inputs),
            limits: super::NativeEmbeddingLimits {
                max_input_tokens: self.info.max_length.min(u32::MAX as usize) as u32,
                max_output_dimensions: self.info.dim.min(u32::MAX as usize) as u32,
            },
            normalization,
        };
        let (admission, prompt_tokens) = admit_server_embedding_request(
            &self.command_tx,
            &self.reservations,
            self.capacity,
            super::MAX_NATIVE_EMBED_INPUTS,
            self.info.dim,
            self.request_timeout,
            &request,
        )?;
        let (respond, reply) = mpsc::channel();
        let sink = Box::new(ServerEmbeddingNativeSink::new(
            respond,
            request.inputs.len(),
            prompt_tokens,
        ));
        let handle = start_embedding_admission(
            self.worker_epoch,
            &self.command_tx,
            &self.reservations,
            admission,
            sink,
        )
        .map_err(server_embedding_error)?;
        Ok(MlxcelServerEmbeddingPending {
            reply,
            reservations: Arc::clone(&self.reservations),
            handle,
            request_timeout: self.request_timeout,
        })
    }

    fn run_private_embedding(
        &self,
        request: ServerEmbeddingOperation,
    ) -> Result<MlxcelServerEmbeddingPending, crate::server::embedding_model::EmbeddingError> {
        validate_server_embedding_operation(&request, &self.info)?;
        let request_id = self
            .next_request_id
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1) as u128;
        if self.reservations.is_at_capacity(self.capacity) {
            return Err(crate::server::embedding_model::EmbeddingError::QueueFull);
        }
        let (reservation_id, release) = self
            .reservations
            .reserve(request_id, self.capacity)
            .map_err(server_embedding_error)?;
        let guard = NativeReservationGuard::new(reservation_id, release);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.reservations
            .bind_cancellation(reservation_id, Arc::clone(&cancelled));
        let (respond, reply) = mpsc::channel();
        let release: Arc<dyn NativeReservationRelease> = self.reservations.clone();
        self.command_tx
            .try_send(NativeEmbeddingCommand::StartServer {
                reservation_id,
                request,
                respond,
                release,
                cancelled,
            })
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => {
                    crate::server::embedding_model::EmbeddingError::QueueFull
                }
                mpsc::TrySendError::Disconnected(_) => {
                    crate::server::embedding_model::EmbeddingError::Internal(
                        "embedding worker thread is no longer running".to_string(),
                    )
                }
            })?;
        guard.commit_started();
        let handle = NativeExecutionHandle::new(
            self.worker_epoch,
            reservation_id,
            NativeExecutionKind::Embed,
        );
        Ok(MlxcelServerEmbeddingPending {
            reply,
            reservations: Arc::clone(&self.reservations),
            handle,
            request_timeout: self.request_timeout,
        })
    }

    pub(crate) fn shutdown(&self) -> bool {
        if self.shutdown_sent.swap(true, Ordering::AcqRel) {
            return true;
        }
        self.command_tx
            .send(NativeEmbeddingCommand::Shutdown)
            .is_ok()
    }
}

impl Drop for MlxcelServerEmbeddingManagedEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct ServerEmbeddingNativeSink {
    respond: Option<
        mpsc::Sender<
            Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError>,
        >,
    >,
    vectors: Vec<Option<crate::embeddings::EmbeddingVector>>,
    prompt_tokens: usize,
    started: bool,
    terminal: bool,
}

impl ServerEmbeddingNativeSink {
    fn new(
        respond: mpsc::Sender<
            Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError>,
        >,
        input_count: usize,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            respond: Some(respond),
            vectors: vec![None; input_count],
            prompt_tokens,
            started: false,
            terminal: false,
        }
    }

    fn reply(
        &mut self,
        result: Result<
            crate::embeddings::EmbedReply,
            crate::server::embedding_model::EmbeddingError,
        >,
    ) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if let Some(respond) = self.respond.take() {
            let _ = respond.send(result);
        }
    }
}

impl NativeEmbeddingSink for ServerEmbeddingNativeSink {
    fn started(&mut self) -> Result<(), NativeRuntimeError> {
        if self.started {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::StaleHandle,
                "server embedding sink received duplicate started handshake",
            ));
        }
        self.started = true;
        Ok(())
    }

    fn vector(&mut self, output: super::NativeEmbeddingOutput) -> Result<(), NativeRuntimeError> {
        output.validate()?;
        let index = usize::try_from(output.input_index).map_err(|_| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidOutput,
                "server embedding output index does not fit this host",
            )
        })?;
        let slot = self.vectors.get_mut(index).ok_or_else(|| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidOutput,
                "server embedding output index is outside the request",
            )
        })?;
        if slot.is_some() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidOutput,
                "server embedding output index was delivered twice",
            ));
        }
        *slot = Some(crate::embeddings::EmbeddingVector {
            values: output.vector.to_vec(),
            shape: vec![output.dimension as usize],
        });
        Ok(())
    }

    fn terminal(&mut self, terminal: super::NativePhysicalTerminal) {
        match terminal {
            super::NativePhysicalTerminal::Completed { .. } => {
                let vectors = self
                    .vectors
                    .iter_mut()
                    .map(Option::take)
                    .collect::<Option<Vec<_>>>();
                match vectors {
                    Some(vectors) => self.reply(Ok(crate::embeddings::EmbedReply {
                        vectors,
                        prompt_tokens: self.prompt_tokens,
                    })),
                    None => self.reply(Err(
                        crate::server::embedding_model::EmbeddingError::Internal(
                            "native embedding terminal arrived before every vector".to_string(),
                        ),
                    )),
                }
            }
            super::NativePhysicalTerminal::Cancelled => {
                self.reply(Err(crate::server::embedding_model::EmbeddingError::Timeout))
            }
            super::NativePhysicalTerminal::Failed(error) => {
                self.reply(Err(server_embedding_error(error)))
            }
        }
    }
}

fn server_embedding_error(
    error: NativeRuntimeError,
) -> crate::server::embedding_model::EmbeddingError {
    match error.kind() {
        NativeRuntimeErrorKind::InvalidRequest | NativeRuntimeErrorKind::UnsupportedCapability => {
            crate::server::embedding_model::EmbeddingError::InvalidInput(error.to_string())
        }
        NativeRuntimeErrorKind::ResourceLimit => {
            crate::server::embedding_model::EmbeddingError::QueueFull
        }
        _ => crate::server::embedding_model::EmbeddingError::Internal(error.to_string()),
    }
}

fn validate_server_embedding_operation(
    request: &ServerEmbeddingOperation,
    info: &crate::server::embedding_worker::EmbeddingWorkerInfo,
) -> Result<(), crate::server::embedding_model::EmbeddingError> {
    use crate::server::embedding_model::EmbeddingError;

    let options = match request {
        ServerEmbeddingOperation::Texts(texts, options) => {
            if texts.is_empty() || texts.len() > super::MAX_NATIVE_EMBED_INPUTS {
                return Err(EmbeddingError::InvalidInput(
                    "embedding input count is zero or exceeds the native bound".to_string(),
                ));
            }
            let mut bytes = 0usize;
            for (index, text) in texts.iter().enumerate() {
                if text.is_empty() || text.len() > super::MAX_NATIVE_TEXT_BYTES {
                    return Err(EmbeddingError::InvalidInput(format!(
                        "input[{index}] is empty or exceeds the native text bound"
                    )));
                }
                bytes = bytes.checked_add(text.len()).ok_or_else(|| {
                    EmbeddingError::InvalidInput("embedding text bytes overflowed".to_string())
                })?;
            }
            if bytes > super::MAX_NATIVE_EMBED_AGGREGATE_BYTES {
                return Err(EmbeddingError::InvalidInput(
                    "embedding text inputs exceed the aggregate native bound".to_string(),
                ));
            }
            options
        }
        ServerEmbeddingOperation::Tokens(rows, options) => {
            if rows.is_empty() || rows.len() > super::MAX_NATIVE_EMBED_INPUTS {
                return Err(EmbeddingError::InvalidInput(
                    "embedding input count is zero or exceeds the native bound".to_string(),
                ));
            }
            let mut tokens = 0usize;
            for (index, row) in rows.iter().enumerate() {
                if row.is_empty() || row.len() > super::MAX_NATIVE_PROMPT_TOKENS {
                    return Err(EmbeddingError::InvalidInput(format!(
                        "input[{index}] is empty or exceeds the native token bound"
                    )));
                }
                tokens = tokens.checked_add(row.len()).ok_or_else(|| {
                    EmbeddingError::InvalidInput("embedding token count overflowed".to_string())
                })?;
            }
            if tokens > super::MAX_NATIVE_EMBED_AGGREGATE_TOKENS {
                return Err(EmbeddingError::InvalidInput(
                    "embedding token inputs exceed the aggregate native bound".to_string(),
                ));
            }
            options
        }
        ServerEmbeddingOperation::Image(image, options) => {
            if !info.supports_images {
                return Err(EmbeddingError::InvalidInput(
                    "this embedding model does not accept image inputs".to_string(),
                ));
            }
            let pixels = usize::try_from(image.image.width())
                .ok()
                .and_then(|width| {
                    usize::try_from(image.image.height())
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or_else(|| {
                    EmbeddingError::InvalidInput("embedding image size overflowed".to_string())
                })?;
            if pixels > super::MAX_NATIVE_PREPARED_BYTES {
                return Err(EmbeddingError::InvalidInput(
                    "embedding image exceeds the native prepared-input bound".to_string(),
                ));
            }
            options
        }
    };
    if options
        .instruction
        .as_ref()
        .is_some_and(|instruction| instruction.len() > super::MAX_NATIVE_TEXT_BYTES)
    {
        return Err(EmbeddingError::InvalidInput(
            "embedding instruction exceeds the native text bound".to_string(),
        ));
    }
    if options
        .dimensions
        .is_some_and(|dimensions| dimensions == 0 || dimensions > info.dim)
    {
        return Err(EmbeddingError::InvalidInput(format!(
            "`dimensions` must be between 1 and {} for this model",
            info.dim
        )));
    }
    Ok(())
}

struct NativeEmbedResolved {
    input_tokens: Vec<usize>,
    dimension: usize,
    multi_vector: bool,
}

fn admit_embedding_request(
    command_tx: &mpsc::SyncSender<NativeEmbeddingCommand>,
    reservations: &Arc<ReservationBook>,
    capacity: usize,
    max_inputs: usize,
    max_dimensions: usize,
    request: &NativeEmbedRequest,
) -> Result<(NativeEmbedAdmission, usize), NativeRuntimeError> {
    request.validate()?;
    if request.inputs.len() > max_inputs {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::ResourceLimit,
            "embedding input count exceeds the resident worker profile",
        ));
    }
    if reservations.is_at_capacity(capacity) {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::ResourceLimit,
            "native worker admission capacity is exhausted",
        ));
    }
    let reply_rx = enqueue_embedding_resolve(command_tx, &request.inputs)?;
    let resolved = reply_rx
        .recv()
        .map_err(|_| NativeManagedEngineClient::worker_gone())??;
    complete_embedding_admission(reservations, capacity, max_dimensions, request, resolved)
}

fn admit_server_embedding_request(
    command_tx: &mpsc::SyncSender<NativeEmbeddingCommand>,
    reservations: &Arc<ReservationBook>,
    capacity: usize,
    max_inputs: usize,
    max_dimensions: usize,
    request_timeout: Duration,
    request: &NativeEmbedRequest,
) -> Result<(NativeEmbedAdmission, usize), crate::server::embedding_model::EmbeddingError> {
    request.validate().map_err(server_embedding_error)?;
    if request.inputs.len() > max_inputs {
        return Err(crate::server::embedding_model::EmbeddingError::QueueFull);
    }
    if reservations.is_at_capacity(capacity) {
        return Err(crate::server::embedding_model::EmbeddingError::QueueFull);
    }
    let reply_rx =
        enqueue_embedding_resolve(command_tx, &request.inputs).map_err(server_embedding_error)?;
    let resolved = match reply_rx.recv_timeout(request_timeout) {
        Ok(result) => result.map_err(server_embedding_error)?,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            return Err(crate::server::embedding_model::EmbeddingError::Timeout);
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err(crate::server::embedding_model::EmbeddingError::Internal(
                "embedding worker thread is no longer running".to_string(),
            ));
        }
    };
    complete_embedding_admission(reservations, capacity, max_dimensions, request, resolved)
        .map_err(server_embedding_error)
}

fn enqueue_embedding_resolve(
    command_tx: &mpsc::SyncSender<NativeEmbeddingCommand>,
    inputs: &Arc<[super::NativeEmbeddingInput]>,
) -> Result<mpsc::Receiver<Result<NativeEmbedResolved, NativeRuntimeError>>, NativeRuntimeError> {
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    command_tx
        .try_send(NativeEmbeddingCommand::Resolve {
            inputs: Arc::clone(inputs),
            reply: reply_tx,
        })
        .map_err(|error| match error {
            mpsc::TrySendError::Full(_) => NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "native embedding operation queue is full",
            ),
            mpsc::TrySendError::Disconnected(_) => NativeManagedEngineClient::worker_gone(),
        })?;
    Ok(reply_rx)
}

fn complete_embedding_admission(
    reservations: &Arc<ReservationBook>,
    capacity: usize,
    max_dimensions: usize,
    request: &NativeEmbedRequest,
    resolved: NativeEmbedResolved,
) -> Result<(NativeEmbedAdmission, usize), NativeRuntimeError> {
    if resolved.multi_vector {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedCapability,
            "multi-vector embedding output is not representable by this native sink",
        ));
    }
    if resolved.dimension > request.limits.max_output_dimensions as usize
        || resolved.dimension > max_dimensions
    {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::ResourceLimit,
            "embedding dimension exceeds the request or worker profile",
        ));
    }
    let prompt_tokens = resolved.input_tokens.iter().sum();
    let validated = request.validated_with_input_tokens(&resolved.input_tokens)?;
    let (reservation_id, release) = reservations.reserve(request.request_id, capacity)?;
    Ok((
        NativeEmbedAdmission::new(reservation_id, validated, release),
        prompt_tokens,
    ))
}

fn start_embedding_admission(
    worker_epoch: u64,
    command_tx: &mpsc::SyncSender<NativeEmbeddingCommand>,
    reservations: &Arc<ReservationBook>,
    admission: NativeEmbedAdmission,
    mut sink: Box<dyn NativeEmbeddingSink + Send>,
) -> Result<NativeExecutionHandle, NativeRuntimeError> {
    let (reservation_id, request, guard) = admission.into_parts();
    let cancelled = Arc::new(AtomicBool::new(false));
    reservations.bind_cancellation(reservation_id, Arc::clone(&cancelled));
    sink.started()?;
    let handle =
        NativeExecutionHandle::new(worker_epoch, reservation_id, NativeExecutionKind::Embed);
    let release: Arc<dyn NativeReservationRelease> = reservations.clone();
    let command = NativeEmbeddingCommand::Start {
        reservation_id,
        request,
        sink,
        release,
        cancelled,
    };
    match command_tx.send(command) {
        Ok(()) => {
            guard.commit_started();
            Ok(handle)
        }
        Err(mpsc::SendError(NativeEmbeddingCommand::Start { mut sink, .. })) => {
            sink.terminal(super::NativePhysicalTerminal::Failed(
                NativeManagedEngineClient::worker_gone(),
            ));
            Ok(handle)
        }
        Err(_) => unreachable!("embedding start send returned a different command"),
    }
}

#[derive(Default)]
struct ReservationBook {
    inner: Mutex<ReservationState>,
    #[cfg(test)]
    release_calls: std::sync::atomic::AtomicUsize,
}

#[derive(Default)]
struct ReservationState {
    next: u128,
    reservations: HashSet<u128>,
    request_ids: HashSet<u128>,
    cancellation: HashMap<u128, Arc<std::sync::atomic::AtomicBool>>,
    request_by_reservation: HashMap<u128, u128>,
}

impl ReservationBook {
    fn is_at_capacity(&self, capacity: usize) -> bool {
        self.inner
            .lock()
            .map_or(true, |state| state.reservations.len() >= capacity)
    }

    fn reserve(
        self: &Arc<Self>,
        request_id: u128,
        capacity: usize,
    ) -> Result<(u128, Arc<dyn NativeReservationRelease>), NativeRuntimeError> {
        let mut state = self.inner.lock().map_err(|_| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                "native reservation state is unavailable",
            )
        })?;
        if state.reservations.len() >= capacity {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "native worker admission capacity is exhausted",
            ));
        }
        if !state.request_ids.insert(request_id) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidRequest,
                "native request id is already reserved or active",
            ));
        }
        state.next = state.next.saturating_add(1).max(1);
        let reservation_id = state.next;
        state.reservations.insert(reservation_id);
        state
            .request_by_reservation
            .insert(reservation_id, request_id);
        let release: Arc<dyn NativeReservationRelease> = self.clone();
        Ok((reservation_id, release))
    }

    fn bind_cancellation(&self, reservation_id: u128, flag: Arc<std::sync::atomic::AtomicBool>) {
        if let Ok(mut state) = self.inner.lock() {
            state.cancellation.insert(reservation_id, flag);
        }
    }

    fn cancel(&self, reservation_id: u128) -> bool {
        let Ok(state) = self.inner.lock() else {
            return false;
        };
        let Some(flag) = state.cancellation.get(&reservation_id) else {
            return false;
        };
        flag.store(true, Ordering::Release);
        true
    }
}

impl NativeReservationRelease for ReservationBook {
    fn release(&self, reservation_id: u128) {
        #[cfg(test)]
        self.release_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut state) = self.inner.lock() {
            state.reservations.remove(&reservation_id);
            state.cancellation.remove(&reservation_id);
            if let Some(request_id) = state.request_by_reservation.remove(&reservation_id) {
                state.request_ids.remove(&request_id);
            }
        }
    }
}

pub(crate) struct NativeEngineFactory {
    build_identity: NativeBuildIdentity,
}

impl NativeEngineFactory {
    pub(crate) fn new(build_identity: NativeBuildIdentity) -> Self {
        Self { build_identity }
    }
}

impl NativeWorkerFactory for NativeEngineFactory {
    fn start_worker(
        &self,
        request: NativeWorkerStartRequest,
    ) -> Result<Box<dyn NativeManagedEngine>, NativeRuntimeError> {
        validate_implemented_worker_contract(
            request.profile.operation_driver(),
            &request.profile.policy().prepared_multimodal,
        )?;
        let worker_id = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
        let worker_epoch = NEXT_WORKER_EPOCH.fetch_add(1, Ordering::Relaxed);
        let profile_identity = request.profile.identity()?;
        let descriptor = NativeEngineDescriptor {
            worker_id,
            worker_epoch,
            build_identity: self.build_identity.digest(),
            package_identity: request.profile.package_identity(),
            profile_identity,
            health: NativeWorkerHealth::Ready,
        };
        let profile = request.profile.clone();
        let capacity = usize::try_from(profile.policy().batch.max_active_sequences)
            .unwrap_or(usize::MAX)
            .saturating_add(
                usize::try_from(profile.policy().batch.max_queued_operations).unwrap_or(usize::MAX),
            );
        let alive = Arc::new(AtomicBool::new(true));
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread_alive = Arc::clone(&alive);
        let channel = match request.profile.operation_driver() {
            NativeOperationDriver::AutoregressiveGenerate
            | NativeOperationDriver::EncoderDecoderGenerate => {
                let (tx, rx) = mpsc::channel();
                std::thread::Builder::new()
                    .name(format!("mlxcel-native-{worker_id}"))
                    .spawn(move || {
                        let result =
                            start_generation_scheduler(request, rx, worker_epoch, ready_tx);
                        thread_alive.store(false, Ordering::Release);
                        if let Err(error) = result {
                            tracing::error!("native managed worker stopped: {error}");
                        }
                    })
                    .map_err(thread_start_error)?;
                NativeEngineChannel::Generate(tx)
            }
            NativeOperationDriver::Embedding => {
                let (tx, rx) = mpsc::sync_channel(capacity.max(1));
                std::thread::Builder::new()
                    .name(format!("mlxcel-native-{worker_id}"))
                    .spawn(move || {
                        let result = start_embedding_worker(request, rx, ready_tx);
                        thread_alive.store(false, Ordering::Release);
                        if let Err(error) = result {
                            tracing::error!("native managed embedding worker stopped: {error}");
                        }
                    })
                    .map_err(thread_start_error)?;
                NativeEngineChannel::Embed(tx)
            }
            NativeOperationDriver::SpeechToText => {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::UnsupportedOperation,
                    "speech-to-text has no managed lifecycle",
                ));
            }
        };
        let loaded = ready_rx.recv().map_err(|_| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                "native worker exited before its load handshake",
            )
        })??;
        Ok(Box::new(NativeManagedEngineClient {
            descriptor,
            profile,
            capacity,
            channel,
            reservations: Arc::new(ReservationBook::default()),
            alive,
            loaded,
        }))
    }
}

fn validate_implemented_worker_contract(
    driver: NativeOperationDriver,
    prepared_multimodal: &super::NativePreparedMultimodalClass,
) -> Result<(), NativeRuntimeError> {
    if driver == NativeOperationDriver::EncoderDecoderGenerate {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedOperation,
            "encoder-decoder generation requires its dedicated physical driver",
        ));
    }
    if !matches!(
        prepared_multimodal,
        super::NativePreparedMultimodalClass::Disabled
    ) {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedCapability,
            "prepared-media is not implemented by the resident M1 physical driver",
        ));
    }
    Ok(())
}

fn validate_generate_driver_input(
    prompt: &super::NativePromptInput,
) -> Result<(), NativeRuntimeError> {
    match prompt {
        super::NativePromptInput::TokenIds(_) | super::NativePromptInput::Text(_) => Ok(()),
        super::NativePromptInput::Rendered(_) => Ok(()),
        super::NativePromptInput::Prepared(_) => Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedCapability,
            "owned prepared-prefill requires a prepared-media worker profile",
        )),
        super::NativePromptInput::ModelSpecific(_) => Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedCapability,
            "model-specific prepared input requires its dedicated physical driver",
        )),
    }
}

fn validate_cache_driver(context: &super::NativeCacheContext) -> Result<(), NativeRuntimeError> {
    if context.allow_read || context.allow_write {
        Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedCapability,
            "native cross-request cache adoption is unavailable for this cache namespace",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn stable_union_token_ids(inputs: &[&[i32]]) -> Result<Vec<i32>, NativeRuntimeError> {
    let capacity = inputs
        .iter()
        .fold(0usize, |total, input| total.saturating_add(input.len()))
        .min(super::MAX_NATIVE_STOP_TOKEN_IDS);
    let mut seen = HashSet::with_capacity(capacity);
    let mut output = Vec::with_capacity(capacity);
    for id in inputs.iter().flat_map(|input| input.iter().copied()) {
        if id < 0 {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::Tokenizer,
                "worker token policy contains a negative token ID",
            ));
        }
        if seen.insert(id) {
            if output.len() == super::MAX_NATIVE_STOP_TOKEN_IDS {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::ResourceLimit,
                    "worker token policy union exceeds the native bound",
                ));
            }
            output.push(id);
        }
    }
    Ok(output)
}

fn thread_start_error(error: std::io::Error) -> NativeRuntimeError {
    NativeRuntimeError::new(
        NativeRuntimeErrorKind::Infrastructure,
        format!("failed to start native worker thread: {error}"),
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct NativeLoadReady {
    statistics: super::NativeLoadStatistics,
    decoder: Option<super::IncrementalDecodeGuarantee>,
}

fn start_generation_scheduler(
    request: NativeWorkerStartRequest,
    rx: mpsc::Receiver<ModelRequest>,
    worker_epoch: u64,
    ready: mpsc::SyncSender<Result<NativeLoadReady, NativeRuntimeError>>,
) -> Result<(), NativeRuntimeError> {
    if !matches!(
        request.profile.operation_driver(),
        NativeOperationDriver::AutoregressiveGenerate
            | NativeOperationDriver::EncoderDecoderGenerate
    ) {
        let error = NativeRuntimeError::new(
            NativeRuntimeErrorKind::UnsupportedOperation,
            "this worker factory currently requires a managed generation profile",
        );
        let _ = ready.send(Err(error.clone()));
        return Err(error);
    }
    let loaded = super::load_residence::load_generation(&request);
    let (loaded, statistics) = match loaded {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            return Err(error);
        }
    };
    let decoder = loaded.tokenizer.guarantee();
    let tokenizer = loaded.tokenizer.shared_tokenizer();
    let policy = request.profile.policy();
    let mut scheduler = crate::server::batch::BatchScheduler::with_config(
        loaded.model,
        tokenizer,
        loaded.eos,
        rx,
        policy.batch.max_active_sequences as usize,
        policy.batch.max_queued_operations as usize,
        Arc::new(crate::server::BatchMetrics::new()),
        Arc::new(crate::server::batch::BatchObservability::new()),
        policy.batch.prefill_chunk_tokens as usize,
        false,
        crate::server::PreemptionPolicy::default(),
        policy.batch.max_active_sequences as usize,
        crate::server::DecodeStorageBackend::Dense,
    )
    .with_native_runtime(loaded.tokenizer, worker_epoch);
    // Loaded arrays own immutable reader backing; tokenizer state owns its bytes.
    drop(request);
    let _ = ready.send(Ok(NativeLoadReady {
        statistics,
        decoder: Some(decoder),
    }));
    scheduler.run();
    Ok(())
}

fn start_embedding_worker(
    request: NativeWorkerStartRequest,
    rx: mpsc::Receiver<NativeEmbeddingCommand>,
    ready: mpsc::SyncSender<Result<NativeLoadReady, NativeRuntimeError>>,
) -> Result<(), NativeRuntimeError> {
    let stream = mlxcel_core::streams::new_thread_local_generation_stream();
    mlxcel_core::streams::install_thread_local_default_stream(stream.as_ref());
    let loaded = super::load_residence::load_embedding(&request);
    let (loaded, statistics) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            return Err(error);
        }
    };
    let batch_size = request.profile.policy().batch.max_active_sequences as usize;
    let engine = crate::embeddings::EmbeddingEngine::new(loaded, batch_size);
    drop(request);
    let _ = ready.send(Ok(NativeLoadReady {
        statistics,
        decoder: None,
    }));
    run_embedding_command_loop(&engine, rx, stream.as_ref());
    Ok(())
}

fn server_embedding_worker_info(
    engine: &crate::embeddings::EmbeddingEngine,
) -> crate::server::embedding_worker::EmbeddingWorkerInfo {
    crate::server::embedding_worker::EmbeddingWorkerInfo {
        dim: engine.dim(),
        max_length: engine.max_length(),
        vocab_size: engine.vocab_size(),
        multi_vector: engine.multi_vector(),
        supports_images: engine.supports_images(),
        batch_size: engine.batch_size(),
        model_type: engine.model_type(),
        pooling: engine.pooling(),
        embd_normalize: engine.default_normalize(),
    }
}

fn run_embedding_command_loop(
    engine: &crate::embeddings::EmbeddingEngine,
    rx: mpsc::Receiver<NativeEmbeddingCommand>,
    stream: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxThreadLocalStream>>,
) {
    for command in rx {
        match command {
            NativeEmbeddingCommand::Resolve { inputs, reply } => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    resolve_embedding_inputs(engine, &inputs)
                }))
                .unwrap_or_else(|payload| Err(embedding_panic_error(payload)));
                let _ = reply.send(result);
            }
            NativeEmbeddingCommand::Start {
                reservation_id,
                request,
                mut sink,
                release,
                cancelled,
            } => {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_embedding_request(engine, &request, sink.as_mut(), &cancelled)
                }));
                if let Err(payload) = outcome {
                    sink.terminal(super::NativePhysicalTerminal::Failed(
                        embedding_panic_error(payload),
                    ));
                }
                release.release(reservation_id);
            }
            NativeEmbeddingCommand::StartServer {
                reservation_id,
                request,
                respond,
                release,
                cancelled,
            } => {
                let result = if cancelled.load(Ordering::Acquire) {
                    Err(crate::server::embedding_model::EmbeddingError::Timeout)
                } else {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_server_embedding_operation(engine, request)
                    }))
                    .unwrap_or_else(|payload| {
                        Err(crate::server::embedding_model::EmbeddingError::Internal(
                            embedding_panic_message(payload),
                        ))
                    })
                };
                let _ = respond.send(result);
                release.release(reservation_id);
            }
            NativeEmbeddingCommand::MemorySnapshot(reply) => {
                let _ = reply.send(capture_memory_snapshot(0, 0));
            }
            NativeEmbeddingCommand::Reclaim(request, reply) => {
                let _ = reply.send(reclaim_idle_allocator(request, 0));
            }
            NativeEmbeddingCommand::Shutdown => break,
        }
        mlxcel_core::streams::synchronize_thread_local_stream(stream);
    }
}

fn run_server_embedding_operation(
    engine: &crate::embeddings::EmbeddingEngine,
    request: ServerEmbeddingOperation,
) -> Result<crate::embeddings::EmbedReply, crate::server::embedding_model::EmbeddingError> {
    match request {
        ServerEmbeddingOperation::Texts(texts, options) => {
            engine.embed_texts(&texts, &options).map_err(Into::into)
        }
        ServerEmbeddingOperation::Tokens(rows, options) => {
            engine.embed_tokens(&rows, &options).map_err(Into::into)
        }
        ServerEmbeddingOperation::Image(image, options) => {
            engine.embed_image(image, &options).map_err(Into::into)
        }
    }
}

fn embedding_panic_error(payload: Box<dyn std::any::Any + Send>) -> NativeRuntimeError {
    NativeRuntimeError::new(
        NativeRuntimeErrorKind::Infrastructure,
        embedding_panic_message(payload),
    )
}

fn embedding_panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let detail = if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "embedding request panicked".to_string()
    };
    tracing::error!(target: "mlxcel::embeddings", "embedding worker recovered from panic: {detail}");
    format!("embedding worker recovered from panic: {detail}")
}

fn resolve_embedding_inputs(
    engine: &crate::embeddings::EmbeddingEngine,
    inputs: &[super::NativeEmbeddingInput],
) -> Result<NativeEmbedResolved, NativeRuntimeError> {
    let input_tokens = inputs
        .iter()
        .map(|input| match input {
            super::NativeEmbeddingInput::TokenIds(ids) => Ok(ids.len()),
            super::NativeEmbeddingInput::Text(text) => {
                engine.resolved_text_tokens(text).map_err(embedding_error)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeEmbedResolved {
        input_tokens,
        dimension: engine.dim(),
        multi_vector: engine.multi_vector(),
    })
}

fn embedding_error(error: crate::embeddings::EmbeddingEngineError) -> NativeRuntimeError {
    let kind = match &error {
        crate::embeddings::EmbeddingEngineError::InvalidInput(_) => {
            NativeRuntimeErrorKind::InvalidRequest
        }
        crate::embeddings::EmbeddingEngineError::Internal(_) => {
            NativeRuntimeErrorKind::Infrastructure
        }
    };
    NativeRuntimeError::new(kind, error.to_string())
}

fn run_embedding_request(
    engine: &crate::embeddings::EmbeddingEngine,
    request: &NativeEmbedRequest,
    sink: &mut dyn NativeEmbeddingSink,
    cancelled: &AtomicBool,
) {
    let normalize = match request.normalization {
        super::NativeEmbeddingNormalization::ModelDefault => None,
        super::NativeEmbeddingNormalization::None => {
            Some(crate::embeddings::normalize::EmbdNormalize::NONE)
        }
        super::NativeEmbeddingNormalization::L2 => {
            Some(crate::embeddings::normalize::EmbdNormalize::EUCLIDEAN)
        }
    };
    let options = crate::embeddings::EmbedOptions {
        instruction: None,
        dimensions: None,
        normalize,
    };
    let normalized = normalize.unwrap_or_else(|| engine.default_normalize())
        != crate::embeddings::normalize::EmbdNormalize::NONE;

    // Preserve the server's physical micro-batching for homogeneous requests.
    // The public DTO permits mixed text/token inputs, which retain the ordinary
    // one-input fallback below because the concrete engine exposes distinct
    // entry points for those operation shapes.
    let batched = if request
        .inputs
        .iter()
        .all(|input| matches!(input, super::NativeEmbeddingInput::Text(_)))
    {
        let texts = request
            .inputs
            .iter()
            .map(|input| match input {
                super::NativeEmbeddingInput::Text(text) => text.to_string(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        Some(engine.embed_texts(&texts, &options))
    } else if request
        .inputs
        .iter()
        .all(|input| matches!(input, super::NativeEmbeddingInput::TokenIds(_)))
    {
        let rows = request
            .inputs
            .iter()
            .map(|input| match input {
                super::NativeEmbeddingInput::TokenIds(ids) => {
                    ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
                }
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        Some(engine.embed_tokens(&rows, &options))
    } else {
        None
    };
    if let Some(reply) = batched {
        match reply {
            Ok(reply) => {
                emit_native_embedding_reply(request, sink, reply, normalized);
            }
            Err(error) => sink.terminal(super::NativePhysicalTerminal::Failed(embedding_error(
                error,
            ))),
        }
        return;
    }

    for (index, input) in request.inputs.iter().enumerate() {
        if cancelled.load(Ordering::Acquire) {
            sink.terminal(super::NativePhysicalTerminal::Cancelled);
            return;
        }
        let reply = match input {
            super::NativeEmbeddingInput::TokenIds(ids) => {
                let row = ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
                engine.embed_tokens(&[row], &options)
            }
            super::NativeEmbeddingInput::Text(text) => {
                engine.embed_texts(&[text.to_string()], &options)
            }
        };
        let reply = match reply {
            Ok(reply) => reply,
            Err(error) => {
                sink.terminal(super::NativePhysicalTerminal::Failed(embedding_error(
                    error,
                )));
                return;
            }
        };
        let Some(vector) = reply.vectors.into_iter().next() else {
            sink.terminal(super::NativePhysicalTerminal::Failed(
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Infrastructure,
                    "embedding engine returned no vector",
                ),
            ));
            return;
        };
        let output = super::NativeEmbeddingOutput {
            input_index: index as u32,
            dimension: vector.values.len() as u32,
            normalized,
            vector: Arc::from(vector.values),
        };
        if let Err(error) = output.validate_for(request) {
            sink.terminal(super::NativePhysicalTerminal::Failed(error));
            return;
        }
        if let Err(error) = sink.vector(output) {
            sink.terminal(super::NativePhysicalTerminal::Failed(error));
            return;
        }
    }
    if cancelled.load(Ordering::Acquire) {
        sink.terminal(super::NativePhysicalTerminal::Cancelled);
    } else {
        sink.terminal(super::NativePhysicalTerminal::Completed {
            reason: super::NativeStopReason::Requested,
        });
    }
}

fn emit_native_embedding_reply(
    request: &NativeEmbedRequest,
    sink: &mut dyn NativeEmbeddingSink,
    reply: crate::embeddings::EmbedReply,
    normalized: bool,
) {
    if reply.vectors.len() != request.inputs.len() {
        sink.terminal(super::NativePhysicalTerminal::Failed(
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidOutput,
                "embedding engine output count differs from the admitted input count",
            ),
        ));
        return;
    }
    for (index, vector) in reply.vectors.into_iter().enumerate() {
        let output = super::NativeEmbeddingOutput {
            input_index: index as u32,
            dimension: vector.values.len() as u32,
            normalized,
            vector: Arc::from(vector.values),
        };
        if let Err(error) = output.validate_for(request) {
            sink.terminal(super::NativePhysicalTerminal::Failed(error));
            return;
        }
        if let Err(error) = sink.vector(output) {
            sink.terminal(super::NativePhysicalTerminal::Failed(error));
            return;
        }
    }
    sink.terminal(super::NativePhysicalTerminal::Completed {
        reason: super::NativeStopReason::Requested,
    });
}

struct NativeManagedEngineClient {
    descriptor: NativeEngineDescriptor,
    profile: NativeWorkerConfig,
    capacity: usize,
    channel: NativeEngineChannel,
    reservations: Arc<ReservationBook>,
    alive: Arc<AtomicBool>,
    loaded: NativeLoadReady,
}

struct NativePreparedReleaseRoute {
    tx: mpsc::Sender<ModelRequest>,
}

impl NativePreparedRelease for NativePreparedReleaseRoute {
    fn release_prepared(&self, reservation_id: u128, sequence: super::NativeSequenceIdentity) {
        let _ = self.tx.send(ModelRequest::Native(
            NativeSchedulerCommand::ReleasePrepared {
                reservation_id,
                sequence,
            },
        ));
    }
}

impl NativeManagedEngineClient {
    fn worker_gone() -> NativeRuntimeError {
        NativeRuntimeError::new(
            NativeRuntimeErrorKind::Infrastructure,
            "native worker command channel is closed",
        )
    }

    fn note_worker_gone(&self) -> NativeRuntimeError {
        self.alive.store(false, Ordering::Release);
        Self::worker_gone()
    }

    fn ensure_alive(&self) -> Result<(), NativeRuntimeError> {
        if self.alive.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Self::worker_gone())
        }
    }
}

impl NativeManagedEngine for NativeManagedEngineClient {
    fn load_statistics(&self) -> super::NativeLoadStatistics {
        self.loaded.statistics
    }

    fn descriptor(&self) -> NativeEngineDescriptor {
        let mut descriptor = self.descriptor.clone();
        descriptor.health = project_worker_health(self.alive.load(Ordering::Acquire));
        descriptor
    }

    fn admit_generate(
        &mut self,
        request: &NativeGenerateRequest,
    ) -> Result<NativeGenerateAdmission, NativeRuntimeError> {
        self.ensure_alive()?;
        if !matches!(
            self.profile.operation_driver(),
            NativeOperationDriver::AutoregressiveGenerate
                | NativeOperationDriver::EncoderDecoderGenerate
        ) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "generation was submitted to a non-generation worker",
            ));
        }
        if self
            .loaded
            .decoder
            .is_none_or(|guarantee| guarantee == super::IncrementalDecodeGuarantee::TerminalOnly)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "terminal-only tokenizer profiles cannot admit managed streaming generation",
            ));
        }
        validate_generate_driver_input(&request.prompt)?;
        if !matches!(
            request.constraint,
            super::NativeConstraintPlan::None | super::NativeConstraintPlan::ValidateOnly
        ) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "this worker has no native compiler for the requested opaque constraint",
            ));
        }
        validate_cache_driver(&request.cache)?;
        if request.speculation.preference == super::NativeSpeculationPreference::Required
            || (!request.speculation.allow_ordinary_fallback
                && request.speculation.preference != super::NativeSpeculationPreference::Disabled)
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "the requested speculative policy cannot use ordinary physical fallback",
            ));
        }
        let policy = self.profile.policy();
        if request.limits.max_new_tokens > policy.max_generation_tokens {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "generation request exceeds the resident worker profile",
            ));
        }
        request.validate()?;
        let NativeEngineChannel::Generate(tx) = &self.channel else {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                "generation worker channel is unavailable",
            ));
        };
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        tx.send(ModelRequest::Native(
            NativeSchedulerCommand::ResolveGenerate {
                request: Arc::new(request.shallow_clone()),
                reply: reply_tx,
            },
        ))
        .map_err(|_| self.note_worker_gone())?;
        let resolution = reply_rx.recv().map_err(|_| self.note_worker_gone())??;
        request.validate_resolved_prompt_tokens(resolution.prompt_token_ids.len())?;
        if resolution.prompt_token_ids.len() > policy.max_prompt_tokens as usize {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "resolved prompt exceeds the resident worker profile",
            ));
        }
        let validated = request.validated_resolved(
            resolution.prompt_token_ids,
            resolution.eos_token_ids,
            resolution.output_suppressed_token_ids,
        )?;
        let (reservation_id, release) = self
            .reservations
            .reserve(request.request_id, self.capacity)?;
        Ok(NativeGenerateAdmission::new(
            reservation_id,
            self.profile.operation_driver(),
            validated,
            release,
        ))
    }

    fn prepare_generate(
        &mut self,
        admission: NativeGenerateAdmission,
    ) -> Result<NativePreparedGenerate, NativeRuntimeError> {
        self.ensure_alive()?;
        let (reservation_id, _, request, guard) = admission.into_parts();
        let owner: Arc<dyn NativeReservationRelease> = self.reservations.clone();
        if !guard.belongs_to(&owner) {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::StaleHandle,
                "generation admission belongs to another native worker",
            ));
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        self.reservations
            .bind_cancellation(reservation_id, Arc::clone(&cancelled));
        let NativeEngineChannel::Generate(tx) = &self.channel else {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "generation was submitted to an embedding worker",
            ));
        };
        let (prepared_tx, prepared_rx) = mpsc::sync_channel(1);
        tx.send(ModelRequest::Native(
            NativeSchedulerCommand::PrepareGenerate {
                reservation_id,
                request: Arc::clone(&request),
                cancelled,
                guard,
                prepared: prepared_tx,
            },
        ))
        .map_err(|_| self.note_worker_gone())?;
        let sequence = prepared_rx.recv().map_err(|_| self.note_worker_gone())??;
        let release: Arc<dyn NativePreparedRelease> =
            Arc::new(NativePreparedReleaseRoute { tx: tx.clone() });
        Ok(NativePreparedGenerate::new(
            reservation_id,
            request,
            sequence,
            release,
        ))
    }

    fn start_generate(
        &mut self,
        prepared: NativePreparedGenerate,
        sink: Box<dyn NativeGenerateSink + Send>,
    ) -> Result<NativeExecutionHandle, NativeRuntimeError> {
        self.ensure_alive()?;
        if prepared.sequence().worker_epoch != self.descriptor.worker_epoch {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::StaleHandle,
                "prepared generation belongs to a different worker epoch",
            ));
        }
        let reservation_id = prepared.reservation_id();
        let sequence = prepared.sequence();
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let NativeEngineChannel::Generate(tx) = &self.channel else {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "generation was submitted to an embedding worker",
            ));
        };
        tx.send(ModelRequest::Native(
            NativeSchedulerCommand::StartGenerate {
                reservation_id,
                sequence,
                sink,
                started: started_tx,
            },
        ))
        .map_err(|_| self.note_worker_gone())?;
        prepared.commit_start();
        started_rx.recv().map_err(|_| self.note_worker_gone())??;
        Ok(NativeExecutionHandle::new(
            self.descriptor.worker_epoch,
            reservation_id,
            NativeExecutionKind::Generate,
        ))
    }

    fn admit_embed(
        &mut self,
        request: &NativeEmbedRequest,
    ) -> Result<NativeEmbedAdmission, NativeRuntimeError> {
        self.ensure_alive()?;
        if self.profile.operation_driver() != NativeOperationDriver::Embedding {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "embedding was submitted to a generation worker",
            ));
        }
        let NativeEngineChannel::Embed(tx) = &self.channel else {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::Infrastructure,
                "embedding worker channel is unavailable",
            ));
        };
        admit_embedding_request(
            tx,
            &self.reservations,
            self.capacity,
            self.profile.policy().max_embedding_inputs as usize,
            self.profile.policy().max_embedding_dimensions as usize,
            request,
        )
        .map(|(admission, _)| admission)
    }

    fn start_embed(
        &mut self,
        admission: NativeEmbedAdmission,
        sink: Box<dyn NativeEmbeddingSink + Send>,
    ) -> Result<NativeExecutionHandle, NativeRuntimeError> {
        let NativeEngineChannel::Embed(tx) = &self.channel else {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "embedding was submitted to a generation worker",
            ));
        };
        start_embedding_admission(
            self.descriptor.worker_epoch,
            tx,
            &self.reservations,
            admission,
            sink,
        )
    }

    fn cancel(&mut self, handle: &NativeExecutionHandle) {
        if handle.worker_epoch() == self.descriptor.worker_epoch {
            let _ = self.reservations.cancel(handle.execution_id());
        }
    }

    fn memory_snapshot(&self) -> NativeMemorySnapshot {
        let (tx, rx) = mpsc::sync_channel(1);
        let sent = match &self.channel {
            NativeEngineChannel::Generate(channel) => channel.send(ModelRequest::Native(
                NativeSchedulerCommand::MemorySnapshot(tx),
            )),
            NativeEngineChannel::Embed(channel) => channel
                .send(NativeEmbeddingCommand::MemorySnapshot(tx))
                .map_err(|_| mpsc::SendError(ModelRequest::Shutdown)),
        };
        if sent.is_err() {
            self.alive.store(false, Ordering::Release);
            return NativeMemorySnapshot::default();
        }
        match rx.recv() {
            Ok(snapshot) => snapshot,
            Err(_) => {
                self.alive.store(false, Ordering::Release);
                NativeMemorySnapshot::default()
            }
        }
    }

    fn reclaim(&mut self, request: NativeReclaimRequest) -> NativeReclaimResult {
        let (tx, rx) = mpsc::sync_channel(1);
        let sent = match &self.channel {
            NativeEngineChannel::Generate(channel) => channel.send(ModelRequest::Native(
                NativeSchedulerCommand::Reclaim(request, tx),
            )),
            NativeEngineChannel::Embed(channel) => channel
                .send(NativeEmbeddingCommand::Reclaim(request, tx))
                .map_err(|_| mpsc::SendError(ModelRequest::Shutdown)),
        };
        if sent.is_err() {
            self.alive.store(false, Ordering::Release);
            return NativeReclaimResult::default();
        }
        match rx.recv() {
            Ok(result) => result,
            Err(_) => {
                self.alive.store(false, Ordering::Release);
                NativeReclaimResult::default()
            }
        }
    }
}

impl Drop for NativeManagedEngineClient {
    fn drop(&mut self) {
        match &self.channel {
            NativeEngineChannel::Generate(tx) => {
                let _ = tx.send(ModelRequest::Shutdown);
            }
            NativeEngineChannel::Embed(tx) => {
                let _ = tx.send(NativeEmbeddingCommand::Shutdown);
            }
        }
    }
}

trait NativeClientMustBeSend: Send {}
impl NativeClientMustBeSend for NativeManagedEngineClient {}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission_test_fixture(request_id: u128) -> (NativeWorkerConfig, NativeGenerateRequest) {
        use std::collections::BTreeMap;
        use std::io::Cursor;

        use sha2::{Digest, Sha256};

        struct Source(BTreeMap<String, Vec<u8>>);
        impl super::super::NativePackageArtifactSource for Source {
            fn open_artifact(
                &mut self,
                label: &str,
            ) -> Result<Box<dyn std::io::Read + Send>, NativeRuntimeError> {
                Ok(Box::new(Cursor::new(
                    self.0.get(label).expect("declared component").clone(),
                )))
            }
        }

        let config = br#"{"model_type":"llama","max_position_embeddings":8}"#.to_vec();
        let crate::tokenizer::MlxcelTokenizer::HuggingFace(tokenizer) =
            crate::tokenizer::MlxcelTokenizer::stub_gemma_sequence_byte_fallback()
        else {
            unreachable!()
        };
        let tokenizer = tokenizer.to_string(false).unwrap().into_bytes();
        let tokenizer_config = b"{}".to_vec();
        let bytes = BTreeMap::from([
            ("config.json".to_string(), config),
            ("tokenizer.json".to_string(), tokenizer),
            ("tokenizer_config.json".to_string(), tokenizer_config),
        ]);
        let declarations = bytes
            .iter()
            .map(|(label, bytes)| {
                super::super::NativePackageComponent::new(
                    label,
                    bytes.len() as u64,
                    Sha256::digest(bytes).into(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let tokenizer_config_component = declarations
            .iter()
            .find(|component| component.label() == "tokenizer_config.json")
            .unwrap()
            .clone();
        let package = super::super::NativeImmutablePackage::acquire(
            &mut Source(bytes),
            declarations,
            super::super::NativePackageLimits::default(),
        )
        .unwrap();
        let report = super::super::model_report::qualify_generation_report_for_orchestration_tests(
            super::super::model_report::inspect_package(&package).unwrap(),
        );
        let profile = NativeWorkerConfig::try_from_report(
            &report,
            super::super::NativeWorkerPolicy {
                operation_driver: NativeOperationDriver::AutoregressiveGenerate,
                scheduler_topology: super::super::NativeSchedulerTopology::SingleSequence,
                sequence_state_topology: super::super::NativeSequenceStateTopology::Independent,
                context: super::super::NativeContextPolicy {
                    max_context_tokens: 8,
                    max_position: 8,
                },
                batch: super::super::NativeBatchPolicy {
                    max_active_sequences: 1,
                    max_queued_operations: 1,
                    max_batch_tokens: 8,
                    prefill_chunk_tokens: 1,
                },
                cache: super::super::NativeCachePolicy {
                    topology: super::super::NativeCacheTopology::Dense,
                    max_resident_bytes: 1,
                    prefix_reuse: false,
                },
                device: super::super::NativeDevicePolicy {
                    class: super::super::NativeDeviceClass::Cpu,
                    offload: super::super::NativeOffloadPolicy::HostVisible,
                    max_device_bytes: 1,
                },
                model_owned_state: super::super::NativeModelOwnedStateMode::Disabled,
                adapter: None,
                drafter: super::super::NativeDrafterTopology::Disabled,
                prepared_multimodal: super::super::NativePreparedMultimodalClass::Disabled,
                max_prompt_tokens: 4,
                max_generation_tokens: 4,
                max_embedding_inputs: 1,
                max_embedding_dimensions: 1,
                max_accepted_chunk_units: super::super::MAX_NATIVE_ACCEPTED_UNITS as u32,
                max_constraint_bytes: 1,
            },
        )
        .unwrap();
        let renderer = super::super::NativePromptRenderer::bind(
            &package,
            &report,
            super::super::NativePromptRendererSpec {
                kind: super::super::NativePromptRendererKind::LiteralUtf8,
                template_component: None,
                tokenizer_config_component,
                limits: super::super::NativePromptRendererLimits::default(),
                add_special_tokens: false,
            },
        )
        .unwrap();
        let prompt = renderer
            .render(&super::super::NativeTemplateValue::string("ok"))
            .unwrap();
        let request = NativeGenerateRequest {
            request_id,
            prompt: super::super::NativePromptInput::Rendered(prompt),
            sampling: super::super::NativeSamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 7,
            },
            limits: super::super::NativeGenerationLimits {
                max_new_tokens: 1,
                max_total_tokens: 3,
            },
            eos_token_ids: Arc::from([7]),
            output_suppressed_token_ids: Arc::from([17]),
            scheduling: super::super::NativeSchedulingHints::default(),
            cache: super::super::NativeCacheContext {
                namespace: Arc::from("test"),
                input_identity: [9; 32],
                allow_read: false,
                allow_write: false,
            },
            constraint: super::super::NativeConstraintPlan::None,
            speculation: super::super::NativeSpeculationPolicy {
                preference: super::super::NativeSpeculationPreference::Disabled,
                minimum_fidelity: super::super::NativeSpeculativeFidelity::ExactTargetDistribution,
                allow_ordinary_fallback: true,
            },
        };
        (profile, request)
    }

    fn admission_test_client(
        profile: NativeWorkerConfig,
    ) -> (
        NativeManagedEngineClient,
        mpsc::Receiver<ModelRequest>,
        Arc<ReservationBook>,
    ) {
        let (tx, rx) = mpsc::channel();
        let reservations = Arc::new(ReservationBook::default());
        let descriptor = NativeEngineDescriptor {
            worker_id: 1,
            worker_epoch: 2,
            build_identity: [3; 32],
            package_identity: profile.package_identity(),
            profile_identity: profile.identity().unwrap(),
            health: NativeWorkerHealth::Ready,
        };
        (
            NativeManagedEngineClient {
                descriptor,
                loaded: NativeLoadReady {
                    statistics: super::super::NativeLoadStatistics::default(),
                    decoder: profile.decoder_guarantee(),
                },
                profile,
                capacity: 2,
                channel: NativeEngineChannel::Generate(tx),
                reservations: Arc::clone(&reservations),
                alive: Arc::new(AtomicBool::new(true)),
            },
            rx,
            reservations,
        )
    }

    #[test]
    fn normal_server_generation_does_not_gate_managed_path_on_tokenizer_inspection() {
        let source = include_str!("engine.rs");
        let spawn = source
            .split_once("pub(crate) fn spawn_mlxcel_server_managed_engine")
            .expect("server managed-engine spawn")
            .1
            .split_once("let (command_tx, command_rx)")
            .expect("spawn command channel")
            .0;
        assert!(
            !spawn.contains("server_tokenizer_guarantee"),
            "normal decoder-only operation selection must not hide an unsupported tokenizer guarantee behind the legacy server path"
        );
    }

    #[test]
    fn native_scheduler_command_has_no_server_projection_field_graph() {
        let production = include_str!("engine.rs")
            .split_once("#[cfg(test)]")
            .expect("engine test boundary")
            .0;
        for forbidden in [
            "MlxcelServerProjection",
            "MlxcelServerNativeSink",
            "StopMatcher",
            "GenerateEvent",
            "response_tx",
            "pending_logprobs",
            "generated_text",
        ] {
            assert!(
                !production.contains(forbidden),
                "native scheduler command retained server projection authority: {forbidden}"
            );
        }
    }

    #[test]
    fn legacy_server_model_table_is_limited_to_dedicated_operation_classes() {
        for model_type in [
            crate::models::ModelType::DiffusionGemma,
            crate::models::ModelType::Llada2Moe,
            crate::models::ModelType::Florence2VLM,
        ] {
            assert!(uses_dedicated_legacy_server_operation(model_type));
        }
        for model_type in [
            crate::models::ModelType::Llama,
            crate::models::ModelType::Qwen3,
            crate::models::ModelType::Gemma4,
            crate::models::ModelType::Gemma4VLM,
        ] {
            assert!(!uses_dedicated_legacy_server_operation(model_type));
        }
    }
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::sampling::LogprobsConfig;
    use mlxcel_core::session::{
        OwnedTensor, PreparedAdapterMode, PreparedAttentionBias, PreparedPositions,
        PreparedPrefill, PreparedTensorDType,
    };

    static_assertions::assert_impl_all!(NativeManagedEngineClient: Send);

    fn server_test_request(
        response_tx: mpsc::Sender<crate::server::model_provider::GenerateEvent>,
    ) -> MlxcelServerGenerateRequest {
        MlxcelServerGenerateRequest {
            prompt: "hello".to_string(),
            prompt_token_ids: Some(vec![1]),
            options: crate::server::ServerGenerateOptions {
                max_tokens: 1,
                sampling: SamplingConfig::default(),
                stop_sequences: None,
                ignore_eos: false,
                priority: crate::server::batch::RequestPriority::default(),
                logprobs: LogprobsConfig::default(),
                reasoning_budget: crate::server::ReasoningBudgetOverride::default(),
                thinking_enter_block_on_start: false,
                reasoning_control: None,
                prompt_cache_ctx: None,
                structured: None,
                image_soft_tokens: None,
            },
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            media: crate::server::MediaRequestMetadata::default(),
            queue_reservation: None,
            response_tx,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    fn server_test_engine(
        capacity: usize,
    ) -> (
        MlxcelServerManagedEngine,
        mpsc::Receiver<ModelRequest>,
        Arc<ReservationBook>,
    ) {
        let (command_tx, command_rx) = mpsc::channel();
        let reservations = Arc::new(ReservationBook::default());
        (
            MlxcelServerManagedEngine {
                worker_epoch: 7,
                capacity,
                command_tx,
                reservations: Arc::clone(&reservations),
                managed_text: true,
                next_request_id: 0,
                shutdown_sent: false,
            },
            command_rx,
            reservations,
        )
    }

    #[test]
    fn reservation_release_is_exact_per_request_while_peers_remain_active() {
        let book = Arc::new(ReservationBook::default());
        let (first, _) = book.reserve(11, 2).unwrap();
        let (peer, _) = book.reserve(22, 2).unwrap();
        assert!(book.reserve(11, 3).is_err());
        book.release(first);
        let (replacement, _) = book.reserve(11, 3).unwrap();
        assert_ne!(replacement, first);
        book.release(peer);
        book.release(replacement);
        let state = book.inner.lock().unwrap();
        assert!(state.reservations.is_empty());
        assert!(state.request_ids.is_empty());
        assert!(state.request_by_reservation.is_empty());
    }

    #[test]
    fn stale_cancel_does_not_mutate_a_peer_reservation() {
        let book = Arc::new(ReservationBook::default());
        let (first, _) = book.reserve(11, 2).unwrap();
        let (peer, _) = book.reserve(22, 2).unwrap();
        let first_flag = Arc::new(AtomicBool::new(false));
        let peer_flag = Arc::new(AtomicBool::new(false));
        book.bind_cancellation(first, Arc::clone(&first_flag));
        book.bind_cancellation(peer, Arc::clone(&peer_flag));
        assert!(book.cancel(first));
        assert!(first_flag.load(Ordering::Acquire));
        assert!(!peer_flag.load(Ordering::Acquire));
        book.release(first);
        assert!(!book.cancel(first));
        assert!(!peer_flag.load(Ordering::Acquire));
    }

    #[test]
    fn server_prestart_failures_including_unsupported_guarantee_release_once_without_sink_output() {
        for kind in [
            NativeRuntimeErrorKind::Tokenizer,
            NativeRuntimeErrorKind::InvalidRequest,
            NativeRuntimeErrorKind::ResourceLimit,
            NativeRuntimeErrorKind::UnsupportedGuarantee,
        ] {
            let (command_tx, command_rx) = mpsc::channel();
            let reservations = Arc::new(ReservationBook::default());
            let mut engine = MlxcelServerManagedEngine {
                worker_epoch: 7,
                capacity: 1,
                command_tx,
                reservations: Arc::clone(&reservations),
                managed_text: true,
                next_request_id: 0,
                shutdown_sent: false,
            };
            let worker = std::thread::spawn(move || {
                let ModelRequest::Native(NativeSchedulerCommand::StartServerGenerate {
                    started,
                    ..
                }) = command_rx.recv().expect("server start command")
                else {
                    panic!("unexpected server command")
                };
                let _ = started.send(Err(NativeRuntimeError::new(kind, "pre-start failure")));
            });
            let (response_tx, response_rx) = mpsc::channel();
            let result = engine.start_server_generate(server_test_request(response_tx));
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("pre-start failure unexpectedly started"),
            };
            assert_eq!(error.kind(), kind);
            worker.join().unwrap();
            assert!(reservations.inner.lock().unwrap().reservations.is_empty());
            assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
            assert!(matches!(
                response_rx.try_recv(),
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected)
            ));
        }
    }

    #[test]
    fn dropping_pending_server_start_before_ack_releases_once() {
        let (mut engine, command_rx, reservations) = server_test_engine(1);
        let (response_tx, _response_rx) = mpsc::channel();
        let pending = engine
            .begin_server_generate(server_test_request(response_tx))
            .unwrap();
        let _command = command_rx.recv().expect("server start command");
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 0);
        drop(pending);
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
    }

    #[test]
    fn rejected_pending_server_start_releases_once() {
        let (mut engine, command_rx, reservations) = server_test_engine(1);
        let (response_tx, _response_rx) = mpsc::channel();
        let pending = engine
            .begin_server_generate(server_test_request(response_tx))
            .unwrap();
        let ModelRequest::Native(NativeSchedulerCommand::StartServerGenerate { started, .. }) =
            command_rx.recv().expect("server start command")
        else {
            panic!("unexpected server command")
        };
        started
            .send(Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "synthetic pre-start rejection",
            )))
            .unwrap();
        let error = match pending.wait_started() {
            Err(error) => error,
            Ok(_) => panic!("rejected server start unexpectedly succeeded"),
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::ResourceLimit);
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
    }

    #[test]
    fn disconnected_pending_server_start_releases_once() {
        let (mut engine, command_rx, reservations) = server_test_engine(1);
        let (response_tx, _response_rx) = mpsc::channel();
        let pending = engine
            .begin_server_generate(server_test_request(response_tx))
            .unwrap();
        let ModelRequest::Native(NativeSchedulerCommand::StartServerGenerate { started, .. }) =
            command_rx.recv().expect("server start command")
        else {
            panic!("unexpected server command")
        };
        drop(started);
        let error = match pending.wait_started() {
            Err(error) => error,
            Ok(_) => panic!("disconnected server start unexpectedly succeeded"),
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::Infrastructure);
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
    }

    #[test]
    fn acknowledged_pending_server_start_commits_until_terminal_release() {
        let (mut engine, command_rx, reservations) = server_test_engine(1);
        let (response_tx, _response_rx) = mpsc::channel();
        let pending = engine
            .begin_server_generate(server_test_request(response_tx))
            .unwrap();
        let ModelRequest::Native(NativeSchedulerCommand::StartServerGenerate {
            reservation_id,
            started,
            release,
            ..
        }) = command_rx.recv().expect("server start command")
        else {
            panic!("unexpected server command")
        };
        started.send(Ok(())).unwrap();
        let handle = pending.wait_started().unwrap();
        assert_eq!(handle.execution_id(), reservation_id);
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 0);
        assert!(
            reservations
                .inner
                .lock()
                .unwrap()
                .reservations
                .contains(&reservation_id)
        );
        release.release(reservation_id);
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
    }

    #[test]
    fn server_frontend_queues_peers_before_either_started_handshake_completes() {
        let (command_tx, command_rx) = mpsc::channel();
        let frontend = Arc::new(MlxcelServerFrontend::from_managed_engine(
            MlxcelServerManagedEngine {
                worker_epoch: 7,
                capacity: 2,
                command_tx,
                reservations: Arc::new(ReservationBook::default()),
                managed_text: true,
                next_request_id: 0,
                shutdown_sent: false,
            },
        ));
        let first = {
            let frontend = Arc::clone(&frontend);
            std::thread::spawn(move || {
                let (response_tx, _response_rx) = mpsc::channel();
                frontend.start_generate(server_test_request(response_tx))
            })
        };
        let first_command = command_rx.recv().expect("first start command");
        let second = {
            let frontend = Arc::clone(&frontend);
            std::thread::spawn(move || {
                let (response_tx, _response_rx) = mpsc::channel();
                frontend.start_generate(server_test_request(response_tx))
            })
        };
        let second_command = command_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .expect("second start command must queue before the first handshake completes");
        for command in [first_command, second_command] {
            let ModelRequest::Native(NativeSchedulerCommand::StartServerGenerate {
                started, ..
            }) = command
            else {
                panic!("unexpected server command")
            };
            let _ = started.send(Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "synthetic pre-start rejection",
            )));
        }
        assert!(first.join().unwrap().is_err());
        assert!(second.join().unwrap().is_err());
    }

    #[test]
    fn operation_driver_rejects_unimplemented_encoder_decoder_and_prepared_media_profiles() {
        assert!(
            validate_implemented_worker_contract(
                NativeOperationDriver::AutoregressiveGenerate,
                &super::super::NativePreparedMultimodalClass::Disabled,
            )
            .is_ok()
        );
        assert_eq!(
            validate_implemented_worker_contract(
                NativeOperationDriver::EncoderDecoderGenerate,
                &super::super::NativePreparedMultimodalClass::Disabled,
            )
            .unwrap_err()
            .kind(),
            NativeRuntimeErrorKind::UnsupportedOperation
        );
        assert_eq!(
            validate_implemented_worker_contract(
                NativeOperationDriver::AutoregressiveGenerate,
                &super::super::NativePreparedMultimodalClass::OwnedPrefill {
                    modalities: super::super::NativeModalitySet {
                        text: true,
                        image: true,
                        audio: false,
                        video: false,
                    },
                    max_media_items: 1,
                    max_payload_bytes: 4,
                },
            )
            .unwrap_err()
            .kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );
    }

    #[test]
    fn token_driver_rejects_both_prepared_request_shapes_before_reservation() {
        assert!(
            validate_generate_driver_input(&super::super::NativePromptInput::TokenIds(Arc::from(
                [1]
            )))
            .is_ok()
        );
        let model_specific = super::super::NativeOwnedPreparedInput::new("schema", [1], 1)
            .expect("bounded prepared input");
        assert_eq!(
            validate_generate_driver_input(&super::super::NativePromptInput::ModelSpecific(
                model_specific,
            ))
            .unwrap_err()
            .kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );

        let prepared = PreparedPrefill {
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
            modalities: Vec::new(),
            adapter_mode: PreparedAdapterMode::Language,
        };
        let prepared = super::super::NativePreparedPrefill::try_from_owned(prepared)
            .expect("bounded prepared-prefill");
        assert_eq!(
            validate_generate_driver_input(&super::super::NativePromptInput::Prepared(prepared))
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::UnsupportedCapability
        );
    }

    #[test]
    fn worker_token_policy_union_is_stable_and_deduplicated() {
        assert_eq!(
            stable_union_token_ids(&[&[5, 3, 5][..], &[3, 7][..], &[9, 5, 7, 11][..],])
                .expect("bounded non-negative union"),
            vec![5, 3, 7, 9, 11]
        );
    }

    #[test]
    fn worker_token_policy_union_rejects_before_overallocation() {
        assert_eq!(
            stable_union_token_ids(&[&[-1]]).unwrap_err().kind(),
            NativeRuntimeErrorKind::Tokenizer
        );
        let over = (0..=super::super::MAX_NATIVE_STOP_TOKEN_IDS as i32).collect::<Vec<_>>();
        assert_eq!(
            stable_union_token_ids(&[&over]).unwrap_err().kind(),
            NativeRuntimeErrorKind::ResourceLimit
        );
    }

    #[test]
    fn rendered_realization_precedes_reservation_in_public_admission() {
        let source = include_str!("engine.rs");
        let body = source
            .split_once("fn admit_generate(")
            .expect("managed admission")
            .1
            .split_once("fn start_generate(")
            .expect("managed admission end")
            .0;
        let resolve = body
            .find("NativeSchedulerCommand::ResolveGenerate")
            .expect("worker-owned prompt realization command");
        let reserve = body
            .find(".reservations\n            .reserve")
            .expect("reservation creation");
        assert!(resolve < reserve);
    }

    struct CountingGenerateSink {
        started: Arc<std::sync::atomic::AtomicUsize>,
        terminal: Arc<std::sync::atomic::AtomicUsize>,
        physical_stopped: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl super::super::NativeGenerateSink for CountingGenerateSink {
        fn started(
            &mut self,
            _sequence: super::super::NativeSequenceIdentity,
        ) -> Result<(), NativeRuntimeError> {
            self.started.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn accepted(
            &mut self,
            _chunk: super::super::NativeAcceptedChunk,
        ) -> super::super::NativeContinuationDirective {
            super::super::NativeContinuationDirective::Stop { consumed_units: 0 }
        }

        fn physical(&mut self, _event: super::super::NativePhysicalEvent) {}

        fn terminal(&mut self, _terminal: super::super::NativePhysicalTerminal) {
            self.terminal.fetch_add(1, Ordering::Relaxed);
        }

        fn physical_stopped(&mut self) {
            self.physical_stopped.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn realization_failure_has_no_reservation_start_or_sink_callback() {
        let (profile, request) = admission_test_fixture(41);
        let (mut client, commands, reservations) = admission_test_client(profile);
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_starts = Arc::clone(&starts);
        let worker = std::thread::spawn(move || {
            let ModelRequest::Native(NativeSchedulerCommand::ResolveGenerate { reply, .. }) =
                commands.recv().expect("realization command")
            else {
                panic!("reservation/start preceded realization");
            };
            reply
                .send(Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::Tokenizer,
                    "synthetic realization failure",
                )))
                .unwrap();
            if matches!(
                commands.recv_timeout(Duration::from_millis(50)),
                Ok(ModelRequest::Native(
                    NativeSchedulerCommand::StartGenerate { .. }
                ))
            ) {
                worker_starts.fetch_add(1, Ordering::Relaxed);
            }
        });

        let error = match client.admit_generate(&request) {
            Err(error) => error,
            Ok(_) => panic!("realization failure unexpectedly reserved"),
        };
        assert_eq!(error.kind(), NativeRuntimeErrorKind::Tokenizer);
        worker.join().unwrap();
        assert_eq!(starts.load(Ordering::Relaxed), 0);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn successful_realization_freezes_tokens_then_reserves_and_starts_once() {
        let (profile, request) = admission_test_fixture(42);
        let (mut client, commands, reservations) = admission_test_client(profile);
        let realizations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sink_started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sink_terminal = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sink_physical_stopped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_realizations = Arc::clone(&realizations);
        let worker_starts = Arc::clone(&starts);
        let worker = std::thread::spawn(move || {
            let ModelRequest::Native(NativeSchedulerCommand::ResolveGenerate { reply, .. }) =
                commands.recv().expect("realization command")
            else {
                panic!("start preceded realization");
            };
            worker_realizations.fetch_add(1, Ordering::Relaxed);
            reply
                .send(Ok(NativeGenerateResolution {
                    prompt_token_ids: Arc::from([21, 22]),
                    eos_token_ids: Arc::from([5, 7]),
                    output_suppressed_token_ids: Arc::from([11, 17]),
                }))
                .unwrap();
            let ModelRequest::Native(NativeSchedulerCommand::PrepareGenerate {
                reservation_id,
                request,
                prepared,
                guard,
                ..
            }) = commands.recv().expect("prepare command")
            else {
                panic!("unexpected command after realization");
            };
            assert!(matches!(
                &request.prompt,
                super::super::NativePromptInput::TokenIds(ids) if ids.as_ref() == [21, 22]
            ));
            assert_eq!(request.eos_token_ids.as_ref(), [5, 7]);
            assert_eq!(request.output_suppressed_token_ids.as_ref(), [11, 17]);
            let identity = super::super::NativeSequenceIdentity {
                request_id: request.request_id,
                sequence_id: 1,
                worker_epoch: 2,
            };
            prepared.send(Ok(identity)).unwrap();

            let ModelRequest::Native(NativeSchedulerCommand::StartGenerate {
                reservation_id: started_reservation,
                sequence,
                mut sink,
                started,
            }) = commands.recv().expect("start command")
            else {
                panic!("unexpected command after prepare");
            };
            worker_starts.fetch_add(1, Ordering::Relaxed);
            assert_eq!(started_reservation, reservation_id);
            assert_eq!(sequence, identity);
            sink.started(identity).unwrap();
            started.send(Ok(())).unwrap();
            sink.terminal(super::super::NativePhysicalTerminal::Completed {
                reason: super::super::NativeStopReason::Requested,
            });
            assert_eq!(worker_starts.load(Ordering::Relaxed), 1);
            sink.physical_stopped();
            drop(guard);
        });

        let admission = client.admit_generate(&request).unwrap();
        assert_eq!(realizations.load(Ordering::Relaxed), 1);
        assert_eq!(
            admission.request().logical_prompt_tokens(),
            Some(2),
            "admission must freeze realized TokenIds"
        );
        let prepared = client.prepare_generate(admission).unwrap();
        assert_eq!(prepared.sequence().worker_epoch, 2);
        assert_eq!(prepared.request().logical_prompt_tokens(), Some(2));
        assert_eq!(sink_started.load(Ordering::Relaxed), 0);
        client
            .start_generate(
                prepared,
                Box::new(CountingGenerateSink {
                    started: Arc::clone(&sink_started),
                    terminal: Arc::clone(&sink_terminal),
                    physical_stopped: Arc::clone(&sink_physical_stopped),
                }),
            )
            .unwrap();
        worker.join().unwrap();
        assert_eq!(realizations.load(Ordering::Relaxed), 1);
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert_eq!(sink_started.load(Ordering::Relaxed), 1);
        assert_eq!(sink_terminal.load(Ordering::Relaxed), 1);
        assert_eq!(sink_physical_stopped.load(Ordering::Relaxed), 1);
        assert!(reservations.inner.lock().unwrap().reservations.is_empty());
        assert_eq!(reservations.release_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unsupported_cache_adoption_rejects_read_and_write_without_identity_aliasing() {
        let context = |namespace: &str, identity: u8, allow_read, allow_write| {
            super::super::NativeCacheContext {
                namespace: Arc::from(namespace),
                input_identity: [identity; 32],
                allow_read,
                allow_write,
            }
        };
        assert!(validate_cache_driver(&context("tenant-a", 1, false, false)).is_ok());
        for candidate in [
            context("tenant-a", 1, true, false),
            context("tenant-a", 1, false, true),
            context("tenant-b", 1, true, true),
            context("tenant-a", 2, true, true),
        ] {
            assert_eq!(
                validate_cache_driver(&candidate).unwrap_err().kind(),
                NativeRuntimeErrorKind::UnsupportedCapability
            );
        }
    }

    #[test]
    fn allocator_projection_and_reclaim_report_only_observed_bytes() {
        let snapshot = mlxcel_core::memory::MemorySnapshot {
            active_bytes: 100,
            peak_bytes: 150,
            cache_bytes: 25,
            limit_bytes: 1_000,
        };
        assert_eq!(
            project_memory_snapshot(snapshot, 2, 3, true),
            NativeMemorySnapshot {
                host_resident_bytes: 0,
                host_peak_bytes: 0,
                device_resident_bytes: 125,
                device_peak_bytes: 150,
                active_sequences: 2,
                queued_operations: 3,
            }
        );
        let after = mlxcel_core::memory::MemorySnapshot {
            cache_bytes: 5,
            ..snapshot
        };
        assert_eq!(
            project_reclaim_result(snapshot, after, 0, true),
            NativeReclaimResult {
                reclaimed_host_bytes: 0,
                reclaimed_device_bytes: 20,
                retained_active_sequences: 0,
            }
        );
        assert_eq!(
            project_reclaim_result(snapshot, after, 2, false),
            NativeReclaimResult {
                reclaimed_host_bytes: 20,
                reclaimed_device_bytes: 0,
                retained_active_sequences: 2,
            }
        );
        assert_eq!(project_worker_health(true), NativeWorkerHealth::Ready);
        assert_eq!(
            project_worker_health(false),
            NativeWorkerHealth::ReloadRequired,
            "a closed worker channel must make a zero memory/reclaim result distinguishable via descriptor health"
        );
    }
}
