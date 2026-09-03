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

//! Single dedicated-thread worker for the embedding model.
//!
//! MLX work is thread-affine (see [`super::audio_worker`] for the full
//! argument): the thread that loads the weights must be the thread that
//! evaluates them. [`EmbeddingWorker`] owns that thread. It installs a
//! thread-local default stream, loads the [`LoadedEmbeddingModel`] on
//! itself, wraps it in an [`EmbeddingEngine`], and serves commands from a
//! bounded managed admission one at a time. Tokenization, length-sorted
//! micro-batching, pooling and normalization all run inside the engine on
//! this thread; the HTTP side only ever sees `Vec<f32>` vectors.
//!
//! The native reservation book bounds active plus queued work (capacity
//! exhaustion = [`EmbeddingError::QueueFull`]), a per-request reply timeout
//! signals cancellation ([`EmbeddingError::Timeout`]), and
//! `catch_unwind` around every engine call so one bad request cannot kill
//! the worker for the rest of the process.

use std::path::Path;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::embeddings::{
    EmbedOptions, EmbedReply, EmbeddingLoadOptions, ImageInput, LoadedEmbeddingModel,
    load_embedding_model_with_options,
};
use crate::models::ModelType;
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};

/// Static facts about the loaded model, sent back once the worker is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbeddingWorkerInfo {
    pub dim: usize,
    pub max_length: usize,
    pub vocab_size: usize,
    pub multi_vector: bool,
    pub supports_images: bool,
    pub batch_size: usize,
    pub model_type: ModelType,
    /// The pooling this checkpoint resolved, reported by `/props` (#1452).
    pub pooling: crate::embeddings::PoolingMode,
    /// The normalization a request that names none gets (#1452).
    pub embd_normalize: crate::embeddings::EmbdNormalize,
}

/// Owns the dedicated thread that loads and runs one embedding model.
#[derive(Debug)]
pub(crate) struct EmbeddingWorker {
    engine: crate::native_runtime::engine::MlxcelServerEmbeddingManagedEngine,
    info: EmbeddingWorkerInfo,
    handle: Option<JoinHandle<()>>,
}

impl EmbeddingWorker {
    /// Spawn the worker thread and block until the model has loaded.
    ///
    /// `loader` runs on the worker thread so every array it creates belongs
    /// to that thread's MLX context. `queue_depth` bounds the command
    /// channel (clamped to at least one), `request_timeout` bounds the reply
    /// wait (a zero falls back to the default), and `batch_size` is the
    /// micro-batch width handed to the engine.
    pub(crate) fn spawn<L>(
        thread_name: &str,
        queue_depth: usize,
        request_timeout: Duration,
        batch_size: usize,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel> + Send + 'static,
    {
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (engine, info, handle) =
            crate::native_runtime::engine::spawn_mlxcel_server_embedding_managed_engine(
                thread_name,
                queue_depth,
                request_timeout,
                batch_size,
                loader,
            )?;
        Ok(Self {
            engine,
            info,
            handle: Some(handle),
        })
    }

    /// Static facts about the loaded model.
    pub(crate) fn info(&self) -> EmbeddingWorkerInfo {
        self.info
    }

    pub(crate) fn embed_texts(
        &self,
        texts: Vec<String>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let pending = self.engine.embed_texts(texts, opts)?;
        pending.wait()
    }

    pub(crate) fn embed_tokens(
        &self,
        rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let pending = self.engine.embed_tokens(rows, opts)?;
        pending.wait()
    }

    pub(crate) fn embed_image(
        &self,
        image: ImageInput,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let pending = self.engine.embed_image(image, opts)?;
        pending.wait()
    }
}

impl Drop for EmbeddingWorker {
    fn drop(&mut self) {
        let _ = self.engine.shutdown();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// [`EmbeddingModelProvider`] backed by an [`EmbeddingWorker`].
pub struct EmbeddingWorkerProvider {
    worker: EmbeddingWorker,
    model_id: String,
    created_at: i64,
}

impl std::fmt::Debug for EmbeddingWorkerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingWorkerProvider")
            .field("model_id", &self.model_id)
            .field("info", &self.worker.info())
            .finish()
    }
}

impl EmbeddingWorkerProvider {
    /// Spawn the worker and load the checkpoint at `model_path` on it.
    ///
    /// `model_id` is the id reported in responses and `/v1/models`. Returns
    /// `Err` when the thread cannot start or the checkpoint fails to load.
    pub fn load(
        model_path: &Path,
        model_id: String,
        batch_size: usize,
        queue_depth: usize,
        request_timeout: Duration,
        load_options: EmbeddingLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        Self::from_loader(
            model_id,
            batch_size,
            queue_depth,
            request_timeout,
            move || load_embedding_model_with_options(&model_path, load_options),
        )
    }

    /// Spawn the worker around an arbitrary loader (the production path
    /// above, or a test stub).
    pub(crate) fn from_loader<L>(
        model_id: String,
        batch_size: usize,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel> + Send + 'static,
    {
        let worker = EmbeddingWorker::spawn(
            "embedding-worker",
            queue_depth,
            request_timeout,
            batch_size,
            loader,
        )?;
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Ok(Self {
            worker,
            model_id,
            created_at,
        })
    }

    /// Static facts about the loaded model.
    pub(crate) fn info(&self) -> EmbeddingWorkerInfo {
        self.worker.info()
    }
}

impl EmbeddingModelProvider for EmbeddingWorkerProvider {
    fn embed_texts(
        &self,
        texts: Vec<String>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_texts(texts, opts)
    }

    fn embed_tokens(
        &self,
        token_rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_tokens(token_rows, opts)
    }

    fn embed_image(
        &self,
        image: ImageInput,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_image(image, opts)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn dim(&self) -> usize {
        self.worker.info().dim
    }

    fn multi_vector(&self) -> bool {
        self.worker.info().multi_vector
    }

    fn pooling(&self) -> crate::embeddings::PoolingMode {
        self.worker.info().pooling
    }

    fn embd_normalize(&self) -> crate::embeddings::EmbdNormalize {
        self.worker.info().embd_normalize
    }

    fn supports_images(&self) -> bool {
        self.worker.info().supports_images
    }

    fn vocab_size(&self) -> usize {
        self.worker.info().vocab_size
    }

    fn max_length(&self) -> usize {
        self.worker.info().max_length
    }
}

#[cfg(test)]
#[path = "embedding_worker_tests.rs"]
mod tests;
