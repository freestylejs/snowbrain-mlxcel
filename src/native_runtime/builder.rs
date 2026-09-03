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

//! Builder and managed-engine source contracts.

use std::fmt;
use std::sync::Arc;

use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::model_report::{NativeImmutablePackage, NativeModelReport, inspect_package};
use super::operation_driver::MlxNativeOperationClass;
use super::profile::{
    NativeBuildIdentity, NativeConsumerBuildFragment, NativeEngineDescriptor, NativeWorkerConfig,
    embedded_native_build_identity,
};
use super::request::{
    NativeEmbedAdmission, NativeEmbedRequest, NativeExecutionHandle, NativeGenerateAdmission,
    NativeGenerateRequest, NativeReclaimRequest, NativeReclaimResult,
};
use super::sink::{NativeEmbeddingSink, NativeGenerateSink, NativeMemorySnapshot};

/// Owned worker-start request. Package authority moves into the worker factory.
///
/// The factory must retain this authority for the complete worker lifetime. A
/// path-based lazy/mapped model loader may return only after its weights have
/// been eagerly materialized and a device barrier has detached runtime state
/// from staging bytes; otherwise the factory must retain proven no-write OS
/// guards for every backing artifact until worker teardown.
pub struct NativeWorkerStartRequest {
    pub package: NativeImmutablePackage,
    pub profile: NativeWorkerConfig,
    pub initial_operation: MlxNativeOperationClass,
}

impl NativeWorkerStartRequest {
    fn validate(&self) -> Result<(), NativeRuntimeError> {
        self.profile.validate()?;
        if self.profile.operation_driver().operation_class() != self.initial_operation {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "worker operation driver does not match the requested operation class",
            ));
        }
        if self.initial_operation == MlxNativeOperationClass::SpeechToText {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedOperation,
                "speech-to-text has no managed request/admission/sink lifecycle",
            ));
        }
        if self.package.identity() != self.profile.package_identity() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidPackage,
                "worker profile is bound to a different immutable package",
            ));
        }
        let report = inspect_package(&self.package)?;
        if report.report_identity() != self.profile.report_identity()
            || report.operation_class() != self.initial_operation
            || report.context_window() != self.profile.model_context_window()
            || report.tokenizer().incremental_decode() != self.profile.decoder_guarantee()
        {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::UnsupportedCapability,
                "worker profile authority disagrees with current immutable inspection",
            ));
        }
        Ok(())
    }
}

/// Physical managed engine. Implementations remain owned by root mlxcel.
///
/// # Lifecycle contract
///
/// `admit_*` must validate every bounded request field before retaining any
/// bytes or reserving capacity. A rejected admission reserves nothing. The
/// returned non-cloneable capability owns its reservation; dropping it before
/// a successful start releases that reservation.
///
/// `start_*` keeps that release guard armed until the corresponding
/// `sink.started(...)` callback succeeds. It may return `Err` only before that
/// handshake, in which case the reservation is released. After the handshake,
/// it returns a non-cloneable handle and reports every later failure exactly
/// once through `sink.terminal(...)`, never through the returned `Result`.
/// Generated chunks, sink-returned continuation directives, and embedding
/// outputs must pass their public validation methods before retention or use;
/// embedding outputs must use request-bound `validate_for`, not only the
/// global shape validator.
pub trait NativeManagedEngine: Send {
    fn descriptor(&self) -> NativeEngineDescriptor;

    fn admit_generate(
        &mut self,
        request: &NativeGenerateRequest,
    ) -> Result<NativeGenerateAdmission, NativeRuntimeError>;

    fn start_generate(
        &mut self,
        admission: NativeGenerateAdmission,
        sink: Box<dyn NativeGenerateSink + Send>,
    ) -> Result<NativeExecutionHandle, NativeRuntimeError>;

    fn admit_embed(
        &mut self,
        request: &NativeEmbedRequest,
    ) -> Result<NativeEmbedAdmission, NativeRuntimeError>;

    fn start_embed(
        &mut self,
        admission: NativeEmbedAdmission,
        sink: Box<dyn NativeEmbeddingSink + Send>,
    ) -> Result<NativeExecutionHandle, NativeRuntimeError>;

    fn cancel(&mut self, handle: &NativeExecutionHandle);

    fn memory_snapshot(&self) -> NativeMemorySnapshot;

    fn reclaim(&mut self, request: NativeReclaimRequest) -> NativeReclaimResult;
}

/// Internal root-mlxcel construction boundary implemented by Order 03. The
/// concrete implementation owns the dedicated physical thread and constructs
/// any non-`Send` tokenizer/decoder on that thread, never inside a sink.
pub(crate) trait NativeWorkerFactory: Send + Sync {
    fn start_worker(
        &self,
        request: NativeWorkerStartRequest,
    ) -> Result<Box<dyn NativeManagedEngine>, NativeRuntimeError>;
}

/// Build/package authority shared by native frontends.
pub struct NativeRuntimeBuilder {
    build_identity: NativeBuildIdentity,
    worker_factory: Arc<dyn NativeWorkerFactory>,
}

impl fmt::Debug for NativeRuntimeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeRuntimeBuilder")
            .field("build_identity", &self.build_identity)
            .finish_non_exhaustive()
    }
}

impl NativeRuntimeBuilder {
    /// Construct the root mlxcel implementation for the current linked build.
    pub fn current_process(
        consumer: NativeConsumerBuildFragment,
    ) -> Result<Self, NativeRuntimeError> {
        let build_identity = embedded_native_build_identity(consumer)?;
        Ok(Self {
            worker_factory: Arc::new(super::engine::NativeEngineFactory::new(
                build_identity.clone(),
            )),
            build_identity,
        })
    }

    /// Root-only constructor. Order 03 supplies the concrete factory; callers
    /// cannot inject a hand-authored build identity into the public builder.
    pub(crate) fn from_embedded_build(
        consumer: NativeConsumerBuildFragment,
        worker_factory: Arc<dyn NativeWorkerFactory>,
    ) -> Result<Self, NativeRuntimeError> {
        Ok(Self {
            build_identity: embedded_native_build_identity(consumer)?,
            worker_factory,
        })
    }

    #[must_use]
    pub fn build_identity(&self) -> NativeBuildIdentity {
        self.build_identity.clone()
    }

    pub fn inspect_package(
        &self,
        package: &NativeImmutablePackage,
    ) -> Result<NativeModelReport, NativeRuntimeError> {
        inspect_package(package)
    }

    pub fn start_worker(
        &self,
        request: NativeWorkerStartRequest,
    ) -> Result<Box<dyn NativeManagedEngine>, NativeRuntimeError> {
        request.validate()?;
        self.worker_factory.start_worker(request)
    }
}
