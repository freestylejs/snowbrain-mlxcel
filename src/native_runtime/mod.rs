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

//! Transport-neutral public contract for mlxcel's managed physical runtime.
//!
//! This module deliberately exposes no server events, OpenAI semantics, MLX
//! arrays, cache objects, or `LoadedModel` variants.

mod builder;
mod detokenizer;
mod error;
mod model_report;
mod operation_driver;
mod profile;
mod request;
mod sink;

pub use builder::{NativeManagedEngine, NativeRuntimeBuilder, NativeWorkerStartRequest};
pub use detokenizer::{
    IncrementalDecodeGuarantee, NativeIncrementalDecoder, NativeTokenizerDecoder, StableDecodeDelta,
};
pub use error::{
    MAX_NATIVE_ERROR_MESSAGE_BYTES, NativeBoundedErrorMessage, NativeErrorMessageTooLong,
    NativeRuntimeError, NativeRuntimeErrorKind, TokenizerError, TokenizerErrorKind,
};
pub use model_report::{
    MAX_NATIVE_INSPECTION_BYTES, MAX_NATIVE_PACKAGE_BYTES, MAX_NATIVE_PACKAGE_COMPONENTS,
    MAX_NATIVE_PACKAGE_LABEL_BYTES, MAX_NATIVE_PACKAGE_LABEL_TOTAL_BYTES, MAX_NATIVE_REPORT_ITEMS,
    MAX_NATIVE_REPORT_TEXT_BYTES, NativeImmutablePackage, NativeManagedCapabilities,
    NativeModalitySet, NativeModelCapabilities, NativeModelReport, NativePackageArtifactSource,
    NativePackageComponent, NativePackageLimits, NativeTokenizerFormat, NativeTokenizerReport,
};
pub use operation_driver::{MlxNativeOperationClass, NativeOperationDriver};
pub use profile::{
    NativeAdapterBinding, NativeBatchPolicy, NativeBuildIdentity, NativeBuildSource,
    NativeCachePolicy, NativeCacheTopology, NativeConsumerBuildFragment, NativeConsumerBuildRecord,
    NativeContextPolicy, NativeDeviceClass, NativeDevicePolicy, NativeDrafterTopology,
    NativeEffectiveBuildInputs, NativeEngineDescriptor, NativeLinkMode, NativeModelOwnedStateMode,
    NativeNamedBuildInput, NativeNativeSource, NativeNativeSourceKind, NativeOffloadPolicy,
    NativePostLinkAttestation, NativePostLinkRecord, NativePreparedMultimodalClass,
    NativeResolvedCargoPackage, NativeSchedulerTopology, NativeSelfBuildPackage,
    NativeSequenceStateTopology, NativeWorkerConfig, NativeWorkerHealth, NativeWorkerPolicy,
};
pub use request::{
    MAX_NATIVE_CONSTRAINT_BYTES, MAX_NATIVE_EMBED_AGGREGATE_BYTES,
    MAX_NATIVE_EMBED_AGGREGATE_TOKENS, MAX_NATIVE_EMBED_INPUTS, MAX_NATIVE_LABEL_BYTES,
    MAX_NATIVE_PREPARED_BYTES, MAX_NATIVE_PREPARED_MEDIA_ITEMS, MAX_NATIVE_PREPARED_MODALITIES,
    MAX_NATIVE_PROMPT_TOKENS, MAX_NATIVE_STOP_TOKEN_IDS, MAX_NATIVE_TENSOR_RANK,
    MAX_NATIVE_TEXT_BYTES, NativeCacheContext, NativeConstraintPlan, NativeEmbedAdmission,
    NativeEmbedRequest, NativeEmbeddingInput, NativeEmbeddingLimits, NativeEmbeddingNormalization,
    NativeExecutionHandle, NativeExecutionKind, NativeGenerateAdmission, NativeGenerateRequest,
    NativeGenerationLimits, NativeOwnedPreparedInput, NativePreparedPrefill, NativePromptInput,
    NativeReclaimRequest, NativeReclaimResult, NativeSamplingConfig, NativeSchedulingHints,
    NativeSpeculationPolicy, NativeSpeculationPreference, NativeSpeculativeFidelity,
};
pub use sink::{
    MAX_NATIVE_ACCEPTED_UNITS, MAX_NATIVE_EMBEDDING_DIMENSION, MAX_NATIVE_STABLE_BYTES_PER_CHUNK,
    MAX_NATIVE_STABLE_BYTES_PER_UNIT, NativeAcceptedChunk, NativeAcceptedSource,
    NativeAcceptedUnit, NativeConstraintTransition, NativeContinuationDirective,
    NativeDirectiveApplication, NativeEmbeddingOutput, NativeEmbeddingSink, NativeGenerateSink,
    NativeMemoryEvent, NativeMemoryPressure, NativeMemorySnapshot, NativePhysicalEvent,
    NativePhysicalTerminal, NativePrefixFacts, NativeRecoverability, NativeResumeCapability,
    NativeSequenceIdentity, NativeStopReason, NativeUsageDelta,
};
