//! Canonical managed model loading from a retained, producer-verified source.
//!
//! This is the physical loader used by the existing worker factory, not another
//! start API. It performs model I/O only. Cold package proof/inspection is not
//! reachable here. Each lazy MLX tensor owns its read-only native reader.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use mlxcel_core::weights::WeightMap;
use serde_json::Value;

use super::{
    IncrementalDecodeGuarantee, NativeBoundTokenizer, NativeLoadStatistics, NativeRuntimeError,
    NativeRuntimeErrorKind, NativeWorkerStartRequest,
};
use crate::embeddings::loader::{LoadedEmbeddingModel, QuantizationParams, quantization_params};
use crate::embeddings::model::EmbeddingModel;
use crate::models::bert::{BertArgs, BertVariant};
use crate::models::bert_heads::BertEmbeddingModel;
use crate::tokenizer::MlxcelTokenizer;

pub(crate) struct LoadedGeneration {
    pub(crate) model: crate::LoadedModel,
    pub(crate) tokenizer: NativeBoundTokenizer,
    pub(crate) eos: Vec<i32>,
}

/// Load the already admitted managed family, using its existing model constructor.
pub(crate) fn load_generation(
    request: &NativeWorkerStartRequest,
) -> Result<(LoadedGeneration, NativeLoadStatistics), NativeRuntimeError> {
    request.validate(request.binding.native_build_identity())?;
    if request.binding.architecture() != "gemma4_unified" {
        return Err(unsupported(
            "managed generation source has no supported family constructor",
        ));
    }
    let mut loader = Loader::new(request);
    let config = loader.json("config.json")?;
    loader.check_config(&config)?;
    let processor = loader.optional_json("processor_config.json")?;
    // Parsing is actual construction input validation, not another package report.
    let geometry: crate::vision::gemma4_unified_config::Gemma4UnifiedConfig =
        serde_json::from_value(config.clone())
            .map_err(|_| invalid("invalid Gemma 4 Unified configuration"))?;
    if !request.binding.modalities().image
        || request.binding.modalities().video != processor.is_some()
    {
        return Err(invalid(
            "model modalities disagree with the inspected load binding",
        ));
    }
    let (tokenizer, guarantee) = loader.tokenizer(&config)?;
    let bound = NativeBoundTokenizer::from_loaded(
        tokenizer,
        guarantee,
        &request.binding,
        &request.profile,
    )?;
    let eos = loader
        .optional_json("generation_config.json")?
        .map(|config| crate::loading::parse_generation_config_defaults(&config).eos_token_ids)
        .unwrap_or_default();
    if eos.len() > super::MAX_NATIVE_STOP_TOKEN_IDS || eos.iter().any(|id| *id < 0) {
        return Err(invalid(
            "generation EOS metadata exceeds native token bounds",
        ));
    }
    let (weights, backing) = loader.gemma_weights()?;
    // Match the existing constructor's optional-audio trigger. Configuration
    // alone does not build an audio embedder; the projection must also exist.
    // This inspects the already loaded weight table, not package proof bytes.
    let has_audio = geometry.audio_config.is_some()
        && weights.keys().any(|name| {
            crate::loading::normalize_gemma4_unified_key(name)
                == "embed_audio.embedding_projection.weight"
        });
    if request.binding.modalities().audio != has_audio {
        return Err(invalid(
            "audio load binding disagrees with the model projection",
        ));
    }
    loader.check_cancelled()?;
    let model = crate::loading::load_gemma4_unified_from_weights(config, weights, backing)
        .map_err(|_| invalid("verified Gemma 4 model construction failed"))?;
    loader.check_cancelled()?;
    Ok((
        LoadedGeneration {
            model,
            tokenizer: bound,
            eos,
        },
        loader.statistics,
    ))
}

/// Load the existing BERT/XLM-R encoder constructor and shared embedding policies.
pub(crate) fn load_embedding(
    request: &NativeWorkerStartRequest,
) -> Result<(LoadedEmbeddingModel, NativeLoadStatistics), NativeRuntimeError> {
    request.validate(request.binding.native_build_identity())?;
    let model_type = match request.binding.architecture() {
        "bert" => crate::models::ModelType::Bert,
        "xlm-roberta" => crate::models::ModelType::XlmRoberta,
        _ => {
            return Err(unsupported(
                "managed embedding source has no supported family constructor",
            ));
        }
    };
    let mut loader = Loader::new(request);
    let config = loader.json("config.json")?;
    // BERT's bound context is the effective embedding ceiling, not necessarily
    // the physical position table length. Validate that ceiling after sidecars.
    if config.get("model_type").and_then(Value::as_str) != Some(request.binding.architecture()) {
        return Err(invalid(
            "model configuration family differs from its inspected binding",
        ));
    }
    let variant = BertVariant::from_model_type(model_type)
        .ok_or_else(|| unsupported("embedding family has no encoder variant"))?;
    let args = BertArgs::from_config(&config, variant)
        .map_err(|_| invalid("invalid embedding encoder geometry"))?;
    let sentence = loader.optional_json("sentence_bert_config.json")?;
    let token_config = loader.optional_json("tokenizer_config.json")?;
    let ceiling = crate::embeddings::limits::derive_max_length_from_configs(
        Some(&config),
        token_config.as_ref(),
        sentence.as_ref(),
        true,
        None,
    )
    .min(args.max_sequence_length());
    if request.binding.max_context_positions().map(|v| v as usize) != Some(ceiling) {
        return Err(invalid(
            "embedding limits disagree with the inspected source binding",
        ));
    }
    let pooling = loader
        .optional_json(crate::embeddings::PoolingConfig::RELATIVE_PATH)?
        .as_ref()
        .map(crate::embeddings::PoolingConfig::parse)
        .transpose()
        .map_err(|_| invalid("embedding pooling configuration is unsupported"))?
        .unwrap_or(BertEmbeddingModel::DEFAULT_POOLING);
    // The canonical source contract consumes package pooling policy only.
    // Server-global CLI/environment overrides belong to the independent server
    // frontend and cannot alter this exact native load binding.
    let (tokenizer, _) = loader.tokenizer_with_config(&config, token_config.as_ref())?;
    let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(tokenizer);
    let pad_token_id = crate::embeddings::limits::resolve_pad_token_id_from_config(
        token_config.as_ref(),
        &tokenizer,
    );
    let vocab_size = crate::embeddings::limits::resolve_vocab_size(&config, &tokenizer);
    if vocab_size == 0 || pad_token_id as usize >= vocab_size {
        return Err(invalid(
            "embedding vocabulary or padding identity is invalid",
        ));
    }
    let mut weights = loader.weights(true)?;
    if crate::models::should_convert_bf16_to_f16()
        && !crate::models::config_has_quantization_metadata(&config)
        && crate::models::convert_bf16_weights(&mut weights)
    {
        crate::models::warn_bf16_precision();
    }
    let weights = crate::models::bert::sanitize(weights, false);
    let quant = quantization_params(&config).unwrap_or(QuantizationParams {
        group_size: 64,
        bits: 4,
    });
    loader.check_cancelled()?;
    let model = Box::new(
        BertEmbeddingModel::from_weights(&weights, args, quant, pooling)
            .map_err(|_| invalid("verified embedding model construction failed"))?,
    );
    let limits = crate::embeddings::limits::EmbeddingLimits {
        max_length: ceiling,
        dim: model.embedding_dim(),
        multi_vector: model.multi_vector(),
    };
    loader.check_cancelled()?;
    Ok((
        LoadedEmbeddingModel {
            model,
            tokenizer,
            limits,
            pad_token_id,
            vocab_size,
            model_type,
        },
        loader.statistics,
    ))
}

struct Loader<'a> {
    request: &'a NativeWorkerStartRequest,
    statistics: NativeLoadStatistics,
}

impl<'a> Loader<'a> {
    fn new(request: &'a NativeWorkerStartRequest) -> Self {
        Self {
            request,
            statistics: NativeLoadStatistics::default(),
        }
    }

    fn check_cancelled(&self) -> Result<(), NativeRuntimeError> {
        if self.request.cancellation.load(Ordering::Acquire) {
            Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::Cancelled,
                "native model loading cancelled",
            ))
        } else {
            Ok(())
        }
    }

    fn json(&mut self, label: &str) -> Result<Value, NativeRuntimeError> {
        let bytes = self.request.source.read_metadata(
            label,
            &self.request.cancellation,
            &mut self.statistics,
        )?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| invalid("declared model metadata is invalid JSON"))?;
        if !value.is_object() {
            return Err(invalid("declared model metadata must be an object"));
        }
        Ok(value)
    }

    fn optional_json(&mut self, label: &str) -> Result<Option<Value>, NativeRuntimeError> {
        if self.request.source.component(label).is_none() {
            return Ok(None);
        }
        self.json(label).map(Some)
    }

    fn check_config(&self, config: &Value) -> Result<(), NativeRuntimeError> {
        if config.get("model_type").and_then(Value::as_str)
            != Some(self.request.binding.architecture())
            || crate::context_window_from_config(config)
                != self
                    .request
                    .binding
                    .max_context_positions()
                    .map(|v| v as usize)
        {
            return Err(invalid(
                "model configuration differs from its inspected binding",
            ));
        }
        Ok(())
    }

    fn tokenizer(
        &mut self,
        config: &Value,
    ) -> Result<(MlxcelTokenizer, IncrementalDecodeGuarantee), NativeRuntimeError> {
        let sidecar = self.optional_json("tokenizer_config.json")?;
        self.tokenizer_with_config(config, sidecar.as_ref())
    }

    fn tokenizer_with_config(
        &mut self,
        config: &Value,
        sidecar: Option<&Value>,
    ) -> Result<(MlxcelTokenizer, IncrementalDecodeGuarantee), NativeRuntimeError> {
        let source = &self.request.source;
        let labels = source
            .components()
            .iter()
            .map(|component| component.label())
            .collect::<Vec<_>>();
        let selected = super::model_report::select_declared_tokenizer_primary(&labels)?
            .ok_or_else(|| invalid("verified source has no declared supported local tokenizer"))?;
        let label = selected.as_ref();
        let bytes =
            source.read_metadata(label, &self.request.cancellation, &mut self.statistics)?;
        let merges = if label == "vocab.json" {
            Some(source.read_metadata(
                "merges.txt",
                &self.request.cancellation,
                &mut self.statistics,
            )?)
        } else {
            None
        };
        let added_tokens = if label == "vocab.json" {
            self.optional_json("added_tokens.json")?
        } else {
            None
        };
        let guarantee = if label == "tokenizer.json" {
            let json: Value = serde_json::from_slice(&bytes)
                .map_err(|_| invalid("declared tokenizer JSON is malformed"))?;
            if super::is_proven_replace_byte_fallback_fuse(&json) {
                IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 }
            } else {
                IncrementalDecodeGuarantee::TerminalOnly
            }
        } else if label.ends_with(".tiktoken") || label == "tiktoken.model" {
            IncrementalDecodeGuarantee::RawTokenBytes
        } else {
            IncrementalDecodeGuarantee::TerminalOnly
        };
        let tokenizer = crate::tokenizer::load_tokenizer_from_declared_bytes(
            label,
            &bytes,
            sidecar,
            config,
            merges.as_deref(),
            added_tokens.as_ref(),
        )
        .map_err(|_| invalid("declared local tokenizer construction failed"))?;
        self.check_cancelled()?;
        Ok((tokenizer, guarantee))
    }

    fn weight_labels(&mut self, embedding: bool) -> Result<Vec<String>, NativeRuntimeError> {
        let index = self.optional_json("model.safetensors.index.json")?;
        let mut indexed = BTreeSet::new();
        if let Some(index) = index {
            let map = index
                .get("weight_map")
                .and_then(Value::as_object)
                .filter(|map| !map.is_empty())
                .ok_or_else(|| invalid("weight index has no tensor map"))?;
            for value in map.values() {
                let label = value
                    .as_str()
                    .ok_or_else(|| invalid("weight index shard is not a label"))?;
                if label.contains('/')
                    || !label.ends_with(".safetensors")
                    || self.request.source.component(label).is_none()
                {
                    return Err(invalid("weight index references an undeclared shard"));
                }
                indexed.insert(label.to_owned());
            }
        }
        let labels = self
            .request
            .source
            .components()
            .iter()
            .filter_map(|component| {
                let label = component.label();
                if !label.ends_with(".safetensors") {
                    return None;
                }
                let parts = label.split('/').collect::<Vec<_>>();
                if parts.iter().any(|p| p.starts_with('.'))
                    || parts.len() > if embedding { 2 } else { 1 }
                {
                    return None;
                }
                let basename = parts.last().copied().unwrap_or("");
                if embedding
                    && [
                        "adapter_model.safetensors",
                        "optimizer.safetensors",
                        "training_args.safetensors",
                    ]
                    .contains(&basename)
                {
                    return None;
                }
                if embedding && parts.len() == 1 && !indexed.is_empty() && !indexed.contains(label)
                {
                    return None;
                }
                Some(label.to_owned())
            })
            .collect::<Vec<_>>();
        if !labels.iter().any(|label| !label.contains('/')) {
            return Err(invalid("verified source has no primary model weight shard"));
        }
        Ok(labels)
    }

    fn gemma_weights(
        &mut self,
    ) -> Result<(WeightMap, crate::models::Gemma4WeightBacking), NativeRuntimeError> {
        let labels = self.weight_labels(false)?;
        let mut mappings = Vec::new();
        mappings.try_reserve_exact(labels.len()).map_err(|_| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::ResourceLimit,
                "native backing table allocation failed",
            )
        })?;
        for label in labels {
            self.check_cancelled()?;
            let backing = self.request.source.materialize(
                &label,
                &self.request.cancellation,
                &mut self.statistics,
            )?;
            #[cfg(unix)]
            mappings.push(backing.map_read_only()?);
            #[cfg(not(unix))]
            return Err(unsupported(
                "immutable model mapping is unavailable on this platform",
            ));
        }
        self.check_cancelled()?;
        crate::models::load_gemma4_unified_weights_from_mmaps(mappings)
            .map_err(|_| invalid("declared Gemma weight conversion failed"))
    }

    fn weights(&mut self, embedding: bool) -> Result<WeightMap, NativeRuntimeError> {
        let labels = self.weight_labels(embedding)?;
        let mut weights = WeightMap::new();
        for label in labels {
            self.check_cancelled()?;
            let backing = self.request.source.materialize(
                &label,
                &self.request.cancellation,
                &mut self.statistics,
            )?;
            #[cfg(unix)]
            let loaded = mlxcel_core::weights::load_safetensors_file(&backing)
                .map_err(|_| invalid("declared weight shard is malformed or unsupported"))?;
            #[cfg(not(unix))]
            let loaded: WeightMap = return Err(unsupported(
                "verified native reader is unavailable on this platform",
            ));
            weights.try_reserve(loaded.len()).map_err(|_| {
                NativeRuntimeError::new(
                    NativeRuntimeErrorKind::ResourceLimit,
                    "weight table allocation failed",
                )
            })?;
            for (name, array) in loaded {
                let name = if let Some((folder, _)) = label.split_once('/') {
                    format!("{folder}.{name}")
                } else {
                    name
                };
                if weights.insert(name, array).is_some() {
                    return Err(invalid(
                        "declared weight shards contain duplicate tensor names",
                    ));
                }
            }
            // Native Load primitives own the duplicated read-only descriptor.
            // Dropping the transfer backing cannot invalidate retained lazy arrays.
            drop(backing);
        }
        self.check_cancelled()?;
        Ok(weights)
    }
}

fn invalid(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidPackage, message)
}
fn unsupported(message: &'static str) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::UnsupportedCapability, message)
}

#[cfg(test)]
#[path = "load_residence_tests.rs"]
mod tests;
