use super::*;
use crate::native_runtime::*;
use mlxcel_core::generate::LanguageModel;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

struct Source {
    bytes: BTreeMap<String, Vec<u8>>,
    reads: Mutex<BTreeMap<String, u64>>,
}
impl NativeVerifiedArtifactSource for Source {
    fn read_at(
        &self,
        label: &str,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<usize, NativeRuntimeError> {
        let bytes = self
            .bytes
            .get(label)
            .ok_or_else(|| invalid("test artifact absent"))?;
        let start = usize::try_from(offset).map_err(|_| invalid("test offset"))?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| invalid("test offset"))?;
        destination.copy_from_slice(bytes.get(start..end).ok_or_else(|| invalid("test range"))?);
        *self
            .reads
            .lock()
            .unwrap()
            .entry(label.to_owned())
            .or_default() += destination.len() as u64;
        Ok(destination.len())
    }
}

fn source(bytes: BTreeMap<String, Vec<u8>>) -> (NativeVerifiedSource, Arc<Source>) {
    let components = bytes
        .iter()
        .map(|(name, bytes)| {
            super::super::NativeVerifiedComponent::new(name, bytes.len() as u64, [6; 32]).unwrap()
        })
        .collect();
    let producer = Arc::new(Source {
        bytes,
        reads: Mutex::new(BTreeMap::new()),
    });
    (
        NativeVerifiedSource::new(
            [1; 32],
            components,
            producer.clone(),
            super::super::NativeLoadLimits {
                max_read_bytes: 1024,
                ..Default::default()
            },
        )
        .unwrap(),
        producer,
    )
}
fn policy(driver: super::super::NativeOperationDriver) -> super::super::NativeWorkerPolicy {
    use super::super::*;
    NativeWorkerPolicy {
        operation_driver: driver,
        scheduler_topology: NativeSchedulerTopology::SingleSequence,
        sequence_state_topology: NativeSequenceStateTopology::Independent,
        context: NativeContextPolicy {
            max_context_tokens: 24,
            max_position: 24,
        },
        batch: NativeBatchPolicy {
            max_active_sequences: 1,
            max_queued_operations: 2,
            max_batch_tokens: 24,
            prefill_chunk_tokens: 24,
        },
        cache: NativeCachePolicy {
            topology: NativeCacheTopology::Dense,
            max_resident_bytes: 1024 * 1024,
            prefix_reuse: false,
        },
        device: NativeDevicePolicy {
            class: NativeDeviceClass::Metal,
            offload: NativeOffloadPolicy::DeviceResident,
            max_device_bytes: 64 * 1024 * 1024,
        },
        model_owned_state: NativeModelOwnedStateMode::Disabled,
        adapter: None,
        drafter: NativeDrafterTopology::Disabled,
        prepared_multimodal: NativePreparedMultimodalClass::Disabled,
        max_prompt_tokens: if driver == NativeOperationDriver::Embedding {
            24
        } else {
            20
        },
        max_generation_tokens: 4,
        max_embedding_inputs: 2,
        max_embedding_dimensions: 32,
        max_accepted_chunk_units: 32,
        max_constraint_bytes: 4096,
    }
}
fn request(
    source: NativeVerifiedSource,
    architecture: &str,
    driver: super::super::NativeOperationDriver,
) -> NativeWorkerStartRequest {
    use super::super::*;
    let binding = NativeWorkerLoadBinding::new(
        [1; 32],
        [2; 32],
        [3; 32],
        architecture,
        true,
        Some(24),
        NativeModalitySet {
            text: true,
            image: architecture == "gemma4_unified",
            audio: architecture == "gemma4_unified",
            video: architecture == "gemma4_unified",
        },
        NativeTokenizerLoadBinding::new([4; 32], "producer-tokenizer").unwrap(),
    )
    .unwrap();
    NativeWorkerStartRequest {
        source,
        profile: NativeWorkerConfig::try_from_binding(&binding, policy(driver)).unwrap(),
        binding,
        initial_operation: driver.operation_class(),
        cancellation: Arc::new(AtomicBool::new(false)),
    }
}
fn safetensors(shapes: BTreeMap<String, Vec<usize>>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape) in shapes {
        let start = payload.len();
        let count = shape.iter().product::<usize>();
        let value = if name.contains("norm") || name.ends_with("layer_scalar") {
            1.0_f32
        } else {
            0.01_f32
        };
        for _ in 0..count {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    while encoded.len() % 8 != 0 {
        encoded.push(b' ');
    }
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend(payload);
    bytes
}
fn gemma_bytes() -> BTreeMap<String, Vec<u8>> {
    gemma_bytes_with_audio_projection(true)
}

fn gemma_bytes_with_audio_projection(present: bool) -> BTreeMap<String, Vec<u8>> {
    let (mut tensors, mut config, processor) = crate::loading::native_unified_fixture();
    if !present {
        tensors.remove("embed_audio.embedding_projection.weight");
    }
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub_gemma_sequence_byte_fallback();
    config["text_config"]["vocab_size"] = 257.into();
    config["text_config"]["max_position_embeddings"] = 24.into();
    let mut shapes = tensors
        .into_iter()
        .map(|(name, metadata)| (name, metadata.shape().to_vec()))
        .collect::<BTreeMap<_, _>>();
    shapes.insert(
        "language_model.model.embed_tokens.weight".to_owned(),
        vec![257, 8],
    );
    BTreeMap::from([
        ("model.safetensors".to_owned(), safetensors(shapes)),
        (
            "config.json".to_owned(),
            serde_json::to_vec(&config).unwrap(),
        ),
        (
            "processor_config.json".to_owned(),
            serde_json::to_vec(&processor).unwrap(),
        ),
        (
            "tokenizer.json".to_owned(),
            tokenizer
                .hf_tokenizer()
                .unwrap()
                .to_string(false)
                .unwrap()
                .into_bytes(),
        ),
    ])
}

#[test]
fn canonical_gemma_audio_config_without_projection_preserves_supported_loading() {
    let (verified, _) = source(gemma_bytes_with_audio_projection(false));
    let mut load = request(
        verified,
        "gemma4_unified",
        NativeOperationDriver::AutoregressiveGenerate,
    );
    let binding = &load.binding;
    let mut modalities = binding.modalities();
    modalities.audio = false;
    load.binding = NativeWorkerLoadBinding::new(
        binding.source_identity(),
        binding.report_identity(),
        binding.native_build_identity(),
        binding.architecture(),
        binding.architecture_supported(),
        binding.max_context_positions(),
        modalities,
        binding.tokenizer().clone(),
    )
    .unwrap();
    load.profile = NativeWorkerConfig::try_from_binding(
        &load.binding,
        policy(NativeOperationDriver::AutoregressiveGenerate),
    )
    .unwrap();
    let (loaded, _) =
        load_generation(&load).expect("the existing family supports absent audio projection");
    drop(load);
    let ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut cache = loaded.model.make_caches();
    let logits = loaded.model.forward(&ids, &mut cache, None);
    mlxcel_core::eval(&logits);
    assert!(!mlxcel_core::array_to_raw_bytes(&logits).is_empty());
}

#[test]
fn canonical_gemma_audio_binding_rejects_absent_projection() {
    let (verified, _) = source(gemma_bytes_with_audio_projection(false));
    let load = request(
        verified,
        "gemma4_unified",
        NativeOperationDriver::AutoregressiveGenerate,
    );
    assert_eq!(
        load_generation(&load).err().unwrap().kind(),
        NativeRuntimeErrorKind::InvalidPackage
    );
}

#[test]
fn canonical_preflight_rejects_wrong_build_and_source_before_reads() {
    let (verified, producer) = source(bert_bytes());
    let mut load = request(verified, "bert", NativeOperationDriver::Embedding);
    assert_eq!(
        load.validate([99; 32]).unwrap_err().kind(),
        NativeRuntimeErrorKind::BuildIdentityMismatch,
    );
    load.source = NativeVerifiedSource::new(
        [9; 32],
        load.source.components().to_vec(),
        producer.clone(),
        load.source.limits(),
    )
    .unwrap();
    assert_eq!(
        load.validate([3; 32]).unwrap_err().kind(),
        NativeRuntimeErrorKind::InvalidPackage,
    );
    assert!(producer.reads.lock().unwrap().is_empty());
}

#[test]
fn canonical_bert_rejects_invalid_geometry_before_weights() {
    let mut bytes = bert_bytes();
    let mut config: serde_json::Value = serde_json::from_slice(&bytes["config.json"]).unwrap();
    config["hidden_size"] = 0.into();
    bytes.insert(
        "config.json".to_owned(),
        serde_json::to_vec(&config).unwrap(),
    );
    let (verified, producer) = source(bytes);
    let load = request(verified, "bert", NativeOperationDriver::Embedding);
    assert_eq!(
        load_embedding(&load).err().unwrap().kind(),
        NativeRuntimeErrorKind::InvalidPackage
    );
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}

#[test]
fn canonical_loader_rejects_malformed_metadata_before_weights() {
    let mut bytes = bert_bytes();
    bytes.insert("config.json".to_owned(), b"{invalid".to_vec());
    let (verified, producer) = source(bytes);
    let load = request(verified, "bert", NativeOperationDriver::Embedding);
    assert_eq!(
        load_embedding(&load).err().unwrap().kind(),
        NativeRuntimeErrorKind::InvalidPackage
    );
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}

#[test]
fn canonical_gemma_source_load_reload_evaluates_after_source_drop() {
    let bytes = gemma_bytes();
    let weight_bytes = bytes["model.safetensors"].len() as u64;
    let (source, producer) = source(bytes);
    let mut models = Vec::new();
    for _ in 0..2 {
        let request = request(
            source.clone(),
            "gemma4_unified",
            super::super::NativeOperationDriver::AutoregressiveGenerate,
        );
        let (loaded, statistics) = load_generation(&request).unwrap();
        assert_eq!(statistics.backing_bytes, weight_bytes);
        assert!(statistics.metadata_bytes > 0);
        assert!(statistics.largest_read <= 1024);
        drop(request);
        models.push(loaded.model);
    }
    assert_eq!(
        producer.reads.lock().unwrap()["model.safetensors"],
        2 * weight_bytes
    );
    let lifetime = Arc::downgrade(&producer);
    drop(source);
    drop(producer);
    assert!(lifetime.upgrade().is_none());
    let mut outputs = Vec::new();
    for model in models {
        let ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
        let mut cache = model.make_caches();
        let logits = model.forward(&ids, &mut cache, None);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&logits);
        let values = mlxcel_core::array_to_raw_bytes(&logits)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
        outputs.push(values);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn canonical_loader_rejects_missing_tokenizer_without_reading_weight_bytes() {
    let mut bytes = gemma_bytes();
    bytes.remove("tokenizer.json");
    let (source, producer) = source(bytes);
    let request = request(
        source,
        "gemma4_unified",
        super::super::NativeOperationDriver::AutoregressiveGenerate,
    );
    assert!(load_generation(&request).is_err());
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}

#[test]
fn immutable_gemma_backing_preserves_fp8_conversion_not_uint8_reinterpretation() {
    let header = serde_json::to_vec(&serde_json::json!({
        "language_model.test.weight": {"dtype":"F8_E4M3", "shape":[2], "data_offsets":[0,2]}
    }))
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&[0x38, 0x40]); // E4M3: 1.0 and 2.0, not integer 56 and 64.
    let (source, producer) = source(BTreeMap::from([("model.safetensors".to_owned(), bytes)]));
    let mut stats = NativeLoadStatistics::default();
    let backing = source
        .materialize("model.safetensors", &AtomicBool::new(false), &mut stats)
        .unwrap();
    let mapping = backing.map_read_only().unwrap();
    drop(backing);
    drop(source);
    drop(producer);
    let (mut weights, residence) =
        crate::models::load_gemma4_unified_weights_from_mmaps(vec![mapping]).unwrap();
    let value = weights.remove("language_model.test.weight").unwrap();
    drop(weights);
    drop(residence);
    let value = mlxcel_core::astype(&value, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&value);
    let actual = mlxcel_core::array_to_raw_bytes(&value)
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, [1.0, 2.0]);
}

fn bert_bytes() -> BTreeMap<String, Vec<u8>> {
    let mut shapes = BTreeMap::from([
        (
            "bert.embeddings.word_embeddings.weight".to_owned(),
            vec![32, 8],
        ),
        (
            "bert.embeddings.position_embeddings.weight".to_owned(),
            vec![24, 8],
        ),
        (
            "bert.embeddings.token_type_embeddings.weight".to_owned(),
            vec![2, 8],
        ),
        ("bert.embeddings.LayerNorm.weight".to_owned(), vec![8]),
    ]);
    for index in 0..2 {
        let base = format!("bert.encoder.layer.{index}");
        for name in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            shapes.insert(format!("{base}.{name}.weight"), vec![8, 8]);
        }
        shapes.insert(format!("{base}.intermediate.dense.weight"), vec![16, 8]);
        shapes.insert(format!("{base}.output.dense.weight"), vec![8, 16]);
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            shapes.insert(format!("{base}.{name}.weight"), vec![8]);
        }
    }
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    BTreeMap::from([
        ("model.safetensors".to_owned(), safetensors(shapes)),
        (
            "config.json".to_owned(),
            serde_json::to_vec(&serde_json::json!({
                "model_type":"bert", "architectures":["BertModel"], "vocab_size":32,
                "hidden_size":8, "num_hidden_layers":2, "num_attention_heads":2,
                "intermediate_size":16, "max_position_embeddings":24, "type_vocab_size":2,
                "hidden_act":"gelu"
            }))
            .unwrap(),
        ),
        (
            "1_Pooling/config.json".to_owned(),
            br#"{"pooling_mode":"mean"}"#.to_vec(),
        ),
        (
            "tokenizer.json".to_owned(),
            tokenizer
                .hf_tokenizer()
                .unwrap()
                .to_string(false)
                .unwrap()
                .into_bytes(),
        ),
    ])
}

#[test]
fn canonical_bert_source_load_reload_evaluates_after_source_drop() {
    let bytes = bert_bytes();
    let expected_weight_bytes = bytes["model.safetensors"].len() as u64;
    let (source, producer) = source(bytes);
    let mut engines = Vec::new();
    for _ in 0..2 {
        let request = request(source.clone(), "bert", NativeOperationDriver::Embedding);
        let (loaded, stats) = load_embedding(&request).unwrap();
        assert_eq!(stats.backing_bytes, expected_weight_bytes);
        assert!(stats.metadata_bytes > 0 && stats.largest_read <= 1024);
        assert_eq!(
            stats.source_bytes,
            stats.metadata_bytes + stats.backing_bytes
        );
        engines.push(crate::embeddings::EmbeddingEngine::new(loaded, 2));
    }
    assert_eq!(
        producer.reads.lock().unwrap()["model.safetensors"],
        2 * expected_weight_bytes
    );
    let weak = Arc::downgrade(&producer);
    drop(source);
    drop(producer);
    assert!(weak.upgrade().is_none());
    let mut replies = Vec::new();
    for engine in engines {
        let result = engine
            .embed_tokens(
                &[vec![1, 2], vec![3]],
                &crate::embeddings::EmbedOptions::default(),
            )
            .unwrap();
        assert_eq!(result.vectors.len(), 2);
        assert_eq!(result.prompt_tokens, 3);
        assert!(
            result
                .vectors
                .iter()
                .all(|v| v.shape == [8] && v.values.iter().all(|x| x.is_finite()))
        );
        replies.push(result);
    }
    assert_eq!(replies[0], replies[1]);
}

#[test]
fn canonical_bert_rejects_changed_input_ceiling_before_weights() {
    let mut bytes = bert_bytes();
    bytes.insert(
        "sentence_bert_config.json".to_owned(),
        br#"{"max_seq_length":12}"#.to_vec(),
    );
    let (source, producer) = source(bytes);
    let request = request(source, "bert", NativeOperationDriver::Embedding);
    assert!(load_embedding(&request).is_err());
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}

#[test]
fn canonical_loader_rejects_undeclared_shard_without_loading_weights() {
    let mut bytes = bert_bytes();
    bytes.insert(
        "model.safetensors.index.json".to_owned(),
        br#"{"weight_map":{"weight":"missing.safetensors"}}"#.to_vec(),
    );
    let (source, producer) = source(bytes);
    let request = request(source, "bert", NativeOperationDriver::Embedding);
    assert!(load_embedding(&request).is_err());
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}

#[test]
fn canonical_loader_rejects_malformed_shard_in_fallible_native_boundary() {
    let mut bytes = bert_bytes();
    bytes.insert("model.safetensors".to_owned(), vec![0; 16]);
    let (source, _) = source(bytes);
    let request = request(source, "bert", NativeOperationDriver::Embedding);
    assert!(load_embedding(&request).is_err());
}

#[test]
fn canonical_loader_rejects_cancellation_before_any_source_read() {
    let (source, producer) = source(gemma_bytes());
    let request = request(
        source,
        "gemma4_unified",
        super::super::NativeOperationDriver::AutoregressiveGenerate,
    );
    request.cancellation.store(true, Ordering::Release);
    assert_eq!(
        load_generation(&request).err().unwrap().kind(),
        NativeRuntimeErrorKind::Cancelled
    );
    assert!(producer.reads.lock().unwrap().is_empty());
}

#[test]
fn canonical_loader_rejects_config_family_substitution_before_weights() {
    let mut bytes = gemma_bytes();
    bytes.insert(
        "config.json".to_owned(),
        br#"{"model_type":"bert"}"#.to_vec(),
    );
    let (source, producer) = source(bytes);
    let request = request(
        source,
        "gemma4_unified",
        super::super::NativeOperationDriver::AutoregressiveGenerate,
    );
    assert!(load_generation(&request).is_err());
    assert!(
        !producer
            .reads
            .lock()
            .unwrap()
            .contains_key("model.safetensors")
    );
}
