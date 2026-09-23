use super::*;

fn embedding_policy() -> NativeWorkerPolicy {
    NativeWorkerPolicy {
        operation_driver: NativeOperationDriver::Embedding,
        scheduler_topology: NativeSchedulerTopology::SingleSequence,
        sequence_state_topology: NativeSequenceStateTopology::Independent,
        context: NativeContextPolicy {
            max_context_tokens: 24,
            max_position: 24,
        },
        batch: NativeBatchPolicy {
            max_active_sequences: 1,
            max_queued_operations: 1,
            max_batch_tokens: 24,
            prefill_chunk_tokens: 1,
        },
        cache: NativeCachePolicy {
            topology: NativeCacheTopology::Dense,
            max_resident_bytes: 1,
            prefix_reuse: false,
        },
        device: NativeDevicePolicy {
            class: NativeDeviceClass::Cpu,
            offload: NativeOffloadPolicy::HostVisible,
            max_device_bytes: 1,
        },
        model_owned_state: NativeModelOwnedStateMode::Disabled,
        adapter: None,
        drafter: NativeDrafterTopology::Disabled,
        prepared_multimodal: NativePreparedMultimodalClass::Disabled,
        max_prompt_tokens: 24,
        max_generation_tokens: 1,
        max_embedding_inputs: 1,
        max_embedding_dimensions: 8,
        max_accepted_chunk_units: 1,
        max_constraint_bytes: 1,
    }
}

fn bert_binding() -> NativeWorkerLoadBinding {
    NativeWorkerLoadBinding::new(
        [1; 32],
        [2; 32],
        [3; 32],
        "bert",
        true,
        Some(24),
        NativeModalitySet {
            text: true,
            ..NativeModalitySet::default()
        },
        NativeTokenizerLoadBinding::new([4; 32], "tokenizers").unwrap(),
    )
    .unwrap()
}

#[test]
fn verified_binding_builds_embedding_profile_without_native_report_or_decoder() {
    let binding = bert_binding();
    let profile = NativeWorkerConfig::try_from_binding(&binding, embedding_policy()).unwrap();

    assert_eq!(profile.package_identity(), binding.source_identity());
    assert_eq!(profile.report_identity(), binding.report_identity());
    assert_eq!(profile.model_context_window(), Some(24));
    assert_eq!(profile.embedding_input_ceiling(), Some(24));
    assert_eq!(profile.load_binding(), Some(&binding));
}
