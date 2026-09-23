//! Exact static/source identities are checked before native model loading.
use mlxcel::native_runtime::{
    NativeModalitySet, NativeTokenizerLoadBinding, NativeWorkerLoadBinding,
};

fn binding(
    source: [u8; 32],
    report: [u8; 32],
    build: [u8; 32],
    architecture: &str,
) -> Result<NativeWorkerLoadBinding, mlxcel::native_runtime::NativeRuntimeError> {
    NativeWorkerLoadBinding::new(
        source,
        report,
        build,
        architecture,
        true,
        Some(24),
        NativeModalitySet {
            text: true,
            image: false,
            audio: false,
            video: false,
        },
        NativeTokenizerLoadBinding::new([4; 32], "opaque-producer-tokenizer").unwrap(),
    )
}

#[test]
fn binding_preserves_opaque_domains_without_loading_or_hashing() {
    let value = binding([1; 32], [2; 32], [3; 32], "bert").unwrap();
    assert_eq!(value.source_identity(), [1; 32]);
    assert_eq!(value.report_identity(), [2; 32]);
    assert_eq!(value.native_build_identity(), [3; 32]);
    assert_eq!(value.architecture(), "bert");
    assert_eq!(value.tokenizer().digest(), [4; 32]);
    assert_eq!(
        value.tokenizer().implementation(),
        "opaque-producer-tokenizer"
    );
    assert_eq!(value.max_context_positions(), Some(24));
}

#[test]
fn binding_rejects_missing_or_unbounded_identity_metadata() {
    for (source, report, build) in [
        ([0; 32], [2; 32], [3; 32]),
        ([1; 32], [0; 32], [3; 32]),
        ([1; 32], [2; 32], [0; 32]),
    ] {
        assert!(binding(source, report, build, "bert").is_err());
    }
    for architecture in ["", "bad\narchitecture"] {
        assert!(binding([1; 32], [2; 32], [3; 32], architecture).is_err());
    }
    assert!(binding([1; 32], [2; 32], [3; 32], &"x".repeat(4097)).is_err());
    assert!(NativeTokenizerLoadBinding::new([0; 32], "implementation").is_err());
    assert!(NativeTokenizerLoadBinding::new([4; 32], "").is_err());
    assert!(NativeTokenizerLoadBinding::new([4; 32], &"x".repeat(4097)).is_err());
}
