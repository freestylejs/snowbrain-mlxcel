//! Verified-source contract: no path capability and no content proof on adoption.
use mlxcel::native_runtime::{
    NativeLoadLimits, NativeLoadStatistics, NativeRuntimeError, NativeVerifiedArtifactSource,
    NativeVerifiedComponent, NativeVerifiedSource,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Bytes {
    data: Vec<u8>,
    calls: AtomicUsize,
    short: bool,
}
impl NativeVerifiedArtifactSource for Bytes {
    fn read_at(
        &self,
        _: &str,
        offset: u64,
        output: &mut [u8],
    ) -> Result<usize, NativeRuntimeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let count = output.len().saturating_sub(usize::from(self.short));
        output[..count].copy_from_slice(&self.data[offset as usize..offset as usize + count]);
        Ok(count)
    }
}
fn component(label: &str, len: u64) -> NativeVerifiedComponent {
    NativeVerifiedComponent::new(label, len, [9; 32]).unwrap()
}
fn limits() -> NativeLoadLimits {
    NativeLoadLimits {
        max_components: 4,
        max_total_bytes: 1024,
        max_metadata_bytes: 128,
        max_read_bytes: 3,
    }
}
fn source(short: bool) -> (NativeVerifiedSource, Arc<Bytes>) {
    let data = Arc::new(Bytes {
        data: b"abcdefgh".to_vec(),
        calls: AtomicUsize::new(0),
        short,
    });
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("config.json", 8)],
        data.clone(),
        limits(),
    )
    .unwrap();
    (source, data)
}

#[test]
fn adopting_verified_source_reads_zero_bytes() {
    let (source, data) = source(false);
    assert_eq!(source.identity(), [7; 32]);
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn source_rejects_unsafe_empty_and_noncanonical_labels() {
    for label in [
        "",
        "/config.json",
        "../config.json",
        "a/../b",
        "a//b",
        "a/./b",
        "a\\b",
        "config.json/",
        "plugin.py",
        "plugin.so",
    ] {
        assert!(
            NativeVerifiedComponent::new(label, 8, [9; 32]).is_err(),
            "{label:?}"
        );
    }
}

#[test]
fn source_rejects_duplicate_overbound_and_overflowing_declarations_before_read() {
    let (_, data) = source(false);
    for declarations in [
        vec![],
        vec![component("a", 8), component("a", 8)],
        vec![component("a", 1025)],
        vec![component("a", u64::MAX), component("b", 1)],
    ] {
        assert!(NativeVerifiedSource::new([7; 32], declarations, data.clone(), limits()).is_err());
    }
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn source_rejects_component_count_above_limit_before_read() {
    let (_, data) = source(false);
    let maximum = limits().max_components;
    let declarations = (0..=maximum)
        .map(|index| component(&format!("part-{index}.json"), 1))
        .collect();
    assert!(NativeVerifiedSource::new([7; 32], declarations, data.clone(), limits()).is_err());
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
    let exact = (0..maximum)
        .map(|index| component(&format!("part-{index}.json"), 1))
        .collect();
    assert!(NativeVerifiedSource::new([7; 32], exact, data.clone(), limits()).is_ok());
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn oversized_producer_result_is_rejected_without_publishing_bytes_or_backing() {
    struct Oversized(AtomicUsize);
    impl NativeVerifiedArtifactSource for Oversized {
        fn read_at(
            &self,
            _: &str,
            _: u64,
            destination: &mut [u8],
        ) -> Result<usize, NativeRuntimeError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(destination.len() + 1)
        }
    }
    let producer = Arc::new(Oversized(AtomicUsize::new(0)));
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("config.json", 8)],
        producer.clone(),
        limits(),
    )
    .unwrap();
    for materialize in [false, true] {
        let mut statistics = NativeLoadStatistics::default();
        let error = if materialize {
            source
                .materialize("config.json", &AtomicBool::new(false), &mut statistics)
                .err()
                .unwrap()
        } else {
            source
                .read_metadata("config.json", &AtomicBool::new(false), &mut statistics)
                .unwrap_err()
        };
        assert_eq!(
            error.kind(),
            mlxcel::native_runtime::NativeRuntimeErrorKind::InvalidPackage
        );
        assert_eq!(statistics.source_read_calls, 1);
        assert_eq!(statistics.source_bytes, 0);
        assert_eq!(statistics.metadata_bytes, 0);
        assert_eq!(statistics.backing_bytes, 0);
    }
    assert_eq!(producer.0.load(Ordering::Relaxed), 2);
}

#[test]
fn metadata_reads_are_chunk_bounded_and_exact() {
    let (source, data) = source(false);
    let mut statistics = NativeLoadStatistics::default();
    let bytes = source
        .read_metadata("config.json", &AtomicBool::new(false), &mut statistics)
        .unwrap();
    assert_eq!(bytes, b"abcdefgh");
    assert_eq!(data.calls.load(Ordering::Relaxed), 3);
    assert_eq!(statistics.metadata_bytes, 8);
    assert_eq!(statistics.largest_read, 3);
    assert_eq!(statistics.source_read_calls, 3);
}

#[test]
fn metadata_rejects_short_read_unknown_label_and_cancellation() {
    let (short, _) = source(true);
    assert!(
        short
            .read_metadata(
                "config.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    let (source, data) = source(false);
    assert!(
        source
            .read_metadata(
                "missing.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    let cancelled = source
        .read_metadata(
            "config.json",
            &AtomicBool::new(true),
            &mut NativeLoadStatistics::default(),
        )
        .unwrap_err();
    assert_eq!(
        cancelled.kind(),
        mlxcel::native_runtime::NativeRuntimeErrorKind::Cancelled
    );
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn invalid_limits_and_overbound_metadata_fail_before_source_read() {
    let (_, data) = source(false);
    for invalid in [
        NativeLoadLimits {
            max_read_bytes: 0,
            ..limits()
        },
        NativeLoadLimits {
            max_components: 0,
            ..limits()
        },
        NativeLoadLimits {
            max_total_bytes: 0,
            ..limits()
        },
        NativeLoadLimits {
            max_read_bytes: usize::MAX,
            ..limits()
        },
    ] {
        assert!(
            NativeVerifiedSource::new(
                [7; 32],
                vec![component("config.json", 8)],
                data.clone(),
                invalid
            )
            .is_err()
        );
    }
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("config.json", 8)],
        data.clone(),
        NativeLoadLimits {
            max_metadata_bytes: 7,
            ..limits()
        },
    )
    .unwrap();
    assert!(
        source
            .read_metadata(
                "config.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn source_panic_is_contained_at_the_read_boundary() {
    struct Panics;
    impl NativeVerifiedArtifactSource for Panics {
        fn read_at(&self, _: &str, _: u64, _: &mut [u8]) -> Result<usize, NativeRuntimeError> {
            panic!("injected source fault");
        }
    }
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("config.json", 8)],
        Arc::new(Panics),
        limits(),
    )
    .unwrap();
    assert!(
        source
            .read_metadata(
                "config.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
}

#[test]
fn cancellation_between_reads_stops_before_next_source_call() {
    struct Cancels(Arc<AtomicBool>);
    impl NativeVerifiedArtifactSource for Cancels {
        fn read_at(&self, _: &str, _: u64, dst: &mut [u8]) -> Result<usize, NativeRuntimeError> {
            assert!(!self.0.swap(true, Ordering::AcqRel));
            dst.fill(1);
            Ok(dst.len())
        }
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("config.json", 8)],
        Arc::new(Cancels(cancelled.clone())),
        limits(),
    )
    .unwrap();
    let mut statistics = NativeLoadStatistics::default();
    assert!(
        source
            .read_metadata("config.json", &cancelled, &mut statistics)
            .is_err()
    );
    assert_eq!(statistics.source_read_calls, 1);
    assert_eq!(statistics.metadata_bytes, 3);
}

#[cfg(unix)]
#[test]
fn materialized_backing_is_unlinked_readonly_and_retained_independently() {
    use std::io::Write;
    use std::os::{fd::AsFd, unix::fs::MetadataExt};
    let (source, data) = source(false);
    let mut statistics = NativeLoadStatistics::default();
    let backing = source
        .materialize("config.json", &AtomicBool::new(false), &mut statistics)
        .unwrap();
    let mut file = std::fs::File::from(backing.as_fd().try_clone_to_owned().unwrap());
    assert_eq!(file.metadata().unwrap().nlink(), 0);
    assert!(file.write_all(b"X").is_err());
    assert_eq!(statistics.backing_bytes, 8);
    assert_eq!(statistics.source_read_calls, 3);
    let weak = Arc::downgrade(&data);
    drop(data);
    drop(source);
    assert!(weak.upgrade().is_none());
    assert_eq!(backing.read_range(0, 8).unwrap(), b"abcdefgh");
    assert!(backing.read_range(0, 9).is_err());
    assert!(backing.read_range(2, 1).is_err());
}

#[cfg(unix)]
#[test]
fn failed_or_cancelled_copy_never_returns_a_backing() {
    let (source, _) = source(true);
    assert!(
        source
            .materialize(
                "config.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    let (source, data) = self::source(false);
    assert!(
        source
            .materialize(
                "config.json",
                &AtomicBool::new(true),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    assert!(
        source
            .materialize(
                "missing",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .is_err()
    );
    assert_eq!(data.calls.load(Ordering::Relaxed), 0);
}

#[cfg(unix)]
#[test]
fn actual_lazy_native_tensor_outlives_source_and_backing_handles() {
    use std::os::fd::AsFd;
    let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.25_f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5_f32).to_le_bytes());
    let source = NativeVerifiedSource::new(
        [7; 32],
        vec![component("model.safetensors", bytes.len() as u64)],
        Arc::new(Bytes {
            data: bytes,
            calls: AtomicUsize::new(0),
            short: false,
        }),
        limits(),
    )
    .unwrap();
    let mut statistics = NativeLoadStatistics::default();
    let backing = source
        .materialize(
            "model.safetensors",
            &AtomicBool::new(false),
            &mut statistics,
        )
        .unwrap();
    let file = std::fs::File::from(backing.as_fd().try_clone_to_owned().unwrap());
    let mut weights = mlxcel_core::weights::load_safetensors_file(&file).unwrap();
    drop(file);
    drop(backing);
    drop(source);
    let array = weights.remove("weight").unwrap();
    drop(weights);
    let values = mlxcel_core::array_to_raw_bytes(&array);
    assert_eq!(
        values,
        [1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat()
    );
    assert!(statistics.backing_bytes > 8);
    assert_eq!(statistics.metadata_bytes, 0);
    assert!(statistics.largest_read <= 3);
}

#[test]
fn source_clone_retains_provider_until_last_owner_drops() {
    let (source, data) = source(false);
    let weak = Arc::downgrade(&data);
    drop(data);
    let clone = source.clone();
    drop(source);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        clone
            .read_metadata(
                "config.json",
                &AtomicBool::new(false),
                &mut NativeLoadStatistics::default()
            )
            .unwrap(),
        b"abcdefgh"
    );
    drop(clone);
    assert!(weak.upgrade().is_none());
}
