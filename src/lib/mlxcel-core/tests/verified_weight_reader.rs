//! Pathless safetensors loading must retain its read-only backing through lazy evaluation.
#![cfg(unix)]

use mlxcel_core::{array_to_raw_bytes, eval, weights::load_safetensors_file};
use std::fs::File;
use std::io::Write;

fn fixture() -> Vec<u8> {
    let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.25_f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5_f32).to_le_bytes());
    bytes
}

fn readonly_unlinked(bytes: &[u8]) -> File {
    let mut temporary = tempfile::NamedTempFile::new().unwrap();
    temporary.write_all(bytes).unwrap();
    temporary.flush().unwrap();
    let file = File::open(temporary.path()).unwrap();
    temporary.close().unwrap();
    file
}

#[test]
fn verified_reader_retains_backing_after_source_file_drop() {
    let file = readonly_unlinked(&fixture());
    let mut weights = load_safetensors_file(&file).unwrap();
    drop(file);
    let array = weights.remove("weight").unwrap();
    drop(weights);
    // Actual MLX lazy loading happens here, after both original holders are gone.
    eval(&array);
    let bytes = array_to_raw_bytes(&array);
    assert_eq!(
        bytes,
        [1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat()
    );
}

#[test]
fn verified_reader_rejects_writable_backing() {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(&fixture()).unwrap();
    assert!(load_safetensors_file(&file).is_err());
}

#[test]
fn verified_reader_reports_truncated_header_without_unwinding() {
    for bytes in [vec![], vec![1, 2, 3], 32_u64.to_le_bytes().to_vec()] {
        assert!(load_safetensors_file(&readonly_unlinked(&bytes)).is_err());
    }
}

#[test]
fn verified_reader_rejects_tensor_extent_outside_backing() {
    let mut bytes = fixture();
    bytes.truncate(bytes.len() - 1);
    assert!(load_safetensors_file(&readonly_unlinked(&bytes)).is_err());
}

#[test]
fn verified_reader_does_not_change_the_borrowed_file_cursor() {
    use std::io::{Seek, SeekFrom};
    let mut file = readonly_unlinked(&fixture());
    file.seek(SeekFrom::Start(3)).unwrap();
    let weights = load_safetensors_file(&file).unwrap();
    eval(&weights["weight"]);
    assert_eq!(file.stream_position().unwrap(), 3);
}

#[test]
fn verified_reader_rejects_invalid_descriptor_and_nonregular_file() {
    assert!(mlxcel_core::mlx_load_safetensors_fd(-1).is_err());
    let directory = tempfile::tempdir().unwrap();
    let file = File::open(directory.path()).unwrap();
    assert!(load_safetensors_file(&file).is_err());
}

#[test]
fn verified_reader_can_reload_one_immutable_backing_independently() {
    let file = readonly_unlinked(&fixture());
    let first = load_safetensors_file(&file).unwrap();
    let second = load_safetensors_file(&file).unwrap();
    drop(file);
    assert_eq!(
        array_to_raw_bytes(&first["weight"]),
        array_to_raw_bytes(&second["weight"])
    );
}
