use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::canonical;

#[derive(Debug, Clone)]
pub struct PostLinkConfig {
    pub artifact: PathBuf,
    pub link_trace: PathBuf,
    pub link_map: Option<PathBuf>,
    pub pre_link_identity: [u8; 32],
    pub artifact_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkTrace {
    pub schema: String,
    pub linker_identity: String,
    pub archiver_identity: Option<String>,
    pub output_name: String,
    pub arguments: Vec<String>,
    pub archives: Vec<ArchiveTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveTrace {
    pub argument_index: usize,
    pub label: String,
    pub sha256: String,
    pub members_sorted: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostLinkSidecar {
    pub schema: String,
    pub pre_link_identity: String,
    pub artifact_kind: String,
    pub artifact_sha256: String,
    pub final_link_trace_sha256: String,
    pub link_map_sha256: Option<String>,
    pub observed_link_arguments: Vec<String>,
}

pub fn produce_post_link_sidecar(config: &PostLinkConfig) -> Result<PostLinkSidecar, String> {
    if config.artifact_kind.is_empty() || config.artifact_kind.len() > 1024 {
        return Err("artifact kind is empty or oversized".to_owned());
    }
    let trace_metadata = fs::metadata(&config.link_trace)
        .map_err(|err| format!("cannot inspect {}: {err}", config.link_trace.display()))?;
    if !trace_metadata.is_file() || trace_metadata.len() > 32 * 1024 * 1024 {
        return Err("link trace is not a regular file or exceeds 32 MiB".to_owned());
    }
    let trace_bytes = fs::read(&config.link_trace)
        .map_err(|err| format!("cannot read {}: {err}", config.link_trace.display()))?;
    let trace: LinkTrace = serde_json::from_slice(&trace_bytes)
        .map_err(|err| format!("link trace is invalid: {err}"))?;
    validate_trace(&trace)?;
    let artifact_name = config
        .artifact
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "artifact name is not strict UTF-8".to_owned())?;
    if trace.output_name != artifact_name {
        return Err("link trace belongs to a different output artifact".to_owned());
    }
    let map_digest = config
        .link_map
        .as_ref()
        .map(|path| canonical::file_sha256(path).map(canonical::hex_digest))
        .transpose()?;
    if let Some(map) = &config.link_map {
        let map_name = map
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "link map name is not strict UTF-8".to_owned())?;
        if !trace.arguments.iter().any(|argument| {
            argument
                .trim_matches('"')
                .replace('\\', "/")
                .ends_with(map_name)
        }) {
            return Err("link trace does not reference the supplied link map".to_owned());
        }
    }
    Ok(PostLinkSidecar {
        schema: "mlxcel.post-link.v1".to_owned(),
        pre_link_identity: canonical::hex_digest(config.pre_link_identity),
        artifact_kind: config.artifact_kind.clone(),
        artifact_sha256: canonical::hex_digest(canonical::file_sha256(&config.artifact)?),
        final_link_trace_sha256: canonical::hex_digest(sha2::Sha256::digest(&trace_bytes).into()),
        link_map_sha256: map_digest,
        observed_link_arguments: trace.arguments,
    })
}

fn validate_trace(trace: &LinkTrace) -> Result<(), String> {
    if trace.schema != "mlxcel.final-link-trace.v1" {
        return Err("link trace has an unknown schema".to_owned());
    }
    if trace.linker_identity.is_empty()
        || trace.output_name.is_empty()
        || trace.arguments.is_empty()
        || trace.arguments.len() > canonical::MAX_CANONICAL_ITEMS
    {
        return Err("link trace is incomplete or oversized".to_owned());
    }
    for value in trace
        .arguments
        .iter()
        .chain([&trace.linker_identity, &trace.output_name])
    {
        validate_trace_text(value)?;
    }
    validate_tool_identity(&trace.linker_identity, "linker")?;
    validate_trace_output(trace)?;
    if let Some(archiver) = &trace.archiver_identity {
        validate_trace_text(archiver)?;
        validate_tool_identity(archiver, "archiver")?;
    }
    if trace.archives.is_empty() != trace.archiver_identity.is_none() {
        return Err("link trace archive evidence and archiver identity disagree".to_owned());
    }
    if trace.archives.len() > canonical::MAX_CANONICAL_ITEMS {
        return Err("link trace archive list exceeds the item bound".to_owned());
    }
    let mut indices = BTreeSet::new();
    let mut previous_index = None;
    let mut member_count = 0_usize;
    for archive in &trace.archives {
        let digest = canonical::parse_sha256(&archive.sha256, "archive digest")?;
        member_count = member_count
            .checked_add(archive.members_sorted.len())
            .ok_or_else(|| "archive member count overflowed".to_owned())?;
        if archive.argument_index >= trace.arguments.len()
            || !indices.insert(archive.argument_index)
            || previous_index.is_some_and(|previous| previous >= archive.argument_index)
            || archive.label != trace.arguments[archive.argument_index]
            || archive
                .members_sorted
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || digest == [0; 32]
            || member_count > canonical::MAX_CANONICAL_ITEMS
        {
            return Err("link trace contains invalid archive evidence".to_owned());
        }
        validate_trace_text(&archive.label)?;
        for member in &archive.members_sorted {
            validate_trace_text(member)?;
        }
        previous_index = Some(archive.argument_index);
    }
    Ok(())
}

fn validate_trace_text(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > canonical::MAX_CANONICAL_TEXT_BYTES
        || value
            .bytes()
            .any(|byte| byte == 0 || byte == b'\r' || byte == b'\n')
    {
        return Err("link trace contains invalid text".to_owned());
    }
    Ok(())
}

fn validate_tool_identity(value: &str, label: &str) -> Result<(), String> {
    let (_, digest_and_version) = value
        .split_once("|sha256=")
        .ok_or_else(|| format!("{label} identity has no executable digest"))?;
    let (digest, version) = digest_and_version
        .split_once('|')
        .ok_or_else(|| format!("{label} identity has no version"))?;
    let digest = canonical::parse_sha256(digest, &format!("{label} executable"))?;
    if digest == [0; 32] || version.is_empty() {
        return Err(format!("{label} identity is incomplete"));
    }
    Ok(())
}

fn validate_trace_output(trace: &LinkTrace) -> Result<(), String> {
    let mut outputs = Vec::new();
    let mut index = 0;
    while index < trace.arguments.len() {
        let argument = &trace.arguments[index];
        if argument == "-o" {
            outputs.push(
                trace
                    .arguments
                    .get(index + 1)
                    .ok_or_else(|| "link trace -o has no output path".to_owned())?
                    .as_str(),
            );
            index += 2;
            continue;
        }
        if argument
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/out:"))
        {
            outputs.push(argument.get(5..).unwrap_or_default());
        } else if let Some(output) = argument.strip_prefix("-Wl,-o,") {
            outputs.push(output);
        } else if let Some(output) = argument.strip_prefix("-Wl,-o=") {
            outputs.push(output);
        }
        index += 1;
    }
    if outputs.len() != 1 {
        return Err("link trace must contain exactly one output selection".to_owned());
    }
    let normalized_output = outputs[0].trim_matches('"').replace('\\', "/");
    let output_name = normalized_output.rsplit('/').next().unwrap_or(outputs[0]);
    if output_name != trace.output_name {
        return Err("link trace output selection disagrees with its artifact name".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn post_link_recomputes_artifact_trace_and_map() {
        let temp = tempfile::tempdir().unwrap();
        let artifact = temp.path().join("runner");
        let map = temp.path().join("runner.map");
        let trace_path = temp.path().join("runner.trace.json");
        fs::write(&artifact, b"artifact").unwrap();
        fs::write(&map, b"map").unwrap();
        let trace = LinkTrace {
            schema: "mlxcel.final-link-trace.v1".to_owned(),
            linker_identity: format!("ld|sha256={}|version", "11".repeat(32)),
            archiver_identity: None,
            output_name: "runner".to_owned(),
            arguments: vec![
                "-o".to_owned(),
                "<artifact>/runner".to_owned(),
                "-Wl,-Map,<artifact>/runner.map".to_owned(),
            ],
            archives: Vec::new(),
        };
        fs::write(&trace_path, serde_json::to_vec(&trace).unwrap()).unwrap();
        let sidecar = produce_post_link_sidecar(&PostLinkConfig {
            artifact: artifact.clone(),
            link_trace: trace_path.clone(),
            link_map: Some(map.clone()),
            pre_link_identity: [7; 32],
            artifact_kind: "runner".to_owned(),
        })
        .unwrap();
        assert_eq!(
            sidecar.artifact_sha256,
            canonical::hex_digest(sha2::Sha256::digest(b"artifact").into())
        );
        assert_eq!(
            sidecar.final_link_trace_sha256,
            canonical::hex_digest(sha2::Sha256::digest(fs::read(trace_path).unwrap()).into())
        );
        assert_eq!(
            sidecar.link_map_sha256.unwrap(),
            canonical::hex_digest(sha2::Sha256::digest(b"map").into())
        );
    }

    #[test]
    fn invalid_archive_mutations_never_skip_success() {
        let mut trace = LinkTrace {
            schema: "mlxcel.final-link-trace.v1".to_owned(),
            linker_identity: format!("ld|sha256={}|version", "11".repeat(32)),
            archiver_identity: Some(format!("ar|sha256={}|version", "22".repeat(32))),
            output_name: "runner".to_owned(),
            arguments: vec!["-o".to_owned(), "runner".to_owned(), "archive.a".to_owned()],
            archives: vec![ArchiveTrace {
                argument_index: 2,
                label: "archive.a".to_owned(),
                sha256: "11".repeat(32),
                members_sorted: vec!["a.o".to_owned(), "b.o".to_owned()],
            }],
        };
        assert!(validate_trace(&trace).is_ok());
        trace.archives[0].members_sorted.swap(0, 1);
        assert!(validate_trace(&trace).is_err());
        trace.archives[0].members_sorted.sort();
        trace.archives[0].sha256 = "00".to_owned();
        assert!(validate_trace(&trace).is_err());
        trace.archives[0].sha256 = "11".repeat(32);
        trace.archives[0].label = "different.a".to_owned();
        assert!(validate_trace(&trace).is_err());
        trace.archives[0].label = trace.arguments[2].clone();
        trace.archiver_identity = None;
        assert!(validate_trace(&trace).is_err());
    }

    #[test]
    fn malformed_trace_text_tool_output_and_output_selection_never_skip_success() {
        let mut trace = LinkTrace {
            schema: "mlxcel.final-link-trace.v1".to_owned(),
            linker_identity: format!("ld|sha256={}|version", "11".repeat(32)),
            archiver_identity: Some(format!("ar|sha256={}|version", "22".repeat(32))),
            output_name: "runner".to_owned(),
            arguments: vec!["-o".to_owned(), "runner".to_owned(), "archive.a".to_owned()],
            archives: vec![ArchiveTrace {
                argument_index: 2,
                label: "archive.a".to_owned(),
                sha256: "33".repeat(32),
                members_sorted: vec!["member.o".to_owned()],
            }],
        };
        assert!(validate_trace(&trace).is_ok());

        trace.linker_identity = "ld|version".to_owned();
        assert!(validate_trace(&trace).is_err());
        trace.linker_identity = format!("ld|sha256={}|version", "11".repeat(32));
        trace.arguments[1] = "different".to_owned();
        assert!(validate_trace(&trace).is_err());
        trace.arguments[1] = "runner".to_owned();
        trace.archives[0].sha256 = "00".repeat(32);
        assert!(validate_trace(&trace).is_err());
        trace.archives[0].sha256 = "33".repeat(32);
        trace.archives[0].members_sorted = vec!["member.o".to_owned(), "member.o".to_owned()];
        assert!(validate_trace(&trace).is_ok());
        trace.archives[0].members_sorted = vec!["member.o".to_owned()];
        trace.archives[0].members_sorted[0] = "bad\nmember".to_owned();
        assert!(validate_trace(&trace).is_err());
    }
}
