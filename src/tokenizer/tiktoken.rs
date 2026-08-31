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

//! Tiktoken BPE tokenizer implementation.
//!
//! Supports loading `.tiktoken` vocabulary files (base64-encoded byte sequences
//! with integer ranks) and performing byte-level BPE encoding/decoding.
//! Used by: HunYuan models

use anyhow::Result;
use base64::Engine;
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::native_added_token_work_is_bounded;

/// Pre-tokenization regex pattern used by HunYuan's tiktoken tokenizer.
/// Matches contractions, letter sequences, digits, punctuation runs, and whitespace.
const HUNYUAN_PAT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct TiktokenTokenizer {
    /// Maps byte sequences to their token IDs (ranks)
    encoder: HashMap<Vec<u8>, u32>,
    /// Maps token IDs back to byte sequences
    decoder: HashMap<u32, Vec<u8>>,
    /// Maps special token strings to their IDs
    special_encoder: HashMap<String, u32>,
    /// Maps special token IDs back to strings
    special_decoder: HashMap<u32, String>,
    /// Special tokens sorted by length descending for greedy matching
    special_tokens_sorted: Vec<(String, u32)>,
    /// Pre-tokenization regex
    pat: Regex,
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` BPE file and tokenizer config.
    ///
    /// The `.tiktoken` file format: each line contains `<base64_token> <rank>`.
    /// Special tokens are derived from the vocabulary size and the standard
    /// HunYuan special token set.
    pub fn from_file(tiktoken_path: &Path, model_path: &Path) -> Result<Self> {
        Self::from_file_with_config(tiktoken_path, model_path, true, false)
    }

    pub(crate) fn from_file_with_declared_config(
        tiktoken_path: &Path,
        model_path: &Path,
        tokenizer_config_declared: bool,
    ) -> Result<Self> {
        Self::from_file_with_config(tiktoken_path, model_path, tokenizer_config_declared, true)
    }

    fn from_file_with_config(
        tiktoken_path: &Path,
        model_path: &Path,
        tokenizer_config_declared: bool,
        enforce_native_bounds: bool,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(tiktoken_path)?;

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token_b64 = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Invalid tiktoken line: missing token"))?;
            let rank_str = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Invalid tiktoken line: missing rank"))?;
            if parts.next().is_some() {
                return Err(anyhow::anyhow!(
                    "Invalid tiktoken line: unexpected trailing fields"
                ));
            }

            let token_bytes = base64::engine::general_purpose::STANDARD.decode(token_b64)?;
            let rank: u32 = rank_str.parse()?;
            if token_bytes.is_empty() {
                return Err(anyhow::anyhow!(
                    "Tiktoken vocabulary contains an empty token"
                ));
            }
            if encoder.contains_key(&token_bytes) {
                return Err(anyhow::anyhow!(
                    "Tiktoken vocabulary contains duplicate token bytes"
                ));
            }
            if decoder.contains_key(&rank) {
                return Err(anyhow::anyhow!(
                    "Tiktoken vocabulary contains duplicate token ranks"
                ));
            }
            decoder.insert(rank, token_bytes.clone());
            encoder.insert(token_bytes, rank);
        }
        if encoder.is_empty() {
            return Err(anyhow::anyhow!("Tiktoken vocabulary is empty"));
        }
        let vocabulary_len = u32::try_from(encoder.len())?;
        if (0..vocabulary_len).any(|rank| !decoder.contains_key(&rank)) {
            return Err(anyhow::anyhow!(
                "Tiktoken vocabulary ranks must be the dense range 0..N"
            ));
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            if !encoder.contains_key([byte].as_slice()) {
                return Err(anyhow::anyhow!(
                    "Tiktoken vocabulary is missing singleton byte 0x{byte:02X}"
                ));
            }
        }
        let special_start_id = vocabulary_len;

        // Build special tokens: same order as Python HYTokenizer
        let special_token_names = Self::build_special_token_list();
        let mut special_encoder = HashMap::new();
        let mut special_decoder = HashMap::new();

        for (i, name) in special_token_names.iter().enumerate() {
            let id = special_start_id
                .checked_add(u32::try_from(i)?)
                .ok_or_else(|| anyhow::anyhow!("Tiktoken special token ID overflowed"))?;
            special_encoder.insert(name.clone(), id);
            special_decoder.insert(id, name.clone());
        }

        // Override with tokenizer_config.json if available
        Self::load_special_tokens_from_config(
            model_path,
            &encoder,
            &decoder,
            &mut special_encoder,
            &mut special_decoder,
            tokenizer_config_declared,
            enforce_native_bounds,
        )?;

        let mut special_tokens_sorted: Vec<(String, u32)> = special_encoder
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        special_tokens_sorted.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let pat = Regex::new(HUNYUAN_PAT)?;

        Ok(Self {
            encoder,
            decoder,
            special_encoder,
            special_decoder,
            special_tokens_sorted,
            pat,
        })
    }

    /// Build the standard HunYuan special token list.
    fn build_special_token_list() -> Vec<String> {
        let mut tokens = vec![
            "<|endoftext|>".to_string(),
            "<|startoftext|>".to_string(),
            "<|bos|>".to_string(),
            "<|eos|>".to_string(),
            "<|pad|>".to_string(),
        ];
        for i in 0..205 {
            tokens.push(format!("<|extra_{i}|>"));
        }
        tokens
    }

    /// Load additional special token mappings from tokenizer_config.json.
    fn load_special_tokens_from_config(
        model_path: &Path,
        regular_encoder: &HashMap<Vec<u8>, u32>,
        regular_decoder: &HashMap<u32, Vec<u8>>,
        special_encoder: &mut HashMap<String, u32>,
        special_decoder: &mut HashMap<u32, String>,
        tokenizer_config_declared: bool,
        enforce_native_bounds: bool,
    ) -> Result<()> {
        if !tokenizer_config_declared {
            return Ok(());
        }
        let config_path = model_path.join("tokenizer_config.json");
        let content = match std::fs::read_to_string(&config_path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let config: serde_json::Value = serde_json::from_str(&content)?;
        if enforce_native_bounds && !native_added_token_work_is_bounded(&config) {
            return Err(anyhow::anyhow!(
                "tokenizer_config added-token work exceeds the native bound"
            ));
        }
        let config = config
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("tokenizer_config root is not an object"))?;
        if let Some(value) = config.get("added_tokens_decoder") {
            let decoder_map = value.as_object().ok_or_else(|| {
                anyhow::anyhow!("tokenizer_config added_tokens_decoder is not an object")
            })?;
            let mut source_ids = HashSet::new();
            let mut source_contents = HashSet::new();
            for (id_str, entry) in decoder_map {
                let id = id_str.parse::<u32>()?;
                if !source_ids.insert(id) {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config has duplicate numeric special token ID {id}"
                    ));
                }
                let token_content = entry
                    .get("content")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        anyhow::anyhow!("tokenizer_config special token {id} has no string content")
                    })?;
                if token_content.is_empty() {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config special token {id} has empty content"
                    ));
                }
                if !source_contents.insert(token_content) {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config has duplicate special token content {token_content:?}"
                    ));
                }
                if regular_decoder.contains_key(&id) {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config special token ID {id} collides with the regular vocabulary"
                    ));
                }
                if let Some(&regular_id) = regular_encoder.get(token_content.as_bytes()) {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config special token {token_content:?} at ID {id} shadows regular vocabulary ID {regular_id}"
                    ));
                }
                if let Some(existing) = special_decoder.get(&id)
                    && existing != token_content
                {
                    return Err(anyhow::anyhow!(
                        "tokenizer_config special token ID {id} has conflicting spellings"
                    ));
                }
                if let Some(previous_id) = special_encoder.get(token_content).copied()
                    && previous_id != id
                {
                    special_decoder.remove(&previous_id);
                }
                special_encoder.insert(token_content.to_string(), id);
                special_decoder.insert(id, token_content.to_string());
            }
        }
        if special_encoder.len() != special_decoder.len()
            || special_encoder
                .iter()
                .any(|(token, id)| special_decoder.get(id) != Some(token))
        {
            return Err(anyhow::anyhow!(
                "tokenizer_config special token mappings are not one-to-one"
            ));
        }
        Ok(())
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Split at special token boundaries first
        let segments = self.split_with_special_tokens(text);

        for segment in segments {
            if let Some(&id) = self.special_encoder.get(&segment) {
                result.push(id);
            } else {
                // Pre-tokenize with regex, then BPE encode each piece
                let matches = self.pat.find_iter(&segment);
                for m in matches {
                    let m = m.map_err(|e| anyhow::anyhow!("Regex error: {}", e))?;
                    let piece = m.as_str().as_bytes();

                    if let Some(&id) = self.encoder.get(piece) {
                        // Single-token fast path
                        result.push(id);
                    } else {
                        // Apply BPE merges
                        let ids = self.bpe_encode(piece)?;
                        result.extend(ids);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Encode without recognizing special-token spellings written into the
    /// input text (`parse_special: false`, #1442).
    pub fn encode_without_special_parsing(&self, text: &str) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        let matches = self.pat.find_iter(text);
        for m in matches {
            let m = m.map_err(|e| anyhow::anyhow!("Regex error: {}", e))?;
            let piece = m.as_str().as_bytes();
            if let Some(&id) = self.encoder.get(piece) {
                result.push(id);
            } else {
                result.extend(self.bpe_encode(piece)?);
            }
        }
        Ok(result)
    }

    /// The id a vocabulary entry holds, by its exact spelling.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.special_encoder
            .get(token)
            .copied()
            .or_else(|| self.encoder.get(token.as_bytes()).copied())
    }

    /// Raw bytes for one token; see `MlxcelTokenizer::token_piece_bytes`.
    ///
    /// The tiktoken vocabulary is byte sequences by construction, so this is
    /// the one backend where no reconstruction is needed.
    pub fn piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if let Some(special) = self.special_decoder.get(&id) {
            return Some(special.clone().into_bytes());
        }
        self.decoder.get(&id).cloned()
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();

        for &id in ids {
            if let Some(special) = self.special_decoder.get(&id) {
                if !skip_special_tokens {
                    bytes.extend_from_slice(special.as_bytes());
                }
            } else if let Some(token_bytes) = self.decoder.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
            // Preserve the root/server decoder's legacy behavior: unknown IDs
            // are skipped. The native exact seam validates token_piece_bytes
            // before decoding and reports a typed InvalidTokenId instead.
        }

        // Preserve existing server/API detokenization semantics. The native
        // incremental decoder has its own strict no-replacement UTF-8 state.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Apply byte-pair encoding to a byte sequence using the rank-based algorithm.
    ///
    /// The algorithm repeatedly finds the pair of adjacent tokens with the lowest
    /// rank in the vocabulary and merges them, until no more merges are possible.
    fn bpe_encode(&self, piece: &[u8]) -> Result<Vec<u32>> {
        if piece.is_empty() {
            return Ok(vec![]);
        }
        if piece.len() == 1 {
            // Single byte — must be in the vocabulary (bytes 0-255)
            return self
                .encoder
                .get(piece)
                .map(|&id| vec![id])
                .ok_or_else(|| anyhow::anyhow!("Tiktoken vocabulary cannot encode input byte"));
        }

        // Start with each byte as its own part
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();

        loop {
            if parts.len() < 2 {
                break;
            }

            // Find the pair with the minimum rank
            let mut min_rank = u32::MAX;
            let mut min_idx = usize::MAX;

            for i in 0..parts.len() - 1 {
                let mut merged = parts[i].clone();
                merged.extend_from_slice(&parts[i + 1]);
                if let Some(&rank) = self.encoder.get(&merged)
                    && rank < min_rank
                {
                    min_rank = rank;
                    min_idx = i;
                }
            }

            if min_idx == usize::MAX {
                break; // No more merges possible
            }

            // Merge the pair at min_idx
            let merged = {
                let mut m = parts[min_idx].clone();
                m.extend_from_slice(&parts[min_idx + 1]);
                m
            };
            parts[min_idx] = merged;
            parts.remove(min_idx + 1);
        }

        // Convert parts to token IDs
        parts
            .iter()
            .map(|part| {
                self.encoder
                    .get(part.as_slice())
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Tiktoken BPE produced an unranked byte span"))
            })
            .collect()
    }

    /// Split text into segments at special token boundaries (greedy longest-match-first).
    fn split_with_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_tokens_sorted.is_empty() {
            return vec![text.to_string()];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;
            for (token, _) in &self.special_tokens_sorted {
                if token.is_empty() {
                    continue;
                }
                if remaining.starts_with(token.as_str()) {
                    segments.push(token.clone());
                    remaining = &remaining[token.len()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Find the next special token occurrence
                let mut next_pos = remaining.len();
                for (token, _) in &self.special_tokens_sorted {
                    if token.is_empty() {
                        continue;
                    }
                    if let Some(pos) = remaining.find(token.as_str())
                        && pos < next_pos
                    {
                        next_pos = pos;
                    }
                }
                segments.push(remaining[..next_pos].to_string());
                remaining = &remaining[next_pos..];
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn test_model_path() -> PathBuf {
        PathBuf::from("models/hunyuan-13b")
    }

    fn tiktoken_file() -> PathBuf {
        test_model_path().join("hy.tiktoken")
    }

    fn load_inline(vocabulary: &str, config: Option<&str>) -> Result<TiktokenTokenizer> {
        let root = TempDir::new()?;
        let vocabulary_path = root.path().join("tiny.tiktoken");
        std::fs::write(&vocabulary_path, vocabulary)?;
        if let Some(config) = config {
            std::fs::write(root.path().join("tokenizer_config.json"), config)?;
        }
        TiktokenTokenizer::from_file(&vocabulary_path, root.path())
    }

    fn all_singleton_bytes() -> String {
        (0u16..=255)
            .map(|rank| {
                let encoded = base64::engine::general_purpose::STANDARD.encode([rank as u8]);
                format!("{encoded} {rank}\n")
            })
            .collect()
    }

    #[test]
    fn exact_tiktoken_rank_and_special_id_invariants_are_enforced() {
        let vocabulary = all_singleton_bytes();
        let valid = load_inline(&vocabulary, None).unwrap();
        assert_eq!(valid.piece_bytes(b'a' as u32).unwrap(), b"a");
        assert_eq!(valid.piece_bytes(b'b' as u32).unwrap(), b"b");
        assert_eq!(valid.encode("all bytes covered", false).unwrap().len(), 17);

        let missing_byte = vocabulary
            .lines()
            .filter(|line| !line.ends_with(" 255"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(load_inline(&missing_byte, None).is_err());
        let mut duplicate_rank = vocabulary.clone();
        duplicate_rank.push_str("YQ== 0\n");
        assert!(load_inline(&duplicate_rank, None).is_err());
        let mut duplicate_token = vocabulary.clone();
        duplicate_token.push_str("YQ== 256\n");
        assert!(load_inline(&duplicate_token, None).is_err());
        assert!(
            load_inline(
                &vocabulary,
                Some(r#"{"added_tokens_decoder":{"1":{"content":"<|hostile|>"}}}"#),
            )
            .is_err()
        );
        assert!(
            load_inline(
                &vocabulary,
                Some(r#"{"added_tokens_decoder":{"300":{"content":""}}}"#),
            )
            .is_err()
        );
        assert!(
            load_inline(
                &vocabulary,
                Some(r#"{"added_tokens_decoder":{"500":{"content":"a"}}}"#),
            )
            .is_err(),
            "a special token cannot shadow regular byte content"
        );
        assert!(
            load_inline(
                &vocabulary,
                Some(r#"{"added_tokens_decoder":{"500":{"content":"<dup>"},"501":{"content":"<dup>"}}}"#),
            )
            .is_err(),
            "duplicate source content is ambiguous even before the maps are built"
        );
        assert!(
            load_inline(
                &vocabulary,
                Some(
                    r#"{"added_tokens_decoder":{"500":{"content":"<a>"},"0500":{"content":"<b>"}}}"#
                ),
            )
            .is_err(),
            "numeric key aliases must not silently overwrite each other"
        );
        assert!(load_inline(&vocabulary, Some("[]")).is_err());
        assert!(load_inline(&vocabulary, Some("{}")).is_ok());
    }

    #[test]
    #[ignore] // Requires model files
    fn test_load_tiktoken() {
        let tokenizer = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path());
        assert!(
            tokenizer.is_ok(),
            "Failed to load tiktoken: {:?}",
            tokenizer.err()
        );
        let t = tokenizer.unwrap();
        assert!(!t.encoder.is_empty());
        assert!(!t.special_encoder.is_empty());
        assert!(t.special_encoder.contains_key("<|eos|>"));
    }

    #[test]
    #[ignore] // Requires model files
    fn test_encode_decode_roundtrip() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "Hello, world!";
        let ids = t.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_encode_chinese() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "你好，世界";
        let ids = t.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_special_tokens() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "<|eos|>";
        let ids = t.encode(text, false).unwrap();
        assert_eq!(ids.len(), 1);
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }
}
