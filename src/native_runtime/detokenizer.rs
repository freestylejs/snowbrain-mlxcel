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

//! Fallible exact incremental decoding for the neutral native runtime seam.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::tokenizer::MlxcelTokenizer;

use super::error::{TokenizerError, TokenizerErrorKind};
use super::sink::{
    MAX_NATIVE_ACCEPTED_UNITS, MAX_NATIVE_STABLE_BYTES_PER_CHUNK, MAX_NATIVE_STABLE_BYTES_PER_UNIT,
};

pub const MAX_PROVEN_LOOKBEHIND_TOKENS: usize = MAX_NATIVE_ACCEPTED_UNITS;
pub const MAX_TERMINAL_ONLY_TOKENS: usize = MAX_NATIVE_ACCEPTED_UNITS;
pub const MAX_DECODER_PROMPT_TOKENS: usize = 1_048_576;
pub const MAX_TOKEN_PIECE_BYTES: usize = MAX_NATIVE_STABLE_BYTES_PER_UNIT;
pub const MAX_PENDING_DECODE_BYTES: usize = MAX_NATIVE_STABLE_BYTES_PER_UNIT;
pub const MAX_STABLE_DELTA_BYTES: usize = MAX_NATIVE_STABLE_BYTES_PER_UNIT;
pub const MAX_DECODED_WINDOW_BYTES: usize = MAX_NATIVE_STABLE_BYTES_PER_CHUNK;

/// The exactness tier proved for one tokenizer/profile pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IncrementalDecodeGuarantee {
    RawTokenBytes,
    ProvenBoundedLookbehind { tokens: usize },
    TerminalOnly,
}

/// Newly stable bytes and the number of raw bytes still pending.
///
/// Fields are private so an external decoder implementation cannot fabricate
/// an output beyond the native accepted-unit ceiling.
///
/// ```compile_fail
/// # use mlxcel::native_runtime::StableDecodeDelta;
/// let _ = StableDecodeDelta { stable_bytes: vec![], pending_bytes: 0 };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StableDecodeDelta {
    stable_bytes: Vec<u8>,
    pending_bytes: usize,
}

impl StableDecodeDelta {
    pub fn try_new(stable_bytes: Vec<u8>, pending_bytes: usize) -> Result<Self, TokenizerError> {
        let value = Self {
            stable_bytes,
            pending_bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), TokenizerError> {
        if self.stable_bytes.len() > MAX_STABLE_DELTA_BYTES
            || self.pending_bytes > MAX_PENDING_DECODE_BYTES
        {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "decoder delta exceeds the native accepted-unit byte ceiling",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn stable_bytes(&self) -> &[u8] {
        &self.stable_bytes
    }

    #[must_use]
    pub const fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    #[must_use]
    pub fn into_stable_bytes(self) -> Vec<u8> {
        self.stable_bytes
    }
}

/// Fallible decoder contract. Implementations never silently replace errors.
///
/// This trait deliberately has no `Send`/`Sync` bound. A native worker creates
/// and owns its concrete tokenizer/decoder on its dedicated physical thread;
/// only checked [`StableDecodeDelta`] bytes may cross into a `Send` sink.
pub trait NativeIncrementalDecoder {
    fn push(&mut self, token_id: i32) -> Result<StableDecodeDelta, TokenizerError>;
    fn flush(&mut self) -> Result<StableDecodeDelta, TokenizerError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeTokenizerKind {
    HuggingFace,
    SentencePiece,
    Tiktoken,
}

trait DecoderBackend {
    fn kind(&self) -> NativeTokenizerKind;
    fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError>;
    fn piece_bytes(&self, id: u32) -> Option<Vec<u8>>;
    fn decoded_len_upper_bound(&self, ids: &[u32]) -> Option<usize>;
}

impl DecoderBackend for MlxcelTokenizer {
    fn kind(&self) -> NativeTokenizerKind {
        match self {
            Self::HuggingFace(_) => NativeTokenizerKind::HuggingFace,
            Self::SentencePiece(_) => NativeTokenizerKind::SentencePiece,
            Self::Tiktoken(_) => NativeTokenizerKind::Tiktoken,
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let result = catch_unwind(AssertUnwindSafe(|| self.decode(ids, false))).map_err(|_| {
            TokenizerError::new(
                TokenizerErrorKind::DecodeFailed,
                "tokenizer panicked while decoding a bounded native window",
            )
        })?;
        result.map(String::into_bytes).map_err(|error| {
            TokenizerError::new(
                TokenizerErrorKind::DecodeFailed,
                format!("tokenizer decode failed: {error}"),
            )
        })
    }

    fn piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.token_piece_bytes(id)
    }

    fn decoded_len_upper_bound(&self, ids: &[u32]) -> Option<usize> {
        ids.iter().try_fold(0usize, |total, &id| {
            let piece = self.token_piece_bytes(id)?;
            total.checked_add(piece.len().checked_add(4)?)
        })
    }
}

/// Decoder over one frozen tokenizer. Only terminal-only construction is
/// public; stronger tiers are selected internally from a package-bound report.
pub struct NativeTokenizerDecoder {
    core: DecoderCore,
}

impl NativeTokenizerDecoder {
    /// Construct the conservative terminal-only tier without making a
    /// tokenizer-specific streaming claim.
    pub fn terminal_only(
        tokenizer: Arc<MlxcelTokenizer>,
        prompt_tokens: &[i32],
    ) -> Result<Self, TokenizerError> {
        Ok(Self {
            core: DecoderCore::new(
                tokenizer,
                prompt_tokens,
                IncrementalDecodeGuarantee::TerminalOnly,
            )?,
        })
    }

    /// Test-only constructor for exercising guarantee algorithms in isolation.
    /// Production workers must use [`Self::from_worker_authority`], whose
    /// opaque argument is created only by immutable package/report binding.
    #[cfg(test)]
    pub(crate) fn from_proven_profile(
        tokenizer: Arc<MlxcelTokenizer>,
        prompt_tokens: &[i32],
        guarantee: IncrementalDecodeGuarantee,
    ) -> Result<Self, TokenizerError> {
        Ok(Self {
            core: DecoderCore::new(tokenizer, prompt_tokens, guarantee)?,
        })
    }

    pub(crate) fn from_worker_authority(
        authority: super::model_report::NativeBoundTokenizer,
        prompt_tokens: &[i32],
    ) -> Result<Self, TokenizerError> {
        let (tokenizer, guarantee) = authority.into_parts();
        Ok(Self {
            core: DecoderCore::new(tokenizer, prompt_tokens, guarantee)?,
        })
    }

    #[must_use]
    pub const fn guarantee(&self) -> IncrementalDecodeGuarantee {
        self.core.guarantee
    }
}

impl NativeIncrementalDecoder for NativeTokenizerDecoder {
    fn push(&mut self, token_id: i32) -> Result<StableDecodeDelta, TokenizerError> {
        self.core.push(token_id)
    }

    fn flush(&mut self) -> Result<StableDecodeDelta, TokenizerError> {
        self.core.flush()
    }
}

struct DecoderCore {
    backend: Arc<dyn DecoderBackend>,
    guarantee: IncrementalDecodeGuarantee,
    all_ids: Vec<u32>,
    read_offset: usize,
    committed: Vec<u8>,
    pending_raw_bytes: usize,
    raw_utf8_pending: Vec<u8>,
    terminal_pending_tokens: usize,
    flushed: bool,
}

impl DecoderCore {
    fn new(
        backend: Arc<dyn DecoderBackend>,
        prompt_tokens: &[i32],
        guarantee: IncrementalDecodeGuarantee,
    ) -> Result<Self, TokenizerError> {
        if let IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } = guarantee
            && (tokens == 0 || tokens > MAX_PROVEN_LOOKBEHIND_TOKENS)
        {
            return Err(TokenizerError::new(
                TokenizerErrorKind::UnsupportedGuarantee,
                "bounded-lookbehind guarantee is zero or exceeds the native decoder bound",
            ));
        }
        if guarantee == IncrementalDecodeGuarantee::RawTokenBytes
            && backend.kind() != NativeTokenizerKind::Tiktoken
        {
            return Err(TokenizerError::new(
                TokenizerErrorKind::UnsupportedGuarantee,
                "raw-token-byte decoding is enabled only for the compositional tiktoken backend",
            ));
        }

        if prompt_tokens.len() > MAX_DECODER_PROMPT_TOKENS {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "decoder prompt token count exceeds the native bound",
            ));
        }
        let prompt_ids = prompt_tokens
            .iter()
            .map(|&id| checked_id(id))
            .collect::<Result<Vec<_>, _>>()?;
        validate_prompt_utf8(backend.as_ref(), &prompt_ids)?;
        let (all_ids, committed) = match guarantee {
            IncrementalDecodeGuarantee::RawTokenBytes => (Vec::new(), Vec::new()),
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } => {
                let start = prompt_ids.len().saturating_sub(tokens);
                let retained = prompt_ids[start..].to_vec();
                validate_piece_sequence(backend.as_ref(), &retained, MAX_PENDING_DECODE_BYTES)?;
                let decoded = bounded_decode(backend.as_ref(), &retained)?;
                (retained, decoded)
            }
            IncrementalDecodeGuarantee::TerminalOnly => {
                validate_piece_sequence(backend.as_ref(), &prompt_ids, MAX_PENDING_DECODE_BYTES)?;
                let decoded = bounded_decode(backend.as_ref(), &prompt_ids)?;
                (prompt_ids, decoded)
            }
        };
        let read_offset = all_ids.len();
        Ok(Self {
            backend,
            guarantee,
            all_ids,
            read_offset,
            committed,
            pending_raw_bytes: 0,
            raw_utf8_pending: Vec::new(),
            terminal_pending_tokens: 0,
            flushed: false,
        })
    }

    fn push(&mut self, token_id: i32) -> Result<StableDecodeDelta, TokenizerError> {
        self.push_inner(token_id)
    }

    fn push_inner(&mut self, token_id: i32) -> Result<StableDecodeDelta, TokenizerError> {
        if self.flushed {
            return Err(TokenizerError::new(
                TokenizerErrorKind::AlreadyFlushed,
                "cannot push a token after the native decoder was flushed",
            ));
        }
        let id = checked_id(token_id)?;
        let piece = self.backend.piece_bytes(id).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::InvalidTokenId,
                format!("token id {token_id} is outside the frozen tokenizer vocabulary"),
            )
        })?;
        if piece.len() > MAX_TOKEN_PIECE_BYTES {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "token piece exceeds the native decoder byte bound",
            ));
        }

        match self.guarantee {
            IncrementalDecodeGuarantee::RawTokenBytes => self.push_raw_utf8(&piece),
            IncrementalDecodeGuarantee::TerminalOnly => {
                let next_count = self.terminal_pending_tokens.checked_add(1).ok_or_else(|| {
                    TokenizerError::new(
                        TokenizerErrorKind::LookbehindExceeded,
                        "terminal-only decoder token count overflowed",
                    )
                })?;
                let next_bytes =
                    self.pending_raw_bytes
                        .checked_add(piece.len())
                        .ok_or_else(|| {
                            TokenizerError::new(
                                TokenizerErrorKind::LookbehindExceeded,
                                "terminal-only decoder byte count overflowed",
                            )
                        })?;
                if next_count > MAX_TERMINAL_ONLY_TOKENS || next_bytes > MAX_PENDING_DECODE_BYTES {
                    return Err(TokenizerError::new(
                        TokenizerErrorKind::LookbehindExceeded,
                        "terminal-only decoder buffer exceeded its native bound",
                    ));
                }
                self.all_ids.push(id);
                self.terminal_pending_tokens = next_count;
                self.pending_raw_bytes = next_bytes;
                StableDecodeDelta::try_new(Vec::new(), self.pending_raw_bytes)
            }
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens } => {
                self.push_bounded(id, &piece, tokens)
            }
        }
    }

    fn push_raw_utf8(&mut self, piece: &[u8]) -> Result<StableDecodeDelta, TokenizerError> {
        let combined_len = self
            .raw_utf8_pending
            .len()
            .checked_add(piece.len())
            .ok_or_else(|| {
                TokenizerError::new(
                    TokenizerErrorKind::LookbehindExceeded,
                    "raw UTF-8 decoder byte count overflowed",
                )
            })?;
        if combined_len > MAX_STABLE_DELTA_BYTES + 3 {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "raw stable delta exceeds the native byte bound",
            ));
        }
        let mut combined = Vec::with_capacity(combined_len);
        combined.extend_from_slice(&self.raw_utf8_pending);
        combined.extend_from_slice(piece);

        let (stable_len, pending) = match std::str::from_utf8(&combined) {
            Ok(_) => (combined.len(), Vec::new()),
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                let pending = combined[valid..].to_vec();
                if pending.len() > 3 {
                    return Err(TokenizerError::new(
                        TokenizerErrorKind::InvalidTerminalBytes,
                        "raw tokenizer produced an impossible UTF-8 tail",
                    ));
                }
                (valid, pending)
            }
            Err(_) => {
                return Err(TokenizerError::new(
                    TokenizerErrorKind::DecodeFailed,
                    "raw tokenizer bytes contain a definite UTF-8 error",
                ));
            }
        };
        let stable_bytes = combined[..stable_len].to_vec();
        let delta = StableDecodeDelta::try_new(stable_bytes, pending.len())?;
        self.raw_utf8_pending = pending;
        Ok(delta)
    }

    fn push_bounded(
        &mut self,
        id: u32,
        piece: &[u8],
        tokens: usize,
    ) -> Result<StableDecodeDelta, TokenizerError> {
        let held_tokens = self.all_ids.len().saturating_sub(self.read_offset);
        let next_pending = self
            .pending_raw_bytes
            .checked_add(piece.len())
            .ok_or_else(|| {
                TokenizerError::new(
                    TokenizerErrorKind::LookbehindExceeded,
                    "bounded decoder byte count overflowed",
                )
            })?;
        if held_tokens + 1 > tokens || next_pending > MAX_PENDING_DECODE_BYTES {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "decoder did not stabilize within the proven lookbehind bound",
            ));
        }
        // Work on a small, guarantee-bounded ID candidate and do not mutate
        // retained state until every fallible decode and exactness check has
        // succeeded. `committed` can be large, so it is deliberately never
        // cloned per token.
        let mut candidate_ids = self.all_ids.clone();
        candidate_ids.push(id);
        let decoded = bounded_decode(self.backend.as_ref(), &candidate_ids)?;
        let suffix = decoded
            .strip_prefix(self.committed.as_slice())
            .ok_or_else(|| {
                TokenizerError::new(
                    TokenizerErrorKind::RetroactiveRewrite,
                    "tokenizer attempted to rewrite bytes already emitted by the native decoder",
                )
            })?;
        let raw = raw_pending_from(self.backend.as_ref(), &candidate_ids[self.read_offset..])?;
        match std::str::from_utf8(&raw) {
            Ok(_) => {
                let stable_bytes = suffix.to_vec();
                if stable_bytes.len() > MAX_STABLE_DELTA_BYTES {
                    return Err(TokenizerError::new(
                        TokenizerErrorKind::LookbehindExceeded,
                        "stable decoder delta exceeds the native byte bound",
                    ));
                }
                let retain_from = candidate_ids.len().saturating_sub(tokens);
                let next_ids = candidate_ids[retain_from..].to_vec();
                let next_committed = bounded_decode(self.backend.as_ref(), &next_ids)?;
                self.read_offset = next_ids.len();
                self.all_ids = next_ids;
                self.committed = next_committed;
                self.pending_raw_bytes = 0;
                StableDecodeDelta::try_new(stable_bytes, 0)
            }
            Err(error) if error.error_len().is_none() => {
                // The whole raw pending region is withheld; reporting only the
                // trailing invalid suffix would understate retained bytes.
                self.all_ids = candidate_ids;
                self.pending_raw_bytes = raw.len();
                StableDecodeDelta::try_new(Vec::new(), self.pending_raw_bytes)
            }
            Err(_) => Err(TokenizerError::new(
                TokenizerErrorKind::DecodeFailed,
                "tokenizer pieces contain invalid bytes that cannot be completed by a later token",
            )),
        }
    }

    fn raw_pending(&self) -> Result<Vec<u8>, TokenizerError> {
        raw_pending_from(self.backend.as_ref(), &self.all_ids[self.read_offset..])
    }

    fn flush(&mut self) -> Result<StableDecodeDelta, TokenizerError> {
        // Flush consumes the decoder even when terminal validation fails. A
        // caller cannot repair an incomplete/invalid terminal sequence by
        // pushing more bytes after observing a terminal error.
        self.flush_inner()
    }

    fn flush_inner(&mut self) -> Result<StableDecodeDelta, TokenizerError> {
        if self.flushed {
            return Err(TokenizerError::new(
                TokenizerErrorKind::AlreadyFlushed,
                "native decoder flush may be called exactly once",
            ));
        }
        self.flushed = true;
        if self.guarantee == IncrementalDecodeGuarantee::RawTokenBytes {
            if !self.raw_utf8_pending.is_empty() {
                return Err(TokenizerError::new(
                    TokenizerErrorKind::IncompleteTerminalBytes,
                    "raw tokenizer ended with an incomplete UTF-8 sequence",
                ));
            }
            return StableDecodeDelta::try_new(Vec::new(), 0);
        }

        let raw = self.raw_pending()?;
        if let Err(error) = std::str::from_utf8(&raw) {
            let kind = if error.error_len().is_none() {
                TokenizerErrorKind::IncompleteTerminalBytes
            } else {
                TokenizerErrorKind::InvalidTerminalBytes
            };
            return Err(TokenizerError::new(
                kind,
                "terminal tokenizer bytes are incomplete or invalid; replacement output is forbidden",
            ));
        }

        if self.all_ids.len().saturating_sub(self.read_offset) > MAX_TERMINAL_ONLY_TOKENS
            || self.pending_raw_bytes > MAX_PENDING_DECODE_BYTES
        {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "terminal decode buffer exceeds the native bound",
            ));
        }
        let decoded = bounded_decode(self.backend.as_ref(), &self.all_ids)?;
        let suffix = decoded
            .strip_prefix(self.committed.as_slice())
            .ok_or_else(|| {
                TokenizerError::new(
                    TokenizerErrorKind::RetroactiveRewrite,
                    "tokenizer attempted to rewrite bytes already emitted by the native decoder",
                )
            })?;
        let stable_bytes = suffix.to_vec();
        if stable_bytes.len() > MAX_STABLE_DELTA_BYTES {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "terminal stable delta exceeds the native byte bound",
            ));
        }
        self.committed = decoded;
        self.read_offset = self.all_ids.len();
        self.pending_raw_bytes = 0;
        StableDecodeDelta::try_new(stable_bytes, 0)
    }
}

/// Validate the complete prompt byte stream without retaining or emitting it.
/// Only the at-most-three-byte incomplete UTF-8 suffix crosses iterations, so
/// a maximum-size prompt cannot cause an aggregate prompt-byte allocation.
fn validate_prompt_utf8(
    backend: &dyn DecoderBackend,
    prompt_ids: &[u32],
) -> Result<(), TokenizerError> {
    let mut pending = Vec::new();
    for &id in prompt_ids {
        let piece = backend.piece_bytes(id).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::InvalidTokenId,
                format!("prompt token id {id} is outside the frozen tokenizer vocabulary"),
            )
        })?;
        if piece.len() > MAX_TOKEN_PIECE_BYTES {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "prompt token piece exceeds the native decoder byte bound",
            ));
        }
        let combined_len = pending.len().checked_add(piece.len()).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "prompt UTF-8 validation buffer overflowed",
            )
        })?;
        let mut combined = Vec::with_capacity(combined_len);
        combined.extend_from_slice(&pending);
        combined.extend_from_slice(&piece);
        match std::str::from_utf8(&combined) {
            Ok(_) => pending.clear(),
            Err(error) if error.error_len().is_none() => {
                pending = combined[error.valid_up_to()..].to_vec();
                if pending.len() > 3 {
                    return Err(TokenizerError::new(
                        TokenizerErrorKind::InvalidTerminalBytes,
                        "prompt tokenizer produced an impossible UTF-8 tail",
                    ));
                }
            }
            Err(_) => {
                return Err(TokenizerError::new(
                    TokenizerErrorKind::DecodeFailed,
                    "prompt tokenizer bytes contain a definite UTF-8 error",
                ));
            }
        }
    }
    if !pending.is_empty() {
        return Err(TokenizerError::new(
            TokenizerErrorKind::IncompleteTerminalBytes,
            "prompt tokenizer bytes end with an incomplete UTF-8 sequence",
        ));
    }
    Ok(())
}

fn validate_piece_sequence(
    backend: &dyn DecoderBackend,
    ids: &[u32],
    max_bytes: usize,
) -> Result<(), TokenizerError> {
    let mut total = 0usize;
    for &id in ids {
        let piece = backend.piece_bytes(id).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::InvalidTokenId,
                format!("token id {id} is outside the frozen tokenizer vocabulary"),
            )
        })?;
        total = total.checked_add(piece.len()).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "decoder piece-byte total overflowed",
            )
        })?;
        if piece.len() > MAX_TOKEN_PIECE_BYTES || total > max_bytes {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "decoder input exceeds its native byte bound",
            ));
        }
    }
    Ok(())
}

fn raw_pending_from(backend: &dyn DecoderBackend, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
    let mut raw = Vec::new();
    for &id in ids {
        let piece = backend.piece_bytes(id).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::InvalidTokenId,
                format!("token id {id} is outside the frozen tokenizer vocabulary"),
            )
        })?;
        let next = raw.len().checked_add(piece.len()).ok_or_else(|| {
            TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "pending raw byte count overflowed",
            )
        })?;
        if piece.len() > MAX_TOKEN_PIECE_BYTES || next > MAX_PENDING_DECODE_BYTES {
            return Err(TokenizerError::new(
                TokenizerErrorKind::LookbehindExceeded,
                "pending raw bytes exceed the native decoder bound",
            ));
        }
        raw.extend_from_slice(&piece);
    }
    Ok(raw)
}

fn bounded_decode(backend: &dyn DecoderBackend, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
    if backend
        .decoded_len_upper_bound(ids)
        .is_some_and(|bound| bound > MAX_DECODED_WINDOW_BYTES)
    {
        return Err(TokenizerError::new(
            TokenizerErrorKind::LookbehindExceeded,
            "tokenizer decoded output upper bound exceeds the native byte bound",
        ));
    }
    let decoded = backend.decode(ids)?;
    if decoded.len() > MAX_DECODED_WINDOW_BYTES {
        return Err(TokenizerError::new(
            TokenizerErrorKind::LookbehindExceeded,
            "tokenizer decoded output exceeds the native byte bound",
        ));
    }
    Ok(decoded)
}

fn checked_id(id: i32) -> Result<u32, TokenizerError> {
    u32::try_from(id).map_err(|_| {
        TokenizerError::new(
            TokenizerErrorKind::InvalidTokenId,
            format!("negative token id {id} is invalid"),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::rc::Rc;

    use base64::Engine;
    use sha2::{Digest, Sha256};

    use super::*;
    use tempfile::TempDir;

    fn materialize_sentencepiece_fixture(root: &Path) {
        const EXPECTED_SIZE: usize = 253_270;
        const EXPECTED_SHA256: [u8; 32] = [
            0x68, 0xe8, 0x2c, 0xf4, 0xc9, 0xc3, 0xef, 0x0b, 0xe1, 0xf2, 0x5a, 0x7c, 0xae, 0x2c,
            0x83, 0x6f, 0xca, 0xfa, 0x7d, 0x3c, 0x83, 0x82, 0x1f, 0xb9, 0x86, 0x41, 0x1e, 0x56,
            0x77, 0x9a, 0xd1, 0xe2,
        ];
        let encoded = include_str!("../../testdata/native_runtime/sentencepiece-toy.model.b64")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("pinned SentencePiece fixture base64 is corrupt");
        assert_eq!(bytes.len(), EXPECTED_SIZE);
        assert_eq!(<[u8; 32]>::from(Sha256::digest(&bytes)), EXPECTED_SHA256);
        fs::write(root.join("tokenizer.model"), bytes).unwrap();
    }

    fn load_exact_local_tokenizer(path: &Path, primary: &str) -> Arc<MlxcelTokenizer> {
        Arc::new(
            crate::tokenizer::load_tokenizer_local_only_selected(path, primary)
                .unwrap_or_else(|error| panic!("cannot load exact tokenizer {primary:?}: {error}")),
        )
    }

    fn all_singleton_tiktoken_vocab() -> String {
        (0u16..=255)
            .map(|rank| {
                let token = base64::engine::general_purpose::STANDARD.encode([rank as u8]);
                format!("{token} {rank}\n")
            })
            .collect()
    }

    struct FakeBackend {
        kind: NativeTokenizerKind,
        pieces: HashMap<u32, Vec<u8>>,
        decode: fn(&[u32]) -> Result<Vec<u8>, TokenizerError>,
    }

    impl DecoderBackend for FakeBackend {
        fn kind(&self) -> NativeTokenizerKind {
            self.kind
        }

        fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
            (self.decode)(ids)
        }

        fn piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
            self.pieces.get(&id).cloned()
        }

        fn decoded_len_upper_bound(&self, ids: &[u32]) -> Option<usize> {
            ids.iter().try_fold(0usize, |total, id| {
                total.checked_add(self.pieces.get(id)?.len().checked_add(4)?)
            })
        }
    }

    struct ThreadConfinedDecoder(Rc<()>);

    impl NativeIncrementalDecoder for ThreadConfinedDecoder {
        fn push(&mut self, _token_id: i32) -> Result<StableDecodeDelta, TokenizerError> {
            let _ = Rc::strong_count(&self.0);
            StableDecodeDelta::try_new(Vec::new(), 0)
        }

        fn flush(&mut self) -> Result<StableDecodeDelta, TokenizerError> {
            StableDecodeDelta::try_new(Vec::new(), 0)
        }
    }

    static_assertions::assert_not_impl_any!(ThreadConfinedDecoder: Send, Sync);

    #[test]
    fn decoder_trait_accepts_dedicated_thread_confined_implementations() {
        let mut decoder: Box<dyn NativeIncrementalDecoder> =
            Box::new(ThreadConfinedDecoder(Rc::new(())));
        assert!(decoder.push(0).unwrap().stable_bytes().is_empty());
        assert!(decoder.flush().unwrap().stable_bytes().is_empty());
    }

    fn concatenating_backend(kind: NativeTokenizerKind, pieces: &[&[u8]]) -> FakeBackend {
        let pieces = pieces
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect::<HashMap<_, _>>();
        FakeBackend {
            kind,
            pieces,
            decode: |ids| {
                let table = [b"a".as_slice(), b"b".as_slice(), "가".as_bytes()];
                let mut output = Vec::new();
                for id in ids {
                    output.extend_from_slice(table.get(*id as usize).copied().unwrap_or_default());
                }
                Ok(output)
            },
        }
    }

    #[test]
    fn tiktoken_raw_bytes_preserve_split_utf8() {
        let backend: Arc<dyn DecoderBackend> = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::Tiktoken,
            pieces: [(0, vec![0xEA]), (1, vec![0xB0]), (2, vec![0x80])]
                .into_iter()
                .collect(),
            decode: |_| Ok("가".as_bytes().to_vec()),
        });
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::RawTokenBytes).unwrap();
        let first = decoder.push(0).unwrap();
        assert!(first.stable_bytes.is_empty());
        assert_eq!(first.pending_bytes, 1);
        let second = decoder.push(1).unwrap();
        assert!(second.stable_bytes.is_empty());
        assert_eq!(second.pending_bytes, 2);
        let third = decoder.push(2).unwrap();
        assert_eq!(third.stable_bytes, "가".as_bytes());
        assert_eq!(third.pending_bytes, 0);
        assert!(decoder.flush().unwrap().stable_bytes.is_empty());
    }

    #[test]
    fn huggingface_byte_fallback_holds_split_utf8_without_replacement() {
        let tokenizer = Arc::new(MlxcelTokenizer::stub_all_byte_fallback());
        let mut decoder = NativeTokenizerDecoder::from_proven_profile(
            tokenizer,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 },
        )
        .unwrap();
        let bytes = "가".as_bytes();
        assert_eq!(decoder.push(bytes[0] as i32).unwrap().pending_bytes, 1);
        assert_eq!(decoder.push(bytes[1] as i32).unwrap().pending_bytes, 2);
        assert_eq!(
            decoder.push(bytes[2] as i32).unwrap().stable_bytes,
            "가".as_bytes()
        );
    }

    #[test]
    fn sentencepiece_terminal_only_emits_only_on_flush() {
        let backend = Arc::new(concatenating_backend(
            NativeTokenizerKind::SentencePiece,
            &[b"a", b"b"],
        ));
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly).unwrap();
        assert!(decoder.push(0).unwrap().stable_bytes.is_empty());
        assert!(decoder.push(1).unwrap().stable_bytes.is_empty());
        assert_eq!(decoder.flush().unwrap().stable_bytes, b"ab");
    }

    #[test]
    fn actual_sentencepiece_loader_is_terminal_only_and_matches_one_shot_decode() {
        // Apache-2.0 sentencepiece crate fixture; read from the installed crate
        // source rather than duplicating the binary model in this repository.
        let root = TempDir::new().unwrap();
        materialize_sentencepiece_fixture(root.path());
        let tokenizer = load_exact_local_tokenizer(root.path(), "tokenizer.model");
        assert!(matches!(&*tokenizer, MlxcelTokenizer::SentencePiece(_)));

        let ids = tokenizer
            .encode("I saw a girl with a telescope.", false)
            .unwrap();
        assert!(ids.len() > 2);
        let expected = tokenizer.decode(&ids, false).unwrap().into_bytes();
        assert!(std::str::from_utf8(&expected).is_ok());
        let mut decoder =
            NativeTokenizerDecoder::terminal_only(Arc::clone(&tokenizer), &[]).unwrap();
        for &id in &ids {
            let delta = decoder.push(id as i32).unwrap();
            assert!(delta.stable_bytes().is_empty());
        }
        assert_eq!(decoder.flush().unwrap().stable_bytes(), expected);

        let mut oov = NativeTokenizerDecoder::terminal_only(tokenizer, &[]).unwrap();
        assert_eq!(
            oov.push(i32::MAX).unwrap_err().kind(),
            TokenizerErrorKind::InvalidTokenId
        );
    }

    #[test]
    fn actual_gemma_hf_decoder_matches_prompt_generation_one_shot_bytes() {
        let root = TempDir::new().unwrap();
        let MlxcelTokenizer::HuggingFace(fixture) =
            MlxcelTokenizer::stub_gemma_sequence_byte_fallback()
        else {
            unreachable!()
        };
        fs::write(
            root.path().join("tokenizer.json"),
            fixture.to_string(false).unwrap(),
        )
        .unwrap();
        let tokenizer = load_exact_local_tokenizer(root.path(), "tokenizer.json");
        assert!(matches!(&*tokenizer, MlxcelTokenizer::HuggingFace(_)));

        let prompt = vec![tokenizer.token_to_id("▁Hi").unwrap()];
        let generated = [0xEA_u8, 0xB0, 0x80]
            .map(|byte| {
                tokenizer
                    .token_to_id(&format!("<0x{byte:02X}>"))
                    .unwrap_or_else(|| panic!("Gemma tokenizer lacks byte fallback {byte:02X}"))
            })
            .to_vec();
        let prompt_text = tokenizer.decode(&prompt, false).unwrap().into_bytes();
        let mut all = prompt.clone();
        all.extend_from_slice(&generated);
        let one_shot = tokenizer.decode(&all, false).unwrap().into_bytes();
        let expected = one_shot
            .strip_prefix(prompt_text.as_slice())
            .expect("generation rewrote the decoded prompt prefix");

        let prompt_i32 = prompt.iter().map(|&id| id as i32).collect::<Vec<_>>();
        let mut decoder = NativeTokenizerDecoder::from_proven_profile(
            tokenizer,
            &prompt_i32,
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 },
        )
        .unwrap();
        let mut emitted = Vec::new();
        for id in generated {
            let delta = decoder.push(id as i32).unwrap();
            emitted.extend_from_slice(delta.stable_bytes());
        }
        emitted.extend_from_slice(decoder.flush().unwrap().stable_bytes());
        assert_eq!(emitted, expected);
        assert_eq!(emitted, "가".as_bytes());
    }

    #[test]
    fn terminal_incomplete_bytes_are_an_error_not_replacement_text() {
        let tokenizer = Arc::new(MlxcelTokenizer::stub_all_byte_fallback());
        let mut decoder = NativeTokenizerDecoder::from_proven_profile(
            tokenizer,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 },
        )
        .unwrap();
        decoder.push(0xEA).unwrap();
        decoder.push(0xB0).unwrap();
        assert_eq!(
            decoder.flush().unwrap_err().kind(),
            TokenizerErrorKind::IncompleteTerminalBytes
        );
        assert_eq!(
            decoder.push(0x80).unwrap_err().kind(),
            TokenizerErrorKind::AlreadyFlushed
        );
    }

    #[test]
    fn decode_errors_are_preserved() {
        let backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::SentencePiece,
            pieces: [(0, b"a".to_vec())].into_iter().collect(),
            decode: |_| {
                Err(TokenizerError::new(
                    TokenizerErrorKind::DecodeFailed,
                    "synthetic decode failure",
                ))
            },
        });
        assert_eq!(
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly,)
                .err()
                .unwrap()
                .kind(),
            TokenizerErrorKind::DecodeFailed
        );
    }

    #[test]
    fn retroactive_rewrite_is_rejected() {
        let backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::HuggingFace,
            pieces: [(0, b"a ".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())]
                .into_iter()
                .collect(),
            decode: |ids| match ids {
                [0] => Ok(b"a ".to_vec()),
                [0, 1] => Ok(b"ab".to_vec()),
                [0, 2] => Ok(b"a c".to_vec()),
                _ => Ok(Vec::new()),
            },
        });
        let mut decoder = DecoderCore::new(
            backend,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 2 },
        )
        .unwrap();
        assert_eq!(decoder.push(0).unwrap().stable_bytes, b"a ");
        assert_eq!(
            decoder.push(1).unwrap_err().kind(),
            TokenizerErrorKind::RetroactiveRewrite
        );
        assert_eq!(decoder.push(2).unwrap().stable_bytes, b"c");
        assert!(decoder.flush().unwrap().stable_bytes.is_empty());
    }

    #[test]
    fn oov_and_second_flush_are_typed_errors() {
        let backend = Arc::new(concatenating_backend(
            NativeTokenizerKind::SentencePiece,
            &[b"a"],
        ));
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly).unwrap();
        assert_eq!(
            decoder.push(99).unwrap_err().kind(),
            TokenizerErrorKind::InvalidTokenId
        );
        decoder.flush().unwrap();
        assert_eq!(
            decoder.flush().unwrap_err().kind(),
            TokenizerErrorKind::AlreadyFlushed
        );
    }

    #[test]
    fn negative_push_after_flush_and_empty_flush_are_typed() {
        let backend = Arc::new(concatenating_backend(
            NativeTokenizerKind::SentencePiece,
            &[b"a"],
        ));
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly).unwrap();
        assert_eq!(
            decoder.push(-1).unwrap_err().kind(),
            TokenizerErrorKind::InvalidTokenId
        );
        assert!(decoder.flush().unwrap().stable_bytes.is_empty());
        assert_eq!(
            decoder.push(0).unwrap_err().kind(),
            TokenizerErrorKind::AlreadyFlushed
        );
    }

    #[test]
    fn pending_bytes_count_the_complete_prefix_and_incomplete_tail() {
        let backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::HuggingFace,
            pieces: [(0, vec![b'a', 0xEA]), (1, vec![0xB0, 0x80])]
                .into_iter()
                .collect(),
            decode: |ids| match ids {
                [0] => Ok(vec![b'a', 0xEA]),
                [0, 1] => Ok("a가".as_bytes().to_vec()),
                _ => Ok(Vec::new()),
            },
        });
        let mut decoder = DecoderCore::new(
            backend,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 4 },
        )
        .unwrap();
        let first = decoder.push(0).unwrap();
        assert!(first.stable_bytes.is_empty());
        assert_eq!(first.pending_bytes, 2);
        assert_eq!(decoder.push(1).unwrap().stable_bytes, "a가".as_bytes());
    }

    struct OversizedBackend;

    impl DecoderBackend for OversizedBackend {
        fn kind(&self) -> NativeTokenizerKind {
            NativeTokenizerKind::HuggingFace
        }
        fn decode(&self, _ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
            panic!("preflight must reject before decode allocation")
        }
        fn piece_bytes(&self, _id: u32) -> Option<Vec<u8>> {
            Some(b"x".to_vec())
        }
        fn decoded_len_upper_bound(&self, _ids: &[u32]) -> Option<usize> {
            Some(MAX_DECODED_WINDOW_BYTES + 1)
        }
    }

    #[test]
    fn oversized_piece_and_decode_are_rejected_before_state_retention() {
        let piece_backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::HuggingFace,
            pieces: [(0, vec![0; MAX_TOKEN_PIECE_BYTES + 1])]
                .into_iter()
                .collect(),
            decode: |_| Ok(Vec::new()),
        });
        let mut decoder = DecoderCore::new(
            piece_backend,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 1 },
        )
        .unwrap();
        assert_eq!(
            decoder.push(0).unwrap_err().kind(),
            TokenizerErrorKind::LookbehindExceeded
        );

        assert_eq!(
            DecoderCore::new(
                Arc::new(OversizedBackend),
                &[0],
                IncrementalDecodeGuarantee::TerminalOnly,
            )
            .err()
            .unwrap()
            .kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
    }

    #[test]
    fn raw_delta_overflow_does_not_commit_the_completing_piece() {
        let backend: Arc<dyn DecoderBackend> = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::Tiktoken,
            pieces: [
                (0, vec![0xc2]),
                (1, {
                    let mut bytes = vec![b'a'; MAX_STABLE_DELTA_BYTES];
                    bytes[0] = 0xa2;
                    bytes
                }),
                (2, vec![0xa2]),
            ]
            .into_iter()
            .collect(),
            decode: |_| Ok(Vec::new()),
        });
        let mut retry = DecoderCore::new(
            Arc::clone(&backend),
            &[],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        assert_eq!(retry.push(0).unwrap().pending_bytes(), 1);
        assert_eq!(
            retry.push(1).unwrap_err().kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
        assert_eq!(retry.push(2).unwrap().stable_bytes(), "¢".as_bytes());
        assert!(retry.flush().unwrap().stable_bytes().is_empty());

        let mut terminal =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::RawTokenBytes).unwrap();
        terminal.push(0).unwrap();
        assert_eq!(
            terminal.push(1).unwrap_err().kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
        assert_eq!(
            terminal.flush().unwrap_err().kind(),
            TokenizerErrorKind::IncompleteTerminalBytes
        );
    }

    #[test]
    fn terminal_only_delayed_token_span_accepts_4096_and_rejects_4097_without_retention() {
        let backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::SentencePiece,
            pieces: [(0, b"a".to_vec())].into_iter().collect(),
            decode: |ids| Ok(vec![b'a'; ids.len()]),
        });
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly).unwrap();
        for _ in 0..MAX_TERMINAL_ONLY_TOKENS {
            let delta = decoder.push(0).unwrap();
            assert!(delta.stable_bytes.is_empty());
        }
        assert_eq!(
            decoder.push(0).unwrap_err().kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
        let terminal = decoder.flush().unwrap();
        assert_eq!(terminal.stable_bytes.len(), MAX_TERMINAL_ONLY_TOKENS);
        assert!(terminal.stable_bytes.iter().all(|byte| *byte == b'a'));
    }

    #[test]
    fn immediately_stable_tiers_do_not_apply_the_terminal_lifetime_ceiling() {
        let raw_backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::Tiktoken,
            pieces: [(0, b"a".to_vec())].into_iter().collect(),
            decode: |ids| Ok(vec![b'a'; ids.len()]),
        });
        let mut raw =
            DecoderCore::new(raw_backend, &[], IncrementalDecodeGuarantee::RawTokenBytes).unwrap();
        for _ in 0..=MAX_TERMINAL_ONLY_TOKENS {
            let delta = raw.push(0).unwrap();
            assert_eq!(delta.stable_bytes(), b"a");
            assert_eq!(delta.pending_bytes(), 0);
        }

        let bounded_backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::HuggingFace,
            pieces: [(0, b"a".to_vec())].into_iter().collect(),
            decode: |ids| Ok(vec![b'a'; ids.len()]),
        });
        let mut bounded = DecoderCore::new(
            bounded_backend,
            &[],
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 1 },
        )
        .unwrap();
        for _ in 0..=MAX_TERMINAL_ONLY_TOKENS {
            let delta = bounded.push(0).unwrap();
            assert_eq!(delta.stable_bytes(), b"a");
            assert_eq!(delta.pending_bytes(), 0);
        }
    }

    #[test]
    fn terminal_only_delayed_byte_span_accepts_one_mib_and_rejects_next_byte() {
        let backend = Arc::new(FakeBackend {
            kind: NativeTokenizerKind::SentencePiece,
            pieces: [
                (0, vec![b'a'; MAX_PENDING_DECODE_BYTES]),
                (1, b"b".to_vec()),
            ]
            .into_iter()
            .collect(),
            decode: |ids| match ids {
                [0] => Ok(vec![b'a'; MAX_PENDING_DECODE_BYTES]),
                [0, 1] => {
                    let mut output = vec![b'a'; MAX_PENDING_DECODE_BYTES];
                    output.push(b'b');
                    Ok(output)
                }
                _ => Ok(Vec::new()),
            },
        });
        let mut decoder =
            DecoderCore::new(backend, &[], IncrementalDecodeGuarantee::TerminalOnly).unwrap();
        assert_eq!(
            decoder.push(0).unwrap().pending_bytes,
            MAX_PENDING_DECODE_BYTES
        );
        assert_eq!(
            decoder.push(1).unwrap_err().kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
        let terminal = decoder.flush().unwrap();
        assert_eq!(terminal.stable_bytes.len(), MAX_PENDING_DECODE_BYTES);
        assert!(terminal.stable_bytes.iter().all(|byte| *byte == b'a'));
    }

    #[test]
    fn each_strategy_concatenates_to_its_authoritative_one_shot_decode() {
        for guarantee in [
            IncrementalDecodeGuarantee::RawTokenBytes,
            IncrementalDecodeGuarantee::ProvenBoundedLookbehind { tokens: 2 },
            IncrementalDecodeGuarantee::TerminalOnly,
        ] {
            let kind = if guarantee == IncrementalDecodeGuarantee::RawTokenBytes {
                NativeTokenizerKind::Tiktoken
            } else {
                NativeTokenizerKind::HuggingFace
            };
            let backend = Arc::new(concatenating_backend(kind, &[b"a", b"b"]));
            let mut decoder = DecoderCore::new(backend, &[], guarantee).unwrap();
            let mut emitted = Vec::new();
            emitted.extend(decoder.push(0).unwrap().stable_bytes);
            emitted.extend(decoder.push(1).unwrap().stable_bytes);
            emitted.extend(decoder.flush().unwrap().stable_bytes);
            assert_eq!(emitted, b"ab");
        }
    }

    #[test]
    fn real_tiktoken_backend_preserves_raw_split_utf8() {
        let root = TempDir::new().unwrap();
        fs::write(
            root.path().join("tiny.tiktoken"),
            all_singleton_tiktoken_vocab(),
        )
        .unwrap();
        let tokenizer = Arc::new(crate::tokenizer::load_tokenizer_local_only(root.path()).unwrap());
        assert!(matches!(tokenizer.as_ref(), MlxcelTokenizer::Tiktoken(_)));
        let split = [0xEA_u32, 0xB0, 0x80];
        assert_eq!(tokenizer.decode(&split, false).unwrap(), "가");
        let mut decoder = NativeTokenizerDecoder::from_proven_profile(
            Arc::clone(&tokenizer),
            &[],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        let mut bytes = Vec::new();
        let first = decoder.push(split[0] as i32).unwrap();
        assert!(first.stable_bytes.is_empty());
        assert_eq!(first.pending_bytes, 1);
        let second = decoder.push(split[1] as i32).unwrap();
        assert!(second.stable_bytes.is_empty());
        assert_eq!(second.pending_bytes, 2);
        bytes.extend(decoder.push(split[2] as i32).unwrap().stable_bytes);
        bytes.extend(decoder.flush().unwrap().stable_bytes);
        assert_eq!(bytes, "가".as_bytes());

        let mut incomplete = NativeTokenizerDecoder::from_proven_profile(
            Arc::clone(&tokenizer),
            &[],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        incomplete.push(split[0] as i32).unwrap();
        assert_eq!(
            incomplete.flush().unwrap_err().kind(),
            TokenizerErrorKind::IncompleteTerminalBytes
        );
        assert_eq!(
            incomplete.push(split[1] as i32).unwrap_err().kind(),
            TokenizerErrorKind::AlreadyFlushed
        );

        let mut invalid = NativeTokenizerDecoder::from_proven_profile(
            Arc::clone(&tokenizer),
            &[],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        assert_eq!(
            invalid.push(0xFF).unwrap_err().kind(),
            TokenizerErrorKind::DecodeFailed
        );
        assert_eq!(invalid.push(b'a' as i32).unwrap().stable_bytes, b"a");
        assert!(invalid.flush().unwrap().stable_bytes.is_empty());
        assert_eq!(tokenizer.decode(&[0xFF], false).unwrap(), "\u{fffd}");

        let mut oov = NativeTokenizerDecoder::from_proven_profile(
            Arc::clone(&tokenizer),
            &[],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        assert_eq!(
            oov.push(999).unwrap_err().kind(),
            TokenizerErrorKind::InvalidTokenId
        );
        assert_eq!(tokenizer.decode(&[999], false).unwrap(), "");
        assert_eq!(oov.push(b'a' as i32).unwrap().stable_bytes, b"a");

        let mut prompt_bound = NativeTokenizerDecoder::from_proven_profile(
            Arc::clone(&tokenizer),
            &[b'a' as i32],
            IncrementalDecodeGuarantee::RawTokenBytes,
        )
        .unwrap();
        let mut generated = Vec::new();
        generated.extend(prompt_bound.push(split[0] as i32).unwrap().stable_bytes);
        generated.extend(prompt_bound.push(split[1] as i32).unwrap().stable_bytes);
        generated.extend(prompt_bound.push(split[2] as i32).unwrap().stable_bytes);
        generated.extend(prompt_bound.flush().unwrap().stable_bytes);
        assert_eq!(generated, "가".as_bytes());
        let mut prompt_and_generated = b"a".to_vec();
        prompt_and_generated.extend_from_slice(&generated);
        assert_eq!(
            prompt_and_generated,
            tokenizer
                .decode(&[b'a' as u32, split[0], split[1], split[2]], false)
                .unwrap()
                .as_bytes()
        );

        assert_eq!(
            NativeTokenizerDecoder::from_proven_profile(
                Arc::clone(&tokenizer),
                &[split[0] as i32],
                IncrementalDecodeGuarantee::RawTokenBytes,
            )
            .err()
            .unwrap()
            .kind(),
            TokenizerErrorKind::IncompleteTerminalBytes
        );
        assert_eq!(
            NativeTokenizerDecoder::from_proven_profile(
                Arc::clone(&tokenizer),
                &[0xFF],
                IncrementalDecodeGuarantee::RawTokenBytes,
            )
            .err()
            .unwrap()
            .kind(),
            TokenizerErrorKind::DecodeFailed
        );
        assert_eq!(
            NativeTokenizerDecoder::from_proven_profile(
                tokenizer,
                &[-1],
                IncrementalDecodeGuarantee::RawTokenBytes,
            )
            .err()
            .unwrap()
            .kind(),
            TokenizerErrorKind::InvalidTokenId
        );
    }

    #[test]
    fn stable_delta_checked_constructor_enforces_each_public_bound() {
        let exact = StableDecodeDelta::try_new(
            vec![b'x'; MAX_STABLE_DELTA_BYTES],
            MAX_PENDING_DECODE_BYTES,
        )
        .unwrap();
        assert_eq!(exact.stable_bytes().len(), MAX_STABLE_DELTA_BYTES);
        assert_eq!(exact.pending_bytes(), MAX_PENDING_DECODE_BYTES);
        assert_eq!(
            StableDecodeDelta::try_new(vec![0; MAX_STABLE_DELTA_BYTES + 1], 0)
                .unwrap_err()
                .kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
        assert_eq!(
            StableDecodeDelta::try_new(Vec::new(), MAX_PENDING_DECODE_BYTES + 1)
                .unwrap_err()
                .kind(),
            TokenizerErrorKind::LookbehindExceeded
        );
    }
}
