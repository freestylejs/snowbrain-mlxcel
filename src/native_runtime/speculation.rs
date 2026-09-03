// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Physical accounting kept separate from frontend semantic commitment.

use super::{NativeDirectiveApplication, NativeUsageDelta};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePhysicalAccounting {
    pub input_tokens_evaluated: u64,
    pub input_tokens_reused: u64,
    pub output_tokens_target_accepted: u64,
    pub output_units_committed: u64,
    pub accepted_suffix_discarded: u64,
    pub speculative_tokens_evaluated: u64,
    pub draft_tokens_rejected: u64,
}

impl NativePhysicalAccounting {
    pub(crate) fn record_chunk(
        &mut self,
        usage: NativeUsageDelta,
        application: NativeDirectiveApplication,
    ) {
        self.input_tokens_evaluated = self
            .input_tokens_evaluated
            .saturating_add(usage.input_tokens_evaluated);
        self.input_tokens_reused = self
            .input_tokens_reused
            .saturating_add(usage.input_tokens_reused);
        self.output_tokens_target_accepted = self
            .output_tokens_target_accepted
            .saturating_add(usage.output_tokens_accepted);
        self.output_units_committed = self
            .output_units_committed
            .saturating_add(application.committed_units() as u64);
        self.accepted_suffix_discarded = self
            .accepted_suffix_discarded
            .saturating_add(application.accepted_suffix_discarded_units() as u64);
        self.speculative_tokens_evaluated = self
            .speculative_tokens_evaluated
            .saturating_add(usage.speculative_tokens_evaluated);
        self.draft_tokens_rejected = self
            .draft_tokens_rejected
            .saturating_add(usage.speculative_tokens_discarded);
    }
}
