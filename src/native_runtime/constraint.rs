// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Sequence-local constraint transition ledger.

use super::{NativeConstraintPlan, NativeConstraintTransition, NativeRuntimeError};

#[derive(Debug, Clone)]
pub(crate) struct NativeConstraintState {
    current: NativeConstraintPlan,
    revision: u64,
    sampled_revision: u64,
}

impl NativeConstraintState {
    pub(crate) fn new(current: NativeConstraintPlan) -> Self {
        Self {
            current,
            revision: 0,
            sampled_revision: 0,
        }
    }

    pub(crate) fn apply(
        &mut self,
        transition: &NativeConstraintTransition,
    ) -> Result<(), NativeRuntimeError> {
        transition.validate()?;
        self.current = match transition {
            NativeConstraintTransition::Unchanged => return Ok(()),
            NativeConstraintTransition::Clear => NativeConstraintPlan::None,
            NativeConstraintTransition::Replace { format, payload } => {
                NativeConstraintPlan::Dynamic {
                    initial_format: format.clone(),
                    initial_payload: payload.clone(),
                }
            }
        };
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub(crate) fn mark_dependent_sample(&mut self) {
        self.sampled_revision = self.revision;
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn sampled_revision(&self) -> u64 {
        self.sampled_revision
    }

    pub(crate) fn current(&self) -> &NativeConstraintPlan {
        &self.current
    }
}
