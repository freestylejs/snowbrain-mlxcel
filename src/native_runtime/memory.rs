// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Bounded worker-health transitions after allocation/evaluation failures.

use super::{NativeRecoverability, NativeWorkerHealth};

pub(crate) const fn health_after_failure(
    current: NativeWorkerHealth,
    recoverability: NativeRecoverability,
) -> NativeWorkerHealth {
    match recoverability {
        NativeRecoverability::Request => current,
        NativeRecoverability::WorkerReload => NativeWorkerHealth::ReloadRequired,
        NativeRecoverability::ProcessInvariant => NativeWorkerHealth::Failed,
    }
}

pub(crate) fn classify_allocation_failure(
    message: &str,
    worker_reload: bool,
) -> Option<NativeRecoverability> {
    let message = message.to_ascii_lowercase();
    let allocation_failure = message.contains("out of memory")
        || message.contains("allocation failed")
        || message.contains("failed to allocate")
        || message.contains("memory limit")
        || message.contains("resource exhausted");
    allocation_failure.then_some(if worker_reload {
        NativeRecoverability::WorkerReload
    } else {
        NativeRecoverability::Request
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_failure_isolated_but_worker_and_process_failures_escalate() {
        let ready = NativeWorkerHealth::Ready;
        assert_eq!(
            health_after_failure(ready, NativeRecoverability::Request),
            NativeWorkerHealth::Ready
        );
        assert_eq!(
            health_after_failure(ready, NativeRecoverability::WorkerReload),
            NativeWorkerHealth::ReloadRequired
        );
        assert_eq!(
            health_after_failure(ready, NativeRecoverability::ProcessInvariant),
            NativeWorkerHealth::Failed
        );
    }

    #[test]
    fn allocation_failures_are_typed_without_misclassifying_generic_eval_errors() {
        assert_eq!(
            classify_allocation_failure("Metal allocation failed: out of memory", false),
            Some(NativeRecoverability::Request)
        );
        assert_eq!(
            classify_allocation_failure("resource exhausted at memory limit", true),
            Some(NativeRecoverability::WorkerReload)
        );
        assert_eq!(
            classify_allocation_failure("invalid graph capture invariant", true),
            None
        );
    }
}
