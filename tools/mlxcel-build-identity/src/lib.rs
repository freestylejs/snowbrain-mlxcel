// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Build-only producer for mlxcel consumer and post-link provenance.
//!
//! A dependency cannot observe its consumer's direct Cargo edge, lockfile, or
//! final linker. Qualification runners use this crate from their build script
//! and as their linker wrapper. No runtime or server semantics live here.

#[path = "../../../build_support/native_identity.rs"]
pub mod canonical;

mod consumer;
mod post_link;

pub use consumer::{
    ConsumerFragment, ConsumerProducerConfig, GitCargoSource, produce_consumer_fragment,
    render_consumer_fragment_rust,
};
pub use post_link::{PostLinkConfig, PostLinkSidecar, produce_post_link_sidecar};
