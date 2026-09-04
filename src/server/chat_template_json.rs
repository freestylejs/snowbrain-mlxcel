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

//! Compatibility facade for the transport-neutral CPython-compatible prompt
//! JSON filter. The implementation lives below the server boundary.

use minijinja::Error;
use minijinja::value::{Kwargs, Value};

/// Preserve the server module's public helper while delegating to the single
/// neutral implementation and the server's established 64 MiB policy ceiling.
pub fn python_tojson(value: Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    crate::native_runtime::python_tojson_for_server(value, indent, kwargs)
}
