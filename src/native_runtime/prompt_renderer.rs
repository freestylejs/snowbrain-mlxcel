// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Immutable-package-bound, transport-neutral prompt rendering.

use std::collections::HashSet;
use std::io::{self, Write};
use std::sync::Arc;

use minijinja::{Environment, ErrorKind, Value};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use sha2::{Digest, Sha256};

use super::error::{NativeRuntimeError, NativeRuntimeErrorKind};
use super::model_report::{NativeImmutablePackage, NativeModelReport, NativePackageComponent};

pub const MAX_NATIVE_PROMPT_TEMPLATE_BYTES: usize = 1024 * 1024;
pub const MAX_NATIVE_PROMPT_CONTEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_PROMPT_CONTEXT_DEPTH: usize = 64;
pub const MAX_NATIVE_PROMPT_CONTEXT_NODES: usize = 65_536;
pub const MAX_NATIVE_RENDERED_PROMPT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NATIVE_PROMPT_RENDER_FUEL: u64 = 50_000_000;
pub const NATIVE_PROMPT_RENDERER_SCHEMA: &str = "mlxcel.native-prompt-renderer.v1";

fn prompt_error(kind: NativeRuntimeErrorKind, message: impl AsRef<str>) -> NativeRuntimeError {
    NativeRuntimeError::new(kind, message)
}

/// Exact per-render ceilings. Construction rejects zero values and ceilings
/// larger than the fixed native schema permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativePromptRendererLimits {
    max_template_bytes: usize,
    max_context_bytes: usize,
    max_context_depth: usize,
    max_context_nodes: usize,
    max_output_bytes: usize,
    fuel: u64,
}

impl NativePromptRendererLimits {
    pub fn try_new(
        max_template_bytes: usize,
        max_context_bytes: usize,
        max_context_depth: usize,
        max_context_nodes: usize,
        max_output_bytes: usize,
        fuel: u64,
    ) -> Result<Self, NativeRuntimeError> {
        let limits = Self {
            max_template_bytes,
            max_context_bytes,
            max_context_depth,
            max_context_nodes,
            max_output_bytes,
            fuel,
        };
        if max_template_bytes == 0
            || max_template_bytes > MAX_NATIVE_PROMPT_TEMPLATE_BYTES
            || max_context_bytes == 0
            || max_context_bytes > MAX_NATIVE_PROMPT_CONTEXT_BYTES
            || max_context_depth == 0
            || max_context_depth > MAX_NATIVE_PROMPT_CONTEXT_DEPTH
            || max_context_nodes == 0
            || max_context_nodes > MAX_NATIVE_PROMPT_CONTEXT_NODES
            || max_output_bytes == 0
            || max_output_bytes > MAX_NATIVE_RENDERED_PROMPT_BYTES
            || fuel == 0
            || fuel > MAX_NATIVE_PROMPT_RENDER_FUEL
        {
            return Err(prompt_error(
                NativeRuntimeErrorKind::ResourceLimit,
                "prompt renderer limits are zero or exceed a native hard ceiling",
            ));
        }
        Ok(limits)
    }

    #[must_use]
    pub const fn max_template_bytes(self) -> usize {
        self.max_template_bytes
    }

    #[must_use]
    pub const fn max_context_bytes(self) -> usize {
        self.max_context_bytes
    }

    #[must_use]
    pub const fn max_context_depth(self) -> usize {
        self.max_context_depth
    }

    #[must_use]
    pub const fn max_context_nodes(self) -> usize {
        self.max_context_nodes
    }

    #[must_use]
    pub const fn max_output_bytes(self) -> usize {
        self.max_output_bytes
    }

    #[must_use]
    pub const fn fuel(self) -> u64 {
        self.fuel
    }
}

impl Default for NativePromptRendererLimits {
    fn default() -> Self {
        Self {
            max_template_bytes: MAX_NATIVE_PROMPT_TEMPLATE_BYTES,
            max_context_bytes: MAX_NATIVE_PROMPT_CONTEXT_BYTES,
            max_context_depth: MAX_NATIVE_PROMPT_CONTEXT_DEPTH,
            max_context_nodes: MAX_NATIVE_PROMPT_CONTEXT_NODES,
            max_output_bytes: MAX_NATIVE_RENDERED_PROMPT_BYTES,
            fuel: MAX_NATIVE_PROMPT_RENDER_FUEL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativePromptRendererKind {
    LiteralUtf8,
    ImmutableJinja,
}

/// Explicit immutable components and policy selected by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePromptRendererSpec {
    pub kind: NativePromptRendererKind,
    pub template_component: Option<NativePackageComponent>,
    pub tokenizer_config_component: NativePackageComponent,
    pub limits: NativePromptRendererLimits,
    pub add_special_tokens: bool,
}

/// Generic semantic-free context value. Object order is retained exactly and
/// duplicate keys are rejected by [`Self::object`] and again before rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeTemplateValue {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    String(Arc<str>),
    Sequence(Arc<[NativeTemplateValue]>),
    Object(Arc<[(Arc<str>, NativeTemplateValue)]>),
}

impl NativeTemplateValue {
    #[must_use]
    pub fn string(value: impl AsRef<str>) -> Self {
        Self::String(Arc::from(value.as_ref()))
    }

    #[must_use]
    pub fn sequence(values: impl IntoIterator<Item = Self>) -> Self {
        Self::Sequence(values.into_iter().collect::<Vec<_>>().into())
    }

    pub fn object<K: AsRef<str>>(
        entries: impl IntoIterator<Item = (K, Self)>,
    ) -> Result<Self, NativeRuntimeError> {
        let mut keys = HashSet::new();
        let mut output = Vec::new();
        for (key, value) in entries {
            let key = key.as_ref();
            if !keys.insert(key.to_owned()) {
                return Err(prompt_error(
                    NativeRuntimeErrorKind::InvalidRequest,
                    "prompt context object contains a duplicate key",
                ));
            }
            output.push((Arc::from(key), value));
        }
        Ok(Self::Object(output.into()))
    }
}

impl Serialize for NativeTemplateValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::I64(value) => serializer.serialize_i64(*value),
            Self::U64(value) => serializer.serialize_u64(*value),
            Self::String(value) => serializer.serialize_str(value),
            Self::Sequence(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values.iter() {
                    sequence.serialize_element(value)?;
                }
                sequence.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries.iter() {
                    map.serialize_entry(key.as_ref(), value)?;
                }
                map.end()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePromptRendererBinding {
    package_identity: [u8; 32],
    report_identity: [u8; 32],
    tokenizer_identity: [u8; 32],
    template_component: Option<NativePackageComponent>,
    tokenizer_config_component: NativePackageComponent,
    renderer_schema: Arc<str>,
    limits: NativePromptRendererLimits,
    add_special_tokens: bool,
    identity: [u8; 32],
}

impl NativePromptRendererBinding {
    #[must_use]
    pub const fn package_identity(&self) -> [u8; 32] {
        self.package_identity
    }

    #[must_use]
    pub const fn report_identity(&self) -> [u8; 32] {
        self.report_identity
    }

    #[must_use]
    pub const fn tokenizer_identity(&self) -> [u8; 32] {
        self.tokenizer_identity
    }

    #[must_use]
    pub fn template_component(&self) -> Option<&NativePackageComponent> {
        self.template_component.as_ref()
    }

    #[must_use]
    pub const fn tokenizer_config_component(&self) -> &NativePackageComponent {
        &self.tokenizer_config_component
    }

    #[must_use]
    pub fn renderer_schema(&self) -> &str {
        &self.renderer_schema
    }

    #[must_use]
    pub const fn limits(&self) -> NativePromptRendererLimits {
        self.limits
    }

    #[must_use]
    pub const fn add_special_tokens(&self) -> bool {
        self.add_special_tokens
    }

    #[must_use]
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

#[derive(Debug, Clone)]
pub struct NativePromptRenderer {
    kind: NativePromptRendererKind,
    template: Option<Arc<str>>,
    binding: NativePromptRendererBinding,
}

impl NativePromptRenderer {
    pub(crate) fn bind(
        package: &NativeImmutablePackage,
        report: &NativeModelReport,
        spec: NativePromptRendererSpec,
    ) -> Result<Self, NativeRuntimeError> {
        validate_component_binding(package, &spec.tokenizer_config_component)?;
        let selected_tokenizer_config =
            package.component("tokenizer_config.json").ok_or_else(|| {
                prompt_error(
                    NativeRuntimeErrorKind::UnsupportedGuarantee,
                    "prompt rendering requires the tokenizer's declared config component",
                )
            })?;
        if selected_tokenizer_config != &spec.tokenizer_config_component {
            return Err(prompt_error(
                NativeRuntimeErrorKind::InvalidRequest,
                "prompt renderer tokenizer config is not the component selected by the tokenizer loader",
            ));
        }
        let tokenizer_config = package.read_component(spec.tokenizer_config_component.label())?;
        std::str::from_utf8(&tokenizer_config).map_err(|_| {
            prompt_error(
                NativeRuntimeErrorKind::InvalidRequest,
                "tokenizer-config component is not UTF-8",
            )
        })?;
        let tokenizer = package.bind_prompt_tokenizer(report)?;
        let (template, template_component) = match spec.kind {
            NativePromptRendererKind::LiteralUtf8 => {
                if spec.template_component.is_some() {
                    return Err(prompt_error(
                        NativeRuntimeErrorKind::InvalidRequest,
                        "literal prompt renderer cannot bind a template component",
                    ));
                }
                (None, None)
            }
            NativePromptRendererKind::ImmutableJinja => {
                let component = spec.template_component.as_ref().ok_or_else(|| {
                    prompt_error(
                        NativeRuntimeErrorKind::InvalidRequest,
                        "Jinja prompt renderer requires an explicit template component",
                    )
                })?;
                validate_component_binding(package, component)?;
                let template_len = usize::try_from(component.size_bytes()).map_err(|_| {
                    prompt_error(
                        NativeRuntimeErrorKind::ResourceLimit,
                        "template size does not fit this host",
                    )
                })?;
                if template_len > spec.limits.max_template_bytes {
                    return Err(prompt_error(
                        NativeRuntimeErrorKind::ResourceLimit,
                        "template exceeds the renderer byte limit",
                    ));
                }
                let bytes = package.read_component(component.label())?;
                let source = std::str::from_utf8(&bytes).map_err(|_| {
                    prompt_error(
                        NativeRuntimeErrorKind::InvalidRequest,
                        "prompt template component is not UTF-8",
                    )
                })?;
                reject_forbidden_template_features(source)?;
                let mut environment = neutral_environment(
                    spec.limits.fuel,
                    spec.limits.max_output_bytes,
                    spec.limits.max_context_nodes,
                );
                environment
                    .add_template("prompt", source)
                    .map_err(template_error)?;
                (Some(Arc::from(source)), Some(component.clone()))
            }
        };
        let mut binding = NativePromptRendererBinding {
            package_identity: tokenizer.package_identity(),
            report_identity: tokenizer.report_identity(),
            tokenizer_identity: tokenizer.tokenizer_identity(),
            template_component,
            tokenizer_config_component: spec.tokenizer_config_component,
            renderer_schema: Arc::from(NATIVE_PROMPT_RENDERER_SCHEMA),
            limits: spec.limits,
            add_special_tokens: spec.add_special_tokens,
            identity: [0; 32],
        };
        binding.identity = binding_identity(&binding, spec.kind);
        Ok(Self {
            kind: spec.kind,
            template,
            binding,
        })
    }

    #[must_use]
    pub const fn binding(&self) -> &NativePromptRendererBinding {
        &self.binding
    }

    #[must_use]
    pub const fn binding_identity(&self) -> [u8; 32] {
        self.binding.identity()
    }

    pub fn render(
        &self,
        context: &NativeTemplateValue,
    ) -> Result<NativeRenderedPrompt, NativeRuntimeError> {
        validate_context(context, self.binding.limits)?;
        let mut output = BoundedWriter::new(self.binding.limits.max_output_bytes);
        match self.kind {
            NativePromptRendererKind::LiteralUtf8 => match context {
                NativeTemplateValue::String(value) => {
                    output.write_all(value.as_bytes()).map_err(|_| {
                        prompt_error(
                            NativeRuntimeErrorKind::ResourceLimit,
                            "rendered prompt exceeds the output byte limit",
                        )
                    })?
                }
                _ => {
                    return Err(prompt_error(
                        NativeRuntimeErrorKind::InvalidRequest,
                        "literal prompt renderer requires a string value",
                    ));
                }
            },
            NativePromptRendererKind::ImmutableJinja => {
                let mut environment = neutral_environment(
                    self.binding.limits.fuel,
                    self.binding.limits.max_output_bytes,
                    self.binding.limits.max_context_nodes,
                );
                environment
                    .add_template(
                        "prompt",
                        self.template.as_deref().expect("bound Jinja template"),
                    )
                    .map_err(template_error)?;
                let template = environment.get_template("prompt").map_err(template_error)?;
                #[allow(deprecated)]
                template
                    .render_to_write(context, &mut output)
                    .map_err(|error| {
                        if output.exceeded || error.kind() == minijinja::ErrorKind::OutOfFuel {
                            prompt_error(
                                NativeRuntimeErrorKind::ResourceLimit,
                                "prompt rendering exceeded its output or fuel limit",
                            )
                        } else {
                            template_error(error)
                        }
                    })?;
            }
        }
        let rendered = String::from_utf8(output.bytes).map_err(|_| {
            prompt_error(
                NativeRuntimeErrorKind::InvalidRequest,
                "prompt renderer produced non-UTF-8 output",
            )
        })?;
        let output_identity = rendered_prompt_identity(&self.binding, rendered.as_bytes());
        Ok(NativeRenderedPrompt(Arc::new(NativeRenderedPromptInner {
            rendered_utf8: Arc::from(rendered),
            binding: self.binding.clone(),
            output_identity,
        })))
    }
}

/// Receipt produced only by a successfully bound native renderer.
///
/// Consumers can inspect its immutable facts but cannot forge one:
///
/// ```compile_fail
/// use mlxcel::native_runtime::NativeRenderedPrompt;
/// let _forged = NativeRenderedPrompt {};
/// ```
#[derive(Debug)]
pub struct NativeRenderedPrompt(Arc<NativeRenderedPromptInner>);

#[derive(Debug)]
struct NativeRenderedPromptInner {
    rendered_utf8: Arc<str>,
    binding: NativePromptRendererBinding,
    output_identity: [u8; 32],
}

impl NativeRenderedPrompt {
    #[must_use]
    pub fn rendered_utf8(&self) -> &str {
        &self.0.rendered_utf8
    }

    #[must_use]
    pub fn binding(&self) -> &NativePromptRendererBinding {
        &self.0.binding
    }

    #[must_use]
    pub fn binding_identity(&self) -> [u8; 32] {
        self.0.binding.identity()
    }

    #[must_use]
    pub fn package_identity(&self) -> [u8; 32] {
        self.0.binding.package_identity()
    }

    #[must_use]
    pub fn report_identity(&self) -> [u8; 32] {
        self.0.binding.report_identity()
    }

    #[must_use]
    pub fn tokenizer_identity(&self) -> [u8; 32] {
        self.0.binding.tokenizer_identity()
    }

    #[must_use]
    pub fn output_identity(&self) -> [u8; 32] {
        self.0.output_identity
    }

    #[must_use]
    pub fn add_special_tokens(&self) -> bool {
        self.0.binding.add_special_tokens()
    }

    pub(crate) fn shallow_clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

fn validate_component_binding(
    package: &NativeImmutablePackage,
    expected: &NativePackageComponent,
) -> Result<(), NativeRuntimeError> {
    let actual = package.component(expected.label()).ok_or_else(|| {
        prompt_error(
            NativeRuntimeErrorKind::InvalidPackage,
            "prompt renderer component label is absent from the immutable package",
        )
    })?;
    if actual != expected {
        return Err(prompt_error(
            NativeRuntimeErrorKind::InvalidRequest,
            "prompt renderer component size or digest disagrees with the immutable package",
        ));
    }
    Ok(())
}

fn reject_forbidden_template_features(source: &str) -> Result<(), NativeRuntimeError> {
    let mut rest = source;
    while let Some(start) = match (rest.find("{#"), rest.find("{%")) {
        (Some(comment), Some(block)) => Some(comment.min(block)),
        (Some(comment), None) => Some(comment),
        (None, Some(block)) => Some(block),
        (None, None) => None,
    } {
        rest = &rest[start..];
        if rest.starts_with("{#") {
            let Some(end) = rest.find("#}") else {
                break;
            };
            rest = &rest[end + 2..];
            continue;
        }
        let Some(end) = rest.find("%}") else {
            break;
        };
        let tag = rest[2..end].trim_start_matches(['-', '+']).trim();
        let keyword = tag.split_whitespace().next().unwrap_or_default();
        if matches!(keyword, "include" | "import" | "from" | "extends") {
            return Err(prompt_error(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "prompt template requests forbidden external template resolution",
            ));
        }
        if tag.contains("strftime_now") {
            return Err(prompt_error(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "prompt template requests a non-deterministic time helper",
            ));
        }
        rest = &rest[end + 2..];
    }
    let mut rest = source;
    while let Some(start) = rest.find("{{") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("}}") else {
            break;
        };
        if rest[..end].contains("strftime_now") {
            return Err(prompt_error(
                NativeRuntimeErrorKind::UnsupportedGuarantee,
                "prompt template requests a non-deterministic time helper",
            ));
        }
        rest = &rest[end + 2..];
    }
    Ok(())
}

#[derive(Debug)]
struct PromptHelperResourceLimit;

impl std::fmt::Display for PromptHelperResourceLimit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("prompt helper resource limit")
    }
}

impl std::error::Error for PromptHelperResourceLimit {}

fn helper_resource_error() -> minijinja::Error {
    minijinja::Error::new(
        ErrorKind::InvalidOperation,
        "prompt helper exceeds its selected byte or node limit",
    )
    .with_source(PromptHelperResourceLimit)
}

fn is_helper_resource_limit(error: &minijinja::Error) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if cause.downcast_ref::<PromptHelperResourceLimit>().is_some() {
            return true;
        }
        current = cause.source();
    }
    false
}

fn checked_replace(
    source: &str,
    old: &str,
    new: &str,
    maximum: usize,
) -> Result<Value, minijinja::Error> {
    let replacements = if old.is_empty() {
        source
            .chars()
            .count()
            .checked_add(1)
            .ok_or_else(helper_resource_error)?
    } else {
        source.match_indices(old).count()
    };
    let removed = replacements
        .checked_mul(old.len())
        .ok_or_else(helper_resource_error)?;
    let inserted = replacements
        .checked_mul(new.len())
        .ok_or_else(helper_resource_error)?;
    let output_bytes = source
        .len()
        .checked_sub(removed)
        .and_then(|length| length.checked_add(inserted))
        .filter(|length| *length <= maximum)
        .ok_or_else(helper_resource_error)?;
    let mut output = String::new();
    output
        .try_reserve(output_bytes)
        .map_err(|_| helper_resource_error())?;
    if old.is_empty() {
        output.push_str(new);
        for character in source.chars() {
            output.push(character);
            output.push_str(new);
        }
    } else {
        let mut rest = source;
        while let Some(index) = rest.find(old) {
            output.push_str(&rest[..index]);
            output.push_str(new);
            rest = &rest[index + old.len()..];
        }
        output.push_str(rest);
    }
    debug_assert_eq!(output.len(), output_bytes);
    Ok(Value::from(output))
}

fn checked_split<'a>(
    parts: impl Clone + Iterator<Item = &'a str>,
    maximum_bytes: usize,
    maximum_nodes: usize,
) -> Result<Value, minijinja::Error> {
    let mut count = 0usize;
    let mut bytes = 0usize;
    for part in parts.clone() {
        count = count.checked_add(1).ok_or_else(helper_resource_error)?;
        bytes = bytes
            .checked_add(part.len())
            .ok_or_else(helper_resource_error)?;
        if count
            .checked_add(1)
            .is_none_or(|nodes| nodes > maximum_nodes)
            || bytes > maximum_bytes
        {
            return Err(helper_resource_error());
        }
    }
    let mut output = Vec::new();
    output
        .try_reserve(count)
        .map_err(|_| helper_resource_error())?;
    output.extend(parts.map(|part| Value::from(part.to_string())));
    Ok(Value::from(output))
}

fn checked_join(
    separator: &str,
    iterable: &Value,
    maximum_bytes: usize,
    maximum_nodes: usize,
) -> Result<Value, minijinja::Error> {
    let mut item_count = 0usize;
    let mut item_bytes = 0usize;
    for value in iterable.try_iter()? {
        item_count = item_count
            .checked_add(1)
            .filter(|count| *count <= maximum_nodes)
            .ok_or_else(helper_resource_error)?;
        let remaining = maximum_bytes.saturating_sub(item_bytes);
        let part_bytes = value.as_str().map_or_else(
            || counted_display_bytes(&value, remaining),
            |part| Ok(part.len()),
        )?;
        item_bytes = item_bytes
            .checked_add(part_bytes)
            .filter(|length| *length <= maximum_bytes)
            .ok_or_else(helper_resource_error)?;
    }
    let separators = item_count.saturating_sub(1);
    let output_bytes = separator
        .len()
        .checked_mul(separators)
        .and_then(|length| length.checked_add(item_bytes))
        .filter(|length| *length <= maximum_bytes)
        .ok_or_else(helper_resource_error)?;
    let mut output = String::new();
    output
        .try_reserve(output_bytes)
        .map_err(|_| helper_resource_error())?;
    for (index, value) in iterable.try_iter()?.enumerate() {
        if index != 0 {
            output.push_str(separator);
        }
        if let Some(part) = value.as_str() {
            output.push_str(part);
        } else {
            std::fmt::write(&mut output, format_args!("{value}"))
                .map_err(|_| helper_resource_error())?;
        }
    }
    debug_assert_eq!(output.len(), output_bytes);
    Ok(Value::from(output))
}

struct BoundedDisplayCounter {
    bytes: usize,
    maximum: usize,
}

impl std::fmt::Write for BoundedDisplayCounter {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.bytes = self
            .bytes
            .checked_add(value.len())
            .filter(|bytes| *bytes <= self.maximum)
            .ok_or(std::fmt::Error)?;
        Ok(())
    }
}

fn counted_display_bytes(value: &Value, maximum: usize) -> Result<usize, minijinja::Error> {
    let mut counter = BoundedDisplayCounter { bytes: 0, maximum };
    std::fmt::write(&mut counter, format_args!("{value}")).map_err(|_| helper_resource_error())?;
    Ok(counter.bytes)
}

fn derived_sequence_allows(
    current_items: usize,
    additional_nodes_per_item: usize,
    maximum_nodes: usize,
) -> bool {
    current_items
        .checked_add(1)
        .and_then(|items| items.checked_mul(additional_nodes_per_item))
        .and_then(|children| children.checked_add(1))
        .is_some_and(|nodes| nodes <= maximum_nodes)
}

pub(crate) fn configure_neutral_prompt_environment(
    env: &mut Environment<'_>,
    fuel: u64,
    max_intermediate_bytes: usize,
    max_intermediate_nodes: usize,
) {
    env.set_keep_trailing_newline(true);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_fuel(Some(fuel));
    env.add_filter("tojson", move |value, indent, kwargs| {
        super::python_tojson_bounded(
            value,
            indent,
            kwargs,
            max_intermediate_bytes,
            max_intermediate_nodes,
        )
    });

    env.set_unknown_method_callback(move |_state, value, method, args| {
        use minijinja::value::ValueKind;

        // --- String methods ---------------------------------------------
        if value.kind() == ValueKind::String {
            let s = value.as_str().unwrap_or_default();
            let arg_str = || args.first().and_then(|a| a.as_str()).unwrap_or_default();

            // `chars` argument for strip/lstrip/rstrip. `None` → whitespace.
            let strip_chars = args.first().and_then(|a| a.as_str());
            let strip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim().to_string(),
                    Some(chars) => s.trim_matches(|c: char| chars.contains(c)).to_string(),
                }
            };
            let lstrip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim_start().to_string(),
                    Some(chars) => s
                        .trim_start_matches(|c: char| chars.contains(c))
                        .to_string(),
                }
            };
            let rstrip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim_end().to_string(),
                    Some(chars) => s.trim_end_matches(|c: char| chars.contains(c)).to_string(),
                }
            };

            match method {
                "split" => {
                    let sep = arg_str();
                    // Python `s.split()` with no arg splits on any whitespace
                    // and drops empty strings; `s.split(sep)` splits on sep.
                    return if sep.is_empty() {
                        checked_split(
                            s.split_whitespace(),
                            max_intermediate_bytes,
                            max_intermediate_nodes,
                        )
                    } else {
                        checked_split(s.split(sep), max_intermediate_bytes, max_intermediate_nodes)
                    };
                }
                "rsplit" => {
                    let sep = arg_str();
                    return if sep.is_empty() {
                        checked_split(
                            s.split_whitespace(),
                            max_intermediate_bytes,
                            max_intermediate_nodes,
                        )
                    } else {
                        checked_split(
                            s.rsplit(sep),
                            max_intermediate_bytes,
                            max_intermediate_nodes,
                        )
                    };
                }
                "strip" => return Ok(Value::from(strip_matches(s))),
                "lstrip" => return Ok(Value::from(lstrip_matches(s))),
                "rstrip" => return Ok(Value::from(rstrip_matches(s))),
                "startswith" => {
                    let prefix = arg_str();
                    return Ok(Value::from(s.starts_with(prefix)));
                }
                "endswith" => {
                    let suffix = arg_str();
                    return Ok(Value::from(s.ends_with(suffix)));
                }
                "index" | "find" => {
                    // Python `str.index(sub)` / `str.find(sub)`: return the
                    // character index of the first occurrence. `index` raises on
                    // absence; `find` returns -1. Used by the Granite 4 Vision
                    // template's image-position dispatcher.
                    let sub = arg_str();
                    match s.find(sub) {
                        Some(byte_idx) => {
                            let char_idx = s[..byte_idx].chars().count() as i64;
                            return Ok(Value::from(char_idx));
                        }
                        None if method == "find" => return Ok(Value::from(-1i64)),
                        None => {
                            return Err(minijinja::Error::new(
                                ErrorKind::InvalidOperation,
                                "substring not found",
                            ));
                        }
                    }
                }
                "replace" => {
                    let old = args.first().and_then(|a| a.as_str()).unwrap_or_default();
                    let new = args.get(1).and_then(|a| a.as_str()).unwrap_or_default();
                    return checked_replace(s, old, new, max_intermediate_bytes);
                }
                "upper" => return Ok(Value::from(s.to_uppercase())),
                "lower" => return Ok(Value::from(s.to_lowercase())),
                "title" => {
                    // Simple title-case: uppercase first letter of each word.
                    let mut result = String::with_capacity(s.len());
                    let mut prev_alphabetic = false;
                    for ch in s.chars() {
                        if ch.is_alphabetic() {
                            if prev_alphabetic {
                                result.extend(ch.to_lowercase());
                            } else {
                                result.extend(ch.to_uppercase());
                            }
                            prev_alphabetic = true;
                        } else {
                            result.push(ch);
                            prev_alphabetic = false;
                        }
                    }
                    return Ok(Value::from(result));
                }
                "capitalize" => {
                    // Python `s.capitalize()` — uppercase first char, lowercase the rest.
                    let mut chars = s.chars();
                    let result = match chars.next() {
                        Some(first) => {
                            let mut out = String::with_capacity(s.len());
                            out.extend(first.to_uppercase());
                            out.extend(chars.flat_map(|c| c.to_lowercase()));
                            out
                        }
                        None => String::new(),
                    };
                    return Ok(Value::from(result));
                }
                "casefold" => {
                    // Python `s.casefold()` — approximation via lowercase.
                    return Ok(Value::from(s.to_lowercase()));
                }
                "swapcase" => {
                    let result: String = s
                        .chars()
                        .flat_map(|c| {
                            if c.is_uppercase() {
                                c.to_lowercase().collect::<Vec<_>>()
                            } else if c.is_lowercase() {
                                c.to_uppercase().collect::<Vec<_>>()
                            } else {
                                vec![c]
                            }
                        })
                        .collect();
                    return Ok(Value::from(result));
                }
                "join" => {
                    // Python `sep.join(iterable)` — called as method on the
                    // separator string, with the iterable as the argument.
                    let Some(iter_arg) = args.first() else {
                        return Err(minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "str.join() requires an iterable argument",
                        ));
                    };
                    return checked_join(
                        s,
                        iter_arg,
                        max_intermediate_bytes,
                        max_intermediate_nodes,
                    );
                }
                "isdigit" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()),
                    ));
                }
                "isalpha" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_alphabetic()),
                    ));
                }
                "isalnum" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()),
                    ));
                }
                "isspace" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_whitespace()),
                    ));
                }
                "isupper" => {
                    return Ok(Value::from(
                        s.chars().any(|c| c.is_uppercase()) && !s.chars().any(|c| c.is_lowercase()),
                    ));
                }
                "islower" => {
                    return Ok(Value::from(
                        s.chars().any(|c| c.is_lowercase()) && !s.chars().any(|c| c.is_uppercase()),
                    ));
                }
                "count" => {
                    let needle = arg_str();
                    if needle.is_empty() {
                        return Ok(Value::from(0_i64));
                    }
                    return Ok(Value::from(s.matches(needle).count() as i64));
                }
                _ => {}
            }
        }

        // --- Map (dict) methods -----------------------------------------
        if value.kind() == ValueKind::Map {
            match method {
                // Python-style `dict.get(key[, default])`: returns the
                // value at `key` or `default`, or Python `None` when the
                // key is absent and no explicit default was supplied.
                // `None` remains falsy, so `m.get('a') or m.get('b')`
                // chains keep working while `is none` / `is defined`
                // now match CPython Jinja semantics for missing keys.
                "get" => {
                    let Some(key) = args.first() else {
                        return Err(minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "dict.get() requires at least one argument",
                        ));
                    };
                    let default = args.get(1).cloned().unwrap_or_else(|| Value::from(()));
                    return match value.get_item(key) {
                        Ok(v) if v.is_undefined() => Ok(default),
                        Ok(v) => Ok(v),
                        Err(_) => Ok(default),
                    };
                }
                // Python-style `dict.items()` / `keys()` / `values()` —
                // return flat sequences usable directly in `{% for %}`.
                "items" => {
                    let mut pairs: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            if !derived_sequence_allows(pairs.len(), 3, max_intermediate_nodes) {
                                return Err(helper_resource_error());
                            }
                            let v = value.get_item(&k).unwrap_or(Value::UNDEFINED);
                            pairs.push(Value::from(vec![k, v]));
                        }
                    }
                    return Ok(Value::from(pairs));
                }
                "keys" => {
                    let mut keys: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            if !derived_sequence_allows(keys.len(), 1, max_intermediate_nodes) {
                                return Err(helper_resource_error());
                            }
                            keys.push(k);
                        }
                    }
                    return Ok(Value::from(keys));
                }
                "values" => {
                    let mut vals: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            if let Ok(v) = value.get_item(&k) {
                                if !derived_sequence_allows(vals.len(), 1, max_intermediate_nodes) {
                                    return Err(helper_resource_error());
                                }
                                vals.push(v);
                            }
                        }
                    }
                    return Ok(Value::from(vals));
                }
                _ => {}
            }
        }

        Err(minijinja::Error::from(ErrorKind::UnknownMethod))
    });
}

fn neutral_environment(
    fuel: u64,
    max_intermediate_bytes: usize,
    max_intermediate_nodes: usize,
) -> Environment<'static> {
    let mut environment = Environment::new();
    configure_neutral_prompt_environment(
        &mut environment,
        fuel,
        max_intermediate_bytes,
        max_intermediate_nodes,
    );
    environment
}

fn template_error(error: minijinja::Error) -> NativeRuntimeError {
    let kind = if error.kind() == minijinja::ErrorKind::OutOfFuel
        || is_helper_resource_limit(&error)
        || super::prompt_json::is_prompt_json_resource_limit(&error)
    {
        NativeRuntimeErrorKind::ResourceLimit
    } else {
        NativeRuntimeErrorKind::InvalidRequest
    };
    prompt_error(kind, format!("prompt template rejected input: {error}"))
}

fn validate_context(
    root: &NativeTemplateValue,
    limits: NativePromptRendererLimits,
) -> Result<(), NativeRuntimeError> {
    let mut pending = vec![(root, 1usize)];
    let mut nodes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes = nodes.checked_add(1).ok_or_else(|| {
            prompt_error(
                NativeRuntimeErrorKind::ResourceLimit,
                "prompt context node count overflowed",
            )
        })?;
        if nodes > limits.max_context_nodes || depth > limits.max_context_depth {
            return Err(prompt_error(
                NativeRuntimeErrorKind::ResourceLimit,
                "prompt context exceeds its node or depth limit",
            ));
        }
        match value {
            NativeTemplateValue::Sequence(values) => {
                validate_context_children(nodes, pending.len(), depth, values.len(), limits)?;
                pending.extend(values.iter().rev().map(|value| (value, depth + 1)));
            }
            NativeTemplateValue::Object(entries) => {
                validate_context_children(nodes, pending.len(), depth, entries.len(), limits)?;
                let mut keys = HashSet::with_capacity(entries.len().min(1024));
                for (key, value) in entries.iter().rev() {
                    if !keys.insert(key.as_ref()) {
                        return Err(prompt_error(
                            NativeRuntimeErrorKind::InvalidRequest,
                            "prompt context object contains a duplicate key",
                        ));
                    }
                    pending.push((value, depth + 1));
                }
            }
            _ => {}
        }
    }
    let mut counter = BoundedCounter::new(limits.max_context_bytes);
    serde_json::to_writer(&mut counter, root).map_err(|_| {
        let message = if counter.exceeded {
            "prompt context exceeds its canonical byte limit"
        } else {
            "prompt context could not be serialized deterministically"
        };
        prompt_error(
            if counter.exceeded {
                NativeRuntimeErrorKind::ResourceLimit
            } else {
                NativeRuntimeErrorKind::InvalidRequest
            },
            message,
        )
    })?;
    Ok(())
}

fn validate_context_children(
    visited_nodes: usize,
    pending_nodes: usize,
    parent_depth: usize,
    child_count: usize,
    limits: NativePromptRendererLimits,
) -> Result<(), NativeRuntimeError> {
    if child_count != 0 && parent_depth >= limits.max_context_depth {
        return Err(prompt_error(
            NativeRuntimeErrorKind::ResourceLimit,
            "prompt context exceeds its depth limit",
        ));
    }
    let retained_nodes = visited_nodes
        .checked_add(pending_nodes)
        .and_then(|count| count.checked_add(child_count))
        .ok_or_else(|| {
            prompt_error(
                NativeRuntimeErrorKind::ResourceLimit,
                "prompt context node count overflowed",
            )
        })?;
    if retained_nodes > limits.max_context_nodes {
        return Err(prompt_error(
            NativeRuntimeErrorKind::ResourceLimit,
            "prompt context exceeds its node limit",
        ));
    }
    Ok(())
}

struct BoundedCounter {
    length: usize,
    maximum: usize,
    exceeded: bool,
}

impl BoundedCounter {
    fn new(maximum: usize) -> Self {
        Self {
            length: 0,
            maximum,
            exceeded: false,
        }
    }
}

impl Write for BoundedCounter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(next) = self.length.checked_add(buffer.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("bounded prompt counter exceeded"));
        };
        if next > self.maximum {
            self.exceeded = true;
            return Err(io::Error::other("bounded prompt counter exceeded"));
        }
        self.length = next;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    maximum: usize,
    exceeded: bool,
}

impl BoundedWriter {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            exceeded: false,
        }
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let next = self.bytes.len().checked_add(buffer.len());
        if next.is_none_or(|next| next > self.maximum) {
            self.exceeded = true;
            return Err(io::Error::other("bounded prompt writer exceeded"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn binding_identity(
    binding: &NativePromptRendererBinding,
    kind: NativePromptRendererKind,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"mlxcel.native-prompt-binding.v1");
    hash_field(&mut hasher, &binding.package_identity);
    hash_field(&mut hasher, &binding.report_identity);
    hash_field(&mut hasher, &binding.tokenizer_identity);
    hasher.update([match kind {
        NativePromptRendererKind::LiteralUtf8 => 0,
        NativePromptRendererKind::ImmutableJinja => 1,
    }]);
    if let Some(component) = &binding.template_component {
        hasher.update([1]);
        hash_component(&mut hasher, component);
    } else {
        hasher.update([0]);
    }
    hash_component(&mut hasher, &binding.tokenizer_config_component);
    hash_field(&mut hasher, binding.renderer_schema.as_bytes());
    hash_limits(&mut hasher, binding.limits);
    hasher.update([u8::from(binding.add_special_tokens)]);
    hasher.finalize().into()
}

fn hash_component(hasher: &mut Sha256, component: &NativePackageComponent) {
    hash_field(hasher, component.label().as_bytes());
    hasher.update(component.size_bytes().to_le_bytes());
    hash_field(hasher, &component.sha256());
}

fn hash_limits(hasher: &mut Sha256, limits: NativePromptRendererLimits) {
    for value in [
        limits.max_template_bytes,
        limits.max_context_bytes,
        limits.max_context_depth,
        limits.max_context_nodes,
        limits.max_output_bytes,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
    hasher.update(limits.fuel.to_le_bytes());
}

fn rendered_prompt_identity(binding: &NativePromptRendererBinding, rendered: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"mlxcel.native-rendered-prompt.v1");
    hash_field(&mut hasher, &binding.identity);
    hash_field(&mut hasher, rendered);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::native_runtime::{
        NativeImmutablePackage, NativePackageArtifactSource, NativePackageComponent,
        NativePackageLimits,
    };

    struct MemorySource(BTreeMap<String, Vec<u8>>);

    impl NativePackageArtifactSource for MemorySource {
        fn open_artifact(
            &mut self,
            label: &str,
        ) -> Result<Box<dyn std::io::Read + Send>, crate::native_runtime::NativeRuntimeError>
        {
            Ok(Box::new(Cursor::new(
                self.0.get(label).expect("declared test component").clone(),
            )))
        }
    }

    struct DirectorySource(PathBuf);

    impl NativePackageArtifactSource for DirectorySource {
        fn open_artifact(
            &mut self,
            label: &str,
        ) -> Result<Box<dyn std::io::Read + Send>, crate::native_runtime::NativeRuntimeError>
        {
            File::open(self.0.join(label))
                .map(|file| Box::new(file) as Box<dyn std::io::Read + Send>)
                .map_err(|error| {
                    crate::native_runtime::NativeRuntimeError::new(
                        crate::native_runtime::NativeRuntimeErrorKind::InvalidPackage,
                        format!("qualification component {label} could not be opened: {error}"),
                    )
                })
        }
    }

    fn component(label: &str, bytes: &[u8]) -> NativePackageComponent {
        NativePackageComponent::new(label, bytes.len() as u64, Sha256::digest(bytes).into())
            .unwrap()
    }

    fn package(
        template: &[u8],
        tokenizer_config: &[u8],
    ) -> (
        NativeImmutablePackage,
        NativePackageComponent,
        NativePackageComponent,
    ) {
        let config = br#"{"model_type":"llama","max_position_embeddings":64}"#;
        let crate::tokenizer::MlxcelTokenizer::HuggingFace(tokenizer) =
            crate::tokenizer::MlxcelTokenizer::stub_gemma_sequence_byte_fallback()
        else {
            unreachable!()
        };
        let tokenizer = tokenizer.to_string(false).unwrap().into_bytes();
        let template_component = component("chat_template.jinja", template);
        let tokenizer_config_component = component("tokenizer_config.json", tokenizer_config);
        let mut bytes = BTreeMap::from([
            ("config.json".to_string(), config.to_vec()),
            ("tokenizer.json".to_string(), tokenizer.clone()),
            ("chat_template.jinja".to_string(), template.to_vec()),
            (
                "tokenizer_config.json".to_string(),
                tokenizer_config.to_vec(),
            ),
        ]);
        let declarations = bytes
            .iter()
            .map(|(label, bytes)| component(label, bytes))
            .collect();
        let package = NativeImmutablePackage::acquire(
            &mut MemorySource(std::mem::take(&mut bytes)),
            declarations,
            NativePackageLimits::default(),
        )
        .unwrap();
        (package, template_component, tokenizer_config_component)
    }

    fn file_component(root: &Path, label: &str) -> NativePackageComponent {
        let bytes = std::fs::read(root.join(label)).expect("qualification component");
        component(label, &bytes)
    }

    fn digest_hex(digest: [u8; 32]) -> String {
        use std::fmt::Write as _;

        let mut output = String::with_capacity(64);
        for byte in digest {
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output
    }

    fn value_from_json(value: &serde_json::Value) -> NativeTemplateValue {
        match value {
            serde_json::Value::Null => NativeTemplateValue::Null,
            serde_json::Value::Bool(value) => NativeTemplateValue::Bool(*value),
            serde_json::Value::Number(value) => value
                .as_i64()
                .map(NativeTemplateValue::I64)
                .or_else(|| value.as_u64().map(NativeTemplateValue::U64))
                .expect("integer qualification context"),
            serde_json::Value::String(value) => NativeTemplateValue::string(value),
            serde_json::Value::Array(values) => {
                NativeTemplateValue::sequence(values.iter().map(value_from_json))
            }
            serde_json::Value::Object(entries) => NativeTemplateValue::object(
                entries
                    .iter()
                    .map(|(key, value)| (key.as_str(), value_from_json(value))),
            )
            .expect("unique JSON object keys"),
        }
    }

    fn spec(
        template: NativePackageComponent,
        tokenizer_config: NativePackageComponent,
    ) -> NativePromptRendererSpec {
        NativePromptRendererSpec {
            kind: NativePromptRendererKind::ImmutableJinja,
            template_component: Some(template),
            tokenizer_config_component: tokenizer_config,
            limits: NativePromptRendererLimits::default(),
            add_special_tokens: false,
        }
    }

    #[test]
    fn immutable_jinja_is_deterministic_and_receipt_is_exactly_bound() {
        let template = b"{{ messages[0].role }}={{ messages[0].content }}";
        let tokenizer_config = br#"{"bos_token":"<s>"}"#;
        let (package, template_component, tokenizer_config_component) =
            package(template, tokenizer_config);
        let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
        let renderer = NativePromptRenderer::bind(
            &package,
            &report,
            spec(template_component, tokenizer_config_component),
        )
        .unwrap();
        let context = NativeTemplateValue::object([(
            "messages",
            NativeTemplateValue::sequence([NativeTemplateValue::object([
                ("role", NativeTemplateValue::string("user")),
                ("content", NativeTemplateValue::string("hello")),
            ])
            .unwrap()]),
        )])
        .unwrap();

        let first = renderer.render(&context).unwrap();
        let second = renderer.render(&context).unwrap();
        assert_eq!(first.rendered_utf8(), "user=hello");
        assert_eq!(first.output_identity(), second.output_identity());
        assert_eq!(first.binding_identity(), renderer.binding_identity());
        assert_eq!(first.package_identity(), package.identity());
        assert_eq!(first.report_identity(), report.report_identity());
        assert!(!first.add_special_tokens());
    }

    #[test]
    fn component_digest_and_label_mismatches_fail_closed() {
        let template = b"{{ value }}";
        let tokenizer_config = b"{}";
        let (bound_package, template_component, tokenizer_config_component) =
            package(template, tokenizer_config);
        let report = crate::native_runtime::model_report::inspect_package(&bound_package).unwrap();

        let wrong_digest = NativePackageComponent::new(
            template_component.label(),
            template_component.size_bytes(),
            [7; 32],
        )
        .unwrap();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &report,
                spec(wrong_digest, tokenizer_config_component.clone()),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );

        let wrong_existing_config = bound_package
            .component("config.json")
            .expect("existing non-tokenizer config component")
            .clone();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &report,
                spec(template_component.clone(), wrong_existing_config),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );

        let wrong_label = NativePackageComponent::new(
            "other.jinja",
            template_component.size_bytes(),
            template_component.sha256(),
        )
        .unwrap();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &report,
                spec(wrong_label, tokenizer_config_component.clone()),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidPackage
        );

        let wrong_config_digest = NativePackageComponent::new(
            tokenizer_config_component.label(),
            tokenizer_config_component.size_bytes(),
            [8; 32],
        )
        .unwrap();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &report,
                spec(template_component.clone(), wrong_config_digest),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );

        let wrong_config_label = NativePackageComponent::new(
            "other-tokenizer-config.json",
            tokenizer_config_component.size_bytes(),
            tokenizer_config_component.sha256(),
        )
        .unwrap();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &report,
                spec(template_component.clone(), wrong_config_label),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidPackage
        );

        let (other_package, _, _) = package(b"{{ other }}", tokenizer_config);
        let other_report =
            crate::native_runtime::model_report::inspect_package(&other_package).unwrap();
        assert_eq!(
            NativePromptRenderer::bind(
                &bound_package,
                &other_report,
                spec(template_component, tokenizer_config_component),
            )
            .unwrap_err()
            .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );
    }

    #[test]
    fn forbidden_resolution_and_time_helpers_are_rejected_at_bind() {
        for source in [
            "{% include 'other' %}",
            "{% import 'other' as x %}",
            "{% from 'other' import x %}",
            "{% extends 'other' %}",
            "{{ strftime_now('%Y') }}",
        ] {
            let (package, template, tokenizer_config) = package(source.as_bytes(), b"{}");
            let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
            assert_eq!(
                NativePromptRenderer::bind(&package, &report, spec(template, tokenizer_config),)
                    .unwrap_err()
                    .kind(),
                crate::native_runtime::NativeRuntimeErrorKind::UnsupportedGuarantee,
                "source={source}"
            );
        }
    }

    #[test]
    fn forbidden_block_before_a_later_comment_is_not_skipped() {
        let source = "{% include 'other' %}{# harmless trailing comment #}";
        let (package, template, tokenizer_config) = package(source.as_bytes(), b"{}");
        let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
        assert_eq!(
            NativePromptRenderer::bind(&package, &report, spec(template, tokenizer_config),)
                .unwrap_err()
                .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::UnsupportedGuarantee
        );
    }

    #[test]
    fn server_wrapper_builds_on_the_neutral_renderer_environment() {
        let server = include_str!("../server/chat_template.rs");
        let configure = server
            .split_once("fn configure_environment")
            .expect("server environment configuration")
            .1
            .split_once("/// Extract a token")
            .expect("server environment configuration end")
            .0;
        assert!(configure.contains("configure_neutral_prompt_environment("));
        assert!(!configure.contains("set_unknown_method_callback"));
        assert!(!configure.contains("add_filter(\"tojson\""));
    }

    #[test]
    fn native_prompt_renderer_has_no_server_dependency() {
        let production = include_str!("prompt_renderer.rs")
            .split_once("#[cfg(test)]")
            .expect("prompt renderer test boundary")
            .0;
        assert!(!production.contains("crate::server"));
    }

    #[test]
    fn helper_intermediates_honor_selected_byte_and_node_bounds() {
        let cases = [
            (
                "replace exact",
                "{% set ignored = value.replace('a', '1234') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("aa"))])
                    .unwrap(),
                8,
                8,
                true,
            ),
            (
                "replace plus one",
                "{% set ignored = value.replace('a', '12345') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("aa"))])
                    .unwrap(),
                8,
                8,
                false,
            ),
            (
                "join exact",
                "{% set ignored = sep.join(values) %}ok",
                NativeTemplateValue::object([
                    ("sep", NativeTemplateValue::string("12")),
                    (
                        "values",
                        NativeTemplateValue::sequence([
                            NativeTemplateValue::string("a"),
                            NativeTemplateValue::string("b"),
                            NativeTemplateValue::string("c"),
                        ]),
                    ),
                ])
                .unwrap(),
                7,
                8,
                true,
            ),
            (
                "join plus one",
                "{% set ignored = sep.join(values) %}ok",
                NativeTemplateValue::object([
                    ("sep", NativeTemplateValue::string("123")),
                    (
                        "values",
                        NativeTemplateValue::sequence([
                            NativeTemplateValue::string("a"),
                            NativeTemplateValue::string("b"),
                            NativeTemplateValue::string("c"),
                        ]),
                    ),
                ])
                .unwrap(),
                8,
                8,
                false,
            ),
            (
                "split exact",
                "{% set ignored = value.split(',') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("a,b,c"))])
                    .unwrap(),
                8,
                4,
                true,
            ),
            (
                "split plus one",
                "{% set ignored = value.split(',') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("a,b,c,d"))])
                    .unwrap(),
                8,
                4,
                false,
            ),
            (
                "rsplit exact",
                "{% set ignored = value.rsplit(',') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("a,b,c"))])
                    .unwrap(),
                8,
                4,
                true,
            ),
            (
                "rsplit plus one",
                "{% set ignored = value.rsplit(',') %}ok",
                NativeTemplateValue::object([("value", NativeTemplateValue::string("a,b,c,d"))])
                    .unwrap(),
                8,
                4,
                false,
            ),
        ];

        for (name, template, context, max_output_bytes, max_context_nodes, succeeds) in cases {
            let (package, template_component, tokenizer_config_component) =
                package(template.as_bytes(), b"{}");
            let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
            let renderer = NativePromptRenderer::bind(
                &package,
                &report,
                NativePromptRendererSpec {
                    kind: NativePromptRendererKind::ImmutableJinja,
                    template_component: Some(template_component),
                    tokenizer_config_component,
                    limits: NativePromptRendererLimits::try_new(
                        template.len(),
                        256,
                        8,
                        max_context_nodes,
                        max_output_bytes,
                        10_000,
                    )
                    .unwrap(),
                    add_special_tokens: false,
                },
            )
            .unwrap();
            let result = renderer.render(&context);
            if succeeds {
                assert_eq!(result.unwrap().rendered_utf8(), "ok", "case={name}");
            } else {
                match result {
                    Err(error) => {
                        assert_eq!(
                            error.kind(),
                            crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit,
                            "case={name}"
                        );
                    }
                    Ok(receipt) => panic!("case={name} unexpectedly rendered {receipt:?}"),
                }
            }
        }
    }

    #[test]
    #[ignore = "requires MLXCEL_GEMMA4_QUALIFICATION_MODEL_DIR"]
    fn real_gemma4_native_renderer_matches_server_bytes_across_semantic_contexts() {
        let root = PathBuf::from(
            std::env::var_os("MLXCEL_GEMMA4_QUALIFICATION_MODEL_DIR")
                .expect("MLXCEL_GEMMA4_QUALIFICATION_MODEL_DIR"),
        );
        let labels = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ];
        let declarations = labels
            .iter()
            .map(|label| file_component(&root, label))
            .collect::<Vec<_>>();
        for (label, size, digest) in [
            (
                "config.json",
                5_415,
                "fbc1c1cb48ed86ec98482b2d41f5a03d3991aba74b7c29a93d430761e6518a38",
            ),
            (
                "tokenizer.json",
                32_169_626,
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            ),
            (
                "tokenizer_config.json",
                2_719,
                "fc1384a911d2c9860ac07bc3ceafff20bff26695991744b7dbe5e1e4522bfa57",
            ),
            (
                "chat_template.jinja",
                17_466,
                "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
            ),
        ] {
            let component = declarations
                .iter()
                .find(|component| component.label() == label)
                .unwrap();
            assert_eq!(component.size_bytes(), size, "component={label}");
            assert_eq!(digest_hex(component.sha256()), digest, "component={label}");
        }
        let template_component = declarations
            .iter()
            .find(|component| component.label() == "chat_template.jinja")
            .unwrap()
            .clone();
        let tokenizer_config_component = declarations
            .iter()
            .find(|component| component.label() == "tokenizer_config.json")
            .unwrap()
            .clone();
        let package = NativeImmutablePackage::acquire(
            &mut DirectorySource(root.clone()),
            declarations,
            NativePackageLimits::default(),
        )
        .unwrap();
        let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
        let renderer = NativePromptRenderer::bind(
            &package,
            &report,
            spec(template_component, tokenizer_config_component),
        )
        .unwrap();
        let template = std::fs::read_to_string(root.join("chat_template.jinja")).unwrap();

        let tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Read weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City"}},
                    "required": ["city"]
                }
            }
        });
        let scenarios = [
            (
                "completion",
                serde_json::json!([{"role": "user", "content": "complete this"}]),
                serde_json::json!([]),
                false,
                false,
            ),
            (
                "chat",
                serde_json::json!([{"role": "user", "content": "hello"}]),
                serde_json::json!([]),
                false,
                true,
            ),
            (
                "system",
                serde_json::json!([
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "hello"}
                ]),
                serde_json::json!([]),
                false,
                true,
            ),
            (
                "developer",
                serde_json::json!([
                    {"role": "developer", "content": "Use metric units"},
                    {"role": "user", "content": "weather"}
                ]),
                serde_json::json!([]),
                false,
                true,
            ),
            (
                "tools",
                serde_json::json!([{"role": "user", "content": "weather in Seoul"}]),
                serde_json::json!([tool.clone()]),
                false,
                true,
            ),
            (
                "tool_history_and_result",
                serde_json::json!([
                    {"role": "user", "content": "weather in Seoul"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call-1",
                            "function": {"name": "weather", "arguments": {"city": "Seoul"}}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": {"celsius": 24}}
                ]),
                serde_json::json!([tool.clone()]),
                false,
                true,
            ),
            (
                "reasoning_off",
                serde_json::json!([{"role": "user", "content": "think"}]),
                serde_json::json!([]),
                false,
                true,
            ),
            (
                "reasoning_on",
                serde_json::json!([{"role": "user", "content": "think"}]),
                serde_json::json!([]),
                true,
                true,
            ),
        ];

        for (name, messages, tools, enable_thinking, add_generation_prompt) in scenarios {
            let tools_typed: Vec<crate::server::types::request::Tool> =
                serde_json::from_value(tools.clone()).unwrap();
            let kwargs = crate::server::chat_template_kwargs::ChatTemplateKwargs::from_json_str(
                &format!(r#"{{"enable_thinking":{enable_thinking}}}"#),
            )
            .unwrap();
            let mut server = crate::server::chat_template::ChatTemplateProcessor::with_template(
                template.clone(),
            );
            server.set_add_generation_prompt(add_generation_prompt);
            let expected = server
                .apply_raw_with_kwargs(&messages, Some(&tools_typed), &kwargs)
                .unwrap();
            let context = NativeTemplateValue::object([
                ("messages", value_from_json(&messages)),
                ("bos_token", NativeTemplateValue::string("")),
                ("eos_token", NativeTemplateValue::string("")),
                (
                    "add_generation_prompt",
                    NativeTemplateValue::Bool(add_generation_prompt),
                ),
                ("tools", value_from_json(&tools)),
                (
                    "enable_thinking",
                    NativeTemplateValue::Bool(enable_thinking),
                ),
            ])
            .unwrap();
            let receipt = renderer.render(&context).unwrap();
            assert_eq!(receipt.rendered_utf8(), expected, "scenario={name}");
            assert_eq!(
                receipt.output_identity(),
                renderer.render(&context).unwrap().output_identity(),
                "scenario={name}"
            );
        }
    }

    #[test]
    fn context_duplicate_depth_node_and_byte_bounds_are_checked() {
        assert!(
            NativeTemplateValue::object([
                ("same", NativeTemplateValue::Null),
                ("same", NativeTemplateValue::Bool(true)),
            ])
            .is_err()
        );

        let exact =
            NativePromptRendererLimits::try_new(MAX_NATIVE_PROMPT_TEMPLATE_BYTES, 5, 2, 3, 8, 100)
                .unwrap();
        let over = NativePromptRendererLimits::try_new(
            MAX_NATIVE_PROMPT_TEMPLATE_BYTES + 1,
            5,
            2,
            3,
            8,
            100,
        )
        .unwrap_err();
        assert_eq!(
            over.kind(),
            crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit
        );
        assert_eq!(exact.max_template_bytes(), MAX_NATIVE_PROMPT_TEMPLATE_BYTES);
    }

    #[test]
    fn every_renderer_hard_ceiling_accepts_the_edge_and_rejects_plus_one() {
        let exact = NativePromptRendererLimits::try_new(
            MAX_NATIVE_PROMPT_TEMPLATE_BYTES,
            MAX_NATIVE_PROMPT_CONTEXT_BYTES,
            MAX_NATIVE_PROMPT_CONTEXT_DEPTH,
            MAX_NATIVE_PROMPT_CONTEXT_NODES,
            MAX_NATIVE_RENDERED_PROMPT_BYTES,
            MAX_NATIVE_PROMPT_RENDER_FUEL,
        )
        .unwrap();
        assert_eq!(exact, NativePromptRendererLimits::default());

        let candidates = [
            NativePromptRendererLimits::try_new(
                MAX_NATIVE_PROMPT_TEMPLATE_BYTES + 1,
                1,
                1,
                1,
                1,
                1,
            ),
            NativePromptRendererLimits::try_new(1, MAX_NATIVE_PROMPT_CONTEXT_BYTES + 1, 1, 1, 1, 1),
            NativePromptRendererLimits::try_new(1, 1, MAX_NATIVE_PROMPT_CONTEXT_DEPTH + 1, 1, 1, 1),
            NativePromptRendererLimits::try_new(1, 1, 1, MAX_NATIVE_PROMPT_CONTEXT_NODES + 1, 1, 1),
            NativePromptRendererLimits::try_new(
                1,
                1,
                1,
                1,
                MAX_NATIVE_RENDERED_PROMPT_BYTES + 1,
                1,
            ),
            NativePromptRendererLimits::try_new(1, 1, 1, 1, 1, MAX_NATIVE_PROMPT_RENDER_FUEL + 1),
        ];
        for candidate in candidates {
            assert_eq!(
                candidate.unwrap_err().kind(),
                crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit
            );
        }
    }

    #[test]
    fn render_checks_template_context_depth_node_output_and_fuel_edges() {
        let template = b"ok";
        let (package, template_component, tokenizer_config_component) = package(template, b"{}");
        let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
        let renderer = |limits| {
            NativePromptRenderer::bind(
                &package,
                &report,
                NativePromptRendererSpec {
                    kind: NativePromptRendererKind::ImmutableJinja,
                    template_component: Some(template_component.clone()),
                    tokenizer_config_component: tokenizer_config_component.clone(),
                    limits,
                    add_special_tokens: false,
                },
            )
        };
        let context =
            NativeTemplateValue::sequence([NativeTemplateValue::Null, NativeTemplateValue::Null]);
        let exact = NativePromptRendererLimits::try_new(2, 11, 2, 3, 2, 100).unwrap();
        assert_eq!(
            renderer(exact)
                .unwrap()
                .render(&context)
                .unwrap()
                .rendered_utf8(),
            "ok"
        );

        assert_eq!(
            renderer(NativePromptRendererLimits::try_new(1, 11, 2, 3, 2, 100).unwrap())
                .unwrap_err()
                .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit
        );
        for limits in [
            NativePromptRendererLimits::try_new(2, 10, 2, 3, 2, 100).unwrap(),
            NativePromptRendererLimits::try_new(2, 11, 1, 3, 2, 100).unwrap(),
            NativePromptRendererLimits::try_new(2, 11, 2, 2, 2, 100).unwrap(),
            NativePromptRendererLimits::try_new(2, 11, 2, 3, 1, 100).unwrap(),
            NativePromptRendererLimits::try_new(2, 11, 2, 3, 2, 1).unwrap(),
        ] {
            assert_eq!(
                renderer(limits)
                    .unwrap()
                    .render(&context)
                    .unwrap_err()
                    .kind(),
                crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit,
                "limits={limits:?}"
            );
        }
    }

    #[test]
    fn literal_renderer_preserves_utf8_and_output_bound() {
        let tokenizer_config = b"{}";
        let (package, _, tokenizer_config_component) = package(b"ignored", tokenizer_config);
        let report = crate::native_runtime::model_report::inspect_package(&package).unwrap();
        let renderer = NativePromptRenderer::bind(
            &package,
            &report,
            NativePromptRendererSpec {
                kind: NativePromptRendererKind::LiteralUtf8,
                template_component: None,
                tokenizer_config_component,
                limits: NativePromptRendererLimits::try_new(1, 16, 4, 8, 5, 100).unwrap(),
                add_special_tokens: true,
            },
        )
        .unwrap();
        assert_eq!(
            renderer
                .render(&NativeTemplateValue::string("éé"))
                .unwrap()
                .rendered_utf8(),
            "éé"
        );
        let error = renderer
            .render(&NativeTemplateValue::String(Arc::from("ééé")))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            crate::native_runtime::NativeRuntimeErrorKind::ResourceLimit
        );
    }

    #[test]
    fn rendered_prompt_defers_total_token_validation_until_realization() {
        use crate::native_runtime::{
            NativeCacheContext, NativeConstraintPlan, NativeGenerateRequest,
            NativeGenerationLimits, NativePromptInput, NativeSamplingConfig, NativeSchedulingHints,
            NativeSpeculationPolicy, NativeSpeculationPreference, NativeSpeculativeFidelity,
        };

        let (bound_package, _, tokenizer_config_component) = package(b"ignored", b"{}");
        let report = crate::native_runtime::model_report::inspect_package(&bound_package).unwrap();
        let rendered = NativePromptRenderer::bind(
            &bound_package,
            &report,
            NativePromptRendererSpec {
                kind: NativePromptRendererKind::LiteralUtf8,
                template_component: None,
                tokenizer_config_component,
                limits: NativePromptRendererLimits::default(),
                add_special_tokens: false,
            },
        )
        .unwrap()
        .render(&NativeTemplateValue::string("hello"))
        .unwrap();
        let request = NativeGenerateRequest {
            request_id: 7,
            prompt: NativePromptInput::Rendered(rendered),
            sampling: NativeSamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: 0,
            },
            limits: NativeGenerationLimits {
                max_new_tokens: 1,
                max_total_tokens: 2,
            },
            eos_token_ids: Arc::from([]),
            output_suppressed_token_ids: Arc::from([]),
            scheduling: NativeSchedulingHints {
                priority: 0,
                deadline_after_ms: None,
            },
            cache: NativeCacheContext {
                namespace: Arc::from("test"),
                input_identity: [0; 32],
                allow_read: false,
                allow_write: false,
            },
            constraint: NativeConstraintPlan::None,
            speculation: NativeSpeculationPolicy {
                preference: NativeSpeculationPreference::Disabled,
                minimum_fidelity: NativeSpeculativeFidelity::ExactTargetDistribution,
                allow_ordinary_fallback: true,
            },
        };
        assert!(request.validate().is_ok());
        assert!(request.validate_resolved_prompt_tokens(1).is_ok());
        assert_eq!(
            request
                .validate_resolved_prompt_tokens(2)
                .unwrap_err()
                .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );
    }

    #[test]
    fn worker_authority_realizes_exact_tokenizer_ids_and_rejects_cross_package_receipts() {
        let (bound_package, _, tokenizer_config_component) = package(b"ignored", b"{}");
        let report = crate::native_runtime::model_report::inspect_package(&bound_package).unwrap();
        let renderer = NativePromptRenderer::bind(
            &bound_package,
            &report,
            NativePromptRendererSpec {
                kind: NativePromptRendererKind::LiteralUtf8,
                template_component: None,
                tokenizer_config_component,
                limits: NativePromptRendererLimits::default(),
                add_special_tokens: true,
            },
        )
        .unwrap();
        let receipt = renderer
            .render(&NativeTemplateValue::string("hello"))
            .unwrap();
        let authority = bound_package.bind_prompt_tokenizer(&report).unwrap();
        let expected = authority
            .clone()
            .into_parts()
            .0
            .encode("hello", true)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        assert_eq!(
            authority.resolve_rendered_prompt(&receipt).unwrap(),
            expected
        );

        let (other_package, _, other_tokenizer_config) = package(b"different", b"{}");
        let other_report =
            crate::native_runtime::model_report::inspect_package(&other_package).unwrap();
        let other_receipt = NativePromptRenderer::bind(
            &other_package,
            &other_report,
            NativePromptRendererSpec {
                kind: NativePromptRendererKind::LiteralUtf8,
                template_component: None,
                tokenizer_config_component: other_tokenizer_config,
                limits: NativePromptRendererLimits::default(),
                add_special_tokens: false,
            },
        )
        .unwrap()
        .render(&NativeTemplateValue::string("hello"))
        .unwrap();
        assert_eq!(
            authority
                .resolve_rendered_prompt(&other_receipt)
                .unwrap_err()
                .kind(),
            crate::native_runtime::NativeRuntimeErrorKind::InvalidRequest
        );
    }
}
