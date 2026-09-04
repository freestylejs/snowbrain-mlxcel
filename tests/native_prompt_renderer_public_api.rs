// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

use mlxcel::native_runtime::{
    NativeImmutablePackage, NativeModelReport, NativePromptInput, NativePromptRenderer,
    NativePromptRendererKind, NativePromptRendererLimits, NativePromptRendererSpec,
    NativeRenderedPrompt, NativeRuntimeBuilder, NativeRuntimeError, NativeTemplateValue,
};

fn bind_from_public_consumer(
    builder: &NativeRuntimeBuilder,
    package: &NativeImmutablePackage,
    report: &NativeModelReport,
    spec: NativePromptRendererSpec,
) -> Result<NativePromptRenderer, NativeRuntimeError> {
    builder.bind_prompt_renderer(package, report, spec)
}

fn inspect_receipt_from_public_consumer(receipt: &NativeRenderedPrompt) {
    let _ = receipt.rendered_utf8();
    let _ = receipt.binding();
    let _ = receipt.binding_identity();
    let _ = receipt.package_identity();
    let _ = receipt.report_identity();
    let _ = receipt.tokenizer_identity();
    let _ = receipt.output_identity();
    let _ = receipt.add_special_tokens();
}

#[test]
fn public_prompt_surface_is_transport_neutral_and_receipt_is_noncloneable() {
    static_assertions::assert_not_impl_any!(NativeRenderedPrompt: Clone, Default);
    let context = NativeTemplateValue::object([
        (
            "messages",
            NativeTemplateValue::sequence([NativeTemplateValue::object([
                ("role", NativeTemplateValue::string("user")),
                ("content", NativeTemplateValue::string("hello")),
            ])
            .unwrap()]),
        ),
        ("enable_thinking", NativeTemplateValue::Bool(false)),
        ("add_generation_prompt", NativeTemplateValue::Bool(true)),
    ])
    .unwrap();
    assert!(matches!(context, NativeTemplateValue::Object(_)));
    assert_eq!(
        NativePromptRendererLimits::default().max_context_depth(),
        64
    );

    let _bind: fn(
        &NativeRuntimeBuilder,
        &NativeImmutablePackage,
        &NativeModelReport,
        NativePromptRendererSpec,
    ) -> Result<NativePromptRenderer, NativeRuntimeError> = bind_from_public_consumer;
    let _inspect: fn(&NativeRenderedPrompt) = inspect_receipt_from_public_consumer;
    let _kind = NativePromptRendererKind::LiteralUtf8;
}

#[test]
fn opaque_receipt_keeps_prompt_transport_pointer_sized() {
    let pointer = std::mem::size_of::<usize>();
    assert!(std::mem::size_of::<NativeRenderedPrompt>() <= 2 * pointer);
    // The pre-existing ModelSpecific variant contains two fat Arcs and a
    // length, so six pointer widths is the transport's established ceiling.
    assert!(std::mem::size_of::<NativePromptInput>() <= 6 * pointer);
}
