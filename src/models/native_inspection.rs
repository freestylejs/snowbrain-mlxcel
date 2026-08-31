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

//! Loader-owned qualification status for neutral native inspection.
//!
//! This table lives beside the root model registry rather than in the native
//! adapter. Adding a [`ModelType`] therefore makes this exhaustive match fail
//! to compile until the root loader family is either backed by a complete
//! inspection descriptor or explicitly deferred. A deferred family can still
//! expose its detected architecture, but it is not executable authority.

use super::ModelType;

/// Root loader families whose complete construction schema is available to
/// immutable native package inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeQualifiedFamily {
    Gemma4Unified,
    Gemma4UnifiedAssistant,
    Bert,
}

/// Exhaustive native-inspection disposition owned by the root loader registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeFamilyInspectionDescriptor {
    Qualified(NativeQualifiedFamily),
    ExplicitlyDeferred(&'static str),
}

const UNQUALIFIED_LOADER_SCHEMA: &str =
    "the root loader has no complete native tensor-schema descriptor yet";

/// Resolve a root [`ModelType`] to its explicit native-inspection status.
///
/// There is intentionally no wildcard arm. This match is the compile-time
/// exhaustiveness gate for the model registry.
pub(crate) fn native_family_inspection_descriptor(
    model_type: ModelType,
) -> NativeFamilyInspectionDescriptor {
    use ModelType::*;
    match model_type {
        Gemma4Unified => {
            NativeFamilyInspectionDescriptor::Qualified(NativeQualifiedFamily::Gemma4Unified)
        }
        Bert => NativeFamilyInspectionDescriptor::Qualified(NativeQualifiedFamily::Bert),
        Llama
        | Llama4
        | Llama4VLM
        | MllamaVLM
        | Qwen2
        | Qwen3
        | Qwen3Moe
        | Qwen3Next
        | Qwen35
        | Qwen35VLM
        | Qwen35Moe
        | Qwen35MoeVLM
        | Gemma
        | Gemma2
        | Gemma3
        | Gemma4
        | DiffusionGemma
        | Llada2Moe
        | Gemma3VLM
        | Gemma4VLM
        | LlavaVLM
        | GraniteVisionVLM
        | Granite4VisionVLM
        | DeepSeekOcrVLM
        | DeepSeekOcr2VLM
        | UnlimitedOcrVLM
        | DeepSeekVL2
        | LlavaBunnyVLM
        | FastVLM
        | Ernie45MoeVLM
        | HunyuanVLM
        | AyaVisionVLM
        | PaliGemmaVLM
        | PixtralVLM
        | Mistral3VLM
        | Qwen2VL
        | Qwen25VL
        | Qwen3VL
        | Qwen3VLMoe
        | Qwen3OmniMoe
        | PaddleOcrVL
        | DotsOcrVL
        | FalconOcrVL
        | JinaVLM
        | Glm4v
        | Glm4vMoe
        | GlmOcr
        | YoutuVLM
        | InternVLChatVLM
        | LocateAnythingVLM
        | SmolVLM
        | Idefics2
        | MiniCPMOVLM
        | MiniCPMV46VLM
        | Moondream3VLM
        | Moondream2VLM
        | Florence2VLM
        | Gemma3n
        | Gemma3nVLM
        | Phi
        | Phixtral
        | Phi3
        | Phi4MMVLM
        | Phi4SigLipVLM
        | Phi3VLM
        | MolmoVLM
        | Molmo2VLM
        | MolmoPointVLM
        | Phi3Small
        | PhiMoe
        | GptOss
        | MiniMax
        | MiniMaxM3
        | MiniMaxM3VL
        | MuseGlimmerVLM
        | Mixtral
        | Qwen2Moe
        | OLMoE
        | Dbrx
        | DeepSeek
        | DeepSeekV2
        | DeepSeekV3
        | DeepSeekV32
        | DeepSeekV4
        | Dots1
        | Cohere
        | Cohere2
        | Cohere2Moe
        | InternLM2
        | InternLM3
        | Baichuan
        | Glm4
        | Glm4Moe
        | Glm4MoeLite
        | GlmMoeDsa
        | Ernie45
        | Ernie45Moe
        | HunyuanMoe
        | HunyuanV1Dense
        | MiMo
        | BailingMoe
        | BailingMoeLinear
        | Afmoe
        | Klear
        | Apertus
        | SeedOss
        | Granite
        | BitNet
        | ExaOne
        | ExaOne4
        | ExaOneMoe
        | SolarOpen
        | Olmo
        | Olmo2
        | Olmo3
        | OpenElm
        | Gpt2
        | GptBigCode
        | GptNeoX
        | StarCoder2
        | Mellum
        | Helium
        | TeleChat3
        | MiniCPM
        | MiniCPM3
        | StableLM
        | SmolLM3
        | Ministral3
        | Mistral3
        | Mistral4
        | Nemotron
        | Mamba
        | Mamba2
        | Jamba
        | NemotronH
        | NemotronHNanoOmniVLM
        | NemotronNAS
        | FalconH1
        | Lfm2
        | Lfm2Moe
        | Lfm2VL
        | Plamo2
        | GraniteMoeHybrid
        | KimiLinear
        | KimiVL
        | KimiK25
        | LongcatFlash
        | LongcatFlashNgram
        | Step3p5
        | Step3p7
        | Rwkv7
        | RecurrentGemma
        | Whisper
        | Kokoro
        | XlmRoberta
        | ModernBert
        | SiglipText
        | Gemma3Embedding
        | Qwen3Embedding
        | Qwen3VLEmbedding
        | Lfm2Embedding
        | Ministral3Embedding
        | LlamaBidirec
        | LlamaNemotronVLEmbedding
        | ColIdefics3
        | ColQwen25
        | SequenceClassifier => {
            NativeFamilyInspectionDescriptor::ExplicitlyDeferred(UNQUALIFIED_LOADER_SCHEMA)
        }
    }
}

/// Resolve the drafter family that has its own loader registry outside
/// [`ModelType`]. Other declarations remain under normal model detection.
pub(crate) fn native_declared_drafter_descriptor(
    declared_model_type: &str,
) -> Option<NativeFamilyInspectionDescriptor> {
    match declared_model_type {
        "gemma4_assistant" | "gemma4_unified_assistant" => {
            Some(NativeFamilyInspectionDescriptor::Qualified(
                NativeQualifiedFamily::Gemma4UnifiedAssistant,
            ))
        }
        "qwen3_5_mtp" => Some(NativeFamilyInspectionDescriptor::ExplicitlyDeferred(
            "the qwen3_5_mtp drafter has no complete native tensor-schema descriptor yet",
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_registry_has_an_explicit_native_inspection_disposition() {
        let mut qualified = Vec::new();
        for &model_type in super::super::ALL_MODEL_TYPES {
            match native_family_inspection_descriptor(model_type) {
                NativeFamilyInspectionDescriptor::Qualified(family) => qualified.push(family),
                NativeFamilyInspectionDescriptor::ExplicitlyDeferred(reason) => {
                    assert!(
                        !reason.trim().is_empty(),
                        "{model_type:?} has an empty reason"
                    );
                }
            }
        }
        assert_eq!(
            qualified,
            [
                NativeQualifiedFamily::Gemma4Unified,
                NativeQualifiedFamily::Bert
            ]
        );
    }

    #[test]
    fn exact_assistant_declarations_select_only_the_qualified_drafter() {
        for declared in ["gemma4_assistant", "gemma4_unified_assistant"] {
            assert_eq!(
                native_declared_drafter_descriptor(declared),
                Some(NativeFamilyInspectionDescriptor::Qualified(
                    NativeQualifiedFamily::Gemma4UnifiedAssistant
                ))
            );
        }
        assert!(matches!(
            native_declared_drafter_descriptor("qwen3_5_mtp"),
            Some(NativeFamilyInspectionDescriptor::ExplicitlyDeferred(_))
        ));
        assert_eq!(native_declared_drafter_descriptor("gemma4_unified"), None);
    }
}
