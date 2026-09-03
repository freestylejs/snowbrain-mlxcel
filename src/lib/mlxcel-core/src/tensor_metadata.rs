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

//! Pure tensor-header validation used beside model constructors.
//!
//! The runtime loaders still own array creation.  These types deliberately
//! describe only the shape and dtype facts a safetensors header can prove, so
//! immutable package inspection can reuse the loader's construction schema
//! without reading payloads or crossing the CXX boundary.

use std::collections::BTreeMap;

use crate::layers::{QuantizedTensorShapes, infer_quantization_mode, validate_quantized_packing};

/// Dtypes understood by the pinned safetensors/MLX loader intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMetadataDtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    C64,
    F8E4M3,
    F8E8M0,
}

impl TensorMetadataDtype {
    /// Learned dense weights, norms, scales, and affine zero points accepted by
    /// the model constructors without a conversion-specific inspection proof.
    #[must_use]
    pub const fn is_learned_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32)
    }

    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::I8
                | Self::U16
                | Self::I16
                | Self::U32
                | Self::I32
                | Self::U64
                | Self::I64
        )
    }
}

/// Owned, payload-free metadata for one acquisition-verified tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    shape: Box<[usize]>,
    dtype: TensorMetadataDtype,
}

impl TensorMetadata {
    #[must_use]
    pub fn new(shape: impl Into<Box<[usize]>>, dtype: TensorMetadataDtype) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub const fn dtype(&self) -> TensorMetadataDtype {
        self.dtype
    }
}

/// Name-to-metadata view consumed by loader-owned schema validators.
pub trait TensorMetadataLookup {
    fn tensor_metadata(&self, name: &str) -> Option<&TensorMetadata>;
}

impl TensorMetadataLookup for BTreeMap<String, TensorMetadata> {
    fn tensor_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.get(name)
    }
}

/// Effective quantization parameters passed by the real array constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorQuantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtypeRequirement {
    LearnedFloat,
    PackedU32,
    Integer,
}

fn dtype_matches(dtype: TensorMetadataDtype, requirement: TensorDtypeRequirement) -> bool {
    match requirement {
        TensorDtypeRequirement::LearnedFloat => dtype.is_learned_float(),
        TensorDtypeRequirement::PackedU32 => dtype == TensorMetadataDtype::U32,
        TensorDtypeRequirement::Integer => dtype.is_integer(),
    }
}

/// Require one exact tensor shape and dtype class.
pub fn validate_required_tensor(
    lookup: &dyn TensorMetadataLookup,
    name: &str,
    expected_shape: &[usize],
    dtype: TensorDtypeRequirement,
) -> Result<(), String> {
    let metadata = lookup
        .tensor_metadata(name)
        .ok_or_else(|| format!("required tensor {name:?} is absent"))?;
    if metadata.shape() != expected_shape {
        return Err(format!(
            "tensor {name:?} has shape {:?}, expected {expected_shape:?}",
            metadata.shape()
        ));
    }
    if !dtype_matches(metadata.dtype(), dtype) {
        return Err(format!(
            "tensor {name:?} has dtype {:?}, which does not satisfy {dtype:?}",
            metadata.dtype()
        ));
    }
    Ok(())
}

/// Validate an optional tensor when the selected constructor consumes it.
pub fn validate_optional_tensor(
    lookup: &dyn TensorMetadataLookup,
    name: &str,
    expected_shape: &[usize],
    dtype: TensorDtypeRequirement,
) -> Result<(), String> {
    if lookup.tensor_metadata(name).is_none() {
        return Ok(());
    }
    validate_required_tensor(lookup, name, expected_shape, dtype)
}

fn shape_as_i32(name: &str, shape: &[usize]) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .copied()
        .map(|dimension| {
            i32::try_from(dimension).map_err(|_| {
                format!("tensor {name:?} dimension {dimension} exceeds the MLX ShapeElem range")
            })
        })
        .collect()
}

/// Validate one `UnifiedLinear` / `UnifiedEmbedding` matrix exactly as the
/// constructor selects it: a dense learned-float matrix, or a packed U32
/// matrix plus learned-float scales and optional affine zero points.
///
/// `allow_linear_bias` identifies the `UnifiedLinear` sidecars: the distinct
/// `.bias` and optional native-NVFP4 `.global_scale`. Embedding tables pass
/// `false` because their constructor ignores both keys.
pub fn validate_required_matrix(
    lookup: &dyn TensorMetadataLookup,
    prefix: &str,
    rows: usize,
    input_width: usize,
    quantization: TensorQuantization,
    allow_linear_bias: bool,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let scales_name = format!("{prefix}.scales");
    let biases_name = format!("{prefix}.biases");
    let global_scale_name = format!("{prefix}.global_scale");
    let bias_name = format!("{prefix}.bias");

    if let Some(scales) = lookup.tensor_metadata(&scales_name) {
        let weight = lookup
            .tensor_metadata(&weight_name)
            .ok_or_else(|| format!("required packed tensor {weight_name:?} is absent"))?;
        if weight.dtype() != TensorMetadataDtype::U32 {
            return Err(format!(
                "packed tensor {weight_name:?} has dtype {:?}, expected U32",
                weight.dtype()
            ));
        }
        if !scales.dtype().is_learned_float() {
            return Err(format!(
                "quantization scales {scales_name:?} have dtype {:?}, expected F16/BF16/F32",
                scales.dtype()
            ));
        }
        if weight.shape().len() != 2 || weight.shape()[0] != rows {
            return Err(format!(
                "packed tensor {weight_name:?} has shape {:?}, expected output axis [{rows}, ...]",
                weight.shape()
            ));
        }
        if scales.shape().len() != 2 || scales.shape()[0] != rows {
            return Err(format!(
                "quantization scales {scales_name:?} have shape {:?}, expected output axis [{rows}, ...]",
                scales.shape()
            ));
        }
        let biases = lookup.tensor_metadata(&biases_name);
        if let Some(biases) = biases
            && !biases.dtype().is_learned_float()
        {
            return Err(format!(
                "affine zero points {biases_name:?} have dtype {:?}, expected F16/BF16/F32",
                biases.dtype()
            ));
        }
        let weight_shape = shape_as_i32(&weight_name, weight.shape())?;
        let scales_shape = shape_as_i32(&scales_name, scales.shape())?;
        let biases_shape = biases
            .map(|metadata| shape_as_i32(&biases_name, metadata.shape()))
            .transpose()?;
        let mode =
            infer_quantization_mode(biases.is_some(), quantization.group_size, quantization.bits);
        validate_quantized_packing(
            prefix,
            &QuantizedTensorShapes {
                weight: &weight_shape,
                scales: &scales_shape,
                biases: biases_shape.as_deref(),
            },
            input_width,
            quantization.group_size,
            quantization.bits,
            mode,
        )?;
        if allow_linear_bias {
            validate_optional_tensor(
                lookup,
                &global_scale_name,
                &[1],
                TensorDtypeRequirement::LearnedFloat,
            )?;
        } else if lookup.tensor_metadata(&global_scale_name).is_some() {
            return Err(format!(
                "embedding tensor {global_scale_name:?} is present but its constructor does not consume it"
            ));
        }
    } else {
        if lookup.tensor_metadata(&biases_name).is_some() {
            return Err(format!(
                "tensor {biases_name:?} is present without the required {scales_name:?} plane"
            ));
        }
        if lookup.tensor_metadata(&global_scale_name).is_some() {
            return Err(format!(
                "tensor {global_scale_name:?} is present without the required {scales_name:?} plane"
            ));
        }
        validate_required_tensor(
            lookup,
            &weight_name,
            &[rows, input_width],
            TensorDtypeRequirement::LearnedFloat,
        )?;
    }

    if allow_linear_bias {
        validate_optional_tensor(
            lookup,
            &bias_name,
            &[rows],
            TensorDtypeRequirement::LearnedFloat,
        )?;
    } else if lookup.tensor_metadata(&bias_name).is_some() {
        return Err(format!(
            "embedding tensor {bias_name:?} is present but its constructor does not consume it"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(shape: &[usize], dtype: TensorMetadataDtype) -> TensorMetadata {
        TensorMetadata::new(shape.to_vec(), dtype)
    }

    #[test]
    fn dense_matrix_requires_exact_shape_and_float_dtype() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            metadata(&[8, 4], TensorMetadataDtype::F16),
        );
        validate_required_matrix(
            &tensors,
            "proj",
            8,
            4,
            TensorQuantization {
                group_size: 64,
                bits: 4,
            },
            true,
        )
        .unwrap();

        tensors.get_mut("proj.weight").unwrap().shape = vec![7, 4].into();
        assert!(
            validate_required_matrix(
                &tensors,
                "proj",
                8,
                4,
                TensorQuantization {
                    group_size: 64,
                    bits: 4,
                },
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn affine_matrix_checks_packed_dtype_output_axis_and_zero_points() {
        let mut tensors = BTreeMap::from([
            (
                "proj.weight".to_string(),
                metadata(&[8, 8], TensorMetadataDtype::U32),
            ),
            (
                "proj.scales".to_string(),
                metadata(&[8, 1], TensorMetadataDtype::Bf16),
            ),
            (
                "proj.biases".to_string(),
                metadata(&[8, 1], TensorMetadataDtype::Bf16),
            ),
        ]);
        validate_required_matrix(
            &tensors,
            "proj",
            8,
            64,
            TensorQuantization {
                group_size: 64,
                bits: 4,
            },
            false,
        )
        .unwrap();

        tensors.get_mut("proj.weight").unwrap().shape = vec![8, 2].into();
        assert!(
            validate_required_matrix(
                &tensors,
                "proj",
                8,
                64,
                TensorQuantization {
                    group_size: 64,
                    bits: 4,
                },
                false,
            )
            .is_err()
        );
        tensors.get_mut("proj.weight").unwrap().shape = vec![8, 8].into();

        tensors.get_mut("proj.weight").unwrap().dtype = TensorMetadataDtype::F16;
        assert!(
            validate_required_matrix(
                &tensors,
                "proj",
                8,
                64,
                TensorQuantization {
                    group_size: 64,
                    bits: 4,
                },
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn quantized_matrix_validates_optional_global_scale_scalar() {
        let mut tensors = BTreeMap::from([
            (
                "proj.weight".to_string(),
                metadata(&[8, 8], TensorMetadataDtype::U32),
            ),
            (
                "proj.scales".to_string(),
                metadata(&[8, 1], TensorMetadataDtype::Bf16),
            ),
            (
                "proj.biases".to_string(),
                metadata(&[8, 1], TensorMetadataDtype::Bf16),
            ),
            (
                "proj.global_scale".to_string(),
                metadata(&[1], TensorMetadataDtype::F32),
            ),
        ]);
        let validate = |tensors: &BTreeMap<String, TensorMetadata>| {
            validate_required_matrix(
                tensors,
                "proj",
                8,
                64,
                TensorQuantization {
                    group_size: 64,
                    bits: 4,
                },
                true,
            )
        };
        validate(&tensors).unwrap();

        tensors.get_mut("proj.global_scale").unwrap().shape = vec![2].into();
        assert!(validate(&tensors).is_err());
        tensors.get_mut("proj.global_scale").unwrap().shape = vec![1].into();
        tensors.get_mut("proj.global_scale").unwrap().dtype = TensorMetadataDtype::U32;
        assert!(validate(&tensors).is_err());

        tensors.get_mut("proj.global_scale").unwrap().dtype = TensorMetadataDtype::F32;
        assert!(
            validate_required_matrix(
                &tensors,
                "proj",
                8,
                64,
                TensorQuantization {
                    group_size: 64,
                    bits: 4,
                },
                false,
            )
            .is_err()
        );
    }
}
