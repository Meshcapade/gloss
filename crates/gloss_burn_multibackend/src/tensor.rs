use burn::tensor::{
    ops::{BoolTensor, FloatTensor, IntTensor},
    quantization::{QTensorPrimitive, QuantScheme},
    DType, Shape, TensorMetadata,
};

use crate::backend::{NdArrayBackend, WgpuBackend};

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum MultiFloatTensor {
    NdArray(FloatTensor<NdArrayBackend>),
    // #[cfg(feature = "wgpu")]
    Wgpu(FloatTensor<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(FloatTensor<burn_autodiff::Autodiff<MultiBackend>>),
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum MultiIntTensor {
    NdArray(IntTensor<NdArrayBackend>),
    // #[cfg(feature = "wgpu")]
    Wgpu(IntTensor<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(IntTensor<burn_autodiff::Autodiff<MultiBackend>>),
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum MultiBoolTensor {
    NdArray(BoolTensor<NdArrayBackend>),
    // #[cfg(feature = "wgpu")]
    Wgpu(BoolTensor<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(BoolTensor<burn_autodiff::Autodiff<MultiBackend>>),
}

// TensorMetadata implementations
impl TensorMetadata for MultiFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        match self {
            MultiFloatTensor::NdArray(t) => t.shape(),
            // #[cfg(feature = "wgpu")]
            MultiFloatTensor::Wgpu(t) => t.shape(),
            // #[cfg(feature = "autodiff")]
            // MultiFloatTensor::Autodiff(t) => t.shape(),
        }
    }
}

impl TensorMetadata for MultiIntTensor {
    fn dtype(&self) -> DType {
        DType::I32
    }

    fn shape(&self) -> Shape {
        match self {
            MultiIntTensor::NdArray(t) => t.shape(),
            // #[cfg(feature = "wgpu")]
            MultiIntTensor::Wgpu(t) => t.shape(),
            // #[cfg(feature = "autodiff")]
            // MultiIntTensor::Autodiff(t) => t.shape(),
        }
    }
}

impl TensorMetadata for MultiBoolTensor {
    fn dtype(&self) -> DType {
        DType::U8
    }

    fn shape(&self) -> Shape {
        match self {
            MultiBoolTensor::NdArray(t) => t.shape(),
            // #[cfg(feature = "wgpu")]
            MultiBoolTensor::Wgpu(t) => t.shape(),
            // #[cfg(feature = "autodiff")]
            // MultiIntTensor::Autodiff(t) => t.shape(),
        }
    }
}

impl QTensorPrimitive for MultiIntTensor {
    fn scheme(&self) -> &QuantScheme {
        unimplemented!("Quantization is not supported")
    }
}
