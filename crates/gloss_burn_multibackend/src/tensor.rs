#[cfg(feature = "burn-candle")]
use crate::backend::CandleBackend;
#[cfg(feature = "burn-ndarray")]
use crate::backend::NdArrayBackend;
#[cfg(feature = "burn-wgpu")]
use crate::backend::WgpuBackend;

use burn::tensor::{
    ops::{BoolTensor, FloatTensor, IntTensor},
    quantization::{QTensorPrimitive, QuantScheme},
    DType, Shape, TensorMetadata,
};

#[derive(Debug, Clone)]
pub enum MultiFloatTensor {
    #[cfg(feature = "burn-candle")]
    Candle(FloatTensor<CandleBackend>),
    #[cfg(feature = "burn-ndarray")]
    NdArray(FloatTensor<NdArrayBackend>),
    #[cfg(feature = "burn-wgpu")]
    Wgpu(FloatTensor<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(FloatTensor<burn_autodiff::Autodiff<MultiBackend>>),
}

#[derive(Debug, Clone)]
pub enum MultiIntTensor {
    #[cfg(feature = "burn-candle")]
    Candle(IntTensor<CandleBackend>),
    #[cfg(feature = "burn-ndarray")]
    NdArray(IntTensor<NdArrayBackend>),
    #[cfg(feature = "burn-wgpu")]
    Wgpu(IntTensor<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(IntTensor<burn_autodiff::Autodiff<MultiBackend>>),
}

#[derive(Debug, Clone)]
pub enum MultiBoolTensor {
    #[cfg(feature = "burn-candle")]
    Candle(BoolTensor<CandleBackend>),
    #[cfg(feature = "burn-ndarray")]
    NdArray(BoolTensor<NdArrayBackend>),
    #[cfg(feature = "burn-wgpu")]
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
            #[cfg(feature = "burn-candle")]
            MultiFloatTensor::Candle(t) => t.shape(),
            #[cfg(feature = "burn-ndarray")]
            MultiFloatTensor::NdArray(t) => t.shape(),
            #[cfg(feature = "burn-wgpu")]
            MultiFloatTensor::Wgpu(t) => t.shape(),
            // #[cfg(feature = "autodiff")]
            // MultiFloatTensor::Autodiff(t) => t.shape(),
        }
    }
}

impl TensorMetadata for MultiIntTensor {
    fn dtype(&self) -> DType {
        match self {
            #[cfg(feature = "burn-candle")]
            MultiIntTensor::Candle(_) => DType::I64,
            #[cfg(feature = "burn-ndarray")]
            MultiIntTensor::NdArray(_) => DType::I32,
            #[cfg(feature = "burn-wgpu")]
            MultiIntTensor::Wgpu(_) => DType::I32,
        }
    }

    fn shape(&self) -> Shape {
        match self {
            #[cfg(feature = "burn-candle")]
            MultiIntTensor::Candle(t) => t.shape(),
            #[cfg(feature = "burn-ndarray")]
            MultiIntTensor::NdArray(t) => t.shape(),
            #[cfg(feature = "burn-wgpu")]
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
            #[cfg(feature = "burn-candle")]
            MultiBoolTensor::Candle(t) => t.shape(),
            #[cfg(feature = "burn-ndarray")]
            MultiBoolTensor::NdArray(t) => t.shape(),
            #[cfg(feature = "burn-wgpu")]
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
