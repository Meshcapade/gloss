use core::panic;

use burn::{
    backend::{candle::CandleDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice, Candle, NdArray, Wgpu},
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};
// use burn::backend::ndarray::PrecisionBridge as NdArrayBridge;
// use burn::backend::candle::PrecisionBridge as CandleBridge;
// use burn::backend::wgpu::WebGpu PrecisionBridge as WgpuBridge;
// use burn::tensor::backend::BackendBridge;

use crate::bshare::{tensor_to_data_float, tensor_to_data_int, ToBurn, ToNalgebraFloat, ToNalgebraInt};
extern crate nalgebra as na;
use bytemuck;
use log::warn;

pub type DefaultBackend = NdArray; // Change this as needed

#[derive(Clone)]
pub enum BurnBackend {
    Candle,
    NdArray,
    Wgpu,
}

/// `DynamicTensor` enum for Dynamic backend tensors in burn
#[derive(Clone, Debug)]
pub enum DynamicTensorFloat1D {
    NdArray(Tensor<NdArray, 1, Float>),
    Wgpu(Tensor<Wgpu, 1, Float>),
    Candle(Tensor<Candle, 1, Float>),
}

/// `DynamicTensor` enum for Dynamic backend tensors in burn
#[derive(Clone, Debug)]
pub enum DynamicTensorFloat2D {
    NdArray(Tensor<NdArray, 2, Float>),
    Wgpu(Tensor<Wgpu, 2, Float>),
    Candle(Tensor<Candle, 2, Float>),
}

/// `DynamicTensor` enum for Dynamic backend tensors in burn
#[derive(Clone, Debug)]
pub enum DynamicTensorInt1D {
    NdArray(Tensor<NdArray, 1, Int>),
    Wgpu(Tensor<Wgpu, 1, Int>),
    Candle(Tensor<Candle, 1, Int>),
}

/// `DynamicTensor` enum for Dynamic backend tensors in burn
#[derive(Clone, Debug)]
pub enum DynamicTensorInt2D {
    NdArray(Tensor<NdArray, 2, Int>),
    Wgpu(Tensor<Wgpu, 2, Int>),
    Candle(Tensor<Candle, 2, Int>),
}

/// From methods for converting from Tensor to `DynamicTensor`
impl DynamicTensorFloat1D {
    pub fn from_ndarray(tensor: Tensor<NdArray, 1, Float>) -> Self {
        DynamicTensorFloat1D::NdArray(tensor)
    }
    pub fn from_wgpu(tensor: Tensor<Wgpu, 1, Float>) -> Self {
        DynamicTensorFloat1D::Wgpu(tensor)
    }
    pub fn from_candle(tensor: Tensor<Candle, 1, Float>) -> Self {
        DynamicTensorFloat1D::Candle(tensor)
    }
}

/// From methods for converting from Tensor to `DynamicTensor`
impl DynamicTensorFloat2D {
    pub fn from_ndarray(tensor: Tensor<NdArray, 2, Float>) -> Self {
        DynamicTensorFloat2D::NdArray(tensor)
    }
    pub fn from_wgpu(tensor: Tensor<Wgpu, 2, Float>) -> Self {
        DynamicTensorFloat2D::Wgpu(tensor)
    }
    pub fn from_candle(tensor: Tensor<Candle, 2, Float>) -> Self {
        DynamicTensorFloat2D::Candle(tensor)
    }
}

/// From methods for converting from Tensor to `DynamicTensor`
impl DynamicTensorInt1D {
    pub fn from_ndarray(tensor: Tensor<NdArray, 1, Int>) -> Self {
        DynamicTensorInt1D::NdArray(tensor)
    }
    pub fn from_wgpu(tensor: Tensor<Wgpu, 1, Int>) -> Self {
        DynamicTensorInt1D::Wgpu(tensor)
    }
    pub fn from_candle(tensor: Tensor<Candle, 1, Int>) -> Self {
        DynamicTensorInt1D::Candle(tensor)
    }
}

/// From methods for converting from Tensor to `DynamicTensor`
impl DynamicTensorInt2D {
    pub fn from_ndarray(tensor: Tensor<NdArray, 2, Int>) -> Self {
        DynamicTensorInt2D::NdArray(tensor)
    }
    pub fn from_wgpu(tensor: Tensor<Wgpu, 2, Int>) -> Self {
        DynamicTensorInt2D::Wgpu(tensor)
    }
    pub fn from_candle(tensor: Tensor<Candle, 2, Int>) -> Self {
        DynamicTensorInt2D::Candle(tensor)
    }
}

// Conversion and Utility Operations for DynamicTensor variants

/// Trait for common `DynamicTensor` operations
pub trait DynamicTensorOps<T> {
    fn as_bytes(&self) -> Vec<u8>;

    fn nrows(&self) -> usize;
    fn shape(&self) -> (usize, usize);

    fn to_vec(&self) -> Vec<T>;
    fn min_vec(&self) -> Vec<T>;
    fn max_vec(&self) -> Vec<T>;
}

/// `DynamicTensorOps` for Float 1D tensors
impl DynamicTensorOps<f32> for DynamicTensorFloat1D {
    fn as_bytes(&self) -> Vec<u8> {
        match self {
            DynamicTensorFloat1D::NdArray(tensor) => bytemuck::cast_slice(&tensor_to_data_float(tensor)).to_vec(),
            DynamicTensorFloat1D::Wgpu(tensor) => bytemuck::cast_slice(&tensor_to_data_float(tensor)).to_vec(),
            DynamicTensorFloat1D::Candle(tensor) => bytemuck::cast_slice(&tensor_to_data_float(tensor)).to_vec(),
        }
    }

    fn nrows(&self) -> usize {
        match self {
            DynamicTensorFloat1D::NdArray(tensor) => tensor.dims()[0],
            DynamicTensorFloat1D::Wgpu(tensor) => tensor.dims()[0],
            DynamicTensorFloat1D::Candle(tensor) => tensor.dims()[0],
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            DynamicTensorFloat1D::NdArray(tensor) => (tensor.dims()[0], 1),
            DynamicTensorFloat1D::Wgpu(tensor) => (tensor.dims()[0], 1),
            DynamicTensorFloat1D::Candle(tensor) => (tensor.dims()[0], 1),
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        match &self {
            DynamicTensorFloat1D::NdArray(tensor) => tensor_to_data_float(tensor),
            DynamicTensorFloat1D::Wgpu(tensor) => tensor_to_data_float(tensor),
            DynamicTensorFloat1D::Candle(tensor) => tensor_to_data_float(tensor),
        }
    }

    fn min_vec(&self) -> Vec<f32> {
        vec![self.to_vec().iter().copied().fold(f32::INFINITY, f32::min)]
    }

    fn max_vec(&self) -> Vec<f32> {
        vec![self.to_vec().iter().copied().fold(f32::NEG_INFINITY, f32::max)]
    }
}

/// `DynamicTensorOps` for Float 2D tensors
impl DynamicTensorOps<f32> for DynamicTensorFloat2D {
    fn as_bytes(&self) -> Vec<u8> {
        match self {
            DynamicTensorFloat2D::NdArray(tensor) => {
                let tensor_data = tensor_to_data_float(tensor);
                bytemuck::cast_slice(&tensor_data).to_vec()
            }
            DynamicTensorFloat2D::Wgpu(tensor) => {
                warn!("Forcing DynamicTensor with Wgpu backend to CPU");
                let tensor_data = tensor_to_data_float(tensor);
                bytemuck::cast_slice(&tensor_data).to_vec()
            }
            DynamicTensorFloat2D::Candle(tensor) => {
                let tensor_data = tensor_to_data_float(tensor);
                bytemuck::cast_slice(&tensor_data).to_vec()
            }
        }
    }

    fn nrows(&self) -> usize {
        match self {
            DynamicTensorFloat2D::NdArray(tensor) => tensor.dims()[0],
            DynamicTensorFloat2D::Wgpu(tensor) => tensor.dims()[0],
            DynamicTensorFloat2D::Candle(tensor) => tensor.dims()[0],
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            DynamicTensorFloat2D::NdArray(tensor) => (tensor.dims()[0], tensor.dims()[1]),
            DynamicTensorFloat2D::Wgpu(tensor) => (tensor.dims()[0], tensor.dims()[1]),
            DynamicTensorFloat2D::Candle(tensor) => (tensor.dims()[0], tensor.dims()[1]),
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        match &self {
            DynamicTensorFloat2D::NdArray(tensor) => tensor_to_data_float(tensor),
            DynamicTensorFloat2D::Wgpu(tensor) => {
                warn!("Forcing DynamicTensor with Wgpu backend to CPU");
                tensor_to_data_float(tensor)
            }
            DynamicTensorFloat2D::Candle(tensor) => tensor_to_data_float(tensor),
        }
    }

    fn min_vec(&self) -> Vec<f32> {
        match &self {
            DynamicTensorFloat2D::NdArray(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_float(&min_tensor)
            }
            DynamicTensorFloat2D::Wgpu(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_float(&min_tensor)
            }
            DynamicTensorFloat2D::Candle(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_float(&min_tensor)
            }
        }
    }

    fn max_vec(&self) -> Vec<f32> {
        match &self {
            DynamicTensorFloat2D::NdArray(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_float(&max_tensor)
            }
            DynamicTensorFloat2D::Wgpu(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_float(&max_tensor)
            }
            DynamicTensorFloat2D::Candle(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_float(&max_tensor)
            }
        }
    }
}

/// `DynamicTensorOps` for Int 1D tensors
impl DynamicTensorOps<u32> for DynamicTensorInt1D {
    fn as_bytes(&self) -> Vec<u8> {
        match self {
            DynamicTensorInt1D::NdArray(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
            DynamicTensorInt1D::Wgpu(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
            DynamicTensorInt1D::Candle(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
        }
    }

    fn nrows(&self) -> usize {
        match self {
            DynamicTensorInt1D::NdArray(tensor) => tensor.dims()[0],
            DynamicTensorInt1D::Wgpu(tensor) => tensor.dims()[0],
            DynamicTensorInt1D::Candle(tensor) => tensor.dims()[0],
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            DynamicTensorInt1D::NdArray(tensor) => (tensor.dims()[0], 1),
            DynamicTensorInt1D::Wgpu(tensor) => (tensor.dims()[0], 1),
            DynamicTensorInt1D::Candle(tensor) => (tensor.dims()[0], 1),
        }
    }

    fn to_vec(&self) -> Vec<u32> {
        match &self {
            DynamicTensorInt1D::NdArray(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt1D::Wgpu(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt1D::Candle(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
        }
    }

    fn min_vec(&self) -> Vec<u32> {
        vec![self.to_vec().into_iter().min().unwrap_or(0)]
    }

    fn max_vec(&self) -> Vec<u32> {
        vec![self.to_vec().into_iter().max().unwrap_or(0)]
    }
}

/// `DynamicTensorOps` for Int 2D tensors
impl DynamicTensorOps<u32> for DynamicTensorInt2D {
    fn as_bytes(&self) -> Vec<u8> {
        match self {
            DynamicTensorInt2D::NdArray(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
            DynamicTensorInt2D::Wgpu(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
            DynamicTensorInt2D::Candle(tensor) => {
                let tensor_data = tensor_to_data_int(tensor);
                let u32_data: Vec<u32> = tensor_data
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect();
                bytemuck::cast_slice(&u32_data).to_vec()
            }
        }
    }

    fn nrows(&self) -> usize {
        match self {
            DynamicTensorInt2D::NdArray(tensor) => tensor.dims()[0],
            DynamicTensorInt2D::Wgpu(tensor) => tensor.dims()[0],
            DynamicTensorInt2D::Candle(tensor) => tensor.dims()[0],
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            DynamicTensorInt2D::NdArray(tensor) => (tensor.dims()[0], tensor.dims()[1]),
            DynamicTensorInt2D::Wgpu(tensor) => (tensor.dims()[0], tensor.dims()[1]),
            DynamicTensorInt2D::Candle(tensor) => (tensor.dims()[0], tensor.dims()[1]),
        }
    }

    fn to_vec(&self) -> Vec<u32> {
        match &self {
            DynamicTensorInt2D::NdArray(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Wgpu(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Candle(tensor) => {
                let data = tensor_to_data_int(tensor);
                data.into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
        }
    }

    fn min_vec(&self) -> Vec<u32> {
        match &self {
            DynamicTensorInt2D::NdArray(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_int(&min_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Wgpu(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_int(&min_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Candle(tensor) => {
                let min_tensor = tensor.clone().min_dim(0);
                tensor_to_data_int(&min_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
        }
    }

    fn max_vec(&self) -> Vec<u32> {
        match &self {
            DynamicTensorInt2D::NdArray(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_int(&max_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Wgpu(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_int(&max_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
            DynamicTensorInt2D::Candle(tensor) => {
                let max_tensor = tensor.clone().max_dim(0);
                tensor_to_data_int(&max_tensor)
                    .into_iter()
                    .map(|x| x.try_into().expect("Negative value found during conversion to u32"))
                    .collect()
            }
        }
    }
}

/// Trait for conversion to and from nalgebra matrices
pub trait DynamicMatrixOps<T> {
    fn from_dmatrix(matrix: &na::DMatrix<T>) -> Self;
    fn to_dmatrix(&self) -> na::DMatrix<T>;
    fn into_dmatrix(self) -> na::DMatrix<T>;
}

/// `DynamicMatrixOps` for `DynamicTensorFloat2D`
impl DynamicMatrixOps<f32> for DynamicTensorFloat2D {
    fn from_dmatrix(matrix: &na::DMatrix<f32>) -> Self {
        match std::any::TypeId::of::<DefaultBackend>() {
            id if id == std::any::TypeId::of::<NdArray>() => {
                let tensor = matrix.to_burn(&NdArrayDevice::Cpu);
                DynamicTensorFloat2D::NdArray(tensor)
            }
            id if id == std::any::TypeId::of::<Candle>() => {
                let tensor = matrix.to_burn(&CandleDevice::Cpu);
                DynamicTensorFloat2D::Candle(tensor)
            }
            id if id == std::any::TypeId::of::<Wgpu>() => {
                let tensor = matrix.to_burn(&WgpuDevice::BestAvailable);
                DynamicTensorFloat2D::Wgpu(tensor)
            }
            _ => panic!("Unsupported backend!"),
        }
    }

    fn to_dmatrix(&self) -> na::DMatrix<f32> {
        match self {
            DynamicTensorFloat2D::NdArray(tensor) => tensor.to_nalgebra(),
            DynamicTensorFloat2D::Wgpu(tensor) => tensor.to_nalgebra(),
            DynamicTensorFloat2D::Candle(tensor) => tensor.to_nalgebra(),
        }
    }

    fn into_dmatrix(self) -> na::DMatrix<f32> {
        match self {
            DynamicTensorFloat2D::NdArray(tensor) => tensor.into_nalgebra(),
            DynamicTensorFloat2D::Wgpu(tensor) => tensor.into_nalgebra(),
            DynamicTensorFloat2D::Candle(tensor) => tensor.into_nalgebra(),
        }
    }
}

/// `DynamicMatrixOps` for `DynamicTensorInt2D`
impl DynamicMatrixOps<u32> for DynamicTensorInt2D {
    fn from_dmatrix(matrix: &na::DMatrix<u32>) -> Self {
        match std::any::TypeId::of::<DefaultBackend>() {
            id if id == std::any::TypeId::of::<NdArray>() => {
                let tensor = matrix.to_burn(&NdArrayDevice::Cpu);
                DynamicTensorInt2D::NdArray(tensor)
            }
            id if id == std::any::TypeId::of::<Candle>() => {
                let tensor = matrix.to_burn(&CandleDevice::Cpu);
                DynamicTensorInt2D::Candle(tensor)
            }
            id if id == std::any::TypeId::of::<Wgpu>() => {
                let tensor = matrix.to_burn(&WgpuDevice::BestAvailable);
                DynamicTensorInt2D::Wgpu(tensor)
            }
            _ => panic!("Unsupported backend!"),
        }
    }

    fn to_dmatrix(&self) -> na::DMatrix<u32> {
        match self {
            DynamicTensorInt2D::NdArray(tensor) => tensor.to_nalgebra(),
            DynamicTensorInt2D::Wgpu(tensor) => tensor.to_nalgebra(),
            DynamicTensorInt2D::Candle(tensor) => tensor.to_nalgebra(),
        }
    }

    fn into_dmatrix(self) -> na::DMatrix<u32> {
        match self {
            DynamicTensorInt2D::NdArray(tensor) => tensor.into_nalgebra(),
            DynamicTensorInt2D::Wgpu(tensor) => tensor.into_nalgebra(),
            DynamicTensorInt2D::Candle(tensor) => tensor.into_nalgebra(),
        }
    }
}

// /////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////// ///// Some burn utilities
// /////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////
/// Normalise a 2D tensor across dim 1
pub fn normalize_tensor<B: Backend>(tensor: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
    let norm = tensor.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // Compute the L2 norm along the last axis (dim = 1)
    tensor.div(norm) // Divide each vector by its norm
}

/// Cross product of 2 2D Tensors
pub fn cross_product<B: Backend>(
    a: &Tensor<B, 2, Float>, // Tensor of shape [N, 3]
    b: &Tensor<B, 2, Float>, // Tensor of shape [N, 3]
) -> Tensor<B, 2, Float> {
    // Split the input tensors along dimension 1 (the 3 components) using chunk
    let a_chunks = a.clone().chunk(3, 1); // Split tensor `a` into 3 chunks: ax, ay, az
    let b_chunks = b.clone().chunk(3, 1); // Split tensor `b` into 3 chunks: bx, by, bz

    let ax: Tensor<B, 1> = a_chunks[0].clone().squeeze(1); // x component of a
    let ay: Tensor<B, 1> = a_chunks[1].clone().squeeze(1); // y component of a
    let az: Tensor<B, 1> = a_chunks[2].clone().squeeze(1); // z component of a

    let bx: Tensor<B, 1> = b_chunks[0].clone().squeeze(1); // x component of b
    let by: Tensor<B, 1> = b_chunks[1].clone().squeeze(1); // y component of b
    let bz: Tensor<B, 1> = b_chunks[2].clone().squeeze(1); // z component of b

    // Compute the components of the cross product
    let cx = ay.clone().mul(bz.clone()).sub(az.clone().mul(by.clone())); // cx = ay * bz - az * by
    let cy = az.mul(bx.clone()).sub(ax.clone().mul(bz)); // cy = az * bx - ax * bz
    let cz = ax.mul(by).sub(ay.mul(bx)); // cz = ax * by - ay * bx

    // Stack the result to form the resulting [N, 3] tensor
    Tensor::stack(vec![cx, cy, cz], 1) // Concatenate along the second dimension
}
