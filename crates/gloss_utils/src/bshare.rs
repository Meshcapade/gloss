use burn::{
    prelude::Backend,
    tensor::{Float, Int, Tensor, TensorKind},
};
use nalgebra as na;
use ndarray as nd;

// ================ Tensor to Data Functions ================
// Handle Float tensors for both 1D and 2D
/// Convert a burn float tensor to a Vec on wasm
#[cfg(target_arch = "wasm32")]
pub fn tensor_to_data_float<B: Backend, const D: usize>(tensor: &Tensor<B, D, Float>) -> Vec<f32> {
    // tensor.to_data().block_on().to_vec::<f32>().unwrap()
    tensor.to_data().to_vec::<f32>().unwrap()
}

/// Convert a burn float tensor to a Vec
#[cfg(not(target_arch = "wasm32"))]
pub fn tensor_to_data_float<B: Backend, const D: usize>(tensor: &Tensor<B, D, Float>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().unwrap()
}

// Handle Int tensors for both 1D and 2D
/// Convert a burn int tensor to a Vec on wasm
#[cfg(target_arch = "wasm32")]
#[allow(clippy::cast_possible_truncation)]
pub fn tensor_to_data_int<B: Backend, const D: usize>(tensor: &Tensor<B, D, Int>) -> Vec<i32> {
    if let Ok(data) = tensor.to_data().to_vec::<i32>() {
        return data;
    }

    // Fallback: Attempt `i64` conversion and downcast to `i32` if `i32` fails
    let data_i64: Vec<i64> = tensor.to_data().to_vec::<i64>().unwrap();
    data_i64.into_iter().map(|x| x as i32).collect()
}

/// Convert a burn int tensor to a Vec
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::cast_possible_truncation)]
pub fn tensor_to_data_int<B: Backend, const D: usize>(tensor: &Tensor<B, D, Int>) -> Vec<i32> {
    // tensor.to_data().to_vec::<i32>().unwrap()
    if let Ok(data) = tensor.to_data().to_vec::<i32>() {
        return data;
    }

    // Fallback: Attempt `i64` conversion and downcast to `i32` if `i32` fails
    let data_i64: Vec<i64> = tensor.to_data().to_vec::<i64>().unwrap();
    data_i64.into_iter().map(|x| x as i32).collect()
}

// ================ To and Into Burn Conversions ================

/// Trait for converting ndarray to burn tensor (generic over Float/Int and
/// dimensionality)
pub trait ToBurn<B: Backend, const D: usize, T: TensorKind<B>> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, D, T>;
    fn into_burn(self, device: &B::Device) -> Tensor<B, D, T>;
}

/// Implementation of the trait for 2D Float ndarray
impl<B: Backend> ToBurn<B, 2, Float> for nd::Array2<f32> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 2, Float> {
        let vec: Vec<f32>;
        let bytes = if self.is_standard_layout() {
            self.as_slice().unwrap()
        } else {
            vec = self.iter().copied().collect();
            vec.as_slice()
        };
        let shape = [self.nrows(), self.ncols()];
        Tensor::<B, 1, Float>::from_floats(bytes, device).reshape(shape)
    }

    fn into_burn(self, device: &B::Device) -> Tensor<B, 2, Float> {
        let vec: Vec<f32>;
        let bytes = if self.is_standard_layout() {
            self.as_slice().expect("Array should have a slice if it's in standard layout")
        } else {
            vec = self.iter().copied().collect();
            vec.as_slice()
        };
        let shape = [self.nrows(), self.ncols()];
        Tensor::<B, 1, Float>::from_floats(bytes, device).reshape(shape)
    }
}

/// Trait implementation for 1D Float ndarray
impl<B: Backend> ToBurn<B, 1, Float> for nd::Array1<f32> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 1, Float> {
        let vec: Vec<f32> = self.iter().copied().collect();
        Tensor::<B, 1, Float>::from_floats(&vec[..], device)
    }

    fn into_burn(self, device: &B::Device) -> Tensor<B, 1, Float> {
        let vec: Vec<f32>;
        let bytes = if self.is_standard_layout() {
            self.as_slice().expect("Array should have a slice if it's in standard layout")
        } else {
            vec = self.iter().copied().collect();
            vec.as_slice()
        };
        Tensor::<B, 1, Float>::from_floats(bytes, device)
    }
}

/// Trait implementation for 2D Int ndarray
impl<B: Backend> ToBurn<B, 2, Int> for nd::Array2<u32> {
    #[allow(clippy::cast_possible_wrap)]
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        let shape = [self.nrows(), self.ncols()];
        Tensor::<B, 1, Int>::from_ints(&vec[..], device).reshape(shape)
    }

    #[allow(clippy::cast_possible_wrap)]
    fn into_burn(self, device: &B::Device) -> Tensor<B, 2, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        let shape = [self.nrows(), self.ncols()];
        Tensor::<B, 1, Int>::from_ints(&vec[..], device).reshape(shape)
    }
}

/// Trait implementation for 1D Int ndarray
impl<B: Backend> ToBurn<B, 1, Int> for nd::Array1<u32> {
    #[allow(clippy::cast_possible_wrap)]
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        Tensor::<B, 1, Int>::from_ints(&vec[..], device)
    }

    #[allow(clippy::cast_possible_wrap)]
    fn into_burn(self, device: &B::Device) -> Tensor<B, 1, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        Tensor::<B, 1, Int>::from_ints(&vec[..], device)
    }
}
impl<B: Backend> ToBurn<B, 3, Float> for nd::Array3<f32> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 3, Float> {
        let vec: Vec<f32>;
        let bytes = if self.is_standard_layout() {
            self.as_slice().unwrap()
        } else {
            vec = self.iter().copied().collect();
            vec.as_slice()
        };
        let shape = [self.shape()[0], self.shape()[1], self.shape()[2]];
        Tensor::<B, 1, Float>::from_floats(bytes, device).reshape(shape)
    }

    fn into_burn(self, device: &B::Device) -> Tensor<B, 3, Float> {
        let vec: Vec<f32>;
        let bytes = if self.is_standard_layout() {
            self.as_slice().expect("Array should have a slice if it's in standard layout")
        } else {
            vec = self.iter().copied().collect();
            vec.as_slice()
        };
        let shape = [self.shape()[0], self.shape()[1], self.shape()[2]];
        Tensor::<B, 1, Float>::from_floats(bytes, device).reshape(shape)
    }
}

/// Trait implementation for 3D Int ndarray
impl<B: Backend> ToBurn<B, 3, Int> for nd::Array3<u32> {
    #[allow(clippy::cast_possible_wrap)]
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 3, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        let shape = [self.shape()[0], self.shape()[1], self.shape()[2]];
        Tensor::<B, 1, Int>::from_ints(&vec[..], device).reshape(shape)
    }

    #[allow(clippy::cast_possible_wrap)]
    fn into_burn(self, device: &B::Device) -> Tensor<B, 3, Int> {
        let array_i32 = self.mapv(|x| x as i32);
        let vec: Vec<i32> = array_i32.into_raw_vec_and_offset().0;
        // let vec: Vec<i32> = array_i32.into_raw_vec();
        let shape = [self.shape()[0], self.shape()[1], self.shape()[2]];
        Tensor::<B, 1, Int>::from_ints(&vec[..], device).reshape(shape)
    }
}
/// Implement `ToBurn` for converting `nalgebra::DMatrix<f32>` to a burn tensor
/// (Float type)
impl<B: Backend> ToBurn<B, 2, Float> for na::DMatrix<f32> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 2, Float> {
        let num_rows = self.nrows();
        let num_cols = self.ncols();
        let flattened: Vec<f32> = self.transpose().as_slice().to_vec();
        Tensor::<B, 1, Float>::from_floats(&flattened[..], device).reshape([num_rows, num_cols])
    }

    fn into_burn(self, device: &B::Device) -> Tensor<B, 2, Float> {
        let num_rows = self.nrows();
        let num_cols = self.ncols();
        let flattened: Vec<f32> = self.transpose().as_slice().to_vec();
        Tensor::<B, 1, Float>::from_floats(&flattened[..], device).reshape([num_rows, num_cols])
    }
}

/// Implement `ToBurn` for converting `nalgebra::DMatrix<u32>` to a burn tensor
/// (Int type)
impl<B: Backend> ToBurn<B, 2, Int> for na::DMatrix<u32> {
    fn to_burn(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let num_rows = self.nrows();
        let num_cols = self.ncols();
        let flattened: Vec<i32> = self
            .transpose()
            .as_slice()
            .iter()
            .map(|&x| i32::try_from(x).expect("Value out of range for i32"))
            .collect();
        Tensor::<B, 1, Int>::from_ints(&flattened[..], device).reshape([num_rows, num_cols])
    }

    fn into_burn(self, device: &B::Device) -> Tensor<B, 2, Int> {
        let num_rows = self.nrows();
        let num_cols = self.ncols();
        let flattened: Vec<i32> = self
            .transpose()
            .as_slice()
            .iter()
            .map(|&x| i32::try_from(x).expect("Value out of range for i32"))
            .collect();
        Tensor::<B, 1, Int>::from_ints(&flattened[..], device).reshape([num_rows, num_cols])
    }
}

// ================ To and Into NdArray Conversions ================

/// Trait for converting burn tensor to ndarray (generic over Float/Int and
/// dimensionality)
pub trait ToNdArray<B: Backend, const D: usize, T> {
    fn to_ndarray(&self) -> nd::Array<T, nd::Dim<[usize; D]>>;
    fn into_ndarray(self) -> nd::Array<T, nd::Dim<[usize; D]>>;
}

/// Trait implementation for converting 2D Float burn tensor to ndarray
impl<B: Backend> ToNdArray<B, 2, f32> for Tensor<B, 2, Float> {
    fn to_ndarray(&self) -> nd::Array2<f32> {
        let tensor_data = tensor_to_data_float(self);
        let shape = self.dims();
        nd::Array2::from_shape_vec((shape[0], shape[1]), tensor_data).unwrap()
    }

    fn into_ndarray(self) -> nd::Array2<f32> {
        let tensor_data = tensor_to_data_float(&self);
        let shape = self.dims();
        nd::Array2::from_shape_vec((shape[0], shape[1]), tensor_data).unwrap()
    }
}

/// Trait implementation for converting 1D Float burn tensor to ndarray
impl<B: Backend> ToNdArray<B, 1, f32> for Tensor<B, 1, Float> {
    fn to_ndarray(&self) -> nd::Array1<f32> {
        let tensor_data = tensor_to_data_float(self);
        nd::Array1::from_vec(tensor_data)
    }

    fn into_ndarray(self) -> nd::Array1<f32> {
        let tensor_data = tensor_to_data_float(&self);
        nd::Array1::from_vec(tensor_data)
    }
}

/// Trait implementation for converting 2D Int burn tensor to ndarray
#[allow(clippy::cast_sign_loss)]
impl<B: Backend> ToNdArray<B, 2, u32> for Tensor<B, 2, Int> {
    fn to_ndarray(&self) -> nd::Array2<u32> {
        let tensor_data = tensor_to_data_int(self);
        let tensor_data_u32: Vec<u32> = tensor_data.into_iter().map(|x| x as u32).collect();
        let shape = self.dims();
        nd::Array2::from_shape_vec((shape[0], shape[1]), tensor_data_u32).unwrap()
    }

    fn into_ndarray(self) -> nd::Array2<u32> {
        let tensor_data = tensor_to_data_int(&self);
        let tensor_data_u32: Vec<u32> = tensor_data.into_iter().map(|x| x as u32).collect();
        let shape = self.dims();
        nd::Array2::from_shape_vec((shape[0], shape[1]), tensor_data_u32).unwrap()
    }
}

/// Trait implementation for converting 1D Int burn tensor to ndarray
#[allow(clippy::cast_sign_loss)]
impl<B: Backend> ToNdArray<B, 1, u32> for Tensor<B, 1, Int> {
    fn to_ndarray(&self) -> nd::Array1<u32> {
        let tensor_data = tensor_to_data_int(self);
        let tensor_data_u32: Vec<u32> = tensor_data.into_iter().map(|x| x as u32).collect();
        nd::Array1::from_vec(tensor_data_u32)
    }

    fn into_ndarray(self) -> nd::Array1<u32> {
        let tensor_data = tensor_to_data_int(&self);
        let tensor_data_u32: Vec<u32> = tensor_data.into_iter().map(|x| x as u32).collect();
        nd::Array1::from_vec(tensor_data_u32)
    }
}

// ================ To and Into Nalgebra Conversions ================

/// Trait for converting `burn` tensor to `nalgebra::DMatrix` or
/// `nalgebra::DVector` (Float type)
pub trait ToNalgebraFloat<B: Backend, const D: usize> {
    fn to_nalgebra(&self) -> na::DMatrix<f32>;
    fn into_nalgebra(self) -> na::DMatrix<f32>;
}

/// Trait for converting `burn` tensor to `nalgebra::DMatrix` or
/// `nalgebra::DVector` (Int type)
pub trait ToNalgebraInt<B: Backend, const D: usize> {
    fn to_nalgebra(&self) -> na::DMatrix<u32>;
    fn into_nalgebra(self) -> na::DMatrix<u32>;
}

/// Implement trait to convert `burn` tensor to `nalgebra::DMatrix<f32>` (Float
/// type)
impl<B: Backend> ToNalgebraFloat<B, 2> for Tensor<B, 2, Float> {
    fn to_nalgebra(&self) -> na::DMatrix<f32> {
        let data = tensor_to_data_float(self);
        let shape = self.shape().dims;
        na::DMatrix::from_vec(shape[1], shape[0], data).transpose()
    }

    fn into_nalgebra(self) -> na::DMatrix<f32> {
        let data = tensor_to_data_float(&self);
        let shape = self.shape().dims;
        na::DMatrix::from_vec(shape[1], shape[0], data).transpose()
    }
}

/// Implement trait to convert `burn` tensor to `nalgebra::DMatrix<u32>` (Int
/// type)
impl<B: Backend> ToNalgebraInt<B, 2> for Tensor<B, 2, Int> {
    #[allow(clippy::cast_sign_loss)]
    fn to_nalgebra(&self) -> na::DMatrix<u32> {
        let data = tensor_to_data_int(self);
        let shape = self.shape().dims;
        let data_u32: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
        na::DMatrix::from_vec(shape[1], shape[0], data_u32).transpose()
    }
    #[allow(clippy::cast_sign_loss)]
    fn into_nalgebra(self) -> na::DMatrix<u32> {
        let data = tensor_to_data_int(&self);
        let shape = self.shape().dims;
        let data_u32: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
        na::DMatrix::from_vec(shape[1], shape[0], data_u32).transpose()
    }
}
