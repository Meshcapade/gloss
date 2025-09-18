use std::ops::Range;

use burn::tensor::{ops::BoolTensorOps, Shape, TensorData};

use crate::{
    backend::{MultiBackend, MultiDevice},
    tensor::{MultiBoolTensor, MultiFloatTensor, MultiIntTensor},
};

#[allow(unused_variables)]
impl BoolTensorOps<Self> for MultiBackend {
    fn bool_from_data(data: TensorData, device: &MultiDevice) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_repeat_dim(tensor: MultiBoolTensor, dim: usize, times: usize) -> MultiBoolTensor {
        unimplemented!()
    }
    async fn bool_into_data(tensor: MultiBoolTensor) -> TensorData {
        unimplemented!()
    }
    fn bool_to_device(tensor: MultiBoolTensor, device: &MultiDevice) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_reshape(tensor: MultiBoolTensor, shape: Shape) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_device(tensor: &MultiBoolTensor) -> MultiDevice {
        unimplemented!()
    }
    fn bool_empty(shape: Shape, device: &MultiDevice) -> MultiBoolTensor {
        unimplemented!()
    }
    // fn bool_zeros(shape: Shape, device: &MultiDevice) -> MultiIntTensor {
    //     unimplemented!()
    // }
    // fn bool_ones(shape: Shape, device: &MultiDevice) -> MultiIntTensor {
    //     unimplemented!()
    // }
    fn bool_slice(tensor: MultiBoolTensor, ranges: &[Range<usize>]) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_slice_assign(tensor: MultiBoolTensor, ranges: &[Range<usize>], value: MultiBoolTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_cat(tensors: Vec<MultiBoolTensor>, dim: usize) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_equal(lhs: MultiBoolTensor, rhs: MultiBoolTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_not(tensor: MultiBoolTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_and(lhs: MultiBoolTensor, rhs: MultiBoolTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_or(lhs: MultiBoolTensor, rhs: MultiBoolTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_into_int(tensor: MultiBoolTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn bool_into_float(tensor: MultiBoolTensor) -> MultiFloatTensor {
        ops_tensor_ret_float!(bool(tensor) => bool_into_float)
    }
    fn bool_swap_dims(tensor: MultiBoolTensor, dim1: usize, dim2: usize) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_permute(tensor: MultiBoolTensor, axes: &[usize]) -> MultiBoolTensor {
        unimplemented!()
    }
    fn bool_flip(tensor: MultiBoolTensor, axes: &[usize]) -> MultiBoolTensor {
        unimplemented!()
    }
    async fn bool_argwhere(tensor: MultiBoolTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn bool_expand(tensor: MultiBoolTensor, shape: Shape) -> MultiBoolTensor {
        unimplemented!()
    }
}
