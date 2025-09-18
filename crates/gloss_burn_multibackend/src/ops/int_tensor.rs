use std::ops::Range;

use burn::tensor::{ops::IntTensorOps, Distribution, Shape, TensorData};

use crate::{
    backend::{MultiBackend, MultiDevice, NdArrayBackend, WgpuBackend},
    tensor::{MultiBoolTensor, MultiFloatTensor, MultiIntTensor},
};

#[allow(unused_variables)]
impl IntTensorOps<Self> for MultiBackend {
    fn int_from_data(data: TensorData, device: &MultiDevice) -> MultiIntTensor {
        ops_rest_device!(int(data ; device) => int_from_data)
    }
    fn int_repeat_dim(tensor: MultiIntTensor, dim: usize, times: usize) -> MultiIntTensor {
        ops_tensor_rest!(int(tensor, dim, times) => int_repeat_dim)
    }
    async fn int_into_data(tensor: MultiIntTensor) -> TensorData {
        match tensor {
            MultiIntTensor::NdArray(t) => <NdArrayBackend as IntTensorOps<NdArrayBackend>>::int_into_data(t).await,
            MultiIntTensor::Wgpu(t) => <WgpuBackend as IntTensorOps<WgpuBackend>>::int_into_data(t).await,
        }
    }
    fn int_to_device(tensor: MultiIntTensor, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_reshape(tensor: MultiIntTensor, shape: Shape) -> MultiIntTensor {
        ops_tensor_rest!(int(tensor, shape) => int_reshape)
    }
    fn int_device(tensor: &MultiIntTensor) -> MultiDevice {
        match tensor {
            MultiIntTensor::NdArray(t) => MultiDevice::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::int_device(t)),
            MultiIntTensor::Wgpu(t) => MultiDevice::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::int_device(t)),
        }
    }
    fn int_empty(shape: Shape, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_slice(tensor: MultiIntTensor, ranges: &[Range<usize>]) -> MultiIntTensor {
        ops_tensor_rest!(int(tensor, ranges) => int_slice)
    }
    fn int_slice_assign(tensor: MultiIntTensor, ranges: &[Range<usize>], value: MultiIntTensor) -> MultiIntTensor {
        ops_tensor_other_values!(int(tensor, ranges, value) => int_slice_assign)
    }
    fn int_cat(tensors: Vec<MultiIntTensor>, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    // fn int_matmul(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
    //     unimplemented!()
    // }
    fn int_equal(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_equal_elem(lhs: MultiIntTensor, rhs: i32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_greater(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_greater_elem(lhs: MultiIntTensor, rhs: i32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_greater_equal(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_greater_equal_elem(lhs: MultiIntTensor, rhs: i32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_lower(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_lower_elem(lhs: MultiIntTensor, rhs: i32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_lower_equal(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_lower_equal_elem(lhs: MultiIntTensor, rhs: i32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn int_add(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_add_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_sub(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_sub_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        ops_tensor_scalar!(int(lhs, rhs) => int_sub_scalar)
        // unimplemented!()
    }
    fn int_mul(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_mul_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_div(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_div_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_remainder(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_remainder_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_neg(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_zeros(shape: Shape, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_ones(shape: Shape, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_full(shape: Shape, fill_value: i32, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_sum(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_sum_dim(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_prod(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_prod_dim(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_mean(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_mean_dim(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_gather(dim: usize, tensor: MultiIntTensor, indices: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_scatter(dim: usize, tensor: MultiIntTensor, indices: MultiIntTensor, value: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_select(tensor: MultiIntTensor, dim: usize, indices: MultiIntTensor) -> MultiIntTensor {
        ops_tensor_dim_indices!(int(tensor, dim, indices) => int_select)
    }
    fn int_select_assign(tensor: MultiIntTensor, dim: usize, indices: MultiIntTensor, value: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_mask_where(tensor: MultiIntTensor, mask: MultiBoolTensor, source: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_mask_fill(tensor: MultiIntTensor, mask: MultiBoolTensor, value: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_argmax(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_argmin(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_max_dim(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_max_dim_with_indices(tensor: MultiIntTensor, dim: usize) -> (MultiIntTensor, MultiIntTensor) {
        unimplemented!()
    }
    fn int_min_dim(tensor: MultiIntTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_min_dim_with_indices(tensor: MultiIntTensor, dim: usize) -> (MultiIntTensor, MultiIntTensor) {
        unimplemented!()
    }
    fn int_clamp_min(tensor: MultiIntTensor, min: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_clamp_max(tensor: MultiIntTensor, max: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_clamp(tensor: MultiIntTensor, min: i32, max: i32) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_abs(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_into_float(tensor: MultiIntTensor) -> MultiFloatTensor {
        unimplemented!()
    }
    fn int_swap_dims(tensor: MultiIntTensor, dim1: usize, dim2: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_random(shape: Shape, distribution: Distribution, device: &MultiDevice) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_arange(range: Range<i64>, device: &MultiDevice) -> MultiIntTensor {
        ops_rest_device!(int(range ; device) => int_arange)
    }
    fn int_permute(tensor: MultiIntTensor, axes: &[usize]) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_flip(tensor: MultiIntTensor, axes: &[usize]) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_sign(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_expand(tensor: MultiIntTensor, shape: Shape) -> MultiIntTensor {
        ops_tensor_rest!(int(tensor, shape) => int_expand)
    }
    fn int_sort(tensor: MultiIntTensor, dim: usize, descending: bool) -> MultiIntTensor {
        unimplemented!()
    }
    fn int_argsort(tensor: MultiIntTensor, dim: usize, descending: bool) -> MultiIntTensor {
        unimplemented!()
    }
    fn bitwise_and(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_or(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_xor(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_not(tensor: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_and_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_or_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_xor_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_left_shift(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_right_shift(lhs: MultiIntTensor, rhs: MultiIntTensor) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_left_shift_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }

    fn bitwise_right_shift_scalar(lhs: MultiIntTensor, rhs: i32) -> MultiIntTensor {
        unimplemented!()
    }

    // fn int_cast(tensor: MultiIntTensor, dtype: DType) -> MultiIntTensor {
    //     unimplemented!()
    // }
}
