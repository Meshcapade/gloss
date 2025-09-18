use std::ops::Range;

use burn::tensor::{ops::FloatTensorOps, Distribution, FloatDType, Shape, TensorData};

use crate::{
    backend::{MultiBackend, MultiDevice, NdArrayBackend, WgpuBackend},
    tensor::{MultiBoolTensor, MultiFloatTensor, MultiIntTensor},
};

#[allow(unused_variables)]
impl FloatTensorOps<Self> for MultiBackend {
    fn float_from_data(data: TensorData, device: &MultiDevice) -> MultiFloatTensor {
        ops_rest_device!(float(data ; device) => float_from_data)
    }
    fn float_random(shape: Shape, distribution: Distribution, device: &MultiDevice) -> MultiFloatTensor {
        ops_rest_device!(float(shape, distribution ; device) => float_random)
    }
    fn float_repeat_dim(tensor: MultiFloatTensor, dim: usize, times: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim, times) => float_repeat_dim)
    }
    fn float_zeros(shape: Shape, device: &MultiDevice) -> MultiFloatTensor {
        ops_rest_device!(float(shape ; device) => float_zeros)
    }
    fn float_ones(shape: Shape, device: &MultiDevice) -> MultiFloatTensor {
        ops_rest_device!(float(shape ; device) => float_ones)
    }
    async fn float_into_data(tensor: MultiFloatTensor) -> TensorData {
        // ops_tensor!(float(tensor) => float_into_data)
        match tensor {
            MultiFloatTensor::NdArray(t) => <NdArrayBackend as FloatTensorOps<NdArrayBackend>>::float_into_data(t).await,
            MultiFloatTensor::Wgpu(t) => <WgpuBackend as FloatTensorOps<WgpuBackend>>::float_into_data(t).await,
        }
    }
    fn float_device(tensor: &MultiFloatTensor) -> MultiDevice {
        match tensor {
            MultiFloatTensor::NdArray(t) => MultiDevice::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::float_device(t)),
            MultiFloatTensor::Wgpu(t) => MultiDevice::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::float_device(t)),
        }
    }
    fn float_to_device(tensor: MultiFloatTensor, device: &MultiDevice) -> MultiFloatTensor {
        match tensor {
            //current tensor is on ndarray
            MultiFloatTensor::NdArray(ref t) => match device {
                MultiDevice::NdArray(_) => {
                    // No need to move anything
                    tensor.clone()
                }
                MultiDevice::Wgpu(d) => {
                    //need to move ndarray to wgpu
                    let data = burn::tensor::try_read_sync(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::float_into_data(t.clone())).expect(
                        "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
                    );
                    MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::float_from_data(data, d))
                }
            },
            //current tensor is on wgpu
            MultiFloatTensor::Wgpu(ref t) => match device {
                MultiDevice::Wgpu(_) => {
                    // No need to move anything
                    tensor.clone()
                }
                MultiDevice::NdArray(d) => {
                    //need to move wgpu to ndarray
                    let data = burn::tensor::try_read_sync(<WgpuBackend as FloatTensorOps<WgpuBackend>>::float_into_data(t.clone())).expect(
                        "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
                    );
                    MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::float_from_data(data, d))
                }
            },
        }
    }
    fn float_empty(shape: Shape, device: &MultiDevice) -> MultiFloatTensor {
        ops_rest_device!(float(shape ; device) => float_empty)
    }
    fn float_add(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_add)
    }
    fn float_add_scalar(lhs: MultiFloatTensor, rhs: f32) -> MultiFloatTensor {
        ops_tensor_scalar!(float(lhs, rhs) => float_add_scalar)
    }
    fn float_sub(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_sub)
    }
    fn float_sub_scalar(lhs: MultiFloatTensor, rhs: f32) -> MultiFloatTensor {
        ops_tensor_scalar!(float(lhs, rhs) => float_sub_scalar)
    }
    fn float_mul(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_mul)
    }
    fn float_mul_scalar(lhs: MultiFloatTensor, rhs: f32) -> MultiFloatTensor {
        ops_tensor_scalar!(float(lhs, rhs) => float_mul_scalar)
    }
    fn float_div(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_div)
    }
    fn float_div_scalar(lhs: MultiFloatTensor, rhs: f32) -> MultiFloatTensor {
        ops_tensor_scalar!(float(lhs, rhs) => float_div_scalar)
    }
    fn float_remainder(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_remainder)
    }
    fn float_remainder_scalar(lhs: MultiFloatTensor, rhs: f32) -> MultiFloatTensor {
        ops_tensor_scalar!(float(lhs, rhs) => float_remainder_scalar)
    }
    fn float_matmul(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_matmul)
    }
    fn float_neg(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_neg)
    }
    fn float_recip(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_recip)
    }
    fn float_swap_dims(tensor: MultiFloatTensor, dim1: usize, dim2: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim1, dim2) => float_swap_dims)
    }
    fn float_reshape(tensor: MultiFloatTensor, shape: Shape) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, shape) => float_reshape)
    }
    fn float_gather(dim: usize, tensor: MultiFloatTensor, indices: MultiIntTensor) -> MultiFloatTensor {
        ops_dim_tensor_indices!(float(dim, tensor, indices) => float_gather)
    }
    fn float_scatter(dim: usize, tensor: MultiFloatTensor, indices: MultiIntTensor, value: MultiFloatTensor) -> MultiFloatTensor {
        ops_dim_tensor_indices_values!(float(dim, tensor, indices, value) => float_scatter)
    }
    fn float_select(tensor: MultiFloatTensor, dim: usize, indices: MultiIntTensor) -> MultiFloatTensor {
        ops_tensor_dim_indices!(float(tensor, dim, indices) => float_select)
    }
    fn float_select_assign(tensor: MultiFloatTensor, dim: usize, indices: MultiIntTensor, value: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_dim_indices_values!(float(tensor, dim, indices, value) => float_select_assign)
    }
    fn float_slice(tensor: MultiFloatTensor, ranges: &[Range<usize>]) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, ranges) => float_slice)
    }
    fn float_slice_assign(tensor: MultiFloatTensor, ranges: &[Range<usize>], value: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_other_values!(float(tensor, ranges, value) => float_slice_assign)
    }
    fn float_mask_where(tensor: MultiFloatTensor, mask: MultiBoolTensor, value: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }
    fn float_mask_fill(tensor: MultiFloatTensor, mask: MultiBoolTensor, value: f32) -> MultiFloatTensor {
        unimplemented!()
    }
    fn float_equal(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_equal_elem(lhs: MultiFloatTensor, rhs: f32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_greater(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_greater_elem(lhs: MultiFloatTensor, rhs: f32) -> MultiBoolTensor {
        ops_tensor_rest_ret_bool!(float(lhs, rhs) => float_greater_elem)
    }
    fn float_greater_equal(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_greater_equal_elem(lhs: MultiFloatTensor, rhs: f32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_lower(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_lower_elem(lhs: MultiFloatTensor, rhs: f32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_lower_equal(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_lower_equal_elem(lhs: MultiFloatTensor, rhs: f32) -> MultiBoolTensor {
        unimplemented!()
    }
    fn float_mean(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_mean)
    }
    fn float_sum(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_sum)
    }
    fn float_sum_dim(tensor: MultiFloatTensor, dim: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim) => float_sum_dim)
    }
    fn float_mean_dim(tensor: MultiFloatTensor, dim: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim) => float_mean_dim)
    }
    fn float_prod(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_prod)
    }
    fn float_prod_dim(tensor: MultiFloatTensor, dim: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim) => float_prod_dim)
    }
    fn float_argmax(tensor: MultiFloatTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn float_argmin(tensor: MultiFloatTensor, dim: usize) -> MultiIntTensor {
        unimplemented!()
    }
    fn float_max_dim(tensor: MultiFloatTensor, dim: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim) => float_max_dim)
    }
    fn float_max_dim_with_indices(tensor: MultiFloatTensor, dim: usize) -> (MultiFloatTensor, MultiIntTensor) {
        unimplemented!()
    }
    fn float_min_dim(tensor: MultiFloatTensor, dim: usize) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim) => float_min_dim)
    }
    fn float_min_dim_with_indices(tensor: MultiFloatTensor, dim: usize) -> (MultiFloatTensor, MultiIntTensor) {
        unimplemented!()
    }
    fn float_exp(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_exp)
    }
    fn float_log(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_log)
    }
    fn float_log1p(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_log1p)
    }
    fn float_powf_scalar(tensor: MultiFloatTensor, value: f32) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, value) => float_powf_scalar)
    }
    fn float_sqrt(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_sqrt)
    }
    fn float_abs(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_abs)
    }
    fn float_cos(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_cos)
    }
    fn float_sin(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_sin)
    }
    fn float_tanh(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_tanh)
    }
    fn float_round(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_round)
    }
    fn float_floor(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_floor)
    }
    fn float_ceil(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_ceil)
    }
    fn float_erf(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_erf)
    }
    fn float_cat(tensors: Vec<MultiFloatTensor>, dim: usize) -> MultiFloatTensor {
        assert!(!tensors.is_empty(), "Cannot concatenate an empty list of tensors");
        match &tensors[0] {
            MultiFloatTensor::NdArray(_) => {
                use crate::backend::NdArrayBackend;
                let inner: Vec<_> = tensors
                    .into_iter()
                    .map(|t| match t {
                        MultiFloatTensor::NdArray(inner) => inner,
                        _ => panic!("Mismatched tensor backends in float_cat: expected NdArray"),
                    })
                    .collect();
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::float_cat(inner, dim))
            }
            MultiFloatTensor::Wgpu(_) => {
                use crate::backend::WgpuBackend;
                let inner: Vec<_> = tensors
                    .into_iter()
                    .map(|t| match t {
                        MultiFloatTensor::Wgpu(inner) => inner,
                        _ => panic!("Mismatched tensor backends in float_cat: expected Wgpu"),
                    })
                    .collect();
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::float_cat(inner, dim))
            }
        }
    }
    fn float_clamp_min(tensor: MultiFloatTensor, min: f32) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, min) => float_clamp_min)
    }
    fn float_clamp_max(tensor: MultiFloatTensor, max: f32) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, max) => float_clamp_max)
    }
    fn float_clamp(tensor: MultiFloatTensor, min: f32, max: f32) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, min, max) => float_clamp)
    }
    fn float_into_int(tensor: MultiFloatTensor) -> MultiIntTensor {
        unimplemented!()
    }
    fn float_powf(lhs: MultiFloatTensor, rhs: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor_tensor!(float(lhs, rhs) => float_powf)
    }
    fn float_permute(tensor: MultiFloatTensor, axes: &[usize]) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, axes) => float_permute)
    }
    fn float_flip(tensor: MultiFloatTensor, axes: &[usize]) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, axes) => float_flip)
    }
    fn float_sign(tensor: MultiFloatTensor) -> MultiFloatTensor {
        ops_tensor!(float(tensor) => float_sign)
    }
    fn float_expand(tensor: MultiFloatTensor, shape: Shape) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, shape) => float_expand)
    }
    fn float_sort(tensor: MultiFloatTensor, dim: usize, descending: bool) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dim, descending) => float_sort)
    }
    fn float_sort_with_indices(tensor: MultiFloatTensor, dim: usize, descending: bool) -> (MultiFloatTensor, MultiIntTensor) {
        unimplemented!()
    }
    fn float_argsort(tensor: MultiFloatTensor, dim: usize, descending: bool) -> MultiIntTensor {
        unimplemented!()
    }
    fn float_cast(tensor: MultiFloatTensor, dtype: FloatDType) -> MultiFloatTensor {
        ops_tensor_rest!(float(tensor, dtype) => float_cast)
    }
}
