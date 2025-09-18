use burn::tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateOptions, MaxPool1dWithIndices, MaxPool2dBackward,
    MaxPool2dWithIndices, ModuleOps,
};

use crate::tensor::MultiIntTensor;
use crate::{backend::MultiBackend, tensor::MultiFloatTensor};

#[allow(unused_variables)]
impl ModuleOps<Self> for MultiBackend {
    fn embedding(weights: MultiFloatTensor, indices: MultiIntTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn embedding_backward(weights: MultiFloatTensor, output: MultiFloatTensor, indices: MultiIntTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn conv1d(x: MultiFloatTensor, weight: MultiFloatTensor, bias: Option<MultiFloatTensor>, options: ConvOptions<1>) -> MultiFloatTensor {
        unimplemented!()
    }

    fn conv2d(x: MultiFloatTensor, weight: MultiFloatTensor, bias: Option<MultiFloatTensor>, options: ConvOptions<2>) -> MultiFloatTensor {
        unimplemented!()
    }

    fn conv3d(x: MultiFloatTensor, weight: MultiFloatTensor, bias: Option<MultiFloatTensor>, options: ConvOptions<3>) -> MultiFloatTensor {
        unimplemented!()
    }

    fn deform_conv2d(
        _x: MultiFloatTensor,
        _offset: MultiFloatTensor,
        _weight: MultiFloatTensor,
        _mask: Option<MultiFloatTensor>,
        _bias: Option<MultiFloatTensor>,
        _options: DeformConvOptions<2>,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn deform_conv2d_backward(
        _x: MultiFloatTensor,
        _offset: MultiFloatTensor,
        _weight: MultiFloatTensor,
        _mask: Option<MultiFloatTensor>,
        _bias: Option<MultiFloatTensor>,
        _out_grad: MultiFloatTensor,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!()
    }

    fn conv_transpose1d(
        x: MultiFloatTensor,
        weight: MultiFloatTensor,
        bias: Option<MultiFloatTensor>,
        options: ConvTransposeOptions<1>,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn conv_transpose2d(
        x: MultiFloatTensor,
        weight: MultiFloatTensor,
        bias: Option<MultiFloatTensor>,
        options: ConvTransposeOptions<2>,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn conv_transpose3d(
        x: MultiFloatTensor,
        weight: MultiFloatTensor,
        bias: Option<MultiFloatTensor>,
        options: ConvTransposeOptions<3>,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn avg_pool1d(x: MultiFloatTensor, kernel_size: usize, stride: usize, padding: usize, count_include_pad: bool) -> MultiFloatTensor {
        unimplemented!()
    }
    fn avg_pool2d(
        x: MultiFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn avg_pool2d_backward(
        x: MultiFloatTensor,
        grad: MultiFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> MultiFloatTensor {
        unimplemented!()
    }

    fn max_pool1d(x: MultiFloatTensor, kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> MultiFloatTensor {
        unimplemented!()
    }

    fn max_pool1d_with_indices(
        x: MultiFloatTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<MultiBackend> {
        unimplemented!()
    }

    fn max_pool2d(x: MultiFloatTensor, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2]) -> MultiFloatTensor {
        unimplemented!()
    }

    fn max_pool2d_with_indices(
        x: MultiFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<MultiBackend> {
        unimplemented!()
    }

    fn max_pool2d_with_indices_backward(
        x: MultiFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: MultiFloatTensor,
        indices: MultiIntTensor,
    ) -> MaxPool2dBackward<MultiBackend> {
        unimplemented!()
    }

    fn adaptive_avg_pool2d(x: MultiFloatTensor, output_size: [usize; 2]) -> MultiFloatTensor {
        unimplemented!()
    }

    fn adaptive_avg_pool2d_backward(x: MultiFloatTensor, grad: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn adaptive_avg_pool1d(x: MultiFloatTensor, output_size: usize) -> MultiFloatTensor {
        unimplemented!()
    }

    fn interpolate(x: MultiFloatTensor, output_size: [usize; 2], options: InterpolateOptions) -> MultiFloatTensor {
        unimplemented!()
    }

    fn interpolate_backward(x: MultiFloatTensor, grad: MultiFloatTensor, output_size: [usize; 2], options: InterpolateOptions) -> MultiFloatTensor {
        unimplemented!()
    }
}
