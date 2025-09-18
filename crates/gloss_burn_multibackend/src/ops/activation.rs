use burn::tensor::ops::ActivationOps;

use crate::{backend::MultiBackend, tensor::MultiFloatTensor};

#[allow(unused_variables)]
//activation ops
impl ActivationOps<Self> for MultiBackend {
    fn relu(tensor: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn gelu(tensor: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn gelu_backward(tensor: MultiFloatTensor, grad: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn sigmoid(tensor: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }

    fn log_sigmoid(tensor: MultiFloatTensor) -> MultiFloatTensor {
        unimplemented!()
    }
}
