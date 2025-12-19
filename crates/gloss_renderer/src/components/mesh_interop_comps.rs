use std::sync::Arc;

#[derive(Clone)]
pub struct VertsPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

#[derive(Clone)]
pub struct ColorsPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

#[derive(Clone)]
pub struct UVsPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

#[derive(Clone)]
pub struct NormalsPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

#[derive(Clone)]
pub struct TangentsPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

#[derive(Clone)]
pub struct FacesPyTensor {
    pub tensor: Arc<tch::Tensor>,
}

// SAFETY: You must ensure that tch::Tensor is never accessed from multiple threads simultaneously.
unsafe impl Send for VertsPyTensor {}
unsafe impl Sync for VertsPyTensor {}
unsafe impl Send for ColorsPyTensor {}
unsafe impl Sync for ColorsPyTensor {}
unsafe impl Send for UVsPyTensor {}
unsafe impl Sync for UVsPyTensor {}
unsafe impl Send for NormalsPyTensor {}
unsafe impl Sync for NormalsPyTensor {}
unsafe impl Send for TangentsPyTensor {}
unsafe impl Sync for TangentsPyTensor {}
unsafe impl Send for FacesPyTensor {}
unsafe impl Sync for FacesPyTensor {}
