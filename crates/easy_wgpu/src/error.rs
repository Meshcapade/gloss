#[cfg(feature = "burn-torch")]
#[derive(Debug, thiserror::Error)]
pub enum CudaInteropError {
    #[error("Tensor must be 4D (N,C,H,W), but got {0} dimensions")]
    InvalidTensorDim(usize),
    #[error("Provided tensor has to be have only one batch but it has first dim as {0}")]
    InvalidBatchSize(usize),
    #[error("Nr channels has to be 4 or less but it has {0} channels")]
    InvalidChannelSize(usize),
    #[error("Interop not allowed for tensor of type {0:?}")]
    InvalidTensorType(tch::Kind),
    #[error("Tensor has to be contiguous in memory")]
    InvalidNonContiguous,
}
