/// Extension trait for turning CUDA error codes into `Result`.
use cust_raw::CUresult;
use std::error::Error;
use std::fmt;

/// Our boxed‐error type for CUDA failures.
#[derive(Debug)]
struct CudaError(CUresult);

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Leverage the enum's Debug to get the variant name, e.g. "CUDA_ERROR_OUT_OF_MEMORY"
        write!(f, "{:?}", self.0)
    }
}

impl Error for CudaError {}

/// Extension trait to map `CUresult` → Result<(), Box<dyn Error>>
pub trait CudaResultExt {
    /// Convert a CUDA `CUresult` into `Result<(), Box<dyn Error>>`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `CUresult` is not `CUDA_SUCCESS`; the error is a boxed
    /// `std::error::Error` describing the CUDA failure.
    fn to_result(self) -> Result<(), Box<dyn Error>>;
}

impl CudaResultExt for CUresult {
    #[inline]
    fn to_result(self) -> Result<(), Box<dyn Error>> {
        if self == CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Box::new(CudaError(self)))
        }
    }
}
