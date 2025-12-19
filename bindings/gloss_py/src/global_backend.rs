use gloss_burn_multibackend::global_backend::GlobalBackend;
use log::info;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "gloss_init_burn_backend")]
#[pyo3(text_signature = "(backend_name: string, idx_gpu: Optional[usize] = None) -> None")]
pub fn init_global_burn_backend(backend_name: &str, idx_gpu: Option<usize>) {
    let device = match backend_name {
        "candle" => GlobalBackend::Candle,
        "ndarray" => GlobalBackend::NdArray,
        "wgpu" => GlobalBackend::Wgpu,
        "torch_cpu" => GlobalBackend::TorchCpu,
        "torch_cuda" => GlobalBackend::TorchCuda(idx_gpu.expect("idx_gpu must be provided when using torch_cuda backend")),
        _ => {
            panic!("Unknown backend: {backend_name}");
        }
    };
    gloss_burn_multibackend::global_backend::init_global_burn_backend(device);
    info!("Gloss burn backend initialized to {device:?}");
}
