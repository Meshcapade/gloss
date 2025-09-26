use gloss_burn_multibackend::global_backend::GlobalBackend;
use log::info;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "gloss_init_burn_backend")]
#[pyo3(text_signature = "(backend_name: string) -> None")]
pub fn init_global_burn_backend(backend_name: &str) {
    let device = match backend_name {
        "candle" => GlobalBackend::Candle,
        "ndarray" => GlobalBackend::NdArray,
        "wgpu" => GlobalBackend::Wgpu,
        _ => {
            panic!("Unknown backend: {backend_name}");
        }
    };
    gloss_burn_multibackend::global_backend::init_global_burn_backend(device);
    info!("Gloss burn backend initialized to {device:?}");
}
