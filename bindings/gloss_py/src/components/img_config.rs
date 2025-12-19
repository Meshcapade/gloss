use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::ImgConfig, scene::Scene};
use pyo3::prelude::*;

#[pyclass(name = "ImgConfig", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyImgConfig {
    pub inner: ImgConfig,
}
#[pymethods]
impl PyImgConfig {
    #[new]
    #[pyo3(signature = (keep_on_cpu=None, fast_upload=None, generate_mipmaps=None, mipmap_generation_cpu=None))]
    #[pyo3(
        text_signature = "(keep_on_cpu: Optional[bool] = None, fast_upload: Optional[bool] = None, generate_mipmaps: Optional[bool], mipmap_generation_cpu: Optional[bool]) -> ImgConfig"
    )]
    pub fn new(keep_on_cpu: Option<bool>, fast_upload: Option<bool>, generate_mipmaps: Option<bool>, mipmap_generation_cpu: Option<bool>) -> Self {
        let def = ImgConfig::default();

        let img_config = ImgConfig {
            keep_on_cpu: keep_on_cpu.unwrap_or(def.keep_on_cpu),
            fast_upload: fast_upload.unwrap_or(def.fast_upload),
            generate_mipmaps: generate_mipmaps.unwrap_or(def.generate_mipmaps),
            mipmap_generation_cpu: mipmap_generation_cpu.unwrap_or(def.mipmap_generation_cpu),
            // ..Default::default()
        };

        PyImgConfig { inner: img_config }
    }
}
