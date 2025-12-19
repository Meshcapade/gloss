use crate::components::img_config::PyImgConfig;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{DiffuseImg, ImgConfig},
    scene::Scene,
};
use pyo3::prelude::*;

#[pyclass(name = "DiffuseImg", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyDiffuseImg {
    pub inner: DiffuseImg,
}
#[pymethods]
impl PyDiffuseImg {
    #[new]
    #[pyo3(text_signature = "(path: str, img_config: Optional[ImgConfig] = None) -> DiffuseImg")]
    pub fn new(path: &str, img_config: Option<PyImgConfig>) -> Self {
        Self {
            inner: DiffuseImg::new_from_path(path, &img_config.map_or(ImgConfig::default(), |x| x.inner)),
        }
    }
}
