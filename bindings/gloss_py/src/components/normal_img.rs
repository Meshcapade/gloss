use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{ImgConfig, NormalImg},
    scene::Scene,
};
use pyo3::prelude::*;

#[pyclass(name = "NormalImg", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyNormalImg {
    pub inner: NormalImg,
}
#[pymethods]
impl PyNormalImg {
    #[new]
    #[pyo3(text_signature = "(path: str) -> NormalImg")]
    pub fn new(path: &str) -> Self {
        Self {
            inner: NormalImg::new_from_path(path, &ImgConfig::default()),
        }
    }
}
