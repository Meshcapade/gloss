use std::sync::Arc;

use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::ColorsPyTensor, scene::Scene};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

#[pyclass(name = "ColorsPyTensor", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyColorsPyTensor {
    pub inner: ColorsPyTensor,
}
#[pymethods]
impl PyColorsPyTensor {
    #[new]
    #[pyo3(text_signature = "(tensor: PyTensor) -> Colors")]
    pub fn new(tensor: PyTensor) -> Self {
        let colors = ColorsPyTensor { tensor: Arc::new(tensor.0) };
        Self { inner: colors }
    }
}
