use std::sync::Arc;

use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::VertsPyTensor, scene::Scene};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

#[pyclass(name = "VertsPyTensor", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyVertsPyTensor {
    pub inner: VertsPyTensor,
}
#[pymethods]
impl PyVertsPyTensor {
    #[new]
    #[pyo3(text_signature = "(tensor: PyTensor) -> Verts")]
    pub fn new(tensor: PyTensor) -> Self {
        let verts = VertsPyTensor { tensor: Arc::new(tensor.0) };
        Self { inner: verts }
    }
}
