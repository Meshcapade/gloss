#[cfg(feature = "burn-torch")]
use crate::PyGpu;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::DiffuseTex, scene::Scene};
use pyo3::prelude::*;
#[cfg(feature = "burn-torch")]
use pyo3_tch::PyTensor;

#[pyclass(name = "DiffuseTex", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyDiffuseTex {
    pub inner: DiffuseTex,
}
#[pymethods]
impl PyDiffuseTex {
    #[cfg(feature = "burn-torch")]
    // fn from_tensor(&mut self, tensor: PyTensor, device: &PyDevice, queue: &PyQueue, adapter: &PyAdapter) {
    fn from_tensor(&mut self, tensor: PyTensor, gpu: &PyGpu) {
        self.inner.0.from_tensor(&tensor, gpu.device(), gpu.queue(), gpu.adapter()).unwrap();
    }
}
