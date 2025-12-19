#[cfg(feature = "burn-torch")]
use crate::PyGpu;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::ColorsGPU, scene::Scene};
use pyo3::prelude::*;
#[cfg(feature = "burn-torch")]
use pyo3_tch::PyTensor;

#[pyclass(name = "ColorsGPU", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyColorsGPU {
    pub inner: ColorsGPU,
}
#[pymethods]
impl PyColorsGPU {
    #[cfg(feature = "burn-torch")]
    #[staticmethod]
    fn new_from_tensor(tensor: PyTensor, gpu: &PyGpu) -> Self {
        let colors_gpu = ColorsGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                &tensor,
                gpu.device(),
                gpu.queue(),
                gpu.adapter(),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("ColorsGPU buffer"),
            ),
            nr_vertices: tensor.size()[1] as u32,
        };
        Self { inner: colors_gpu }
    }
    #[cfg(feature = "burn-torch")]
    // fn from_tensor(&mut self, tensor: PyTensor, device: &PyDevice, queue: &PyQueue, adapter: &PyAdapter) {
    fn copy_from_tensor(&mut self, tensor: PyTensor, gpu: &PyGpu) {
        self.inner
            .buf
            .copy_from_tensor(&tensor, gpu.device(), gpu.queue(), gpu.adapter())
            .unwrap();
        self.inner.nr_vertices = tensor.size()[1] as u32;
    }
}
