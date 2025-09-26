use pyo3::prelude::*;

#[pyclass(name = "Gpu", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyGpu {
    obj_ptr: *const easy_wgpu::gpu::Gpu,
}
impl std::ops::Deref for PyGpu {
    type Target = easy_wgpu::gpu::Gpu;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.obj_ptr }
    }
}
impl PyGpu {
    pub fn new(obj_ptr: *const easy_wgpu::gpu::Gpu) -> Self {
        PyGpu { obj_ptr }
    }
}
