use pyo3::prelude::*;
use wgpu;

#[pyclass(name = "Device", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyDevice {
    obj_ptr: *const wgpu::Device,
}
impl std::ops::Deref for PyDevice {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.obj_ptr }
    }
}
impl PyDevice {
    pub fn new(obj_ptr: *const wgpu::Device) -> Self {
        PyDevice { obj_ptr }
    }
}
