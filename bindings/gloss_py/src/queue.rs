use pyo3::prelude::*;
use wgpu;

#[pyclass(name = "Queue", module = "gloss", unsendable)] // it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyQueue {
    queue_ptr: *const wgpu::Queue,
}
impl std::ops::Deref for PyQueue {
    type Target = wgpu::Queue;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.queue_ptr }
    }
}
impl PyQueue {
    pub fn new(queue_ptr: *const wgpu::Queue) -> Self {
        PyQueue { queue_ptr }
    }
}
