use crate::components::transport_config::PyTransportConfig;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::network::SceneSender;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;

#[pyclass(name = "SceneSender", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PySceneSender {
    pub inner: SceneSender,
}
#[pymethods]
impl PySceneSender {
    #[new]
    #[pyo3(text_signature = "(TransportConfig) -> SceneSender")]
    pub fn new(transport_config: PyTransportConfig) -> Self {
        PySceneSender {
            inner: SceneSender::new(transport_config.inner),
        }
    }

    pub fn try_connect_to_receiver(&mut self) {
        self.inner.try_connect_to_receiver();
    }

    pub fn start_listening(&mut self) {
        self.inner.start_listening().expect("Failed to start listening");
    }

    #[pyo3(text_signature = "($self) -> int")]
    pub fn get_ptr(&mut self) -> u64 {
        // println!("get ptr_viewer addr {:p}", &self.0);
        let obj_ptr: *mut SceneSender = &mut self.inner;
        obj_ptr as u64
    }
}
