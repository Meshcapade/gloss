use crate::components::transport_config::PyTransportConfig;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::network::SceneReceiver;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;

#[pyclass(name = "SceneReceiver", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PySceneReceiver {
    pub inner: SceneReceiver,
}
#[pymethods]
impl PySceneReceiver {
    #[new]
    #[pyo3(text_signature = "(TransportConfig) -> SceneReceiver")]
    pub fn new(transport_config: PyTransportConfig) -> Self {
        PySceneReceiver {
            inner: SceneReceiver::new(transport_config.inner),
        }
    }

    pub fn connect_to_sender(&mut self) {
        self.inner.connect_to_sender().expect("Failed to connect to sender");
    }

    pub fn try_connect_to_sender(&mut self) -> bool {
        self.inner.connect_to_sender().is_ok()
    }

    #[pyo3(text_signature = "($self) -> int")]
    pub fn get_ptr(&mut self) -> u64 {
        // println!("get ptr_viewer addr {:p}", &self.0);
        let obj_ptr: *mut SceneReceiver = &mut self.inner;
        obj_ptr as u64
    }
}
