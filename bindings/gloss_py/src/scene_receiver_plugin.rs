use gloss_renderer::network::receiver_plugin::SceneReceiverPlugin;
use gloss_renderer::plugin_manager::Plugins;
use gloss_renderer::scene::Scene;

use pyo3::prelude::*;

#[pyclass(name = "SceneReceiverPlugin", module = "gloss.plugins", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PySceneReceiverPlugin {
    inner: SceneReceiverPlugin,
}

#[pymethods]
impl PySceneReceiverPlugin {
    #[new]
    #[pyo3(text_signature = "(autorun: bool) -> SceneReceiverPlugin")]
    pub fn new(autorun: bool) -> Self {
        PySceneReceiverPlugin {
            inner: SceneReceiverPlugin::new(autorun),
        }
    }
    #[pyo3(text_signature = "($self, plugin_ptr_idx: int, scene_ptr_idx: int) -> None")]
    pub fn insert_plugin(&mut self, plugin_ptr_idx: u64, scene_ptr_idx: u64) {
        let plugin_ptr: *mut Plugins = plugin_ptr_idx as *mut Plugins;
        let plugin_list: &mut Plugins = unsafe { &mut *plugin_ptr };
        let scene_ptr: *mut Scene = scene_ptr_idx as *mut Scene;
        plugin_list.insert_plugin(&self.inner, unsafe { &mut *scene_ptr });
    }
}
