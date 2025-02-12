use gloss_py_macros::PtrDeref;
use gloss_renderer::plugin_manager::Plugins;
use pyo3::prelude::*;

#[pyclass(name = "PluginList", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(PtrDeref)]
pub struct PyPluginList {
    obj_ptr: *mut Plugins,
}
impl PyPluginList {
    pub fn new(obj_ptr: *mut Plugins) -> Self {
        PyPluginList { obj_ptr }
    }
}
#[pymethods]
impl PyPluginList {}
