use crate::{actor::PyActorMut, camera::PyCamera, scene::PyScene};

use gloss_renderer::{camera::Camera, config::Config, plugin_manager::Plugins, scene::Scene, viewer_dummy::ViewerDummy};

use pyo3::prelude::*;

#[pyclass(name = "ViewerDummy", module = "gloss", unsendable)] // it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyViewerDummy(pub ViewerDummy);
impl std::ops::Deref for PyViewerDummy {
    type Target = ViewerDummy;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[pymethods]
impl PyViewerDummy {
    #[new]
    #[pyo3(signature = (config_path=None))]
    #[pyo3(text_signature = "(config_path: Optional[str] = None) -> ViewerHeadless")]
    pub fn new(config_path: Option<&str>) -> Self {
        Self(ViewerDummy::new_with_config(&Config::new(config_path)))
    }
    #[pyo3(text_signature = "($self, name: str) -> Entity")]
    pub fn get_or_create_entity(&mut self, name: &str) -> PyActorMut {
        let scene: &mut Scene = &mut self.0.scene;
        let entity = scene.get_or_create_entity(name).entity();
        PyActorMut::new(entity, &mut self.0.scene)
    }
    #[pyo3(text_signature = "($self, component: Any) -> None")]
    pub fn add_resource(&mut self, pycomp: Py<PyAny>) {
        let mut pyscene = self.get_scene();
        pyscene.add_resource(pycomp);
    }
    #[pyo3(text_signature = "($self) -> Scene")]
    pub fn get_scene(&mut self) -> PyScene {
        let obj_ptr: *mut Scene = &mut self.0.scene;
        PyScene::new(obj_ptr)
    }
    #[pyo3(text_signature = "($self) -> Camera")]
    pub fn get_camera(&mut self) -> PyCamera {
        let obj_ptr: *mut Camera = &mut self.0.camera;
        PyCamera::new(obj_ptr, self.get_scene())
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn get_plugin_list_ptr(&mut self) -> u64 {
        let obj_ptr: *mut Plugins = &mut self.0.plugins;
        obj_ptr as u64
    }
    #[pyo3(text_signature = "($self, plugin: Any) -> None")]
    pub fn insert_plugin(mut slf: PyRefMut<'_, Self>, pycomp: Py<PyAny>) {
        // let obj_ptr: *mut Camera = &mut self.0.camera;
        Python::with_gil(|py| {
            let pyany = pycomp.bind(py);
            let args = (slf.get_plugin_list_ptr(),);
            let _ = pyany.call_method("insert_plugin", args, None).unwrap();
        });
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn run_manual_plugins(&mut self) {
        let v = &mut self.0;
        v.run_manual_plugins();
    }
}
