use gloss_py_macros::PtrDeref;
use gloss_renderer::scene::Scene;
use pyo3::{prelude::*, types::PyType};

use crate::actor::PyActorMut;
#[pyclass(name = "Scene", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(PtrDeref)]
pub struct PyScene {
    pub obj_ptr: *mut Scene,
}
impl PyScene {
    pub fn new(obj_ptr: *mut Scene) -> Self {
        PyScene { obj_ptr }
    }
}
#[pymethods]
impl PyScene {
    #[pyo3(text_signature = "($self, name: str) -> Entity")]
    pub fn get_or_create_entity(&mut self, name: &str) -> PyActorMut {
        let scene_native: &mut Scene = self;
        let entity = scene_native.get_or_create_entity(name).entity();
        PyActorMut::new(entity, self.obj_ptr)
    }
    #[pyo3(text_signature = "($self) -> List[str]")]
    pub fn get_renderable_names(&mut self) -> Vec<String> {
        let scene_native: &mut Scene = self;
        scene_native.get_renderable_names()
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn ptr_idx(&mut self) -> u64 {
        self.obj_ptr as u64
    }
    /// Removes a renderable entity from the scene by its name.
    ///
    /// # Errors
    /// Will return an error if the entity with the given name cannot be found
    /// or if there is an issue with the underlying scene operations.
    #[pyo3(text_signature = "($self, name: str) -> None")]
    pub fn remove_entity(&mut self, name: &str) -> PyResult<()> {
        let scene_native: &mut Scene = self;
        scene_native.despawn_with_name(name);
        Ok(())
    }
    #[pyo3(text_signature = "($self, component: Any) -> None")]
    pub fn add_resource(&mut self, pycomp: Py<PyAny>) {
        let scene: &mut Scene = self;
        let entity = scene.get_entity_resource();

        Python::with_gil(|py| {
            let pyany = pycomp.bind(py);
            let args = (entity.to_bits().get(), self.obj_ptr as u64);
            let _result = pyany.call_method("insert_to_entity", args, None).unwrap();
        });
    }
    #[pyo3(text_signature = "($self, cls: Type[T]) -> T")]
    pub fn get_resource(slf: PyRefMut<'_, Self>, cls: &Bound<'_, PyType>) -> Py<PyAny> {
        let result = Python::with_gil(|py| {
            let entity = slf.get_entity_resource();
            let args = (entity.to_bits().get(), slf.obj_ptr as u64);
            let result = cls.call_method("get", args, None);
            let pyany_ref = result.unwrap();
            pyany_ref.into_py(py)
        });
        result
    }
}
