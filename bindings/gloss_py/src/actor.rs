use gloss_hecs::Entity;
use gloss_renderer::{actor::Actor, scene::Scene};
// use numpy::Element;
use pyo3::{prelude::*, types::PyType};

//https://stackoverflow.com/questions/67412827/pyo3-deriving-frompyobject-for-enums
// https://stackoverflow.com/questions/75779700/retrieve-a-pyclass-from-an-attribute-of-an-arbitrary-pyany
//https://pyo3.rs/v0.19.2/conversions/traits
// https://github.com/PyO3/pyo3/pull/573
// https://github.com/PyO3/pyo3/issues/696

//mut
#[pyclass(name = "Entity", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, Copy)]
pub struct PyActorMut {
    pub actor: Actor,
    scene_ptr: *mut Scene, /* we only store a ptr to the scene because we pyo3 cannot deal with lifetime so EntityMut cannot have a reference to
                            * the world as it does on the rust side */
}
impl PyActorMut {
    pub fn new(entity: Entity, scene_ptr: *mut Scene) -> Self {
        Self {
            actor: Actor::from_entity(entity),
            scene_ptr,
        }
    }
    pub fn scene(&self) -> &Scene {
        unsafe { &*self.scene_ptr }
    }
    pub fn scene_mut(&mut self) -> &mut Scene {
        unsafe { &mut *self.scene_ptr }
    }
}
#[pymethods]
impl PyActorMut {
    #[pyo3(text_signature = "($self, component: Any) -> Entity")]
    pub fn insert(slf: PyRefMut<'_, Self>, pycomp: Py<PyAny>) -> PyRefMut<'_, Self> {
        Python::with_gil(|py| {
            let pyany = pycomp.bind(py);
            // Ideally we would use a function that takes PyEntityMut as parameter. However,
            // PyEntityMut cannot be shared between multiple crates due to this: https://github.com/PyO3/pyo3/issues/1444
            // therefore we pass around just the raw data for the entity and the scene_ptr
            let args = (slf.actor.entity.to_bits().get(), slf.scene_ptr as u64);
            let _ = pyany.call_method("insert_to_entity", args, None).unwrap();
        });
        slf
    }

    #[pyo3(text_signature = "($self, builder: Any) -> Entity")]
    pub fn insert_builder(slf: PyRefMut<'_, Self>, pybuilder: Py<PyAny>) -> PyRefMut<'_, Self> {
        //we can't use this because we may have other builders made by other libraries like Smpl-rs they can't create a PyEntityBuilder because of this https://github.com/PyO3/pyo3/issues/1444
        //so we just treat the builder as any other component and make the builder
        // insert itself into the entity let entity = slf.entity;
        // {
        //     let scene = slf.scene_mut();
        //     let _ = scene.world.insert(entity, builder.build());
        // }

        Self::insert(slf, pybuilder)
    }

    #[pyo3(text_signature = "($self, cls: Type[T]) -> T")]
    pub fn get(slf: PyRefMut<'_, Self>, cls: &Bound<'_, PyType>) -> Py<PyAny> {
        let result = Python::with_gil(|py| {
            //ideally we would use a function that takes PyEntityMut as parameter. However,
            // PyEntityMut cannot be shared between multiple crates due to this: https://github.com/PyO3/pyo3/issues/1444
            // therefore we pass around just the raw data for the entity and the scene_ptr
            let args = (slf.actor.entity.to_bits().get(), slf.scene_ptr as u64);
            let result = cls.call_method("get", args, None);
            let pyany_ref = result.unwrap();
            pyany_ref.into_py(py)
        });
        result
    }

    #[pyo3(text_signature = "($self, cls: Type[T]) -> bool")]
    pub fn has(slf: PyRefMut<'_, Self>, cls: &Bound<'_, PyType>) -> Py<PyAny> {
        let result = Python::with_gil(|py| {
            //ideally we would use a function that takes PyEntityMut as parameter. However,
            // PyEntityMut cannot be shared between multiple crates due to this: https://github.com/PyO3/pyo3/issues/1444
            // therefore we pass around just the raw data for the entity and the scene_ptr
            let args = (slf.actor.entity.to_bits().get(), slf.scene_ptr as u64);
            let result = cls.call_method("exists", args, None);
            let pyany_ref = result.unwrap();
            pyany_ref.into_py(py)
        });
        result
    }

    #[pyo3(text_signature = "($self, cls: Type[T]) -> bool")]
    pub fn remove(slf: PyRefMut<'_, Self>, cls: &Bound<'_, PyType>) {
        //ideally we would use a function that takes PyEntityMut as parameter. However,
        // PyEntityMut cannot be shared between multiple crates due to this: https://github.com/PyO3/pyo3/issues/1444
        // therefore we pass around just the raw data for the entity and the scene_ptr
        let args = (slf.actor.entity.to_bits().get(), slf.scene_ptr as u64);
        let _ = cls.call_method("remove", args, None).unwrap();
    }

    #[pyo3(text_signature = "($self) -> int")]
    pub fn entity(slf: PyRefMut<'_, Self>) -> u64 {
        slf.actor.entity.to_bits().get()
    }

    #[pyo3(text_signature = "($self) -> None")]
    pub fn apply_model_matrix(&mut self) {
        let mut actor = self.actor; // we need to clone because of the borrow checker and having mutable ref to both
                                    // self and scene
        let scene: &mut Scene = self.scene_mut();
        actor.apply_model_matrix(scene);
    }

    #[pyo3(text_signature = "($self, path: str) -> None")]
    pub fn save_obj(&self, path: &str) {
        let scene: &Scene = self.scene();
        self.actor.save_obj(scene, path);
    }

    #[pyo3(text_signature = "($self, path: str) -> None")]
    pub fn save_ply(&self, path: &str) {
        let scene: &Scene = self.scene();
        self.actor.save_ply(scene, path);
    }
}
