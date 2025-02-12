use gloss_hecs::{Entity, EntityBuilder};
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;

#[pyclass(name = "EntityBuilder", module = "gloss.builders", unsendable)]
pub struct PyEntityBuilder {
    pub inner: Option<EntityBuilder>, //need to have it as an option because we need to be able to move out inner using .take()
}
impl PyEntityBuilder {
    pub fn new(builder: EntityBuilder) -> Self {
        Self { inner: Some(builder) }
    }
}
#[pymethods]
impl PyEntityBuilder {
    #[pyo3(text_signature = "($self, entity_bits: int, scene_ptr_idx: int) -> None")]
    pub fn insert_to_entity(&mut self, entity_bits: u64, scene_ptr_idx: u64) {
        let entity = Entity::from_bits(entity_bits).unwrap();
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &mut Scene = unsafe { &mut *scene_ptr };

        scene.world.insert(entity, self.inner.take().unwrap().build()).ok();
    }
}
