use gloss_hecs::Entity;

use crate::scene::Scene;

pub trait LocalEntData {
    fn new(entity: Entity, scene: &Scene) -> Self;
}
