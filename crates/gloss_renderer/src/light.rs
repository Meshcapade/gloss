extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use crate::{
    components::{PosLookat, Projection, ShadowCaster, SpotLightBundle},
    scene::Scene,
};
use gloss_hecs::Entity;

/// Lights implements most of the functionality related to lights and inherits
/// some of the components from camera also.
pub struct Light {
    pub entity: Entity,
}

impl Light {
    #[allow(clippy::missing_panics_doc)] //really will never panic because the entity definitelly already exists in the
                                         // world
    pub fn new(name: &str, scene: &mut Scene) -> Self {
        let entity = scene
            .get_or_create_hidden_entity(name)
            .insert_bundle(SpotLightBundle::default())
            .insert(ShadowCaster::default())
            .entity();

        Self { entity }
    }

    pub fn from_entity(entity: Entity) -> Self {
        Self { entity }
    }

    /// # Panics
    /// Will panic if the ``PosLookat`` component does not exist for this entity
    pub fn view_matrix(&self, scene: &Scene) -> na::Matrix4<f32> {
        let pos_lookat = scene.get_comp::<&PosLookat>(&self.entity).unwrap();
        pos_lookat.view_matrix()
    }

    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix(&self, scene: &Scene) -> na::Matrix4<f32> {
        let proj = scene.get_comp::<&Projection>(&self.entity).unwrap();
        match *proj {
            Projection::WithFov(ref proj) => proj.proj_matrix(),
            Projection::WithIntrinsics(_) => {
                panic!("We don't deal with light that have projection as intrinsics")
            }
        }
    }

    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix_reverse_z(&self, scene: &Scene) -> na::Matrix4<f32> {
        let proj = scene.get_comp::<&Projection>(&self.entity).unwrap();
        match *proj {
            Projection::WithFov(ref proj) => proj.proj_matrix_reverse_z(),
            Projection::WithIntrinsics(_) => {
                panic!("We don't deal with light that have projection as intrinsics")
            }
        }
    }

    /// returns the intensity the light should have so that a certain point in
    /// space, after attenuating, receives a desired intensity of light
    pub fn intensity_for_point(light_pos: &na::Point3<f32>, point: &na::Point3<f32>, desired_intensity_at_point: f32) -> f32 {
        let dist = (light_pos - point).norm();
        let attenuation = 1.0 / (dist * dist);
        //power that the point receive is m_power*attenuation. If we want the power
        // there to be power: power=m_power*attenuation. Therefore the intensity of the
        // light is:
        desired_intensity_at_point / attenuation
    }
}
