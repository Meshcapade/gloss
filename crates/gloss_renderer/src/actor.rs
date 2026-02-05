use gloss_geometry::geom;
use gloss_hecs::{CommandBuffer, Entity};
use log::{error, warn};

use crate::{
    components::{Colors, Faces, ModelMatrix, Normals, UVs, Verts},
    scene::Scene,
};

/// Contains a reference to an entity in the world so that any mesh processing
/// will directly affect the relevant entity in the world.
#[derive(Clone, Copy)]
pub struct Actor {
    pub entity: Entity,
}

impl Actor {
    pub fn new(name: &str, scene: &mut Scene) -> Self {
        //check if there is an entity with the same name, if no then spawn one
        let entity = scene.get_or_create_hidden_entity(name).entity();
        Self { entity }
    }

    pub fn from_entity(entity: Entity) -> Self {
        Self { entity }
    }

    pub fn apply_model_matrix(&mut self, scene: &mut Scene) {
        let mut command_buffer = CommandBuffer::new();

        {
            let Some(model_matrix) = scene.get_comp::<&ModelMatrix>(&self.entity).ok() else {
                warn!("No model matrix to apply");
                return;
            };

            //verts
            if let Ok(verts) = scene.get_comp::<&Verts>(&self.entity) {
                let new_verts = geom::transform_verts(&verts.0, &model_matrix.0);
                command_buffer.insert_one(self.entity, Verts(new_verts));
            }

            //normals
            if let Ok(normals) = scene.get_comp::<&Normals>(&self.entity) {
                let new_normals = geom::transform_vectors(&normals.0, &model_matrix.0);
                command_buffer.insert_one(self.entity, Normals(new_normals));
            }

            //model matrix is now identity
            command_buffer.insert_one(self.entity, ModelMatrix::default());
        }

        scene.world_mut().run_command_buffer(&mut command_buffer);
    }

    //methods that are not static and act directly on the entity
    pub fn save_obj(&self, scene: &Scene, path: &str) {
        let Some(verts) = scene.get_comp::<&Verts>(&self.entity).ok() else {
            error!("No vertices present on entity, cannot save as obj");
            return;
        };
        let faces = scene.get_comp::<&Faces>(&self.entity).ok();
        let normals = scene.get_comp::<&Normals>(&self.entity).ok();
        let uvs = scene.get_comp::<&UVs>(&self.entity).ok();

        //TODO modify data given the model matrix
        geom::save_obj(
            &verts.0,
            faces.as_ref().map(|v| &v.0),
            uvs.as_ref().map(|v| &v.0),
            normals.as_ref().map(|v| &v.0),
            path,
        );
    }

    pub fn save_ply(&self, scene: &Scene, path: &str) {
        let Some(verts) = scene.get_comp::<&Verts>(&self.entity).ok() else {
            error!("No vertices present on entity, cannot save as obj");
            return;
        };

        let faces = scene.get_comp::<&Faces>(&self.entity).ok();
        let normals = scene.get_comp::<&Normals>(&self.entity).ok();
        let uvs = scene.get_comp::<&UVs>(&self.entity).ok();
        let colors = scene.get_comp::<&Colors>(&self.entity).ok();

        //TODO modify data given the model matrix

        geom::save_ply(
            &verts.0,
            faces.as_ref().map(|v| &v.0),
            uvs.as_ref().map(|v| &v.0),
            normals.as_ref().map(|v| &v.0),
            colors.as_ref().map(|v| &v.0),
            path,
        );
    }
}
