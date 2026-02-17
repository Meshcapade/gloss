#![allow(clippy::doc_markdown)]

use gloss_hecs::{
    CommandBuffer, Component, ComponentError, ComponentRef, DynamicBundle, Entity, EntityBuilder, EntityRef, NoSuchEntity, Query, QueryMut, World,
};
use log::{error, trace};

use crate::{
    actor::Actor,
    builders,
    camera::Camera,
    components::{
        BoundingBox, CamController, ColorsGPU, DiffuseImg, DiffuseTex, EdgesGPU, EnvironmentMapGpu, FacesGPU, ImgConfig, LightEmit, MeshColorType,
        MetalnessTex, ModelMatrix, Name, NormalTex, NormalsGPU, OnGui, PosLookat, Projection, ProjectionWithFov, Renderable, RoughnessTex,
        ShadowCaster, ShadowMap, TangentsGPU, UVsGPU, Verts, VertsGPU, VisLines, VisMesh, VisPoints,
    },
    config::{Config, FloorTexture, FloorType, LightConfig},
    light::Light,
    network::{ComponentTypeId, SceneSender},
    selector::Selectable,
};

use gloss_geometry::geom;
use gloss_utils::abi_stable_aliases::std_types::{RHashMap, RString};
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;
use nalgebra as na;

pub static GLOSS_FLOOR_NAME: &str = "floor";
pub static GLOSS_CAM_NAME: &str = "gloss_camera";

// TODO make parametric checkerboard
static CHECKERBOARD_BYTES: &[u8; 2324] = include_bytes!("../data/checkerboard.png");

/// Scene contains the ECS world and various functionality to interact with it.
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
#[allow(non_upper_case_globals, non_camel_case_types)]
pub struct Scene {
    world: World,
    name2entity: RHashMap<RString, Entity>,
    pub command_buffer: CommandBuffer, //defer insertions and deletion of scene entities for whenever we apply this command buffer
    entity_resource: Entity,           //unique entity that contains resources, so unique componentes
}
impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        let mut world = World::default();
        let name2entity = RHashMap::<RString, Entity>::new();
        let command_buffer = CommandBuffer::new();
        let entity_resource = world.spawn((Name("entity_resource".to_string()),));
        Self {
            world,
            name2entity,
            command_buffer,
            entity_resource,
        }
    }
    /// Gets the entity that contains all the resources
    pub fn get_entity_resource(&self) -> Entity {
        self.entity_resource
    }

    /// Creates a name that is guaranteed to be unused
    pub fn get_unused_name(&self) -> String {
        let mut cur_nr = self.get_renderables(false).len();
        loop {
            let name = String::from("ent_") + &cur_nr.to_string();
            let r_name = RString::from(name.clone());
            if !self.name2entity.contains_key(&r_name) {
                return name;
            }
            cur_nr += 1;
        }
    }

    /// Creates a entity with a name or gets the one that already exists with
    /// that concrete name You can keep adding components to this entity
    /// with .insert()
    pub fn get_or_create_hidden_entity(&mut self, name: &str) -> EntityMut<'_> {
        let r_name = RString::from(name.to_string());
        let entity_ref = self
            .name2entity
            .entry(r_name)
            .or_insert_with(|| self.world.spawn((Name(name.to_string()),))); //to insert a single component we use a tuple like (x,)
        EntityMut::new(&mut self.world, *entity_ref)
    }

    /// Creates a entity with a name or gets the one that already exists with
    /// that concrete name You can keep adding components to this entity
    /// with .insert() Also inserts a renderable component and a selectable component
    pub fn get_or_create_entity(&mut self, name: &str) -> EntityMut {
        let r_name = RString::from(name.to_string());
        let entity_ref = self
            .name2entity
            .entry(r_name)
            .or_insert_with(|| self.world.spawn((Name(name.to_string()), Renderable, Selectable)));

        //regardless of weather it exists or not we issue a spawn message
        if let Ok(mut sender) = self.world.get::<&mut SceneSender>(self.entity_resource) {
            sender.send_entity_spawn(name, true, false);
        }

        EntityMut::new(&mut self.world, *entity_ref)
    }

    /// Creates a entity that will have a Gui marker component indicating that it will only be visible in the Gui
    /// This is mostly helpful when trying to visualize images in a window
    /// in which case we will spawn a gui entity and add the DiffuseImg component
    /// The upload pass will take care of uploading the Img to a Texture and then the gui will display it
    pub fn get_or_create_gui_entity(&mut self, name: &str) -> EntityMut {
        let r_name = RString::from(name.to_string());
        let entity_ref = self
            .name2entity
            .entry(r_name)
            .or_insert_with(|| self.world.spawn((Name(name.to_string()), OnGui)));

        //regardless of weather it exists or not we issue a spawn message
        if let Ok(mut sender) = self.world.get::<&mut SceneSender>(self.entity_resource) {
            sender.send_entity_spawn(name, false, true);
        }

        EntityMut::new(&mut self.world, *entity_ref)
    }

    /// Despawns entity with a certain name and all it's components
    pub fn despawn_with_name(&mut self, name: &str) {
        //get the entity from the world if there is one, despawn it from the world and
        // remove it from our internal hashmap
        if let Some(entity) = self.get_entity_with_name(name) {
            let _ = self.world.despawn(entity);
            let r_name = RString::from(name.to_string());
            let _ = self.name2entity.remove(&r_name);
        }
    }

    pub fn despawn(&mut self, entity: Entity) {
        //if the entity has a name get it so we can remove it form our hashmap of
        // name2entity
        let name = self.get_comp::<&Name>(&entity).map(|x| RString::from(x.0.to_string()));
        if let Ok(name) = name {
            let _ = self.name2entity.remove(&name);
        }
        let _ = self.world.despawn(entity);
    }

    /// # Panics
    /// Will panic if no entity has that name
    pub fn get_entity_with_name(&self, name: &str) -> Option<Entity> {
        self.name2entity.get(name).copied()
    }

    /// # Panics
    /// Will panic if no entity has that name
    pub fn get_entity_mut_with_name(&mut self, name: &str) -> Option<EntityMut<'_>> {
        let entity_opt = self.name2entity.get(name);
        entity_opt.map(|ent| EntityMut::new(&mut self.world, *ent))
    }

    /// # Panics
    /// Will return None if no entity with id found
    pub fn find_entity_with_id(&mut self, id: u8) -> Option<EntityRef<'_>> {
        let entities = self.get_all_entities(false);

        for entity in entities {
            if u32::from(id) == entity.id() {
                let e_ref = self.world.entity(entity).unwrap();
                return Some(e_ref);
            }
        }
        None
    }

    /// # Panics
    /// Will panic if there is no camera added yet
    pub fn get_current_cam(&self) -> Option<Camera> {
        // TODO this has to be done better that with just a hard coded name. Maybe a
        // marker component on the camera?
        let entity_opt = self.name2entity.get(GLOSS_CAM_NAME);
        entity_opt.map(|ent| Camera::from_entity(*ent))
    }

    /// Use to create a unique component, similar to resources in Bevy
    /// # Panics
    /// Will panic if the entity that contains all the resources has not been
    /// created
    pub fn add_resource<T: gloss_hecs::Component>(&mut self, component: T) {
        self.world.insert_one(self.entity_resource, component).unwrap();
    }

    /// # Panics
    /// This function will panic if the entity that contains all the resources
    /// has not been created.
    ///
    /// # Errors
    /// This function will return an error if the required component does not
    /// exist.
    pub fn remove_resource<T: gloss_hecs::Component>(&mut self) -> Result<T, gloss_hecs::ComponentError> {
        self.world.remove_one::<T>(self.entity_resource) //DO NOT unwrap. this
                                                         // functions throws
                                                         // error if the
                                                         // component was
                                                         // already removed but
                                                         // we don't care if we
                                                         // do repeted
                                                         // remove_resource
    }

    /// # Panics
    /// Will panic if the entity that contains all the resources has not been
    /// created
    pub fn has_resource<T: gloss_hecs::Component>(&self) -> bool {
        self.world.has::<T>(self.entity_resource).unwrap()
    }

    /// Gets a resource which is a component shared between all entities
    /// Use with: scene.get_resource::<&mut Component>();
    /// # Errors
    /// Will error if the entity that contains all the resources has not been
    /// created
    pub fn get_resource<'a, T: gloss_hecs::ComponentRef<'a>>(&'a self) -> Result<<T as ComponentRef<'a>>::Ref, gloss_hecs::ComponentError> {
        self.world.get::<T>(self.entity_resource)
    }

    /// # Panics
    /// Will panic if the entity has not been created
    pub fn insert_if_doesnt_exist<T: gloss_hecs::Component + Default>(&mut self, entity: Entity) {
        if !self.world.has::<T>(entity).unwrap() {
            let _ = self.world.insert_one(entity, T::default());
        }
    }

    /// Generic function to get a component for a certain entity. Merely
    /// syntactic sugar. Use with: scene.get_comp::<&mut
    /// Component>(&entity);
    ///
    /// # Errors
    /// Will error if the component or the entity does not exist
    pub fn get_comp<'a, T: gloss_hecs::ComponentRef<'a>>(
        &'a self,
        entity: &Entity,
    ) -> Result<<T as ComponentRef<'a>>::Ref, gloss_hecs::ComponentError> {
        self.world.get::<T>(*entity)
    }

    /// # Panics
    /// Will panic if the entity does not have a name assigned
    pub fn get_lights(&self, sorted_by_name: bool) -> Vec<Entity> {
        let mut entities_with_name = Vec::new();
        for (entity_light, (name, _)) in self.world.query::<(&Name, &LightEmit)>().iter() {
            entities_with_name.push((entity_light, name.0.clone()));
        }
        //sort by name
        if sorted_by_name {
            entities_with_name.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        let entities = entities_with_name.iter().map(|x| x.0).collect();
        entities
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn get_all_entities(&self, sorted_by_name: bool) -> Vec<Entity> {
        let mut entities_with_name = Vec::new();
        for (entity, name) in self.world.query::<&Name>().iter() {
            entities_with_name.push((entity, name.0.clone()));
        }
        //sort by name
        if sorted_by_name {
            entities_with_name.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        let entities = entities_with_name.iter().map(|x| x.0).collect();
        entities
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn get_renderables(&self, sorted_by_name: bool) -> Vec<Entity> {
        let mut entities_with_name = Vec::new();
        for (entity, (name, _)) in self.world.query::<(&Name, &Renderable)>().iter() {
            entities_with_name.push((entity, name.0.clone()));
        }
        //sort by name
        if sorted_by_name {
            entities_with_name.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        let entities = entities_with_name.iter().map(|x| x.0).collect();
        entities
    }

    pub fn get_renderable_names(&self) -> Vec<String> {
        self.world
            .query::<(&Name, &Renderable)>()
            .iter()
            .map(|(_, (name, _))| name.0.clone())
            .collect()
    }

    // # Errors
    // Will return an error if the entity with the given name is not found.
    // # Panics
    // Will panic if no entity has that name
    // pub fn remove_renderable(&mut self, name: &str) -> Result<(), String> {
    //     let r_name = RString::from(name.to_string());
    //     if let Some(&entity) = self.name2entity.get(&r_name) {
    //         self.command_buffer.despawn(entity);
    //         self.command_buffer.run_on(&mut self.world);
    //         self.name2entity.remove(&r_name);
    //         Ok(())
    //     } else {
    //         Err(format!("Entity with name '{name}' not found"))
    //     }
    // }

    #[allow(clippy::cast_possible_truncation)]
    pub fn nr_renderables(&self) -> u32 {
        self.get_renderables(false).len() as u32
    }

    /// get two points that define the minimal point of the scene in all
    /// dimensions and the maximum point of the scene in all directions. These
    /// two points would form a rectangle containing the whole scene without the
    /// floor
    pub fn get_bounding_points(&self) -> (na::Point3<f32>, na::Point3<f32>) {
        let mut min_point_global = na::Point3::<f32>::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_point_global = na::Point3::<f32>::new(f32::MIN, f32::MIN, f32::MIN);

        for (_entity, (verts, model_matrix_opt, name, _)) in self.world.query::<(&Verts, Option<&ModelMatrix>, &Name, &Renderable)>().iter() {
            if name.0 == GLOSS_FLOOR_NAME {
                continue;
            }

            //Some meshes may not have a model matrix yet because the prepass hasn't run
            let model_matrix = if let Some(mm) = model_matrix_opt {
                mm.clone()
            } else {
                ModelMatrix::default()
            };

            trace!("scale for mesh {}", name.0);

            //get min and max vertex in obj coords
            let min_coord_vec: Vec<f32> = verts.0.column_iter().map(|c| c.min()).collect();
            let max_coord_vec: Vec<f32> = verts.0.column_iter().map(|c| c.max()).collect();
            let min_point = na::Point3::<f32>::from_slice(&min_coord_vec);
            let max_point = na::Point3::<f32>::from_slice(&max_coord_vec);

            //get the points to world coords
            let min_point_w = model_matrix.0 * min_point;
            let max_point_w = model_matrix.0 * max_point;

            //get the min/max between these points of this mesh and the global one
            min_point_global = min_point_global.inf(&min_point_w);
            max_point_global = max_point_global.sup(&max_point_w);
        }

        //for entities that have no verts, but have a bounding box, we compute the bounding points based on that.
        //this can happen for entities that are created directly on gpu, and they have only VertsGPU and no Verts components.
        for (_entity, (bbox, name, _)) in self.world.query::<(&BoundingBox, &Name, &Renderable)>().without::<&Verts>().iter() {
            if name.0 == GLOSS_FLOOR_NAME {
                continue;
            }
            min_point_global = min_point_global.inf(&bbox.min);
            max_point_global = max_point_global.sup(&bbox.max);
        }

        (min_point_global, max_point_global)
    }

    pub fn get_min_y(&self) -> f32 {
        let mut min_y_global = f32::MAX;

        for (_entity, (verts, model_matrix_opt, name, _)) in self.world.query::<(&Verts, Option<&ModelMatrix>, &Name, &Renderable)>().iter() {
            if name.0 == GLOSS_FLOOR_NAME {
                continue;
            }

            let model_matrix = if let Some(mm) = model_matrix_opt {
                mm.clone()
            } else {
                ModelMatrix::default()
            };

            //get min and max vertex in obj coords
            let v_world = geom::transform_verts(&verts.0, &model_matrix.0);
            let min_y_cur = v_world.column(1).min();
            min_y_global = min_y_global.min(min_y_cur);
        }

        //for entities that have no verts, but have a bounding box, we compute the min_y based on that.
        //this can happen for entities that are created directly on gpu, and they have only VertsGPU and no Verts components.
        for (_entity, (bbox, name, _)) in self.world.query::<(&BoundingBox, &Name, &Renderable)>().without::<&Verts>().iter() {
            if name.0 == GLOSS_FLOOR_NAME {
                continue;
            }
            min_y_global = min_y_global.min(bbox.min.y);
        }

        min_y_global
    }

    /// get the scale of the scene. useful for adding camera or lights
    /// automtically
    pub fn get_scale(&self) -> f32 {
        if self.get_renderables(false).is_empty() {
            error!("scale: no renderables, returning 1.0");
            return 1.0;
        }

        let (min_point_global, max_point_global) = self.get_bounding_points();

        //get scale as the maximum distance between any of the coordinates
        let scale = (max_point_global - min_point_global).abs().max();

        if scale.is_infinite() {
            1.0
        } else {
            scale
        }
    }

    pub fn get_centroid(&self) -> na::Point3<f32> {
        if self.get_renderables(false).is_empty() {
            error!("centroid: no renderables, returning 1.0");
            return na::Point3::<f32>::origin();
        }

        let (min_point_global, max_point_global) = self.get_bounding_points();

        //exactly the miggle between min and max
        min_point_global.lerp(&max_point_global, 0.5)
    }

    pub fn init_3_point_light(&self, light_config: &mut LightConfig, idx: usize, scale: f32, centroid: &na::Point3<f32>, flip_y: bool) {
        let (mut dir_movement, intensity_at_point) = match idx {
            0 => {
                let dir_movement = na::Vector3::new(0.5, 0.6, 0.5);
                let intensity_at_point = 2.9;
                (dir_movement, intensity_at_point)
            }
            1 => {
                let dir_movement = na::Vector3::new(-0.5, 0.6, 0.5);
                let intensity_at_point = 1.0;
                (dir_movement, intensity_at_point)
            }
            2 => {
                let dir_movement = na::Vector3::new(-0.1, 0.6, -0.5);
                let intensity_at_point = 3.5;
                (dir_movement, intensity_at_point)
            }
            //rest of light that are not in the 3 point light
            _ => {
                let dir_movement = na::Vector3::new(0.0, 0.6, 0.5);
                let intensity_at_point = 2.9;
                (dir_movement, intensity_at_point)
            }
        };

        if flip_y {
            dir_movement[1] *= -1.0;
        }

        dir_movement = dir_movement.normalize();
        let lookat = centroid;
        // let position = centroid + dir_movement * 3.5 * scale; //move the light
        // starting from the center in the direction by a certain amout so that in
        // engulfs the whole scene
        let position = centroid + dir_movement * 8.0 * scale; //move the light starting from the center in the direction by a certain amout
                                                              // so that in engulfs the whole scene
        let intensity = Light::intensity_for_point(&position, lookat, intensity_at_point);

        if light_config.position.is_none() {
            light_config.position = Some(position);
        }
        if light_config.lookat.is_none() {
            light_config.lookat = Some(*lookat);
        }
        if light_config.near.is_none() {
            light_config.near = Some(scale * 0.5);
        }
        if light_config.far.is_none() {
            light_config.far = Some(scale * 30.0);
        }
        if light_config.intensity.is_none() {
            light_config.intensity = Some(intensity);
        }
        if light_config.range.is_none() {
            light_config.range = Some(scale * 30.0);
        }
        if light_config.radius.is_none() {
            light_config.radius = Some(scale * 1.5);
        }
        if light_config.shadow_res.is_none() {
            light_config.shadow_res = Some(2048);
        }
        if light_config.shadow_bias_fixed.is_none() {
            light_config.shadow_bias_fixed = Some(2e-6);
        }
        if light_config.shadow_bias.is_none() {
            light_config.shadow_bias = Some(2e-6);
        }
        if light_config.shadow_bias_normal.is_none() {
            light_config.shadow_bias_normal = Some(2e-6);
        }
    }

    pub fn make_concrete_config(&self, config: &mut Config) {
        let scale = self.get_scale();
        let centroid = self.get_centroid();
        let min_y = self.get_min_y();

        //core
        let floor_scale_multiplier = 300.0;
        if config.core.floor_scale.is_none() {
            config.core.floor_scale = Some(scale * floor_scale_multiplier); //just make a realy large plane. At some point we should change it to be actualyl infinite
        }
        if config.core.floor_origin.is_none() {
            config.core.floor_origin = Some(na::Point3::<f32>::new(centroid.x, min_y, centroid.z));
        }
        if config.core.floor_uv_scale.is_none() {
            config.core.floor_uv_scale = Some(scale * floor_scale_multiplier * 1.4);
        }

        //camera
        let position = centroid + na::Vector3::z_axis().scale(2.0 * scale) + na::Vector3::y_axis().scale(0.5 * scale);
        if config.scene.cam.position.is_none() {
            config.scene.cam.position = Some(position);
        }
        if config.scene.cam.lookat.is_none() {
            config.scene.cam.lookat = Some(centroid);
        }
        if config.scene.cam.near.is_none() {
            let near = (centroid - position).norm() * 0.02;
            config.scene.cam.near = Some(near);
        }
        if config.scene.cam.far.is_none() {
            let far = (centroid - position).norm() * 50.0; //far plane can be quite big. The near plane shouldn't be too tiny because it
                                                           // make the depth have very little precision
            config.scene.cam.far = Some(far);
        }

        //render
        if config.render.distance_fade_center.is_none() {
            config.render.distance_fade_center = Some(centroid);
        }
        if config.render.distance_fade_start.is_none() {
            config.render.distance_fade_start = Some(scale * 1.0);
        }
        if config.render.distance_fade_end.is_none() {
            config.render.distance_fade_end = Some(scale * 8.5);
        }

        //lights
        // let three_point_lights = self.create_3_point_light_configs(scale, centroid);
        for (idx, light_config) in config.scene.lights.iter_mut().enumerate() {
            self.init_3_point_light(light_config, idx, scale, &centroid, config.render.flip_light_position_y);
        }

        //specify that it is now concrete so we don't rerun this function
        config.set_concrete();
    }

    /// # Panics
    /// Will panic if the camera entity has not yet been created
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::needless_update)]
    #[allow(clippy::too_many_lines)]
    pub fn from_config(&mut self, config: &mut Config, width: u32, height: u32) {
        //camera=============
        let cam = self.get_current_cam().expect("Camera should be created");
        //add a pos lookat if there is none
        if !self.world.has::<PosLookat>(cam.entity).unwrap() {
            self.world
                .insert(
                    cam.entity,
                    (
                        PosLookat::new(config.scene.cam.position.unwrap(), config.scene.cam.lookat.unwrap()),
                        CamController::new(
                            config.scene.cam.limit_max_dist,
                            config.scene.cam.limit_max_vertical_angle,
                            config.scene.cam.limit_min_vertical_angle,
                        ),
                    ),
                )
                .unwrap();
        }
        //add a projection if there is none
        if !self.world.has::<Projection>(cam.entity).unwrap() {
            let aspect_ratio = width as f32 / height as f32;
            self.world
                .insert(
                    cam.entity,
                    (Projection::WithFov(ProjectionWithFov {
                        aspect_ratio,
                        fovy: config.scene.cam.fovy,
                        near: config.scene.cam.near.unwrap(),
                        far: config.scene.cam.far.unwrap(),
                        ..Default::default()
                    }),),
                )
                .unwrap();
        }

        //create lights
        for (idx, light_config) in config.scene.lights.iter_mut().enumerate() {
            let entity = self
                .get_or_create_hidden_entity(("light_".to_owned() + idx.to_string().as_str()).as_str())
                .insert(PosLookat::new(light_config.position.unwrap(), light_config.lookat.unwrap()))
                .insert(Projection::WithFov(ProjectionWithFov {
                    aspect_ratio: 1.0,
                    fovy: light_config.fovy, //radians
                    near: light_config.near.unwrap(),
                    far: light_config.far.unwrap(),
                    ..Default::default()
                }))
                .insert(LightEmit {
                    color: light_config.color,
                    intensity: light_config.intensity.unwrap(),
                    range: light_config.range.unwrap(),
                    radius: light_config.radius.unwrap(),
                    ..Default::default()
                })
                .entity;
            //shadow
            let shadow_res = light_config.shadow_res.unwrap_or(0);
            if shadow_res != 0 {
                self.world
                    .insert_one(
                        entity,
                        ShadowCaster {
                            shadow_res: light_config.shadow_res.unwrap(),
                            shadow_bias_fixed: light_config.shadow_bias_fixed.unwrap(),
                            shadow_bias: light_config.shadow_bias.unwrap(),
                            shadow_bias_normal: light_config.shadow_bias_normal.unwrap(),
                        },
                    )
                    .ok();
            }
            // .insert(ShadowCaster { shadow_res: 2048 });
        }

        //floor
        if config.core.auto_add_floor {
            self.create_floor(config);
        }

        //specify that it is now consumed so we don't rerun this function again
        config.set_consumed();
    }

    pub fn create_floor(&mut self, config: &Config) {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let floor_builder = match config.core.floor_type {
            FloorType::Solid => builders::build_plane(
                config.core.floor_origin.unwrap(),
                na::Vector3::<f32>::new(0.0, 1.0, 0.0),
                config.core.floor_scale.unwrap(),
                config.core.floor_scale.unwrap(),
                false,
            ),
            FloorType::Grid => builders::build_grid(
                config.core.floor_origin.unwrap(),
                na::Vector3::<f32>::new(0.0, 1.0, 0.0),
                (config.core.floor_scale.unwrap() / 1.0) as u32,
                (config.core.floor_scale.unwrap() / 1.0) as u32,
                config.core.floor_scale.unwrap(),
                config.core.floor_scale.unwrap(),
                false,
            ),
        };

        #[allow(clippy::cast_possible_truncation)]
        let floor_ent = self
            .get_or_create_entity(GLOSS_FLOOR_NAME)
            .insert_builder(floor_builder)
            .insert(VisMesh {
                show_mesh: config.core.floor_type == FloorType::Solid,
                solid_color: na::Vector4::<f32>::new(0.08, 0.08, 0.08, 1.0), //106
                // perceptual_roughness: 0.75,
                perceptual_roughness: 0.70,
                uv_scale: config.core.floor_uv_scale.unwrap(),
                // metalness: 1.0,
                color_type: MeshColorType::Texture,
                ..Default::default()
            })
            .insert(VisPoints::default())
            .insert(VisLines {
                show_lines: config.core.floor_type == FloorType::Grid,
                line_color: na::Vector4::<f32>::new(0.2, 0.2, 0.2, 1.0),
                line_width: config.core.floor_grid_line_width,
                antialias_edges: true,
                ..Default::default()
            })
            .entity();

        if config.core.floor_texture == FloorTexture::Checkerboard {
            let texture_checkerboard = DiffuseImg::new_from_buf(
                CHECKERBOARD_BYTES,
                &ImgConfig {
                    mipmap_generation_cpu: true, /* we keep the generation of mipmaps on cpu because GPU one doesn't run on wasm due to webgl2 not
                                                  * allowing reading and writing from the same texture even if it's different mipmaps :( */
                    ..Default::default()
                },
            );
            let _ = self.world.insert_one(floor_ent, texture_checkerboard);
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn get_floor(&self) -> Option<Actor> {
        let ent_opt = self.name2entity.get(GLOSS_FLOOR_NAME);
        ent_opt.map(|ent| Actor::from_entity(*ent))
    }

    pub fn has_floor(&self) -> bool {
        self.name2entity.get(GLOSS_FLOOR_NAME).is_some()
    }

    pub fn create_camera(&mut self) {
        Camera::new(GLOSS_CAM_NAME, self, false);
    }

    pub fn remove_all_gpu_components(&mut self) {
        //TODO this is very brittle, need to somehow mark the gpu components somehow
        let mut command_buffer = CommandBuffer::new();
        for (entity, ()) in self.world.query::<()>().iter() {
            command_buffer.remove_one::<VertsGPU>(entity);
            command_buffer.remove_one::<UVsGPU>(entity);
            command_buffer.remove_one::<NormalsGPU>(entity);
            command_buffer.remove_one::<ColorsGPU>(entity);
            command_buffer.remove_one::<EdgesGPU>(entity);
            command_buffer.remove_one::<FacesGPU>(entity);
            command_buffer.remove_one::<TangentsGPU>(entity);
            command_buffer.remove_one::<DiffuseTex>(entity);
            command_buffer.remove_one::<NormalTex>(entity);
            command_buffer.remove_one::<MetalnessTex>(entity);
            command_buffer.remove_one::<RoughnessTex>(entity);
            command_buffer.remove_one::<EnvironmentMapGpu>(entity);
            command_buffer.remove_one::<ShadowMap>(entity);
        }

        command_buffer.run_on(&mut self.world);
    }

    //useful for when a network receiver creates entities in in the scene using a command buffer but doesn't update the name2entity. We can then use this to keep the name2entity mapping in sync
    //ideally we would make the scene.run_command buffer call this but it's difficult to work with the internals of the command buffer
    pub fn update_name2entity(&mut self, name_new_entity: &str) {
        for entity in self.get_all_entities(false) {
            let name = self.get_comp::<&Name>(&entity).unwrap().0.clone();
            if name == name_new_entity {
                self.name2entity.insert(name_new_entity.to_string().into(), entity);
            }
        }
    }

    //provide ONLY unmutable reference to the world.
    //we want all the mutable functions to go dirreclty thoguh scene (scene.insert_one rather than scene.world.insert_one) so that we can optionally send the components through the network
    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> WorldMut {
        WorldMut {
            world: &mut self.world,
            entity_resource: self.entity_resource,
        }
    }
}
//a mutable wrapper around the world functions . We do this so anything that mutates the world can optionally also send components through the network
pub struct WorldMut<'w> {
    world: &'w mut World,
    entity_resource: Entity,
}
impl WorldMut<'_> {
    pub fn run_command_buffer(&mut self, command_buffer: &mut CommandBuffer) {
        command_buffer.run_on(self.world);

        // self.clear();
    }
    pub fn query_mut<Q: Query>(&mut self) -> QueryMut<'_, Q> {
        self.world.query_mut::<Q>()
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn insert_one<T: Component>(&mut self, entity: Entity, component: T) -> Result<(), NoSuchEntity> {
        // Try to send the component over the network using the sender's component registry
        let type_id = ComponentTypeId::of::<T>();
        if let Ok(mut sender) = self.world.get::<&mut SceneSender>(self.entity_resource) {
            // Check if this component type is sendable
            if sender.is_component_sendable::<T>() {
                let entity_name = self.world.get::<&Name>(entity).ok().map(|n| n.0.clone()).unwrap_or_default();

                // Try to serialize the component using the sender's registry
                if let Some(serialized_component) = sender.try_serialize_component(&type_id, &component as &dyn std::any::Any) {
                    if let Err(e) = sender.send_component(&entity_name, serialized_component) {
                        log::warn!("Failed to send component: {e}");
                    }
                }
            }
        }

        // Insert the component locally
        self.world.insert_one(entity, component)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn insert(&mut self, entity: Entity, component: impl DynamicBundle) -> Result<(), NoSuchEntity> {
        self.world.insert(entity, component)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn remove_one<T: Component>(&mut self, entity: Entity) -> Result<T, ComponentError> {
        self.world.remove_one::<T>(entity)
    }

    pub fn set_trackers_changed(&mut self) {
        self.world.set_trackers_changed();
    }
    pub fn clear_trackers(&mut self) {
        self.world.clear_trackers();
    }
}

/// A mutable reference to a particular [`Entity`] and all of its components
pub struct EntityMut<'w> {
    world: &'w mut World,
    entity: Entity,
    // location: EntityLocation,
}

//similar to bevys entitymut https://docs.rs/bevy_ecs/latest/src/bevy_ecs/world/entity_ref.rs.html#183
impl<'w> EntityMut<'w> {
    pub(crate) fn new(world: &'w mut World, entity: Entity) -> Self {
        EntityMut { world, entity }
    }

    /// Inserts a component to this entity.
    /// This will overwrite any previous value(s) of the same component type.
    pub fn insert<T: Component>(&mut self, component: T) -> &mut Self {
        self.insert_bundle((component,))
    }

    /// Inserts a [`Bundle`] of components to the entity.
    /// This will overwrite any previous value(s) of the same component type.
    pub fn insert_bundle(&mut self, bundle: impl DynamicBundle) -> &mut Self {
        let _ = self.world.insert(self.entity, bundle);
        self
    }

    pub fn remove<T: Component>(&mut self) -> &mut Self {
        let _ = self.world.remove_one::<T>(self.entity);
        self
    }

    /// Inserts a [`EntityBuilder`] of components to the entity.
    /// This will overwrite any previous value(s) of the same component type.
    pub fn insert_builder(&mut self, mut builder: EntityBuilder) -> &mut Self {
        let _ = self.world.insert(self.entity, builder.build());
        self
    }
    /// Convenience function to get a component. Mostly useful for the python
    /// bindings. # Panics
    /// Will panic if the entity does not exist in the world
    pub fn get_comp<'a, T: gloss_hecs::ComponentRef<'a>>(&'a self) -> <T as ComponentRef<'a>>::Ref {
        let comp = self.world.get::<T>(self.entity).unwrap();
        comp
    }
    pub fn entity(&self) -> Entity {
        self.entity
    }
}
