extern crate nalgebra as na;

use crate::{
    camera::Camera,
    components::{
        Colors, DiffuseTex, Edges, EnvironmentMapGpu, Faces, LightEmit, ModelMatrix, Name, NormalTex, Normals, Renderable, RoughnessTex,
        ShadowCaster, ShadowMap, Tangents, UVs, Verts, VisLines, VisMesh, VisNormals, VisOutline, VisPoints, VisWireframe,
    },
    config::Config,
    scene::Scene,
};
use easy_wgpu::{
    gpu::Gpu,
    texture::{TexParams, Texture},
};
use gloss_geometry::geom::{self, PerVertexNormalsWeightingType};
use gloss_hecs::{Changed, CommandBuffer, Entity};
use log::{debug, warn};
/// Makes sure that the meshes are all with with correct components. Add model
/// matrices to the ones that will be rendered and dummy textures so that we can
/// use the same pipeline for all of them
pub struct PrePass {
    command_buffer: CommandBuffer, //defer insertions and deletion of scene entities for whenever we apply this command buffer
}
impl Default for PrePass {
    fn default() -> Self {
        Self::new()
    }
}
impl PrePass {
    pub fn new() -> Self {
        let command_buffer = CommandBuffer::new();
        Self { command_buffer }
    }

    pub fn add_auto_components(&mut self, gpu: &Gpu, scene: &mut Scene) {
        //this adds a lot of components automatically if they are needed
        self.add_model_matrix(scene);
        self.add_vertex_normals(scene);
        scene.world_mut().run_command_buffer(&mut self.command_buffer); //run the command buffer so now all entities actually have Normals, this is
                                                                        // necessary so that the add_tangents correctly computes tangents for all
                                                                        // entities which have Verts,Faces,Normals and UVs
        self.add_tangents(scene); //keep before adding dummy uvs so if we don't have uvs we just add some dumym
                                  // tangents
        self.add_dummy_uvs(scene);
        self.add_dummy_colors(scene);

        self.add_dummy_diffuse_tex(scene, gpu);
        self.add_dummy_normal_tex(scene, gpu);
        self.add_dummy_roughness_tex(scene, gpu);
        self.add_dummy_environment_map(scene, gpu);

        //auto vis options depending on the components
        self.add_vis_lines(scene);
        self.add_vis_wireframe(scene);
        self.add_vis_normals(scene);
        self.add_vis_points(scene);
        self.add_vis_mesh(scene);
        self.add_vis_outline(scene);
        scene.world_mut().run_command_buffer(&mut self.command_buffer); //in order to actually
                                                                        // create the model matrix
                                                                        // so that adding lights
                                                                        // works
    }

    pub fn run(
        &mut self,
        gpu: &Gpu,
        camera: &mut Camera,
        scene: &mut Scene,
        // width: u32,
        // height: u32,
        config: &mut Config,
    ) {
        self.begin_pass();

        //sanity checks
        self.check_entity_names(scene); //checks that all entities have names

        //automatic camera and lights only on first render

        //this adds a lot of components automatically if they are needed
        self.add_auto_components(gpu, scene);

        //if we have objects in the scene, make the config objects that were set to
        // auto, to a value that is concrete
        if !config.is_concrete() && scene.nr_renderables() != 0 {
            scene.make_concrete_config(config);
        }
        if config.is_concrete() && !config.is_consumed() {
            let (width, height) = camera.get_target_res(scene);
            scene.from_config(config, width, height);
            self.add_auto_components(gpu, scene); //whatever we now added in
                                                  // the scene like lights,etc
                                                  // might also need auto
                                                  // components
        }

        //add new objects the first time we render like the floor and lights
        //requires that the objects already have model_matrix
        // if scene.get_lights(false).is_empty() && scene.get_renderables(false).len()
        // != 0 {     scene.add_auto_lights();
        // }
        self.add_shadow_maps(scene, gpu);

        // if !scene.has_floor() && scene.get_renderables(false).len() != 0 {
        //     scene.add_floor();
        // }

        // //setup camera
        // if !scene.world.has::<PosLookat>(camera.entity).unwrap()
        //     && scene.get_renderables(false).len() != 0
        // {
        //     self.set_auto_cam(camera, scene);
        // }

        self.end_pass_sanity_check(scene);

        self.end_pass(scene);
    }

    fn begin_pass(&self) {}

    fn end_pass(&mut self, scene: &mut Scene) {
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn check_entity_names(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<()>().without::<&Name>();
        for (_entity, _comp) in query.iter() {
            warn!("Entity does not have a name, please assign name to all of them");
        }
    }

    fn add_vis_lines(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<Option<&Faces>>().with::<(&Verts, &Edges)>().without::<&VisLines>();
        for (entity, faces) in query.iter() {
            //we automatically enable vis_lines if we don't have faces so we have only
            // edges component
            let show_lines = faces.is_none();

            self.command_buffer.insert_one(
                entity,
                VisLines {
                    added_automatically: true,
                    show_lines,
                    ..Default::default()
                },
            );
        }
    }

    fn add_vis_wireframe(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<()>().with::<(&Verts, &Faces)>().without::<&VisWireframe>();
        for (entity, _comp) in query.iter() {
            self.command_buffer.insert_one(
                entity,
                VisWireframe {
                    added_automatically: true,
                    ..Default::default()
                },
            );
        }
    }

    fn add_vis_normals(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<()>().with::<(&Verts, &Normals)>().without::<&VisNormals>();
        for (entity, _comp) in query.iter() {
            self.command_buffer.insert_one(
                entity,
                VisNormals {
                    added_automatically: true,
                    ..Default::default()
                },
            );
        }
    }

    fn add_vis_points(&mut self, scene: &mut Scene) {
        let mut query = scene
            .world()
            .query::<(Option<&Faces>, Option<&Edges>)>()
            .with::<&Verts>()
            .without::<&VisPoints>();
        for (entity, (faces, edges)) in query.iter() {
            //we automatically enable vis_points if we don't have faces or edges component
            // and we have only verts
            let show_points = faces.is_none() && edges.is_none();

            self.command_buffer.insert_one(
                entity,
                VisPoints {
                    added_automatically: true,
                    show_points,
                    ..Default::default()
                },
            );
        }
    }

    fn add_vis_mesh(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<()>().with::<(&Verts, &Faces)>().without::<&VisMesh>();
        for (entity, _comp) in query.iter() {
            self.command_buffer.insert_one(
                entity,
                VisMesh {
                    added_automatically: true,
                    ..Default::default()
                },
            );
        }
    }

    fn add_vis_outline(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<&Name>().with::<(&Verts, &Faces)>().without::<&VisOutline>();
        for (entity, name) in query.iter() {
            // don't add outline to floor
            if name.0 == "floor" {
                continue;
            }
            self.command_buffer.insert_one(
                entity,
                VisOutline {
                    added_automatically: true,
                    ..Default::default()
                },
            );
        }
    }

    fn add_model_matrix(&mut self, scene: &mut Scene) {
        let mut query = scene.world().query::<()>().with::<&Renderable>().without::<&ModelMatrix>();
        for (entity, _comp) in query.iter() {
            self.command_buffer.insert_one(entity, ModelMatrix::default());
        }
    }

    fn add_vertex_normals(&mut self, scene: &mut Scene) {
        // We panic if prepass sees Wgpu backend tensors for now and if faces and verts
        // are on different backends
        let insert_normals = |entity: Entity, verts: &Verts, faces: &Faces, command_buffer: &mut CommandBuffer| {
            let normals = geom::compute_per_vertex_normals(&verts.0, &faces.0, &PerVertexNormalsWeightingType::Area);
            command_buffer.insert_one(entity, Normals(normals));
        };

        //add normals to all entities that have verts and faces but don't have Normals
        let mut query = scene.world().query::<(&Verts, &Faces)>().with::<&Renderable>().without::<&Normals>();
        for (entity, (verts, faces)) in query.iter() {
            insert_normals(entity, verts, faces, &mut self.command_buffer);
        }

        //also all the entities that have verts,faces AND normals but the normals don't
        // correspond in size to the verts this can happen when we update a
        // entity with a different Verts and Faces but don't update the Normals for some
        // reason
        let mut query = scene.world().query::<(&Verts, &Faces, &Normals)>().with::<&Renderable>();
        for (entity, (verts, faces, normals)) in query.iter() {
            if verts.0.nrows() != normals.0.nrows() {
                insert_normals(entity, verts, faces, &mut self.command_buffer);
            }
        }
    }
    #[allow(clippy::too_many_lines)]
    fn add_tangents(&mut self, scene: &mut Scene) {
        let insert_tangents = |entity: Entity, verts: &Verts, faces: &Faces, normals: &Normals, uvs: &UVs, command_buffer: &mut CommandBuffer| {
            let tangents = geom::compute_tangents(&verts.0, &faces.0, &normals.0, &uvs.0);
            command_buffer.insert_one(entity, Tangents(tangents));
        };

        //all the entities that have verts, faces, normals and uvs but NO tangents
        let mut query = scene
            .world()
            .query::<(&Verts, &Faces, &Normals, &UVs)>()
            .with::<&Renderable>()
            .without::<&Tangents>();
        for (entity, (verts, faces, normals, uvs)) in query.iter() {
            insert_tangents(entity, verts, faces, normals, uvs, &mut self.command_buffer);
        }

        //also all the entities that have verts,faces, normals,uvs AND tangents but the
        // tangents don't correspond in size to the verts this can happen when
        // we update a entity with a different Verts and Faces but don't update the
        // Tangents for some reason
        let mut query = scene.world().query::<(&Verts, &Faces, &Normals, &UVs, &Tangents)>().with::<&Renderable>();
        for (entity, (verts, faces, normals, uvs, tangents)) in query.iter() {
            if verts.0.nrows() != tangents.0.nrows() {
                insert_tangents(entity, verts, faces, normals, uvs, &mut self.command_buffer);
            }
        }

        //if we don't have real uvs then we add dummy tangents and the next function
        // will also just add dummy uvs
        let mut query = scene.world().query::<(&Verts, &Faces)>().with::<&Renderable>().without::<&UVs>();
        for (entity, (verts, _faces)) in query.iter() {
            let tangents = geom::compute_dummy_tangents(verts.0.nrows());
            self.command_buffer.insert_one(entity, Tangents(tangents));
        }

        // If we have verts, faces, NO uvs but we do have tangents, make sure the
        // tangents have the same size as verts
        let mut query = scene
            .world()
            .query::<(&Verts, &Faces, &Tangents)>()
            .with::<&Renderable>()
            .without::<&UVs>();

        for (entity, (verts, _faces, tangents)) in query.iter() {
            if verts.0.nrows() != tangents.0.nrows() {
                let tangents = geom::compute_dummy_tangents(verts.0.nrows());
                self.command_buffer.insert_one(entity, Tangents(tangents));
            }
        }
    }

    // fn add_dummy_uvs(&mut self, scene: &mut Scene) {
    //     //add dummy uvs if we don't have them
    //     let mut query = scene
    //         .world
    //         .query::<(&Verts, &Faces)>()
    //         .with::<&Renderable>()
    //         .without::<&UVs>();
    //     for (entity, (verts, _faces)) in query.iter() {
    //         let uvs = geom::compute_dummy_uvs(verts.0.nrows());
    //         let uvs_tensor = DynamicTensorFloat2D::NdArray(uvs);
    //         self.command_buffer.insert_one(entity, UVs(uvs_tensor));
    //     }

    //     //if we do have uvs, we make sure that they are the same size
    //     let mut query = scene
    //         .world
    //         .query::<(&Verts, &Faces, &UVs)>()
    //         .with::<&Renderable>();
    //     for (entity, (verts, _faces, uvs)) in query.iter() {
    //         if verts.0.nrows() != uvs.0.nrows() {
    //             let uvs = geom::compute_dummy_uvs(verts.0.nrows());
    //             let uvs_tensor = DynamicTensorFloat2D::NdArray(uvs);
    //             self.command_buffer.insert_one(entity, UVs(uvs_tensor));
    //         }
    //     }
    // }
    fn add_dummy_uvs(&mut self, scene: &mut Scene) {
        // Add dummy uvs if we don't have them
        let mut query = scene.world().query::<(&Verts, &Faces)>().with::<&Renderable>().without::<&UVs>();

        for (entity, (verts, _faces)) in query.iter() {
            let uvs = geom::compute_dummy_uvs(verts.0.nrows());
            self.command_buffer.insert_one(entity, UVs(uvs));
        }

        // If we do have uvs, make sure that they are the same size
        let mut query = scene.world().query::<(&Verts, &Faces, &UVs)>().with::<&Renderable>();

        for (entity, (verts, _faces, uvs)) in query.iter() {
            if verts.0.nrows() != uvs.0.nrows() {
                let uvs = geom::compute_dummy_uvs(verts.0.nrows());
                self.command_buffer.insert_one(entity, UVs(uvs));
            }
        }
    }

    // fn add_dummy_colors(&mut self, scene: &mut Scene) {
    //     //we add dummy colors if we don't have them
    //     let mut query = scene
    //         .world
    //         .query::<&Verts>()
    //         .with::<&Renderable>()
    //         .without::<&Colors>();
    //     for (entity, verts) in query.iter() {
    //         let colors = geom::compute_dummy_colors(verts.0.nrows());
    //         let colors_tensor = DynamicTensorFloat2D::NdArray(colors);
    //         self.command_buffer
    //             .insert_one(entity, Colors(colors_tensor));
    //     }
    //     //if we do have colors, we make sure they are the same size as verts
    //     let mut query = scene
    //         .world
    //         .query::<(&Verts, &Colors)>()
    //         .with::<&Renderable>();
    //     for (entity, (verts, colors)) in query.iter() {
    //         if verts.0.nrows() != colors.0.nrows() {
    //             let colors = geom::compute_dummy_colors(verts.0.nrows());
    //             let colors_tensor = DynamicTensorFloat2D::NdArray(colors);
    //             self.command_buffer
    //                 .insert_one(entity, Colors(colors_tensor));
    //         }
    //     }
    // }
    fn add_dummy_colors(&mut self, scene: &mut Scene) {
        // We add dummy colors if we don't have them
        let mut query = scene.world().query::<&Verts>().with::<&Renderable>().without::<&Colors>();

        for (entity, verts) in query.iter() {
            let colors = geom::compute_dummy_colors(verts.0.nrows());
            self.command_buffer.insert_one(entity, Colors(colors));
        }

        // If we do have colors, make sure they are the same size as verts
        let mut query = scene.world().query::<(&Verts, &Colors)>().with::<&Renderable>();

        for (entity, (verts, colors)) in query.iter() {
            if verts.0.nrows() != colors.0.nrows() {
                let colors = geom::compute_dummy_colors(verts.0.nrows());
                self.command_buffer.insert_one(entity, Colors(colors));
            }
        }
    }

    fn add_dummy_diffuse_tex(&mut self, scene: &mut Scene, gpu: &Gpu) {
        let mut query = scene.world().query::<()>().with::<&Renderable>().without::<&DiffuseTex>();
        for (entity, _comp) in query.iter() {
            let tex = Texture::create_default_texture(gpu.device(), gpu.queue());
            self.command_buffer.insert_one(entity, DiffuseTex(tex));
        }
    }

    fn add_dummy_normal_tex(&mut self, scene: &mut Scene, gpu: &Gpu) {
        let mut query = scene.world().query::<()>().with::<&Renderable>().without::<&NormalTex>();
        for (entity, _comp) in query.iter() {
            let tex = Texture::create_default_texture(gpu.device(), gpu.queue());
            self.command_buffer.insert_one(entity, NormalTex(tex));
        }
    }

    fn add_dummy_roughness_tex(&mut self, scene: &mut Scene, gpu: &Gpu) {
        let mut query = scene.world().query::<()>().with::<&Renderable>().without::<&RoughnessTex>();
        for (entity, _comp) in query.iter() {
            let tex = Texture::create_default_texture(gpu.device(), gpu.queue());
            self.command_buffer.insert_one(entity, RoughnessTex(tex));
        }
    }

    fn add_dummy_environment_map(&mut self, scene: &mut Scene, gpu: &Gpu) {
        //environment map
        if !scene.has_resource::<EnvironmentMapGpu>() {
            let env = EnvironmentMapGpu::new_dummy(gpu.device(), gpu.queue());
            scene.add_resource(env);
        }
    }

    //creates a new shadow map for all the lights that have the shadowcasting
    // component
    fn add_shadow_maps(&mut self, scene: &mut Scene, gpu: &Gpu) {
        let mut query = scene
            .world()
            .query::<(&ShadowCaster, Changed<ShadowCaster>, Option<&ShadowMap>)>()
            .with::<&LightEmit>();

        for (entity, (shadow_caster, is_shadow_changed, shadow_map)) in query.iter() {
            if is_shadow_changed || shadow_map.is_none() {
                debug!(
                    "creating shadow map, because is_shadow_changed {} or shadow_map.is_none() {}",
                    is_shadow_changed,
                    shadow_map.is_none()
                );
                let tex_depth = easy_wgpu::texture::Texture::new(
                    gpu.device(),
                    shadow_caster.shadow_res,
                    shadow_caster.shadow_res,
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    TexParams::default(),
                );
                // let tex_depth_moments = easy_wgpu::texture::Texture::new(
                //     gpu.device(),
                //     shadow_caster.shadow_res,
                //     shadow_caster.shadow_res,
                //     wgpu::TextureFormat::Rg32Float,
                //     wgpu::TextureUsages::RENDER_ATTACHMENT |
                // wgpu::TextureUsages::TEXTURE_BINDING, );
                self.command_buffer.insert_one(
                    entity,
                    ShadowMap {
                        tex_depth,
                        // tex_depth_moments,
                    },
                );
            }
        }
    }

    fn end_pass_sanity_check(&mut self, scene: &mut Scene) {
        //check that entities that have verts and colors, have the same number of
        // vertices and colors
        let mut query = scene.world().query::<(&Verts, &Colors)>();
        for (_entity, (verts, colors)) in query.iter() {
            assert!(
                verts.0.shape() == colors.0.shape(),
                "verts is {:?} and colors is{:?}",
                verts.0.shape(),
                colors.0.shape()
            );
        }
    }

    // fn set_auto_cam(&mut self, cam: &mut Camera, scene: &mut Scene) {
    //     let scale = scene.get_scale();
    //     let centroid = scene.get_centroid();

    //     //get all the automatic values we can get
    //     let position = centroid
    //         + na::Vector3::z_axis().scale(2.0 * scale)
    //         + na::Vector3::y_axis().scale(0.5 * scale);
    //     let near = (centroid - position).norm() * 0.01;
    //     let far = (centroid - position).norm() * 1000.0; //far plane can be quite
    // big. The near plane shouldn't be too tiny because it make the depth have very
    // little precision

    //     //add a pos lookat
    //     scene
    //         .world
    //         .insert(cam.entity, (PosLookat::new(position, centroid),))
    //         .unwrap();
    //     //modify also the near and far
    //     let mut cam_proj = scene.world.get::<&mut
    // Projection>(cam.entity).unwrap();     cam_proj.near = near;
    //     cam_proj.far = far;
    // }
}
