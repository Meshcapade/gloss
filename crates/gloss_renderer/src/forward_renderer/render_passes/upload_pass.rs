#![allow(clippy::cast_precision_loss)]

extern crate nalgebra as na;

// use crate::backend_specific_action;

use crate::{
    camera::Camera,
    components::{
        Colors, ColorsGPU, DiffuseImg, DiffuseTex, Edges, EdgesV1, EdgesV1GPU, EdgesV2, EdgesV2GPU, EnvironmentMap, EnvironmentMapGpu, Faces,
        FacesGPU, GpuAtrib, LightEmit, MeshColorType, Name, NormalImg, NormalTex, Normals, NormalsGPU, PosLookat, Projection, ProjectionWithFov,
        Renderable, RoughnessImg, RoughnessTex, ShadowCaster, Tangents, TangentsGPU, UVs, UVsGPU, Verts, VertsGPU, VisMesh,
    },
    config::RenderConfig,
    scene::Scene,
};

use easy_wgpu::{
    bind_group::BindGroupBuilder,
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
    gpu::Gpu,
    mipmap::RenderMipmapGenerator,
    texture::Texture,
};
use gloss_utils::tensor::{DynamicMatrixOps, DynamicTensorFloat2D, DynamicTensorOps};

use gloss_hecs::{Changed, CommandBuffer, Component, Entity};
use gloss_utils::numerical::{align, align_usz};
use log::{debug, info, warn};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use encase;

pub const MAX_NUM_LIGHTS: usize = 20; //lower than 20 causes wasm to throw error because the uniform is too small..
pub const MAX_NUM_SHADOWS: usize = 3; //HAS to be lower than MAX_NUM_LIGHTS.

pub fn index_vertices_from_edges(matrix: &na::DMatrix<f32>, v_indices: &na::DMatrix<u32>, col_id: usize) -> na::DMatrix<f32> {
    let index_slice = v_indices.column(col_id).into_owned();
    let indices: Vec<usize> = index_slice.iter().copied().map(|x| x as usize).collect();

    // Select rows based on indices
    let mut selected_rows = Vec::new();
    for &index in &indices {
        let row = matrix.row(index);
        selected_rows.push(row);
    }
    na::DMatrix::from_rows(&selected_rows)
}

/// Upload pass which uploads to GPU any data that is necessary, like vertex
/// buffers for meshes and camera parameters.
pub struct UploadPass {
    //all the buffers for per_frame stuff like light positions, cam parameters, etc. This are stuff that don't change from mesh to mesh
    per_frame_uniforms: PerFrameUniforms,
    mipmapper: Option<RenderMipmapGenerator>,
    //the local stuff that changes from mesh to mesh is allocated by each pass, because each pass might need something different from the mesh
    pub command_buffer: CommandBuffer, //defer insertions and deletion of scene entities for whenever we apply this command buffer
    pub staging_buffer: Option<Buffer>,
}

impl UploadPass {
    pub fn new(gpu: &Gpu, params: &RenderConfig) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<PerFrameSceneCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameCamCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameLightCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameParamsCPU>() % 16 == 0);

        let per_frame_uniforms = PerFrameUniforms::new(gpu);

        // cfg_if::cfg_if! {
        //     if #[cfg(target_arch = "wasm32")] {
        //         let mipmapper= None;
        //     }else{
        //         let mipmapper = Some(RenderMipmapGenerator::new_with_format_hints(
        //             gpu.device(),
        //             &[
        //                 wgpu::TextureFormat::Rgba8Unorm, //for normal maps
        //                 wgpu::TextureFormat::Rgba8UnormSrgb, //for diffuse maps
        //                 wgpu::TextureFormat::R8Unorm, //for roughness maps
        //             ],
        //         ));
        //     }
        // }

        let mipmapper = Some(RenderMipmapGenerator::new_with_format_hints(
            gpu.device(),
            &[
                wgpu::TextureFormat::Rgba8Unorm,     //for normal maps
                wgpu::TextureFormat::Rgba8UnormSrgb, //for diffuse maps
                wgpu::TextureFormat::R8Unorm,        //for roughness maps
            ],
        ));

        let command_buffer = CommandBuffer::new();

        let staging_buffer = if params.preallocated_staging_buffer_bytes != 0 {
            info!(
                "Using preallocated staging buffer with {} MB",
                params.preallocated_staging_buffer_bytes / (1024 * 1024)
            );
            Some(Buffer::new_empty(
                gpu.device(),
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
                Some("gloss_staging_buffer"),
                align_usz(params.preallocated_staging_buffer_bytes as usize, 256),
            ))
        } else {
            None
        };

        Self {
            per_frame_uniforms,
            mipmapper,
            command_buffer,
            staging_buffer,
        }
    }

    pub fn run(&mut self, gpu: &Gpu, camera: &Camera, scene: &mut Scene, render_params: &RenderConfig) -> &PerFrameUniforms {
        //upload each component (all of these are needed for the mesh)
        self.upload_v(gpu, scene);
        self.upload_e(gpu, scene);
        self.upload_f(gpu, scene);
        self.upload_uv(gpu, scene);
        self.upload_nv(gpu, scene);
        self.upload_t(gpu, scene);
        self.upload_c(gpu, scene);
        self.upload_textures(gpu, scene);

        self.upload_scene(gpu, scene);
        self.upload_cam(gpu, camera, scene);
        self.upload_lights(gpu, scene);
        self.upload_params(gpu, scene, render_params);

        &self.per_frame_uniforms
    }

    pub fn upload_textures(&mut self, gpu: &Gpu, scene: &mut Scene) {
        self.upload_diffuse_tex(gpu, scene);
        self.upload_normal_tex(gpu, scene);
        self.upload_roughness_tex(gpu, scene);
        self.upload_environment_map(gpu, scene);
    }

    #[allow(clippy::unnecessary_unwrap)] //I know it's unnecesary but it makes everything more compact the two cases
                                         // more explicit
    fn upload_dynamic_vertex_atrib<T, C: DynamicTensorOps<T> + Component, G: GpuAtrib + Component>(
        &mut self,
        entity: Entity,
        atrib: &C,
        atrib_gpu: Option<&mut G>,
        gpu: &Gpu,
        additional_usage: wgpu::BufferUsages, // scene: &mut Scene,
        label: &str,
    ) {
        // TODO: If DynamicTensor of Wgpu backend, do a direct buffer to buffer copy
        let verts_bytes = atrib.as_bytes();
        let size_bytes = verts_bytes.len();
        if atrib_gpu.is_none() || atrib_gpu.as_ref().unwrap().data_ref().size() != std::convert::TryInto::<u64>::try_into(size_bytes).unwrap() {
            // Allocate new memory for the GPU buffer if it doesn't exist or size has
            // changed
            let desc = wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &verts_bytes, // Use the raw data directly
                usage: additional_usage | wgpu::BufferUsages::COPY_DST,
            };

            let buf: wgpu::Buffer = gpu.device().create_buffer_init(&desc);

            // Insert the new GPU buffer component into the entity
            self.command_buffer
                .insert_one(entity, G::new_from(buf, u32::try_from(atrib.nrows()).unwrap()));
        } else {
            gpu.queue().write_buffer(
                atrib_gpu.unwrap().data_ref(),
                0,
                &verts_bytes, // Use the raw data directly
            );
        }
    }

    /// Functions for uploading each component of the mesh
    fn upload_v(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&Verts, Option<&mut VertsGPU>, Changed<Verts>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;

        for (ent, (verts, mut verts_gpu, changed_verts)) in query {
            if changed_verts {
                self.upload_dynamic_vertex_atrib(ent, &verts.0, verts_gpu.as_deref_mut(), gpu, usage, "verts");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_e(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(
                &Verts,
                &Edges,
                Option<&mut EdgesV1GPU>,
                Option<&mut EdgesV2GPU>,
                Changed<Verts>,
                Changed<Edges>,
            )>()
            .with::<&Renderable>();

        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (verts, edges, mut edges_v1_gpu, mut edges_v2_gpu, changed_verts, changed_edges)) in query {
            if changed_verts || changed_edges {
                let edges_v1_mat = index_vertices_from_edges(&verts.0.to_dmatrix(), &edges.0.to_dmatrix(), 0);
                let edges_v2_mat = index_vertices_from_edges(&verts.0.to_dmatrix(), &edges.0.to_dmatrix(), 1);

                let edges_v1_mat_tensor = DynamicTensorFloat2D::from_dmatrix(&edges_v1_mat);
                let edges_v2_mat_tensor = DynamicTensorFloat2D::from_dmatrix(&edges_v2_mat);
                let edges_v1 = EdgesV1(edges_v1_mat_tensor);
                let edges_v2 = EdgesV2(edges_v2_mat_tensor);

                self.upload_dynamic_vertex_atrib(ent, &edges_v1.0, edges_v1_gpu.as_deref_mut(), gpu, usage, "edges_v1");
                self.upload_dynamic_vertex_atrib(ent, &edges_v2.0, edges_v2_gpu.as_deref_mut(), gpu, usage, "edges_v2");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_f(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&Faces, Option<&mut FacesGPU>, Changed<Faces>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::INDEX;
        for (ent, (faces, mut faces_gpu, changed_faces)) in query {
            if changed_faces {
                self.upload_dynamic_vertex_atrib(ent, &faces.0, faces_gpu.as_deref_mut(), gpu, usage, "faces");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }
    fn upload_uv(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene.world.query_mut::<(&UVs, Option<&mut UVsGPU>, Changed<UVs>)>().with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (uvs, mut uvs_gpu, changed_uvs)) in query {
            if changed_uvs {
                self.upload_dynamic_vertex_atrib(ent, &uvs.0, uvs_gpu.as_deref_mut(), gpu, usage, "uv");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_nv(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&Normals, Option<&mut NormalsGPU>, Changed<Normals>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (normals, mut normals_gpu, changed_normals)) in query {
            if changed_normals {
                self.upload_dynamic_vertex_atrib(ent, &normals.0, normals_gpu.as_deref_mut(), gpu, usage, "normals");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_t(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&Tangents, Option<&mut TangentsGPU>, Changed<Tangents>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (tangents, mut tangents_gpu, changed_tangents)) in query {
            if changed_tangents {
                self.upload_dynamic_vertex_atrib(ent, &tangents.0, tangents_gpu.as_deref_mut(), gpu, usage, "tangents");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_c(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&Colors, Option<&mut ColorsGPU>, Changed<Colors>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (colors, mut colors_gpu, changed_colors)) in query {
            if changed_colors {
                self.upload_dynamic_vertex_atrib(ent, &colors.0, colors_gpu.as_deref_mut(), gpu, usage, "colors");
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_diffuse_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut modified_entities = Vec::new();
        {
            let mut query = scene
                .world
                .query::<(&mut DiffuseImg, Option<&mut DiffuseTex>, Changed<DiffuseImg>)>()
                .with::<&Renderable>();
            for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
                if changed_img && img.generic_img.cpu_img.is_some() {
                    debug!("DiffuseImg changed for entity {entity:?}");
                    let nr_channels = img.generic_img.img_ref().color().channel_count();
                    if nr_channels != 4 {
                        warn!("unoptimal use of memory: diffuse does not have 4 channels, it has {nr_channels}");
                    }
                    modified_entities.push(entity);
                    let is_srgb = true; //only true for diffuse since they are in srgb space in the png but we want to
                                        // sample linear colors
                                        // let tex = Texture::from_path(&img.0, gpu.device(), gpu.queue(), is_srgb);
                    let keep_on_cpu = img.generic_img.config.keep_on_cpu;
                    let staging_buffer = if img.generic_img.config.fast_upload {
                        None
                    } else {
                        //using slow upload through a preallocated staging buffer
                        if self.staging_buffer.is_none() {
                            warn!("The diffuse image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
                        }
                        self.staging_buffer.as_ref()
                    };

                    //either create a new tex or update the existing one
                    let mut tex_uploaded = false;
                    if let Some(mut existing_tex) = tex_opt {
                        let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
                        let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
                        let old_tex_extent = existing_tex.0.extent();
                        let old_format = existing_tex.0.texture.format();
                        if new_tex_format == old_format && new_tex_extent == old_tex_extent {
                            debug!("reusing diffuse tex");
                            existing_tex.0.update_from_img(
                                img.generic_img.img_ref(),
                                gpu.device(),
                                gpu.queue(),
                                is_srgb,
                                img.generic_img.config.generate_mipmaps,
                                img.generic_img.config.mipmap_generation_cpu,
                                staging_buffer,
                                self.mipmapper.as_ref(),
                            );
                            tex_uploaded = true;
                        }
                    }
                    //we create a new one if we couldn't update an existing one
                    if !tex_uploaded {
                        let tex = Texture::from_img(
                            img.generic_img.img_ref(),
                            gpu.device(),
                            gpu.queue(),
                            is_srgb,
                            img.generic_img.config.generate_mipmaps,
                            img.generic_img.config.mipmap_generation_cpu,
                            staging_buffer,
                            self.mipmapper.as_ref(),
                        );
                        self.command_buffer.insert_one(entity, DiffuseTex(tex));
                    }

                    if !keep_on_cpu {
                        // self.command_buffer.remove_one::<DiffuseImg>(entity);
                        let _ = img.generic_img.cpu_img.take();
                    }
                }
            }

            //set those meshes to actually visualize the mesh
            for entity in modified_entities {
                // let mut vis_mesh = scene.get_comp::<&mut VisMesh>(&entity);
                if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
                    if vis_mesh.added_automatically {
                        vis_mesh.color_type = MeshColorType::Texture;
                    }
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_normal_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut modified_entities = Vec::new();
        {
            let mut query = scene
                .world
                .query::<(&mut NormalImg, Option<&mut NormalTex>, Changed<NormalImg>)>()
                .with::<&Renderable>();
            for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
                if changed_img && img.generic_img.cpu_img.is_some() {
                    debug!("NormalImg changed for entity {entity:?}");
                    let nr_channels = img.generic_img.img_ref().color().channel_count();
                    if nr_channels != 4 {
                        warn!("unoptimal use of memory: normal does not have 4 channels, it has {nr_channels}");
                    }
                    modified_entities.push(entity);
                    let is_srgb = false; //only true for diffuse since they are in srgb space in the png but we want to
                                         // sample linear colors
                    let keep_on_cpu = img.generic_img.config.keep_on_cpu;
                    let staging_buffer = if img.generic_img.config.fast_upload {
                        None
                    } else {
                        //using slow upload through a preallocated staging buffer
                        if self.staging_buffer.is_none() {
                            warn!("The normal image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
                        }
                        self.staging_buffer.as_ref()
                    };

                    //either create a new tex or update the existing one
                    let mut tex_uploaded = false;
                    if let Some(mut existing_tex) = tex_opt {
                        let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
                        let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
                        let old_tex_extent = existing_tex.0.extent();
                        let old_format = existing_tex.0.texture.format();
                        if new_tex_format == old_format && new_tex_extent == old_tex_extent {
                            debug!("reusing normal tex");
                            existing_tex.0.update_from_img(
                                img.generic_img.img_ref(),
                                gpu.device(),
                                gpu.queue(),
                                is_srgb,
                                img.generic_img.config.generate_mipmaps,
                                img.generic_img.config.mipmap_generation_cpu,
                                staging_buffer,
                                self.mipmapper.as_ref(),
                            );
                            tex_uploaded = true;
                        }
                    }
                    //we create a new one if we couldn't update an existing one
                    if !tex_uploaded {
                        let tex = Texture::from_img(
                            img.generic_img.img_ref(),
                            gpu.device(),
                            gpu.queue(),
                            is_srgb,
                            img.generic_img.config.generate_mipmaps,
                            img.generic_img.config.mipmap_generation_cpu,
                            staging_buffer,
                            self.mipmapper.as_ref(),
                        );
                        self.command_buffer.insert_one(entity, NormalTex(tex));
                    }

                    if !keep_on_cpu {
                        // self.command_buffer.remove_one::<NormalImg>(entity);
                        let _ = img.generic_img.cpu_img.take();
                    }
                }
            }

            //set those meshes to actually visualize the mesh
            for entity in modified_entities {
                if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
                    if vis_mesh.added_automatically {
                        vis_mesh.color_type = MeshColorType::Texture;
                    }
                }
            }
        }

        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_roughness_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut modified_entities = Vec::new();
        {
            let mut query = scene
                .world
                .query::<(&mut RoughnessImg, Option<&mut RoughnessTex>, Changed<RoughnessImg>)>()
                .with::<&Renderable>();
            for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
                if changed_img && img.generic_img.cpu_img.is_some() {
                    debug!("RoughnessImg changed for entity {entity:?}");
                    let nr_channels = img.generic_img.img_ref().color().channel_count();
                    if nr_channels != 1 {
                        warn!("unoptimal use of memory: roughness does not have 1 channels, it has {nr_channels}");
                    }
                    modified_entities.push(entity);
                    let is_srgb = false; //only true for diffuse since they are in srgb space in the png but we want to
                                         // sample linear colors
                    let keep_on_cpu = img.generic_img.config.keep_on_cpu;
                    let staging_buffer = if img.generic_img.config.fast_upload {
                        None
                    } else {
                        //using slow upload through a preallocated staging buffer
                        if self.staging_buffer.is_none() {
                            warn!("The roughness image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
                        }
                        self.staging_buffer.as_ref()
                    };

                    //either create a new tex or update the existing one
                    let mut tex_uploaded = false;
                    if let Some(mut existing_tex) = tex_opt {
                        let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
                        let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
                        let old_tex_extent = existing_tex.0.extent();
                        let old_format = existing_tex.0.texture.format();
                        if new_tex_format == old_format && new_tex_extent == old_tex_extent {
                            debug!("reusing roughness tex");
                            existing_tex.0.update_from_img(
                                img.generic_img.img_ref(),
                                gpu.device(),
                                gpu.queue(),
                                is_srgb,
                                img.generic_img.config.generate_mipmaps,
                                img.generic_img.config.mipmap_generation_cpu,
                                staging_buffer,
                                self.mipmapper.as_ref(),
                            );
                            tex_uploaded = true;
                        }
                    }
                    //we create a new one if we couldn't update an existing one
                    if !tex_uploaded {
                        let tex = Texture::from_img(
                            img.generic_img.img_ref(),
                            gpu.device(),
                            gpu.queue(),
                            is_srgb,
                            img.generic_img.config.generate_mipmaps,
                            img.generic_img.config.mipmap_generation_cpu,
                            staging_buffer,
                            self.mipmapper.as_ref(),
                        );
                        self.command_buffer.insert_one(entity, RoughnessTex(tex));
                    }

                    if !keep_on_cpu {
                        // self.command_buffer.remove_one::<RoughnessImg>(entity);
                        let _ = img.generic_img.cpu_img.take();
                    }
                }
            }

            //set those meshes to actually visualize the mesh
            for entity in modified_entities {
                if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
                    if vis_mesh.added_automatically {
                        vis_mesh.color_type = MeshColorType::Texture;
                    }
                }
            }
        }

        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_environment_map(&mut self, gpu: &Gpu, scene: &mut Scene) {
        // if scene.has_resource::<EnvironmentMap>() {
        let query = scene.world.query_mut::<(&EnvironmentMap, Changed<EnvironmentMap>)>();
        for (entity, (env_map, changed_env)) in query {
            if changed_env {
                let diffue_raw_data = std::fs::read(env_map.diffuse_path.clone()).unwrap();
                let diffuse_reader = ktx2::Reader::new(diffue_raw_data.as_slice()).expect("Can't create diffuse_reader");
                let specular_raw_data = std::fs::read(env_map.specular_path.clone()).unwrap();
                let specular_reader = ktx2::Reader::new(specular_raw_data.as_slice()).expect("Can't create specular_reader");

                let diffuse_tex = EnvironmentMapGpu::reader2texture(&diffuse_reader, gpu.device(), gpu.queue());
                let specular_tex = EnvironmentMapGpu::reader2texture(&specular_reader, gpu.device(), gpu.queue());

                let env_map_gpu = EnvironmentMapGpu { diffuse_tex, specular_tex };

                // scene.add_resource(env_map);
                self.command_buffer.insert_one(entity, env_map_gpu);
            }
        }

        self.command_buffer.run_on(&mut scene.world);
    }

    fn upload_scene(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let entities_lights = scene.get_lights(false);
        let env_map = scene.get_resource::<&EnvironmentMapGpu>().unwrap();
        let environment_map_smallest_specular_mip_level = env_map.specular_tex.texture.mip_level_count() - 1;

        let per_frame_scene_data = PerFrameSceneCPU {
            nr_lights: u32::try_from(entities_lights.len()).unwrap(),
            environment_map_smallest_specular_mip_level,
            pad_1: 0,
            pad_2: 0,
        };

        self.per_frame_uniforms.scene_buf.push_cpu_chunk_packed(&per_frame_scene_data);
        self.per_frame_uniforms.scene_buf.upload_from_cpu_chunks(gpu.queue());
        self.per_frame_uniforms.scene_buf.reset_chunks_offset();
    }

    fn upload_cam(&mut self, gpu: &Gpu, camera: &Camera, scene: &mut Scene) {
        let pos_lookat = if let Ok(pos_lookat) = scene.world.get::<&mut PosLookat>(camera.entity) {
            pos_lookat.clone()
        } else {
            PosLookat::default()
        };

        let view_matrix = pos_lookat.view_matrix();
        let view_inv_matrix = pos_lookat.view_matrix_isometry().inverse().to_matrix();

        //get projection info but also take into account that if there are no entities
        // yet in the scene, there is also no projection matrix so we just set some
        // reasonable defaults
        let proj_matrix;
        let near;
        let far;
        if scene.world.has::<Projection>(camera.entity).unwrap() {
            proj_matrix = camera.proj_matrix_reverse_z(scene);
            (near, far) = camera.near_far(scene);
        } else {
            let proj = ProjectionWithFov::default();
            proj_matrix = proj.proj_matrix_reverse_z();
            (near, far) = (proj.near, proj.far);
        }
        let (width, height) = camera.get_target_res(scene);
        let aspect_ratio = width as f32 / height as f32;
        let proj_inv_matrix = proj_matrix.try_inverse().unwrap();

        let vp_matrix = proj_matrix * view_matrix;
        let pos_world = pos_lookat.position.coords;

        #[allow(clippy::cast_precision_loss)]
        let per_frame_cam_data = PerFrameCamCPU {
            view_matrix,
            view_inv_matrix,
            proj_matrix,
            proj_inv_matrix,
            vp_matrix,
            pos_world,
            near,
            far,
            aspect_ratio,
            width: width as f32,
            height: height as f32,
        };

        self.per_frame_uniforms.cam_buf.push_cpu_chunk_packed(&per_frame_cam_data);
        self.per_frame_uniforms.cam_buf.upload_from_cpu_chunks(gpu.queue());
        self.per_frame_uniforms.cam_buf.reset_chunks_offset();
    }

    fn upload_lights(&mut self, gpu: &Gpu, scene: &mut Scene) {
        self.per_frame_uniforms.idx_ubo2light.clear();

        let query = scene
            .world
            .query_mut::<(&Name, &PosLookat, &Projection, &LightEmit, Option<&ShadowCaster>)>();
        for (idx_light, (entity, (name, pos_lookat, proj, light_emit, shadow_caster))) in query.into_iter().enumerate() {
            let view_matrix = pos_lookat.view_matrix();
            let proj_matrix = match *proj {
                Projection::WithFov(ref proj) => proj.proj_matrix_reverse_z(),
                Projection::WithIntrinsics(_) => {
                    panic!("We don't deal with light that have projection as intrinsics")
                }
            };
            let (near, far) = proj.near_far();
            let vp_matrix = proj_matrix * view_matrix;
            let pos_world = pos_lookat.position.coords;
            let lookat_dir_world = pos_lookat.direction();

            let color = light_emit.color;
            let intensity = light_emit.intensity;
            let range = light_emit.range;
            let inverse_square_range = 1.0 / (range * range);
            let radius = light_emit.radius;
            let is_shadow_casting_bool = shadow_caster.is_some();
            let is_shadow_casting: u32 = u32::from(is_shadow_casting_bool);

            let shadow_bias_fixed = if let Some(shadow_caster) = shadow_caster {
                shadow_caster.shadow_bias_fixed
            } else {
                0.0
            };
            let shadow_bias = if let Some(shadow_caster) = shadow_caster {
                shadow_caster.shadow_bias
            } else {
                0.0
            };
            let shadow_bias_normal = if let Some(shadow_caster) = shadow_caster {
                shadow_caster.shadow_bias_normal
            } else {
                0.0
            };

            // let outer_angle = proj.fovy / 2.0; //we can use the fov as the angle because
            // we know the fov_y is the same as fov_x because the shadowmaps are  always
            // square let outer_angle = 1.57; //we can use the fov as the angle
            // because we know the fov_y is the same as fov_x because the shadowmaps are
            // always square
            let outer_angle = light_emit.outer_angle;
            let inner_angle = light_emit.inner_angle;

            //encase
            let per_frame_light_data = PerFrameLightCPU {
                view_matrix,
                proj_matrix,
                vp_matrix,
                pos_world,
                lookat_dir_world,
                color,
                intensity,
                range,
                inverse_square_range,
                radius,
                // spot_scale: 1.0,
                outer_angle,
                inner_angle,
                near,
                far,
                is_shadow_casting,
                shadow_bias_fixed,
                shadow_bias,
                shadow_bias_normal,
                pad_b: 1.0,
                pad_c: 1.0,
                pad_d: 1.0,
            };

            //push packed because we will expose it as an array inside the shader
            self.per_frame_uniforms.lights_buf.push_cpu_chunk_packed(&per_frame_light_data);

            //save also a mapping between light name and the idx in the whole light buffer
            self.per_frame_uniforms
                .light2idx_ubo
                .insert(name.0.clone(), u32::try_from(idx_light).unwrap());
            self.per_frame_uniforms.idx_ubo2light.push(entity);
        }

        self.per_frame_uniforms.lights_buf.upload_from_cpu_chunks(gpu.queue());
        self.per_frame_uniforms.lights_buf.reset_chunks_offset();
    }

    fn upload_params(&mut self, gpu: &Gpu, _scene: &mut Scene, render_params: &RenderConfig) {
        let per_frame_params_data = PerFrameParamsCPU {
            ambient_factor: render_params.ambient_factor,
            environment_factor: render_params.environment_factor,
            bg_color: render_params.bg_color,
            enable_distance_fade: u32::from(render_params.enable_distance_fade.unwrap_or(false)),
            distance_fade_center: render_params.distance_fade_center.unwrap_or_default().coords,
            distance_fade_start: render_params.distance_fade_start.unwrap_or(0.0),
            distance_fade_end: render_params.distance_fade_end.unwrap_or(0.0),
            apply_lighting: u32::from(render_params.apply_lighting),
            saturation: render_params.saturation,
            gamma: render_params.gamma,
            exposure: render_params.exposure,
            shadow_filter_method: render_params.shadow_filter_method as i32, // post_saturation: render_params.post_saturation,
            pad_b: 0.0,
            pad_c: 0.0,
            pad_d: 0.0,
        };

        self.per_frame_uniforms.params_buf.push_cpu_chunk_packed(&per_frame_params_data);
        self.per_frame_uniforms.params_buf.upload_from_cpu_chunks(gpu.queue());
        self.per_frame_uniforms.params_buf.reset_chunks_offset();
    }
}

#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct PerFrameSceneCPU {
    nr_lights: u32,
    environment_map_smallest_specular_mip_level: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_1: u32,
    pad_2: u32,
}
/// Contains camera data that will be sent to the GPU once a frame.
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct PerFrameCamCPU {
    view_matrix: na::Matrix4<f32>,
    view_inv_matrix: na::Matrix4<f32>,
    proj_matrix: na::Matrix4<f32>,
    proj_inv_matrix: na::Matrix4<f32>,
    vp_matrix: na::Matrix4<f32>, /* proj*view //order matter because we multiply from the left this matrix so we first do the view_matrix and then
                                  * proj */
    pos_world: na::Vector3<f32>,
    near: f32,
    far: f32,
    aspect_ratio: f32,
    width: f32,
    height: f32,
}
/// Contains light data that will be sent to the GPU once a frame.
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PerFrameLightCPU {
    view_matrix: na::Matrix4<f32>,
    proj_matrix: na::Matrix4<f32>,
    vp_matrix: na::Matrix4<f32>, /* proj*view //order matter because we multiply from the left this matrix so we first do the view_matrix and then
                                  * proj */
    pos_world: na::Vector3<f32>,
    lookat_dir_world: na::Vector3<f32>,
    color: na::Vector3<f32>,
    intensity: f32,
    range: f32,
    inverse_square_range: f32, //just 1/(range*range) because we don't want to compute this on gpu
    radius: f32,
    outer_angle: f32,
    inner_angle: f32,
    near: f32,
    far: f32,
    is_shadow_casting: u32, //should be bool but that is not host-sharable: https://www.w3.org/TR/WGSL/#host-shareable-types
    shadow_bias_fixed: f32,
    shadow_bias: f32,
    shadow_bias_normal: f32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_b: f32,
    pad_c: f32,
    pad_d: f32,
}
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct PerFrameParamsCPU {
    ambient_factor: f32,
    environment_factor: f32,
    bg_color: na::Vector4<f32>,
    enable_distance_fade: u32,
    distance_fade_center: na::Vector3<f32>,
    distance_fade_start: f32,
    distance_fade_end: f32,
    //color grading, applied before tonemapping
    apply_lighting: u32,
    saturation: f32,
    gamma: f32,
    exposure: f32,
    shadow_filter_method: i32,
    // post_saturation: f32, //applied after tonemapping
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_b: f32,
    pad_c: f32,
    pad_d: f32,
}

/// All the buffers that are the same for all meshes. Contains things like
/// camera parameters, lights, and global setting.
#[non_exhaustive]
pub struct PerFrameUniforms {
    scene_buf: Buffer,  //group 0, binding 0
    cam_buf: Buffer,    //group 0, binding 1
    lights_buf: Buffer, //group 0, binding 2
    params_buf: Buffer, //group 0, binding 3

    #[allow(dead_code)]
    //storing the samplers is not needed since the bind group consumes them but it makes things more explicit
    sampler_nearest: wgpu::Sampler, //group 0, binding 4
    #[allow(dead_code)]
    sampler_linear: wgpu::Sampler, //group 0, binding 5
    #[allow(dead_code)]
    sampler_comparison: wgpu::Sampler, //group 0, binding 6
    //we save also the bind_group since we will not need to recreate it (the buffer will not be reallocated)
    //the layout we keep as a associated function because we want to call it without the object.
    pub bind_group: wgpu::BindGroup,
    //misc
    pub light2idx_ubo: HashMap<String, u32>,
    pub idx_ubo2light: Vec<Entity>,
}
impl PerFrameUniforms {
    pub fn new(gpu: &Gpu) -> Self {
        let scene_buf = Buffer::new_empty(
            gpu.device(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("global_scene_uniform"),
            align_usz(std::mem::size_of::<PerFrameSceneCPU>(), 256),
        );
        //allocate buffers on gpu to hold the corresponding cpu data
        let cam_buf = Buffer::new_empty(
            gpu.device(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("global_cam_uniform"),
            align_usz(std::mem::size_of::<PerFrameCamCPU>(), 256),
        );
        //allocate space fo MAX_NUM_LIGHTS lights
        let lights_buf = Buffer::new_empty(
            gpu.device(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            // wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("global_lights_uniform"),
            MAX_NUM_LIGHTS * align_usz(std::mem::size_of::<PerFrameLightCPU>(), 256),
        );
        let params_buf = Buffer::new_empty(
            gpu.device(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("global_params_uniform"),
            align_usz(std::mem::size_of::<PerFrameParamsCPU>(), 256),
        );

        //samplers for nearest and linear
        let sampler_nearest = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_nearest"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let sampler_linear = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_linear"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_comparison = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_shadow_map"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::Greater),
            ..Default::default()
        });

        let layout = Self::create_layout(gpu);
        let bind_group = BindGroupBuilder::new()
            .label("per_frame_bind_group")
            .add_entry_buf(&scene_buf.buffer)
            .add_entry_buf(&cam_buf.buffer)
            .add_entry_buf(&lights_buf.buffer)
            .add_entry_buf(&params_buf.buffer)
            .add_entry_sampler(&sampler_nearest)
            .add_entry_sampler(&sampler_linear)
            .add_entry_sampler(&sampler_comparison)
            .build_bind_group(gpu.device(), &layout);

        Self {
            scene_buf,
            cam_buf,
            lights_buf,
            params_buf,
            sampler_nearest,
            sampler_linear,
            sampler_comparison,
            bind_group,
            light2idx_ubo: HashMap::new(),
            idx_ubo2light: Vec::new(),
        }
    }

    //keep as associated function so we can call it in the pipeline creation
    // without and object
    pub fn create_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
        let global_bind_group_layout = Self::build_layout_desc().into_bind_group_layout(gpu.device());
        global_bind_group_layout
    }

    /// # Panics
    /// Will panic if the texture is deleted while it's being copied
    pub fn build_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("locals_layout")
            //scene
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                false,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<PerFrameSceneCPU>()).unwrap(), 256))),
            )
            //cam
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                false,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<PerFrameCamCPU>()).unwrap(), 256))),
            )
            //light
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                false,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<PerFrameLightCPU>()).unwrap(), 256))),
            )
            //params
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                false,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<PerFrameParamsCPU>()).unwrap(), 256))),
            )
            //samplers
            .add_entry_sampler(wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::NonFiltering)
            .add_entry_sampler(wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering)
            .add_entry_sampler(wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Comparison)
            .build()
    }
}
