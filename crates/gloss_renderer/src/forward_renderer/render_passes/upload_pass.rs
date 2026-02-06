#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)] //needed to supress warning of the check function of encase. To be removed after encase 0.11
extern crate nalgebra as na;

// use crate::backend_specific_action;

#[cfg(feature = "burn-torch")]
use crate::components::{ColorsPyTensor, FacesPyTensor, NormalsPyTensor, TangentsPyTensor, UVsPyTensor, VertsPyTensor};
use crate::{
    camera::Camera,
    components::{
        Colors, ColorsGPU, CpuAtrib, DiffuseImg, DiffuseTex, Edges, EdgesV1, EdgesV1GPU, EdgesV2, EdgesV2GPU, EnvironmentMap, EnvironmentMapGpu,
        Faces, FacesGPU, GenericImageGetter, GpuAtrib, LightEmit, MeshColorType, Name, NormalImg, NormalTex, Normals, NormalsGPU, PosLookat,
        Projection, ProjectionWithFov, Renderable, RoughnessImg, RoughnessTex, ShadowCaster, Tangents, TangentsGPU, TextureGetter, UVs, UVsGPU,
        Verts, VertsGPU, VisMesh,
    },
    config::RenderConfig,
    scene::Scene,
};

// use abi_stable::reexports::SelfOps;
use easy_wgpu::{
    bind_group::BindGroupBuilder,
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
    gpu::Gpu,
    mipmap::RenderMipmapGenerator,
    texture::Texture,
};

use gloss_hecs::{Changed, CommandBuffer, Component, Entity};
use gloss_utils::numerical::{align, align_usz};
use log::{debug, info, warn};
#[cfg(not(target_arch = "wasm32"))]
use pollster::FutureExt;
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::sync::mpsc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures;
use wgpu::util::DeviceExt;

use encase;

pub const MAX_NUM_LIGHTS: usize = 20; //lower than 20 causes wasm to throw error because the uniform is too small..
pub const MAX_NUM_SHADOWS: usize = 3; //HAS to be lower than MAX_NUM_LIGHTS.

// pub struct PendingDiffuseUpload;
// pub struct PendingNormalUpload;
// pub struct PendingRoughnessUpload;

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

trait TextureUploadable {
    type Img: Component + Clone + GenericImageGetter;
    type Tex: Component + Clone + TextureGetter;

    fn tex_name() -> &'static str;
    fn new_tex(texture: Texture) -> Self::Tex;
    fn is_srgb() -> bool;
    #[cfg(target_arch = "wasm32")]
    fn texture_receiver(upload_pass: &mut UploadPass) -> &mut Option<mpsc::Receiver<(Entity, Self::Tex)>>;
}

struct DiffuseUploadable;
impl TextureUploadable for DiffuseUploadable {
    type Img = DiffuseImg;
    type Tex = DiffuseTex;

    fn tex_name() -> &'static str {
        "diffuse"
    }
    fn new_tex(texture: Texture) -> Self::Tex {
        DiffuseTex(texture)
    }

    fn is_srgb() -> bool {
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn texture_receiver(upload_pass: &mut UploadPass) -> &mut Option<mpsc::Receiver<(Entity, Self::Tex)>> {
        &mut upload_pass.diffuse_receiver
    }
}

struct NormalUploadable;
impl TextureUploadable for NormalUploadable {
    type Img = NormalImg;
    type Tex = NormalTex;

    fn tex_name() -> &'static str {
        "normal"
    }
    fn new_tex(texture: Texture) -> Self::Tex {
        NormalTex(texture)
    }

    fn is_srgb() -> bool {
        false
    }

    #[cfg(target_arch = "wasm32")]
    fn texture_receiver(upload_pass: &mut UploadPass) -> &mut Option<mpsc::Receiver<(Entity, Self::Tex)>> {
        &mut upload_pass.normal_receiver
    }
}

struct RoughnessUploadable;
impl TextureUploadable for RoughnessUploadable {
    type Img = RoughnessImg;
    type Tex = RoughnessTex;

    fn tex_name() -> &'static str {
        "roughness"
    }
    fn new_tex(texture: Texture) -> Self::Tex {
        RoughnessTex(texture)
    }

    fn is_srgb() -> bool {
        false
    }

    #[cfg(target_arch = "wasm32")]
    fn texture_receiver(upload_pass: &mut UploadPass) -> &mut Option<mpsc::Receiver<(Entity, Self::Tex)>> {
        &mut upload_pass.roughness_receiver
    }
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
    #[cfg(target_arch = "wasm32")]
    diffuse_receiver: Option<mpsc::Receiver<(Entity, DiffuseTex)>>,
    #[cfg(target_arch = "wasm32")]
    normal_receiver: Option<mpsc::Receiver<(Entity, NormalTex)>>,
    #[cfg(target_arch = "wasm32")]
    roughness_receiver: Option<mpsc::Receiver<(Entity, RoughnessTex)>>,
}

impl UploadPass {
    pub fn new(gpu: &Gpu, params: &RenderConfig) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<PerFrameSceneCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameCamCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameLightCPU>() % 16 == 0);
        const_assert!(std::mem::size_of::<PerFrameParamsCPU>() % 16 == 0);

        let per_frame_uniforms = PerFrameUniforms::new(gpu);

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
            #[cfg(target_arch = "wasm32")]
            diffuse_receiver: None,
            #[cfg(target_arch = "wasm32")]
            normal_receiver: None,
            #[cfg(target_arch = "wasm32")]
            roughness_receiver: None,
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

        //interop between tensors and gpu buffers
        #[cfg(feature = "burn-torch")]
        {
            self.upload_v_interop(gpu, scene);
            self.upload_f_interop(gpu, scene);
            self.upload_uv_interop(gpu, scene);
            self.upload_nv_interop(gpu, scene);
            self.upload_t_interop(gpu, scene);
            self.upload_c_interop(gpu, scene);
        }

        &self.per_frame_uniforms
    }

    pub fn upload_textures(&mut self, gpu: &Gpu, scene: &mut Scene) {
        #[cfg(target_arch = "wasm32")]
        self.process_completed_texture_uploads(scene);

        self.upload_texture::<DiffuseUploadable>(gpu, scene);
        self.upload_texture::<NormalUploadable>(gpu, scene);
        self.upload_texture::<RoughnessUploadable>(gpu, scene);
        self.upload_environment_map(gpu, scene);
    }

    #[allow(clippy::unnecessary_unwrap)] //I know it's unnecesary but it makes everything more compact the two cases
                                         // more explicit
    fn upload_vertex_atrib<T: na::Scalar + bytemuck::Pod, C: CpuAtrib<T> + Component, G: GpuAtrib + Component>(
        &mut self,
        entity: Entity,
        atrib: &C,
        atrib_gpu: Option<&mut G>,
        gpu: &Gpu,
        additional_usage: wgpu::BufferUsages, // scene: &mut Scene,
        label: &str,
    ) {
        let verts = atrib.data_ref();
        let verts_t = verts.transpose();
        let size_bytes = verts_t.len() * atrib.byte_size_element();
        if atrib_gpu.is_none() || atrib_gpu.as_ref().unwrap().data_ref().size() != std::convert::TryInto::<u64>::try_into(size_bytes).unwrap() {
            //allocate new memory if it's the first time or if the size_bytes doesn't fit in it
            let desc = wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(verts_t.data.as_slice()),
                usage: additional_usage | wgpu::BufferUsages::COPY_DST,
            };
            let buf: wgpu::Buffer = gpu.device().create_buffer_init(&desc);

            self.command_buffer
                .insert_one(entity, G::new_from(buf, u32::try_from(atrib.data_ref().nrows()).unwrap()));
            //it either creates the component or it overwrites it
        } else {
            //buffer exists as it has to correct size, so we just write to it
            gpu.queue()
                .write_buffer(atrib_gpu.unwrap().data_ref(), 0, bytemuck::cast_slice(verts_t.data.as_slice()));
        }
    }

    /// Functions for uploading each component of the mesh
    fn upload_v(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world.query_mut::<(&Verts, Option<&mut VertsGPU>, Changed<Verts>)>().with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;

        for (ent, (verts, mut verts_gpu, changed_verts)) in query {
            if changed_verts {
                self.upload_vertex_atrib(ent, verts, verts_gpu.as_deref_mut(), gpu, usage, "verts");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn upload_e(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world
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
                let edges_v1_mat = index_vertices_from_edges(&verts.0, &edges.0, 0);
                let edges_v2_mat = index_vertices_from_edges(&verts.0, &edges.0, 1);

                let edges_v1 = EdgesV1(edges_v1_mat);
                let edges_v2 = EdgesV2(edges_v2_mat);

                self.upload_vertex_atrib(ent, &edges_v1, edges_v1_gpu.as_deref_mut(), gpu, usage, "edges_v1");
                self.upload_vertex_atrib(ent, &edges_v2, edges_v2_gpu.as_deref_mut(), gpu, usage, "edges_v2");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn upload_f(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world.query_mut::<(&Faces, Option<&mut FacesGPU>, Changed<Faces>)>().with::<&Renderable>();
        let usage = wgpu::BufferUsages::INDEX;
        for (ent, (faces, mut faces_gpu, changed_faces)) in query {
            if changed_faces {
                self.upload_vertex_atrib(ent, faces, faces_gpu.as_deref_mut(), gpu, usage, "faces");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }
    fn upload_uv(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world.query_mut::<(&UVs, Option<&mut UVsGPU>, Changed<UVs>)>().with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (uvs, mut uvs_gpu, changed_uvs)) in query {
            if changed_uvs {
                self.upload_vertex_atrib(ent, uvs, uvs_gpu.as_deref_mut(), gpu, usage, "uv");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn upload_nv(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world
            .query_mut::<(&Normals, Option<&mut NormalsGPU>, Changed<Normals>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (normals, mut normals_gpu, changed_normals)) in query {
            if changed_normals {
                self.upload_vertex_atrib(ent, normals, normals_gpu.as_deref_mut(), gpu, usage, "normals");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn upload_t(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world
            .query_mut::<(&Tangents, Option<&mut TangentsGPU>, Changed<Tangents>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (tangents, mut tangents_gpu, changed_tangents)) in query {
            if changed_tangents {
                self.upload_vertex_atrib(ent, tangents, tangents_gpu.as_deref_mut(), gpu, usage, "tangents");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    fn upload_c(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut world = scene.world_mut();
        let query = world
            .query_mut::<(&Colors, Option<&mut ColorsGPU>, Changed<Colors>)>()
            .with::<&Renderable>();
        let usage = wgpu::BufferUsages::VERTEX;
        for (ent, (colors, mut colors_gpu, changed_colors)) in query {
            if changed_colors {
                self.upload_vertex_atrib(ent, colors, colors_gpu.as_deref_mut(), gpu, usage, "colors");
            }
        }
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    #[cfg(target_arch = "wasm32")]
    // Insert textures to entity once they are completed and clear the receiver
    fn process_completed_texture_uploads(&mut self, scene: &mut Scene) {
        fn process_texture_receiver<T: TextureUploadable>(
            texture_receiver: &mut Option<mpsc::Receiver<(Entity, T::Tex)>>,
            command_buffer: &mut CommandBuffer,
        ) {
            if let Some(recv) = texture_receiver {
                let mut processed = false;
                while let Ok((entity, texture)) = recv.try_recv() {
                    command_buffer.insert_one(entity, texture);
                    processed = true;
                }
                if processed {
                    *texture_receiver = None;
                }
            }
        }

        process_texture_receiver::<DiffuseUploadable>(&mut self.diffuse_receiver, &mut self.command_buffer);
        process_texture_receiver::<NormalUploadable>(&mut self.normal_receiver, &mut self.command_buffer);
        process_texture_receiver::<RoughnessUploadable>(&mut self.roughness_receiver, &mut self.command_buffer);

        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    #[allow(clippy::too_many_lines)]
    fn upload_texture<T: TextureUploadable>(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let mut modified_entities = Vec::new();
        {
            let mut query = scene.world().query::<(&mut T::Img, Option<&mut T::Tex>, Changed<T::Img>)>();

            for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
                if changed_img && img.generic_img().cpu_img.is_some() {
                    debug!("{} changed for entity {entity:?}", T::tex_name());
                    let nr_channels = img.generic_img().img_ref().color().channel_count();
                    if nr_channels != 4 {
                        warn!("unoptimal use of memory: diffuse does not have 4 channels, it has {nr_channels}");
                    }
                    modified_entities.push(entity);
                    let is_srgb = T::is_srgb(); //only true for diffuse since they are in srgb space in the png but we want to sample linear colors
                    let keep_on_cpu = img.generic_img().config.keep_on_cpu;

                    #[cfg(not(target_arch = "wasm32"))]
                    let staging_buffer = if img.generic_img().config.fast_upload {
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
                    #[allow(unused_mut)]
                    if let Some(mut existing_tex) = tex_opt {
                        let new_tex_extent = Texture::extent_from_img(img.generic_img().img_ref());
                        let new_tex_format = Texture::format_from_img(img.generic_img().img_ref(), is_srgb);
                        let old_tex_extent = existing_tex.texture().extent();
                        let old_format = existing_tex.texture().texture.format();
                        if new_tex_format == old_format && new_tex_extent == old_tex_extent {
                            debug!("reusing diffuse tex");

                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                existing_tex
                                    .texture_mut()
                                    .update_from_img(
                                        img.generic_img().img_ref(),
                                        gpu.device(),
                                        gpu.queue(),
                                        is_srgb,
                                        img.generic_img().config.generate_mipmaps,
                                        img.generic_img().config.mipmap_generation_cpu,
                                        staging_buffer,
                                        self.mipmapper.as_ref(),
                                    )
                                    .block_on()
                                    .unwrap();
                            }

                            #[cfg(target_arch = "wasm32")]
                            {
                                // We can safely clone a lot of the wgpu types because internally they are behind arcs
                                let img_clone = img.generic_img().img_ref().clone();
                                let device = gpu.device().clone();
                                let queue = gpu.queue().clone();
                                let generate_mipmaps = img.generic_img().config.generate_mipmaps;
                                let mipmap_generation_cpu = img.generic_img().config.mipmap_generation_cpu;
                                let mipmapper_clone = self.mipmapper.clone();
                                let mut existing_tex_clone = existing_tex.clone();

                                wasm_bindgen_futures::spawn_local(async move {
                                    match existing_tex_clone
                                        .texture_mut()
                                        .update_from_img(
                                            &img_clone,
                                            &device,
                                            &queue,
                                            is_srgb,
                                            generate_mipmaps,
                                            mipmap_generation_cpu,
                                            None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
                                            mipmapper_clone.as_ref(),
                                        )
                                        .await
                                    {
                                        Ok(_) => {}
                                        Err(e) => {
                                            log::error!("Texture update failed: {:?}", e);
                                        }
                                    }
                                });
                            }

                            tex_uploaded = true;
                        }
                    }
                    //we create a new one if we couldn't update an existing one
                    if !tex_uploaded {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let tex = Texture::from_img(
                                img.generic_img().img_ref(),
                                gpu.device(),
                                gpu.queue(),
                                is_srgb,
                                img.generic_img().config.generate_mipmaps,
                                img.generic_img().config.mipmap_generation_cpu,
                                staging_buffer,
                                self.mipmapper.as_ref(),
                            )
                            .block_on()
                            .unwrap();
                            self.command_buffer.insert_one(entity, T::new_tex(tex));
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            let (sender, receiver) = mpsc::channel();
                            let texture_receiver = T::texture_receiver(self);
                            if texture_receiver.is_none() {
                                *texture_receiver = Some(receiver);
                            }

                            let img_clone = img.generic_img().img_ref().clone();
                            let device = gpu.device().clone();
                            let queue = gpu.queue().clone();
                            let generate_mipmaps = img.generic_img().config.generate_mipmaps;
                            let mipmap_generation_cpu = img.generic_img().config.mipmap_generation_cpu;
                            let mipmapper_clone = self.mipmapper.clone();

                            wasm_bindgen_futures::spawn_local(async move {
                                match Texture::from_img(
                                    &img_clone,
                                    &device,
                                    &queue,
                                    is_srgb,
                                    generate_mipmaps,
                                    mipmap_generation_cpu,
                                    None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
                                    mipmapper_clone.as_ref(),
                                )
                                .await
                                {
                                    Ok(texture) => {
                                        let _ = sender.send((entity, T::new_tex(texture)));
                                    }
                                    Err(e) => {
                                        log::error!("Texture creation failed: {:?}", e);
                                    }
                                }
                            });
                        }
                    }

                    if !keep_on_cpu {
                        let _ = img.generic_img_mut().cpu_img.take();
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
        scene.world_mut().run_command_buffer(&mut self.command_buffer);
    }

    // #[allow(clippy::too_many_lines)]
    // fn upload_diffuse_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
    //     let mut modified_entities = Vec::new();
    //     {
    //         let mut query = scene
    //             .world
    //             .query::<(&mut DiffuseImg, Option<&mut DiffuseTex>, Changed<DiffuseImg>)>()
    //             .with::<&Renderable>();

    //         for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
    //             if changed_img && img.generic_img.cpu_img.is_some() {
    //                 debug!("DiffuseImg changed for entity {entity:?}");
    //                 let nr_channels = img.generic_img.img_ref().color().channel_count();
    //                 if nr_channels != 4 {
    //                     warn!("unoptimal use of memory: diffuse does not have 4 channels, it has {nr_channels}");
    //                 }
    //                 modified_entities.push(entity);
    //                 let is_srgb = true; //only true for diffuse since they are in srgb space in the png but we want to sample linear colors
    //                 let keep_on_cpu = img.generic_img.config.keep_on_cpu;

    //                 #[cfg(not(target_arch = "wasm32"))]
    //                 let staging_buffer = if img.generic_img.config.fast_upload {
    //                     None
    //                 } else {
    //                     //using slow upload through a preallocated staging buffer
    //                     if self.staging_buffer.is_none() {
    //                         warn!("The diffuse image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
    //                     }
    //                     self.staging_buffer.as_ref()
    //                 };

    //                 //either create a new tex or update the existing one
    //                 let mut tex_uploaded = false;
    //                 #[allow(unused_mut)]
    //                 if let Some(mut existing_tex) = tex_opt {
    //                     let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
    //                     let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
    //                     let old_tex_extent = existing_tex.0.extent();
    //                     let old_format = existing_tex.0.texture.format();
    //                     if new_tex_format == old_format && new_tex_extent == old_tex_extent {
    //                         debug!("reusing diffuse tex");

    //                         #[cfg(not(target_arch = "wasm32"))]
    //                         {
    //                             existing_tex
    //                                 .0
    //                                 .update_from_img(
    //                                     img.generic_img.img_ref(),
    //                                     gpu.device(),
    //                                     gpu.queue(),
    //                                     is_srgb,
    //                                     img.generic_img.config.generate_mipmaps,
    //                                     img.generic_img.config.mipmap_generation_cpu,
    //                                     staging_buffer,
    //                                     self.mipmapper.as_ref(),
    //                                 )
    //                                 .block_on()
    //                                 .unwrap();
    //                         }

    //                         #[cfg(target_arch = "wasm32")]
    //                         {
    //                             let (_sender, receiver) = mpsc::channel();
    //                             if self.diffuse_receiver.is_none() {
    //                                 self.diffuse_receiver = Some(receiver);
    //                             }
    //                             // We can safely clone a lot of the wgpu types because internally they are behind arcs
    //                             let img_clone = img.generic_img.img_ref().clone();
    //                             let device = gpu.device().clone();
    //                             let queue = gpu.queue().clone();
    //                             let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                             let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                             let mipmapper_clone = self.mipmapper.clone();
    //                             let mut existing_tex_clone = existing_tex.clone();

    //                             wasm_bindgen_futures::spawn_local(async move {
    //                                 match existing_tex_clone
    //                                     .0
    //                                     .update_from_img(
    //                                         &img_clone,
    //                                         &device,
    //                                         &queue,
    //                                         is_srgb,
    //                                         generate_mipmaps,
    //                                         mipmap_generation_cpu,
    //                                         None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                         mipmapper_clone.as_ref(),
    //                                     )
    //                                     .await
    //                                 {
    //                                     Ok(_) => {}
    //                                     Err(e) => {
    //                                         log::error!("Texture update failed: {:?}", e);
    //                                     }
    //                                 }
    //                             });

    //                             // Mark entity as having pending upload
    //                             // self.command_buffer.insert_one(entity, PendingDiffuseUpload);
    //                         }

    //                         tex_uploaded = true;
    //                     }
    //                 }
    //                 //we create a new one if we couldn't update an existing one
    //                 if !tex_uploaded {
    //                     #[cfg(not(target_arch = "wasm32"))]
    //                     {
    //                         let tex = Texture::from_img(
    //                             img.generic_img.img_ref(),
    //                             gpu.device(),
    //                             gpu.queue(),
    //                             is_srgb,
    //                             img.generic_img.config.generate_mipmaps,
    //                             img.generic_img.config.mipmap_generation_cpu,
    //                             staging_buffer,
    //                             self.mipmapper.as_ref(),
    //                         )
    //                         .block_on()
    //                         .unwrap();
    //                         self.command_buffer.insert_one(entity, DiffuseTex(tex));
    //                     }
    //                     #[cfg(target_arch = "wasm32")]
    //                     {
    //                         let (sender, receiver) = mpsc::channel();
    //                         if self.diffuse_receiver.is_none() {
    //                             self.diffuse_receiver = Some(receiver);
    //                         }

    //                         let img_clone = img.generic_img.img_ref().clone();
    //                         let device = gpu.device().clone();
    //                         let queue = gpu.queue().clone();
    //                         let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                         let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                         let mipmapper_clone = self.mipmapper.clone();

    //                         wasm_bindgen_futures::spawn_local(async move {
    //                             match Texture::from_img(
    //                                 &img_clone,
    //                                 &device,
    //                                 &queue,
    //                                 is_srgb,
    //                                 generate_mipmaps,
    //                                 mipmap_generation_cpu,
    //                                 None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                 mipmapper_clone.as_ref(),
    //                             )
    //                             .await
    //                             {
    //                                 Ok(texture) => {
    //                                     let _ = sender.send((entity, DiffuseTex(texture)));
    //                                 }
    //                                 Err(e) => {
    //                                     log::error!("Texture creation failed: {:?}", e);
    //                                 }
    //                             }
    //                         });

    //                         // Mark entity as having pending upload
    //                         // self.command_buffer.insert_one(entity, PendingDiffuseUpload);
    //                     }
    //                 }

    //                 if !keep_on_cpu {
    //                     // self.command_buffer.remove_one::<DiffuseImg>(entity);
    //                     let _ = img.generic_img.cpu_img.take();
    //                 }
    //             }
    //         }

    //         //set those meshes to actually visualize the mesh
    //         for entity in modified_entities {
    //             // let mut vis_mesh = scene.get_comp::<&mut VisMesh>(&entity);
    //             if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
    //                 if vis_mesh.added_automatically {
    //                     vis_mesh.color_type = MeshColorType::Texture;
    //                 }
    //             }
    //         }
    //     }
    //     self.command_buffer.run_on(&mut scene.world);
    // }

    // #[allow(clippy::too_many_lines)]
    // fn upload_normal_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
    //     let mut modified_entities = Vec::new();
    //     {
    //         let mut query = scene
    //             .world
    //             .query::<(&mut NormalImg, Option<&mut NormalTex>, Changed<NormalImg>)>()
    //             .with::<&Renderable>();
    //         for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
    //             if changed_img && img.generic_img.cpu_img.is_some() {
    //                 debug!("NormalImg changed for entity {entity:?}");
    //                 let nr_channels = img.generic_img.img_ref().color().channel_count();
    //                 if nr_channels != 4 {
    //                     warn!("unoptimal use of memory: normal does not have 4 channels, it has {nr_channels}");
    //                 }
    //                 modified_entities.push(entity);
    //                 let is_srgb = false; //only true for diffuse since they are in srgb space in the png but we want to sample linear colors
    //                 let keep_on_cpu = img.generic_img.config.keep_on_cpu;

    //                 #[cfg(not(target_arch = "wasm32"))]
    //                 let staging_buffer = if img.generic_img.config.fast_upload {
    //                     None
    //                 } else {
    //                     //using slow upload through a preallocated staging buffer
    //                     if self.staging_buffer.is_none() {
    //                         warn!("The normal image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
    //                     }
    //                     self.staging_buffer.as_ref()
    //                 };

    //                 //either create a new tex or update the existing one
    //                 let mut tex_uploaded = false;
    //                 #[allow(unused_mut)]
    //                 if let Some(mut existing_tex) = tex_opt {
    //                     let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
    //                     let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
    //                     let old_tex_extent = existing_tex.0.extent();
    //                     let old_format = existing_tex.0.texture.format();
    //                     if new_tex_format == old_format && new_tex_extent == old_tex_extent {
    //                         debug!("reusing normal tex");

    //                         #[cfg(not(target_arch = "wasm32"))]
    //                         {
    //                             existing_tex
    //                                 .0
    //                                 .update_from_img(
    //                                     img.generic_img.img_ref(),
    //                                     gpu.device(),
    //                                     gpu.queue(),
    //                                     is_srgb,
    //                                     img.generic_img.config.generate_mipmaps,
    //                                     img.generic_img.config.mipmap_generation_cpu,
    //                                     staging_buffer,
    //                                     self.mipmapper.as_ref(),
    //                                 )
    //                                 .block_on()
    //                                 .unwrap();
    //                         }

    //                         #[cfg(target_arch = "wasm32")]
    //                         {
    //                             let (_sender, receiver) = mpsc::channel();
    //                             if self.normal_receiver.is_none() {
    //                                 self.normal_receiver = Some(receiver);
    //                             }
    //                             // We can safely clone a lot of the wgpu types because internally they are behind arcs
    //                             let img_clone = img.generic_img.img_ref().clone();
    //                             let device = gpu.device().clone();
    //                             let queue = gpu.queue().clone();
    //                             let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                             let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                             let mipmapper_clone = self.mipmapper.clone();
    //                             let mut existing_tex_clone = existing_tex.clone();

    //                             wasm_bindgen_futures::spawn_local(async move {
    //                                 match existing_tex_clone
    //                                     .0
    //                                     .update_from_img(
    //                                         &img_clone,
    //                                         &device,
    //                                         &queue,
    //                                         is_srgb,
    //                                         generate_mipmaps,
    //                                         mipmap_generation_cpu,
    //                                         None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                         mipmapper_clone.as_ref(),
    //                                     )
    //                                     .await
    //                                 {
    //                                     Ok(_) => {}
    //                                     Err(e) => {
    //                                         log::error!("Texture update failed: {:?}", e);
    //                                     }
    //                                 }
    //                             });

    //                             // Mark entity as having pending upload
    //                             // self.command_buffer.insert_one(entity, PendingNormalUpload);
    //                         }

    //                         tex_uploaded = true;
    //                     }
    //                 }
    //                 //we create a new one if we couldn't update an existing one
    //                 if !tex_uploaded {
    //                     #[cfg(not(target_arch = "wasm32"))]
    //                     {
    //                         let tex = Texture::from_img(
    //                             img.generic_img.img_ref(),
    //                             gpu.device(),
    //                             gpu.queue(),
    //                             is_srgb,
    //                             img.generic_img.config.generate_mipmaps,
    //                             img.generic_img.config.mipmap_generation_cpu,
    //                             staging_buffer,
    //                             self.mipmapper.as_ref(),
    //                         )
    //                         .block_on()
    //                         .unwrap();
    //                         self.command_buffer.insert_one(entity, NormalTex(tex));
    //                     }

    //                     #[cfg(target_arch = "wasm32")]
    //                     {
    //                         let (sender, receiver) = mpsc::channel();
    //                         if self.normal_receiver.is_none() {
    //                             self.normal_receiver = Some(receiver);
    //                         }

    //                         let img_clone = img.generic_img.img_ref().clone();
    //                         let device = gpu.device().clone();
    //                         let queue = gpu.queue().clone();
    //                         let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                         let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                         let mipmapper_clone = self.mipmapper.clone();

    //                         wasm_bindgen_futures::spawn_local(async move {
    //                             match Texture::from_img(
    //                                 &img_clone,
    //                                 &device,
    //                                 &queue,
    //                                 is_srgb,
    //                                 generate_mipmaps,
    //                                 mipmap_generation_cpu,
    //                                 None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                 mipmapper_clone.as_ref(),
    //                             )
    //                             .await
    //                             {
    //                                 Ok(texture) => {
    //                                     let _ = sender.send((entity, NormalTex(texture)));
    //                                 }
    //                                 Err(e) => {
    //                                     log::error!("Texture creation failed: {:?}", e);
    //                                 }
    //                             }
    //                         });

    //                         // Mark entity as having pending upload
    //                         // self.command_buffer.insert_one(entity, PendingNormalUpload);
    //                     }
    //                 }

    //                 if !keep_on_cpu {
    //                     // self.command_buffer.remove_one::<NormalImg>(entity);
    //                     let _ = img.generic_img.cpu_img.take();
    //                 }
    //             }
    //         }

    //         //set those meshes to actually visualize the mesh
    //         for entity in modified_entities {
    //             if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
    //                 if vis_mesh.added_automatically {
    //                     vis_mesh.color_type = MeshColorType::Texture;
    //                 }
    //             }
    //         }
    //     }

    //     self.command_buffer.run_on(&mut scene.world);
    // }

    // #[allow(clippy::too_many_lines)]
    // fn upload_roughness_tex(&mut self, gpu: &Gpu, scene: &mut Scene) {
    //     let mut modified_entities = Vec::new();
    //     {
    //         let mut query = scene
    //             .world
    //             .query::<(&mut RoughnessImg, Option<&mut RoughnessTex>, Changed<RoughnessImg>)>()
    //             .with::<&Renderable>();
    //         for (entity, (mut img, tex_opt, changed_img)) in query.iter() {
    //             if changed_img && img.generic_img.cpu_img.is_some() {
    //                 debug!("RoughnessImg changed for entity {entity:?}");
    //                 let nr_channels = img.generic_img.img_ref().color().channel_count();
    //                 if nr_channels != 1 {
    //                     warn!("unoptimal use of memory: roughness does not have 1 channels, it has {nr_channels}");
    //                 }
    //                 modified_entities.push(entity);
    //                 let is_srgb = false; //only true for diffuse since they are in srgb space in the png but we want to
    //                                      // sample linear colors
    //                 let keep_on_cpu = img.generic_img.config.keep_on_cpu;

    //                 #[cfg(not(target_arch = "wasm32"))]
    //                 let staging_buffer = if img.generic_img.config.fast_upload {
    //                     None
    //                 } else {
    //                     //using slow upload through a preallocated staging buffer
    //                     if self.staging_buffer.is_none() {
    //                         warn!("The roughness image is set to slow upload which would require a preallocated staging buffer. However no bytes have been allocated for it. Check the config.toml for the preallocated_staging_buffer. Now we default to fast upload through wgpu staging buffer which might use more memory than necessary.");
    //                     }
    //                     self.staging_buffer.as_ref()
    //                 };

    //                 //either create a new tex or update the existing one
    //                 let mut tex_uploaded = false;
    //                 #[allow(unused_mut)]
    //                 if let Some(mut existing_tex) = tex_opt {
    //                     let new_tex_extent = Texture::extent_from_img(img.generic_img.img_ref());
    //                     let new_tex_format = Texture::format_from_img(img.generic_img.img_ref(), is_srgb);
    //                     let old_tex_extent = existing_tex.0.extent();
    //                     let old_format = existing_tex.0.texture.format();
    //                     if new_tex_format == old_format && new_tex_extent == old_tex_extent {
    //                         debug!("reusing roughness tex");

    //                         #[cfg(not(target_arch = "wasm32"))]
    //                         {
    //                             existing_tex
    //                                 .0
    //                                 .update_from_img(
    //                                     img.generic_img.img_ref(),
    //                                     gpu.device(),
    //                                     gpu.queue(),
    //                                     is_srgb,
    //                                     img.generic_img.config.generate_mipmaps,
    //                                     img.generic_img.config.mipmap_generation_cpu,
    //                                     staging_buffer,
    //                                     self.mipmapper.as_ref(),
    //                                 )
    //                                 .block_on()
    //                                 .unwrap();
    //                         }

    //                         #[cfg(target_arch = "wasm32")]
    //                         {
    //                             let (_sender, receiver) = mpsc::channel();
    //                             if self.roughness_receiver.is_none() {
    //                                 self.roughness_receiver = Some(receiver);
    //                             }
    //                             // We can safely clone a lot of the wgpu types because internally they are behind arcs
    //                             let img_clone = img.generic_img.img_ref().clone();
    //                             let device = gpu.device().clone();
    //                             let queue = gpu.queue().clone();
    //                             let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                             let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                             let mipmapper_clone = self.mipmapper.clone();
    //                             let mut existing_tex_clone = existing_tex.clone();

    //                             wasm_bindgen_futures::spawn_local(async move {
    //                                 match existing_tex_clone
    //                                     .0
    //                                     .update_from_img(
    //                                         &img_clone,
    //                                         &device,
    //                                         &queue,
    //                                         is_srgb,
    //                                         generate_mipmaps,
    //                                         mipmap_generation_cpu,
    //                                         None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                         mipmapper_clone.as_ref(),
    //                                     )
    //                                     .await
    //                                 {
    //                                     Ok(_) => {}
    //                                     Err(e) => {
    //                                         log::error!("Texture update failed: {:?}", e);
    //                                     }
    //                                 }
    //                             });

    //                             // Mark entity as having pending upload
    //                             // self.command_buffer.insert_one(entity, PendingRoughnessUpload);
    //                         }

    //                         tex_uploaded = true;
    //                     }
    //                 }
    //                 //we create a new one if we couldn't update an existing one
    //                 if !tex_uploaded {
    //                     #[cfg(not(target_arch = "wasm32"))]
    //                     {
    //                         let tex = Texture::from_img(
    //                             img.generic_img.img_ref(),
    //                             gpu.device(),
    //                             gpu.queue(),
    //                             is_srgb,
    //                             img.generic_img.config.generate_mipmaps,
    //                             img.generic_img.config.mipmap_generation_cpu,
    //                             staging_buffer,
    //                             self.mipmapper.as_ref(),
    //                         )
    //                         .block_on()
    //                         .unwrap();
    //                         self.command_buffer.insert_one(entity, RoughnessTex(tex));
    //                     }

    //                     #[cfg(target_arch = "wasm32")]
    //                     {
    //                         let (sender, receiver) = mpsc::channel();
    //                         if self.roughness_receiver.is_none() {
    //                             self.roughness_receiver = Some(receiver);
    //                         }

    //                         let img_clone = img.generic_img.img_ref().clone();
    //                         let device = gpu.device().clone();
    //                         let queue = gpu.queue().clone();
    //                         let generate_mipmaps = img.generic_img.config.generate_mipmaps;
    //                         let mipmap_generation_cpu = img.generic_img.config.mipmap_generation_cpu;
    //                         let mipmapper_clone = self.mipmapper.clone();

    //                         wasm_bindgen_futures::spawn_local(async move {
    //                             match Texture::from_img(
    //                                 &img_clone,
    //                                 &device,
    //                                 &queue,
    //                                 is_srgb,
    //                                 generate_mipmaps,
    //                                 mipmap_generation_cpu,
    //                                 None, //TODO: Forcing fast upload on WASM to avoid parking issues, look into fixes for this
    //                                 mipmapper_clone.as_ref(),
    //                             )
    //                             .await
    //                             {
    //                                 Ok(texture) => {
    //                                     let _ = sender.send((entity, RoughnessTex(texture)));
    //                                 }
    //                                 Err(e) => {
    //                                     log::error!("Texture creation failed: {:?}", e);
    //                                 }
    //                             }
    //                         });

    //                         // Mark entity as having pending upload
    //                         // self.command_buffer.insert_one(entity, PendingRoughnessUpload);
    //                     }
    //                 }

    //                 if !keep_on_cpu {
    //                     // self.command_buffer.remove_one::<RoughnessImg>(entity);
    //                     let _ = img.generic_img.cpu_img.take();
    //                 }
    //             }
    //         }

    //         //set those meshes to actually visualize the mesh
    //         for entity in modified_entities {
    //             if let Ok(mut vis_mesh) = scene.get_comp::<&mut VisMesh>(&entity) {
    //                 if vis_mesh.added_automatically {
    //                     vis_mesh.color_type = MeshColorType::Texture;
    //                 }
    //             }
    //         }
    //     }

    //     self.command_buffer.run_on(&mut scene.world);
    // }

    fn upload_environment_map(&mut self, gpu: &Gpu, scene: &mut Scene) {
        // if scene.has_resource::<EnvironmentMap>() {
        let mut world = scene.world_mut();
        let query = world.query_mut::<(&EnvironmentMap, Changed<EnvironmentMap>)>();
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

        scene.world_mut().run_command_buffer(&mut self.command_buffer);
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
        let pos_lookat = if let Ok(pos_lookat) = scene.world().get::<&mut PosLookat>(camera.entity) {
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
        if scene.world().has::<Projection>(camera.entity).unwrap() {
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

        let mut world = scene.world_mut();
        let query = world.query_mut::<(&Name, &PosLookat, &Projection, &LightEmit, Option<&ShadowCaster>)>();
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

    #[cfg(feature = "burn-torch")]
    fn upload_v_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&VertsPyTensor, Option<&mut VertsGPU>, Changed<VertsPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (verts, verts_gpu, changed_verts)) in query {
            if changed_verts {
                if let Some(mut verts_gpu) = verts_gpu {
                    // println!("VertsGPU already exists for entity {:?}", ent);
                    verts_gpu
                        .buf
                        .copy_from_tensor(&verts.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy vertex tensor to GPU buffer");
                    verts_gpu.nr_vertices = verts.tensor.size()[1] as u32;
                } else {
                    // println!("Uploading vertex interop for entity {:?}", ent);
                    let verts_gpu = VertsGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &verts.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            Some("VertsGPU buffer"),
                        ),
                        nr_vertices: verts.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, verts_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    #[cfg(feature = "burn-torch")]
    fn upload_f_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&FacesPyTensor, Option<&mut FacesGPU>, Changed<FacesPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (faces, faces_gpu, changed_faces)) in query {
            if changed_faces {
                if let Some(mut faces_gpu) = faces_gpu {
                    faces_gpu
                        .buf
                        .copy_from_tensor(&faces.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy face tensor to GPU buffer");
                    faces_gpu.nr_triangles = faces.tensor.size()[1] as u32;
                } else {
                    let faces_gpu = FacesGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &faces.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                            Some("FacesGPU buffer"),
                        ),
                        nr_triangles: faces.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, faces_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    #[cfg(feature = "burn-torch")]
    fn upload_uv_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&UVsPyTensor, Option<&mut UVsGPU>, Changed<UVsPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (uvs, uvs_gpu, changed_uvs)) in query {
            if changed_uvs {
                if let Some(mut uvs_gpu) = uvs_gpu {
                    uvs_gpu
                        .buf
                        .copy_from_tensor(&uvs.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy UVs tensor to GPU buffer");
                    uvs_gpu.nr_vertices = uvs.tensor.size()[1] as u32;
                } else {
                    let uvs_gpu = UVsGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &uvs.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            Some("UVsGPU buffer"),
                        ),
                        nr_vertices: uvs.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, uvs_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    #[cfg(feature = "burn-torch")]
    fn upload_nv_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&NormalsPyTensor, Option<&mut NormalsGPU>, Changed<NormalsPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (normals, normals_gpu, changed_normals)) in query {
            if changed_normals {
                if let Some(mut normals_gpu) = normals_gpu {
                    normals_gpu
                        .buf
                        .copy_from_tensor(&normals.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy Normals tensor to GPU buffer");
                    normals_gpu.nr_vertices = normals.tensor.size()[1] as u32;
                } else {
                    let normals_gpu = NormalsGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &normals.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            Some("NormalsGPU buffer"),
                        ),
                        nr_vertices: normals.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, normals_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    #[cfg(feature = "burn-torch")]
    fn upload_t_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&TangentsPyTensor, Option<&mut TangentsGPU>, Changed<TangentsPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (tangents, tangents_gpu, changed_tangents)) in query {
            if changed_tangents {
                if let Some(mut tangents_gpu) = tangents_gpu {
                    tangents_gpu
                        .buf
                        .copy_from_tensor(&tangents.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy Tangents tensor to GPU buffer");
                    tangents_gpu.nr_vertices = tangents.tensor.size()[1] as u32;
                } else {
                    let tangents_gpu = TangentsGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &tangents.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            Some("TangentsGPU buffer"),
                        ),
                        nr_vertices: tangents.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, tangents_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
    }

    #[cfg(feature = "burn-torch")]
    fn upload_c_interop(&mut self, gpu: &Gpu, scene: &mut Scene) {
        let query = scene
            .world
            .query_mut::<(&ColorsPyTensor, Option<&mut ColorsGPU>, Changed<ColorsPyTensor>)>()
            .with::<&Renderable>();

        for (ent, (colors, colors_gpu, changed_colors)) in query {
            if changed_colors {
                if let Some(mut colors_gpu) = colors_gpu {
                    colors_gpu
                        .buf
                        .copy_from_tensor(&colors.tensor, gpu.device(), gpu.queue(), gpu.adapter())
                        .expect("Failed to copy Colors tensor to GPU buffer");
                    colors_gpu.nr_vertices = colors.tensor.size()[1] as u32;
                } else {
                    let colors_gpu = ColorsGPU {
                        buf: easy_wgpu::buffer::Buffer::new_from_tensor(
                            &colors.tensor,
                            gpu.device(),
                            gpu.queue(),
                            gpu.adapter(),
                            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            Some("ColorsGPU buffer"),
                        ),
                        nr_vertices: colors.tensor.size()[1] as u32,
                    };
                    self.command_buffer.insert_one(ent, colors_gpu);
                }
            }
        }
        self.command_buffer.run_on(&mut scene.world);
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
