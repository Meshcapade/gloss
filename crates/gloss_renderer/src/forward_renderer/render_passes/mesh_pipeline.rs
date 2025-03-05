use std::collections::HashMap;

use crate::{
    components::{
        ColorsGPU, DiffuseTex, EnvironmentMapGpu, FacesGPU, ModelMatrix, Name, NormalTex, NormalsGPU, Renderable, RoughnessTex, ShadowCaster,
        ShadowMap, TangentsGPU, UVsGPU, VertsGPU, VisMesh,
    },
    config::RenderConfig,
    forward_renderer::{bind_group_collection::BindGroupCollection, locals::LocalEntData},
    light::Light,
    scene::Scene,
};
use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupDesc, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
};
// use gloss_utils::log;

use easy_wgpu::{
    gpu::Gpu,
    texture::{TexParams, Texture},
    utils::create_empty_group,
};
use gloss_hecs::Entity;
use log::debug;

use super::{pipeline_runner::PipelineRunner, upload_pass::PerFrameUniforms};

use easy_wgpu::pipeline::RenderPipelineDescBuilder;

use super::upload_pass::MAX_NUM_SHADOWS;
use encase;

use gloss_utils::numerical::align;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/gbuffer_mesh_vert.wgsl")]
mod vert_shader_code {}
#[allow(clippy::approx_constant)]
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/gbuffer_mesh_frag.wgsl")]
mod frag_shader_code {}

/// Render all the meshes from the scene to the `GBuffer`
pub struct MeshPipeline {
    render_pipeline: wgpu::RenderPipeline,
    _empty_group: wgpu::BindGroup,
    locals_uniform: Buffer, // a uniform buffer that we suballocate for the locals of every mesh
    locals_bind_groups: LocalsBindGroups,
    /// layout of the input to the mesh pass. Usually contains gbuffer textures,
    /// shadow maps, etc.
    input_layout: wgpu::BindGroupLayout,
    input_bind_group: Option<BindGroupWrapper>,
    //misc
    /// stores some entities that should be local to this pass and don't need to
    /// be stored in the main scene
    local_scene: Scene,
    /// Light that is used as a dummy. Serves to bind to the [``input_layout``]
    /// and [``input_bind_group``] and satisfy the layout needs even if we don't
    /// actually use this light.
    passthrough_light: Light,
}

impl MeshPipeline {
    /// # Panics
    /// Will panic if the gbuffer does not have the correct textures that are
    /// needed for the pipeline creation
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<Locals>() % 16 == 0);

        let input_layout_desc = Self::input_layout_desc();
        let input_layout = input_layout_desc.clone().into_bind_group_layout(gpu.device());

        //render pipeline
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("mesh_pipeline")
            //Code has to be sparated between vert and frag because we use derivatives with dpdx in frag shader and that fails to compiles on wasm when in the same file as the vert shader: https://github.com/gfx-rs/wgpu/issues/4368
            .shader_code_vert(vert_shader_code::SOURCE)
            .shader_code_frag(frag_shader_code::SOURCE)
            .shader_label("mesh_shader")
            .add_bind_group_layout_desc(PerFrameUniforms::build_layout_desc())
            .add_bind_group_layout_desc(input_layout_desc)
            .add_bind_group_layout_desc(LocalsBindGroups::build_layout_desc())
            .add_vertex_buffer_layout(VertsGPU::vertex_buffer_layout::<0>())
            .add_vertex_buffer_layout(UVsGPU::vertex_buffer_layout::<1>())
            .add_vertex_buffer_layout(NormalsGPU::vertex_buffer_layout::<2>())
            .add_vertex_buffer_layout(TangentsGPU::vertex_buffer_layout::<3>())
            .add_vertex_buffer_layout(ColorsGPU::vertex_buffer_layout::<4>())
            .add_render_target(wgpu::ColorTargetState {
                format: color_target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
            .depth_state(Some(wgpu::DepthStencilState {
                format: depth_target_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }))
            .multisample(wgpu::MultisampleState {
                count: params.msaa_nr_samples,
                ..Default::default()
            })
            .build_pipeline(gpu.device());

        let empty_group = create_empty_group(gpu.device());

        let size_bytes = 0x10000;
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM;
        let locals_uniform = Buffer::new_empty(gpu.device(), usage, Some("local_buffer"), size_bytes);

        let locals_bind_groups = LocalsBindGroups::new(gpu);

        //we want to add a dummy light and we do it in a local world so we don't have
        // to depend on the main scene
        let mut local_scene = Scene::new();
        // let passthrough_tex = Texture::create_default_texture(gpu.device(),
        // gpu.queue());
        let passthrough_tex = Texture::new(
            gpu.device(),
            4,
            4,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
            TexParams::default(),
        );
        // let passthrough_tex2 = Texture::create_default_texture(gpu.device(),
        // gpu.queue());
        let passthrough_light = Light::new("compose_pass_passthrough_light", &mut local_scene);
        let _ = local_scene.world.insert_one(
            passthrough_light.entity,
            ShadowMap {
                tex_depth: passthrough_tex,
                // tex_depth_moments: passthrough_tex2,
            },
        );

        Self {
            render_pipeline,
            _empty_group: empty_group,
            locals_uniform,
            locals_bind_groups,
            input_layout,
            input_bind_group: None,
            //misc
            local_scene,
            passthrough_light,
        }
    }
}
impl PipelineRunner for MeshPipeline {
    type QueryItems<'a> = (
        &'a VertsGPU,
        &'a FacesGPU,
        &'a UVsGPU,
        &'a NormalsGPU,
        &'a TangentsGPU,
        &'a ColorsGPU,
        &'a DiffuseTex,
        &'a NormalTex,
        &'a RoughnessTex,
        &'a VisMesh,
        &'a Name,
    );
    type QueryState<'a> = gloss_hecs::QueryBorrow<'a, gloss_hecs::With<Self::QueryItems<'a>, &'a Renderable>>;

    fn query_state(scene: &Scene) -> Self::QueryState<'_> {
        scene.world.query::<Self::QueryItems<'_>>().with::<&Renderable>()
    }

    fn prepare<'a>(&mut self, gpu: &Gpu, per_frame_uniforms: &PerFrameUniforms, scene: &'a Scene) -> Self::QueryState<'a> {
        self.begin_pass();

        self.update_locals(gpu, scene);

        //update the bind group in case the input_texture or the shadowmaps changed
        self.update_input_bind_group(gpu, scene, per_frame_uniforms);

        Self::query_state(scene)
    }

    /// # Panics
    /// Will panic if the input bind groups are not created
    #[allow(clippy::too_many_lines)]
    fn run<'r>(
        &'r mut self,
        render_pass: &mut wgpu::RenderPass<'r>,
        per_frame_uniforms: &'r PerFrameUniforms,
        _render_params: &RenderConfig,
        query_state: &'r mut Self::QueryState<'_>,
    ) {
        //completely skip this if there are no entities to draw
        if query_state.iter().count() == 0 {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);

        //global binding
        render_pass.set_bind_group(0, &per_frame_uniforms.bind_group, &[]);
        //input binding
        render_pass.set_bind_group(1, self.input_bind_group.as_ref().unwrap().bg(), &[]);

        for (_id, (verts, faces, uvs, normals, tangents, colors, _diffuse_tex, _normal_tex, _roughness_tex, vis_mesh, name)) in query_state.iter() {
            if !vis_mesh.show_mesh {
                continue;
            }

            //local bindings
            let (local_bg, offset) = &self.locals_bind_groups.mesh2local_bind[&name.0.clone()];
            render_pass.set_bind_group(2, local_bg.bg(), &[*offset]);

            render_pass.set_vertex_buffer(0, verts.buf.slice(..));
            render_pass.set_vertex_buffer(1, uvs.buf.slice(..));
            render_pass.set_vertex_buffer(2, normals.buf.slice(..));
            render_pass.set_vertex_buffer(3, tangents.buf.slice(..));
            render_pass.set_vertex_buffer(4, colors.buf.slice(..));
            render_pass.set_index_buffer(faces.buf.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..faces.nr_triangles * 3, 0, 0..1);
        }
    }

    fn begin_pass(&mut self) {}

    fn input_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("compose_input_layout")
            // diffuse cubemap
            .add_entry_cubemap(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            //specular cubemap
            .add_entry_cubemap(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            //shadow_maps
            .add_entries_tex(
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Depth, /* float textures cannot be linearly filtered on webgpu :( But you can use a sampler with
                                                 * comparison and it does a hardware 2x2 pcf */
                // wgpu::TextureSampleType::Float { filterable: false }, //float textures cannot be linearly filtered on webgpu :(
                MAX_NUM_SHADOWS,
            )
            .build()
    }

    fn update_input_bind_group(&mut self, gpu: &Gpu, scene: &Scene, per_frame_uniforms: &PerFrameUniforms) {
        //get envmap
        let env_map = scene.get_resource::<&EnvironmentMapGpu>().unwrap();

        let mut shadow_maps = Vec::new();

        //attempt adding shadow maps for all the lights, first the real lights that are
        // actually in the scene and afterwards we will with dummy values
        for i in 0..MAX_NUM_SHADOWS {
            let is_within_valid_lights: bool = i < per_frame_uniforms.idx_ubo2light.len();
            if is_within_valid_lights && scene.world.has::<ShadowCaster>(per_frame_uniforms.idx_ubo2light[i]).unwrap() {
                let entity = per_frame_uniforms.idx_ubo2light[i];
                let shadow = scene
                    .get_comp::<&ShadowMap>(&entity)
                    .expect("The lights who have a ShadowCaster should also have ShadowMap at this point.");
                shadow_maps.push(shadow);
            } else {
                //dummy passthough shadowmap
                let shadow: gloss_hecs::Ref<'_, ShadowMap> = self
                    .local_scene
                    .get_comp::<&ShadowMap>(&self.passthrough_light.entity)
                    .expect("Dummy light should have ShadowMap");
                shadow_maps.push(shadow);
            }
        }

        let entries = BindGroupBuilder::new()
            .add_entry_tex(&env_map.diffuse_tex)
            .add_entry_tex(&env_map.specular_tex)
            .add_entry_tex(&shadow_maps[0].tex_depth)
            .add_entry_tex(&shadow_maps[1].tex_depth)
            .add_entry_tex(&shadow_maps[2].tex_depth)
            .build_entries();
        let stale = self.input_bind_group.as_ref().map_or(true, |b| b.is_stale(&entries)); //returns true if the bg has not been created or if stale
        if stale {
            debug!("compose input bind group is stale, recreating");
            self.input_bind_group = Some(BindGroupDesc::new("compose_input_bg", entries).into_bind_group_wrapper(gpu.device(), &self.input_layout));
        }
    }

    /// update the local information that need to be sent to the gpu for each
    /// mesh like te model matrix
    fn update_locals(&mut self, gpu: &Gpu, scene: &Scene) {
        Self::update_locals_inner::<Locals, _>(
            gpu,
            scene,
            &mut self.locals_uniform,
            &mut self.locals_bind_groups,
            &mut Self::query_state(scene),
        );
    }
}

/// Keep in sync with shader `gbuffer_mesh.wgsl`
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct Locals {
    model_matrix: nalgebra::Matrix4<f32>,
    color_type: i32,
    solid_color: nalgebra::Vector4<f32>,
    metalness: f32,
    perceptual_roughness: f32,
    roughness_black_lvl: f32,
    uv_scale: f32,
    is_floor: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    // pad_b: f32,
    pad_c: f32,
    pad_d: f32,
}
impl LocalEntData for Locals {
    fn new(entity: Entity, scene: &Scene) -> Self {
        let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap().0.to_homogeneous();
        let vis_mesh = scene.get_comp::<&VisMesh>(&entity).unwrap();
        let color_type = vis_mesh.color_type as i32;
        let is_floor = if let Some(floor) = scene.get_floor() {
            floor.entity == entity
        } else {
            false
        };
        let is_floor = u32::from(is_floor);
        Locals {
            model_matrix,
            color_type,
            solid_color: vis_mesh.solid_color,
            metalness: vis_mesh.metalness,
            perceptual_roughness: vis_mesh.perceptual_roughness,
            roughness_black_lvl: vis_mesh.roughness_black_lvl,
            uv_scale: vis_mesh.uv_scale,
            is_floor,
            pad_c: 0.0,
            pad_d: 0.0,
        }
    }
}

struct LocalsBindGroups {
    layout: wgpu::BindGroupLayout,
    pub mesh2local_bind: HashMap<String, (BindGroupWrapper, u32)>,
}
impl BindGroupCollection for LocalsBindGroups {
    fn new(gpu: &Gpu) -> Self {
        Self {
            layout: Self::build_layout_desc().into_bind_group_layout(gpu.device()),
            mesh2local_bind: HashMap::default(),
        }
    }

    //keep as associated function so we can call it in the pipeline creation
    // without and object
    fn build_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("gbuffer_pass_locals_layout")
            //locals
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                true,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<Locals>()).unwrap(), 256))),
            )
            //diffuse tex
            .add_entry_tex(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            //normal tex
            .add_entry_tex(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            //roughness tex
            .add_entry_tex(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            .build()
    }

    fn update_bind_group(&mut self, entity: Entity, gpu: &Gpu, mesh_name: &str, ubo: &Buffer, offset_in_ubo: u32, scene: &Scene) {
        //extract textures for this entity
        let diffuse_tex = &scene.get_comp::<&DiffuseTex>(&entity).unwrap().0;
        let normal_tex = &scene.get_comp::<&NormalTex>(&entity).unwrap().0;
        let roughness_tex = &scene.get_comp::<&RoughnessTex>(&entity).unwrap().0;

        let entries = BindGroupBuilder::new()
            .add_entry_buf_chunk::<Locals>(&ubo.buffer)
            .add_entry_tex(diffuse_tex)
            .add_entry_tex(normal_tex)
            .add_entry_tex(roughness_tex)
            .build_entries();

        self.update_if_stale(mesh_name, entries, offset_in_ubo, gpu);
    }

    fn get_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
    fn get_mut_entity2binds(&mut self) -> &mut HashMap<String, (BindGroupWrapper, u32)> {
        &mut self.mesh2local_bind
    }
}
