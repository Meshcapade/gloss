use easy_wgpu::{
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
    gpu::Gpu,
};
use gloss_hecs::Query;

use crate::{
    components::Name,
    config::RenderConfig,
    forward_renderer::{bind_group_collection::BindGroupCollection, locals::LocalEntData},
    scene::Scene,
};

use super::upload_pass::PerFrameUniforms;

pub trait PipelineRunner {
    type QueryItems<'r>: Query;
    type QueryState<'r>;

    fn query_state(scene: &Scene) -> Self::QueryState<'_>;
    fn prepare<'a>(&mut self, gpu: &Gpu, per_frame_uniforms: &PerFrameUniforms, scene: &'a Scene) -> Self::QueryState<'a>;
    fn run<'r>(
        &'r mut self,
        render_pass: &mut wgpu::RenderPass<'r>,
        per_frame_uniforms: &'r PerFrameUniforms,
        _render_params: &RenderConfig,
        query_state: &'r mut Self::QueryState<'_>,
    );
    fn begin_pass(&mut self);
    fn input_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new().label("empty_layout").build()
    }
    //is optional if there is no input to this pipeline so by default can be empty
    fn update_input_bind_group(&mut self, _gpu: &Gpu, _scene: &Scene, _per_frame_uniforms: &PerFrameUniforms) {}
    fn update_locals(&mut self, gpu: &Gpu, scene: &Scene);

    fn update_locals_inner<Locals: LocalEntData + encase::ShaderType + encase::internal::WriteInto, Q: Query>(
        gpu: &Gpu,
        scene: &Scene,
        locals_uniform: &mut Buffer,
        locals_bind_groups: &mut impl BindGroupCollection,
        query_state: &mut gloss_hecs::QueryBorrow<'_, Q>,
    ) {
        locals_uniform.reset_chunks_offset_if_necessary();
        // Update the local binding groups for the meshes we render. We do it in two
        // passes because the binding group cannot be created and consumed in the same
        // loop
        for (id, _comps) in query_state.iter() {
            let name = scene.get_comp::<&Name>(&id).unwrap().0.clone();

            // upload local stuff
            let locals = Locals::new(id, scene);
            let offset_in_ubo = locals_uniform.push_cpu_chunk_aligned::<Locals>(&locals);

            // chekc if we need to recreate bind group (for example when any of the textures
            // of the meshes have changed)
            locals_bind_groups.update_bind_group(id, gpu, &name, locals_uniform, offset_in_ubo, scene);
        }
        locals_uniform.upload_from_cpu_chunks(gpu.queue()); //important to
                                                            // upload everything
                                                            // to gpu at the end
    }
}
