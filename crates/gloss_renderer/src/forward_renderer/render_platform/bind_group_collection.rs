use std::collections::HashMap;

use easy_wgpu::{
    bind_group::{BindGroupDesc, BindGroupEntry, BindGroupWrapper},
    bind_group_layout::BindGroupLayoutDesc,
    buffer::Buffer,
    gpu::Gpu,
};
use gloss_hecs::Entity;
use smallvec::SmallVec;
use wgpu::BindGroupLayout;

use crate::scene::Scene;

/// trait that defines a collection of uniforms each with the same layout. Each
/// entity can be associated with a certain
pub trait BindGroupCollection {
    fn new(gpu: &Gpu) -> Self;
    fn build_layout_desc() -> BindGroupLayoutDesc;

    fn update_if_stale(&mut self, ent_name: &str, entries: SmallVec<[BindGroupEntry<'_>; 16]>, offset_in_ubo: u32, gpu: &Gpu) {
        //if there is no entry for the bind group or if the current one is stale, we
        // recreate it
        if !self.get_mut_entity2binds().contains_key(ent_name) || self.get_mut_entity2binds()[ent_name].0.is_stale(&entries) {
            let bg = BindGroupDesc::new("local_bg", entries).into_bind_group_wrapper(gpu.device(), self.get_layout());
            let bg_and_offset = (bg, offset_in_ubo);
            self.get_mut_entity2binds().insert(ent_name.to_string(), bg_and_offset);
        }

        //sometimes just the offset of the bind group changes so we also make sure to
        // update this.
        self.get_mut_entity2binds()
            .entry(ent_name.to_string())
            .and_modify(|r| r.1 = offset_in_ubo);
    }
    fn update_bind_group(&mut self, _entity: Entity, gpu: &Gpu, mesh_name: &str, ubo: &Buffer, offset_in_ubo: u32, _scene: &Scene);

    //getters
    fn get_layout(&self) -> &BindGroupLayout;
    fn get_mut_entity2binds(&mut self) -> &mut HashMap<String, (BindGroupWrapper, u32)>;
}
