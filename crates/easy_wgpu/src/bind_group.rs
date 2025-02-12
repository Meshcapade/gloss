use smallvec::SmallVec;
use wgpu;

use crate::texture::Texture;

pub fn align(size: usize, alignment: usize) -> usize {
    ((size + alignment - 1) / alignment) * alignment
}

/// Since we want the `BindGroupWrapper` to keep a vector of the ids and the ids
/// are all typed, we use an enum to deal with heterogeneous types
#[derive(PartialEq, Clone)]
pub enum BgEntriesId {
    Tex(wgpu::Id<wgpu::Texture>),
    Buf(wgpu::Id<wgpu::Buffer>),
    Sampler(wgpu::Id<wgpu::Sampler>),
}

/// Wrapper for a bind group that also keeps the ids of the entries. This helps
/// with keeping track if the textures in the entries have changed and the bind
/// group needs to be recreated
pub struct BindGroupWrapper {
    bind_group: wgpu::BindGroup,
    ids: SmallVec<[BgEntriesId; 16]>,
}
impl<'a> BindGroupWrapper {
    fn new(bind_group: wgpu::BindGroup, ids: SmallVec<[BgEntriesId; 16]>) -> Self {
        Self { bind_group, ids }
    }
    pub fn bg(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    pub fn is_stale(&self, entries: &SmallVec<[BindGroupEntry<'a>; 16]>) -> bool {
        if self.ids.len() != entries.len() {
            return true;
        }

        for it in entries.iter().zip(self.ids.iter()) {
            let (entry, id) = it;
            //if the ids don't match we return true because the binding group needs to be
            // recreated
            if entry.id != *id {
                return true;
            }
        }

        false
    }
}

/// Stores both an entry and the ``global_id`` of the resources it points to in
/// order to track if the bind group is stale Required wgpu to enable the
/// expose-ids feature
pub struct BindGroupEntry<'a> {
    pub entry: wgpu::BindGroupEntry<'a>,
    pub id: BgEntriesId,
}

/// Describes a bind group a series of entries
pub struct BindGroupDesc<'a> {
    pub label: Option<String>,
    pub bind_group_entries: SmallVec<[BindGroupEntry<'a>; 16]>,
    last_binding_number: u32,
}
impl Default for BindGroupDesc<'_> {
    fn default() -> Self {
        Self {
            label: None,
            bind_group_entries: SmallVec::new(),
            last_binding_number: 0,
        }
    }
}
impl<'a> BindGroupDesc<'a> {
    pub fn new(label: &str, entries: SmallVec<[BindGroupEntry<'a>; 16]>) -> Self {
        Self {
            label: Some(String::from(label)),
            bind_group_entries: entries,
            last_binding_number: 0,
        }
    }
    pub fn into_bind_group_wrapper(self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> BindGroupWrapper {
        //make the vector of bg entries
        let mut vec_entries: SmallVec<[wgpu::BindGroupEntry; 16]> = SmallVec::new();
        let mut ids: SmallVec<[BgEntriesId; 16]> = SmallVec::new();
        for bg_entry in self.bind_group_entries {
            //moves self.bg_entries and invalidates them
            vec_entries.push(bg_entry.entry);
            ids.push(bg_entry.id);
        }
        //create bg
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            // entries: self.bind_group_entries.as_slice(),
            entries: vec_entries.as_slice(),
            label: self.label.as_deref(),
        });
        BindGroupWrapper::new(bind_group, ids) //TODO add the ids of the
                                               // texture entries
    }
}

/// Builder for bind groups
/// IMPORTANT: order of adding entries sets also the binding index that should
/// correspond with the layout and the shader
pub struct BindGroupBuilder<'a> {
    bind_group_desc: Option<BindGroupDesc<'a>>,
}
impl<'a> Default for BindGroupBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a> BindGroupBuilder<'a> {
    pub fn new() -> Self {
        Self {
            bind_group_desc: Some(BindGroupDesc::default()),
        }
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn label(mut self, label: &str) -> Self {
        self.bind_group_desc.as_mut().unwrap().label = Some(String::from(label));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_empty(mut self) -> Self {
        self.bind_group_desc.as_mut().unwrap().last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_tex(mut self, tex: &'a Texture) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.bind_group_desc.as_ref().unwrap().last_binding_number;
        //entry and id
        let entry = BindGroupEntry {
            entry: wgpu::BindGroupEntry {
                binding: binding_number,
                resource: wgpu::BindingResource::TextureView(&tex.view),
            },
            id: BgEntriesId::Tex(tex.texture.global_id()),
        };
        //add
        self.bind_group_desc.as_mut().unwrap().bind_group_entries.push(entry);
        self.bind_group_desc.as_mut().unwrap().last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_buf(mut self, buffer: &'a wgpu::Buffer) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.bind_group_desc.as_ref().unwrap().last_binding_number;
        //entry and id
        let entry = BindGroupEntry {
            entry: wgpu::BindGroupEntry {
                binding: binding_number,
                resource: buffer.as_entire_binding(),
            },
            id: BgEntriesId::Buf(buffer.global_id()),
        };
        //add
        self.bind_group_desc.as_mut().unwrap().bind_group_entries.push(entry);
        self.bind_group_desc.as_mut().unwrap().last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_buf_chunk<T>(mut self, buffer: &'a wgpu::Buffer) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.bind_group_desc.as_ref().unwrap().last_binding_number;
        //chunk of the buffer
        let binding = wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: wgpu::BufferSize::new(u64::try_from(align(std::mem::size_of::<T>(), 256)).unwrap()),
        };
        //entry and id
        let entry = BindGroupEntry {
            entry: wgpu::BindGroupEntry {
                binding: binding_number,
                resource: wgpu::BindingResource::Buffer(binding),
            },
            id: BgEntriesId::Buf(buffer.global_id()),
        };
        //add
        self.bind_group_desc.as_mut().unwrap().bind_group_entries.push(entry);
        self.bind_group_desc.as_mut().unwrap().last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_sampler(mut self, sampler: &'a wgpu::Sampler) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.bind_group_desc.as_ref().unwrap().last_binding_number;
        //entry and id
        let entry = BindGroupEntry {
            entry: wgpu::BindGroupEntry {
                binding: binding_number,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            id: BgEntriesId::Sampler(sampler.global_id()),
        };
        // add
        self.bind_group_desc.as_mut().unwrap().bind_group_entries.push(entry);
        self.bind_group_desc.as_mut().unwrap().last_binding_number += 1;
        self
    }

    // pub fn get_or_build(
    //     &mut self,
    //     device: &wgpu::Device,
    //     layout: &wgpu::BindGroupLayout,
    //     old_bind_group: Option<BindGroupWrapper>,
    // ) -> BindGroupWrapper {
    //     //check if it's stale by comparing if the id's match
    //     let stale = true;
    //     let ret = if stale {
    //         let desc = self.bind_group_desc.take().unwrap();
    //         desc.into_bind_group_wrapper(device, layout)
    //     } else {
    //         old_bind_group.unwrap()
    //     };
    //     ret
    // }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> BindGroupWrapper {
        let desc = self.bind_group_desc.take().unwrap();
        desc.into_bind_group_wrapper(device, layout)
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build_bind_group(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        let desc = self.bind_group_desc.take().unwrap();
        desc.into_bind_group_wrapper(device, layout).bind_group
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build_entries(&mut self) -> SmallVec<[BindGroupEntry<'a>; 16]> {
        self.bind_group_desc.take().unwrap().bind_group_entries
    }
}
