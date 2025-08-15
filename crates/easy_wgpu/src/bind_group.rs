use smallvec::SmallVec;
use wgpu;

use crate::texture::Texture;

pub fn align(size: usize, alignment: usize) -> usize {
    size.div_ceil(alignment) * alignment
}

/// Since we want the `BindGroupWrapper` to keep a vector of the types of resources that are bound,
/// we use an enum to deal with heterogeneous types
#[derive(PartialEq, Clone)]
pub enum BgResource {
    TexView(wgpu::TextureView),
    Buf(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

/// Wrapper for a bind group that also keep a handle on the resources in the bingroup. This helps
/// with keeping track if the textures in the entries have changed and the bind
/// group needs to be recreated
pub struct BindGroupWrapper {
    bind_group: wgpu::BindGroup,
    resources: SmallVec<[BgResource; 16]>,
}
impl BindGroupWrapper {
    fn new(bind_group: wgpu::BindGroup, resources: SmallVec<[BgResource; 16]>) -> Self {
        Self { bind_group, resources }
    }
    pub fn bg(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    pub fn is_stale(&self, entries: &SmallVec<[wgpu::BindGroupEntry; 16]>) -> bool {
        if self.resources.len() != entries.len() {
            return true;
        }

        for it in entries.iter().zip(self.resources.iter()) {
            let (entry, res) = it;
            let is_different = match &entry.resource {
                wgpu::BindingResource::Buffer(buffer_binding) => BgResource::Buf(buffer_binding.buffer.clone()) != *res,
                wgpu::BindingResource::Sampler(sampler) => BgResource::Sampler((*sampler).clone()) != *res,
                wgpu::BindingResource::TextureView(view) => BgResource::TexView((*view).clone()) != *res,
                _ => unimplemented!("other binding types"),
            };

            if is_different {
                return true;
            }
        }

        false
    }
}

/// Describes a bind group a series of entries
pub struct BindGroupDesc<'a> {
    pub label: Option<String>,
    pub bind_group_entries: SmallVec<[wgpu::BindGroupEntry<'a>; 16]>,
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
    pub fn new(label: &str, entries: SmallVec<[wgpu::BindGroupEntry<'a>; 16]>) -> Self {
        Self {
            label: Some(String::from(label)),
            bind_group_entries: entries,
            last_binding_number: 0,
        }
    }
    pub fn into_bind_group_wrapper(self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> BindGroupWrapper {
        //create bg
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: self.bind_group_entries.as_slice(),
            label: self.label.as_deref(),
        });
        //create some owned resources by cloning the texture,buffers,etc. Since its all Arc<> it's actually just a reference count
        let mut bg_resources = SmallVec::new();
        for res in self.bind_group_entries {
            match res.resource {
                wgpu::BindingResource::Buffer(buffer_binding) => bg_resources.push(BgResource::Buf(buffer_binding.buffer.clone())),
                wgpu::BindingResource::Sampler(sampler) => bg_resources.push(BgResource::Sampler(sampler.clone())),
                wgpu::BindingResource::TextureView(texture_view) => bg_resources.push(BgResource::TexView(texture_view.clone())),
                _ => todo!(),
            }
        }
        BindGroupWrapper::new(bind_group, bg_resources)
        // texture entries
    }
}

/// Builder for bind groups
/// IMPORTANT: order of adding entries sets also the binding index that should
/// correspond with the layout and the shader
pub struct BindGroupBuilder<'a> {
    bind_group_desc: Option<BindGroupDesc<'a>>,
}
impl Default for BindGroupBuilder<'_> {
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
        //entry
        let entry = wgpu::BindGroupEntry {
            binding: binding_number,
            resource: wgpu::BindingResource::TextureView(&tex.view),
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
        //entry
        let entry = wgpu::BindGroupEntry {
            binding: binding_number,
            resource: buffer.as_entire_binding(),
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
        //entry
        let entry = wgpu::BindGroupEntry {
            binding: binding_number,
            resource: wgpu::BindingResource::Buffer(binding),
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
        //entry
        let entry = wgpu::BindGroupEntry {
            binding: binding_number,
            resource: wgpu::BindingResource::Sampler(sampler),
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
    pub fn build_entries(&mut self) -> SmallVec<[wgpu::BindGroupEntry<'a>; 16]> {
        self.bind_group_desc.take().unwrap().bind_group_entries
    }
}
