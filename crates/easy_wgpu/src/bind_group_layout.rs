use core::num::NonZeroU64;

/// Convenience layout for the bind group. Can potentially be hashed and used in
/// a arena-kind of way
#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct BindGroupLayoutDesc {
    label: Option<String>,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}
impl BindGroupLayoutDesc {
    pub fn into_bind_group_layout(self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let desc = wgpu::BindGroupLayoutDescriptor {
            entries: self.entries.as_slice(),
            label: self.label.as_deref(),
        };
        device.create_bind_group_layout(&desc)
    }
    pub fn empty() -> Self {
        Self {
            label: Some(String::from("emtpy_bgl")),
            entries: Vec::new(),
        }
    }
}

/// Convenience builder to build the layout of a bind group
/// # Example
/// ```
///  # use easy_wgpu::bind_group_layout::BindGroupLayoutBuilder;
/// let desc = BindGroupLayoutBuilder::new()
///     .add_entry_uniform(
///         wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
///         false,
///         None,
///     )
///     .add_entry_sampler(
///         wgpu::ShaderStages::FRAGMENT,
///         wgpu::SamplerBindingType::NonFiltering,
///     )
///     .build();
/// ```

pub struct BindGroupLayoutBuilder {
    layout_desc: Option<BindGroupLayoutDesc>,
    last_binding_number: u32,
}
impl Default for BindGroupLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl BindGroupLayoutBuilder {
    pub fn new() -> Self {
        Self {
            layout_desc: Some(BindGroupLayoutDesc::empty()),
            last_binding_number: 0,
        }
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn label(mut self, label: &str) -> Self {
        self.layout_desc.as_mut().unwrap().label = Some(String::from(label));
        self
    }

    #[must_use]
    pub fn add_entry_empty(mut self) -> Self {
        self.last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_tex(mut self, visibility: wgpu::ShaderStages, sample_type: wgpu::TextureSampleType) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.last_binding_number;
        //entry and id
        let entry = wgpu::BindGroupLayoutEntry {
            binding: binding_number, //matches with the @binding in the shader
            visibility,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type,
            },
            count: None,
        };
        //add
        self.layout_desc.as_mut().unwrap().entries.push(entry);
        self.last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entries_tex(self, visibility: wgpu::ShaderStages, sample_type: wgpu::TextureSampleType, num_textures: usize) -> Self {
        let mut builder = self;
        for _i in 0..num_textures {
            builder = builder.add_entry_tex(visibility, sample_type);
        }
        builder
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_cubemap(mut self, visibility: wgpu::ShaderStages, sample_type: wgpu::TextureSampleType) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.last_binding_number;
        //entry and id
        let entry = wgpu::BindGroupLayoutEntry {
            binding: binding_number, //matches with the @binding in the shader
            visibility,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::Cube,
                sample_type,
            },
            count: None,
        };
        //add
        self.layout_desc.as_mut().unwrap().entries.push(entry);
        self.last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_sampler(mut self, visibility: wgpu::ShaderStages, sampler_type: wgpu::SamplerBindingType) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.last_binding_number;
        //entry and id
        let entry = wgpu::BindGroupLayoutEntry {
            binding: binding_number, //matches with the @binding in the shader
            visibility,
            ty: wgpu::BindingType::Sampler(sampler_type),
            count: None,
        };
        //add
        self.layout_desc.as_mut().unwrap().entries.push(entry);
        self.last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_entry_uniform(mut self, visibility: wgpu::ShaderStages, has_dynamic_offset: bool, _min_binding_size: Option<NonZeroU64>) -> Self {
        //each entry we add will have sequential binding_indices
        //this should correspond with the binding in the shader
        let binding_number = self.last_binding_number;
        // println!("BINDING NUM {binding_number}");
        // println!("MIN SIZE {:?}", min_binding_size);
        // println!("LABEL {:?}", self.layout_desc.clone().unwrap().label);

        //entry and id
        // TODO: Why does min_binding_size have an issue here?
        let entry = wgpu::BindGroupLayoutEntry {
            binding: binding_number, //----- keep in sync with the binding in create_bind_group and also the shader
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset, //important because we will change the offset intot his buffer for each mesh
                // min_binding_size,
                min_binding_size: None,
            },
            count: None,
        };
        //add
        self.layout_desc.as_mut().unwrap().entries.push(entry);
        self.last_binding_number += 1;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build(&mut self) -> BindGroupLayoutDesc {
        self.layout_desc.take().unwrap()
    }
}
