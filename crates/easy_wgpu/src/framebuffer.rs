use crate::texture::{TexParams, Texture};
use enum_map::EnumMap;

/// Contains all the render targets used inside a framebuffer and their formats.
/// Uses [enum-map](https://docs.rs/enum-map/latest/enum_map/struct.EnumMap.html) to create render targets that you can later reference using the enum
/// # Example
/// ```no_run
/// use easy_wgpu::{framebuffer::FrameBufferBuilder, texture::TexParams};
/// use enum_map::{Enum, EnumMap};
/// use pollster::FutureExt;
/// use wgpu;
///
/// let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .block_on()
///     .unwrap();
/// let (device, queue) = adapter
///     .request_device(&wgpu::DeviceDescriptor::default(), None)
///     .block_on()
///     .unwrap();
///
/// #[derive(Debug, Enum)]
/// pub enum Target {
///     Albedo,
///     Depth,
/// }
/// let fb_builder = FrameBufferBuilder::<Target>::new(128, 128);
/// let framebuffer = fb_builder
///     .add_render_target(
///         &device,
///         Target::Albedo,
///         wgpu::TextureFormat::Rgba8Unorm,
///         wgpu::TextureUsages::RENDER_ATTACHMENT,
///         TexParams::default(),
///     )
///     .add_render_target(
///         &device,
///         Target::Depth,
///         wgpu::TextureFormat::Depth32Float,
///         wgpu::TextureUsages::RENDER_ATTACHMENT,
///         TexParams::default(),
///     )
///     .build(&device);
/// //access gbuffer textures using the enum
/// let tex = framebuffer.get(Target::Albedo);
/// ```
pub struct FrameBuffer<T: enum_map::EnumArray<Option<Texture>>> {
    targets: EnumMap<T, Option<Texture>>, //acts like a map of keys and values where the keys are elements of the enum and values are the textures
    pub width: u32,
    pub height: u32,
    pub bind_group_layout: wgpu::BindGroupLayout,
    //TODO binding might not always be necesary if we don't want to read from the textures inside a shader. However for a gbuffer it might be
    // necessary
    pub bind_group: Option<wgpu::BindGroup>,
    // pub requires_clear: bool, //the first time we write to the gbuffer we do loadop clear, all the subsequent times we write within the same frame
    // we will just write to gbuffer without clearing
}
impl<T: enum_map::EnumArray<Option<Texture>> + std::fmt::Debug> FrameBuffer<T> {
    // dissallow to call new on a gbuffer outside of this module
    pub(self) fn new(device: &wgpu::Device, targets: EnumMap<T, Option<Texture>>, width: u32, height: u32, create_bind_group: bool) -> Self {
        //bind layout
        let mut layout_entries = Vec::new();
        for (idx, tex) in targets.values().enumerate() {
            if let Some(tex) = tex {
                //creates changes the sample type to depth if it was created with depth format
                let mut sample_type = wgpu::TextureSampleType::Float { filterable: false };
                if tex.texture.format().is_depth_stencil_format() {
                    sample_type = wgpu::TextureSampleType::Depth;
                }

                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: u32::try_from(idx).unwrap(),
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        multisampled: tex.texture.sample_count() > 1,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type,
                    },
                    count: None,
                });
            }
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer Bind Group Layout"),
            entries: layout_entries.as_slice(),
        });

        //we keep this as a associated function as we can call it from within new

        let bind_group = if create_bind_group {
            Some(Self::create_bind_group(device, &targets, &layout))
        } else {
            None
        };

        Self {
            targets,
            width,
            height,
            // requires_clear: true,
            bind_group_layout: layout,
            bind_group,
        }
    }

    pub fn get(&self, target_type: T) -> Option<&Texture> {
        let tex = self.targets[target_type].as_ref();
        tex
    }

    pub fn get_mut(&mut self, target_type: T) -> Option<&mut Texture> {
        let tex = self.targets[target_type].as_mut();
        tex
    }

    #[allow(clippy::missing_panics_doc)] //really should not panic
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // resize all textures
        // recreate the bind_group
        for tex in self.targets.values_mut() {
            let tex = tex.as_mut().unwrap();
            tex.resize(device, width, height);
        }
        self.width = width;
        self.height = height;

        if self.bind_group.is_some() {
            self.bind_group = Some(Self::create_bind_group(device, &self.targets, &self.bind_group_layout));
        }
    }

    //keep as associated function so we can call it from new
    fn create_bind_group(device: &wgpu::Device, targets: &EnumMap<T, Option<Texture>>, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        let mut bind_group_entries = Vec::new();
        for (idx, tex) in targets.values().enumerate() {
            //TODO do we need this view when the texture is depth?
            // wgpu::BindGroupEntry {
            //         binding: 2,
            //         resource: wgpu::BindingResource::TextureView(&depth_tex.create_view(
            //             &wgpu::TextureViewDescriptor {
            //                 aspect: wgpu::TextureAspect::DepthOnly,
            //                 ..Default::default()
            //             },
            //         )),
            //     }
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(idx).unwrap(),
                resource: wgpu::BindingResource::TextureView(&tex.as_ref().unwrap().view),
            });
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: bind_group_entries.as_slice(),
            label: Some("gbuffer group"),
        });
        bind_group
    }
}

pub struct FrameBufferBuilder<T: enum_map::EnumArray<Option<Texture>>> {
    targets: EnumMap<T, Option<Texture>>, //acts like a map of keys and values where the keys are elements of the enum and values are the textures
    pub width: u32,
    pub height: u32,
    pub create_bind_group: bool,
}
impl<T: enum_map::EnumArray<Option<Texture>> + std::fmt::Debug> FrameBufferBuilder<T> {
    pub fn new(width: u32, height: u32) -> Self {
        let targets = EnumMap::default();
        Self {
            targets,
            width,
            height,
            create_bind_group: false,
        }
    }

    /// # Panics
    /// Will panic if texture usage is empty
    #[must_use]
    pub fn add_render_target(
        mut self,
        device: &wgpu::Device,
        target_type: T,
        format: wgpu::TextureFormat,
        usages: wgpu::TextureUsages,
        tex_params: TexParams,
    ) -> Self {
        assert_ne!(usages, wgpu::TextureUsages::empty(), "Texture usage cannot be empty");
        // let usages = wgpu::TextureUsages::RENDER_ATTACHMENT
        //     | wgpu::TextureUsages::TEXTURE_BINDING
        //     | additional_usages;
        let tex = Texture::new(device, self.width, self.height, format, usages, tex_params);
        self.targets[target_type] = Some(tex);
        self
    }

    #[must_use]
    pub fn create_bind_group(mut self) -> Self {
        self.create_bind_group = true;
        self
    }

    /// # Panics
    /// Will panic if no targets were added to the gbuffer
    pub fn build(self, device: &wgpu::Device) -> FrameBuffer<T> {
        assert!(
            self.targets.len() != 0,
            "You haven't assigned any render targets. You have to add render targets using add_render_target()"
        );
        FrameBuffer::new(device, self.targets, self.width, self.height, self.create_bind_group)
    }
}
