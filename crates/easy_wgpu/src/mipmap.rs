//mostly from
// https://github.com/jshrake/wgpu-mipmap/blob/main/src/backends/render.rs

use crate::utils::get_mip_extent;
use std::collections::HashMap;
use thiserror::Error;
use wgpu::{
    util::make_spirv, AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoder, Device, FilterMode, FragmentState, FrontFace, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PrimitiveState, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerDescriptor,
    ShaderModuleDescriptor, Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

/// Generates mipmaps for textures with output attachment usage.
#[derive(Debug)]
pub struct RenderMipmapGenerator {
    sampler: Sampler,
    layout_cache: HashMap<TextureSampleType, BindGroupLayout>,
    pipeline_cache: HashMap<TextureFormat, RenderPipeline>,
}

#[allow(clippy::match_same_arms)]
fn to_sample_type(format: TextureFormat) -> TextureSampleType {
    match format {
        TextureFormat::R8Uint
        | TextureFormat::R16Uint
        | TextureFormat::Rg8Uint
        | TextureFormat::R32Uint
        | TextureFormat::Rg16Uint
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rg32Uint
        | TextureFormat::Rgba16Uint
        | TextureFormat::Rgba32Uint => TextureSampleType::Uint,

        TextureFormat::R8Sint
        | TextureFormat::R16Sint
        | TextureFormat::Rg8Sint
        | TextureFormat::R32Sint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba32Sint => TextureSampleType::Sint,

        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R16Float
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::R32Float
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Float
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba16Float
        | TextureFormat::Rgba32Float
        | TextureFormat::Depth32Float
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8
        | TextureFormat::Bc1RgbaUnorm
        | TextureFormat::Bc1RgbaUnormSrgb
        | TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb
        | TextureFormat::Bc4RUnorm
        | TextureFormat::Bc4RSnorm
        | TextureFormat::Bc5RgUnorm
        | TextureFormat::Bc5RgSnorm
        | TextureFormat::Bc6hRgbUfloat
        // | TextureFormat::Bc6hRgbSfloat
        | TextureFormat::Bc7RgbaUnorm
        | TextureFormat::Bc7RgbaUnormSrgb
        // | TextureFormat::Etc2RgbUnorm
        // | TextureFormat::Etc2RgbUnormSrgb
        // | TextureFormat::Etc2RgbA1Unorm
        // | TextureFormat::Etc2RgbA1UnormSrgb
        // | TextureFormat::Etc2RgbA8Unorm
        // | TextureFormat::Etc2RgbA8UnormSrgb
        // | TextureFormat::EacRUnorm
        // | TextureFormat::EacRSnorm
        // | TextureFormat::EtcRgUnorm
        // | TextureFormat::EtcRgSnorm
        // | TextureFormat::Astc4x4RgbaUnorm
        // | TextureFormat::Astc4x4RgbaUnormSrgb
        // | TextureFormat::Astc5x4RgbaUnorm
        // | TextureFormat::Astc5x4RgbaUnormSrgb
        // | TextureFormat::Astc5x5RgbaUnorm
        // | TextureFormat::Astc5x5RgbaUnormSrgb
        // | TextureFormat::Astc6x5RgbaUnorm
        // | TextureFormat::Astc6x5RgbaUnormSrgb
        // | TextureFormat::Astc6x6RgbaUnorm
        // | TextureFormat::Astc6x6RgbaUnormSrgb
        // | TextureFormat::Astc8x5RgbaUnorm
        // | TextureFormat::Astc8x5RgbaUnormSrgb
        // | TextureFormat::Astc8x6RgbaUnorm
        // | TextureFormat::Astc8x6RgbaUnormSrgb
        // | TextureFormat::Astc10x5RgbaUnorm
        // | TextureFormat::Astc10x5RgbaUnormSrgb
        // | TextureFormat::Astc10x6RgbaUnorm
        // | TextureFormat::Astc10x6RgbaUnormSrgb
        // | TextureFormat::Astc8x8RgbaUnorm
        // | TextureFormat::Astc8x8RgbaUnormSrgb
        // | TextureFormat::Astc10x8RgbaUnorm
        // | TextureFormat::Astc10x8RgbaUnormSrgb
        // | TextureFormat::Astc10x10RgbaUnorm
        // | TextureFormat::Astc10x10RgbaUnormSrgb
        // | TextureFormat::Astc12x10RgbaUnorm
        // | TextureFormat::Astc12x10RgbaUnormSrgb
        // | TextureFormat::Astc12x12RgbaUnorm
        // | TextureFormat::Astc12x12RgbaUnormSrgb
         => TextureSampleType::Float { filterable: true },

        _ => TextureSampleType::Float { filterable: true },
    }
}

impl RenderMipmapGenerator {
    /// Returns the texture usage `RenderMipmapGenerator` requires for mipmap
    /// generation.
    pub fn required_usage() -> TextureUsages {
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING
    }

    /// Creates a new `RenderMipmapGenerator`. Once created, it can be used
    /// repeatedly to generate mipmaps for any texture with format specified
    /// in `format_hints`.
    #[allow(clippy::too_many_lines)]
    pub fn new_with_format_hints(device: &Device, format_hints: &[TextureFormat]) -> Self {
        // A sampler for box filter with clamp to edge behavior
        // In practice, the final result may be implementation dependent
        // - [Vulkan](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#textures-texel-linear-filtering)
        // - [Metal](https://developer.apple.com/documentation/metal/mtlsamplerminmagfilter/linear)
        // - [DX12](https://docs.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_filter)
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("wgpu-mipmap-sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let render_layout_cache = {
            let mut layout_cache = HashMap::new();
            // For now, we only cache a bind group layout for floating-point textures
            #[allow(clippy::single_element_loop)]
            for &sample_type in &[TextureSampleType::Float { filterable: true }] {
                let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(&format!("wgpu-mipmap-bg-layout-{sample_type:?}")),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                view_dimension: TextureViewDimension::D2,
                                sample_type,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
                layout_cache.insert(sample_type, bind_group_layout);
            }
            layout_cache
        };

        let render_pipeline_cache = {
            let mut pipeline_cache = HashMap::new();
            let vertex_module = device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: make_spirv(include_bytes!("../shaders/triangle.vert.spv")),
            });
            let box_filter = device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: make_spirv(include_bytes!("../shaders/box.frag.spv")),
            });
            for format in format_hints {
                let fragment_module = &box_filter;

                let sample_type = to_sample_type(*format);
                if let Some(bind_group_layout) = render_layout_cache.get(&sample_type) {
                    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[bind_group_layout],
                        push_constant_ranges: &[],
                    });
                    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                        label: Some(&format!("wgpu-mipmap-render-pipeline-{format:?}")),
                        layout: Some(&layout),
                        vertex: VertexState {
                            module: &vertex_module,
                            entry_point: "main",
                            buffers: &[],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        },
                        primitive: PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            front_face: FrontFace::Ccw,
                            cull_mode: Some(wgpu::Face::Back),
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: MultisampleState {
                            count: 1,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        fragment: Some(FragmentState {
                            module: fragment_module,
                            entry_point: "main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: *format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                        multiview: None,
                        cache: None,
                    });
                    pipeline_cache.insert(*format, pipeline);
                } else {
                    log::warn!("RenderMipmapGenerator does not support requested format {:?}", format);
                    continue;
                }
            }
            pipeline_cache
        };

        Self {
            sampler,
            layout_cache: render_layout_cache,
            pipeline_cache: render_pipeline_cache,
        }
    }

    /// Generate mipmaps from level 0 of `src_texture` to
    /// levels `dst_mip_offset..dst_texture_descriptor.mip_level_count`
    // of `dst_texture`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub(crate) fn generate_src_dst(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_texture: &Texture,
        dst_texture: &Texture,
        src_texture_descriptor: &TextureDescriptor,
        dst_texture_descriptor: &TextureDescriptor,
        dst_mip_offset: u32,
    ) -> Result<(), Error> {
        let src_format = src_texture_descriptor.format;
        let src_mip_count = src_texture_descriptor.mip_level_count;
        let src_ext = src_texture_descriptor.size;
        let src_dim = src_texture_descriptor.dimension;
        let src_usage = src_texture_descriptor.usage;
        let src_next_mip_ext = get_mip_extent(&src_ext, 1);

        let dst_format = dst_texture_descriptor.format;
        let dst_mip_count = dst_texture_descriptor.mip_level_count;
        let dst_ext = dst_texture_descriptor.size;
        let dst_dim = dst_texture_descriptor.dimension;
        let dst_usage = dst_texture_descriptor.usage;
        // invariants that we expect callers to uphold
        if src_format != dst_format {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture formats must be equal");
        }
        if src_dim != dst_dim {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture dimensions must be eqaul");
        }
        if !((src_mip_count == dst_mip_count && src_ext == dst_ext) || (src_next_mip_ext == dst_ext)) {
            dbg!(src_texture_descriptor);
            dbg!(dst_texture_descriptor);
            panic!("src and dst texture extents must match or dst must be half the size of src");
        }

        if src_dim != TextureDimension::D2 {
            return Err(Error::UnsupportedDimension(src_dim));
        }
        // src texture must be sampled
        if !src_usage.contains(TextureUsages::TEXTURE_BINDING) {
            return Err(Error::UnsupportedUsage(src_usage));
        }
        // dst texture must be sampled and output attachment
        if !dst_usage.contains(Self::required_usage()) {
            return Err(Error::UnsupportedUsage(dst_usage));
        }
        let format = src_format;
        let pipeline = self.pipeline_cache.get(&format).ok_or(Error::UnknownFormat(format))?;
        let sample_type = to_sample_type(format);
        let layout = self.layout_cache.get(&sample_type).ok_or(Error::UnknownFormat(format))?;
        let views = (0..src_mip_count)
            .map(|mip_level| {
                // The first view is mip level 0 of the src texture
                // Subsequent views are for the dst_texture
                let (texture, base_mip_level) = if mip_level == 0 {
                    (src_texture, 0)
                } else {
                    (dst_texture, mip_level - dst_mip_offset)
                };
                texture.create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: TextureAspect::All,
                    base_mip_level,
                    mip_level_count: Some(1),
                    array_layer_count: None,
                    base_array_layer: 0,
                })
            })
            .collect::<Vec<_>>();
        for mip in 1..src_mip_count as usize {
            let src_view = &views[mip - 1];
            let dst_view = &views[mip];
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(src_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

impl RenderMipmapGenerator {
    /// # Errors
    ///
    /// Will return `Err` if the texture cannot be mipmapped
    pub fn generate(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        texture_descriptor: &TextureDescriptor,
    ) -> Result<(), Error> {
        self.generate_src_dst(device, encoder, texture, texture, texture_descriptor, texture_descriptor, 0)
    }
}

/// An error that occurred during mipmap generation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("Unsupported texture usage `{0:?}`.\nYour texture usage must contain one of: 1. TextureUsage::STORAGE, 2. TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED, 3. TextureUsage::COPY_SRC | TextureUsage::COPY_DST")]
    UnsupportedUsage(wgpu::TextureUsages),
    #[error("Unsupported texture dimension `{0:?}. You texture dimension must be TextureDimension::D2`")]
    UnsupportedDimension(wgpu::TextureDimension),
    #[error("Unsupported texture format `{0:?}`. Try using the render backend.")]
    UnsupportedFormat(wgpu::TextureFormat),
    #[error("Unsupported texture size. Texture size must be a power of 2.")]
    NpotTexture,
    #[error("Unknown texture format `{0:?}`.\nDid you mean to specify it in `MipmapGeneratorDescriptor::formats`?")]
    UnknownFormat(wgpu::TextureFormat),
}
