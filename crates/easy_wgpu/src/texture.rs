use gloss_img::DynImage;
use image::imageops::FilterType;
// use image::GenericImage;
use image::{EncodableLayout, GenericImageView, ImageBuffer};
use log::{debug, warn};
use pollster::FutureExt;
use std::borrow::Cow;
use wgpu::{util::DeviceExt, CommandEncoderDescriptor, TextureFormat}; //enabled create_texture_with_data

// use gloss_utils::gloss_image;
use gloss_utils::numerical;

use crate::{buffer::Buffer, mipmap::RenderMipmapGenerator};

//aditional parameters for texture creation that usually you can leave as
// default
#[derive(Clone, Copy)]
pub struct TexParams {
    pub sample_count: u32,
    pub mip_level_count: u32,
}
impl Default for TexParams {
    fn default() -> Self {
        Self {
            sample_count: 1,
            mip_level_count: 1,
        }
    }
}
impl TexParams {
    pub fn from_desc(desc: &wgpu::TextureDescriptor) -> Self {
        Self {
            sample_count: desc.sample_count,
            mip_level_count: desc.mip_level_count,
        }
    }
    pub fn apply(&self, desc: &mut wgpu::TextureDescriptor) {
        desc.sample_count = self.sample_count;
        desc.mip_level_count = self.mip_level_count;
    }
}

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler, //TODO should be optional or rather we should create a nearest and linear sampler as a global per frame uniform
    // pub width: u32,
    // pub height: u32,
    // pub bind_group: Option<wgpu::BindGroup>, //cannot lazily create because it depends on the binding locations
    pub tex_params: TexParams,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        tex_params: TexParams,
    ) -> Self {
        debug!("New texture");
        // let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            usage,
            label: None,
            view_formats: &[],
        };
        tex_params.apply(&mut texture_desc);

        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tex_params,
            // width,
            // height,
            // bind_group: None,
        }
    }

    /// # Panics
    /// Will panic if bytes cannot be decoded into a image representation
    pub fn from_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str) -> Self {
        let img = image::load_from_memory(bytes).unwrap();
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::DynamicImage, label: Option<&str>) -> Self {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let desc = wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let tex_params = TexParams::from_desc(&desc);
        let texture = device.create_texture(&desc);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tex_params, /* width: dimensions.0,
                         * height: dimensions.1,
                         * bind_group: None, */
        }
    }

    /// reads image from format and into this texture
    /// if `is_srgb` is set then the reading will perform a conversion from
    /// gamma space to linear space when sampling the texture in a shader
    /// When writing to the texture, the opposite conversion takes place.
    /// # Panics
    /// Will panic if the path cannot be found
    pub fn from_path(path: &str, device: &wgpu::Device, queue: &wgpu::Queue, is_srgb: bool) -> Self {
        //read to cpu
        let img = image::ImageReader::open(path).unwrap().decode().unwrap();
        Self::from_img(
            &img.try_into().unwrap(),
            device,
            queue,
            is_srgb,
            true,
            false, //TODO what do we set as default here?
            None,
            None,
        )
    }

    /// # Panics
    /// Will panic if textures that have more than 1 byte per channel or more
    /// than 4 channels.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_img(
        img: &DynImage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        is_srgb: bool,
        generate_mipmaps: bool,
        mipmap_generation_cpu: bool,
        staging_buffer: Option<&Buffer>,
        mipmaper: Option<&RenderMipmapGenerator>,
    ) -> Self {
        let dimensions = img.dimensions();
        let nr_channels = img.color().channel_count();
        let bytes_per_channel = img.color().bytes_per_pixel() / nr_channels;
        assert!(bytes_per_channel == 1, "We are only supporting textures which have 1 byte per channel.");
        //convert 3 channels to 4 channels and keep 2 channels as 2 channels
        let img_vec;
        let img_buf = match nr_channels {
            1 | 2 | 4 => img.as_bytes(),
            3 => {
                img_vec = img.to_rgba8().into_vec();
                img_vec.as_bytes()
            }
            _ => panic!("Format with more than 4 channels not supported"),
        };

        let tex_format = Self::format_from_img(img, is_srgb);

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let mut nr_mip_maps = 1;
        let mut usages = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if generate_mipmaps {
            nr_mip_maps = size.max_mips(wgpu::TextureDimension::D2);
        }
        if mipmaper.is_some() && generate_mipmaps {
            usages |= RenderMipmapGenerator::required_usage();
        }

        let desc = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: nr_mip_maps,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: tex_format,
            usage: usages,
            view_formats: &[],
        };
        let tex_params = TexParams::from_desc(&desc);

        let texture = device.create_texture(&desc); //create with all mips but upload only 1 mip

        Self::upload_single_mip(&texture, device, queue, &desc, img_buf, staging_buffer, 0);

        //mipmaps
        if generate_mipmaps {
            Self::generate_mipmaps(
                img,
                &texture,
                device,
                queue,
                &desc,
                nr_mip_maps,
                mipmap_generation_cpu,
                staging_buffer,
                mipmaper,
            );
        }

        // let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            mip_level_count: Some(nr_mip_maps),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tex_params, /* width: dimensions.0,
                         * height: dimensions.1,
                         * bind_group: None, */
        }
    }

    /// # Panics
    /// Will panic if the image has more than 1 byte per channel
    #[allow(clippy::too_many_arguments)]
    pub fn update_from_img(
        &mut self,
        img: &DynImage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        is_srgb: bool,
        generate_mipmaps: bool,
        mipmap_generation_cpu: bool,
        staging_buffer: Option<&Buffer>,
        mipmaper: Option<&RenderMipmapGenerator>,
    ) {
        // let dimensions = img.dimensions();
        let nr_channels = img.color().channel_count();
        let bytes_per_channel = img.color().bytes_per_pixel() / nr_channels;
        assert!(bytes_per_channel == 1, "We are only supporting textures which have 1 byte per channel.");

        // TODO refactor this into its own func because there is a lot of duplication
        // with the from_img function convert 3 channels to 4 channels and keep
        // 2 channels as 2 channels
        let img_vec;
        let img_buf = match nr_channels {
            1 | 2 | 4 => img.as_bytes(),
            3 => {
                img_vec = img.to_rgba8().into_vec();
                img_vec.as_bytes()
            }
            _ => panic!("Format with more than 4 channels not supported"),
        };

        let size = Self::extent_from_img(img);
        let tex_format = Self::format_from_img(img, is_srgb);
        let mut nr_mip_maps = 1;
        let mut usages = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if generate_mipmaps {
            nr_mip_maps = size.max_mips(wgpu::TextureDimension::D2);
        }
        if mipmaper.is_some() && generate_mipmaps {
            usages |= RenderMipmapGenerator::required_usage();
        }

        let desc = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: nr_mip_maps,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: tex_format,
            usage: usages,
            view_formats: &[],
        };

        Self::upload_single_mip(&self.texture, device, queue, &desc, img_buf, staging_buffer, 0);

        //mipmaps
        if generate_mipmaps {
            Self::generate_mipmaps(
                img,
                &self.texture,
                device,
                queue,
                &desc,
                nr_mip_maps,
                mipmap_generation_cpu,
                staging_buffer,
                mipmaper,
            );
        }

        // let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            mip_level_count: Some(nr_mip_maps),
            ..Default::default()
        });

        //update
        self.view = view;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_mipmaps(
        img: &DynImage,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &wgpu::TextureDescriptor,
        nr_mip_maps: u32,
        mipmap_generation_cpu: bool,
        staging_buffer: Option<&Buffer>,
        mipmaper: Option<&RenderMipmapGenerator>,
    ) {
        let nr_channels = img.color().channel_count();
        if mipmap_generation_cpu {
            //CPU generation
            //similar to https://github.com/DGriffin91/bevy_mod_mipmap_generator/blob/main/src/lib.rs
            let mut img_mip = DynImage::new(1, 1, image::ColorType::L8);
            for mip_lvl in 1..nr_mip_maps {
                let mip_size = desc.mip_level_size(mip_lvl).unwrap();
                let prev_img_mip = if mip_lvl == 1 { img } else { &img_mip };
                img_mip = prev_img_mip.resize_exact(mip_size.width, mip_size.height, FilterType::Triangle);
                debug!("mip lvl {} has size {:?}", mip_lvl, mip_size);

                let img_mip_vec;
                let img_mip_buf = match nr_channels {
                    1 | 2 | 4 => img_mip.as_bytes(),
                    3 => {
                        img_mip_vec = img_mip.to_rgba8().into_vec();
                        img_mip_vec.as_bytes()
                    }
                    _ => panic!("Format with more than 4 channels not supported"),
                };

                Self::upload_single_mip(texture, device, queue, desc, img_mip_buf, staging_buffer, mip_lvl);
            }
        } else {
            //GPU mipmaps generation
            if let Some(mipmaper) = mipmaper {
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
                mipmaper.generate(device, &mut encoder, texture, desc).unwrap();
                queue.submit(std::iter::once(encoder.finish()));
            } else {
                warn!("Couldn't generate mipmaps since the mipmapper was not provided");
            }
        }
    }

    pub fn extent_from_img(img: &DynImage) -> wgpu::Extent3d {
        let dimensions = img.dimensions();
        wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        }
    }

    /// # Panics
    /// Will panic if the image has more than 1 byte per channel
    pub fn format_from_img(img: &DynImage, is_srgb: bool) -> wgpu::TextureFormat {
        let nr_channels = img.color().channel_count();
        let bytes_per_channel = img.color().bytes_per_pixel() / nr_channels;
        assert!(bytes_per_channel == 1, "We are only supporting textures which have 1 byte per channel.");

        //get a format for the texture
        let mut tex_format = match nr_channels {
            1 => wgpu::TextureFormat::R8Unorm,
            2 => wgpu::TextureFormat::Rg8Unorm,
            3 | 4 => wgpu::TextureFormat::Rgba8Unorm,
            _ => panic!("Format with more than 4 channels not supported"),
        };
        if is_srgb {
            tex_format = tex_format.add_srgb_suffix();
        }

        tex_format
    }

    /// Basically the same as `device.create_texture_with_data` but without the
    /// creation part and the data is assumed to contain only one mip # Panics
    /// Will panic if the data does not fit in the defined mipmaps described in
    /// textureDescriptor
    pub fn upload_single_mip(
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &wgpu::TextureDescriptor,
        data: &[u8],
        staging_buffer: Option<&Buffer>,
        mip: u32,
    ) {
        let mut mip_size = desc.mip_level_size(mip).unwrap();
        // copying layers separately
        if desc.dimension != wgpu::TextureDimension::D3 {
            mip_size.depth_or_array_layers = 1;
        }

        // Will return None only if it's a combined depth-stencil format
        // If so, default to 4, validation will fail later anyway since the depth or
        // stencil aspect needs to be written to individually
        let block_size = desc.format.block_copy_size(None).unwrap_or(4);
        let (block_width, block_height) = desc.format.block_dimensions();

        // When uploading mips of compressed textures and the mip is supposed to be
        // a size that isn't a multiple of the block size, the mip needs to be uploaded
        // as its "physical size" which is the size rounded up to the nearest block
        // size.
        let mip_physical = mip_size.physical_size(desc.format);

        // All these calculations are performed on the physical size as that's the
        // data that exists in the buffer.
        let width_blocks = mip_physical.width / block_width;
        let height_blocks = mip_physical.height / block_height;

        let bytes_per_row = width_blocks * block_size;
        // let data_size = bytes_per_row * height_blocks *
        // mip_size.depth_or_array_layers;

        // let end_offset = binary_offset + data_size as usize;

        if let Some(staging_buffer) = staging_buffer {
            warn!("Using slow CPU->GPU transfer for texture upload. Might use less memory that staging buffer using by wgpu but it will be slower.");

            //get some metadata
            let bytes_per_row_unpadded = texture.format().block_copy_size(None).unwrap() * mip_size.width;
            let bytes_per_row_padded = numerical::align(bytes_per_row_unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

            //map buffer and copy into it
            // https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#mapping-buffers
            //the mapping range has to be aligned to COPY_BUFFER_ALIGNMENT(4 bytes)
            let slice_size = numerical::align(u32::try_from(data.len()).unwrap(), u32::try_from(wgpu::COPY_BUFFER_ALIGNMENT).unwrap());
            {
                let buffer_slice = staging_buffer.buffer.slice(0..u64::from(slice_size));
                // NOTE: We have to create the mapping THEN device.poll() before await
                // the future. Otherwise the application will freeze.
                let (tx, rx) = futures::channel::oneshot::channel();
                buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
                    tx.send(result).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                rx.block_on().unwrap().unwrap();
                let mut buf_data = buffer_slice.get_mapped_range_mut();

                //copy into it
                buf_data.get_mut(0..data.len()).unwrap().clone_from_slice(data);
            }

            //finish
            staging_buffer.buffer.unmap();

            //copy from buffer to texture
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &staging_buffer.buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row_padded),
                        rows_per_image: Some(mip_size.height),
                    },
                },
                wgpu::ImageCopyTexture {
                    aspect: wgpu::TextureAspect::All,
                    texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::Extent3d {
                    width: mip_size.width,
                    height: mip_size.height,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(Some(encoder.finish()));

            //wait to finish because we might be reusing the staging buffer for
            // something else later TODO maybe this is not needed
            // since the mapping will block either way if the buffer is still in
            // use device.poll(wgpu::Maintain::Wait);
        } else {
            //Use wgpu write_texture which schedules internally the transfer to happen
            // later
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height_blocks),
                },
                mip_physical,
            );
        }

        //-=---------------------
    }

    /// Basically the same as `device.create_texture_with_data` but without the
    /// creation part Assumes the data contains info for all mips
    /// # Panics
    /// Will panic if the data does not fit in the defined mipmaps described in
    /// textureDescriptor
    pub fn upload_all_mips(
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &wgpu::TextureDescriptor,
        data: &[u8],
        staging_buffer: Option<&Buffer>,
    ) {
        // Will return None only if it's a combined depth-stencil format
        // If so, default to 4, validation will fail later anyway since the depth or
        // stencil aspect needs to be written to individually
        let block_size = desc.format.block_copy_size(None).unwrap_or(4);
        let (block_width, block_height) = desc.format.block_dimensions();
        let layer_iterations = desc.array_layer_count();

        let (min_mip, max_mip) = (0, desc.mip_level_count);

        let mut binary_offset = 0;
        for layer in 0..layer_iterations {
            for mip in min_mip..max_mip {
                let mut mip_size = desc.mip_level_size(mip).unwrap();
                // copying layers separately
                if desc.dimension != wgpu::TextureDimension::D3 {
                    mip_size.depth_or_array_layers = 1;
                }

                // When uploading mips of compressed textures and the mip is supposed to be
                // a size that isn't a multiple of the block size, the mip needs to be uploaded
                // as its "physical size" which is the size rounded up to the nearest block
                // size.
                let mip_physical = mip_size.physical_size(desc.format);

                // All these calculations are performed on the physical size as that's the
                // data that exists in the buffer.
                let width_blocks = mip_physical.width / block_width;
                let height_blocks = mip_physical.height / block_height;

                let bytes_per_row = width_blocks * block_size;
                let data_size = bytes_per_row * height_blocks * mip_size.depth_or_array_layers;

                let end_offset = binary_offset + data_size as usize;

                if let Some(staging_buffer) = staging_buffer {
                    warn!("Using slow CPU->GPU transfer for texture upload. Might use less memory that staging buffer using by wgpu but it will be slower.");

                    //get some metadata
                    let bytes_per_row_unpadded = texture.format().block_copy_size(None).unwrap() * mip_size.width;
                    let bytes_per_row_padded = numerical::align(bytes_per_row_unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

                    //map buffer and copy into it
                    // https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#mapping-buffers
                    let data_to_copy = &data[binary_offset..end_offset];
                    //the mapping range has to be aligned to COPY_BUFFER_ALIGNMENT(4 bytes)
                    let slice_size = numerical::align(
                        u32::try_from(data_to_copy.len()).unwrap(),
                        u32::try_from(wgpu::COPY_BUFFER_ALIGNMENT).unwrap(),
                    );
                    {
                        let buffer_slice = staging_buffer.buffer.slice(0..u64::from(slice_size));
                        // NOTE: We have to create the mapping THEN device.poll() before await
                        // the future. Otherwise the application will freeze.
                        let (tx, rx) = futures::channel::oneshot::channel();
                        buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
                            tx.send(result).unwrap();
                        });
                        device.poll(wgpu::Maintain::Wait);
                        rx.block_on().unwrap().unwrap();
                        let mut buf_data = buffer_slice.get_mapped_range_mut();

                        //copy into it
                        buf_data.get_mut(0..data_to_copy.len()).unwrap().clone_from_slice(data_to_copy);
                    }

                    //finish
                    staging_buffer.buffer.unmap();

                    //copy from buffer to texture
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBuffer {
                            buffer: &staging_buffer.buffer,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(bytes_per_row_padded),
                                rows_per_image: Some(mip_size.height),
                            },
                        },
                        wgpu::ImageCopyTexture {
                            aspect: wgpu::TextureAspect::All,
                            texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        wgpu::Extent3d {
                            width: mip_size.width,
                            height: mip_size.height,
                            depth_or_array_layers: 1,
                        },
                    );
                    queue.submit(Some(encoder.finish()));

                    //wait to finish because we might be reusing the staging
                    // buffer for something else later
                    // TODO maybe this is not needed since the mapping will
                    // block either way if the buffer is still in use
                    // device.poll(wgpu::Maintain::Wait);
                } else {
                    //Use wgpu write_texture which schedules internally the transfer to happen
                    // later
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &data[binary_offset..end_offset],
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: Some(height_blocks),
                        },
                        mip_physical,
                    );
                }

                binary_offset = end_offset;
            }
        }
    }

    pub fn upload_from_cpu_with_staging_buffer(
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &wgpu::TextureDescriptor,
        data: &[u8],
        staging_buffer: &Buffer,
        mip_lvl: u32,
    ) {
        let mip_size = desc.mip_level_size(mip_lvl).unwrap();

        //map buffer and copy into it
        // https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#mapping-buffers
        {
            let buffer_slice = staging_buffer.buffer.slice(0..data.len() as u64);
            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let (tx, rx) = futures::channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.block_on().unwrap().unwrap();
            let mut buf_data = buffer_slice.get_mapped_range_mut();

            //copy into it
            buf_data.clone_from_slice(data);
        }

        //finish
        staging_buffer.buffer.unmap();

        //get some metadata
        let bytes_per_row_unpadded = texture.format().block_copy_size(None).unwrap() * mip_size.width;
        let bytes_per_row_padded = numerical::align(bytes_per_row_unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

        //copy from buffer to texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_padded),
                    rows_per_image: Some(mip_size.height),
                },
            },
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture,
                mip_level: mip_lvl,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::Extent3d {
                width: mip_size.width,
                height: mip_size.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        //wait to finish because we might be reusing the staging buffer for something
        // else later
        device.poll(wgpu::Maintain::Wait);
    }

    pub async fn download_to_cpu(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> DynImage {
        // create buffer
        let bytes_per_row_unpadded = self.texture.format().block_copy_size(None).unwrap() * self.width();
        let bytes_per_row_padded = numerical::align(bytes_per_row_unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer_size = u64::from(bytes_per_row_padded * self.height());
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
        // this tells wpgu that we want to read this buffer from the cpu
        | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };

        let output_buffer = device.create_buffer(&output_buffer_desc);

        //copy from texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_padded),
                    rows_per_image: Some(self.height()),
                },
            },
            wgpu::Extent3d {
                width: self.width(),
                height: self.height(),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        // map and get to cpu
        // We need to scope the mapping variables so that we can unmap the buffer

        // let mut buffer = DynImage::new(self.width(), self.height(),
        // self.texture.format()); let mut buffer = match self.texture.format()
        // {     TextureFormat::Rgba8Unorm => DynImage::new_rgba8(self.width(),
        // self.height()),     TextureFormat::Depth32Float =>
        // DynImage::new_luma32f(self.width(), self.height()),     _ => panic!("
        // Texture format not implemented!"), };

        let img: Option<DynImage> = {
            let buffer_slice = output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            //TODO maybe change the future_intrusive to futures. Future_intrusive seems to
            // give some issues on wasm
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();

            //TODO remove padding and copy into image
            // https://github.com/rerun-io/rerun/blob/93146b6d04f8f494258901c8b892eee0bb31b1a8/crates/re_renderer/src/texture_info.rs#L57
            let data_unpadded = Texture::remove_padding(data.as_bytes(), bytes_per_row_unpadded, bytes_per_row_padded, self.height());

            // let copy_from = data_unpadded.as_bytes();
            // buffer.copy_from_bytes(self.width(), self.height(), copy_from);
            let w = self.width();
            let h = self.height();
            match self.texture.format() {
                TextureFormat::Rgba8Unorm => ImageBuffer::from_raw(w, h, data_unpadded.to_vec()).map(DynImage::ImageRgba8),
                TextureFormat::Bgra8Unorm => {
                    let bgra_data = data_unpadded.to_vec();
                    // Convert BGRA to RGBA by swapping channels
                    let mut rgba_data = bgra_data.clone();
                    for chunk in rgba_data.chunks_exact_mut(4) {
                        chunk.swap(0, 2); // Swap B and R
                    }
                    ImageBuffer::from_raw(w, h, rgba_data).map(DynImage::ImageRgba8)
                }
                TextureFormat::Rgba32Float => ImageBuffer::from_raw(w, h, numerical::u8_to_f32_vec(&data_unpadded)).map(DynImage::ImageRgba32F),
                TextureFormat::Depth32Float => ImageBuffer::from_raw(w, h, numerical::u8_to_f32_vec(&data_unpadded)).map(DynImage::ImageLuma32F),
                x => panic!("Texture format not implemented! {x:?}"),
            }
        };
        output_buffer.unmap();
        img.unwrap()
    }

    pub fn remove_padding(buffer: &[u8], bytes_per_row_unpadded: u32, bytes_per_row_padded: u32, nr_rows: u32) -> Cow<'_, [u8]> {
        // re_tracing::profile_function!();

        // assert_eq!(buffer.len() as wgpu::BufferAddress, self.buffer_size_padded);

        if bytes_per_row_padded == bytes_per_row_unpadded {
            return Cow::Borrowed(buffer);
        }

        let mut unpadded_buffer = Vec::with_capacity((bytes_per_row_unpadded * nr_rows) as _);

        for row in 0..nr_rows {
            let offset = (bytes_per_row_padded * row) as usize;
            unpadded_buffer.extend_from_slice(&buffer[offset..(offset + bytes_per_row_unpadded as usize)]);
        }

        unpadded_buffer.into()
    }

    pub fn create_bind_group_layout(device: &wgpu::Device, binding_tex: u32, binding_sampler: u32) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: binding_tex, //matches with the @binding in the shader
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: binding_sampler, //matches with the @binding in the shader
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        })
    }
    #[must_use]
    pub fn depth_linearize(&self, device: &wgpu::Device, queue: &wgpu::Queue, near: f32, far: f32) -> DynImage {
        //panics if depth map retrieval is attempted with MSAA sample count set to > 1
        assert!(
            !(self.texture.sample_count() > 1 && self.texture.format() == TextureFormat::Depth32Float),
            "InvalidSampleCount: Depth maps not supported for MSAA sample count {} (Use a config to set msaa_nr_samples as 1)",
            self.texture.sample_count()
        );

        let dynamic_img = pollster::block_on(self.download_to_cpu(device, queue));
        let w = dynamic_img.width();
        let h = dynamic_img.height();
        let c = dynamic_img.channels();
        assert!(c == 1, "Depth maps should have only 1 channel");

        let linearized_img = match dynamic_img {
            DynImage::ImageLuma32F(v) => {
                let img_vec_ndc = v.to_vec();
                let img_vec: Vec<f32> = img_vec_ndc.iter().map(|&x| numerical::linearize_depth_reverse_z(x, near, far)).collect();
                DynImage::ImageLuma32F(ImageBuffer::from_raw(w, h, img_vec).unwrap())
            }
            _ => panic!("Texture format not implemented for remap (Only for depths)!"),
        };
        linearized_img
    }

    pub fn create_bind_group(&self, device: &wgpu::Device, binding_tex: u32, binding_sampler: u32) -> wgpu::BindGroup {
        //create bind group
        //recreate the bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &Self::create_bind_group_layout(device, binding_tex, binding_sampler),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: binding_tex,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: binding_sampler,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("bind_group"),
        });
        bind_group
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        //essentially creates a whole new texture with the same format and usage
        let format = self.texture.format();
        let usage = self.texture.usage();
        let mut new = Self::new(device, width, height, format, usage, self.tex_params);
        std::mem::swap(self, &mut new);
    }

    //make a default 4x4 texture that can be used as a dummy texture
    pub fn create_default_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // //read to cpu
        // let img = ImageReader::open(path).unwrap().decode().unwrap();
        // let rgba = img.to_rgba8();

        //we make a 4x4 texture because some gbus don't allow 1x1 or 2x2 so 4x4 seems
        // to be the minimum allowed
        let width = 4;
        let height = 4;

        let mut img_data: Vec<u8> = Vec::new();
        for _ in 0..height {
            for _ in 0..width {
                //assume 4 channels
                img_data.push(255);
                img_data.push(0);
                img_data.push(0);
                img_data.push(0);
            }
        }

        // let rgba = img.to_rgba8();
        // let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let desc = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let tex_params = TexParams::from_desc(&desc);
        let texture = device.create_texture_with_data(queue, &desc, wgpu::util::TextureDataOrder::LayerMajor, img_data.as_slice());

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tex_params, /* width,
                         * height, */
        }
    }

    pub fn create_default_cubemap(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // //read to cpu
        // let img = ImageReader::open(path).unwrap().decode().unwrap();
        // let rgba = img.to_rgba8();

        //we make a 4x4 texture because some gbus don't allow 1x1 or 2x2 so 4x4 seems
        // to be the minimum allowed
        let width = 4;
        let height = 4;

        let mut img_data: Vec<u8> = Vec::new();
        for _ in 0..6 {
            for _ in 0..height {
                for _ in 0..width {
                    //assume 4 channels
                    img_data.push(255);
                    img_data.push(0);
                    img_data.push(0);
                    img_data.push(0);
                }
            }
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 6,
        };
        // let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let desc = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let tex_params = TexParams::from_desc(&desc);
        let texture = device.create_texture_with_data(queue, &desc, wgpu::util::TextureDataOrder::LayerMajor, img_data.as_slice());

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tex_params, /* width,
                         * height, */
        }
    }

    pub fn width(&self) -> u32 {
        self.texture.width()
    }
    pub fn height(&self) -> u32 {
        self.texture.height()
    }
    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width(),
            height: self.height(),
            depth_or_array_layers: 1,
        }
    }
    // pub fn clone(&self) -> Self {
    //     Self {
    //         texture: self.texture,
    //         view: (),
    //         sampler: (),
    //         width: (),
    //         height: (),
    //     }
    // }
}
