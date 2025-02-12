// TODO: We have a crate for this now

use crate::numerical;
use image::{ImageBuffer, Luma, Rgba};
use wgpu::TextureFormat;

pub enum DynamicImageBuffer {
    RgbaImage(ImageBuffer<Rgba<u8>, Vec<u8>>),
    LumaImage(ImageBuffer<Luma<f32>, Vec<f32>>),
}

pub struct DynamicImage {
    pub buffer: DynamicImageBuffer,
    pub format: TextureFormat,
}

pub fn from_rgba_image(rgba_image: ImageBuffer<Rgba<u8>, Vec<u8>>) -> DynamicImageBuffer {
    DynamicImageBuffer::RgbaImage(rgba_image)
}

pub fn from_luma_image(luma_image: ImageBuffer<Luma<f32>, Vec<f32>>) -> DynamicImageBuffer {
    DynamicImageBuffer::LumaImage(luma_image)
}

impl DynamicImage {
    pub fn new(width: u32, height: u32, format: TextureFormat) -> Self {
        let buffer = match format {
            TextureFormat::Rgba8Unorm => {
                from_rgba_image(ImageBuffer::<Rgba<u8>, _>::new(width, height))
            }
            TextureFormat::Depth32Float => {
                from_luma_image(ImageBuffer::<Luma<f32>, _>::new(width, height))
            }
            _ => panic!("Texture format not implemented!"),
        };
        Self { buffer, format }
    }

    pub fn as_rgba_image(&mut self) -> Option<&ImageBuffer<Rgba<u8>, Vec<u8>>> {
        match &self.buffer {
            DynamicImageBuffer::RgbaImage(rgba_image) => Some(rgba_image),
            DynamicImageBuffer::LumaImage(_) => None,
        }
    }
    pub fn as_luma_image(&mut self) -> Option<&ImageBuffer<Luma<f32>, Vec<f32>>> {
        match &self.buffer {
            DynamicImageBuffer::LumaImage(luma_image) => Some(luma_image),
            DynamicImageBuffer::RgbaImage(_) => None,
        }
    }
    pub fn copy_from_bytes(&mut self, width: u32, height: u32, copy_from: &[u8]) {
        let num_floats = copy_from.len() / std::mem::size_of::<f32>();
        let mut f32_values = vec![0.0; num_floats];
        numerical::convert_to_f32(copy_from, f32_values.as_mut_slice());

        match self.format {
            TextureFormat::Rgba8Unorm => {
                self.buffer = from_rgba_image(
                    ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, copy_from.to_vec())
                        .unwrap(),
                );
            }
            TextureFormat::Depth32Float => {
                self.buffer = from_luma_image(
                    ImageBuffer::<Luma<f32>, _>::from_raw(width, height, f32_values).unwrap(),
                );
            }
            _ => panic!("Texture format not implemented!"),
        };
    }
    // pub fn numpy(&mut self) -> Py<PyArray3<f32>> {
    // }
    // For depth remapping
    // pub fn depth_remap_reverse_z(&mut self, near: f32, far: f32) -> Py<PyArray2<f32>> {
    // }

    pub fn save(&mut self, path: &str) {
        match self.format {
            TextureFormat::Rgba8Unorm => {
                let rgba_img_buffer = self.as_rgba_image().unwrap();
                _ = rgba_img_buffer.save(path);
            }
            // TextureFormat::Depth32Float => {
            //     let luma_img_buffer = self.as_luma_image().unwrap().as_flat_samples();
            //     _ = luma_img_buffer.save(path);
            // }
            _ => panic!("Texture format not implemented!"),
        }
    }
}
