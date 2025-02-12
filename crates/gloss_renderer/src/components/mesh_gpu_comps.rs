//! These represent GPU components which are usually added automatically and
//! uploaded from CPU to GPU from the corresponding CPU components. When
//! creating entities, you usually just add CPU components.

use easy_wgpu::texture::TexParams;
use ktx2;
use wgpu;

extern crate nalgebra as na;
#[derive(Debug)]
pub struct VertsGPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl VertsGPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
    pub fn vertex_buffer_layout_instanced<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
}

#[derive(Debug)]
pub struct EdgesV1GPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl EdgesV1GPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
    pub fn vertex_buffer_layout_instanced<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
}
#[derive(Debug)]
pub struct EdgesV2GPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl EdgesV2GPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
    pub fn vertex_buffer_layout_instanced<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
}

pub struct UVsGPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl UVsGPU {
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (2 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x2
            ],
        }
    }
}
pub struct NormalsGPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl NormalsGPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
}

pub struct TangentsGPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl TangentsGPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            //we make it 4 because we need to also store the handetness
            array_stride: (4 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x4
            ],
        }
    }
}

pub struct ColorsGPU {
    pub buf: wgpu::Buffer,
    pub nr_vertices: u32,
}
impl ColorsGPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
    pub fn vertex_buffer_layout_instanced<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x3
            ],
        }
    }
}
#[derive(Debug)]
pub struct EdgesGPU {
    pub buf: wgpu::Buffer,
    pub nr_edges: u32,
}
impl EdgesGPU {
    //returning a vertex layout witha  dynamic shader location //https://github.com/gfx-rs/wgpu/discussions/2050
    pub fn vertex_buffer_layout<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (2 * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x2
            ],
        }
    }
    pub fn vertex_buffer_layout_instanced<const SHADER_LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (2 * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
            SHADER_LOCATION => Float32x2
            ],
        }
    }
}

pub struct FacesGPU {
    pub buf: wgpu::Buffer,
    pub nr_triangles: u32,
}

pub struct DiffuseTex(pub easy_wgpu::texture::Texture);
pub struct NormalTex(pub easy_wgpu::texture::Texture);
pub struct MetalnessTex(pub easy_wgpu::texture::Texture);
pub struct RoughnessTex(pub easy_wgpu::texture::Texture);

pub struct EnvironmentMapGpu {
    pub diffuse_tex: easy_wgpu::texture::Texture,
    pub specular_tex: easy_wgpu::texture::Texture,
}
impl EnvironmentMapGpu {
    pub fn new_dummy(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let diffuse_tex = easy_wgpu::texture::Texture::create_default_cubemap(device, queue);
        let specular_tex = easy_wgpu::texture::Texture::create_default_cubemap(device, queue);
        Self { diffuse_tex, specular_tex }
    }

    /// # Panics
    /// Will panic if the environment map is not the expected format
    #[allow(clippy::similar_names)]
    pub fn reader2texture(reader: &ktx2::Reader<&[u8]>, device: &wgpu::Device, queue: &wgpu::Queue) -> easy_wgpu::texture::Texture {
        // Get general texture information.
        let header = reader.header();
        assert_eq!(header.format, Some(ktx2::Format::R16G16B16A16_SFLOAT));
        let nr_mips = header.level_count;
        let width = header.pixel_width;
        let height = header.pixel_height;
        // println!("nr mips {}", nr_mips);
        // println!("header is {:?}", header);

        //make texture
        let usages = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 6, //it's a cube txture so it's always 6 layers
            },
            mip_level_count: nr_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: usages,
            view_formats: &[],
        };
        let tex_params = TexParams::from_desc(&desc);
        let texture = device.create_texture(&desc); //create with all mips but upload only 1 mip

        //TODO maybe possible to upload all the mips at the same time using reader.data
        // which gives you all the mips

        // Read iterator over slices of each mipmap level.
        let levels = reader.levels().collect::<Vec<_>>();
        for (mip, level_data) in levels.iter().enumerate() {
            easy_wgpu::texture::Texture::upload_single_mip(&texture, device, queue, &desc, level_data, None, u32::try_from(mip).unwrap());
        }

        //view and sampler
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

        easy_wgpu::texture::Texture {
            texture,
            view,
            sampler,
            tex_params,
        }
    }
}

//implement some atributes for the vertex atributes so we can use them in a
// generic function
pub trait GpuAtrib {
    fn data_ref(&self) -> &wgpu::Buffer;
    fn new_from(buf: wgpu::Buffer, nr_rows: u32) -> Self;
}

impl GpuAtrib for VertsGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for EdgesV1GPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for EdgesV2GPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for EdgesGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_edges: u32) -> Self {
        Self { buf, nr_edges }
    }
}
impl GpuAtrib for FacesGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_triangles: u32) -> Self {
        Self { buf, nr_triangles }
    }
}
impl GpuAtrib for UVsGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for NormalsGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for TangentsGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}
impl GpuAtrib for ColorsGPU {
    fn data_ref(&self) -> &wgpu::Buffer {
        &self.buf
    }
    fn new_from(buf: wgpu::Buffer, nr_vertices: u32) -> Self {
        Self { buf, nr_vertices }
    }
}

//so we can use the Components inside the Mutex<Hashmap> in the scene and wasm
// https://stackoverflow.com/a/73773940/22166964
// shenanigans
//vertsgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for VertsGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for VertsGPU {}
//vertsgpue1
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EdgesV1GPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EdgesV1GPU {}
//vertsgpue2
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EdgesV2GPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EdgesV2GPU {}
//edgesgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EdgesGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EdgesGPU {}
//facesgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for FacesGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for FacesGPU {}
//uvsgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for UVsGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for UVsGPU {}
//normalsgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for NormalsGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for NormalsGPU {}
//tangentsgpu
#[cfg(target_arch = "wasm32")]
unsafe impl Send for TangentsGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for TangentsGPU {}
//colors
#[cfg(target_arch = "wasm32")]
unsafe impl Send for ColorsGPU {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for ColorsGPU {}
//Diffusetex
#[cfg(target_arch = "wasm32")]
unsafe impl Send for DiffuseTex {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for DiffuseTex {}
//Normaltex
#[cfg(target_arch = "wasm32")]
unsafe impl Send for NormalTex {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for NormalTex {}
//Metalnesstex
#[cfg(target_arch = "wasm32")]
unsafe impl Send for MetalnessTex {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for MetalnessTex {}
//Roughnesstex
#[cfg(target_arch = "wasm32")]
unsafe impl Send for RoughnessTex {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for RoughnessTex {}
//EnvironmentMap
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EnvironmentMapGpu {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EnvironmentMapGpu {}
