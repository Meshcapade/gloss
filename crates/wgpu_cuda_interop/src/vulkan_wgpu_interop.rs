use ash::vk;
use wgpu::Device;
use wgpu_hal::api::Vulkan;

use crate::cuda::CudaSharedMemory;
use crate::cuda_vulkan_interop::VkBufferCudaMem;
use std::sync::Arc;

// #[derive(Clone)]
pub struct WgpuBufferCudaMem {
    pub buffer: wgpu::Buffer,
    pub cuda_mem: Arc<CudaSharedMemory>,
    pub vk_buffer: Arc<VkBufferCudaMem>,
}

pub fn create_wgpu_buffer_from_vk_buffer(
    device: &Device,
    buffer: vk::Buffer,
    size: u64,
    // usage: wgpu::BufferUsages,
    is_in: bool,
) -> wgpu::Buffer {
    let buffer: wgpu_hal::vulkan::Buffer = unsafe { wgpu::hal::vulkan::Device::buffer_from_raw(buffer) };

    unsafe {
        device.create_buffer_from_hal::<Vulkan>(
            buffer,
            &wgpu::BufferDescriptor {
                label: None,
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | (if is_in {
                        wgpu::BufferUsages::COPY_SRC
                    } else {
                        wgpu::BufferUsages::COPY_DST
                    }), // usage: usage,
            },
        )
    }
}

pub fn create_wgpu_cuda_buffer_from_vk_cuda_buffer(device: &Device, vk_buffer_cuda: VkBufferCudaMem, size: u64, is_in: bool) -> WgpuBufferCudaMem {
    let wgpu_buffer = create_wgpu_buffer_from_vk_buffer(device, vk_buffer_cuda.buffer, size, is_in);

    WgpuBufferCudaMem {
        buffer: wgpu_buffer,
        cuda_mem: vk_buffer_cuda.cuda_mem.clone(), //justr an ARC clone,
        vk_buffer: Arc::new(vk_buffer_cuda),       //also keeps this alive since if it dies we lose the memory
    }
}
