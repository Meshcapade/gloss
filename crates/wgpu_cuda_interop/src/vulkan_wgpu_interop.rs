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
impl WgpuBufferCudaMem {
    pub fn new(device: &Device, vk_buffer_cuda: VkBufferCudaMem, size: u64, additional_usages: wgpu::BufferUsages) -> Self {
        let wgpu_buffer = wrap_vk_buffer_with_wgpu(device, vk_buffer_cuda.buffer, size, additional_usages);

        WgpuBufferCudaMem {
            buffer: wgpu_buffer,
            cuda_mem: vk_buffer_cuda.cuda_mem.clone(), //justr an ARC clone,
            vk_buffer: Arc::new(vk_buffer_cuda),       //also keeps this alive since if it dies we lose the memory
        }
    }
}

// converts a raw vk_buffer into a wgpu buffer
pub fn wrap_vk_buffer_with_wgpu(device: &Device, buffer: vk::Buffer, size: u64, additional_usages: wgpu::BufferUsages) -> wgpu::Buffer {
    let buffer: wgpu_hal::vulkan::Buffer = unsafe { wgpu::hal::vulkan::Device::buffer_from_raw(buffer) };

    unsafe {
        device.create_buffer_from_hal::<Vulkan>(
            buffer,
            &wgpu::BufferDescriptor {
                label: None,
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE | additional_usages,
            },
        )
    }
}
