use crate::cuda::allocate_shared_cuda_memory;
use crate::cuda::CudaSharedMemory;
use crate::AllocSize;
use crate::VulkanGpu;
use ash::vk::{self, BufferCreateInfo};
use std::sync::Arc;
use wgpu_hal::api::Vulkan;

//this is a vk buffer that is backed by cuda memory so changing data from cuda_mem.device_ptr will change the memory in the vulkan buffer
pub struct VkBufferCudaMem {
    pub buffer: vk::Buffer,
    pub cuda_mem: Arc<CudaSharedMemory>,
    // WGPU device handle (to access raw Vulkan for cleanup)
    device: wgpu::Device,
    // Raw Vulkan device memory imported from CUDA FD
    memory: vk::DeviceMemory,
}

impl Drop for VkBufferCudaMem {
    fn drop(&mut self) {
        unsafe {
            // // Destroy the vulkan buffer but this doesn't seem to be necessary and causes issue when WgpuCudaMem gets dropped
            // let _ = self.device.as_hal::<Vulkan, _, _>(|d| {
            //     d.map(|dv| dv.raw_device().destroy_buffer(self.buffer, None));
            // });

            // free the Vulkan device memory
            self.device.as_hal::<Vulkan, _, _>(|d| {
                if let Some(dv) = d {
                    dv.raw_device().free_memory(self.memory, None);
                }
            });
        }
    }
}

/// Wrap a CUDA shared memory file descriptor into a Vulkan buffer and imported device memory.
///
/// # Safety
/// This function is `unsafe` because it imports a raw file descriptor into Vulkan and operates on raw Vulkan handles;
/// callers must ensure that:
/// - `raw_gpu` contains valid and live Vulkan `instance/device/physical_device` handles compatible with imported external memory,
/// - `cuda_mem.shared_handle` is a valid file descriptor that refers to memory exported from CUDA and that duplicating/using the FD is permitted,
/// - the returned Vulkan buffer and device memory are used and destroyed in a way that does not outlive the underlying Vulkan objects or the CUDA-exported memory.
///   Incorrect usage can cause undefined behavior, resource leaks, or double-closing of file descriptors.
///
/// # Errors
/// Returns `Err(vk::Result)` when any Vulkan operation (buffer creation, memory allocation, or binding) fails; the error is propagated to the caller.
///
/// Note: Vulkan takes ownership of the imported FD; this function duplicates the FD before importing to avoid double-close of the original handle.
pub unsafe fn wrap_cuda_mem_with_vk_buffer(raw_gpu: &VulkanGpu, cuda_mem: &CudaSharedMemory) -> Result<(vk::Buffer, vk::DeviceMemory), vk::Result> {
    let raw_device = &raw_gpu.device;
    let raw_instance = &raw_gpu.instance;
    let physical_device = &raw_gpu.physical_device;

    let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

    //vulkan takes ownership of the shared FD handle, so if we don't duplicate it, the CudaMemShared will double-drop it. In order to avoid this, we duplicate it here, so one shared handle is for vulkan and one for cudamemshared
    let duplicated_fd = libc::fcntl(i32::try_from(cuda_mem.shared_handle).unwrap(), libc::F_DUPFD_CLOEXEC, 0);

    let mut import_memory_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle_type)
        .fd(duplicated_fd as std::ffi::c_int);

    let mut ext_create_info = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);

    let buffer_create_info = BufferCreateInfo::default()
        .push_next(&mut ext_create_info)
        .size(cuda_mem.cuda_alloc_size as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let raw_buffer = raw_device.create_buffer(&buffer_create_info, None)?;

    // figure out a suitable memory type index
    let mem_req = raw_device.get_buffer_memory_requirements(raw_buffer);
    let memory_type_index = pick_device_local_memory_type_raw(raw_instance, physical_device, mem_req.memory_type_bits);

    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(cuda_mem.cuda_alloc_size as u64)
        .push_next(&mut import_memory_info)
        .memory_type_index(memory_type_index);

    let allocated_memory = raw_device.allocate_memory(&allocate_info, None)?;

    raw_device.bind_buffer_memory(raw_buffer, allocated_memory, 0)?;

    Ok::<_, vk::Result>((raw_buffer, allocated_memory))
}

/// Create a Vulkan buffer backed by CUDA shared memory for the given WGPU device and allocation size.
///
/// # Errors
/// Returns an error if converting the provided `wgpu::Device` into its raw Vulkan handles fails
/// or if creating the underlying Vulkan buffer/device memory fails.
///
/// Note: this function assumes the `wgpu::Device` is using the Vulkan backend; the current
/// implementation calls `unwrap()` on the backend conversion and will panic if the device is not
/// backed by Vulkan.
pub fn create_vk_buffer_backed_by_cuda_memory(device: &wgpu::Device, size: AllocSize) -> Result<VkBufferCudaMem, Box<dyn std::error::Error>> {
    unsafe {
        let vk_cuda_buffer_mem = device
            .as_hal::<Vulkan, _, _>(|hal_device: Option<&wgpu_hal::vulkan::Device>| {
                hal_device.map(|hal_device| {
                    let raw_device = hal_device.raw_device();
                    let raw_instance = hal_device.shared_instance().raw_instance();
                    let physical_device = hal_device.raw_physical_device();
                    let raw_gpu = VulkanGpu {
                        device: raw_device.clone(),
                        instance: raw_instance.clone(),
                        physical_device,
                    };

                    create_vk_buffer_backed_by_cuda_memory_raw(device, &raw_gpu, size)
                })
            })
            .unwrap()?; // TODO: unwrap

        Ok(vk_cuda_buffer_mem)
    }
}

/// Create a Vulkan buffer backed by CUDA shared memory for the given WGPU device and allocation size.
///
/// # Errors
/// Returns an error if:
/// - allocating the shared CUDA memory fails (returned from `allocate_shared_cuda_memory`), or
/// - wrapping the CUDA memory into a Vulkan buffer/device memory fails (returned from `wrap_cuda_mem_with_vk_buffer`).
///   The error is returned boxed as `Box<dyn std::error::Error>`.
pub fn create_vk_buffer_backed_by_cuda_memory_raw(
    device: &wgpu::Device,
    raw_gpu: &VulkanGpu,
    size: AllocSize,
) -> Result<VkBufferCudaMem, Box<dyn std::error::Error>> {
    let cuda_mem = allocate_shared_cuda_memory(size)?;

    unsafe {
        let (raw_buffer, allocated_memory) = wrap_cuda_mem_with_vk_buffer(raw_gpu, &cuda_mem).unwrap();

        Ok(VkBufferCudaMem {
            buffer: raw_buffer,
            cuda_mem: Arc::new(cuda_mem),
            device: device.clone(),
            memory: allocated_memory,
        })
    }
}

// A helper to pick a `DEVICE_LOCAL` memory type
// fn pick_device_local_memory_type(device: &wgpu_hal::vulkan::Device, type_bits: u32) -> u32 {
//     pick_device_local_memory_type_raw(device.shared_instance().raw_instance(), &device.raw_physical_device(), type_bits)
// }

/// A helper to pick a `DEVICE_LOCAL` memory type
#[allow(clippy::trivially_copy_pass_by_ref)]
fn pick_device_local_memory_type_raw(raw_instance: &ash::Instance, physical_device: &ash::vk::PhysicalDevice, type_bits: u32) -> u32 {
    let mem_props = unsafe { raw_instance.get_physical_device_memory_properties(*physical_device) };
    for i in 0..mem_props.memory_type_count as usize {
        if (type_bits & (1 << i)) != 0 && mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return u32::try_from(i).unwrap();
        }
    }
    panic!("no DEVICE_LOCAL memory type found");
}
