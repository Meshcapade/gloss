use crate::cuda;
use crate::cuda_vulkan_interop;
use crate::vulkan_wgpu_interop;
use crate::vulkan_wgpu_interop::WgpuBufferCudaMem;
use crate::AllocSize;

// Copies an image from a cuda device pointer to a wgpu texture using a staging buffer that is backed by cuda memory
pub fn cuda_img_to_wgpu(
    source_ptr: cust_raw::CUdeviceptr,
    img_size: AllocSize,
    staging_buffer: &WgpuBufferCudaMem,
    dst_texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    // cuda::cuda_synchronize(); //waits for whatever work was done to create source_ptr to be finished. TODO not sure if needed sicne this goes on the same queue as the cuda_2d_copy_on_device
    device.poll(wgpu::PollType::Wait).unwrap(); //waits for the staging buffer to not be in use anymore, so we can safely copy into it. Actually blocks untils all work of wgpu is done, ideally it would check only the staging buffer. TODO makes this wait only on staging buffer copies

    let tex_cuda_mem = &staging_buffer.cuda_mem;

    //copy from tensor to cuda shared memory (which is the same memory as the staging_buffer)
    let _ = cuda::cuda_2d_copy_on_device(
        img_size, //width, height, stride which is width*4 (os bytes of width)
        tex_cuda_mem.device_ptr,
        source_ptr,
        1,
        tex_cuda_mem.vulkan_pitch_alignment,
    );
    cuda::cuda_synchronize(); //waits for cuda to finish copying, now the buffer is filled

    // //copy from buffer to texture
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let size = wgpu::Extent3d {
        width: u32::try_from(img_size.width).unwrap(),
        height: u32::try_from(img_size.height).unwrap(),
        depth_or_array_layers: 1,
    };
    let output_stride = u32::try_from(img_size.stride).unwrap();
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padding = (align - output_stride % align) % align;
    let padded_stride = output_stride + padding;
    let in_buf = &staging_buffer.buffer;
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: in_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_stride),
                rows_per_image: None,
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
    // }
    let sub_index = queue.submit(Some(encoder.finish()));
    //for some reason without, we sometimes end up in a freeze or a crash
    device.poll(wgpu::PollType::WaitForSubmissionIndex(sub_index)).unwrap(); //TODO probably not needed
}

// Copies a buffer from a cuda device pointer to a wgpu buffer using a staging buffer that is backed by cuda memory
pub fn cuda_buffer_to_wgpu(
    source_ptr: cust_raw::CUdeviceptr,
    buf_size: AllocSize,
    staging_buffer: &WgpuBufferCudaMem,
    dst_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    // cuda::cuda_synchronize(); //waits for whatever work was done to create source_ptr to be finished. TODO not sure if needed sicne this goes on the same queue as the cuda_2d_copy_on_device
    device.poll(wgpu::PollType::Wait).unwrap(); //waits for the staging buffer to not be in use anymore, so we can safely copy into it. Actually blocks untils all work of wgpu is done, ideally it would check only the staging buffer. TODO makes this wait only on staging buffer copies

    let buf_cuda_mem = &staging_buffer.cuda_mem;

    //copy from tensor to cuda shared memory (which is the same memory as the staging_buffer)
    // For a generic buffer, set height=1, width=1, stride=total bytes
    let _ = crate::cuda::cuda_2d_copy_on_device(
        buf_size,
        buf_cuda_mem.device_ptr,
        source_ptr,
        1,                                   // dst_alignment
        buf_cuda_mem.vulkan_pitch_alignment, // src_alignment
    );
    cuda::cuda_synchronize(); //waits for cuda to finish copying, now the buffer is filled

    // Copy from staging buffer to wgpu buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&staging_buffer.buffer, 0, dst_buffer, 0, buf_size.stride as u64);

    //for some reason without, we sometimes end up in a freeze or a crash so we need to add this wait here
    let sub_index = queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::WaitForSubmissionIndex(sub_index)).unwrap(); //TODO why is this needed?
}

pub fn create_wgpu_cuda_buffer(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    size: AllocSize,
    additional_usages: wgpu::BufferUsages,
) -> WgpuBufferCudaMem {
    assert!(
        adapter.get_info().backend == wgpu::Backend::Vulkan,
        "cuda-wgpu interop only available for vulkan backend"
    );

    let vk_buffer_cuda_mem = cuda_vulkan_interop::VkBufferCudaMem::new(device, size);

    vulkan_wgpu_interop::WgpuBufferCudaMem::new(device, vk_buffer_cuda_mem, size.full_size_padded() as u64, additional_usages)
}
