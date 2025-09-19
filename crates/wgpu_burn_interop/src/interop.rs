use core::panic;

use burn::tensor::{Int, Tensor, TensorMetadata};

use burn_cubecl::tensor::CubeTensor;
use cubecl::wgpu::WgpuRuntime;
use gloss_burn_multibackend::{backend::MultiBackend, tensor::MultiFloatTensor, tensor::MultiIntTensor};

pub fn tensor_float2wgpu_buffer(tensor: Tensor<MultiBackend, 2>, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    // Get underlying cube tensor.
    let cube_tensor = tensor.into_primitive().tensor();
    let MultiFloatTensor::Wgpu(cube_tensor) = cube_tensor else {
        panic!("Expected wgpu tensor got {:?}", cube_tensor.dtype())
    };

    cubewgpu_tensor2wgpu_buffer(cube_tensor, device, queue)
}

pub fn tensor_int2wgpu_buffer(tensor: Tensor<MultiBackend, 2, Int>, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    // Get underlying cube tensor.
    let cube_tensor = tensor.into_primitive();
    let MultiIntTensor::Wgpu(cube_tensor) = cube_tensor else {
        panic!("Expected wgpu tensor got {:?}", cube_tensor.dtype())
    };

    cubewgpu_tensor2wgpu_buffer(cube_tensor, device, queue)
}

fn cubewgpu_tensor2wgpu_buffer(tensor: CubeTensor<WgpuRuntime>, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    // Get the 'resource' from the client
    let client = tensor.client;
    let binding = client.get_resource(tensor.handle.clone().binding());
    let resource = binding.resource();

    // Which has the wgpu buffer.
    let buffer = resource.buffer();

    // But do note it only uses a part of the buffer, see offset + size.
    let offset = resource.offset();
    let size = resource.size();

    // Client buffers the pending work, so flush first in order to make sure it's queued.
    // no need to sync since since on the same queue as wgpu submission submission queue (assuming you are using the same device between burn and wgpu) (see crate wgpu_burn_global_device)
    client.flush();
    // client.sync().block_on();

    // Create destination buffer
    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tensor2wgpu_buffer_dst"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });

    // Encode the copy
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tensor2wgpu_buffer_copy_encoder"),
    });

    encoder.copy_buffer_to_buffer(buffer, offset, &dst_buffer, 0, size);

    // Submit
    queue.submit(Some(encoder.finish()));

    dst_buffer
}
