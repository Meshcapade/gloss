use encase::{
    self,
    private::{AlignmentValue, Writer},
};
use wgpu;

#[cfg(feature = "burn-torch")]
use crate::error::CudaInteropError;
#[cfg(feature = "burn-torch")]
use cust_raw;

#[cfg(feature = "burn-torch")]
use std::sync::Arc;
#[cfg(feature = "burn-torch")]
use tch::Tensor;
#[cfg(feature = "burn-torch")]
use wgpu_cuda_interop::{vulkan_wgpu_interop::WgpuBufferCudaMem, AllocSize};

#[cfg(feature = "burn-torch")]
use log::debug;

/// A wrapper for `wgpu::Buffer`. Allows writing of aligned or packed data into
/// it
#[derive(Clone)]
pub struct Buffer {
    pub buffer: wgpu::Buffer,
    pub size_bytes: usize,
    cpu_byte_buffer: Vec<u8>,
    offset: usize,
    alignment: AlignmentValue,

    //optional stuff for vulkan-cuda interop
    #[cfg(feature = "burn-torch")]
    pub staging_buffer_backed_by_cuda_mem: Option<Arc<WgpuBufferCudaMem>>,
}

impl Buffer {
    pub fn new_empty(device: &wgpu::Device, usage: wgpu::BufferUsages, label: wgpu::Label, size_bytes: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: size_bytes as u64,
            usage,
            mapped_at_creation: false,
        });

        let cpu_byte_buffer = Vec::new();

        Self {
            buffer,
            size_bytes,
            //for a packed one
            cpu_byte_buffer,
            offset: 0,
            alignment: AlignmentValue::new(256),
            #[cfg(feature = "burn-torch")]
            staging_buffer_backed_by_cuda_mem: None,
        }
    }

    pub fn new_from_buffer(buffer: wgpu::Buffer) -> Self {
        let size_bytes = usize::try_from(buffer.size()).unwrap();
        let cpu_byte_buffer = Vec::new();

        Self {
            buffer,
            size_bytes,
            //for a packed one
            cpu_byte_buffer,
            offset: 0,
            alignment: AlignmentValue::new(256),
            #[cfg(feature = "burn-torch")]
            staging_buffer_backed_by_cuda_mem: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }

    pub fn write_buffer(&mut self, queue: &wgpu::Queue, offset_bytes: usize, data: &[u8]) {
        queue.write_buffer(&self.buffer, offset_bytes as u64, bytemuck::cast_slice(data));
    }

    /// Writes data to the `cpu_byte_buffer` and adds padding in order to align
    /// to 256 bytes. This is useful for adding local data of each mesh into a
    /// gpu buffer and having the bind groups with an offset into the buffer. #
    /// Panics Will panic if the `cpu_byte_buffer` is too small to write the
    /// current chunk
    pub fn push_cpu_chunk_aligned<T: encase::ShaderType + encase::internal::WriteInto>(&mut self, chunk: &T) -> u32 {
        let offset = self.offset;
        let mut writer = Writer::new(chunk, &mut self.cpu_byte_buffer, offset).unwrap();
        chunk.write_into(&mut writer);
        // self.offset_packed += chunk.size().get() as usize;
        self.offset += usize::try_from(self.alignment.round_up(chunk.size().get())).unwrap();
        u32::try_from(offset).unwrap()
    }

    /// Writes data to the `cpu_byte_buffer` but adds no padding. This is useful
    /// when adding lots of structures to the buffer and they need to be exposed
    /// as a var<uniform> structs : array<MyStruct,20> to the shader. Useful for
    /// adding a vector of lights for example # Panics
    /// Will panic if the `cpu_byte_buffer` is too small to write the current
    /// chunk
    pub fn push_cpu_chunk_packed<T: encase::ShaderType + encase::internal::WriteInto>(&mut self, chunk: &T) {
        let offset = self.offset;
        let mut writer = Writer::new(chunk, &mut self.cpu_byte_buffer, offset).unwrap();
        chunk.write_into(&mut writer);
        self.offset += usize::try_from(chunk.size().get()).unwrap();
    }

    /// Uploads from the `cpu_byte_buffer` to gpu
    pub fn upload_from_cpu_chunks(&mut self, queue: &wgpu::Queue) {
        //write byte_buffers to gpu
        queue.write_buffer(&self.buffer, 0, self.cpu_byte_buffer.as_slice());
    }

    /// Sets the offset back to 0 so we can start a new round of upload from the
    /// beggining of the buffer
    pub fn reset_chunks_offset(&mut self) {
        self.offset = 0;
    }

    /// Sets the offset back to 0 only if we are approaching the end of the
    /// buffer. Usually we don't want to always reset to 0 since that
    /// subpart of the buffer may be used for rendering when we want to write to
    /// it, rather we want to use as most of the buffer as possible without
    pub fn reset_chunks_offset_if_necessary(&mut self) {
        //this is quite conservative, essentially we are using only half of the buffer
        if self.offset > self.size_bytes / 2 {
            self.offset = 0;
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    #[cfg(feature = "burn-torch")]
    pub fn new_from_tensor(
        tensor: &Tensor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        usage: wgpu::BufferUsages,
        label: wgpu::Label,
    ) -> Self {
        // create dummy Buffer first
        let mut buffer = Self::new_empty(device, usage, label, 4);

        //copy te tensor data into it (will recreate the inner buffer if the size is different)
        buffer
            .copy_from_tensor(tensor, device, queue, adapter)
            .expect("Failed to copy from tensor");

        buffer
    }

    #[cfg(feature = "burn-torch")]
    pub fn copy_from_tensor(
        &mut self,
        tensor: &Tensor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
    ) -> Result<(), CudaInteropError> {
        // Check tensor requirements
        if tensor.size().is_empty() || tensor.size()[0] != 1 {
            return Err(CudaInteropError::InvalidBatchSize(tensor.size().get(0).copied().unwrap_or(0) as usize));
        }

        // Compute buffer size and stride
        let shape = tensor.size();
        let elem_size = match tensor.kind() {
            tch::Kind::Float => std::mem::size_of::<f32>(),
            tch::Kind::Int => std::mem::size_of::<i32>(),
            _ => {
                return Err(CudaInteropError::InvalidTensorType(tensor.kind()));
            }
        };
        let num_elements: i64 = shape.iter().skip(1).product(); // skip batch dim
        let num_elements = usize::try_from(num_elements).unwrap();
        let buf_size = AllocSize {
            height: 1,
            width: 1,
            stride: num_elements * elem_size,
        };

        // Check for contiguous tensor
        if !tensor.is_contiguous() {
            return Err(CudaInteropError::InvalidNonContiguous);
        }

        // Recreate staging buffer if necessary
        if self.staging_buffer_backed_by_cuda_mem.is_none()
            || self.staging_buffer_backed_by_cuda_mem.as_ref().unwrap().cuda_mem.alloc_size != buf_size
        {
            debug!("staging_buffer_backed_by_cuda_mem creating because it is none or the size is different");
            let wgpu_cuda = wgpu_cuda_interop::interop::create_wgpu_cuda_buffer(device, adapter, buf_size, wgpu::BufferUsages::COPY_SRC);
            self.staging_buffer_backed_by_cuda_mem = Some(Arc::new(wgpu_cuda));
        }

        // recreate the buffer if the size is different
        if self.size_bytes != buf_size.stride {
            debug!("recreating the wgpu buffer because the size is different");
            self.size_bytes = buf_size.stride;
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer::from_tensor wgpu buffer"),
                size: self.size_bytes as u64,
                usage: self.buffer.usage(),
                mapped_at_creation: false,
            });
        }

        // Copy tensor data to staging buffer and then to wgpu buffer
        let source_ptr = tensor.data_ptr() as cust_raw::CUdeviceptr;
        if let Some(staging_buffer) = self.staging_buffer_backed_by_cuda_mem.as_ref() {
            // Copy from CUDA memory to staging buffer
            wgpu_cuda_interop::interop::cuda_buffer_to_wgpu(source_ptr, buf_size, staging_buffer, &self.buffer, device, queue);
        }

        Ok(())
    }
}
