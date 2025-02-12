use encase::{
    self,
    private::{AlignmentValue, Writer},
};
use wgpu;

/// A wrapper for `wgpu::Buffer`. Allows writing of aligned or packed data into
/// it
pub struct Buffer {
    pub buffer: wgpu::Buffer,
    pub size_bytes: usize,
    cpu_byte_buffer: Vec<u8>,
    offset: usize,
    alignment: AlignmentValue,
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
}
