#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_ext;
#[cfg(feature = "cuda")]
pub mod cuda_vulkan_interop;
#[cfg(feature = "cuda")]
pub mod interop;
#[cfg(feature = "cuda")]
pub mod vulkan_wgpu_interop;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AllocSize {
    pub height: usize,
    pub width: usize,
    pub stride: usize, //stride between rows (width * nr_channels * bytes per channel)
}
impl AllocSize {
    pub fn full_size(&self) -> usize {
        self.height * self.stride
    }
    pub fn stride_padded(&self) -> usize {
        let output_stride = self.stride;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padding = (align - output_stride % align) % align;
        output_stride + padding
    }
    pub fn full_size_padded(&self) -> usize {
        self.height * self.stride_padded()
    }
}

#[derive(Clone)]
pub struct VulkanGpu {
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
}
