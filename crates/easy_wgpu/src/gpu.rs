/// Convenience function for passing around gpu-related data liek device and
/// queue
#[derive(Debug)]
pub struct Gpu {
    // The order of properties in a struct is the order in which items are dropped.
    // wgpu seems to require that the device be dropped last, otherwise there is a resouce
    // leak.
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    queue: wgpu::Queue,
    device: wgpu::Device,
    limits: wgpu::Limits,
}

impl Gpu {
    pub fn new(adapter: wgpu::Adapter, instance: wgpu::Instance, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let limits = adapter.limits();
        Self {
            adapter,
            instance,
            queue,
            device,
            limits,
        }
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }
}
