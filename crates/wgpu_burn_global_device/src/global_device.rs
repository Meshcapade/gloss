use cubecl::wgpu::WgpuSetup;
use once_cell::sync::OnceCell;
use wgpu::Instance;

// Global static that will be initialized once
pub static GLOBAL_CUBECL_WGPU_DEVICE: OnceCell<cubecl::wgpu::WgpuDevice> = OnceCell::new();

/// Get access to the global ``WgpuDevice`` for Burn if it has been initialized
pub fn get_global_wgpu_device() -> Option<cubecl::wgpu::WgpuDevice> {
    GLOBAL_CUBECL_WGPU_DEVICE.get().cloned()
}

/// Check if the global ``WgpuDevice`` for Burn has been initialized
pub fn has_global_wgpu_device() -> bool {
    GLOBAL_CUBECL_WGPU_DEVICE.get().is_some()
}

pub fn init_global_device(instance: &Instance, adapter: &wgpu::Adapter, device: &wgpu::Device, queue: &wgpu::Queue) {
    let wgpu_setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device: device.clone(),
        queue: queue.clone(),
        backend: adapter.get_info().backend,
    };
    let runtime_options = cubecl::wgpu::RuntimeOptions::default();
    let cubecl_device = cubecl::wgpu::init_device(wgpu_setup, runtime_options);
    GLOBAL_CUBECL_WGPU_DEVICE.set(cubecl_device).unwrap();
}
