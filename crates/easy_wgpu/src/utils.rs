//create an empty layout for when a pipeline doesn't use a specific group
pub fn create_empty_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[],
        label: Some("empty_layout"),
    })
}
pub fn create_empty_group(device: &wgpu::Device) -> wgpu::BindGroup {
    let layout = create_empty_layout(device);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_grou["),
        layout: &layout,
        entries: &[],
    })
}

//from
// https://github.com/jshrake/wgpu-mipmap/blob/main/src/util.rs#L278
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn get_mip_extent(extent: &wgpu::Extent3d, level: u32) -> wgpu::Extent3d {
    let mip_width = ((extent.width as f32) / (2u32.pow(level) as f32)).floor() as u32;
    let mip_height = ((extent.height as f32) / (2u32.pow(level) as f32)).floor() as u32;
    let mip_depth = ((extent.depth_or_array_layers as f32) / (2u32.pow(level) as f32)).floor() as u32;
    wgpu::Extent3d {
        width: mip_width.max(1),
        height: mip_height.max(1),
        depth_or_array_layers: mip_depth.max(1),
    }
}
