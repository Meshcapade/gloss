#[cfg(feature = "selector")]
use crate::{
    components::{CamController, Name, VisOutline},
    plugin_manager::{plugins::InternalPlugin, GpuSystem, RunnerState},
    scene::Scene,
    viewer::GpuResources,
};

#[cfg(feature = "selector")]
use crossbeam_channel;
#[cfg(all(target_arch = "wasm32", feature = "selector"))]
use wasm_bindgen_futures;

pub struct Selector {
    pub current_selected: Option<String>,
}

pub struct Selectable;
/// Resource that holds the receiver for pixel data downloaded from ent id pass.
/// The data stored here is the entity id of the selected entity.
#[cfg(feature = "selector")]
pub struct SelectorPixelReceiver(pub crossbeam_channel::Receiver<u8>);

impl Default for Selector {
    fn default() -> Self {
        Self::new()
    }
}

impl Selector {
    pub fn new() -> Self {
        Self { current_selected: None }
    }
}

#[derive(Clone)]
pub struct SelectorPlugin {
    pub autorun: bool,
}

impl SelectorPlugin {
    pub fn new(autorun: bool) -> Self {
        Self { autorun }
    }
}

#[cfg(feature = "selector")]
impl InternalPlugin for SelectorPlugin {
    fn autorun(&self) -> bool {
        self.autorun
    }
    fn gpu_systems(&self) -> Vec<GpuSystem> {
        vec![
            GpuSystem::new(add_selector_resource).with_name("selector_add_resource_system"),
            GpuSystem::new(queue_pixel_download).with_name("selector_queue_download_system"),
            GpuSystem::new(redraw_if_queued).with_name("selector_redraw_if_queued_system"),
            GpuSystem::new(process_received_pixels).with_name("selector_process_pixels_system"),
        ]
    }
}

// we add this resource as long as the feature is enabled
// we can after choose to remove it if we want to disable click to select
#[cfg(feature = "selector")]
pub fn add_selector_resource(scene: &mut Scene, runner: &mut RunnerState, _gpu_res: &GpuResources) {
    if runner.is_first_time() {
        scene.add_resource(Selector::default());
    }
}

/// This system redraws if the `SelectorPixelReceiver` resource is present.
#[cfg(feature = "selector")]
pub fn redraw_if_queued(scene: &mut Scene, runner: &mut RunnerState, _gpu_res: &GpuResources) {
    if scene.has_resource::<SelectorPixelReceiver>() {
        runner.request_redraw();
    }
}

/// This system listens to `SelectorPixelReceiver` resource and handles the pixel data.
#[cfg(feature = "selector")]
pub fn process_received_pixels(scene: &mut Scene, _runner: &mut RunnerState, _gpu_res: &GpuResources) {
    // Try to receive pixel data
    let pixel_data = scene
        .get_resource::<&SelectorPixelReceiver>()
        .ok()
        .and_then(|receiver| receiver.0.try_recv().ok());

    // Early return if no data was received
    if let Some(pixel_data) = pixel_data {
        let entity_id = pixel_data;

        // Switch off selection for previous entity
        if let Ok(mut selector) = scene.get_resource::<&mut Selector>() {
            if let Some(current_selected) = &selector.current_selected {
                if let Some(prev_entity) = scene.get_entity_with_name(current_selected) {
                    if let Ok(mut vis_outline) = scene.world().get::<&mut VisOutline>(prev_entity) {
                        vis_outline.show_outline = false;
                    }
                }
            }
            selector.current_selected = None;
        }

        // Process the new selection
        if entity_id != 0 {
            let entity_ref = scene.find_entity_with_id(entity_id);

            if let Some(e_ref) = entity_ref {
                let name = e_ref.get::<&Name>().expect("The entity has no name").0.clone();

                if let Some(mut vis_outline) = e_ref.get::<&mut VisOutline>() {
                    vis_outline.show_outline = true;
                }

                scene.add_resource(Selector {
                    current_selected: Some(name.clone()),
                });
            }
        }

        let _ = scene.remove_resource::<SelectorPixelReceiver>();
    }
}

/// This system detects clicks and queues up the download of pixels.
#[cfg(feature = "selector")]
pub fn queue_pixel_download(scene: &mut Scene, runner: &mut RunnerState, gpu_res: &GpuResources) {
    // Early return if LMB is not clicked or if selector resource is not present
    if !scene.get_current_cam().unwrap().is_click(scene) || !scene.has_resource::<Selector>() {
        return;
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let cursor_pos = scene
        .get_current_cam()
        .and_then(|camera| scene.get_comp::<&CamController>(&camera.entity).ok())
        .and_then(|cam_control| cam_control.cursor_position)
        .map(|pos| (pos.x as u32, pos.y as u32));

    let Some((x, y)) = cursor_pos else { return };

    let gpu = &gpu_res.gpu;

    // Get the entity id texture from the renderer
    let entity_id_texture = gpu_res.renderer.entity_id_buffer();
    let scaled_x = x / entity_id_texture.tex_params.scale_factor;
    let scaled_y = y / entity_id_texture.tex_params.scale_factor;

    let (sender, receiver) = crossbeam_channel::unbounded();

    // We only need the pixel at selection so we dont really need to download the whole thing
    #[cfg(not(target_arch = "wasm32"))]
    {
        let single_pixel_img =
            pollster::block_on(entity_id_texture.download_pixel_to_cpu(gpu.device(), gpu.queue(), wgpu::TextureAspect::All, scaled_x, scaled_y));
        // We send only the first byte of the pixel data, which is the entity id
        let _ = sender.send(single_pixel_img.as_bytes().to_vec()[0]);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let texture_clone = entity_id_texture.clone();
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();

        wasm_bindgen_futures::spawn_local(async move {
            let pixel_data = texture_clone
                .download_pixel_to_cpu(&device, &queue, wgpu::TextureAspect::All, scaled_x, scaled_y)
                .await;
            // We send only the first byte of the pixel data, which is the entity id
            let _ = sender.send(pixel_data.as_bytes().to_vec()[0]);
        });
    }

    scene.add_resource(SelectorPixelReceiver(receiver));
    runner.request_redraw();
}
