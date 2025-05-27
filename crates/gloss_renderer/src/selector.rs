use crate::{
    components::{CamController, Name, VisOutline},
    plugin_manager::{plugins::InternalPlugin, GpuSystem, RunnerState},
    scene::Scene,
    viewer::GpuResources,
};

pub struct Selector {
    pub current_selected: String,
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

impl InternalPlugin for SelectorPlugin {
    fn autorun(&self) -> bool {
        self.autorun
    }

    fn gpu_systems(&self) -> Vec<GpuSystem> {
        vec![GpuSystem::new(handle_selection_click).with_name("selector_gpu_system")]
    }
}

fn handle_selection_click(scene: &mut Scene, runner: &mut RunnerState, gpu_res: &GpuResources) {
    // Early return if LMB is not clicked
    if !scene.get_current_cam().unwrap().is_click(scene) {
        return;
    }

    // Get the cursor position if it exists, otherwise return
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
    // Scale as per the scale factor
    let scaled_x = x / entity_id_texture.tex_params.scale_factor;
    let scaled_y = y / entity_id_texture.tex_params.scale_factor;

    // We only need the pixel at selection so we dont really need to download the whole thing
    let single_pixel_img =
        pollster::block_on(entity_id_texture.download_pixel_to_cpu(gpu.device(), gpu.queue(), wgpu::TextureAspect::All, scaled_x, scaled_y));
    let entity_id = single_pixel_img.as_bytes()[0];

    // Switch off selection for previous entity using the name in the selector
    // Always do this, every click regardless of where should switch off the previous selection
    if let Ok(selector) = scene.get_resource::<&mut Selector>() {
        if let Some(prev_entity) = scene.get_entity_with_name(&selector.current_selected) {
            if let Ok(mut vis_outline) = scene.world.get::<&mut VisOutline>(prev_entity) {
                vis_outline.show_outline = false;
            }
        }
    }
    let _ = scene.remove_resource::<Selector>();

    // For pixels with no entity, we get 0, dont do anything in that case.
    // If entity_id is not 0, we can look up the entity in the scene to select
    if entity_id != 0 {
        // Look for an entity with given ID (internally iterates over all ents)
        let entity_ref = scene.find_entity_with_id(entity_id);

        // Modify selector and VisOutline state if entity is found
        if let Some(e_ref) = entity_ref {
            let name = e_ref.get::<&Name>().expect("The entity has no name").0.clone();
            // Only ents with VisOutline are candidates for visual selection
            if let Some(mut vis_outline) = e_ref.get::<&mut VisOutline>() {
                vis_outline.show_outline = true;
            }
            // Add the selector resource to the scene
            scene.add_resource(Selector {
                current_selected: name.clone(),
            });
        }
    }

    runner.request_redraw(); //need to redraw again so the next frame we show the outline
}
