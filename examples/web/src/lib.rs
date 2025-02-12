use gloss_renderer::{config::Config, viewer::Viewer};

#[cfg(target_arch = "wasm32")]
use gloss_renderer::geom::Geom;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
#[allow(dead_code)]
#[allow(unused_mut)]
#[allow(clippy::unused_async)]
async fn start() {
    gloss_setup_logger(LogLevel::Warn, None); //call only once per process
    let mut config = Config::default();
    config.core.canvas_id = Some(String::from("viewer-canvas"));
    let mut viewer = Viewer::new_with_config(&config);

    #[cfg(target_arch = "wasm32")]
    create_test_scene_wasm(&mut viewer).await;

    viewer.run();
}

#[allow(dead_code)]
#[cfg(target_arch = "wasm32")]
async fn create_test_scene_wasm(viewer: &mut Viewer) {
    let path_mesh = "./assets/bust.obj";

    viewer
        .scene
        .get_or_create_entity("test_mesh")
        .insert_builder(Geom::build_from_file_async(path_mesh).await);
}
