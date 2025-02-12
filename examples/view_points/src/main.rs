use gloss_renderer::{
    components::{DiffuseImg, ImgConfig, NormalImg, VisMesh, VisPoints},
    config::LogLevel,
    geom::Geom,
    gloss_setup_logger,
    viewer::Viewer,
};
use pollster::FutureExt;

fn main() {
    // TODO: We dont use the log level from the config?
    gloss_setup_logger(LogLevel::Info, None); // Call only once per process
    let mut viewer = Viewer::new(Some("./config/example_view_mesh.toml"));
    create_test_scene(&mut viewer).block_on();
    viewer.run();
}

async fn create_test_scene(viewer: &mut Viewer) {
    let path_mesh = "./data/bust.obj";
    let path_diffuse = "./data/bust_alb.jpg";
    let path_normal = "./data/bust_nrm.png";
    let name = "test_mesh";

    viewer
        .scene
        .get_or_create_entity(name)
        .insert_builder(Geom::build_from_file(path_mesh))
        .insert(DiffuseImg::new_from_path_async(path_diffuse, &ImgConfig::default()).await)
        .insert(NormalImg::new_from_path_async(path_normal, &ImgConfig::default()).await)
        .insert(VisMesh {
            show_mesh: false,
            ..Default::default()
        })
        .insert(VisPoints {
            show_points: true,
            ..Default::default()
        });
}
