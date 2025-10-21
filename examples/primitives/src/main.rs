use gloss_renderer::{builders, components::VisMesh, config::LogLevel, gloss_setup_logger, viewer::Viewer};
use nalgebra as na;

fn main() {
    gloss_setup_logger(LogLevel::Warn, None); // Call only once per process

    let mut viewer = Viewer::new(Some("./config/example_primitives.toml"));

    // Cube
    viewer
        .scene_mut()
        .get_or_create_entity("cube")
        .insert_builder(builders::build_cube(na::Point3::<f32>::new(0.0, 1.0, 0.0), 1.0))
        .insert(VisMesh {
            solid_color: na::Vector4::<f32>::new(1.0, 1.0, 0.0, 1.0),
            ..Default::default()
        });

    // Plane
    viewer
        .scene_mut()
        .get_or_create_entity("plane")
        .insert_builder(builders::build_plane(
            na::Point3::<f32>::new(0.0, 0.0, 0.0),
            na::Vector3::<f32>::new(0.0, 1.0, 0.0),
            7.0,
            7.0,
            false,
        ))
        .insert(VisMesh {
            solid_color: na::Vector4::<f32>::new(1.0, 0.0, 0.0, 1.0),
            ..Default::default()
        });

    viewer.run();
}
