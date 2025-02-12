use gloss_renderer::{components::VisMesh, config::LogLevel, geom::Geom, gloss_setup_logger, viewer::Viewer};
use nalgebra as na;

fn main() {
    gloss_setup_logger(LogLevel::Warn, None); // Call only once per process

    let mut viewer = Viewer::new(Some("./config/example_primitives.toml"));

    // Cube
    viewer
        .scene
        .get_or_create_entity("cube")
        .insert_builder(Geom::build_cube(na::Point3::<f32>::new(0.0, 1.0, 0.0)))
        .insert(VisMesh {
            solid_color: na::Vector4::<f32>::new(1.0, 1.0, 0.0, 1.0),
            ..Default::default()
        });

    // Plane
    viewer
        .scene
        .get_or_create_entity("plane")
        .insert_builder(Geom::build_plane(
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
