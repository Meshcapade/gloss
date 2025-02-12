use gloss_renderer::{config::LogLevel, gloss_setup_logger, viewer::Viewer};

fn main() {
    gloss_setup_logger(LogLevel::Info, None); // Call only once per process
    let mut viewer = Viewer::new(Some("./config/example_minimal.toml"));
    viewer.run();
}
