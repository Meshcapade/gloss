use config::{Config as ConfigLib, File};
use log::LevelFilter;
use nalgebra as na;
use std::{collections::HashMap, path::Path};

use crate::components::ConfigChanges;
use gloss_utils::{config::MyTomlFile, convert_enum_into};

// Things marked as optional will be sometimes automatically filled depending on
// the scene scale or other factors known at runtime
#[derive(Clone, serde::Deserialize, Debug)]
pub struct Config {
    pub core: CoreConfig,
    pub render: RenderConfig,
    pub scene: SceneConfig,
    is_concrete: Option<bool>, // Some configs are set to "auto", when they are made concrete, this bool gets set to true
    is_consumed: Option<bool>, // Using the config to create a scene will set this to true so that we don't rerun it
}

#[derive(Clone, serde::Deserialize, Debug)]
#[allow(unused)]
#[allow(clippy::struct_excessive_bools)]
pub struct CoreConfig {
    pub enable_gui: bool,
    pub gui_start_hidden: bool,
    pub auto_add_floor: bool,
    pub floor_type: FloorType,
    pub floor_scale: Option<f32>,
    pub floor_origin: Option<na::Point3<f32>>,
    pub floor_texture: FloorTexture,
    pub floor_uv_scale: Option<f32>,
    pub floor_grid_line_width: f32,
    pub canvas_id: Option<String>,
    pub auto_create_logger: bool,
    pub log_level: LogLevel,
    pub log_level_caps: HashMap<String, LogLevel>,
    pub enable_memory_profiling_callstacks: bool,
}

#[derive(Clone, serde::Deserialize, Debug)]
#[allow(unused)]
pub struct RenderConfig {
    pub ambient_factor: f32,
    pub environment_factor: f32,
    pub bg_color: na::Vector4<f32>,
    pub enable_distance_fade: Option<bool>,
    pub distance_fade_center: Option<na::Point3<f32>>,
    pub distance_fade_start: Option<f32>,
    pub distance_fade_end: Option<f32>,
    // Color grading, applied before tonemapping
    pub apply_lighting: bool,
    pub saturation: f32,
    pub gamma: f32,
    pub exposure: f32,
    pub shadow_filter_method: ShadowFilteringMethod,
    pub msaa_nr_samples: u32,
    pub preallocated_staging_buffer_bytes: u32,
    pub offscreen_color_float_tex: bool,
}

#[derive(Clone, serde::Deserialize, Debug)]
#[allow(unused)]
pub struct SceneConfig {
    pub cam: CamConfig,
    pub lights: Vec<LightConfig>,
}

#[derive(Clone, serde::Deserialize, Debug)]
#[allow(unused)]
pub struct CamConfig {
    pub position: Option<na::Point3<f32>>,
    pub lookat: Option<na::Point3<f32>>,
    pub fovy: f32,
    pub near: Option<f32>,
    pub far: Option<f32>,
    pub limit_max_dist: Option<f32>,
    pub limit_max_vertical_angle: Option<f32>,
    pub limit_min_vertical_angle: Option<f32>,
}

#[derive(Clone, serde::Deserialize, Debug)]
#[allow(unused)]
pub struct LightConfig {
    pub position: Option<na::Point3<f32>>,
    pub lookat: Option<na::Point3<f32>>,
    pub fovy: f32, //radians
    pub near: Option<f32>,
    pub far: Option<f32>,
    pub color: na::Vector3<f32>,
    pub intensity: Option<f32>,
    pub range: Option<f32>,
    pub radius: Option<f32>,
    pub shadow_res: Option<u32>,
    pub shadow_bias_fixed: Option<f32>,
    pub shadow_bias: Option<f32>,
    pub shadow_bias_normal: Option<f32>,
}

#[derive(Debug, Copy, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ShadowFilteringMethod {
    /// Hardware 2x2.
    ///
    /// Fast but poor quality.
    Hardware2x2 = 0,
    /// Method by Ignacio Castaño for The Witness using 9 samples and smart
    /// filtering to achieve the same as a regular 5x5 filter kernel.
    ///
    /// Good quality, good performance.
    Castano13,
}

#[derive(Debug, Copy, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FloorType {
    Solid = 0,
    Grid,
}

#[derive(Debug, Copy, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FloorTexture {
    None = 0,
    Checkerboard,
}

#[derive(Debug, Copy, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// A level lower than all log levels.
    Off,
    /// Corresponds to the `Error` log level.
    Error,
    /// Corresponds to the `Warn` log level.
    Warn,
    /// Corresponds to the `Info` log level.
    Info,
    /// Corresponds to the `Debug` log level.
    Debug,
    /// Corresponds to the `Trace` log level.
    Trace,
}
// https://stackoverflow.com/questions/59984712/rust-macro-to-convert-between-identical-enums
convert_enum_into!(LogLevel, LevelFilter, Off, Error, Warn, Info, Debug, Trace,);

impl Default for Config {
    fn default() -> Config {
        Config::new(None)
    }
}

impl Config {
    /// # Panics
    /// Will panic if the path is not valid unicode
    pub fn new(config_path: Option<&str>) -> crate::config::Config {
        let default_file = File::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/config/default.toml")), MyTomlFile);

        // Provide the sources and build the config object:
        let mut builder = ConfigLib::builder().add_source(default_file.required(true));

        if let Some(config_path) = config_path {
            // Read the config file path either absolute or relative
            let config_path_abs = if Path::new(config_path).is_relative() {
                Path::new(env!("CARGO_MANIFEST_DIR")).join(config_path)
            } else {
                Path::new(config_path).to_path_buf()
            };
            let config_file = File::new(config_path_abs.to_str().unwrap(), MyTomlFile);
            builder = builder.add_source(config_file.required(true));
        }

        let settings = builder.build().unwrap();

        settings.try_deserialize().unwrap()
    }

    /// # Panics
    /// Will panic if the path is not valid unicode
    pub fn new_from_str(config_content: &str) -> crate::config::Config {
        let default_file = File::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/config/default.toml")), MyTomlFile);

        let config_file = File::from_str(config_content, MyTomlFile);

        // Provide the sources and build the config object:
        let builder = ConfigLib::builder()
            .add_source(default_file.required(true))
            .add_source(config_file.required(true));

        let settings = builder.build().unwrap();

        settings.try_deserialize().unwrap()
    }
    pub fn is_concrete(&self) -> bool {
        self.is_concrete.is_some()
    }
    pub fn set_concrete(&mut self) {
        self.is_concrete = Some(true);
    }
    pub fn is_consumed(&self) -> bool {
        self.is_consumed.is_some()
    }
    pub fn set_consumed(&mut self) {
        self.is_consumed = Some(true);
    }

    pub fn apply_deltas(&mut self, changes: &ConfigChanges) {
        if let Some(ref mut p) = self.render.distance_fade_center {
            p.clone_from(&changes.new_distance_fade_center);
        }
    }
}
