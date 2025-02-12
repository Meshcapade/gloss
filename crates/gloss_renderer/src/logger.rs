#[cfg(not(target_arch = "wasm32"))]
use log::error;

use log::LevelFilter;

use crate::config::{Config, LogLevel};
use std::collections::HashMap;

#[derive(Clone)]
pub struct LogLevelCaps {
    pub caps: HashMap<String, LogLevel>,
}
impl Default for LogLevelCaps {
    fn default() -> Self {
        let config = Config::default();
        Self {
            caps: config.core.log_level_caps,
        }
    }
}

pub fn gloss_setup_logger_from_config_file(config_path: Option<&str>) {
    let config = Config::new(config_path);
    gloss_setup_logger_from_config(&config);
}

pub fn gloss_setup_logger_from_config(config: &Config) {
    gloss_setup_logger(
        config.core.log_level,
        Some(LogLevelCaps {
            caps: config.core.log_level_caps.clone(),
        }),
    );
}

// If there is no level caps provided we use the default level caps
pub fn gloss_setup_logger(log_level: LogLevel, log_level_caps: Option<LogLevelCaps>) {
    let lvl_filter_default: LevelFilter = log_level.into();

    let log_level_caps = log_level_caps.unwrap_or_default();

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let mut builder = env_logger::filter::Builder::new();
            builder.filter(None, lvl_filter_default); //by default everything info and above will be printed
            // Apply filters
            for (module_name, log_lvl) in log_level_caps.caps.iter(){
                let lvl_filter = <crate::config::LogLevel as Into<LevelFilter>>::into(*log_lvl);
                let log_lvl_least_verbose = lvl_filter_default.min(lvl_filter);
                builder.filter_module(module_name, log_lvl_least_verbose);
            }
            wasm_log::init(wasm_log::Config::new(builder.build()));
        } else {
            // Setup logger with output for fileline and color
            // https://stackoverflow.com/a/65084368
            let mut builder =  env_logger::Builder::from_default_env();
            builder.format( utils_rs::logging::format );
            builder.filter(None, lvl_filter_default); //by default everything info and above will be printed
            // Apply filters
            for (module_name, log_lvl) in log_level_caps.caps.iter(){
                let lvl_filter = <crate::config::LogLevel as Into<LevelFilter>>::into(*log_lvl);
                let log_lvl_least_verbose = lvl_filter_default.min(lvl_filter);
                builder.filter_module(module_name, log_lvl_least_verbose);
            }
            if let Err(err) = builder.try_init(){
                error!("Error: {err}.
                    If you have auto_\"auto_create_logger=false\" in your config file, please make sure you are calling gloss_setup_logger() once per process. 
                    If you have auto_\"auto_create_logger=true\" in your config file, the gloss_setup_logger() is called automatically so please make sure you are creating the Viewer() only once per proces.");
            }

        }
    }
}
