#[cfg(not(target_arch = "wasm32"))]
use log::error;

use crate::config::{Config, LogLevel};
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use env_logger::filter::Builder;
#[cfg(target_arch = "wasm32")]
use env_logger::filter::Filter;
use log::LevelFilter;
#[cfg(target_arch = "wasm32")]
use log::{Level, Log, Metadata, Record};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

/// Specify what to be logged
#[cfg(target_arch = "wasm32")]
pub struct WasmLogConfig {
    filter: Filter,
    message_location: MessageLocation,
}

/// Specify where the message will be logged.
#[cfg(target_arch = "wasm32")]
pub enum MessageLocation {
    /// The message will be on the same line as other info (level, path...)
    SameLine,
    /// The message will be on its own line, a new after other info.
    NewLine,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmLogConfig {
    fn default() -> Self {
        let mut builder = Builder::new();
        Self {
            // level: Level::Debug,
            filter: builder.build(),
            message_location: MessageLocation::SameLine,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl WasmLogConfig {
    /// Specify the maximum level you want to log
    pub fn new(filter: Filter) -> Self {
        Self {
            // level,
            filter,
            message_location: MessageLocation::SameLine,
        }
    }
    /// Put the message on a new line, separated from other information
    /// such as level, file path, line number.
    #[allow(clippy::return_self_not_must_use)]
    pub fn message_on_new_line(mut self) -> Self {
        self.message_location = MessageLocation::NewLine;
        self
    }
}

/// The log styles
#[cfg(target_arch = "wasm32")]
struct Style {
    lvl_trace: String,
    lvl_debug: String,
    lvl_info: String,
    lvl_warn: String,
    lvl_error: String,
    tgt: String,
    args: String,
}

#[cfg(target_arch = "wasm32")]
impl Style {
    fn new() -> Style {
        let base = String::from("color: white; padding: 0 3px; background:");
        Style {
            lvl_trace: format!("{base} gray;"),
            lvl_debug: format!("{base} blue;"),
            lvl_info: format!("{base} green;"),
            lvl_warn: format!("{base} orange;"),
            lvl_error: format!("{base} darkred;"),
            tgt: String::from("font-weight: bold; color: inherit"),
            args: String::from("background: inherit; color: inherit"),
        }
    }
}

/// The logger
#[cfg(target_arch = "wasm32")]
struct WasmLogger {
    config: WasmLogConfig,
    style: Style,
}

#[cfg(target_arch = "wasm32")]
impl Log for WasmLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.config.filter.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            let style = &self.style;
            let message_separator = match self.config.message_location {
                MessageLocation::NewLine => "\n",
                MessageLocation::SameLine => " ",
            };
            let s = format!(
                "%c{}%c {}:{}%c{}{}",
                record.level(),
                record.file().unwrap_or_else(|| record.target()),
                record.line().map_or_else(|| "[Unknown]".to_string(), |line| line.to_string()),
                message_separator,
                record.args(),
            );
            let s = JsValue::from_str(&s);
            let tgt_style = JsValue::from_str(&style.tgt);
            let args_style = JsValue::from_str(&style.args);

            match record.level() {
                Level::Trace => console::debug_4(&s, &JsValue::from(&style.lvl_trace), &tgt_style, &args_style),
                Level::Debug => console::log_4(&s, &JsValue::from(&style.lvl_debug), &tgt_style, &args_style),
                Level::Info => console::info_4(&s, &JsValue::from(&style.lvl_info), &tgt_style, &args_style),
                Level::Warn => console::warn_4(&s, &JsValue::from(&style.lvl_warn), &tgt_style, &args_style),
                Level::Error => console::error_4(&s, &JsValue::from(&style.lvl_error), &tgt_style, &args_style),
            }
        }
    }

    fn flush(&self) {}
}
#[cfg(target_arch = "wasm32")]

pub fn init(config: WasmLogConfig) {
    match try_init(config) {
        Ok(_) => {}
        Err(e) => console::error_1(&JsValue::from(e.to_string())),
    }
}

#[cfg(target_arch = "wasm32")]
pub fn try_init(config: WasmLogConfig) -> Result<(), log::SetLoggerError> {
    let max_level = config.filter.filter();
    let wl = WasmLogger { config, style: Style::new() };

    match log::set_boxed_logger(Box::new(wl)) {
        Ok(_) => {
            log::set_max_level(max_level);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

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
            init(WasmLogConfig::new(builder.build()));
        } else {
            // Setup logger with output for fileline and color
            // https://stackoverflow.com/a/65084368
            let mut builder =  env_logger::Builder::from_default_env();
            builder.format( gloss_utils::logging::format );
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
