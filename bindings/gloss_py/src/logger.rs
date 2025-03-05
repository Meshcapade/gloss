use std::collections::HashMap;

use gloss_renderer::{config::LogLevel, logger::LogLevelCaps};
use gloss_utils::convert_enum_from;
use pyo3::prelude::*;

#[pyclass(name = "LogLevel", module = "gloss.log", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyLogLevel {
    Off = 0,
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
convert_enum_from!(PyLogLevel, LogLevel, Off, Error, Warn, Info, Debug, Trace,);

#[pyclass(module = "gloss.log", name = "LogLevelCaps", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone)]
pub struct PyLogLevelCaps {
    inner: LogLevelCaps,
}
#[pymethods]
impl PyLogLevelCaps {
    #[staticmethod]
    #[pyo3(text_signature = "() -> LogLevelCaps")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: LogLevelCaps::default(),
        }
    }
    #[new]
    #[pyo3(text_signature = "(caps: Dict[str, LogLevel]) -> LogLevelCaps")]
    pub fn new(caps: HashMap<String, PyLogLevel>) -> Self {
        let mut native_log_level_cap: HashMap<String, LogLevel> = HashMap::new();
        for (key, val) in caps.iter() {
            let lvl: LogLevel = (*val).into();
            native_log_level_cap.insert(key.clone(), lvl);
        }
        Self {
            inner: LogLevelCaps { caps: native_log_level_cap },
        }
    }
}

#[pyfunction]
#[pyo3(signature = (config_path=None))]
#[pyo3(text_signature = "(config_path: Optional[str] = None) -> None")]
pub fn gloss_setup_logger_from_config_file(config_path: Option<&str>) {
    gloss_renderer::gloss_setup_logger_from_config_file(config_path);
}

#[pyfunction]
#[pyo3(signature = (log_level, log_level_caps=None))]
#[pyo3(text_signature = "(log_level: LogLevel, log_level_caps: Optional[LogLevelCaps] = None) -> None")]
pub fn gloss_setup_logger(log_level: PyLogLevel, log_level_caps: Option<PyLogLevelCaps>) {
    gloss_renderer::gloss_setup_logger(log_level.into(), log_level_caps.map(|v| v.inner));
}
