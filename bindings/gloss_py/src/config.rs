#![allow(clippy::new_without_default)]

use gloss_renderer::config::Config;

use pyo3::prelude::*;

#[pyclass(name = "Config", module = "gloss", unsendable)] // it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone)]
pub struct PyConfig(pub Config);
impl std::ops::Deref for PyConfig {
    type Target = Config;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (config_path=None))]
    #[pyo3(text_signature = "(config_path: Optional[str] = None) -> Config")]
    pub fn new(config_path: Option<&str>) -> Self {
        Self(Config::new(config_path))
    }
    #[staticmethod]
    #[pyo3(text_signature = "(config_content: str) -> Config")]
    pub fn new_from_str(config_content: &str) -> Self {
        Self(Config::new_from_str(config_content))
    }
}
