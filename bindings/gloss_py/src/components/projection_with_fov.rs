use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::ProjectionWithFov, scene::Scene};
use pyo3::prelude::*;

#[pyclass(name = "ProjectionWithFov", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyProjectionWithFov {
    pub inner: ProjectionWithFov,
}
#[pymethods]
impl PyProjectionWithFov {
    #[new]
    #[pyo3(text_signature = "(aspect_ratio: float, fovy: float, near: float, far: float) -> ProjectionWithFov")]
    pub fn new(aspect_ratio: f32, fovy: f32, near: f32, far: f32) -> Self {
        PyProjectionWithFov {
            inner: ProjectionWithFov {
                aspect_ratio,
                fovy,
                near,
                far,
            },
        }
    }
}
