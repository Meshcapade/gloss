use crate::entity_builder::PyEntityBuilder;
use crate::PyEdges;
use crate::PyFaces;
use crate::PyVerts;
use crate::PyVisLines;
use crate::PyVisMesh;
use gloss_renderer::builders;
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

//Builders---------------------

#[pyclass(name = "builders", module = "gloss.builders", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyBuilders;

#[pymethods]
impl PyBuilders {
    #[staticmethod]
    #[pyo3(text_signature = "(center: NDArray[np.float32], scale: float) -> EntityBuilder")]
    pub fn build_cube(center: PyArrayLike1<'_, f32, AllowTypeChange>, scale: f32) -> PyEntityBuilder {
        assert_eq!(center.len(), 3, "center should have 3 components");
        let center_na: na::DMatrix<f32> = center.try_readonly().unwrap().as_matrix().into();
        let center_point = na::Point3::<f32>::new(center_na.row(0)[0], center_na.row(1)[0], center_na.row(2)[0]);
        PyEntityBuilder::new(builders::build_cube(center_point, scale))
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> EntityBuilder")]
    pub fn build_floor() -> PyEntityBuilder {
        PyEntityBuilder::new(builders::build_floor())
    }
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> EntityBuilder")]
    pub fn build_from_file(path: &str) -> PyEntityBuilder {
        PyEntityBuilder::new(builders::build_from_file(path))
    }
    #[staticmethod]
    #[pyo3(text_signature = "(aspect_ratio: float, yfovy: float) -> EntityBuilder")]
    pub fn build_camera_frustum(aspect_ratio: f32, yfovy: f32) -> PyEntityBuilder {
        PyEntityBuilder::new(builders::build_camera_frustum(aspect_ratio, yfovy))
    }
    #[staticmethod]
    #[pyo3(text_signature = "(aspect_ratio: float, yfovy: float) -> EntityBuilder")]
    pub fn build_camera_frustum_comps(aspect_ratio: f32, yfovy: f32) -> (PyVerts, PyEdges, PyFaces, PyVisLines, PyVisMesh) {
        let comps = builders::build_camera_frustum_comps(aspect_ratio, yfovy);
        (
            PyVerts { inner: comps.0 },
            PyEdges { inner: comps.1 },
            PyFaces { inner: comps.2 },
            PyVisLines { inner: comps.3 },
            PyVisMesh { inner: comps.4 },
        )
    }
}
