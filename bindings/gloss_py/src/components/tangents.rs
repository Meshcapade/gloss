use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::Tangents, scene::Scene};
use gloss_utils::tensor::{DynamicMatrixOps, DynamicTensorFloat2D};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass(name = "Tangents", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyTangents {
    pub inner: Tangents,
}
#[pymethods]
impl PyTangents {
    #[new]
    #[pyo3(text_signature = "(array: NDArray[np.float32]) -> Tangents")]
    pub fn new(array: PyReadonlyArray2<f32>) -> Self {
        Self {
            inner: Tangents(DynamicTensorFloat2D::from_dmatrix(&array.as_matrix().into())),
        }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.0.to_dmatrix().to_pyarray_bound(py).into()
    }
}
