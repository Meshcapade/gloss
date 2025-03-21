use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::Colors, scene::Scene};
use gloss_utils::tensor::{DynamicMatrixOps, DynamicTensorFloat2D};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
// use nshare::ToNdarray2;

#[pyclass(name = "Colors", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyColors {
    pub inner: Colors,
}
#[pymethods]
impl PyColors {
    #[new]
    #[pyo3(text_signature = "(array: NDArray[np.float32]) -> Colors")]
    pub fn new(array: PyReadonlyArray2<f32>) -> Self {
        let shape = array.shape();
        assert_eq!(shape[1], 3, "Colors need to be a Nx3 matrix but it has shape {shape:?}");
        PyColors {
            inner: Colors(DynamicTensorFloat2D::from_dmatrix(&array.as_matrix().into_owned())),
        }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.0.to_dmatrix().to_pyarray_bound(py).into()
    }
}
