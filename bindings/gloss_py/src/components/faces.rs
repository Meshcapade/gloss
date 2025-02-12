use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::Faces, scene::Scene};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use utils_rs::tensor::{DynamicMatrixOps, DynamicTensorInt2D};

#[pyclass(name = "Faces", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyFaces {
    pub inner: Faces,
}
#[pymethods]
impl PyFaces {
    #[new]
    #[pyo3(text_signature = "(array: NDArray[np.uint32]) -> Faces")]
    pub fn new(array: PyReadonlyArray2<u32>) -> Self {
        let shape = array.shape();
        assert_eq!(shape[1], 3, "Faces need to be a Nx3 matrix but it has shape {shape:?}");
        PyFaces {
            inner: Faces(DynamicTensorInt2D::from_dmatrix(&array.as_matrix().into_owned())),
        }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.uint32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray2<u32>> {
        self.inner.0.to_dmatrix().to_pyarray_bound(py).into()
    }
}
