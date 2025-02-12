use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::ModelMatrix, scene::Scene};
use nalgebra as na;
use numpy::{AllowTypeChange, PyArray2, PyArrayLike1, PyArrayLike2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;

#[pyclass(name = "ModelMatrix", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyModelMatrix {
    pub inner: ModelMatrix,
}
#[pymethods]
impl PyModelMatrix {
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    #[pyo3(text_signature = "() -> ModelMatrix")]
    pub fn default() -> Self {
        Self {
            inner: ModelMatrix::default(),
        }
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> ModelMatrix")]
    pub fn with_translation(&mut self, array: PyArrayLike1<'_, f32, AllowTypeChange>) -> Self {
        assert_eq!(array.len(), 3, "Translation should have only 3 components");
        let arr = array.as_array();
        let mm = self.inner.clone();
        let mm = mm.with_translation(&na::Vector3::<f32>::new(arr[0], arr[1], arr[2]));
        Self { inner: mm }
    }
    #[pyo3(text_signature = "($self, rot3: NDArray[np.float32]) -> ModelMatrix")]
    pub fn with_rotation_rot3(&self, rot3: PyArrayLike2<'_, f32, AllowTypeChange>) -> Self {
        assert_eq!(rot3.shape(), [3, 3], "Rotation should be 3x3");
        let rot = rot3.as_matrix().clone_owned();
        let mm = self.inner.clone();
        let mm = mm.with_rotation_rot3(&na::Rotation3::<f32>::from_matrix(&rot.fixed_view::<3, 3>(0, 0).clone_owned()));
        Self { inner: mm }
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> ModelMatrix")]
    pub fn with_rotation_axis_angle(&self, array: PyArrayLike1<'_, f32, AllowTypeChange>) -> Self {
        assert_eq!(array.len(), 3, "Axis angle should have only 3 components");
        let arr = array.as_array();
        let mm = self.inner.clone();
        let mm = mm.with_rotation_axis_angle(&na::Vector3::<f32>::new(arr[0], arr[1], arr[2]));
        Self { inner: mm }
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> ModelMatrix")]
    pub fn with_rotation_euler(&self, array: PyArrayLike1<'_, f32, AllowTypeChange>) -> Self {
        assert_eq!(array.len(), 3, "Axis angle should have only 3 components");
        let arr = array.as_array();
        let mm = self.inner.clone();
        let mm = mm.with_rotation_euler(&na::Vector3::<f32>::new(arr[0], arr[1], arr[2]));
        Self { inner: mm }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.0.to_homogeneous().to_pyarray_bound(py).into()
    }
}
