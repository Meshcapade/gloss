use crate::{components::model_matrix::PyModelMatrix, entity_builder::PyEntityBuilder, img::PyDynImage};
use gloss_py_macros::DirectDeref;
use gloss_renderer::geom::{Geom, IndirRemovalPolicy, SplatType};
use gloss_utils::convert_enum_from;
use nalgebra as na;
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

#[pyclass(name = "SplatType", module = "gloss.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PySplatType {
    Avg = 0,
    Sum,
}
convert_enum_from!(PySplatType, SplatType, Avg, Sum,);

#[pyclass(name = "IndirRemovalPolicy", module = "gloss.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyIndirRemovalPolicy {
    RemoveInvalidRows = 0,
    RemoveInvalidCols,
}
convert_enum_from!(PyIndirRemovalPolicy, IndirRemovalPolicy, RemoveInvalidRows, RemoveInvalidCols,);

//Geom---------------------

#[pyclass(name = "geom", module = "gloss.geom", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(DirectDeref)]
pub struct PyGeom {
    inner: Geom,
}
#[pymethods]
impl PyGeom {
    #[staticmethod]
    #[pyo3(text_signature = "(center: NDArray[np.float32]) -> EntityBuilder")]
    pub fn build_cube(center: PyArrayLike1<'_, f32, AllowTypeChange>) -> PyEntityBuilder {
        assert_eq!(center.len(), 3, "center should have 3 components");
        let center_na: na::DMatrix<f32> = center.try_readonly().unwrap().as_matrix().into();
        let center_point = na::Point3::<f32>::new(center_na.row(0)[0], center_na.row(1)[0], center_na.row(2)[0]);
        PyEntityBuilder::new(Geom::build_cube(center_point))
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> EntityBuilder")]
    pub fn build_floor() -> PyEntityBuilder {
        PyEntityBuilder::new(Geom::build_floor())
    }
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> EntityBuilder")]
    pub fn build_from_file(path: &str) -> PyEntityBuilder {
        PyEntityBuilder::new(Geom::build_from_file(path))
    }

    #[staticmethod]
    #[pyo3(signature = (verts, mm=None))]
    #[pyo3(text_signature = "(verts: NDArray[np.float32], mm: Optional[ModelMatrix] = None) -> Tuple[NDArray[np.float32], NDArray[np.float32]]")]
    pub fn get_bounding_points(py: Python<'_>, verts: PyReadonlyArray2<f32>, mm: Option<PyModelMatrix>) -> (Py<PyArray1<f32>>, Py<PyArray1<f32>>) {
        let (min, max) = Geom::get_bounding_points(&verts.as_matrix().clone_owned(), mm.map(|v| v.inner.0));
        (
            min.coords.as_slice().to_pyarray_bound(py).into(),
            max.coords.as_slice().to_pyarray_bound(py).into(),
        )
    }

    #[staticmethod]
    #[pyo3(text_signature = "(verts: NDArray[np.float32], mat: ModelMatrix) -> NDArray[np.float32]")]
    pub fn transform_verts(py: Python<'_>, verts: PyReadonlyArray2<f32>, mat: PyModelMatrix) -> Py<PyArray2<f32>> {
        Geom::transform_verts(&verts.as_matrix().clone_owned(), &mat.inner.0)
            .to_pyarray_bound(py)
            .into()
    }

    #[staticmethod]
    #[pyo3(text_signature = "(img: DynImage, uvs: NDArray[np.float32], is_srgb: bool) -> NDArray[np.float32]")]
    pub fn sample_img_with_uvs(py: Python<'_>, img: &PyDynImage, uvs: PyReadonlyArray2<f32>, is_srgb: bool) -> Py<PyArray2<f32>> {
        let samples = Geom::sample_img_with_uvs(&img.inner, &uvs.as_matrix().clone_owned(), is_srgb);
        samples.to_pyarray_bound(py).into()
    }

    #[staticmethod]
    #[pyo3(
        text_signature = "(mat: NDArray[np.float32], mask: NDArray[np.bool_], keep: bool) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]"
    )]
    #[allow(clippy::type_complexity)]
    pub fn filter_rows(
        py: Python<'_>,
        mat: PyReadonlyArray2<f32>,
        mask: PyReadonlyArray1<bool>,
        keep: bool,
    ) -> (Py<PyArray2<f32>>, Py<PyArray1<i32>>, Py<PyArray1<i32>>) {
        let (filtered, orig2filtered, filtered2orig) = Geom::filter_rows(&mat.as_matrix().clone_owned(), &mask.to_vec().unwrap(), keep);
        (
            filtered.to_pyarray_bound(py).into(),
            orig2filtered.to_pyarray_bound(py).into(),
            filtered2orig.to_pyarray_bound(py).into(),
        )
    }

    #[staticmethod]
    #[pyo3(text_signature = "(mat: NDArray[np.float32], indices_orig2splatted: NDArray[np.uint32], splat_type: SplatType) -> NDArray[np.float32]")]
    pub fn splat_rows(
        py: Python<'_>,
        mat: PyReadonlyArray2<f32>,
        indices_orig2splatted: PyReadonlyArray1<u32>,
        splat_type: PySplatType,
    ) -> Py<PyArray2<f32>> {
        let splatted = Geom::splat_rows(
            &mat.as_matrix().clone_owned(),
            &indices_orig2splatted.to_vec().unwrap(),
            &splat_type.into(),
        );
        splatted.to_pyarray_bound(py).into()
    }

    #[staticmethod]
    #[pyo3(
        text_signature = "(mat: NDArray[np.uint32], indices_orig2destin: NDArray[np.int32], removal_policy: IndirRemovalPolicy) -> Tuple[NDArray[np.uint32], NDArray[np.bool_]]"
    )]
    pub fn apply_indirection(
        py: Python<'_>,
        mat: PyReadonlyArray2<u32>,
        indices_orig2destin: PyReadonlyArray1<i32>,
        removal_policy: PyIndirRemovalPolicy,
    ) -> (Py<PyArray2<u32>>, Py<PyArray1<bool>>) {
        let (reindexed, mask) = Geom::apply_indirection(
            &mat.as_matrix().clone_owned(),
            &indices_orig2destin.to_vec().unwrap(),
            &removal_policy.into(),
        );
        (reindexed.to_pyarray_bound(py).into(), mask.to_pyarray_bound(py).into())
    }

    //methods that are not static and act directly on the entity
}
