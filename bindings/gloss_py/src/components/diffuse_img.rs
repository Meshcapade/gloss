use crate::components::img_config::PyImgConfig;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{DiffuseImg, ImgConfig},
    scene::Scene,
};
use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass(name = "DiffuseImg", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyDiffuseImg {
    pub inner: DiffuseImg,
}
#[pymethods]
impl PyDiffuseImg {
    #[new]
    #[pyo3(text_signature = "(path: str, img_config: Optional[ImgConfig] = None) -> DiffuseImg")]
    pub fn new(path: &str, img_config: Option<PyImgConfig>) -> Self {
        Self {
            inner: DiffuseImg::new_from_path(path, &img_config.map_or(ImgConfig::default(), |x| x.inner)),
        }
    }
    #[staticmethod]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_numpy_u8_hw3(array: PyReadonlyArray3<u8>) -> Self {
        let shape = array.shape();
        assert_eq!(
            shape[2], 3,
            "array of pixels needs to be a 3d matrix of shape HW3 but it has shape {shape:?}"
        );
        let h = shape[0] as u32;
        let w = shape[1] as u32;
        assert!(h > 0 && w > 0, "array of pixels needs to have non-zero height and width");
        Self {
            inner: DiffuseImg::new_from_raw_pixels(array.as_array().to_slice().unwrap().to_vec(), w, h, 3, &ImgConfig::default()),
        }
    }
}
