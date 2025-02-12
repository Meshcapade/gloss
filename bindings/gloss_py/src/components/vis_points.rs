use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{PointColorType, VisPoints},
    scene::Scene,
};
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use utils_rs::convert_enum_from;

#[pyclass(name = "PointColorType", module = "gloss.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyPointColorType {
    Solid = 0,
    PerVert,
}

// https://stackoverflow.com/questions/59984712/rust-macro-to-convert-between-identical-enums
convert_enum_from!(PyPointColorType, PointColorType, Solid, PerVert,);

#[pyclass(name = "VisPoints", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyVisPoints {
    pub inner: VisPoints,
}
#[pymethods]
impl PyVisPoints {
    #[new]
    #[pyo3(signature = (show_points=None, point_color=None, point_size=None, color_type=None, zbuffer=None, show_points_indices=None))]
    #[pyo3(
        text_signature = "(show_points: Optional[bool] = None, point_color: Optional[NDArray[np.float32]] = None, point_size: Optional[float] = None, color_type: Optional[PointColorType] = None, zbuffer: Optional[bool] = None, show_points_indices: Optional[bool] = None) -> VisPoints"
    )]
    pub fn new(
        show_points: Option<bool>,
        point_color: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        point_size: Option<f32>,
        color_type: Option<PyPointColorType>,
        zbuffer: Option<bool>,
        show_points_indices: Option<bool>,
    ) -> Self {
        let def = VisPoints::default();

        #[allow(clippy::cast_possible_truncation)]
        let point_color = if let Some(point_color) = point_color {
            assert_eq!(point_color.len(), 4, "point_color should have 4 components");
            na::Vector4::<f32>::from_vec(point_color.to_vec().unwrap())
        } else {
            def.point_color
        };

        let vis_points = VisPoints {
            show_points: show_points.unwrap_or(def.show_points),
            point_color,
            point_size: point_size.unwrap_or(def.point_size),
            color_type: PointColorType::from(color_type.unwrap_or(PyPointColorType::Solid)),
            zbuffer: zbuffer.unwrap_or(def.zbuffer),
            show_points_indices: show_points_indices.unwrap_or(def.show_points_indices),
            ..Default::default()
        };

        PyVisPoints { inner: vis_points }
    }
}
