use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{LineColorType, VisLines},
    scene::Scene,
};
use gloss_utils::convert_enum_from;
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass(name = "LineColorType", module = "gloss.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyLineColorType {
    Solid = 0,
    PerVert,
}

// https://stackoverflow.com/questions/59984712/rust-macro-to-convert-between-identical-enums
convert_enum_from!(PyLineColorType, LineColorType, Solid, PerVert,);

#[pyclass(name = "VisLines", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyVisLines {
    pub inner: VisLines,
}
#[pymethods]
impl PyVisLines {
    #[new]
    #[pyo3(signature = (show_lines=None, line_color=None, line_width=None, color_type=None, zbuffer=None))]
    #[pyo3(
        text_signature = "(show_lines: Optional[bool] = None, line_color: Optional[NDArray[np.float32]] = None, line_width: Optional[float] = None, color_type: Optional[LineColorType] = None, zbuffer: Optional[bool] = None) -> VisLines"
    )]
    pub fn new(
        show_lines: Option<bool>,
        line_color: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        line_width: Option<f32>,
        color_type: Option<PyLineColorType>,
        zbuffer: Option<bool>,
    ) -> Self {
        let def = VisLines::default();

        #[allow(clippy::cast_possible_truncation)]
        let line_color = if let Some(line_color) = line_color {
            assert_eq!(line_color.len(), 4, "line_color should have 4 components");
            na::Vector4::<f32>::from_vec(line_color.to_vec().unwrap())
        } else {
            def.line_color
        };

        let vis_lines = VisLines {
            show_lines: show_lines.unwrap_or(def.show_lines),
            line_color,
            line_width: line_width.unwrap_or(def.line_width),
            color_type: LineColorType::from(color_type.unwrap_or(PyLineColorType::Solid)),
            zbuffer: zbuffer.unwrap_or(def.zbuffer),
            ..Default::default()
        };

        PyVisLines { inner: vis_lines }
    }
}
