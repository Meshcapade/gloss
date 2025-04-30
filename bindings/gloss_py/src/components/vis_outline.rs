use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{components::VisOutline, scene::Scene};
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass(name = "VisOutline", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyVisOutline {
    pub inner: VisOutline,
}
#[pymethods]
impl PyVisOutline {
    #[new]
    #[pyo3(signature = (outline_color=None, outline_width=None))]
    #[pyo3(text_signature = "(outline_color: Optional[NDArray[np.float32]] = None, outline_width: Optional[float] = None) -> VisOutline")]
    pub fn new(outline_color: Option<PyArrayLike1<'_, f32, AllowTypeChange>>, outline_width: Option<f32>) -> Self {
        let def = VisOutline::default();

        #[allow(clippy::cast_possible_truncation)]
        let outline_color = if let Some(outline_color) = outline_color {
            assert_eq!(outline_color.len(), 4, "outline_color should have 4 components");
            na::Vector4::<f32>::from_vec(outline_color.to_vec().unwrap())
        } else {
            def.outline_color
        };

        let vis_outline = VisOutline {
            show_outline: def.show_outline,
            outline_color,
            outline_width: outline_width.unwrap_or(def.outline_width),
            ..Default::default()
        };

        PyVisOutline { inner: vis_outline }
    }
}
