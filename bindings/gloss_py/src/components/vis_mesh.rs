use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::{
    components::{MeshColorType, VisMesh},
    scene::Scene,
};
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use utils_rs::convert_enum_from;

#[pyclass(name = "MeshColorType", module = "gloss.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyMeshColorType {
    Solid = 0,
    PerVert,
    Texture,
    UV,
    Normal,
    NormalViewCoords,
}

// https://stackoverflow.com/questions/59984712/rust-macro-to-convert-between-identical-enums
convert_enum_from!(PyMeshColorType, MeshColorType, Solid, PerVert, Texture, UV, Normal, NormalViewCoords,);

#[pyclass(name = "VisMesh", module = "gloss.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyVisMesh {
    pub inner: VisMesh,
}
#[pymethods]
impl PyVisMesh {
    #[new]
    #[pyo3(signature = (show_mesh=None, solid_color=None, color_type=None, uv_scale=None))]
    #[pyo3(
        text_signature = "(show_mesh: Optional[bool] = None, solid_color: Optional[NDArray[np.float32]] = None, color_type: Optional[MeshColorType] = None, uv_scale: Optional[float] = None) -> VisMesh"
    )]
    pub fn new(
        show_mesh: Option<bool>,
        solid_color: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        color_type: Option<PyMeshColorType>,
        uv_scale: Option<f32>,
    ) -> Self {
        let def = VisMesh::default();

        #[allow(clippy::cast_possible_truncation)]
        let solid_color = if let Some(solid_color) = solid_color {
            assert_eq!(solid_color.len(), 4, "solid_color should have 4 components");
            na::Vector4::<f32>::from_vec(solid_color.to_vec().unwrap())
        } else {
            def.solid_color
        };

        let vis_mesh = VisMesh {
            show_mesh: show_mesh.unwrap_or(def.show_mesh),
            solid_color,
            color_type: MeshColorType::from(color_type.unwrap_or(PyMeshColorType::Solid)),
            uv_scale: uv_scale.unwrap_or(1.0),
            ..Default::default()
        };

        PyVisMesh { inner: vis_mesh }
    }
}
