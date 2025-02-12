use crate::scene::PyScene;
use gloss_renderer::{
    camera::Camera,
    components::{PosLookat, Projection, ProjectionWithIntrinsics},
    scene::Scene,
};
use nalgebra as na;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass(name = "Camera", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyCamera {
    obj_ptr: *mut Camera,
    py_scene: PyScene,
}
impl std::ops::Deref for PyCamera {
    type Target = Camera;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.obj_ptr }
    }
}
impl std::ops::DerefMut for PyCamera {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.obj_ptr }
    }
}
impl PyCamera {
    pub fn new(obj_ptr: *mut Camera, py_scene: PyScene) -> Self {
        PyCamera { obj_ptr, py_scene }
    }
}
#[pymethods]
impl PyCamera {
    #[pyo3(text_signature = "($self) -> Tuple[float, float]")]
    pub fn get_near_far(&mut self) -> (f32, f32) {
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<Projection>(ent);
        let projection = self.py_scene.get_comp::<&mut Projection>(&ent).unwrap();
        projection.near_far()
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> None")]
    pub fn set_position(&mut self, array: PyArrayLike1<'_, f32, AllowTypeChange>) {
        assert_eq!(array.len(), 3, "position should have 3 components");
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<PosLookat>(ent);
        let mut pos_lookat = self.py_scene.get_comp::<&mut PosLookat>(&ent).unwrap();
        pos_lookat.position = na::Point3::<f32>::from_slice(array.as_slice().unwrap());
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> None")]
    pub fn set_lookat(&mut self, array: PyArrayLike1<'_, f32, AllowTypeChange>) {
        assert_eq!(array.len(), 3, "lookat should have 3 components");
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<PosLookat>(ent);
        let mut pos_lookat = self.py_scene.get_comp::<&mut PosLookat>(&ent).unwrap();
        pos_lookat.lookat = na::Point3::<f32>::from_slice(array.as_slice().unwrap());
    }
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> None")]
    pub fn set_up(&mut self, array: PyArrayLike1<'_, f32, AllowTypeChange>) {
        assert_eq!(array.len(), 3, "up should have 3 components");
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<PosLookat>(ent);
        let mut pos_lookat = self.py_scene.get_comp::<&mut PosLookat>(&ent).unwrap();
        pos_lookat.up = na::Vector3::<f32>::from_vec(array.to_vec().unwrap());
    }
    /// Returns the look-at position of the camera.
    ///
    /// # Errors
    /// Will return an error if the position look-at component cannot be
    /// retrieved.
    #[pyo3(text_signature = "($self) -> Tuple[float, float, float]")]
    pub fn get_lookat(&self) -> PyResult<(f32, f32, f32)> {
        let ent = self.entity;
        let pos_lookat = self.py_scene.get_comp::<&PosLookat>(&ent).unwrap();
        let lookat = pos_lookat.lookat;
        Ok((lookat.x, lookat.y, lookat.z))
    }
    /// Returns the position of the camera.
    ///
    /// # Errors
    /// Will return an error if the position component cannot be retrieved.
    #[pyo3(text_signature = "($self) -> Tuple[float, float, float]")]
    pub fn get_position(&self) -> PyResult<(f32, f32, f32)> {
        let ent = self.entity;
        let pos_lookat = self.py_scene.get_comp::<&PosLookat>(&ent).unwrap();
        let position = pos_lookat.position;
        Ok((position.x, position.y, position.z))
    }
    /// Returns the up vector of the camera.
    ///
    /// # Errors
    /// Will return an error if the up vector component cannot be retrieved.
    #[pyo3(text_signature = "($self) -> Tuple[float, float, float]")]
    pub fn get_up(&self) -> PyResult<(f32, f32, f32)> {
        let ent = self.entity;
        let pos_lookat = self.py_scene.get_comp::<&PosLookat>(&ent).unwrap();
        let up = pos_lookat.up;
        Ok((up.x, up.y, up.z))
    }
    //TODO this should be a function of the camera because right now it only lives
    // on the python side but there is no equivalent set_extrinsics in rust
    #[pyo3(text_signature = "($self, array: NDArray[np.float32]) -> None")]
    pub fn set_extrinsics(&mut self, array: PyArrayLike2<'_, f32, AllowTypeChange>) {
        assert_eq!(array.shape(), [4, 4], "extrinsics matrix has to be 4x4");

        let extr: na::DMatrix<f32> = array.readonly().as_matrix().into();

        let rot = extr.fixed_view::<3, 3>(0, 0);
        let trans = extr.fixed_view::<3, 1>(0, 3);
        let center = -rot.transpose() * trans;

        //set new values
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<PosLookat>(ent);
        let mut pos_lookat = self.py_scene.get_comp::<&mut PosLookat>(&ent).unwrap();
        let new_lookat = rot.row(2).transpose() + center;
        pos_lookat.position = center.into();
        pos_lookat.lookat = new_lookat.into();
        pos_lookat.up = -rot.row(1).transpose();

        // self.set_position(center.to_pyarray(py));
        // self.set_lookat(
        //     rot[(2, 0)] + center[(0, 0)],
        //     rot[(2, 1)] + center[(1, 0)],
        //     rot[(2, 2)] + center[(2, 0)],
        // );
        // self.set_up(-rot[(1, 0)], -rot[(1, 1)], -rot[(1, 2)]);
    }
    #[pyo3(text_signature = "($self, width: int, height: int) -> None")]
    pub fn set_width_height(&mut self, width: u32, height: u32) {
        let mut cam = Camera::from_entity(self.entity); //we couldn't get this exact camera as a reference because we already have a
                                                        // reference to scene but we can create a new camera from the same entity which
                                                        // should be quite cheap
        let scene_native: &mut Scene = &mut self.py_scene;
        cam.set_target_res(width, height, scene_native);
    }
    #[pyo3(signature = (fx, fy, cx, cy, near=None, far=None))]
    #[pyo3(text_signature = "($self, fx: float, fy: float, cx: float, cy: float, near: Optional[float] = None, far: Optional[float] = None) -> None")]
    pub fn set_intrinsics(&mut self, fx: f32, fy: f32, cx: f32, cy: f32, near: Option<f32>, far: Option<f32>) {
        let ent = self.entity;
        let _ = self.py_scene.world.insert_one(
            ent,
            Projection::WithIntrinsics(ProjectionWithIntrinsics {
                fx,
                fy,
                cx,
                cy,
                near: near.unwrap_or(0.01),
                far: far.unwrap_or(100.0),
            }),
        );
    }
    #[pyo3(text_signature = "($self, degrees: float) -> None")]
    pub fn orbit_y(&mut self, degrees: f32) {
        let ent = self.entity;
        self.py_scene.insert_if_doesnt_exist::<PosLookat>(ent);
        let mut pos_lookat = self.py_scene.get_comp::<&mut PosLookat>(&ent).unwrap();
        pos_lookat.orbit_y(degrees);
    }
}
