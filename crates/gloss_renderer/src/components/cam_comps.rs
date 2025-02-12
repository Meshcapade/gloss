// use gloss_hecs::Bundle;

use std::collections::HashMap;
use winit::event::Touch;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

#[derive(Eq, PartialEq, Debug, Default)]
pub enum TargetResolutionUpdate {
    Fixed,
    #[default]
    WindowSize,
}

#[derive(Default)]
pub struct TargetResolution {
    pub width: u32,
    pub height: u32,
    pub update_mode: TargetResolutionUpdate,
}

#[derive(Eq, PartialEq, Debug)]
pub enum CamMode {
    Rotation,
    Translation,
}

/// Component usually used on camera or lights. Defines a position and a lookat.
/// This can be used to convert to a `view_matrix`
#[derive(Clone)]
pub struct PosLookat {
    pub position: na::Point3<f32>, //position in world coordinates
    pub lookat: na::Point3<f32>,
    pub up: na::Vector3<f32>,
}

// #[derive(Default)]
pub enum Projection {
    // #[default]
    WithFov(ProjectionWithFov),
    WithIntrinsics(ProjectionWithIntrinsics),
}

/// Component usually used on camera on lights. Defines a projection matrix
#[derive(Clone)]
pub struct ProjectionWithFov {
    pub aspect_ratio: f32,
    pub fovy: f32, //radians
    pub near: f32,
    pub far: f32,
}

#[derive(Clone)]
pub struct ProjectionWithIntrinsics {
    pub fx: f32,
    pub fy: f32, //radians
    pub cx: f32,
    pub cy: f32,
    // pub height: f32,
    // pub width: f32,
    pub near: f32,
    pub far: f32,
}

/// Component usually used on camera, allows to keep track of camera state while
/// handling mouse events
pub struct CamController {
    pub mouse_mode: CamMode,
    pub mouse_pressed: bool,
    pub prev_mouse_pos_valid: bool,
    pub prev_mouse: na::Vector2<f32>,
    pub limit_max_dist: Option<f32>,
    pub limit_max_vertical_angle: Option<f32>,
    pub limit_min_vertical_angle: Option<f32>,
    pub id2active_touches: HashMap<u64, Touch>, /* when we have multiple touch events each fingers gets an unique id. This maps from the unique id
                                                 * to an "active" or touching finger */
}

//implementations
//PosLookAt
impl Default for PosLookat {
    fn default() -> Self {
        Self {
            position: na::Point3::<f32>::new(0.0, 1.0, 3.0),
            lookat: na::Point3::<f32>::new(0.0, 0.0, 0.0),
            up: na::Vector3::<f32>::new(0.0, 1.0, 0.0),
        }
    }
}
impl PosLookat {
    pub fn new(position: na::Point3<f32>, lookat: na::Point3<f32>) -> Self {
        Self {
            position,
            lookat,
            ..Default::default()
        }
    }

    /// Initializes from a model matrix and a distance to lookat. Assumes the up
    /// vector is (0,1,0)
    pub fn new_from_model_matrix(model_matrix: na::SimilarityMatrix3<f32>, dist_lookat: f32) -> Self {
        let position = model_matrix.isometry.translation.vector;
        let mat = model_matrix.isometry.to_matrix();
        let rot: na::Matrix3<f32> = mat.fixed_view::<3, 3>(0, 0).into();
        let axis_lookat = rot.column(1); //Y axis
        let lookat = position + axis_lookat * dist_lookat;

        let position = na::Point3::<f32>::from(position);
        let lookat = na::Point3::<f32>::from(lookat);

        Self {
            position,
            lookat,
            up: na::Vector3::<f32>::new(0.0, 1.0, 0.0),
        }
    }

    /// Get view matrix as a mat4x4. View matrix maps from world to camera
    /// coordinates
    pub fn view_matrix(&self) -> na::Matrix4<f32> {
        self.view_matrix_isometry().to_matrix()
    }

    /// Get view matrix as a isometry matrix. View matrix maps from world to
    /// camera coordinates. Isometry matrix allows for faster inverse than a
    /// mat4x4.
    pub fn view_matrix_isometry(&self) -> na::IsometryMatrix3<f32> {
        na::IsometryMatrix3::<f32>::look_at_rh(&self.position, &self.lookat, &self.up)
    }

    /// Direction in which we are looking at in world coordinates
    pub fn direction(&self) -> na::Vector3<f32> {
        (self.lookat - self.position).normalize()
    }

    /// Get ``model_matrix`` as a isometry matrix. Model matrix maps from camera
    /// coordinates to world coordinates. Isometry matrix allows for faster
    /// inverse than a mat4x4
    pub fn model_matrix_isometry(&self) -> na::IsometryMatrix3<f32> {
        self.view_matrix_isometry().inverse()
    }

    /// Get ``model_matrix`` matrix as a isometry matrix. Model matrix maps from
    /// camera coordinates to world coordinates.
    pub fn model_matrix(&self) -> na::Matrix4<f32> {
        self.model_matrix_isometry().to_matrix()
    }

    /// Cam axes as columns of a 3x3 matrix. The columns represent a right
    /// handed coordinate system where x is towards right, y is up and z is
    /// outwards from the screen.
    pub fn cam_axes(&self) -> na::Matrix3<f32> {
        let model_matrix = self.model_matrix();
        let rot = model_matrix.fixed_view::<3, 3>(0, 0);
        rot.into()
    }

    /// Moves the camera along the direction of lookat
    pub fn dolly(&mut self, s: f32) {
        let eye_look_vec = self.lookat - self.position; //just a vector from eye to lookat
        let movement = eye_look_vec * s;
        self.position += movement;
    }

    /// Rotates the camera around the lookat point.
    pub fn orbit(&mut self, rot: na::Rotation3<f32>) {
        //we apply rotations around the lookat point so we have to substract, apply
        // rotation and then add back the lookat point
        let model_matrix = self.model_matrix_isometry();

        let trans_to_look_at = na::Translation3::from(-self.lookat);
        let trans_back = na::Translation3::from(self.lookat);

        let model_matrix_rotated = trans_back * rot * trans_to_look_at * model_matrix;

        //set the new position
        self.position = model_matrix_rotated.translation.vector.into();

        //fix the issue of the camera rotation above the object so that it becomes
        // upside down. When the camera is upside down then it's up vector shouldn't be
        // 0,1,0 anymore but rather 0,-1,0. This is because look_at_rh does a cross
        // product to get the Right vector and using 0,1,0 as up vector would mean that
        // we flip the right vector. Here we check if we are upside down and flip the up
        // vector accordingly.
        let model_matrix_rotated_mat = model_matrix_rotated.to_matrix();
        let cam_axes_after = model_matrix_rotated_mat.fixed_view::<3, 3>(0, 0);
        let up_cam_axis = cam_axes_after.column(1);
        let dot_up = self.up.dot(&up_cam_axis);
        if dot_up < 0.0 {
            self.up = -self.up;
        }
    }

    //convenience function to rotate just around the y axis by x degrees
    pub fn orbit_y(&mut self, degrees: f32) {
        let axis = na::Vector3::y_axis();
        let rot = na::Rotation3::from_axis_angle(&axis, degrees.to_radians());

        self.orbit(rot);
    }

    /// Moves the camera position to be at a new position and also rigidly
    /// shifts the lookat point, without rotating camera
    pub fn shift_cam(&mut self, pos: na::Point3<f32>) {
        let displacement = pos - self.position;
        self.position += displacement;
        self.lookat += displacement;
    }

    /// Moves the lookat at a new position and also rigidly shifts the cam
    /// point, without rotating camera
    pub fn shift_lookat(&mut self, pos: na::Point3<f32>) {
        let displacement = pos - self.lookat;
        self.position += displacement;
        self.lookat += displacement;
    }

    /// Returns the distance from camera to the lookat point
    pub fn dist_lookat(&self) -> f32 {
        (self.position - self.lookat).norm()
    }
}

impl Default for Projection {
    fn default() -> Self {
        Self::WithFov(ProjectionWithFov::default())
    }
}
impl Projection {
    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix(&self, width: u32, height: u32) -> na::Matrix4<f32> {
        match self {
            Projection::WithFov(proj) => proj.proj_matrix(),
            Projection::WithIntrinsics(proj) => proj.proj_matrix(width, height),
        }
    }
    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix_reverse_z(&self, width: u32, height: u32) -> na::Matrix4<f32> {
        match self {
            Projection::WithFov(proj) => proj.proj_matrix_reverse_z(),
            Projection::WithIntrinsics(proj) => proj.proj_matrix_reverse_z(width, height),
        }
    }
    pub fn near_far(&self) -> (f32, f32) {
        match self {
            Projection::WithFov(proj) => (proj.near, proj.far),
            Projection::WithIntrinsics(proj) => (proj.near, proj.far),
        }
    }
    pub fn set_near(&mut self, val: f32) {
        match self {
            Projection::WithFov(ref mut proj) => proj.near = val,
            Projection::WithIntrinsics(ref mut proj) => proj.near = val,
        }
    }
    pub fn set_far(&mut self, val: f32) {
        match self {
            Projection::WithFov(ref mut proj) => proj.far = val,
            Projection::WithIntrinsics(ref mut proj) => proj.far = val,
        }
    }
}

//ProjectionWithFov
impl Default for ProjectionWithFov {
    fn default() -> Self {
        Self {
            aspect_ratio: 1.6,
            fovy: 0.7, //radians
            near: 0.01,
            far: 100.0,
        }
    }
}
impl ProjectionWithFov {
    // right hand perspective-view frustum with a depth range of 0 to 1
    // https://github.com/toji/gl-matrix/issues/369
    pub fn proj_matrix(&self) -> na::Matrix4<f32> {
        glm::perspective_rh_zo(self.aspect_ratio, self.fovy, self.near, self.far)
    }
    /// Creates an infinite reverse right-handed perspective projection matrix
    /// with `[0,1]` depth range
    /// <https://docs.rs/glam/latest/src/glam/f32/sse2/mat4.rs.html#969-982>
    /// <https://github.com/bevyengine/bevy/blob/main/crates/bevy_render/src/camera/projection.rs#L172>
    pub fn proj_matrix_reverse_z(&self) -> na::Matrix4<f32> {
        //infinite Zfar
        // let mut mat = glm::TMat4::zeros();
        // let f = 1.0 / (0.5 * self.fovy).tan();
        // mat[(0, 0)] = f / (self.aspect_ratio);
        // mat[(1, 1)] = f;
        // mat[(2, 3)] = self.near;
        // mat[(3, 2)] = -1.0;
        // mat

        let mat = self.proj_matrix();

        // let mut depth_remap = glm::TMat4::identity();
        // depth_remap[(2, 2)] = -1.0;
        // depth_remap[(2, 3)] = 1.0;

        // depth_remap * mat
        // let mut mat = self.proj_matrix(width, height);

        let mut depth_remap = glm::TMat4::identity();
        depth_remap[(2, 2)] = -1.0;
        depth_remap[(2, 3)] = 1.0;

        depth_remap * mat
        // mat[(2, 2)] *= -1.0;
        // mat[(2, 3)] *= -1.0;
        // mat
    }
}

impl ProjectionWithIntrinsics {
    #[allow(clippy::cast_precision_loss)]
    pub fn proj_matrix(&self, width: u32, height: u32) -> na::Matrix4<f32> {
        let mut projection_matrix = na::Matrix4::<f32>::zeros();

        // Calculate the projection matrix with the given fx, fy, and normalised cx, cy
        projection_matrix[(0, 0)] = 2.0 * self.fx / width as f32;
        projection_matrix[(1, 1)] = 2.0 * self.fy / height as f32;
        projection_matrix[(0, 2)] = 1.0 - (2.0 * self.cx / width as f32);
        projection_matrix[(1, 2)] = (2.0 * self.cy / height as f32) - 1.0;
        // projection_matrix[(2, 2)] = -(self.far + self.near) / (self.far - self.near);
        // projection_matrix[(2, 3)] = -2.0 * self.far * self.near / (self.far -
        // self.near); projection_matrix[(3, 2)] = -1.0;
        projection_matrix[(2, 2)] = -self.far / (self.far - self.near);
        projection_matrix[(2, 3)] = -self.far * self.near / (self.far - self.near);
        projection_matrix[(3, 2)] = -1.0;
        projection_matrix
    }

    pub fn proj_matrix_reverse_z(&self, width: u32, height: u32) -> na::Matrix4<f32> {
        let mat = self.proj_matrix(width, height);

        let mut depth_remap = glm::TMat4::identity();
        depth_remap[(2, 2)] = -1.0;
        depth_remap[(2, 3)] = 1.0;

        depth_remap * mat
        // mat[(2, 2)] *= -1.0;
        // mat[(2, 3)] *= -1.0;
        // mat
    }
}
//CamController
impl Default for CamController {
    fn default() -> Self {
        Self {
            mouse_mode: CamMode::Rotation,
            mouse_pressed: false,
            prev_mouse_pos_valid: false,
            prev_mouse: na::Vector2::<f32>::zeros(),
            limit_max_dist: None,
            limit_max_vertical_angle: None,
            limit_min_vertical_angle: None,
            id2active_touches: HashMap::new(),
        }
    }
}
impl CamController {
    pub fn new(limit_max_dist: Option<f32>, limit_max_vertical_angle: Option<f32>, limit_min_vertical_angle: Option<f32>) -> Self {
        Self {
            limit_max_dist,
            limit_max_vertical_angle,
            limit_min_vertical_angle,
            ..Default::default()
        }
    }
}
