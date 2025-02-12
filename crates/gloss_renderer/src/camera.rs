extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use log::warn;
use winit::{
    dpi::PhysicalPosition,
    event::{MouseButton, MouseScrollDelta, Touch},
};

use crate::{
    components::{CamController, CamMode, PosLookat, Projection, ProjectionWithIntrinsics, TargetResolution, TargetResolutionUpdate},
    scene::Scene,
};
use gloss_hecs::Entity;

/// Camera implements most of the functionality related to cameras. It deals
/// with processing of mouse events and with moving the camera in an orbit like
/// manner. It contains a reference to the entity in the world so that changes
/// done by the camera object will directly affect the entity.
#[repr(C)]
pub struct Camera {
    pub entity: Entity,
}

impl Camera {
    #[allow(clippy::missing_panics_doc)] //really will never panic because the entity definitelly already exists in the
                                         // world
    pub fn new(name: &str, scene: &mut Scene, initialize: bool) -> Self {
        let entity = scene
            .get_or_create_hidden_entity(name)
            .insert(CamController::default())
            .insert(TargetResolution::default())
            .entity();
        if initialize {
            scene.world.insert_one(entity, PosLookat::default()).ok();
            scene.world.insert_one(entity, Projection::default()).ok();
        }
        Self { entity }
    }

    pub fn from_entity(entity: Entity) -> Self {
        Self { entity }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn is_initialized(&self, scene: &Scene) -> bool {
        scene.world.has::<PosLookat>(self.entity).unwrap()
            && (scene.world.has::<Projection>(self.entity).unwrap() || scene.world.has::<ProjectionWithIntrinsics>(self.entity).unwrap())
    }

    /// # Panics
    /// Will panic if the ``PosLookat`` component does not exist for this entity
    pub fn view_matrix(&self, scene: &Scene) -> na::Matrix4<f32> {
        let pos_lookat = scene.get_comp::<&PosLookat>(&self.entity).unwrap();
        pos_lookat.view_matrix()
    }

    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix(&self, scene: &Scene) -> na::Matrix4<f32> {
        let proj = scene.get_comp::<&Projection>(&self.entity).unwrap();
        let (width, height) = self.get_target_res(scene);
        proj.proj_matrix(width, height)
    }

    /// # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn proj_matrix_reverse_z(&self, scene: &Scene) -> na::Matrix4<f32> {
        let proj = scene.get_comp::<&Projection>(&self.entity).unwrap();
        let (width, height) = self.get_target_res(scene);
        proj.proj_matrix_reverse_z(width, height)
    }

    /// Performs a two-axis rotation of the camera by mapping the xy
    /// coordianates of the mouse to rotation around the Y axis of the world and
    /// the X axis of the camera.
    pub fn two_axis_rotation(
        cam_axis_x: na::Vector3<f32>,
        viewport_size: na::Vector2<f32>,
        speed: f32,
        prev_mouse: na::Vector2<f32>,
        current_mouse: na::Vector2<f32>,
    ) -> (na::Rotation3<f32>, na::Rotation3<f32>) {
        // rotate around Y axis of the world (the vector 0,1,0)
        let angle_y = std::f32::consts::PI * (prev_mouse.x - current_mouse.x) / viewport_size.x * speed;
        let rot_y = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), angle_y);

        //rotate around x axis of the camera coordinate
        let axis_x = cam_axis_x;
        let axis_x = na::Unit::new_normalize(axis_x);
        let angle_x = std::f32::consts::PI * (prev_mouse.y - current_mouse.y) / viewport_size.y * speed;
        let rot_x = na::Rotation3::from_axis_angle(&axis_x, angle_x);

        (rot_y, rot_x)
    }

    /// Projects from 3D world to 2D screen coordinates in the range [0,
    /// `viewport_width`] and [0, `viewport_height`]
    pub fn project(
        &self,
        point_world: na::Point3<f32>,
        view: na::Matrix4<f32>,
        proj: na::Matrix4<f32>,
        viewport_size: na::Vector2<f32>,
    ) -> na::Vector3<f32> {
        //get the point from world to screen space
        let p_view = view * point_world.to_homogeneous();
        let mut p_proj = proj * p_view;
        p_proj = p_proj / p_proj.w;
        p_proj = p_proj * 0.5 + na::Vector4::<f32>::new(0.5, 0.5, 0.5, 0.5);
        p_proj.x *= viewport_size.x;
        p_proj.y *= viewport_size.y;

        p_proj.fixed_rows::<3>(0).clone_owned()
    }

    /// Unprojects from 2D screen coordinates in range [0, `viewport_width`] and
    /// [0, `viewport_height`] to 3D world # Panics
    /// Will panic if the proj*view matrix is not invertable
    pub fn unproject(
        &self,
        win: na::Point3<f32>,
        view: na::Matrix4<f32>,
        proj: na::Matrix4<f32>,
        viewport_size: na::Vector2<f32>,
    ) -> na::Vector3<f32> {
        let inv = (proj * view).try_inverse().unwrap();

        let mut tmp = win.to_homogeneous();
        tmp.x /= viewport_size.x;
        tmp.y /= viewport_size.y;
        tmp = tmp * 2.0 - na::Vector4::<f32>::new(-1.0, -1.0, -1.0, -1.0);

        let mut obj = inv * tmp;
        obj = obj / obj.w;

        let scene = obj.fixed_rows::<3>(0).clone_owned();

        scene
    }

    /// Handle the event of touching with a finger
    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    pub fn touch_pressed(&mut self, touch_event: &Touch, scene: &mut Scene) {
        // println!("mouse pressed");
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        cam_control.id2active_touches.insert(touch_event.id, *touch_event);

        cam_control.mouse_pressed = true;

        //adding a finger already invalidates the prev mouse point. If there was one
        // finger before we added, then it means we switch from having the previous
        // position be at the finger to it being in between the two fingers so we
        // invalidate previous state
        cam_control.prev_mouse_pos_valid = false;
    }

    /// Handle the event of pressing mouse
    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    pub fn touch_released(&mut self, touch_event: &Touch, scene: &mut Scene) {
        // println!("mouse pressed");
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        cam_control.id2active_touches.remove(&touch_event.id);

        //no fingers are touching the screen
        if cam_control.id2active_touches.is_empty() {
            cam_control.mouse_pressed = false;
            cam_control.prev_mouse_pos_valid = false;
        }
        //releasing a finger already invalidates the prev mouse point. If there was one
        // finger before we removed, then we definitelly want to invalidate. If there
        // were two then it means we switch from the center of the two fingers to just
        // one finger so we also invalidate the previous pos
        cam_control.prev_mouse_pos_valid = false;
    }

    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    pub fn reset_all_touch_presses(&mut self, scene: &mut Scene) {
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        cam_control.id2active_touches.clear();

        cam_control.mouse_pressed = false;
        cam_control.prev_mouse_pos_valid = false;
        cam_control.prev_mouse_pos_valid = false;
    }

    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_lossless)]
    pub fn process_touch_move(&mut self, touch_event: &Touch, viewport_width: u32, viewport_height: u32, scene: &mut Scene) {
        let touches = {
            let cam_control = scene.get_comp::<&CamController>(&self.entity).unwrap();
            let all_touches: Vec<Touch> = cam_control.id2active_touches.values().copied().collect();
            all_touches
        };

        //if we have only one finger, we rotate
        if touches.len() == 1 {
            {
                let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();
                cam_control.mouse_mode = CamMode::Rotation;
            }
            self.process_mouse_move(
                touch_event.location.x as f32,
                touch_event.location.y as f32,
                viewport_width,
                viewport_height,
                scene,
            );
            //update position of this finger
            let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();
            cam_control.id2active_touches.insert(touch_event.id, *touch_event);
        } else if touches.len() == 2 {
            //calculate difference between the two fingers, did it increase or decrease
            // from the previous frame
            let pos0 = touches[0].location;
            let pos1 = touches[1].location;
            let pos0_na = na::Vector2::<f64>::new(pos0.x / viewport_width as f64, pos0.y / viewport_height as f64);
            let pos1_na = na::Vector2::<f64>::new(pos1.x / viewport_width as f64, pos1.y / viewport_height as f64);
            let diff_prev = (pos0_na - pos1_na).norm();
            // //update position of this finger
            let current_touches = {
                let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();
                cam_control.id2active_touches.insert(touch_event.id, *touch_event);
                let current_touches: Vec<Touch> = cam_control.id2active_touches.values().copied().collect();
                current_touches
            };
            //current difference between fingers
            let pos0 = current_touches[0].location;
            let pos1 = current_touches[1].location;
            let pos0_na = na::Vector2::<f64>::new(pos0.x / viewport_width as f64, pos0.y / viewport_height as f64);
            let pos1_na = na::Vector2::<f64>::new(pos1.x / viewport_width as f64, pos1.y / viewport_height as f64);
            let diff_cur = (pos0_na - pos1_na).norm();

            //check if the difference increased or not
            {
                let inc = diff_cur - diff_prev;
                let speed = 1.0;
                let mut pos_lookat = scene.get_comp::<&mut PosLookat>(&self.entity).unwrap();
                pos_lookat.dolly(inc as f32 * speed);
            }

            //also move the camera depening on where the center of the two fingers moved
            {
                let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();
                cam_control.mouse_mode = CamMode::Translation;
            }
            let center = na::Vector2::<f64>::new((pos0.x + pos1.x) / 2.0, (pos0.y + pos1.y) / 2.0);
            self.process_mouse_move(center.x as f32, center.y as f32, viewport_width, viewport_height, scene);
        }
    }

    /// Handle the event of pressing mouse
    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    pub fn mouse_pressed(&mut self, mouse_button: &MouseButton, scene: &mut Scene) {
        // println!("mouse pressed");
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        match mouse_button {
            MouseButton::Left => {
                cam_control.mouse_mode = CamMode::Rotation;
                cam_control.mouse_pressed = true;
            }
            MouseButton::Right => {
                cam_control.mouse_mode = CamMode::Translation;
                cam_control.mouse_pressed = true;
            }
            _ => {
                cam_control.mouse_pressed = false;
            }
        }
    }

    /// Handle the event of releasing mouse
    /// # Panics
    /// Will panic if the ``CamController`` component does not exist for this
    /// entity
    pub fn mouse_released(&mut self, scene: &mut Scene) {
        // println!("mouse released");
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        // println!("mouse released");
        cam_control.mouse_pressed = false;
        cam_control.prev_mouse_pos_valid = false;
    }

    /// Handle the event of dragging the mouse on the window
    /// # Panics
    /// Will panic if the ``PosLookat``, ``Projection``, ``CamController``
    /// component does not exist for this entity
    pub fn process_mouse_move(&mut self, x: f32, y: f32, viewport_width: u32, viewport_height: u32, scene: &mut Scene) {
        let proj = self.proj_matrix(scene);
        let mut pos_lookat = scene.get_comp::<&mut PosLookat>(&self.entity).unwrap();
        let mut cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        let current_mouse = na::Vector2::<f32>::new(x, y);
        #[allow(clippy::cast_precision_loss)] //it's ok, we don't have very big viewport sizes
        let viewport_size = na::Vector2::<f32>::new(viewport_width as f32, viewport_height as f32);

        if cam_control.mouse_pressed {
            if cam_control.mouse_mode == CamMode::Rotation && cam_control.prev_mouse_pos_valid {
                let speed = 2.0;
                let (rot_y, mut rot_x) = Self::two_axis_rotation(
                    na::Vector3::from(pos_lookat.cam_axes().column(0)),
                    viewport_size,
                    speed,
                    cam_control.prev_mouse,
                    current_mouse,
                );

                //calculate the new position as if we apply this rotation
                let mut new_pos_lookat = pos_lookat.clone();
                new_pos_lookat.orbit(rot_x);
                //calculate vertical angle
                let dot_up = new_pos_lookat.direction().dot(&na::Vector3::<f32>::y_axis());
                let angle_vertical = dot_up.acos();
                // println!("angle vertical {}", angle_vertical);
                if let Some(max_vertical_angle) = cam_control.limit_max_vertical_angle {
                    if angle_vertical > max_vertical_angle || new_pos_lookat.up == na::Vector3::<f32>::new(0.0, -1.0, 0.0) {
                        rot_x = na::Rotation3::<f32>::identity();
                    }
                }
                if let Some(min_vertical_angle) = cam_control.limit_min_vertical_angle {
                    if angle_vertical < min_vertical_angle {
                        rot_x = na::Rotation3::<f32>::identity();
                    }
                }

                let rot = rot_y * rot_x;
                pos_lookat.orbit(rot);
            } else if cam_control.mouse_mode == CamMode::Translation && cam_control.prev_mouse_pos_valid {
                let view = pos_lookat.view_matrix();

                let coord = self.project(pos_lookat.lookat, view, proj, viewport_size);
                let down_mouse_z = coord.z;

                let pos1 = self.unproject(na::Point3::<f32>::new(x, viewport_size.y - y, down_mouse_z), view, proj, viewport_size);
                let pos0 = self.unproject(
                    na::Point3::<f32>::new(cam_control.prev_mouse.x, viewport_size.y - cam_control.prev_mouse.y, down_mouse_z),
                    view,
                    proj,
                    viewport_size,
                );
                let diff = pos1 - pos0;
                // diff.array()*=speed_multiplier;
                let new_pos = pos_lookat.position - diff;
                pos_lookat.shift_cam(new_pos);
            }
            cam_control.prev_mouse = current_mouse;
            cam_control.prev_mouse_pos_valid = true;
        }
    }

    /// Handle event of scrolling the mouse wheel. It performs a zoom.
    /// # Panics
    /// Will panic if the ``PosLookat``, ``CamController`` component does not
    /// exist for this entity
    pub fn process_mouse_scroll(&mut self, delta: &MouseScrollDelta, scene: &mut Scene) {
        let mut pos_lookat = scene.get_comp::<&mut PosLookat>(&self.entity).unwrap();
        let cam_control = scene.get_comp::<&mut CamController>(&self.entity).unwrap();

        let scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => f64::from(scroll * 0.5),
            #[allow(clippy::cast_precision_loss)] //it's ok, we don't have very big numbers
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll,
        };

        let mut s = if scroll > 0.0 { 0.1 } else { -0.1 };

        if let Some(max_dist) = cam_control.limit_max_dist {
            let cur_dist = pos_lookat.dist_lookat();
            if cur_dist > max_dist && s < 0.0 {
                s = 0.0;
            }
        }

        // self.push_away(val);
        pos_lookat.dolly(s);
    }

    /// Resizing the window means that the projection matrix of the camera has
    /// to change accordingly so as to not squish the scene # Panics
    /// Will panic if the ``Projection`` component does not exist for this
    /// entity
    pub fn set_aspect_ratio(&mut self, val: f32, scene: &mut Scene) {
        //extract data
        let mut proj = scene.get_comp::<&mut Projection>(&self.entity).unwrap();
        if let Projection::WithFov(ref mut proj) = *proj {
            proj.aspect_ratio = val;
        }
    }

    /// Resizing the window means that the projection matrix of the camera has
    /// to change accordingly so as to not squish the scene
    pub fn set_aspect_ratio_maybe(&mut self, val: f32, scene: &mut Scene) {
        if let Ok(mut proj) = scene.get_comp::<&mut Projection>(&self.entity) {
            if let Projection::WithFov(ref mut proj) = *proj {
                proj.aspect_ratio = val;
            }
        } else if scene.nr_renderables() != 0 {
            //if the nr of renderabled is 0 then might not have a projection matrix and
            // that's fine because when we add renderables, the prepass will run and add a
            // projection matrix
            warn!("No Projection component yet so we couldn't set aspect ratio. This may not be an issue since the prepass might fix this. Ideally this warning should only appear at most once");
        }
    }

    pub fn near_far(&self, scene: &mut Scene) -> (f32, f32) {
        let proj = scene.get_comp::<&Projection>(&self.entity).unwrap();
        proj.near_far()
    }

    /// Unconditionally sets the target res, regardless of the update mode
    pub fn set_target_res(&mut self, width: u32, height: u32, scene: &mut Scene) {
        {
            let mut res = scene.get_comp::<&mut TargetResolution>(&self.entity).unwrap();
            res.width = width;
            res.height = height;
        }

        //sync also aspect ratio
        #[allow(clippy::cast_precision_loss)]
        self.set_aspect_ratio_maybe(width as f32 / height as f32, scene);
    }

    /// Sets the target res on window resizing, only updates if the updatemode
    /// is `WindowSize`
    pub fn on_window_resize(&mut self, width: u32, height: u32, scene: &mut Scene) {
        {
            let mut res = scene.get_comp::<&mut TargetResolution>(&self.entity).unwrap();
            if res.update_mode == TargetResolutionUpdate::WindowSize {
                res.width = width;
                res.height = height;
            }
        }

        //sync also aspect ratio
        #[allow(clippy::cast_precision_loss)]
        self.set_aspect_ratio_maybe(width as f32 / height as f32, scene);
    }

    pub fn get_target_res(&self, scene: &Scene) -> (u32, u32) {
        let res = scene.get_comp::<&TargetResolution>(&self.entity).unwrap();
        (res.width, res.height)
    }
}
