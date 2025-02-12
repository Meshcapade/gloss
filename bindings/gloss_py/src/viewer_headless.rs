use crate::{actor::PyActorMut, camera::PyCamera, device::PyDevice, queue::PyQueue, scene::PyScene, texture::PyTexture};

use gloss_renderer::{camera::Camera, config::Config, plugin_manager::Plugins, scene::Scene, viewer_headless::ViewerHeadless};

use easy_wgpu::texture::Texture;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use wgpu;

#[pyclass(name = "ViewerHeadless", module = "gloss", unsendable)] // it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PyViewerHeadless(pub ViewerHeadless);
impl std::ops::Deref for PyViewerHeadless {
    type Target = ViewerHeadless;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[pymethods]
impl PyViewerHeadless {
    #[new]
    #[pyo3(signature = (width, height, config_path=None))]
    #[pyo3(text_signature = "(width: int, height: int, config_path: Optional[str] = None) -> ViewerHeadless")]
    pub fn new(width: u32, height: u32, config_path: Option<&str>) -> Self {
        Self(ViewerHeadless::new_with_config(width, height, &Config::new(config_path)))
    }
    #[pyo3(text_signature = "($self, name: str) -> Entity")]
    pub fn get_or_create_entity(&mut self, name: &str) -> PyActorMut {
        let scene: &mut Scene = &mut self.0.scene;
        let entity = scene.get_or_create_entity(name).entity();
        PyActorMut::new(entity, &mut self.0.scene)
    }
    #[pyo3(text_signature = "($self, component: Any) -> None")]
    pub fn add_resource(&mut self, pycomp: Py<PyAny>) {
        let mut pyscene = self.get_scene();
        pyscene.add_resource(pycomp);
    }
    #[pyo3(text_signature = "($self) -> float")]
    pub fn start_frame(&mut self) -> f32 {
        let dt = self.0.start_frame();
        dt.as_secs_f32()
    }
    #[pyo3(text_signature = "($self) -> Device")]
    pub fn get_device(&mut self) -> PyDevice {
        let obj_ptr: *const wgpu::Device = self.0.gpu.device();
        PyDevice::new(obj_ptr)
    }
    #[pyo3(text_signature = "($self) -> Queue")]
    pub fn get_queue(&mut self) -> PyQueue {
        let obj_ptr: *const wgpu::Queue = self.0.gpu.queue();
        PyQueue::new(obj_ptr)
    }
    #[pyo3(text_signature = "($self) -> Scene")]
    pub fn get_scene(&mut self) -> PyScene {
        let obj_ptr: *mut Scene = &mut self.0.scene;
        PyScene::new(obj_ptr)
    }
    #[pyo3(text_signature = "($self) -> Camera")]
    pub fn get_camera(&mut self) -> PyCamera {
        let obj_ptr: *mut Camera = &mut self.0.camera;
        PyCamera::new(obj_ptr, self.get_scene())
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn update(&mut self) {
        self.0.update();
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn render_next_frame(&mut self) {
        self.start_frame();
        self.update();
    }
    #[pyo3(text_signature = "($self, path: str) -> None")]
    pub fn save_last_render(&mut self, path: &str) {
        let mut last_render = self.get_final_tex();
        last_render.save_to_file(&self.get_device(), &self.get_queue(), path);
    }
    #[pyo3(text_signature = "($self, camera: Camera) -> None")]
    pub fn render_from_cam(&mut self, cam: &mut PyCamera) {
        self.0.render_from_cam(cam);
    }
    #[pyo3(text_signature = "($self) -> Texture")]
    pub fn get_final_tex(&mut self) -> PyTexture {
        let ptr: *const Texture = self.0.get_final_tex();
        PyTexture::new(ptr)
    }
    #[pyo3(text_signature = "($self) -> Texture")]
    pub fn get_final_depth(&mut self) -> PyTexture {
        let ptr: *const Texture = self.0.get_final_depth();
        PyTexture::new(ptr)
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn get_linearised_depth(&mut self) -> Py<PyUntypedArray> {
        let (znear, zfar) = self.0.camera.near_far(&mut self.0.scene);
        self.get_final_depth().depth_linearize(&self.get_device(), &self.get_queue(), znear, zfar)
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn get_plugin_list_ptr(&mut self) -> u64 {
        let obj_ptr: *mut Plugins = &mut self.0.plugins;
        obj_ptr as u64
    }
    #[pyo3(text_signature = "($self, plugin: Any) -> None")]
    pub fn insert_plugin(mut slf: PyRefMut<'_, Self>, pycomp: Py<PyAny>) {
        // let obj_ptr: *mut Camera = &mut self.0.camera;
        Python::with_gil(|py| {
            let pyany = pycomp.bind(py);
            let args = (slf.get_plugin_list_ptr(),);
            let _ = pyany.call_method("insert_plugin", args, None).unwrap();
        });
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn run_manual_plugins(&mut self) {
        let v = &mut self.0;
        v.run_manual_plugins();
    }
}
