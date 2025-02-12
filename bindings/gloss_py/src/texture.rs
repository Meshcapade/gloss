use crate::{img::PyDynImage, PyDevice, PyQueue};
use easy_wgpu::texture::Texture;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use wgpu::TextureFormat;

#[pyclass(name = "Texture", module = "gloss", unsendable)]
pub struct PyTexture {
    obj_ptr: *const Texture,
}
//non python method
impl PyTexture {
    pub fn new(obj_ptr: *const Texture) -> Self {
        PyTexture { obj_ptr }
    }
}
#[pymethods]
impl PyTexture {
    #[pyo3(text_signature = "($self, device: Device, queue: Queue, path: str) -> None")]
    pub fn save_to_file(&mut self, device: &PyDevice, queue: &PyQueue, path: &str) {
        let tex = unsafe { &*self.obj_ptr };
        let dyn_img = pollster::block_on(tex.download_to_cpu(device, queue));
        let _ = dyn_img.save(path);
    }
    #[pyo3(text_signature = "($self, device: Device, queue: Queue) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>, device: &PyDevice, queue: &PyQueue) -> Py<PyUntypedArray> {
        let tex = unsafe { &*self.obj_ptr };

        //panics if depth map retrieval is attempted with MSAA sample count set to > 1
        assert!(
            !(tex.texture.sample_count() > 1 && tex.texture.format() == TextureFormat::Depth32Float),
            "InvalidSampleCount: Depth maps not supported for MSAA sample count {} (Use a config to set msaa_nr_samples as 1)",
            tex.texture.sample_count()
        );

        let dynamic_img = pollster::block_on(tex.download_to_cpu(device, queue));
        let pydyn_img = PyDynImage { inner: dynamic_img };
        pydyn_img.numpy(py)
    }
    #[pyo3(text_signature = "($self, device: Device, queue: Queue, near: float, far: float) -> NDArray[np.float32]")]
    pub fn depth_linearize(&mut self, device: &PyDevice, queue: &PyQueue, near: f32, far: f32) -> Py<PyUntypedArray> {
        let tex = unsafe { &*self.obj_ptr };
        let linearized_dynamic_img = tex.depth_linearize(device, queue, near, far);
        let pydyn_img = PyDynImage {
            inner: linearized_dynamic_img,
        };
        Python::with_gil(|py| pydyn_img.numpy(py))
    }
}
