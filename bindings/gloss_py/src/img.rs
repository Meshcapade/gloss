use gloss_hecs::Entity;
use gloss_img::DynImage;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use image::ImageBuffer;
use numpy::{dtype_bound, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, ToPyArray};
use pyo3::{exceptions::PyTypeError, prelude::*};
#[pyclass(name = "DynImage", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyDynImage {
    pub inner: DynImage,
}
#[pymethods]
impl PyDynImage {
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> DynImage")]
    #[allow(clippy::missing_errors_doc)]
    pub fn new_from_file(path: &str) -> PyResult<Self> {
        Ok(PyDynImage {
            inner: gloss_img::dynamic_image::open(path).unwrap(),
        })
    }

    #[staticmethod]
    #[pyo3(text_signature = "(array: NDArray[np.float32]) -> DynImage")]
    #[allow(clippy::missing_errors_doc)]
    pub fn new_from_numpy(py: Python, array: &Bound<'_, PyUntypedArray>) -> PyResult<Self> {
        fn impl_u8(array: &Bound<'_, PyArray3<u8>>, h: u32, w: u32, c: u32) -> PyResult<PyDynImage> {
            let buf = array.to_vec().unwrap();

            match c {
                1 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLuma8).unwrap(),
                }),
                3 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgb8).unwrap(),
                }),
                4 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgba8).unwrap(),
                }),
                _ => Err(PyTypeError::new_err(format!("Unsupported element type u8 and channel_nr: {c}"))),
            }
        }
        fn impl_f32(array: &Bound<'_, PyArray3<f32>>, h: u32, w: u32, c: u32) -> PyResult<PyDynImage> {
            let buf = array.to_vec().unwrap();

            match c {
                1 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLuma32F).unwrap(),
                }),
                3 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgb32F).unwrap(),
                }),
                4 => Ok(PyDynImage {
                    inner: ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgba32F).unwrap(),
                }),
                _ => Err(PyTypeError::new_err(format!("Unsupported element type f32 and channel_nr: {c}"))),
            }
        }

        assert!(
            array.ndim() == 3,
            "numpy array needs to have 3 dimensions corresponding to HWC but it has dim= {}",
            array.ndim()
        );
        let shape = array.shape();
        let c = u32::try_from(shape[2]).unwrap();
        let h = u32::try_from(shape[0]).unwrap();
        let w = u32::try_from(shape[1]).unwrap();
        assert!(c <= 4, "numpy array has to be hwc where c<=4 but the shape is {shape:?}");

        //dispatch depending on type
        let element_type = array.dtype();
        if element_type.is_equiv_to(&dtype_bound::<f32>(py)) {
            let array: &Bound<'_, PyArray3<f32>> = array.downcast().unwrap();
            impl_f32(array, h, w, c)
        } else if element_type.is_equiv_to(&dtype_bound::<u8>(py)) {
            let array: &Bound<'_, PyArray3<u8>> = array.downcast().unwrap();
            impl_u8(array, h, w, c)
        } else {
            Err(PyTypeError::new_err(format!("Unsupported element type: {element_type}")))
        }
    }

    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&self, py: Python<'_>) -> Py<PyUntypedArray> {
        let c = self.inner.channels() as usize;
        let h = self.inner.height() as usize;
        let w = self.inner.width() as usize;
        if let Some(flat_samples) = self.inner.as_flat_samples_u8() {
            let pyarr = flat_samples.samples.to_pyarray_bound(py);
            // let pyarr = PyArray::from_vec_bound(py, flat_samples.samples.to_vec());
            let pyarr = pyarr.reshape((h, w, c)).unwrap();
            let pyatt_untyped: &PyUntypedArray = pyarr.into_gil_ref();
            pyatt_untyped.into_py(py)
        } else if let Some(flat_samples) = self.inner.as_flat_samples_f32() {
            let pyarr = flat_samples.samples.to_pyarray_bound(py);
            let pyarr = pyarr.reshape((h, w, c)).unwrap();
            let pyatt_untyped: &PyUntypedArray = pyarr.into_gil_ref();
            pyatt_untyped.into_py(py)
        } else {
            panic!("Unkown conversion to numpy");
        }
    }

    #[pyo3(text_signature = "($self) -> int")]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn channels(&self) -> u32 {
        self.inner.channels()
    }
}
