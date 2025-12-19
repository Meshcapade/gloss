// use burn::backend::{Wgpu};
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use crate::tensor_utils::{burn2pytensor, pytensor2burn};

#[pyclass(name = "Torch2BurnTest", module = "gloss", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyTorch2BurnTest {
    pub inner: f32, //just a dummy inner value
}
#[pymethods]
impl PyTorch2BurnTest {
    #[new]
    #[pyo3(text_signature = "() -> Torch2BurnTest")]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { inner: 0.0 }
    }

    #[pyo3(text_signature = "($self, input: PyTensor, weights: PyTensor) -> PyTensor")]
    pub fn forward(&mut self, input: PyTensor, weights: PyTensor) -> PyTensor {
        let input = pytensor2burn::<2>(input);
        let weights = pytensor2burn::<2>(weights);

        // run operations on the burn tensor
        let out = input.clone().matmul(weights.clone()) + input.matmul(weights).sin() * 0.1;

        // get back to a PyTensor and return it
        burn2pytensor(out)
    }
}
