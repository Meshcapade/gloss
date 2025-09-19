use burn::{
    prelude::Backend,
    tensor::{backend::DeviceOps, ops::Device},
};

use crate::tensor::MultiBoolTensor;
use crate::tensor::MultiFloatTensor;
use crate::tensor::MultiIntTensor;

pub type NdArrayBackend = burn::backend::NdArray<f32, i32>;
pub type WgpuBackend = burn::backend::Wgpu<f32, i32>;

#[derive(Clone, Copy, Default, Debug)]
pub struct MultiBackend;

impl Backend for MultiBackend {
    type Device = MultiDevice;
    type FloatTensorPrimitive = MultiFloatTensor;
    type IntTensorPrimitive = MultiIntTensor;
    type BoolTensorPrimitive = MultiBoolTensor;
    type QuantizedTensorPrimitive = MultiIntTensor;

    type FloatElem = f32;

    type IntElem = i32;

    type BoolElem = u8;

    fn name(device: &Self::Device) -> String {
        match device {
            MultiDevice::NdArray(_) => "ndarray",
            MultiDevice::Wgpu(_) => "wgpu",
        }
        .to_string()
    }

    fn seed(_seed: u64) {
        //with a newer version of burn we have here access to the device so we can use a match statement
        todo!()
    }

    type QuantizedEncoding = f32;

    fn ad_enabled() -> bool {
        false
    }

    fn sync(_device: &Self::Device) {}
}

#[non_exhaustive]
#[allow(non_snake_case)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultiDevice {
    NdArray(Device<NdArrayBackend>),
    // #[cfg(feature = "wgpu")]
    Wgpu(Device<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(Box<Device<MultiBackend>>),
}
impl Default for MultiDevice {
    fn default() -> Self {
        // Self::NdArray(NdArrayDevice::default())
        //If the viewer has already been initialized, we want to use the same wgpu device, if not we create a new one
        let existing_wgpu_device = wgpu_burn_global_device::get_global_wgpu_device();
        Self::Wgpu(existing_wgpu_device.unwrap_or_default())
    }
}

#[allow(non_snake_case)]
impl DeviceOps for MultiDevice {
    fn id(&self) -> burn::tensor::backend::DeviceId {
        match self {
            MultiDevice::NdArray(_) => burn::tensor::backend::DeviceId::new(0, 0),
            MultiDevice::Wgpu(_) => burn::tensor::backend::DeviceId::new(1, 0),
        }
    }
}
