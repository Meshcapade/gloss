use crate::global_backend;
use burn::{
    prelude::Backend,
    tensor::{backend::DeviceOps, ops::Device},
};

use crate::tensor::MultiBoolTensor;
use crate::tensor::MultiFloatTensor;
use crate::tensor::MultiIntTensor;

//TODO maybe switch to i32 for all backends?
//IF YOU CHANGE THIS, CHANGE THE IntTensorOps int_from_data also together with TensorMetadata for MultiIntTensor
#[cfg(feature = "burn-candle")]
pub type CandleBackend = burn::backend::Candle<f32, i64>;
#[cfg(feature = "burn-ndarray")]
pub type NdArrayBackend = burn::backend::NdArray<f32, i32>;
#[cfg(feature = "burn-wgpu")]
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

    // TODO this probably needs to be i64 if candle is used
    type IntElem = i32;

    type BoolElem = u8;

    fn name(device: &Self::Device) -> String {
        match device {
            #[cfg(feature = "burn-candle")]
            MultiDevice::Candle(_) => "candle",
            #[cfg(feature = "burn-ndarray")]
            MultiDevice::NdArray(_) => "ndarray",
            #[cfg(feature = "burn-wgpu")]
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

#[allow(non_snake_case)]
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultiDevice {
    #[cfg(feature = "burn-candle")]
    Candle(Device<CandleBackend>),
    #[cfg(feature = "burn-ndarray")]
    NdArray(Device<NdArrayBackend>),
    #[cfg(feature = "burn-wgpu")]
    Wgpu(Device<WgpuBackend>),
    // #[cfg(feature = "autodiff")]
    // Autodiff(Box<Device<MultiBackend>>),
}
impl Default for MultiDevice {
    fn default() -> Self {
        //if we set a global device, we select backend based on that
        #[allow(unreachable_patterns)]
        if let Some(global_device) = global_backend::get_global_burn_backend() {
            match global_device {
                #[cfg(feature = "burn-candle")]
                global_backend::GlobalBackend::Candle => return Self::Candle(burn::backend::candle::CandleDevice::default()),
                #[cfg(feature = "burn-ndarray")]
                global_backend::GlobalBackend::NdArray => return Self::NdArray(burn::backend::ndarray::NdArrayDevice::default()),
                #[cfg(feature = "burn-wgpu")]
                global_backend::GlobalBackend::Wgpu => {
                    //If the viewer has already been initialized, we want to use the same wgpu device, if not we create a new one
                    let existing_wgpu_device = wgpu_burn_global_device::get_global_wgpu_device();
                    return Self::Wgpu(existing_wgpu_device.unwrap_or_default());
                }
                _ => {
                    panic!("This global device {global_device:?} is not available because the corresponding feature is not enabled. Please enable the feature in Cargo.toml.");
                }
            }
        }

        //if no global device is set, we default to candle if available, otherwise ndarray, otherwise wgpu
        #[cfg(feature = "burn-candle")]
        {
            Self::Candle(burn::backend::candle::CandleDevice::default())
        }
        #[cfg(all(not(feature = "burn-candle"), feature = "burn-ndarray"))]
        {
            Self::NdArray(burn::backend::ndarray::NdArrayDevice::default());
        }
        #[cfg(all(not(feature = "burn-candle"), not(feature = "burn-ndarray"), feature = "burn-wgpu"))]
        {
            //If the viewer has already been initialized, we want to use the same wgpu device, if not we create a new one
            let existing_wgpu_device = wgpu_burn_global_device::get_global_wgpu_device();
            Self::Wgpu(existing_wgpu_device.unwrap_or_default())
        }
        #[cfg(all(not(feature = "burn-candle"), not(feature = "burn-ndarray"), not(feature = "burn-wgpu")))]
        {
            compile_error!("No backend feature enabled. Please enable at least one of the features: burn-candle, burn-ndarray, burn-wgpu");
        }
    }
}

#[allow(non_snake_case)]
impl DeviceOps for MultiDevice {
    fn id(&self) -> burn::tensor::backend::DeviceId {
        match self {
            #[cfg(feature = "burn-candle")]
            MultiDevice::Candle(_) => burn::tensor::backend::DeviceId::new(0, 0),
            #[cfg(feature = "burn-ndarray")]
            MultiDevice::NdArray(_) => burn::tensor::backend::DeviceId::new(1, 0),
            #[cfg(feature = "burn-wgpu")]
            MultiDevice::Wgpu(_) => burn::tensor::backend::DeviceId::new(2, 0),
        }
    }
}
