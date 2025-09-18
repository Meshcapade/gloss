macro_rules! ops_tensor_tensor {
    (float($($t:ident),*) => $op:ident) => {
        ops_tensor_tensor!($($t),* | FloatTensorOps | $op | MultiFloatTensor)
    };

    (int($($t:ident),*) => $op:ident) => {
        ops_tensor_tensor!($($t),* | IntTensorOps | $op | MultiIntTensor)
    };

    ($($t:ident),* | $trait:ident | $op:ident | $tensor:ident) => {
        match ($($t,)*) {
            ($($tensor::NdArray($t),)*) => {
                use crate::backend::NdArrayBackend;

                $tensor::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op($($t,)*))
            }
            ($($tensor::Wgpu($t),)*) => {
                use crate::backend::WgpuBackend;

                $tensor::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op($($t,)*))
            }
            _ => panic!("Invalid device."),
        }

    };
}

macro_rules! ops_tensor {
    // Entry point for float operations
    (float($tensor:ident) => $op:ident) => {
        ops_tensor!(@impl $tensor, FloatTensorOps, $op, MultiFloatTensor)
    };

    // Entry point for int operations
    (int($tensor:ident) => $op:ident) => {
        ops_tensor!(@impl $tensor, IntTensorOps, $op, MultiIntTensor)
    };

    // Implementation that handles the actual dispatching
    (@impl $tensor:ident, $trait:ident, $op:ident, $tensor_type:ident) => {
        match $tensor {
            $tensor_type::NdArray(t) => {
                use crate::backend::NdArrayBackend;

                $tensor_type::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op(t))
            }
            $tensor_type::Wgpu(t) => {
                use crate::backend::WgpuBackend;

                $tensor_type::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op(t))
            }
        }
    };
}

macro_rules! ops_tensor_scalar {
    (float($t:ident, $s:ident) => $op:ident) => {
        ops_tensor_scalar!(($t, $s) | FloatTensorOps | $op | MultiFloatTensor)
    };

    (int($t:ident, $s:ident) => $op:ident) => {
        ops_tensor_scalar!(($t, $s) | IntTensorOps | $op | MultiIntTensor)
    };

    (($t:ident, $s:ident) | $trait:ident | $op:ident | $tensor:ident) => {
        match ($t, $s) {
            ($tensor::NdArray($t), $s) => {
                use crate::backend::NdArrayBackend;

                $tensor::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op($t, $s.into()))
            }
            ($tensor::Wgpu($t), $s) => {
                use crate::backend::WgpuBackend;

                $tensor::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op($t, $s))
            } // _ => panic!("Invalid device."),
        }
    };
}

macro_rules! ops_tensor_rest {
    // Entry point for float operations
    (float($tensor:ident $(, $rest:expr)*) => $op:ident) => {
        ops_tensor_rest!(@impl $tensor, FloatTensorOps, $op, MultiFloatTensor $(, $rest)*)
    };

    // Entry point for int operations
    (int($tensor:ident $(, $rest:expr)*) => $op:ident) => {
        ops_tensor_rest!(@impl $tensor, IntTensorOps, $op, MultiIntTensor $(, $rest)*)
    };

    // Implementation that handles the actual dispatching
    (@impl $tensor:ident, $trait:ident, $op:ident, $tensor_type:ident $(, $rest:expr)*) => {
        match $tensor {
            $tensor_type::NdArray(t) => {
                use crate::backend::NdArrayBackend;

                $tensor_type::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op(t $(, $rest)*))
            }
            $tensor_type::Wgpu(t) => {
                use crate::backend::WgpuBackend;

                $tensor_type::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op(t $(, $rest)*))
            }
        }
    };
}

macro_rules! ops_rest_device {
    // Entry point for float operations - explicit device parameter
    (float($($args:expr),* ; $device:expr) => $op:ident) => {
        match $device {
            MultiDevice::NdArray(device) => {
                use crate::backend::NdArrayBackend;

                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op($($args),*, device))
            }
            MultiDevice::Wgpu(device) => {
                use crate::backend::WgpuBackend;

                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op($($args),*, device))
            }
        }
    };

    // Entry point for int operations - explicit device parameter
    (int($($args:expr),* ; $device:expr) => $op:ident) => {
        match $device {
            MultiDevice::NdArray(device) => {
                use crate::backend::NdArrayBackend;

                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op($($args),*, device))
            }
            MultiDevice::Wgpu(device) => {
                use crate::backend::WgpuBackend;

                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op($($args),*, device))
            }
        }
    };
}

macro_rules! ops_dim_tensor_indices {
    // For float tensors
    (float($dim:expr, $tensor:ident, $indices:ident) => $op:ident) => {
        match ($tensor, $indices) {
            (MultiFloatTensor::NdArray(t), MultiIntTensor::NdArray(i)) => {
                use crate::backend::NdArrayBackend;
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op($dim, t, i))
            }
            (MultiFloatTensor::Wgpu(t), MultiIntTensor::Wgpu(i)) => {
                use crate::backend::WgpuBackend;
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op($dim, t, i))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
    // For int tensors
    (int($dim:expr, $tensor:ident, $indices:ident) => $op:ident) => {
        match ($tensor, $indices) {
            (MultiIntTensor::NdArray(t), MultiIntTensor::NdArray(i)) => {
                use crate::backend::NdArrayBackend;
                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op($dim, t, i))
            }
            (MultiIntTensor::Wgpu(t), MultiIntTensor::Wgpu(i)) => {
                use crate::backend::WgpuBackend;
                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op($dim, t, i))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
}

macro_rules! ops_dim_tensor_indices_values {
    // For float tensors
    (float($dim:expr, $tensor:ident, $indices:ident, $value:ident) => $op:ident) => {
        match ($tensor, $indices, $value) {
            (MultiFloatTensor::NdArray(t), MultiIntTensor::NdArray(i), MultiFloatTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op($dim, t, i, v))
            }
            (MultiFloatTensor::Wgpu(t), MultiIntTensor::Wgpu(i), MultiFloatTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op($dim, t, i, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
    // For int tensors
    (int($dim:expr, $tensor:ident, $indices:ident, $value:ident) => $op:ident) => {
        match (tensor, $indices, $value) {
            (MultiIntTensor::NdArray(t), MultiIntTensor::NdArray(i), MultiIntTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op($dim, t, i, v))
            }
            (MultiIntTensor::Wgpu(t), MultiIntTensor::Wgpu(i), MultiIntTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op($dim, t, i, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
}

macro_rules! ops_tensor_dim_indices {
    // For float tensors
    (float($tensor:ident, $dim:expr, $indices:ident) => $op:ident) => {
        match ($tensor, $indices) {
            (MultiFloatTensor::NdArray(t), MultiIntTensor::NdArray(i)) => {
                use crate::backend::NdArrayBackend;
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op(t, $dim, i))
            }
            (MultiFloatTensor::Wgpu(t), MultiIntTensor::Wgpu(i)) => {
                use crate::backend::WgpuBackend;
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op(t, $dim, i))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
    // For int tensors
    (int($tensor:ident, $dim:expr, $indices:ident) => $op:ident) => {
        match ($tensor, $indices) {
            (MultiIntTensor::NdArray(t), MultiIntTensor::NdArray(i)) => {
                use crate::backend::NdArrayBackend;
                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op(t, $dim, i))
            }
            (MultiIntTensor::Wgpu(t), MultiIntTensor::Wgpu(i)) => {
                use crate::backend::WgpuBackend;
                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op(t, $dim, i))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
}

macro_rules! ops_tensor_dim_indices_values {
    // For float tensors
    (float($tensor:ident, $dim:expr, $indices:ident, $value:ident) => $op:ident) => {
        match ($tensor, $indices, $value) {
            (MultiFloatTensor::NdArray(t), MultiIntTensor::NdArray(i), MultiFloatTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op(t, $dim, i, v))
            }
            (MultiFloatTensor::Wgpu(t), MultiIntTensor::Wgpu(i), MultiFloatTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op(t, $dim, i, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
    // For int tensors
    (int($tensor:ident, $dim:expr, $indices:ident, $value:ident) => $op:ident) => {
        match ($tensor, $indices, $value) {
            (MultiIntTensor::NdArray(t), MultiIntTensor::NdArray(i), MultiIntTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op(t, $dim, i, v))
            }
            (MultiIntTensor::Wgpu(t), MultiIntTensor::Wgpu(i), MultiIntTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op(t, $dim, i, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
}

macro_rules! ops_tensor_other_values {
    // For float tensors
    (float($tensor:ident, $other:ident, $value:ident) => $op:ident) => {
        match ($tensor, $value) {
            (MultiFloatTensor::NdArray(t), MultiFloatTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiFloatTensor::NdArray(<NdArrayBackend as FloatTensorOps<NdArrayBackend>>::$op(t, $other, v))
            }
            (MultiFloatTensor::Wgpu(t), MultiFloatTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiFloatTensor::Wgpu(<WgpuBackend as FloatTensorOps<WgpuBackend>>::$op(t, $other, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
    // For int tensors
    (int($tensor:ident, $other:ident, $value:ident) => $op:ident) => {
        match ($tensor, $value) {
            (MultiIntTensor::NdArray(t), MultiIntTensor::NdArray(v)) => {
                use crate::backend::NdArrayBackend;
                MultiIntTensor::NdArray(<NdArrayBackend as IntTensorOps<NdArrayBackend>>::$op(t, $other, v))
            }
            (MultiIntTensor::Wgpu(t), MultiIntTensor::Wgpu(v)) => {
                use crate::backend::WgpuBackend;
                MultiIntTensor::Wgpu(<WgpuBackend as IntTensorOps<WgpuBackend>>::$op(t, $other, v))
            }
            _ => panic!("Mismatched tensor backends"),
        }
    };
}

//things related to bool
macro_rules! ops_tensor_rest_ret_bool {
    // Entry point for float operations
    (float($tensor:ident $(, $rest:expr)*) => $op:ident) => {
        ops_tensor_rest_ret_bool!(@impl $tensor, FloatTensorOps, $op, MultiFloatTensor $(, $rest)*)
    };

    // Entry point for int operations
    (int($tensor:ident $(, $rest:expr)*) => $op:ident) => {
        ops_tensor_rest_ret_bool!(@impl $tensor, IntTensorOps, $op, MultiIntTensor $(, $rest)*)
    };

    // Implementation that handles the actual dispatching
    (@impl $tensor:ident, $trait:ident, $op:ident, $tensor_type:ident $(, $rest:expr)*) => {
        match $tensor {
            $tensor_type::NdArray(t) => {
                use crate::backend::NdArrayBackend;

                MultiBoolTensor::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op(t $(, $rest)*))
            }
            $tensor_type::Wgpu(t) => {
                use crate::backend::WgpuBackend;

                MultiBoolTensor::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op(t $(, $rest)*))
            }
        }
    };
}

macro_rules! ops_tensor_ret_float {
    // Entry point for float operations
    (float($tensor:ident) => $op:ident) => {
        ops_tensor_ret_float!(@impl $tensor, FloatTensorOps, $op, MultiFloatTensor)
    };

    // Entry point for int operations
    (int($tensor:ident) => $op:ident) => {
        ops_tensor_ret_float!(@impl $tensor, IntTensorOps, $op, MultiIntTensor)
    };

    // Entry point for bool operations
    (bool($tensor:ident) => $op:ident) => {
        ops_tensor_ret_float!(@impl $tensor, BoolTensorOps, $op, MultiBoolTensor)
    };

    // Implementation that handles the actual dispatching
    (@impl $tensor:ident, $trait:ident, $op:ident, $tensor_type:ident) => {
        match $tensor {
            $tensor_type::NdArray(t) => {
                use crate::backend::NdArrayBackend;

                MultiFloatTensor::NdArray(<NdArrayBackend as $trait<NdArrayBackend>>::$op(t))
            }
            $tensor_type::Wgpu(t) => {
                use crate::backend::WgpuBackend;

                MultiFloatTensor::Wgpu(<WgpuBackend as $trait<WgpuBackend>>::$op(t))
            }
        }
    };
}
