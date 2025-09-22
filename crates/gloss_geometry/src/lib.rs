pub mod csr;
pub mod cubecl_normals;
pub mod cubecl_tangents;
pub mod geom;

pub mod cubecl {
    pub use crate::cubecl_normals::launch_normals::compute_per_vertex_normals_cubecl;
    pub use crate::cubecl_tangents::launch_tangents::compute_tangents_cubecl;

    use burn::tensor::{Int, Tensor, TensorMetadata, TensorPrimitive};
    use burn_cubecl::tensor::CubeTensor;
    use core::panic;
    use gloss_burn_multibackend::{backend::MultiBackend, tensor::MultiFloatTensor, tensor::MultiIntTensor};

    pub fn tensor2cube<const D: usize>(tensor: Tensor<MultiBackend, D>) -> CubeTensor<cubecl::wgpu::WgpuRuntime> {
        let prim = tensor.into_primitive();
        let TensorPrimitive::Float(t) = prim else {
            panic!("Expected float tensor got {:?}", prim.dtype())
        };
        match t {
            MultiFloatTensor::Wgpu(t) => t,
            _ => panic!("Expected wgpu tensor got {:?}", t.dtype()),
        }
    }

    pub fn tensor2cube_int<const D: usize>(tensor: Tensor<MultiBackend, D, Int>) -> CubeTensor<cubecl::wgpu::WgpuRuntime> {
        let prim = tensor.into_primitive();
        match prim {
            MultiIntTensor::Wgpu(t) => t,
            _ => panic!("Expected wgpu tensor got {:?}", prim.dtype()),
        }
    }

    pub fn cube2tensor<const D: usize>(t: CubeTensor<cubecl::wgpu::WgpuRuntime>) -> Tensor<MultiBackend, D> {
        Tensor::from_primitive(TensorPrimitive::Float(MultiFloatTensor::Wgpu(t)))
    }
}
