// Required imports
use burn::tensor::{Int, Tensor};

use crate::cubecl::{cube2tensor, tensor2cube, tensor2cube_int};
use crate::{csr::VertexFaceCSRBurn, cubecl_normals};

use gloss_burn_multibackend::backend::MultiBackend;

pub fn compute_per_vertex_normals_cubecl(
    verts: Tensor<MultiBackend, 2>,
    faces: Tensor<MultiBackend, 2, Int>,
    csr: &VertexFaceCSRBurn<MultiBackend>,
) -> Tensor<MultiBackend, 2> {
    let verts_cube = tensor2cube(verts);
    let faces_cube = tensor2cube_int(faces);
    let row_ptr_cube = tensor2cube_int(csr.row_ptr.clone());
    let col_idx_cube = tensor2cube_int(csr.col_idx.clone());
    let num_vertices = csr.num_vertices;

    let faces_normals_cube = cubecl_normals::compute_normals::face_normals_launch::<
        cubecl::wgpu::WgpuRuntime,
        <MultiBackend as burn::prelude::Backend>::FloatElem,
        <MultiBackend as burn::prelude::Backend>::IntElem,
        <MultiBackend as burn::prelude::Backend>::BoolElem,
    >(verts_cube.clone(), faces_cube.clone());

    let vert_normals_cube = cubecl_normals::compute_normals::vertex_normals_launch::<
        cubecl::wgpu::WgpuRuntime,
        <MultiBackend as burn::prelude::Backend>::FloatElem,
        <MultiBackend as burn::prelude::Backend>::IntElem,
        <MultiBackend as burn::prelude::Backend>::BoolElem,
    >(faces_normals_cube.clone(), row_ptr_cube.clone(), col_idx_cube.clone(), num_vertices);

    cube2tensor(vert_normals_cube)
}
