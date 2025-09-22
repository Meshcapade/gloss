// Required imports
use burn::tensor::{Int, Tensor};

use crate::cubecl::{cube2tensor, tensor2cube, tensor2cube_int};
use crate::{csr::VertexFaceCSRBurn, cubecl_tangents};

use gloss_burn_multibackend::backend::MultiBackend;

pub fn compute_tangents_cubecl(
    verts: Tensor<MultiBackend, 2>,
    faces: Tensor<MultiBackend, 2, Int>,
    normals: Tensor<MultiBackend, 2>,
    uv: Tensor<MultiBackend, 2>,
    csr: &VertexFaceCSRBurn<MultiBackend>,
) -> Tensor<MultiBackend, 2> {
    let verts_cube = tensor2cube(verts);
    let faces_cube = tensor2cube_int(faces);
    let normals_cube = tensor2cube(normals);
    let uv_cube = tensor2cube(uv);
    let row_ptr_cube = tensor2cube_int(csr.row_ptr.clone());
    let col_idx_cube = tensor2cube_int(csr.col_idx.clone());
    let num_vertices = csr.num_vertices;

    let (faces_tangents_cube, faces_bitangents_cube) = cubecl_tangents::compute_tangents::face_tangents_launch::<
        cubecl::wgpu::WgpuRuntime,
        <MultiBackend as burn::prelude::Backend>::FloatElem,
        <MultiBackend as burn::prelude::Backend>::IntElem,
        <MultiBackend as burn::prelude::Backend>::BoolElem,
    >(verts_cube.clone(), uv_cube.clone(), faces_cube.clone());

    let vert_tangents_cube = cubecl_tangents::compute_tangents::vertex_tangents_launch::<
        cubecl::wgpu::WgpuRuntime,
        <MultiBackend as burn::prelude::Backend>::FloatElem,
        <MultiBackend as burn::prelude::Backend>::IntElem,
        <MultiBackend as burn::prelude::Backend>::BoolElem,
    >(
        faces_tangents_cube.clone(),
        faces_bitangents_cube.clone(),
        row_ptr_cube.clone(),
        col_idx_cube.clone(),
        normals_cube.clone(),
        num_vertices,
    );

    cube2tensor(vert_tangents_cube)
}
