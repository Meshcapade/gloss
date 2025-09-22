use burn::tensor::TensorMetadata;
use burn::tensor::{
    ops::{FloatTensor, IntTensor},
    Shape,
};
use burn_cubecl::{tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use cubecl::prelude::*;

#[cube(launch)]
#[allow(clippy::similar_names)]
pub fn face_tangents_kernel<F: Float>(
    verts: &Tensor<F>,               // flattened [num_verts, 3]
    uvs: &Tensor<F>,                 // flattened [num_verts, 2]
    faces: &Tensor<i32>,             // flattened [num_faces, 3]
    face_tangents: &mut Tensor<F>,   // flattened [num_faces, 3]
    face_bitangents: &mut Tensor<F>, // flattened [num_faces, 3]
) {
    let fid = ABSOLUTE_POS_X;

    let num_faces = face_tangents.shape(0);
    if fid >= num_faces {
        terminate!();
    }

    // Get vertex indices for this face
    let mut v_indices = Line::<i32>::empty(3u32);
    for i in 0..3 {
        let face_idx = fid * 3 + i;
        v_indices[i] = faces[face_idx] as i32;
    }

    // Load vertex positions
    let mut v0 = Line::<F>::empty(3u32);
    let mut v1 = Line::<F>::empty(3u32);
    let mut v2 = Line::<F>::empty(3u32);

    for c in 0..3 {
        v0[c] = verts[v_indices[0] as u32 * 3 + c];
        v1[c] = verts[v_indices[1] as u32 * 3 + c];
        v2[c] = verts[v_indices[2] as u32 * 3 + c];
    }

    // Load UV coordinates
    let mut uv0 = Line::<F>::empty(2u32);
    let mut uv1 = Line::<F>::empty(2u32);
    let mut uv2 = Line::<F>::empty(2u32);

    for c in 0..2 {
        uv0[c] = uvs[v_indices[0] as u32 * 2 + c];
        uv1[c] = uvs[v_indices[1] as u32 * 2 + c];
        uv2[c] = uvs[v_indices[2] as u32 * 2 + c];
    }

    // Compute position deltas
    let mut delta_pos1 = Line::<F>::empty(3u32);
    let mut delta_pos2 = Line::<F>::empty(3u32);
    for c in 0..3 {
        delta_pos1[c] = v1[c] - v0[c];
        delta_pos2[c] = v2[c] - v0[c];
    }

    // Compute UV deltas
    let mut delta_uv1 = Line::<F>::empty(2u32);
    let mut delta_uv2 = Line::<F>::empty(2u32);
    for c in 0..2 {
        delta_uv1[c] = uv1[c] - uv0[c];
        delta_uv2[c] = uv2[c] - uv0[c];
    }

    // denominator (for solving tangent/bitangent)
    let denom = delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0];
    let eps = F::new(1e-6);
    let r = F::new(1.0) / (denom + eps);

    // tangent = (deltaPos1 * dv2 - deltaPos2 * dv1) * r
    // bitangent = (deltaPos2 * du1 - deltaPos1 * du2) * r
    let mut tangent = Line::<F>::empty(3u32);
    let mut bitangent = Line::<F>::empty(3u32);

    for c in 0..3 {
        tangent[c] = (delta_pos1[c] * delta_uv2[1] - delta_pos2[c] * delta_uv1[1]) * r;
        bitangent[c] = (delta_pos2[c] * delta_uv1[0] - delta_pos1[c] * delta_uv2[0]) * r;
    }

    // Normalize tangent and bitangent (helps stability)
    let mut tlen_sq = F::new(0.0);
    let mut blen_sq = F::new(0.0);
    for c in 0..3 {
        tlen_sq += tangent[c] * tangent[c];
        blen_sq += bitangent[c] * bitangent[c];
    }

    let t_inv = F::new(1.0) / (F::sqrt(tlen_sq) + eps);
    let b_inv = F::new(1.0) / (F::sqrt(blen_sq) + eps);

    // Write normalized results
    for c in 0..3 {
        face_tangents[fid * 3 + c] = tangent[c] * t_inv;
        face_bitangents[fid * 3 + c] = bitangent[c] * b_inv;
    }
}

#[cube(launch)]
#[allow(clippy::similar_names)]
pub fn vertex_tangents_kernel<F: Float>(
    face_tangents: &Tensor<F>,       // flattened [num_faces, 3]
    face_bitangents: &Tensor<F>,     // flattened [num_faces, 3]
    row_ptr: &Tensor<i32>,           // [num_vertices + 1]
    col_idx: &Tensor<i32>,           // [total_incidents] (face indices)
    normals: &Tensor<F>,             // flattened [num_vertices, 3]
    vertex_tangents: &mut Tensor<F>, // flattened [num_vertices, 4] (x,y,z,handness)
) {
    let v = ABSOLUTE_POS_X;

    let num_vertices = vertex_tangents.shape(0);
    if v >= num_vertices {
        terminate!();
    }

    // read CSR range
    let start_i: i32 = row_ptr[v];
    let end_i: i32 = row_ptr[v + 1];

    // accumulator for tangent vector accumulated over all incident faces
    let mut ax: F = F::new(0.0);
    let mut ay: F = F::new(0.0);
    let mut az: F = F::new(0.0);

    //accumulators for bitangent vector accumulated over all incident faces
    let mut bx: F = F::new(0.0);
    let mut by: F = F::new(0.0);
    let mut bz: F = F::new(0.0);

    // accumulate tangents/bitangents touching vertex v
    let mut i = start_i;
    while i < end_i {
        #[allow(clippy::cast_sign_loss)]
        let face_idx = col_idx[i as u32] as u32;
        let base = face_idx * 3;
        ax += face_tangents[base];
        ay += face_tangents[base + 1];
        az += face_tangents[base + 2];

        bx += face_bitangents[base];
        by += face_bitangents[base + 1];
        bz += face_bitangents[base + 2];

        i += 1;
    }

    // If accumulated tangent is zero-length, write zeros
    let eps = F::new(1e-6);
    let tlen2 = ax * ax + ay * ay + az * az;
    if tlen2 <= eps * eps {
        vertex_tangents[v * 4] = F::new(0.0);
        vertex_tangents[v * 4 + 1] = F::new(0.0);
        vertex_tangents[v * 4 + 2] = F::new(0.0);
        vertex_tangents[v * 4 + 3] = F::new(0.0);
        terminate!();
    }

    // Gram-Schmidt: make tangent orthogonal to normal
    let nx = normals[v * 3];
    let ny = normals[v * 3 + 1];
    let nz = normals[v * 3 + 2];

    // dot = normal . tangent
    let dot = nx * ax + ny * ay + nz * az;

    // t = t - n * dot
    let tx = ax - nx * dot;
    let ty = ay - ny * dot;
    let tz = az - nz * dot;

    // normalize t
    let tlen = F::sqrt(tx * tx + ty * ty + tz * tz);
    let invt = F::new(1.0) / (tlen + eps);
    let ntx = tx * invt;
    let nty = ty * invt;
    let ntz = tz * invt;

    // compute handedness: sign( (tangent cross bitangent) dot normal )
    // cross = tangent x bitangent
    // unclear weather we do the cross with original accumulations or with normalized tangent and accumulated bitangent
    // for now I just do it with the accumulations ( so unnormalized tangent and unnormalized bitangent )
    let c_x = ay * bz - az * by;
    let c_y = az * bx - ax * bz;
    let c_z = ax * by - ay * bx;

    // The following is with the normalized tangent instead of the accumulated tangent
    // let c_x = nty * bz - ntz * by;
    // let c_y = ntz * bx - ntx * bz;
    // let c_z = ntx * by - nty * bx;

    let handed_dot = c_x * nx + c_y * ny + c_z * nz;

    // handedness sign: +1 or -1
    let hand = if handed_dot >= F::new(0.0) { F::new(1.0) } else { F::new(-1.0) };

    // write tangent (x,y,z) and handness as w
    vertex_tangents[v * 4] = ntx;
    vertex_tangents[v * 4 + 1] = nty;
    vertex_tangents[v * 4 + 2] = ntz;
    vertex_tangents[v * 4 + 3] = hand;
}

//launchers
#[allow(clippy::type_complexity)]
pub fn face_tangents_launch<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    verts: FloatTensor<CubeBackend<R, F, I, BT>>, // [num_verts, 3]
    uvs: FloatTensor<CubeBackend<R, F, I, BT>>,   // [num_verts, 2]
    faces: IntTensor<CubeBackend<R, F, I, BT>>,   // [num_faces, 3]
) -> (
    FloatTensor<CubeBackend<R, F, I, BT>>, // face_tangents [num_faces, 3]
    FloatTensor<CubeBackend<R, F, I, BT>>, // face_bitangents [num_faces, 3]
) {
    verts.assert_is_on_same_device(&uvs);
    verts.assert_is_on_same_device(&faces);

    let num_faces = faces.shape().dims::<2>()[0];

    // Allocate buffers for tangents and bitangents
    let shape_out = Shape::from(vec![num_faces, 3usize]);
    let bytes = shape_out.num_elements() * core::mem::size_of::<F>();
    let buffer_tangent = verts.client.empty(bytes);
    let buffer_bitangent = verts.client.empty(bytes);

    // Create CubeTensors
    let face_tangents = CubeTensor::new_contiguous(verts.client.clone(), verts.device.clone(), shape_out.clone(), buffer_tangent, F::dtype());
    let face_bitangents = CubeTensor::new_contiguous(verts.client.clone(), verts.device.clone(), shape_out, buffer_bitangent, F::dtype());

    // Workgroup/cube config
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    #[allow(clippy::cast_possible_truncation)]
    let cubes_needed_in_x = num_faces.div_ceil(cube_dim.x as usize) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, 1, 1);

    // Launch kernel
    face_tangents_kernel::launch::<F, R>(
        &verts.client,
        cube_count,
        cube_dim,
        verts.as_tensor_arg::<F>(1),
        uvs.as_tensor_arg::<F>(1),
        faces.as_tensor_arg::<F>(1),
        face_tangents.as_tensor_arg::<F>(1),
        face_bitangents.as_tensor_arg::<F>(1),
    );

    (face_tangents, face_bitangents)
}

pub fn vertex_tangents_launch<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    face_tangents: FloatTensor<CubeBackend<R, F, I, BT>>,   // [num_faces, 3]
    face_bitangents: FloatTensor<CubeBackend<R, F, I, BT>>, // [num_faces, 3]
    row_ptr: IntTensor<CubeBackend<R, F, I, BT>>,           // [num_vertices + 1]
    col_idx: IntTensor<CubeBackend<R, F, I, BT>>,           // [total_incidents]
    normals: FloatTensor<CubeBackend<R, F, I, BT>>,         // [num_vertices, 3]
    num_vertices: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    // Device checks
    face_tangents.assert_is_on_same_device(&face_bitangents);
    face_tangents.assert_is_on_same_device(&row_ptr);
    face_tangents.assert_is_on_same_device(&col_idx);
    face_tangents.assert_is_on_same_device(&normals);

    // Output: [num_vertices, 4] (tangent vec3 + handedness scalar)
    let shape_out = Shape::from(vec![num_vertices, 4usize]);
    let bytes = shape_out.num_elements() * core::mem::size_of::<F>();
    let buffer = face_tangents.client.empty(bytes);

    // wrap the buffer Handle into CubeTensor primitive for output.
    let output = CubeTensor::new_contiguous(face_tangents.client.clone(), face_tangents.device.clone(), shape_out, buffer, F::dtype());

    // Workgroup config: one thread per vertex
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    #[allow(clippy::cast_possible_truncation)]
    let cubes_needed_in_x = num_vertices.div_ceil(cube_dim.x as usize) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, 1, 1);

    // Launch kernel
    vertex_tangents_kernel::launch::<F, R>(
        &face_tangents.client,
        cube_count,
        cube_dim,
        face_tangents.as_tensor_arg::<F>(1),
        face_bitangents.as_tensor_arg::<F>(1),
        row_ptr.as_tensor_arg::<I>(1),
        col_idx.as_tensor_arg::<I>(1),
        normals.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
    );

    output
}
