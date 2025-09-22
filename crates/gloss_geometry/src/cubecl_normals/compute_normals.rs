use burn::tensor::TensorMetadata;
use burn::tensor::{
    ops::{FloatTensor, IntTensor},
    Shape,
};
use burn_cubecl::{tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use cubecl::prelude::*; // CubeCL macros and helpers

// ----------------- CubeCL kernels -----------------

#[cube(launch)]
pub fn face_normals_kernel<F: Float>(verts: &Tensor<F>, faces: &Tensor<i32>, face_normals: &mut Tensor<F>) {
    // Each thread handles one face
    let f = ABSOLUTE_POS_X; // X dimension corresponds to face index

    let num_faces = face_normals.shape(0); // assume [num_faces, 3]
    if f >= num_faces {
        terminate!();
    }

    // Get the vertex indices for this face
    let mut v_indices = Line::<i32>::empty(3u32);
    for i in 0..3 {
        let face_idx = f * 3 + i;
        v_indices[i] = faces[face_idx] as i32; // faces[f, i]
    }

    // Load vertices
    let mut v0 = Line::<F>::empty(3u32);
    let mut v1 = Line::<F>::empty(3u32);
    let mut v2 = Line::<F>::empty(3u32);

    for c in 0..3 {
        v0[c] = verts[v_indices[0] as u32 * 3 + c];
        v1[c] = verts[v_indices[1] as u32 * 3 + c];
        v2[c] = verts[v_indices[2] as u32 * 3 + c];
    }

    // Compute edges
    let mut d1 = Line::<F>::empty(3u32);
    let mut d2 = Line::<F>::empty(3u32);
    for c in 0..3 {
        d1[c] = v1[c] - v0[c];
        d2[c] = v2[c] - v0[c];
    }

    // Cross product: normal = d1 x d2
    let cx: F = d1[1] * d2[2] - d1[2] * d2[1];
    let cy: F = d1[2] * d2[0] - d1[0] * d2[2];
    let cz: F = d1[0] * d2[1] - d1[1] * d2[0];

    // Normalize and write result
    let len = F::sqrt(cx * cx + cy * cy + cz * cz);
    let eps = F::new(1e-6);
    let inv = F::new(1.0) / (len + eps);

    face_normals[f * 3] = cx * inv;
    face_normals[f * 3 + 1] = cy * inv;
    face_normals[f * 3 + 2] = cz * inv;
}

#[cube(launch)]
pub fn vertex_normals_kernel<F: Float>(
    face_normals: &Tensor<F>,       // flattened [num_faces, 3]
    row_ptr: &Tensor<i32>,          // [num_vertices + 1]
    col_idx: &Tensor<i32>,          // [total_incidents]  (face indices)
    vertex_normals: &mut Tensor<F>, // flattened [num_vertices, 3]
) {
    // one thread per vertex
    let v = ABSOLUTE_POS_X; // vertex index

    let num_vertices = vertex_normals.shape(0);
    if v >= num_vertices {
        terminate!();
    }

    // read CSR range
    let start_i: i32 = row_ptr[v];
    let end_i: i32 = row_ptr[v + 1];

    // Accumulators for the normals components from all the incident faces
    let mut ax: F = F::new(0.0);
    let mut ay: F = F::new(0.0);
    let mut az: F = F::new(0.0);

    // If start_i == end_i this vertex has no incident faces -> leave zeros
    let mut i = start_i;
    #[allow(clippy::cast_sign_loss)]
    while i < end_i {
        // get face index (i32), convert to usize for indexing
        let face_idx_i: i32 = col_idx[i as u32];
        let face_idx = face_idx_i as u32;
        let base = face_idx * 3;

        // accumulate face normal components
        ax += face_normals[base];
        ay += face_normals[base + 1];
        az += face_normals[base + 2];

        i += 1;
    }

    // normalize accumulated normal (with tiny epsilon)
    let len = F::sqrt(ax * ax + ay * ay + az * az);
    let eps = F::new(1e-6);
    let inv = F::new(1.0) / (len + eps);

    vertex_normals[v * 3] = ax * inv;
    vertex_normals[v * 3 + 1] = ay * inv;
    vertex_normals[v * 3 + 2] = az * inv;
}

//launchers
pub fn face_normals_launch<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    verts: FloatTensor<CubeBackend<R, F, I, BT>>,
    faces: IntTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    verts.assert_is_on_same_device(&faces);

    let num_faces = faces.shape().dims::<2>()[0];

    // Build output primitive: shape [num_faces, 3]
    let shape_out = Shape::from(vec![num_faces, 3usize]);
    let bytes = shape_out.num_elements() * core::mem::size_of::<F>();
    let buffer = verts.client.empty(bytes);

    // wrap the buffer Handle into CubeTensor primitive for output.
    let output = CubeTensor::new_contiguous(verts.client.clone(), verts.device.clone(), shape_out, buffer, F::dtype());

    // Choose cube/workgroup sizes (tune as needed)
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 }; // e.g. one face per x-thread or tune accordingly
    #[allow(clippy::cast_possible_truncation)]
    let cubes_needed_in_x = num_faces.div_ceil(cube_dim.x as usize) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, 1, 1);

    // Launch the kernel
    face_normals_kernel::launch::<F, R>(
        &verts.client,
        cube_count,
        cube_dim,
        verts.as_tensor_arg::<F>(1),
        faces.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
    );

    output
}

pub fn vertex_normals_launch<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    face_normals: FloatTensor<CubeBackend<R, F, I, BT>>, // [num_faces, 3]
    row_ptr: IntTensor<CubeBackend<R, F, I, BT>>,        // [num_vertices + 1]
    col_idx: IntTensor<CubeBackend<R, F, I, BT>>,        // [total_incidents]
    num_vertices: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    face_normals.assert_is_on_same_device(&row_ptr);
    face_normals.assert_is_on_same_device(&col_idx);

    // Build output primitive: shape [num_vertices, 3]
    let shape_out = Shape::from(vec![num_vertices, 3usize]);
    let bytes = shape_out.num_elements() * core::mem::size_of::<F>();
    let buffer = face_normals.client.empty(bytes);

    // wrap the buffer Handle into CubeTensor primitive for output.
    let output = CubeTensor::new_contiguous(face_normals.client.clone(), face_normals.device.clone(), shape_out, buffer, F::dtype());

    // Each thread handles one vertex
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    #[allow(clippy::cast_possible_truncation)]
    let cubes_needed_in_x = num_vertices.div_ceil(cube_dim.x as usize) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, 1, 1);

    // Launch vertex_normals_kernel
    vertex_normals_kernel::launch::<F, R>(
        &face_normals.client,
        cube_count,
        cube_dim,
        face_normals.as_tensor_arg::<F>(1),
        row_ptr.as_tensor_arg::<I>(1),
        col_idx.as_tensor_arg::<I>(1),
        output.as_tensor_arg::<F>(1),
    );

    output
}
