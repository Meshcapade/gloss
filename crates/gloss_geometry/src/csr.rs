extern crate nalgebra as na;
use burn::prelude::Backend;
use burn::tensor::{Device, Int, Tensor};

/// Compressed-Sparse-Row (CSR) mapping from vertex -> incident faces.
///
/// This struct is intended to be used by a GPU "vertex kernel" which,
/// for each vertex `v`, will read `row_ptr[v]..row_ptr[v+1]` from `col_idx`
/// to find all faces that reference `v`, then sum those face normals
/// and normalize to produce a per-vertex normal.
///
/// Semantics / invariants:
/// - `num_vertices == row_ptr.len() - 1`.
/// - `col_idx.len() == row_ptr[num_vertices]` (total incidence count, usually 3*F).
/// - Every value `col_idx[i]` is a face id in range `0 .. num_faces`.
/// - The order of face indices within a vertex is determined by input face order.
///
/// Example read pattern for vertex `v`:
/// ```text
/// let start = csr.row_ptr[v] as usize;
/// let end   = csr.row_ptr[v + 1] as usize;
/// for i in start..end {
///     let fid = csr.col_idx[i] as usize; // fid in 0..num_faces
///     // use face_normals[fid]
/// }
/// ```
#[derive(Clone, Debug)]
pub struct VertexFaceCSR {
    /// `row_ptr`: length `num_vertices` + 1
    /// `row_ptr`[v]..`row_ptr`[v+1] (half-open) index into `col_idx`.
    pub row_ptr: Vec<u32>,

    /// `col_idx`: flattened list of face indices (u32), grouped by vertex.
    /// Range of values: 0..num_faces-1.
    pub col_idx: Vec<u32>,

    /// convenience
    pub num_vertices: usize,
    pub num_faces: usize,
}

/// Burn-friendly CSR representation: buffers are Tensors on a chosen backend/device.
///
/// These can be uploaded to GPU and used inside `CubeCL` kernels.
#[derive(Clone, Debug)]
pub struct VertexFaceCSRBurn<B: Backend> {
    /// Tensor of length = `num_vertices` + 1
    /// `row_ptr`[v]..`row_ptr`[v+1] indexes into `col_idx` for vertex v
    pub row_ptr: Tensor<B, 1, Int>,
    /// Tensor of length = 3 * `num_faces`
    /// Flat list of incident face IDs
    pub col_idx: Tensor<B, 1, Int>,
    /// Mesh metadata
    pub num_vertices: usize,
    pub num_faces: usize,
}

impl VertexFaceCSR {
    /// Build a CSR mapping from a (F x 3) faces matrix.
    ///
    /// # Arguments
    /// * `faces` - A `DMatrix<u32>` with shape `(F, 3)`.
    ///
    /// Each row is a triangle `(i0, i1, i2)` with vertex indices.
    pub fn from_faces(faces: &na::DMatrix<u32>) -> Self {
        assert_eq!(faces.ncols(), 3, "Faces matrix must have exactly 3 columns (triangle vertex indices)");

        let num_faces = faces.nrows();

        // Find maximum vertex index -> num_vertices
        let max_idx = faces.iter().copied().max().unwrap_or(0);
        let num_vertices = (max_idx as usize) + 1;

        // 1) Count degrees (number of incident faces) per vertex
        let mut degree = vec![0usize; num_vertices];
        for idx in faces.iter() {
            degree[*idx as usize] += 1;
        }

        // 2) Build row_ptr (prefix sum of degrees)
        let mut row_ptr = Vec::with_capacity(num_vertices + 1);
        row_ptr.push(0);
        #[allow(clippy::cast_possible_truncation)]
        for &d in &degree {
            let last = *row_ptr.last().unwrap();
            row_ptr.push(last + d as u32);
        }

        // 3) Allocate col_idx and temporary cursors
        let total_incidents = *row_ptr.last().unwrap() as usize;
        let mut col_idx = vec![0u32; total_incidents];
        let mut cursor = vec![0usize; num_vertices];

        // 4) Fill col_idx: for each face, push its ID into each vertex’s bucket
        #[allow(clippy::cast_possible_truncation)]
        for fid in 0..num_faces {
            let row = faces.row(fid);
            for j in 0..3 {
                let v = row[j] as usize;
                let base = row_ptr[v] as usize;
                let pos = base + cursor[v];
                col_idx[pos] = fid as u32;
                cursor[v] += 1;
            }
        }

        VertexFaceCSR {
            row_ptr,
            col_idx,
            num_vertices,
            num_faces,
        }
    }

    /// Get the slice of incident face IDs for vertex v.
    pub fn incident_faces(&self, v: usize) -> &[u32] {
        assert!(v < self.num_vertices);
        let start = self.row_ptr[v] as usize;
        let end = self.row_ptr[v + 1] as usize;
        &self.col_idx[start..end]
    }

    /// Convert CPU CSR to Burn tensor version.
    ///
    /// # Arguments
    /// * `device` - The device where tensors should be allocated (CPU, CUDA, WGPU…).
    #[allow(clippy::cast_possible_wrap)]
    pub fn to_burn<B: Backend>(&self, device: &Device<B>) -> VertexFaceCSRBurn<B> {
        // Convert Vec<u32> to Vec<i32> because Burn's `Int` usually maps to i32
        let row_ptr_i32: Vec<i32> = self.row_ptr.iter().map(|&x| x as i32).collect();
        let col_idx_i32: Vec<i32> = self.col_idx.iter().map(|&x| x as i32).collect();

        let row_ptr_tensor: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::from_ints(row_ptr_i32.as_slice(), &device.clone());
        let col_idx_tensor: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::from_ints(col_idx_i32.as_slice(), &device.clone());

        VertexFaceCSRBurn {
            row_ptr: row_ptr_tensor,
            col_idx: col_idx_tensor,
            num_vertices: self.num_vertices,
            num_faces: self.num_faces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn csr_small_example() {
        // faces: F=3
        // face 0: (0,1,2)
        // face 1: (0,2,3)
        // face 2: (2,4,5)
        let data: Vec<u32> = vec![0, 1, 2, 0, 2, 3, 2, 4, 5];
        let faces = DMatrix::from_row_slice(3, 3, &data);
        let csr = VertexFaceCSR::from_faces(&faces);

        assert_eq!(csr.num_faces, 3);
        assert_eq!(csr.num_vertices, 6);

        assert_eq!(csr.incident_faces(0), &[0, 1]);
        assert_eq!(csr.incident_faces(1), &[0]);
        assert_eq!(csr.incident_faces(2), &[0, 1, 2]);
        assert_eq!(csr.incident_faces(3), &[1]);
        assert_eq!(csr.incident_faces(4), &[2]);
        assert_eq!(csr.incident_faces(5), &[2]);
    }
}
