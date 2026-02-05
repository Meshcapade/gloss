//! Serializable versions of CPU components for network transmission.
//!
//! These types mirror the CPU components but implement `Serialize` and `Deserialize`
//! for network transmission. They can be converted to/from the original component types.

use nalgebra::DMatrix;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};

use crate::components::{Colors, Edges, Faces, Normals, UVs, Verts, VisLines, VisMesh, VisPoints};

/// Serializable version of Verts component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableVerts {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl ToSerializable<SerializableVerts> for Verts {
    fn to_serializable(&self) -> SerializableVerts {
        SerializableVerts {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableVerts> for Verts {
    fn from_serializable(s: &SerializableVerts) -> Verts {
        Verts(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of Faces component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableFaces {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u32>,
}
impl ToSerializable<SerializableFaces> for Faces {
    fn to_serializable(&self) -> SerializableFaces {
        SerializableFaces {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableFaces> for Faces {
    fn from_serializable(s: &SerializableFaces) -> Faces {
        Faces(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of Normals component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableNormals {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl ToSerializable<SerializableNormals> for Normals {
    fn to_serializable(&self) -> SerializableNormals {
        SerializableNormals {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableNormals> for Normals {
    fn from_serializable(s: &SerializableNormals) -> Normals {
        Normals(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of Colors component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableColors {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl ToSerializable<SerializableColors> for Colors {
    fn to_serializable(&self) -> SerializableColors {
        SerializableColors {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableColors> for Colors {
    fn from_serializable(s: &SerializableColors) -> Colors {
        Colors(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of UVs component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableUVs {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl ToSerializable<SerializableUVs> for UVs {
    fn to_serializable(&self) -> SerializableUVs {
        SerializableUVs {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableUVs> for UVs {
    fn from_serializable(s: &SerializableUVs) -> UVs {
        UVs(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of Edges component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableEdges {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u32>,
}
impl ToSerializable<SerializableEdges> for Edges {
    fn to_serializable(&self) -> SerializableEdges {
        SerializableEdges {
            rows: self.0.nrows(),
            cols: self.0.ncols(),
            data: self.0.as_slice().to_vec(),
        }
    }
}
impl FromSerializable<SerializableEdges> for Edges {
    fn from_serializable(s: &SerializableEdges) -> Edges {
        Edges(DMatrix::from_vec(s.rows, s.cols, s.data.clone()))
    }
}

/// Serializable version of `VisMesh` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableVisMesh {
    pub show_mesh: bool,
    pub solid_color: [f32; 4],
    pub metalness: f32,
    pub perceptual_roughness: f32,
    pub roughness_black_lvl: f32,
    pub uv_scale: f32,
    pub opacity: f32,
    pub needs_sss: bool,
    pub color_type: u8, // MeshColorType as u8
    pub added_automatically: bool,
}

impl ToSerializable<SerializableVisMesh> for VisMesh {
    fn to_serializable(&self) -> SerializableVisMesh {
        SerializableVisMesh {
            show_mesh: self.show_mesh,
            solid_color: self.solid_color.into(),
            metalness: self.metalness,
            perceptual_roughness: self.perceptual_roughness,
            roughness_black_lvl: self.roughness_black_lvl,
            uv_scale: self.uv_scale,
            opacity: self.opacity,
            needs_sss: self.needs_sss,
            color_type: self.color_type as u8,
            added_automatically: self.added_automatically,
        }
    }
}

impl FromSerializable<SerializableVisMesh> for VisMesh {
    fn from_serializable(s: &SerializableVisMesh) -> VisMesh {
        VisMesh {
            show_mesh: s.show_mesh,
            solid_color: s.solid_color.into(),
            metalness: s.metalness,
            perceptual_roughness: s.perceptual_roughness,
            roughness_black_lvl: s.roughness_black_lvl,
            uv_scale: s.uv_scale,
            opacity: s.opacity,
            needs_sss: s.needs_sss,
            color_type: FromPrimitive::from_u8(s.color_type).unwrap(),
            added_automatically: s.added_automatically,
        }
    }
}

/// Serializable version of `VisPoints` component
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableVisPoints {
    pub show_points: bool,
    pub show_points_indices: bool,
    pub point_color: [f32; 4],
    pub point_size: f32,
    pub is_point_size_in_world_space: bool,
    pub color_type: u8, // PointColorType as u8
    pub zbuffer: bool,
    pub added_automatically: bool,
}

impl ToSerializable<SerializableVisPoints> for VisPoints {
    fn to_serializable(&self) -> SerializableVisPoints {
        SerializableVisPoints {
            show_points: self.show_points,
            show_points_indices: self.show_points_indices,
            point_color: self.point_color.into(),
            point_size: self.point_size,
            is_point_size_in_world_space: self.is_point_size_in_world_space,
            color_type: self.color_type as u8,
            zbuffer: self.zbuffer,
            added_automatically: self.added_automatically,
        }
    }
}

impl FromSerializable<SerializableVisPoints> for VisPoints {
    fn from_serializable(s: &SerializableVisPoints) -> VisPoints {
        VisPoints {
            show_points: s.show_points,
            show_points_indices: s.show_points_indices,
            point_color: s.point_color.into(),
            point_size: s.point_size,
            is_point_size_in_world_space: s.is_point_size_in_world_space,
            color_type: FromPrimitive::from_u8(s.color_type).unwrap(),
            zbuffer: s.zbuffer,
            added_automatically: s.added_automatically,
        }
    }
}

/// Serializable version of `VisLines` component
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableVisLines {
    pub show_lines: bool,
    pub line_color: [f32; 4],
    pub line_width: f32,
    pub color_type: u8, // LineColorType as u8
    pub zbuffer: bool,
    pub antialias_edges: bool,
    pub added_automatically: bool,
}

impl ToSerializable<SerializableVisLines> for VisLines {
    fn to_serializable(&self) -> SerializableVisLines {
        SerializableVisLines {
            show_lines: self.show_lines,
            line_color: self.line_color.into(),
            line_width: self.line_width,
            color_type: self.color_type as u8,
            zbuffer: self.zbuffer,
            antialias_edges: self.antialias_edges,
            added_automatically: self.added_automatically,
        }
    }
}

impl FromSerializable<SerializableVisLines> for VisLines {
    fn from_serializable(s: &SerializableVisLines) -> VisLines {
        VisLines {
            show_lines: s.show_lines,
            line_color: s.line_color.into(),
            line_width: s.line_width,
            color_type: FromPrimitive::from_u8(s.color_type).unwrap(),
            zbuffer: s.zbuffer,
            antialias_edges: s.antialias_edges,
            added_automatically: s.added_automatically,
        }
    }
}

/// Trait for converting between original and serializable types
pub trait ToSerializable<S> {
    fn to_serializable(&self) -> S;
}

/// Trait for converting from serializable back to original type
pub trait FromSerializable<S> {
    fn from_serializable(s: &S) -> Self;
}

crate::impl_network_sendable_and_receivable!(
    //cpu components
    SerializableVerts,
    SerializableFaces,
    SerializableUVs,
    SerializableNormals,
    SerializableColors,
    SerializableEdges,
    //vis components
    SerializableVisLines,
    SerializableVisPoints,
    SerializableVisMesh
);
