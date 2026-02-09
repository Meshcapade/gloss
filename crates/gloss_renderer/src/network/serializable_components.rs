//! Serializable versions of CPU components for network transmission.
//!
//! These types mirror the CPU components but implement `Serialize` and `Deserialize`
//! for network transmission. They can be converted to/from the original component types.

use nalgebra::DMatrix;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};

use crate::components::{
    Colors, DiffuseImg, Edges, Faces, GenericImg, ImgConfig, Normals, ProjectionWithFov, UVs, Verts, VisLines, VisMesh, VisPoints,
};

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

/// Serializable version of `VisLines` component
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableProjectionWithFov {
    pub aspect_ratio: f32,
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
}

impl ToSerializable<SerializableProjectionWithFov> for ProjectionWithFov {
    fn to_serializable(&self) -> SerializableProjectionWithFov {
        SerializableProjectionWithFov {
            aspect_ratio: self.aspect_ratio,
            fovy: self.fovy,
            near: self.near,
            far: self.far,
        }
    }
}

impl FromSerializable<SerializableProjectionWithFov> for ProjectionWithFov {
    fn from_serializable(s: &SerializableProjectionWithFov) -> ProjectionWithFov {
        ProjectionWithFov {
            aspect_ratio: s.aspect_ratio,
            fovy: s.fovy,
            near: s.near,
            far: s.far,
        }
    }
}

/// Serializable version of `ImgConfig` component
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableImgConfig {
    pub keep_on_cpu: bool,
    pub fast_upload: bool,
    pub generate_mipmaps: bool,
    pub mipmap_generation_cpu: bool,
}

/// Serializable version of `GenericImg` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableGenericImg {
    pub path: Option<String>,
    // Image data serialized as raw bytes with metadata
    pub image_data: Option<SerializableImageData>,
    pub config: SerializableImgConfig,
}

/// Serializable image data with format information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableImageData {
    pub width: u32,
    pub height: u32,
    pub channels: u8, // 1=Luma, 2=LumaA, 3=RGB, 4=RGBA
    pub data: Vec<u8>,
}

/// Serializable version of `DiffuseImg` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableDiffuseImg {
    pub generic_img: SerializableGenericImg,
}

impl ToSerializable<SerializableDiffuseImg> for DiffuseImg {
    fn to_serializable(&self) -> SerializableDiffuseImg {
        SerializableDiffuseImg {
            generic_img: self.generic_img.to_serializable(),
        }
    }
}

impl FromSerializable<SerializableDiffuseImg> for DiffuseImg {
    fn from_serializable(s: &SerializableDiffuseImg) -> DiffuseImg {
        DiffuseImg {
            generic_img: GenericImg::from_serializable(&s.generic_img),
        }
    }
}

impl ToSerializable<SerializableGenericImg> for GenericImg {
    fn to_serializable(&self) -> SerializableGenericImg {
        let image_data = if let Some(ref img) = self.cpu_img {
            use gloss_img::DynImage;

            // let (width, height) = img.dimensions();
            let (width, height) = (img.width(), img.height());
            let (channels, data) = match img {
                DynImage::ImageLuma8(img) => (1, img.as_raw().clone()),
                DynImage::ImageLumaA8(img) => (2, img.as_raw().clone()),
                DynImage::ImageRgb8(img) => (3, img.as_raw().clone()),
                DynImage::ImageRgba8(img) => (4, img.as_raw().clone()),
                // Convert other formats to RGBA for simplicity
                _ => {
                    let rgba = img.to_rgba8();
                    (4, rgba.as_raw().clone())
                }
            };

            Some(SerializableImageData {
                width,
                height,
                channels,
                data,
            })
        } else {
            None
        };

        SerializableGenericImg {
            path: self.path.clone(),
            image_data,
            config: self.config.to_serializable(),
        }
    }
}

impl FromSerializable<SerializableGenericImg> for GenericImg {
    fn from_serializable(s: &SerializableGenericImg) -> GenericImg {
        let cpu_img = if let Some(ref img_data) = s.image_data {
            use image::{GrayAlphaImage, GrayImage, RgbImage, RgbaImage};

            let dynamic_img = match img_data.channels {
                1 => {
                    let gray_img = GrayImage::from_raw(img_data.width, img_data.height, img_data.data.clone())
                        .expect("Failed to create grayscale image from serialized data");
                    image::DynamicImage::ImageLuma8(gray_img)
                }
                2 => {
                    let gray_alpha_img = GrayAlphaImage::from_raw(img_data.width, img_data.height, img_data.data.clone())
                        .expect("Failed to create grayscale+alpha image from serialized data");
                    image::DynamicImage::ImageLumaA8(gray_alpha_img)
                }
                3 => {
                    let rgb_img = RgbImage::from_raw(img_data.width, img_data.height, img_data.data.clone())
                        .expect("Failed to create RGB image from serialized data");
                    image::DynamicImage::ImageRgb8(rgb_img)
                }
                4 => {
                    let rgba_img = RgbaImage::from_raw(img_data.width, img_data.height, img_data.data.clone())
                        .expect("Failed to create RGBA image from serialized data");
                    image::DynamicImage::ImageRgba8(rgba_img)
                }
                _ => panic!(
                    "Unsupported number of channels: {}. Supported: 1 (Luma), 2 (LumaA), 3 (RGB), 4 (RGBA)",
                    img_data.channels
                ),
            };

            Some(
                dynamic_img
                    .try_into()
                    .expect("Failed to convert image::DynamicImage to gloss_img::DynImage"),
            )
        } else {
            None
        };

        GenericImg {
            path: s.path.clone(),
            cpu_img,
            config: ImgConfig::from_serializable(&s.config),
        }
    }
}

impl ToSerializable<SerializableImgConfig> for ImgConfig {
    fn to_serializable(&self) -> SerializableImgConfig {
        SerializableImgConfig {
            keep_on_cpu: self.keep_on_cpu,
            fast_upload: self.fast_upload,
            generate_mipmaps: self.generate_mipmaps,
            mipmap_generation_cpu: self.mipmap_generation_cpu,
        }
    }
}

impl FromSerializable<SerializableImgConfig> for ImgConfig {
    fn from_serializable(s: &SerializableImgConfig) -> ImgConfig {
        ImgConfig {
            keep_on_cpu: s.keep_on_cpu,
            fast_upload: s.fast_upload,
            generate_mipmaps: s.generate_mipmaps,
            mipmap_generation_cpu: s.mipmap_generation_cpu,
        }
    }
}

//TRAITS/////

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
    SerializableVisMesh,
    //cam components
    SerializableProjectionWithFov,
    //imgs
    SerializableDiffuseImg,
    SerializableGenericImg,
    SerializableImgConfig,
    SerializableImageData,
);
