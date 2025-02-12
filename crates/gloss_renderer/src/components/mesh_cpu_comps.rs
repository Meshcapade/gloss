//! These represent CPU components which you usually add to entities when you
//! spawn them.

extern crate nalgebra as na;
// use burn::backend::ndarray::NdArrayDevice;
// use burn::backend::candle::CandleDevice;
// use burn::backend::Candle;
use gloss_img::DynImage;
use image::ImageReader;
use na::DMatrix;
use std::io::{BufReader, Cursor, Read, Seek};
use utils_rs::{
    io::FileLoader,
    tensor::{DynamicTensorFloat2D, DynamicTensorInt2D},
};
// use burn::backend::{Candle, NdArray, Wgpu};

/// Component that modifications to the config
#[derive(Clone)]
pub struct ConfigChanges {
    pub new_distance_fade_center: na::Point3<f32>,
}

/// Defines the color type an entity which is displayed as a point cloud
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointColorType {
    Solid = 0,
    PerVert,
}

/// Defines the color type an entity which is displayed as a point cloud
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineColorType {
    Solid = 0,
    PerVert,
}

/// Defines the color type an entity which is displayed as a mesh
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeshColorType {
    Solid = 0,
    PerVert,
    Texture,
    UV,
    Normal,
    NormalViewCoords,
}

/// Component for visualization options of lines
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct VisLines {
    pub show_lines: bool,
    pub line_color: na::Vector4<f32>,
    pub line_width: f32,
    pub color_type: LineColorType,
    pub zbuffer: bool,
    pub antialias_edges: bool,
    //if this components was added automatically by the renderer this will be set to true. This is useful to know since when we upload textures to
    // gpu the vis_mesh.color_type will be set to Texture but this should happen ONLY if the VisMesh was added automatically. If the used adds this
    // component manually then we shouldn't override his VisMesh.colortype.
    pub added_automatically: bool,
}
/// Component for visualization options of wireframe
#[derive(Clone)]
pub struct VisWireframe {
    pub show_wireframe: bool,
    pub wire_color: na::Vector4<f32>,
    pub wire_width: f32,
    //if this components was added automatically by the renderer this will be set to true. This is useful to know since when we upload textures to
    // gpu the vis_mesh.color_type will be set to Texture but this should happen ONLY if the VisMesh was added automatically. If the used adds this
    // component manually then we shouldn't override his VisMesh.colortype.
    pub added_automatically: bool,
}
/// Component for visualization options of normals
#[derive(Clone)]
pub struct VisNormals {
    pub show_normals: bool,
    pub normals_color: na::Vector4<f32>,
    pub normals_width: f32,
    pub normals_scale: f32, //the scale of the arrows for the normal.
    //if this components was added automatically by the renderer this will be set to true. This is useful to know since when we upload textures to
    // gpu the vis_mesh.color_type will be set to Texture but this should happen ONLY if the VisMesh was added automatically. If the used adds this
    // component manually then we shouldn't override his VisMesh.colortype.
    pub added_automatically: bool,
}
/// Component for visualization options of point clouds
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct VisPoints {
    pub show_points: bool,
    pub show_points_indices: bool,
    pub point_color: na::Vector4<f32>,
    pub point_size: f32,
    pub is_point_size_in_world_space: bool,
    pub color_type: PointColorType,
    pub zbuffer: bool,
    //if this components was added automatically by the renderer this will be set to true. This is useful to know since when we upload textures to
    // gpu the vis_mesh.color_type will be set to Texture but this should happen ONLY if the VisMesh was added automatically. If the used adds this
    // component manually then we shouldn't override his VisMesh.colortype.
    pub added_automatically: bool,
}
/// Component for visualization options of meshes
#[derive(Clone)]
pub struct VisMesh {
    pub show_mesh: bool,
    pub solid_color: na::Vector4<f32>,
    pub metalness: f32,
    pub perceptual_roughness: f32,
    pub roughness_black_lvl: f32,
    pub uv_scale: f32,
    pub opacity: f32,
    pub needs_sss: bool,
    pub color_type: MeshColorType,
    //if this components was added automatically by the renderer this will be set to true. This is useful to know since when we upload textures to
    // gpu the vis_mesh.color_type will be set to Texture but this should happen ONLY if the VisMesh was added automatically. If the used adds this
    // component manually then we shouldn't override his VisMesh.colortype.
    pub added_automatically: bool,
}
//implementations of default vis options
impl Default for VisLines {
    fn default() -> VisLines {
        VisLines {
            show_lines: false,
            line_color: na::Vector4::<f32>::new(1.0, 0.1, 0.1, 1.0),
            line_width: 1.0,
            color_type: LineColorType::Solid,
            zbuffer: true,
            antialias_edges: false,
            added_automatically: false,
        }
    }
}
impl Default for VisWireframe {
    fn default() -> VisWireframe {
        VisWireframe {
            show_wireframe: false,
            wire_color: na::Vector4::<f32>::new(1.0, 0.0, 0.0, 1.0),
            wire_width: 1.0,
            added_automatically: false,
        }
    }
}
impl Default for VisNormals {
    fn default() -> VisNormals {
        VisNormals {
            show_normals: false,
            normals_color: na::Vector4::<f32>::new(1.0, 0.0, 0.0, 1.0),
            normals_width: 1.0,
            normals_scale: 1.0,
            added_automatically: false,
        }
    }
}
impl Default for VisPoints {
    fn default() -> VisPoints {
        VisPoints {
            show_points: false,
            show_points_indices: false,
            point_color: na::Vector4::<f32>::new(245.0 / 255.0, 175.0 / 255.0, 110.0 / 255.0, 1.0),
            point_size: 1.0,
            is_point_size_in_world_space: false,
            color_type: PointColorType::Solid,
            zbuffer: true,
            added_automatically: false,
        }
    }
}
impl Default for VisMesh {
    fn default() -> VisMesh {
        VisMesh {
            show_mesh: true,
            solid_color: na::Vector4::<f32>::new(1.0, 206.0 / 255.0, 143.0 / 255.0, 1.0),
            metalness: 0.0,
            perceptual_roughness: 0.5,
            roughness_black_lvl: 0.0,
            uv_scale: 1.0,
            opacity: 1.0,
            needs_sss: false,
            color_type: MeshColorType::Solid,
            added_automatically: false,
        }
    }
}

/// Component that transforms from object coordinates to world. Usually added
/// automatically but you can also add it yourself.
#[derive(Clone)]
pub struct ModelMatrix(pub na::SimilarityMatrix3<f32>); //transform from object coordinates to world corresponds to TfWorldObj
                                                        // pub struct ModelMatrix(pub na::Affine3<f32>); //transform from object
                                                        // coordinates to world corresponds to TfWorldObj
impl Default for ModelMatrix {
    fn default() -> ModelMatrix {
        ModelMatrix(na::SimilarityMatrix3::<f32>::identity())
    }
}
impl ModelMatrix {
    #[must_use]
    pub fn with_translation(self, t: &na::Vector3<f32>) -> Self {
        let mut mat = self;
        mat.0.append_translation_mut(&na::Translation3::new(t[0], t[1], t[2]));
        mat
    }
    #[must_use]
    pub fn with_rotation_rot3(self, r: &na::Rotation3<f32>) -> Self {
        let mut mat = self;
        mat.0.append_rotation_mut(r);
        mat
    }
    #[must_use]
    pub fn with_rotation_axis_angle(self, v: &na::Vector3<f32>) -> Self {
        let mut mat = self;
        mat.0
            .append_rotation_mut(&na::Rotation3::from_axis_angle(&na::UnitVector3::<f32>::new_normalize(*v), v.norm()));
        mat
    }
    #[must_use]
    pub fn with_rotation_euler(self, e: &na::Vector3<f32>) -> Self {
        let mut mat = self;
        mat.0.append_rotation_mut(&na::Rotation3::from_euler_angles(e.x, e.y, e.z));
        mat
    }
}

/// Component that represents a track of camera extrinsics - num frames x 16
#[derive(Clone, Debug)]
pub struct CamTrack(pub DMatrix<f32>);

#[derive(Clone, Debug)]
pub struct Verts(pub DynamicTensorFloat2D);

/// Component that represents a matrix of vertex positions for the first vertex
/// of an edge
#[derive(Clone, Debug)]
pub struct EdgesV1(pub DynamicTensorFloat2D);
/// Component that represents a matrix of vertex positions for the second vertex
/// of an edge
#[derive(Clone, Debug)]
pub struct EdgesV2(pub DynamicTensorFloat2D);

/// Component that represents a matrix of face indices for rendering a triangle
/// mesh
// #[derive(Clone)]
// pub struct Faces(pub DMatrix<u32>);
#[derive(Clone)]
pub struct Faces(pub DynamicTensorInt2D);

/// Component that represents a matrix of edges as Nx2 where each row is the
/// start idx and end idx of an edge which indexes into [``Verts``]
#[derive(Clone, Debug)]
pub struct Edges(pub DynamicTensorInt2D);

/// Component that represents UV coordinates
#[derive(Clone)]
pub struct UVs(pub DynamicTensorFloat2D);
// / Component that represents the scale of the uv coordinates. Increasing it
// will tile the texture on the mesh. #[derive(Clone)]
// pub struct UvScale(pub f32);

/// Component that represents normal vectors
#[derive(Clone)]
pub struct Normals(pub DynamicTensorFloat2D);

/// Component that represents tangents
#[derive(Clone)]
pub struct Tangents(pub DynamicTensorFloat2D);

/// Component that represents per vertex colors
#[derive(Clone)]
pub struct Colors(pub DynamicTensorFloat2D);

#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImgConfig {
    pub keep_on_cpu: bool,
    pub fast_upload: bool, //fast upload uses the staging memory of wgpu but potentially uses more memory
    pub generate_mipmaps: bool,
    pub mipmap_generation_cpu: bool,
}
impl Default for ImgConfig {
    fn default() -> Self {
        Self {
            keep_on_cpu: true,
            fast_upload: true,
            generate_mipmaps: true,
            mipmap_generation_cpu: false,
        }
    }
}

/// A generic Img that has a path and img
/// Concrete images like [``DiffuseImg``] will contain the generic img
#[derive(Clone)]
pub struct GenericImg {
    pub path: Option<String>,
    pub cpu_img: Option<DynImage>, //keep it as an option so we can drop it and release the memory
    pub config: ImgConfig,
}
impl GenericImg {
    /// # Panics
    /// Will panic if the path cannot be opened.
    pub fn new_from_path(path: &str, config: &ImgConfig) -> Self {
        let cpu_img = Some(ImageReader::open(path).unwrap().decode().unwrap());

        Self {
            path: Some(path.to_string()),
            cpu_img: cpu_img.map(|v| v.try_into().unwrap()),
            config: config.clone(),
        }
    }

    /// # Panics
    /// Will panic if the path cannot be opened.
    pub async fn new_from_path_async(path: &str, config: &ImgConfig) -> Self {
        let reader = ImageReader::new(BufReader::new(FileLoader::open(path).await))
            .with_guessed_format()
            .expect("Cursor io never fails");

        let cpu_img = Some(reader.decode().unwrap());

        Self {
            path: Some(path.to_string()),
            cpu_img: cpu_img.map(|v| v.try_into().unwrap()),
            config: config.clone(),
        }
    }

    /// # Panics
    /// Will panic if the img cannot be processed and decoded.
    pub fn new_from_buf(buf: &[u8], config: &ImgConfig) -> Self {
        Self::new_from_reader(Cursor::new(buf), config)
    }

    /// # Panics
    /// Will panic if the img cannot be processed and decoded.
    pub fn new_from_reader<R: Read + Seek>(reader: R, config: &ImgConfig) -> Self {
        let reader_img = ImageReader::new(BufReader::new(reader))
            .with_guessed_format()
            .expect("Format for image should be something known and valid");

        let cpu_img = Some(reader_img.decode().unwrap());

        Self {
            path: None,
            cpu_img: cpu_img.map(|v| v.try_into().unwrap()),
            config: config.clone(),
        }
    }

    pub fn img_ref(&self) -> &DynImage {
        self.cpu_img.as_ref().unwrap()
    }

    pub fn img_ref_mut(&mut self) -> &mut DynImage {
        self.cpu_img.as_mut().unwrap()
    }
}

/// Component which represents a diffuse img.
#[derive(Clone)]
pub struct DiffuseImg {
    pub generic_img: GenericImg,
}

impl DiffuseImg {
    pub fn new_from_path(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path(path, config);
        Self { generic_img }
    }

    pub async fn new_from_path_async(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path_async(path, config).await;
        Self { generic_img }
    }

    pub fn new_from_buf(buf: &[u8], config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_buf(buf, config);
        Self { generic_img }
    }

    pub fn new_from_reader<R: Read + Seek>(reader: R, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_reader(reader, config);
        Self { generic_img }
    }
}

/// Component which represents a normal img.
#[derive(Clone)]
pub struct NormalImg {
    pub generic_img: GenericImg,
}

impl NormalImg {
    pub fn new_from_path(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path(path, config);
        Self { generic_img }
    }

    pub async fn new_from_path_async(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path_async(path, config).await;
        Self { generic_img }
    }

    pub fn new_from_buf(buf: &[u8], config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_buf(buf, config);
        Self { generic_img }
    }

    pub fn new_from_reader<R: Read + Seek>(reader: R, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_reader(reader, config);
        Self { generic_img }
    }
}

/// Component which represents a metalness img.
pub struct MetalnessImg {
    pub generic_img: GenericImg,
}
impl MetalnessImg {
    pub fn new_from_path(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path(path, config);
        Self { generic_img }
    }

    pub async fn new_from_path_async(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path_async(path, config).await;
        Self { generic_img }
    }

    pub fn new_from_buf(buf: &[u8], config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_buf(buf, config);
        Self { generic_img }
    }

    pub fn new_from_reader<R: Read + Seek>(reader: R, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_reader(reader, config);
        Self { generic_img }
    }
}

/// Component which represents a roughness img. Assumes it is stored as
/// perceptual roughness/
pub struct RoughnessImg {
    pub generic_img: GenericImg,
}
impl RoughnessImg {
    pub fn new_from_path(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path(path, config);
        Self { generic_img }
    }

    pub async fn new_from_path_async(path: &str, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_path_async(path, config).await;
        Self { generic_img }
    }

    pub fn new_from_buf(buf: &[u8], config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_buf(buf, config);
        Self { generic_img }
    }

    pub fn new_from_reader<R: Read + Seek>(reader: R, config: &ImgConfig) -> Self {
        let generic_img = GenericImg::new_from_reader(reader, config);
        Self { generic_img }
    }
}

//implement some atributes for the vertex atributes so we can use them in a
// generic function also implement common things like the atom size of an
// element which is f32 for most or u32 for faces and edges https://stackoverflow.com/a/53085395
//https://stackoverflow.com/a/66794115
// pub trait CpuAtrib<T> {
//     // type TypeElement;
//     fn byte_size_element(&self) -> usize;
//     // pub fn get_data(&self) -> DMatrix<Self::TypeElement>;
//     fn data_ref(&self) -> &DMatrix<T>;
// }

// impl CpuAtrib<f32> for Colors {
//     fn byte_size_element(&self) -> usize {
//         std::mem::size_of::<f32>()
//     }
//     fn data_ref(&self) -> &DMatrix<f32> {
//         &self.0
//     }
// }

/// Environment map based ambient lighting representing light from distant
/// scenery.
///
/// When added as a resource to the scene, this component adds indirect light
/// to every point of the scene (including inside, enclosed areas) based on
/// an environment cubemap texture. This is similar to [`crate::AmbientLight`],
/// but higher quality, and is intended for outdoor scenes.
///
/// The environment map must be prefiltered into a diffuse and specular cubemap
/// based on the [split-sum approximation](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf).
///
/// To prefilter your environment map, you can use `KhronosGroup`'s [glTF-IBL-Sampler](https://github.com/KhronosGroup/glTF-IBL-Sampler).
/// The diffuse map uses the Lambertian distribution, and the specular map uses
/// the GGX distribution.
///
/// `KhronosGroup` also has several prefiltered environment maps that can be found [here](https://github.com/KhronosGroup/glTF-Sample-Environments)
pub struct EnvironmentMap {
    pub diffuse_path: String,
    pub specular_path: String,
}
impl EnvironmentMap {
    pub fn new_from_path(diffuse_path: &str, specular_path: &str) -> Self {
        Self {
            diffuse_path: String::from(diffuse_path),
            specular_path: String::from(specular_path),
        }
    }
}

// /so we can use the Components inside the Mutex<Hashmap> in the scene and wasm
// https://stackoverflow.com/a/73773940/22166964
// shenanigans
//ConfigDeltas
#[cfg(target_arch = "wasm32")]
unsafe impl Send for ConfigChanges {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for ConfigChanges {}
//verts
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Verts {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Verts {}
//vertse1
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EdgesV1 {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EdgesV1 {}
//vertse2
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EdgesV2 {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EdgesV2 {}
//edges
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Edges {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Edges {}
//faces
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Faces {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Faces {}
//uvs
#[cfg(target_arch = "wasm32")]
unsafe impl Send for UVs {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for UVs {}
//normalss
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Normals {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Normals {}
//tangents
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Tangents {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Tangents {}
//colors
#[cfg(target_arch = "wasm32")]
unsafe impl Send for Colors {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Colors {}
//Diffuseimg
#[cfg(target_arch = "wasm32")]
unsafe impl Send for DiffuseImg {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for DiffuseImg {}
//Normalimg
#[cfg(target_arch = "wasm32")]
unsafe impl Send for NormalImg {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for NormalImg {}
//Metalnessimg
#[cfg(target_arch = "wasm32")]
unsafe impl Send for MetalnessImg {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for MetalnessImg {}
//Roughnessimg
#[cfg(target_arch = "wasm32")]
unsafe impl Send for RoughnessImg {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for RoughnessImg {}
//EnvironmentMap
#[cfg(target_arch = "wasm32")]
unsafe impl Send for EnvironmentMap {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for EnvironmentMap {}
