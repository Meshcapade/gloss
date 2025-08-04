#![allow(clippy::missing_panics_doc)] //a lot of operations require inserting or removing component from a entity but
                                      // the entity will for sure exists so it will never panic

use crate::components::{Colors, Edges};
use gloss_hecs::EntityBuilder;
use gloss_utils::tensor::{DynamicMatrixOps, DynamicTensorFloat2D, DynamicTensorInt2D};
use log::debug;
use nalgebra_glm::{Vec2, Vec3};

extern crate nalgebra as na;
use crate::components::{Faces, ModelMatrix, Normals, UVs, Verts};
use core::f32;
use gloss_utils::io::FileType;
#[allow(unused_imports)]
use log::{error, info, warn};
use na::DMatrix;
use std::path::Path;
use tobj;

use ply_rs::{parser, ply};

/// Creates a cube
#[must_use]
pub fn build_cube(center: na::Point3<f32>) -> EntityBuilder {
    //makes a 1x1x1 vox in NDC. which has Z going into the screen
    let verts = DMatrix::<f32>::from_row_slice(
        8,
        3,
        &[
            //behind face (which has negative Z as the camera is now looking in the positive Z direction and this will be the face that is
            // behind the camera)
            -1.0, -1.0, -1.0, //bottom-left
            1.0, -1.0, -1.0, //bottom-right
            1.0, 1.0, -1.0, //top-right
            -1.0, 1.0, -1.0, //top-left
            //front face
            -1.0, -1.0, 1.0, //bottom-left
            1.0, -1.0, 1.0, //bottom-right
            1.0, 1.0, 1.0, //top-right
            -1.0, 1.0, 1.0, //top-left
        ],
    );

    //faces (2 triangles per faces, with 6 faces which makes 12 triangles)
    let faces = DMatrix::<u32>::from_row_slice(
        12,
        3,
        &[
            2, 1, 0, //
            2, 0, 3, //
            4, 5, 6, //
            7, 4, 6, //
            5, 0, 1, //
            4, 0, 5, //
            7, 6, 3, //
            3, 6, 2, //
            3, 0, 4, //
            3, 4, 7, //
            6, 5, 1, //
            6, 1, 2, //
        ],
    );

    //since we want each vertex at the corner to have multiple normals, we just
    // duplicate them
    let mut verts_dup = Vec::new();
    let mut faces_dup = Vec::new();
    for f in faces.row_iter() {
        let p1 = verts.row(f[0] as usize);
        let p2 = verts.row(f[2] as usize);
        let p3 = verts.row(f[1] as usize);
        verts_dup.push(p1);
        verts_dup.push(p2);
        verts_dup.push(p3);

        #[allow(clippy::cast_possible_truncation)]
        let v_size = verts_dup.len() as u32;
        // let new_face = na::RowDVector::<u32>::new(v_size-1, v_size-2,
        // v_size-3);//corresponds to p3,p2 and p1
        let new_face = na::RowDVector::<u32>::from_row_slice(&[v_size - 1, v_size - 2, v_size - 3]); //corresponds to p3,p2 and p1
        faces_dup.push(new_face);
    }

    let verts = na::DMatrix::<f32>::from_rows(verts_dup.as_slice());
    let faces = na::DMatrix::<u32>::from_rows(faces_dup.as_slice());

    let mut model_matrix = na::SimilarityMatrix3::<f32>::identity();
    // model_matrix.append_scaling_mut(0.1);
    model_matrix.append_translation_mut(&center.coords.into());
    let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
    let faces_tensor = DynamicTensorInt2D::from_dmatrix(&faces);

    let mut builder = EntityBuilder::new();
    builder.add(Verts(verts_tensor)).add(Faces(faces_tensor)).add(ModelMatrix(model_matrix));

    builder
}

/// Creates a plane based on a center and normal. If `transform_cpu_data` is
/// true, then the vertices of the plane are actually translated and rotated
/// on the CPU. Otherwise the transformation is done on the GPU using the
/// model matrix. This is useful when creating light planes for example when
/// we want the model matrix to be consistent with the orientation of the
/// light and therefore we set `transform_cpu_data=false`
#[must_use]
pub fn build_plane(center: na::Point3<f32>, normal: na::Vector3<f32>, size_x: f32, size_y: f32, transform_cpu_data: bool) -> EntityBuilder {
    //make 4 vertices
    let mut verts = DMatrix::<f32>::from_row_slice(
        4,
        3,
        &[
            -1.0 * size_x,
            0.0,
            -1.0 * size_y, //
            1.0 * size_x,
            0.0,
            -1.0 * size_y, //
            1.0 * size_x,
            0.0,
            1.0 * size_y, //
            -1.0 * size_x,
            0.0,
            1.0 * size_y, //
        ],
    );
    //make 2 faces
    let faces = DMatrix::<u32>::from_row_slice(
        2,
        3,
        &[
            2, 1, 0, //
            3, 2, 0,
        ],
    );

    //uvs
    let uvs = DMatrix::<f32>::from_row_slice(
        4,
        2,
        &[
            0.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            0.0, 1.0, //
        ],
    );

    //make a model matrix
    let up = na::Vector3::<f32>::new(0.0, 1.0, 0.0);
    let lookat = center + normal * 1.0;
    //up and normal are colinear so face_towards would fail, we just set to
    // identity
    let mut model_matrix = if up.angle(&normal.normalize()) < 1e-6 {
        let mut mm = na::SimilarityMatrix3::<f32>::identity();
        mm.append_translation_mut(&na::Translation3::from(center));
        mm
    } else {
        let mut m = na::SimilarityMatrix3::<f32>::face_towards(&center, &lookat, &up, 1.0);
        m = m
            * na::Rotation3::<f32>::from_axis_angle(&na::Vector3::z_axis(), std::f32::consts::FRAC_PI_2)
            * na::Rotation3::<f32>::from_axis_angle(&na::Vector3::x_axis(), std::f32::consts::FRAC_PI_2); //rotate 90 degrees
        m
    };

    if transform_cpu_data {
        //transform directly the verts
        for mut vert in verts.row_iter_mut() {
            let v_modif = model_matrix * na::Point3::from(vert.fixed_columns::<3>(0).transpose());
            vert.copy_from_slice(v_modif.coords.as_slice());
        }
        //reset to identity
        model_matrix = na::SimilarityMatrix3::<f32>::identity();
    }
    let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
    let faces_tensor = DynamicTensorInt2D::from_dmatrix(&faces);
    let uvs_tensor = DynamicTensorFloat2D::from_dmatrix(&uvs);

    let mut builder = EntityBuilder::new();
    builder
        .add(Verts(verts_tensor))
        .add(Faces(faces_tensor))
        .add(UVs(uvs_tensor))
        .add(ModelMatrix(model_matrix));

    builder
}

#[must_use]
pub fn build_floor() -> EntityBuilder {
    //make 4 vertices
    let verts = DMatrix::<f32>::from_row_slice(
        4,
        3,
        &[
            -1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0, //
            1.0, 0.0, 1.0, //
            -1.0, 0.0, 1.0, //
        ],
    );
    //make 2 faces
    let faces = DMatrix::<u32>::from_row_slice(
        2,
        3,
        &[
            2, 1, 0, //
            3, 2, 0,
        ],
    );

    //uvs
    let uvs = DMatrix::<f32>::from_row_slice(
        4,
        2,
        &[
            0.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            0.0, 1.0, //
        ],
    );
    let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
    let faces_tensor = DynamicTensorInt2D::from_dmatrix(&faces);
    let uvs_tensor = DynamicTensorFloat2D::from_dmatrix(&uvs);
    let mut builder = EntityBuilder::new();
    builder.add(Verts(verts_tensor)).add(Faces(faces_tensor)).add(UVs(uvs_tensor));

    builder
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn build_grid(
    center: na::Point3<f32>,
    normal: na::Vector3<f32>,
    nr_lines_x: u32,
    nr_lines_y: u32,
    size_x: f32,
    size_y: f32,
    transform_cpu_data: bool,
) -> EntityBuilder {
    //a grid has at least 2 lines in each dimension, because each square needs 2
    // lines, so we cap the number
    let nr_lines_x = nr_lines_x.max(2);
    let nr_lines_y = nr_lines_y.max(2);

    //in order to be consistent with build_plane we multiply the size in x and y by
    // 2 since it's technically the size from the center until the outer edge
    // let size_x = size_x * 2.0;
    // let size_y = size_y * 2.0;

    let size_cell_x = size_x / nr_lines_x as f32;
    let size_cell_y = size_y / nr_lines_y as f32;
    let grid_half_size_x = size_x / 2.0;
    let grid_half_size_y = size_y / 2.0;

    // println!("nr_linex_x {}", size_cell_x);
    // println!("size_cell_x {}", size_cell_x);
    // println!("grid_half_size_x {}", grid_half_size_x);

    //make points
    let mut verts = Vec::new();
    for idx_y in 0..nr_lines_y {
        for idx_x in 0..nr_lines_x {
            verts.push(idx_x as f32 * size_cell_x - grid_half_size_x);
            verts.push(0.0);
            verts.push(idx_y as f32 * size_cell_y - grid_half_size_y);
        }
    }

    //make edges horizontally
    let mut edges_h = Vec::new();
    for idx_y in 0..nr_lines_y {
        for idx_x in 0..nr_lines_x - 1 {
            let idx_cur = idx_y * nr_lines_x + idx_x;
            let idx_next = idx_y * nr_lines_x + idx_x + 1;
            edges_h.push(idx_cur);
            edges_h.push(idx_next);
        }
    }

    //make edges vertically
    let mut edges_v = Vec::new();
    for idx_y in 0..nr_lines_y - 1 {
        for idx_x in 0..nr_lines_x {
            let idx_cur = idx_y * nr_lines_x + idx_x;
            let idx_next = (idx_y + 1) * nr_lines_x + idx_x;
            edges_v.push(idx_cur);
            edges_v.push(idx_next);
        }
    }

    let mut edges = edges_h;
    edges.extend(edges_v);

    //make nalgebra matrices
    let mut verts = DMatrix::<f32>::from_row_slice(verts.len() / 3, 3, &verts);
    let edges = DMatrix::<u32>::from_row_slice(edges.len() / 2, 2, &edges);

    //make a model matrix
    let up = na::Vector3::<f32>::new(0.0, 1.0, 0.0);
    let lookat = center + normal * 1.0;
    //up and normal are colinear so face_towards would fail, we just set to
    // identity
    let mut model_matrix = if up.angle(&normal.normalize()) < 1e-6 {
        let mut mm = na::SimilarityMatrix3::<f32>::identity();
        mm.append_translation_mut(&na::Translation3::from(center));
        mm
    } else {
        let mut m = na::SimilarityMatrix3::<f32>::face_towards(&center, &lookat, &up, 1.0);
        m = m
            * na::Rotation3::<f32>::from_axis_angle(&na::Vector3::z_axis(), std::f32::consts::FRAC_PI_2)
            * na::Rotation3::<f32>::from_axis_angle(&na::Vector3::x_axis(), std::f32::consts::FRAC_PI_2); //rotate 90 degrees
        m
    };

    if transform_cpu_data {
        //transform directly the verts
        for mut vert in verts.row_iter_mut() {
            let v_modif = model_matrix * na::Point3::from(vert.fixed_columns::<3>(0).transpose());
            vert.copy_from_slice(v_modif.coords.as_slice());
        }
        //reset to identity
        model_matrix = na::SimilarityMatrix3::<f32>::identity();
    }
    let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
    let edges_tensor = DynamicTensorInt2D::from_dmatrix(&edges);

    let mut builder = EntityBuilder::new();
    builder.add(Verts(verts_tensor)).add(Edges(edges_tensor)).add(ModelMatrix(model_matrix));

    builder
}

pub fn build_from_file(path: &str) -> EntityBuilder {
    //get filetype
    let filetype = match Path::new(path).extension() {
        Some(extension) => FileType::find_match(extension.to_str().unwrap_or("")),
        None => FileType::Unknown,
    };

    #[allow(clippy::single_match_else)]
    match filetype {
        FileType::Obj => build_from_obj(Path::new(path)),
        FileType::Ply => build_from_ply(Path::new(path)),
        FileType::Unknown => {
            error!("Could not read file {:?}", path);
            EntityBuilder::new() //empty builder
        }
    }
}

/// # Panics
/// Will panic if the path cannot be opened
#[cfg(target_arch = "wasm32")]
pub async fn build_from_file_async(path: &str) -> EntityBuilder {
    //get filetype
    let filetype = match Path::new(path).extension() {
        Some(extension) => FileType::find_match(extension.to_str().unwrap_or("")),
        _ => FileType::Unknown,
    };

    match filetype {
        FileType::Obj => build_from_obj_async(Path::new(path)).await,
        // FileType::Ply => (),
        _ => {
            error!("Could not read file {:?}", path);
            EntityBuilder::new() //empty builder
        }
    }
}

/// # Panics
/// Will panic if the path cannot be opened
#[allow(clippy::unused_async)] //uses async for wasm
#[allow(clippy::identity_op)] //identity ops makes some things more explixit
fn build_from_obj(path: &Path) -> EntityBuilder {
    info!("reading obj from {path:?}");

    //Native read
    let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS).expect("Failed to OBJ load file");

    model_obj_to_entity_builder(&models)
}

/// # Panics
/// Will panic if the path cannot be opened
#[allow(clippy::unused_async)] //uses async for wasm
#[cfg(target_arch = "wasm32")]
#[allow(deprecated)]
async fn build_from_obj_async(path: &Path) -> EntityBuilder {
    //WASM read
    let mut file_wasm = gloss_utils::io::FileLoader::open(path.to_str().unwrap()).await;
    let (models, _) = tobj::load_obj_buf_async(&mut file_wasm, &tobj::GPU_LOAD_OPTIONS, move |p| async move {
        match p.as_str() {
            _ => unreachable!(),
        }
    })
    .await
    .expect("Failed to OBJ load file");

    model_obj_to_entity_builder(&models)
}

/// # Panics
/// Will panic if the path cannot be opened
#[allow(clippy::unused_async)] //uses async for wasm
#[allow(clippy::identity_op)] //identity ops makes some things more explixit
pub fn build_from_obj_buf(buf: &[u8]) -> EntityBuilder {
    let mut reader = std::io::BufReader::new(buf);

    //Native read
    let (models, _) = tobj::load_obj_buf(&mut reader, &tobj::GPU_LOAD_OPTIONS, move |_p| Err(tobj::LoadError::MaterialParseError))
        .expect("Failed to OBJ load file");

    model_obj_to_entity_builder(&models)
}

#[allow(clippy::identity_op)] //identity ops makes some things more explixit
fn model_obj_to_entity_builder(models: &[tobj::Model]) -> EntityBuilder {
    // fn model_obj_to_entity_builder(model: &ObjData) -> EntityBuilder{

    let mesh = &models[0].mesh;
    debug!("obj: nr indices {}", mesh.indices.len() / 3);
    debug!("obj: nr positions {}", mesh.positions.len() / 3);
    debug!("obj: nr normals {}", mesh.normals.len() / 3);
    debug!("obj: nr texcoords {}", mesh.texcoords.len() / 2);

    let nr_verts = mesh.positions.len() / 3;
    let nr_faces = mesh.indices.len() / 3;
    let nr_normals = mesh.normals.len() / 3;
    let nr_texcoords = mesh.texcoords.len() / 2;

    let mut builder = EntityBuilder::new();

    if nr_verts > 0 {
        debug!("read_obj: file has verts");
        let verts = DMatrix::<f32>::from_row_slice(nr_verts, 3, mesh.positions.as_slice());
        let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
        builder.add(Verts(verts_tensor));
    }

    if nr_faces > 0 {
        debug!("read_obj: file has faces");
        let faces = DMatrix::<u32>::from_row_slice(nr_faces, 3, mesh.indices.as_slice());
        let faces_tensor = DynamicTensorInt2D::from_dmatrix(&faces);
        builder.add(Faces(faces_tensor));
    }

    if nr_normals > 0 {
        debug!("read_obj: file has normals");
        let normals = DMatrix::<f32>::from_row_slice(nr_normals, 3, mesh.normals.as_slice());
        let normals_tensor = DynamicTensorFloat2D::from_dmatrix(&normals);
        builder.add(Normals(normals_tensor));
    }

    if nr_texcoords > 0 {
        debug!("read_obj: file has texcoords");
        let uv = DMatrix::<f32>::from_row_slice(nr_texcoords, 2, mesh.texcoords.as_slice());
        let uvs_tensor = DynamicTensorFloat2D::from_dmatrix(&uv);
        builder.add(UVs(uvs_tensor));
    }

    // if !mesh.faces_original_index.is_empty() {
    //     builder.add(FacesOriginalIndex(mesh.faces_original_index.clone()));
    // }

    builder
}

/// # Panics
/// Will panic if the path cannot be opened
#[allow(clippy::unused_async)] //uses async for wasm
#[allow(clippy::identity_op)] //identity ops makes some things more explixit
#[allow(clippy::too_many_lines)] //identity ops makes some things more explixit
fn build_from_ply(path: &Path) -> EntityBuilder {
    #[derive(Debug, Default)]
    pub struct Vertex {
        pos: Vec3,
        color: Vec3,
        normal: Vec3,
        uv: Vec2,
    }

    #[derive(Debug)]
    pub struct Face {
        vertex_index: Vec<u32>,
    }

    // The structs need to implement the PropertyAccess trait, otherwise the parser
    // doesn't know how to write to them. Most functions have default, hence
    // you only need to implement, what you expect to need.
    impl ply::PropertyAccess for Vertex {
        fn new() -> Self {
            Self::default()
        }
        fn set_property(&mut self, key: String, property: ply::Property) {
            match (key.as_ref(), property) {
                ("x", ply::Property::Float(v)) => self.pos.x = v,
                ("y", ply::Property::Float(v)) => self.pos.y = v,
                ("z", ply::Property::Float(v)) => self.pos.z = v,
                ("red", ply::Property::UChar(v)) => {
                    self.color.x = f32::from(v) / 255.0;
                }
                ("green", ply::Property::UChar(v)) => self.color.y = f32::from(v) / 255.0,
                ("blue", ply::Property::UChar(v)) => self.color.z = f32::from(v) / 255.0,
                //normal
                ("nx", ply::Property::Float(v)) => self.normal.x = v,
                ("ny", ply::Property::Float(v)) => self.normal.y = v,
                ("nz", ply::Property::Float(v)) => self.normal.z = v,
                //uv
                ("u" | "s", ply::Property::Float(v)) => self.uv.x = v,
                ("v" | "t", ply::Property::Float(v)) => self.uv.y = v,
                // (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
                // (k, prop) => {println!("unknown key {} of type {:?}", k, prop)},
                (k, prop) => {
                    warn!("unknown key {} of type {:?}", k, prop);
                }
            }
        }
    }

    // same thing for Face
    impl ply::PropertyAccess for Face {
        fn new() -> Self {
            Face { vertex_index: Vec::new() }
        }
        #[allow(clippy::cast_sign_loss)]
        fn set_property(&mut self, key: String, property: ply::Property) {
            match (key.as_ref(), property.clone()) {
                ("vertex_indices" | "vertex_index", ply::Property::ListInt(vec)) => {
                    self.vertex_index = vec.iter().map(|x| *x as u32).collect();
                }
                ("vertex_indices" | "vertex_index", ply::Property::ListUInt(vec)) => {
                    self.vertex_index = vec;
                }
                (k, _) => {
                    panic!("Face: Unexpected key/value combination: key, val: {k} {property:?}")
                }
            }
        }
    }

    info!("reading ply from {path:?}");
    // set up a reader, in this a file.
    let f = std::fs::File::open(path).unwrap();
    // The header of a ply file consists of ascii lines, BufRead provides useful
    // methods for that.
    let mut f = std::io::BufReader::new(f);

    // Create a parser for each struct. Parsers are cheap objects.
    let vertex_parser = parser::Parser::<Vertex>::new();
    let face_parser = parser::Parser::<Face>::new();

    // lets first consume the header
    // We also could use `face_parser`, The configuration is a parser's only state.
    // The reading position only depends on `f`.
    let header = vertex_parser.read_header(&mut f).unwrap();

    // Depending on the header, read the data into our structs..
    let mut vertex_list = Vec::new();
    let mut face_list = Vec::new();
    for (_ignore_key, element) in &header.elements {
        // we could also just parse them in sequence, but the file format might change
        match element.name.as_ref() {
            "vertex" | "point" => {
                vertex_list = vertex_parser.read_payload_for_element(&mut f, element, &header).unwrap();
            }
            "face" => {
                face_list = face_parser.read_payload_for_element(&mut f, element, &header).unwrap();
            }
            unknown_name => panic!("Unexpected element! {unknown_name}"),
        }
    }

    let mut builder = EntityBuilder::new();

    //pos
    let mut verts = DMatrix::<f32>::zeros(vertex_list.len(), 3);
    for (idx, v) in vertex_list.iter().enumerate() {
        verts.row_mut(idx)[0] = v.pos.x;
        verts.row_mut(idx)[1] = v.pos.y;
        verts.row_mut(idx)[2] = v.pos.z;
    }
    let verts_tensor = DynamicTensorFloat2D::from_dmatrix(&verts);
    builder.add(Verts(verts_tensor));
    //color
    let mut colors = DMatrix::<f32>::zeros(vertex_list.len(), 3);
    for (idx, v) in vertex_list.iter().enumerate() {
        colors.row_mut(idx)[0] = v.color.x;
        colors.row_mut(idx)[1] = v.color.y;
        colors.row_mut(idx)[2] = v.color.z;
    }
    //TODO need a better way to detect if we actually have color info
    if colors.min() != 0.0 || colors.max() != 0.0 {
        debug!("read_ply: file has colors");
        let colors_tensor = DynamicTensorFloat2D::from_dmatrix(&colors);
        builder.add(Colors(colors_tensor));
    }
    //normal
    let mut normals = DMatrix::<f32>::zeros(vertex_list.len(), 3);
    for (idx, v) in vertex_list.iter().enumerate() {
        normals.row_mut(idx)[0] = v.normal.x;
        normals.row_mut(idx)[1] = v.normal.y;
        normals.row_mut(idx)[2] = v.normal.z;
    }
    //TODO need a better way to detect if we actually have normal info
    if normals.min() != 0.0 || normals.max() != 0.0 {
        debug!("read_ply: file has normals");
        let normals_tensor = DynamicTensorFloat2D::from_dmatrix(&normals);
        builder.add(Normals(normals_tensor));
    }
    //uv
    let mut uvs = DMatrix::<f32>::zeros(vertex_list.len(), 2);
    for (idx, v) in vertex_list.iter().enumerate() {
        uvs.row_mut(idx)[0] = v.uv.x;
        uvs.row_mut(idx)[1] = v.uv.y;
    }
    //TODO need a better way to detect if we actually have normal info
    if uvs.min() != 0.0 || uvs.max() != 0.0 {
        debug!("read_ply: file has uvs");
        let uvs_tensor = DynamicTensorFloat2D::from_dmatrix(&uvs);
        builder.add(UVs(uvs_tensor));
    }

    if !face_list.is_empty() {
        debug!("read_ply: file has verts");
        let mut faces = DMatrix::<u32>::zeros(face_list.len(), 3);
        #[allow(clippy::cast_sign_loss)]
        for (idx, f) in face_list.iter().enumerate() {
            faces.row_mut(idx)[0] = f.vertex_index[0];
            faces.row_mut(idx)[1] = f.vertex_index[1];
            faces.row_mut(idx)[2] = f.vertex_index[2];
        }
        let faces_tensor = DynamicTensorInt2D::from_dmatrix(&faces);

        builder.add(Faces(faces_tensor));
    }

    builder
}
