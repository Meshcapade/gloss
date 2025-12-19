// Geom contains functionality related to mesh-like entities, like reading from
// files, or fixing and processing mesh data. Contains mostly static functions

#![allow(clippy::missing_panics_doc)] //a lot of operations require inserting or removing component from a entity but
                                      // the entity will for sure exists so it will never panic

use gloss_img::DynImage;
use image::{GenericImageView, Pixel};
extern crate nalgebra as na;
use core::f32;
use itertools::izip;
#[allow(unused_imports)]
use log::{error, info, warn};
use na::DMatrix;
use obj_exporter::{Geometry, ObjSet, Object, Primitive, Shape, TVertex, Vertex};
use std::ops::AddAssign;

use ply_rs::ply::{self, Addable, Ply};

use burn::tensor::{backend::Backend, Float, Int, Tensor};
use gloss_utils::tensor;

#[derive(PartialEq)]
pub enum PerVertexNormalsWeightingType {
    /// Incident face normals have uniform influence on vertex normal
    Uniform,
    /// Incident face normals are averaged weighted by area
    Area,
}

#[derive(PartialEq)]
pub enum SplatType {
    /// Averages all the values that end up in the same index
    Avg,
    /// Sums all the values that end up in the same index
    Sum,
}

#[derive(PartialEq)]
pub enum IndirRemovalPolicy {
    RemoveInvalidRows,
    RemoveInvalidCols,
}

pub fn save_obj(verts: &DMatrix<f32>, faces: Option<&DMatrix<u32>>, uv: Option<&DMatrix<f32>>, normals: Option<&DMatrix<f32>>, path: &str) {
    let verts_obj: Vec<Vertex> = verts
        .row_iter()
        .map(|row| Vertex {
            x: f64::from(row[0]),
            y: f64::from(row[1]),
            z: f64::from(row[2]),
        })
        .collect();
    let faces_obj: Vec<Shape> = if let Some(faces) = faces {
        faces
            .row_iter()
            .map(|row| Shape {
                primitive: Primitive::Triangle(
                    // (row[0] as usize, Some(row[0] as usize), None),
                    // (row[1] as usize, Some(row[1] as usize), None),
                    // (row[2] as usize, Some(row[2] as usize), None),
                    (row[0] as usize, uv.map(|_| row[0] as usize), normals.map(|_| row[0] as usize)),
                    (row[1] as usize, uv.map(|_| row[1] as usize), normals.map(|_| row[1] as usize)),
                    (row[2] as usize, uv.map(|_| row[2] as usize), normals.map(|_| row[2] as usize)),
                ),
                groups: vec![],
                smoothing_groups: vec![],
            })
            .collect()
    } else {
        Vec::new()
    };
    let uv_obj: Vec<TVertex> = if let Some(uv) = uv {
        uv.row_iter()
            .map(|row| TVertex {
                u: f64::from(row[0]),
                v: f64::from(row[1]),
                w: 0.0,
            })
            .collect()
    } else {
        Vec::<TVertex>::new()
    };

    let normals_obj: Vec<Vertex> = if let Some(normals) = normals {
        normals
            .row_iter()
            .map(|row| Vertex {
                x: f64::from(row[0]),
                y: f64::from(row[1]),
                z: f64::from(row[2]),
            })
            .collect()
    } else {
        Vec::<Vertex>::new()
    };

    let set = ObjSet {
        material_library: None,
        objects: vec![Object {
            name: "exported_mesh".to_owned(),
            vertices: verts_obj,
            tex_vertices: uv_obj,
            normals: normals_obj,
            geometry: vec![Geometry {
                material_name: None,
                shapes: faces_obj,
            }],
        }],
    };

    obj_exporter::export_to_file(&set, path).unwrap();
}

#[allow(clippy::too_many_lines)]
pub fn save_ply(
    verts: &DMatrix<f32>,
    faces: Option<&DMatrix<u32>>,
    uvs: Option<&DMatrix<f32>>,
    normals: Option<&DMatrix<f32>>,
    colors: Option<&DMatrix<f32>>,
    path: &str,
) {
    let ply_float_prop = ply::PropertyType::Scalar(ply::ScalarType::Float);
    let ply_uchar_prop = ply::PropertyType::Scalar(ply::ScalarType::UChar);

    #[allow(clippy::approx_constant)]
    let mut ply = {
        let mut ply = Ply::<ply::DefaultElement>::new();
        // ply.header.encoding = ply::Encoding::BinaryLittleEndian;
        ply.header.encoding = ply::Encoding::Ascii; //for some reason meshlab only opens these type of ply files
        ply.header.comments.push("Gloss Ply file".to_string());

        // Define the elements we want to write. In our case we write a 2D Point.
        // When writing, the `count` will be set automatically to the correct value by
        // calling `make_consistent`
        let mut point_element = ply::ElementDef::new("vertex".to_string());
        let p = ply::PropertyDef::new("x".to_string(), ply_float_prop.clone());
        point_element.properties.add(p);
        let p = ply::PropertyDef::new("y".to_string(), ply_float_prop.clone());
        point_element.properties.add(p);
        let p = ply::PropertyDef::new("z".to_string(), ply_float_prop.clone());
        point_element.properties.add(p);

        if normals.is_some() {
            let p = ply::PropertyDef::new("nx".to_string(), ply_float_prop.clone());
            point_element.properties.add(p);
            let p = ply::PropertyDef::new("ny".to_string(), ply_float_prop.clone());
            point_element.properties.add(p);
            let p = ply::PropertyDef::new("nz".to_string(), ply_float_prop.clone());
            point_element.properties.add(p);
        }

        //neds to be called s,t because that's what blender expects
        if uvs.is_some() {
            let p = ply::PropertyDef::new("s".to_string(), ply_float_prop.clone());
            point_element.properties.add(p);
            let p = ply::PropertyDef::new("t".to_string(), ply_float_prop.clone());
            point_element.properties.add(p);
        }

        //neds to be called s,t because that's what blender expects
        if colors.is_some() {
            let p = ply::PropertyDef::new("red".to_string(), ply_uchar_prop.clone());
            point_element.properties.add(p);
            let p = ply::PropertyDef::new("green".to_string(), ply_uchar_prop.clone());
            point_element.properties.add(p);
            let p = ply::PropertyDef::new("blue".to_string(), ply_uchar_prop.clone());
            point_element.properties.add(p);
        }

        ply.header.elements.add(point_element);

        //face
        let mut face_element = ply::ElementDef::new("face".to_string());
        //x
        let f = ply::PropertyDef::new(
            "vertex_indices".to_string(),
            ply::PropertyType::List(ply::ScalarType::UChar, ply::ScalarType::UInt), //has to be kept as uchar, uint for meshlab to read it
        );
        face_element.properties.add(f);
        ply.header.elements.add(face_element);

        // Add points
        let mut points_list = Vec::new();
        for (idx, vert) in verts.row_iter().enumerate() {
            let mut point_elem = ply::DefaultElement::new();
            point_elem.insert("x".to_string(), ply::Property::Float(vert[0]));
            point_elem.insert("y".to_string(), ply::Property::Float(vert[1]));
            point_elem.insert("z".to_string(), ply::Property::Float(vert[2]));

            if let Some(normals) = normals {
                let normal = normals.row(idx);
                point_elem.insert("nx".to_string(), ply::Property::Float(normal[0]));
                point_elem.insert("ny".to_string(), ply::Property::Float(normal[1]));
                point_elem.insert("nz".to_string(), ply::Property::Float(normal[2]));
            }

            if let Some(uvs) = uvs {
                let uv = uvs.row(idx);
                point_elem.insert("s".to_string(), ply::Property::Float(uv[0]));
                point_elem.insert("t".to_string(), ply::Property::Float(uv[1]));
            }

            #[allow(clippy::cast_sign_loss)]
            if let Some(colors) = colors {
                let color = colors.row(idx);
                #[allow(clippy::cast_possible_truncation)]
                point_elem.insert("red".to_string(), ply::Property::UChar((color[0] * 255.0) as u8));
                #[allow(clippy::cast_possible_truncation)]
                point_elem.insert("green".to_string(), ply::Property::UChar((color[1] * 255.0) as u8));
                #[allow(clippy::cast_possible_truncation)]
                point_elem.insert("blue".to_string(), ply::Property::UChar((color[2] * 255.0) as u8));
            }

            points_list.push(point_elem);
        }
        ply.payload.insert("vertex".to_string(), points_list);

        // Add faces
        if let Some(faces) = faces {
            let mut faces_list = Vec::new();
            for face in faces.row_iter() {
                let mut face_elem = ply::DefaultElement::new();
                face_elem.insert("vertex_indices".to_string(), ply::Property::ListUInt(face.iter().copied().collect()));
                faces_list.push(face_elem);
            }
            ply.payload.insert("face".to_string(), faces_list);
        }

        // only `write_ply` calls this by itself, for all other methods the client is
        // responsible to make the data structure consistent.
        // We do it here for demonstration purpose.
        ply.make_consistent().unwrap();
        ply
    };

    let mut file = std::fs::File::create(path).unwrap();
    let w = ply_rs::writer::Writer::new();
    let written = w.write_ply(&mut file, &mut ply).unwrap();
    println!("{written} bytes written");
}

/// Computes per vertex normals
pub fn compute_per_vertex_normals(
    verts: &na::DMatrix<f32>,
    faces: &na::DMatrix<u32>,
    weighting_type: &PerVertexNormalsWeightingType,
) -> na::DMatrix<f32> {
    match weighting_type {
        PerVertexNormalsWeightingType::Uniform => {
            let faces_row_major = faces.transpose(); //for more efficient row iteration
            let verts_row_major = verts.transpose(); //for more efficient row iteration
            let mut per_vertex_normal_row_major = DMatrix::<f32>::zeros(3, verts.nrows());
            for face in faces_row_major.column_iter() {
                // let v0 = verts_row_major.column(face[0] as usize);
                // let v1 = verts_row_major.column(face[1] as usize);
                // let v2 = verts_row_major.column(face[2] as usize);

                let v0 = verts_row_major.fixed_view::<3, 1>(0, face[0] as usize);
                let v1 = verts_row_major.fixed_view::<3, 1>(0, face[1] as usize);
                let v2 = verts_row_major.fixed_view::<3, 1>(0, face[2] as usize);
                let d1 = v1 - v0;
                let d2 = v2 - v0;
                let face_normal = d1.cross(&d2).normalize();
                //splat to vertices
                per_vertex_normal_row_major.column_mut(face[0] as usize).add_assign(&face_normal);
                per_vertex_normal_row_major.column_mut(face[1] as usize).add_assign(&face_normal);
                per_vertex_normal_row_major.column_mut(face[2] as usize).add_assign(&face_normal);
            }
            // take average via normalization
            for mut row in per_vertex_normal_row_major.column_iter_mut() {
                row /= row.norm();
            }
            per_vertex_normal_row_major.transpose()
        }
        PerVertexNormalsWeightingType::Area => {
            //compute per face normals
            let face_normals = compute_per_face_normals(verts, faces);
            //calculate weights for each face that depend on the area.
            let w = compute_double_face_areas(verts, faces);
            //per vertex
            let mut per_vertex_normal = DMatrix::<f32>::zeros(verts.nrows(), 3);
            for ((face, face_normal), &face_weight) in faces.row_iter().zip(face_normals.row_iter()).zip(w.iter()) {
                for &v_idx in face.iter() {
                    per_vertex_normal.row_mut(v_idx as usize).add_assign(face_weight * face_normal);
                }
            }
            // take average via normalization
            for mut row in per_vertex_normal.row_iter_mut() {
                row /= row.norm();
            }
            per_vertex_normal
        }
    }
}

/// Computes per vertex normals for Burn Tensors
pub fn compute_per_vertex_normals_burn<B: Backend>(
    verts: &Tensor<B, 2, Float>, // Tensor of shape [NUM_VERTS, 3]
    faces: &Tensor<B, 2, Int>,   // Tensor of shape [NUM_FACES, 3]
    weighting_type: &PerVertexNormalsWeightingType,
) -> Tensor<B, 2, Float> {
    let num_verts = verts.shape().dims[0];
    let num_faces = faces.shape().dims[0];
    let mut per_vertex_normals = Tensor::<B, 2, Float>::zeros([num_verts, 3], &verts.device());
    // let now = wasm_timer::Instant::now();
    match weighting_type {
        PerVertexNormalsWeightingType::Uniform => {
            let all_idxs: Tensor<B, 1, Int> = faces.clone().flatten(0, 1);
            let all_tris_as_verts = verts.clone().select(0, all_idxs).reshape([num_faces, 3, 3]);
            let all_tris_as_verts_chunks = all_tris_as_verts.chunk(3, 1); // Split the tensor along the second dimension (3 components)
                                                                          // println!("2 ---- {:?}", now.elapsed());

            // Now assign each chunk to v0, v1, v2
            let v0 = all_tris_as_verts_chunks[0].clone().squeeze(1); // First chunk (v0) [num_faces, 3]
            let v1 = all_tris_as_verts_chunks[1].clone().squeeze(1); // Second chunk (v1) [num_faces, 3]
            let v2 = all_tris_as_verts_chunks[2].clone().squeeze(1); // Third chunk (v2) [num_faces, 3]
                                                                     // println!("3 ---- {:?}", now.elapsed());

            // Compute v1 - v0 and v2 - v0
            let d1 = v1.sub(v0.clone()); // Shape: [num_faces, 3]
            let d2 = v2.sub(v0.clone()); // Shape: [num_faces, 3]
                                         // println!("4 ---- {:?}", now.elapsed());

            // Perform batch cross product between d1 and d2
            let cross_product = tensor::cross_product(&d1, &d2); // Shape: [num_faces, 3]
            let face_normals = tensor::normalize_tensor(cross_product);

            // Scatter the face normals back into the per-vertex normals tensor
            let face_indices_expanded: Tensor<B, 1, Int> = faces.clone().flatten(0, 1); // Shape [num_faces * 3]
            let mut face_normals_repeated: Tensor<B, 3> = face_normals.unsqueeze_dim(1);

            face_normals_repeated = face_normals_repeated.repeat(&[1, 3, 1]);

            let face_normals_to_scatter = face_normals_repeated.reshape([num_faces * 3, 3]); // Repeat face normals for each vertex
                                                                                             // println!("8 ---- {:?}", now.elapsed());

            per_vertex_normals = per_vertex_normals.select_assign(0, face_indices_expanded, face_normals_to_scatter);
            // Normalize the per-vertex normals to get the average
            per_vertex_normals = tensor::normalize_tensor(per_vertex_normals);
            per_vertex_normals
        }
        PerVertexNormalsWeightingType::Area => {
            ////////////////////////////////////////////////////////////////////////
            let all_idxs: Tensor<B, 1, Int> = faces.clone().flatten(0, 1);
            let all_tris_as_verts = verts.clone().select(0, all_idxs).reshape([num_faces, 3, 3]);
            let all_tris_as_verts_chunks = all_tris_as_verts.chunk(3, 1); // Split into v0, v1, v2

            let v0 = all_tris_as_verts_chunks[0].clone().squeeze(1); // [num_faces, 3]
            let v1 = all_tris_as_verts_chunks[1].clone().squeeze(1); // [num_faces, 3]
            let v2 = all_tris_as_verts_chunks[2].clone().squeeze(1); // [num_faces, 3]

            let d1 = v1.sub(v0.clone()); // [num_faces, 3]
            let d2 = v2.sub(v0.clone()); // [num_faces, 3]

            let face_normals = tensor::cross_product(&d1, &d2); // [num_faces, 3]

            let face_areas = face_normals.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // [num_faces]

            let weighted_face_normals = face_normals.div(face_areas); // [num_faces, 3]

            let face_indices_expanded: Tensor<B, 1, Int> = faces.clone().flatten(0, 1); // [num_faces * 3]

            let mut weighted_face_normals_repeated: Tensor<B, 3> = weighted_face_normals.unsqueeze_dim(1);
            weighted_face_normals_repeated = weighted_face_normals_repeated.repeat(&[1, 3, 1]);

            let weighted_normals_to_scatter = weighted_face_normals_repeated.reshape([num_faces * 3, 3]);

            per_vertex_normals = per_vertex_normals.select_assign(0, face_indices_expanded, weighted_normals_to_scatter);

            per_vertex_normals = tensor::normalize_tensor(per_vertex_normals);
            per_vertex_normals
        }
    }
}

/// Computes per face normals
pub fn compute_per_face_normals(verts: &na::DMatrix<f32>, faces: &na::DMatrix<u32>) -> na::DMatrix<f32> {
    let verts_row_major = verts.transpose(); //for more efficient row iteration
                                             // let mut face_normals = DMatrix::<f32>::zeros(faces.nrows(), 3);
    let mut face_normals = unsafe { DMatrix::<core::mem::MaybeUninit<f32>>::uninit(na::Dyn(faces.nrows()), na::Dyn(3)).assume_init() };

    for (face, mut face_normal) in faces.row_iter().zip(face_normals.row_iter_mut()) {
        // let v0 = verts.row(face[0] as usize);
        // let v1 = verts.row(face[1] as usize);
        // let v2 = verts.row(face[2] as usize);
        let v0 = verts_row_major.fixed_view::<3, 1>(0, face[0] as usize);
        let v1 = verts_row_major.fixed_view::<3, 1>(0, face[1] as usize);
        let v2 = verts_row_major.fixed_view::<3, 1>(0, face[2] as usize);
        let d1 = v1 - v0;
        let d2 = v2 - v0;
        let normal = d1.cross(&d2).normalize();
        //TODO make the face normals to be row major and transpose after we have
        // written to them
        face_normal.copy_from(&normal.transpose());
    }

    face_normals
}

///computes twice the area for each input triangle[quad]
pub fn compute_double_face_areas(verts: &na::DMatrix<f32>, faces: &na::DMatrix<u32>) -> na::DMatrix<f32> {
    //helper function
    let proj_doublearea =
            // |verts: &na::DMatrix<f32>, faces: &na::DMatrix<f32>, x, y, f| -> f32 { 3.0 };
            |x, y, f| -> f32 {
                let fx = faces[(f,0)] as usize;
                let fy = faces[(f,1)] as usize;
                let fz = faces[(f,2)] as usize;
                let rx = verts[(fx,x)]-verts[(fz,x)];
                let sx = verts[(fy,x)]-verts[(fz,x)];
                let ry = verts[(fx,y)]-verts[(fz,y)];
                let sy = verts[(fy,y)]-verts[(fz,y)];
                rx*sy - ry*sx
            };

    let mut double_face_areas = DMatrix::<f32>::zeros(faces.nrows(), 1);

    for f in 0..faces.nrows() {
        for d in 0..3 {
            let double_area = proj_doublearea(d, (d + 1) % 3, f);
            double_face_areas[(f, 0)] += double_area * double_area;
        }
    }
    //sqrt for every value
    double_face_areas = double_face_areas.map(f32::sqrt);

    double_face_areas
}

#[allow(clippy::similar_names)]
/// Compute tangents given verts, faces, normals and uvs
pub fn compute_tangents(verts: &na::DMatrix<f32>, faces: &na::DMatrix<u32>, normals: &na::DMatrix<f32>, uvs: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    //if we have UV per vertex then we can calculate a tangent that is aligned with
    // the U direction code from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
    //more explanation in https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // https://gamedev.stackexchange.com/questions/68612/how-to-compute-tangent-and-bitangent-vectors

    // let mut tangents = DMatrix::<f32>::zeros(verts.nrows(), 3);
    // let mut handness = DMatrix::<f32>::zeros(verts.nrows(), 1);
    // let mut bitangents = DMatrix::<f32>::zeros(verts.nrows(), 3);
    // let mut degree_vertices = vec![0; verts.nrows()];

    //convert to rowmajor for more efficient traversal
    let faces_row_major = faces.transpose(); //for more efficient row iteration
    let verts_row_major = verts.transpose(); //for more efficient row iteration
    let normals_row_major = normals.transpose(); //for more efficient row iteration
    let uvs_row_major = uvs.transpose(); //for more efficient row iteration
    let mut tangents_rm = DMatrix::<f32>::zeros(3, verts.nrows());
    let mut handness_rm = DMatrix::<f32>::zeros(1, verts.nrows());
    let mut bitangents_rm = DMatrix::<f32>::zeros(3, verts.nrows());

    //put verts and uv together so it's faster to sample
    let mut v_uv_rm = DMatrix::<f32>::zeros(3 + 2, verts.nrows());
    v_uv_rm.view_mut((0, 0), (3, verts.nrows())).copy_from(&verts_row_major);
    v_uv_rm.view_mut((3, 0), (2, verts.nrows())).copy_from(&uvs_row_major);

    for face in faces_row_major.column_iter() {
        // //increase the degree for the vertices we touch
        // degree_vertices[face[0] as usize]+=1;
        // degree_vertices[face[1] as usize]+=1;
        // degree_vertices[face[2] as usize]+=1;
        let id0 = face[0] as usize;
        let id1 = face[1] as usize;
        let id2 = face[2] as usize;

        // let v0 = verts_row_major.column(id0);
        // let v1 = verts_row_major.column(id1);
        // let v2 = verts_row_major.column(id2);

        // let uv0 = uvs_row_major.column(id0);
        // let uv1 = uvs_row_major.column(id1);
        // let uv2 = uvs_row_major.column(id2);

        //sampel vertex position and uvs
        // let v_uv0 = v_uv_rm.column(id0);
        // let v_uv1 = v_uv_rm.column(id1);
        // let v_uv2 = v_uv_rm.column(id2);
        let v_uv0 = v_uv_rm.fixed_view::<5, 1>(0, id0);
        let v_uv1 = v_uv_rm.fixed_view::<5, 1>(0, id1);
        let v_uv2 = v_uv_rm.fixed_view::<5, 1>(0, id2);

        //get vert and uv
        let v0 = v_uv0.fixed_rows::<3>(0);
        let v1 = v_uv1.fixed_rows::<3>(0);
        let v2 = v_uv2.fixed_rows::<3>(0);
        //uv
        let uv0 = v_uv0.fixed_rows::<2>(3);
        let uv1 = v_uv1.fixed_rows::<2>(3);
        let uv2 = v_uv2.fixed_rows::<2>(3);

        // Edges of the triangle : position delta
        let delta_pos1 = v1 - v0;
        let delta_pos2 = v2 - v0;

        // UV delta
        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        let r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0]);
        let tangent = (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1]) * r;
        let bitangent = (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0]) * r;

        //splat to vertices
        //it can happen that delta_uv1 and delta_uv2 is zero (in the case where there
        // is a degenerate face that has overlapping vertices) and in that case the norm
        // of the tangent would nan, so we don't splat that one
        if r.is_finite() {
            tangents_rm.column_mut(id0).add_assign(&tangent);
            tangents_rm.column_mut(id1).add_assign(&tangent);
            tangents_rm.column_mut(id2).add_assign(&tangent);
            bitangents_rm.column_mut(id0).add_assign(&bitangent);
            bitangents_rm.column_mut(id1).add_assign(&bitangent);
            bitangents_rm.column_mut(id2).add_assign(&bitangent);
        }
    }

    // for (idx, ((mut tangent,normal), bitangent)) in
    // tangents.row_iter_mut().zip(normals.row_iter()).zip(bitangents.row_iter()).
    // enumerate(){
    for (idx, ((mut tangent, normal), bitangent)) in tangents_rm
        .column_iter_mut()
        .zip(normals_row_major.column_iter())
        .zip(bitangents_rm.column_iter())
        .enumerate()
    {
        // https://foundationsofgameenginedev.com/FGED2-sample.pdf
        let dot = tangent.dot(&normal);
        // Gram-Schmidt orthogonalize
        let new_tangent = (&tangent - normal * dot).normalize();
        // Calculate handedness
        let cross_vec = tangent.cross(&bitangent);
        let dot: f32 = cross_vec.dot(&normal);
        handness_rm.column_mut(idx).fill(dot.signum()); //if >0 we write a 1.0, else we write a -1.0

        tangent.copy_from_slice(new_tangent.data.as_slice());
        // tangent[0] = new_tangent[0];
        // tangent[1] = new_tangent[1];
        // tangent[2] = new_tangent[2];
    }

    // let mut tangents_and_handness = DMatrix::<f32>::zeros(verts.nrows(), 4);
    // let mut tangents_and_handness_rm = DMatrix::<f32>::zeros(4, verts.nrows());
    // // for i in (0..verts.nrows()){
    // for ((t, h), mut out) in tangents_rm
    //     .column_iter()
    //     .zip(handness_rm.iter())
    //     .zip(tangents_and_handness_rm.column_iter_mut())
    // {
    //     out[0] = t[0];
    //     out[1] = t[1];
    //     out[2] = t[2];
    //     out[3] = *h;
    // }

    let mut tangents_and_handness_rm = DMatrix::<f32>::zeros(4, verts.nrows());
    tangents_and_handness_rm.view_mut((0, 0), (3, verts.nrows())).copy_from(&tangents_rm);
    tangents_and_handness_rm.view_mut((3, 0), (1, verts.nrows())).copy_from(&handness_rm);

    tangents_and_handness_rm.transpose()
}

#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
/// Compute tangents given verts, faces, normals and uvs for Burn Tensors
pub fn compute_tangents_burn<B: Backend>(
    verts: &Tensor<B, 2, Float>,   // Vertex positions (NUM_VERTS, 3)
    faces: &Tensor<B, 2, Int>,     // Faces (NUM_FACES, 3)
    normals: &Tensor<B, 2, Float>, // Normals (NUM_VERTS, 3)
    uvs: &Tensor<B, 2, Float>,     // UV coordinates (NUM_VERTS, 2)
) -> Tensor<B, 2, Float> {
    // We might be able to optimise a lot of stuff here
    // A lot of these operations are quite slow for Cpu backends (NdArray and
    // Candle)

    let num_faces = faces.shape().dims[0];
    let num_verts = verts.shape().dims[0];
    // let now = wasm_timer::Instant::now();

    // Flatten faces to extract indices // similar to per vert normals
    let all_idxs: Tensor<B, 1, Int> = faces.clone().flatten(0, 1);
    // println!("7.0 -- {:?}", now.elapsed());

    // Extract vertices and UVs corresponding to face indices
    let all_tris_as_verts = verts.clone().select(0, all_idxs.clone()).reshape([num_faces, 3, 3]);
    let all_tris_as_uvs = uvs.clone().select(0, all_idxs.clone()).reshape([num_faces, 3, 2]);
    // println!("7.1 -- {:?}", now.elapsed());

    let v0: Tensor<B, 2> = all_tris_as_verts.clone().slice([0..num_faces, 0..1, 0..3]).squeeze(1);
    let v1: Tensor<B, 2> = all_tris_as_verts.clone().slice([0..num_faces, 1..2, 0..3]).squeeze(1);
    let v2: Tensor<B, 2> = all_tris_as_verts.clone().slice([0..num_faces, 2..3, 0..3]).squeeze(1);
    // println!("7.2 -- {:?}", now.elapsed());

    let uv0 = all_tris_as_uvs.clone().slice([0..num_faces, 0..1, 0..2]).squeeze(1);
    let uv1 = all_tris_as_uvs.clone().slice([0..num_faces, 1..2, 0..2]).squeeze(1);
    let uv2 = all_tris_as_uvs.clone().slice([0..num_faces, 2..3, 0..2]).squeeze(1);

    let delta_pos1 = v1.clone().sub(v0.clone());
    let delta_pos2 = v2.clone().sub(v0.clone());

    let delta_uv1 = uv1.clone().sub(uv0.clone());
    let delta_uv2 = uv2.clone().sub(uv0.clone());
    // println!("7.3 -- {:?}", now.elapsed());

    let delta_uv1_0 = delta_uv1.clone().select(1, Tensor::<B, 1, Int>::from_ints([0], &verts.device())); // delta_uv1[0]
    let delta_uv1_1 = delta_uv1.clone().select(1, Tensor::<B, 1, Int>::from_ints([1], &verts.device())); // delta_uv1[1]

    let delta_uv2_0 = delta_uv2.clone().select(1, Tensor::<B, 1, Int>::from_ints([0], &verts.device())); // delta_uv2[0]
    let delta_uv2_1 = delta_uv2.clone().select(1, Tensor::<B, 1, Int>::from_ints([1], &verts.device())); // delta_uv2[1]
                                                                                                         // println!("7.4 -- {:?}", now.elapsed());

    let denominator = delta_uv1_0
        .clone()
        .mul(delta_uv2_1.clone())
        .sub(delta_uv1_1.clone().mul(delta_uv2_0.clone()));

    let r_mask = denominator.clone().abs().greater_elem(f32::EPSILON).float(); // Only consider denominators that are not too small
    let r = denominator.recip().mul(r_mask.clone()); // Apply the mask to the reciprocal
                                                     // println!("7.5 -- {:?}", now.elapsed());

    let tangent = delta_pos1
        .clone()
        .mul(delta_uv2_1.clone().repeat(&[1, 3]))
        .sub(delta_pos2.clone().mul(delta_uv1_1.clone().repeat(&[1, 3])))
        .mul(r.clone());
    // println!("7.6 -- {:?}", now.elapsed());

    let bitangent = delta_pos2
        .clone()
        .mul(delta_uv1_0.clone())
        .sub(delta_pos1.clone().mul(delta_uv2_0.clone()))
        .mul(r.clone()); // mask the inf values here itself
                         // println!("7.7 -- {:?}", now.elapsed());

    let mut tangents_rm = Tensor::<B, 2, Float>::zeros([verts.dims()[0], 3], &verts.device());
    let mut handness_rm = Tensor::<B, 2, Float>::zeros([verts.dims()[0], 1], &verts.device());
    let mut bitangents_rm = Tensor::<B, 2, Float>::zeros([verts.dims()[0], 3], &verts.device());
    let face_indices_expanded: Tensor<B, 1, Int> = faces.clone().flatten(0, 1); // Shape [num_faces * 3]
                                                                                // println!("7.8 -- {:?}", now.elapsed());

    let mut face_tangents_repeated: Tensor<B, 3> = tangent.unsqueeze_dim(1);
    face_tangents_repeated = face_tangents_repeated.repeat(&[1, 3, 1]);

    let face_tangents_to_scatter = face_tangents_repeated.reshape([num_faces * 3, 3]); // Repeat face normals for each vertex
                                                                                       // println!("7.9 -- {:?}", now.elapsed());

    tangents_rm = tangents_rm.select_assign(0, face_indices_expanded.clone(), face_tangents_to_scatter.clone());
    // println!("7.9.1 -- {:?}", now.elapsed());

    let mut face_bitangents_repeated: Tensor<B, 3> = bitangent.unsqueeze_dim(1);
    face_bitangents_repeated = face_bitangents_repeated.repeat(&[1, 3, 1]);

    let face_bitangents_to_scatter = face_bitangents_repeated.reshape([num_faces * 3, 3]); // Repeat face normals for each vertex
                                                                                           // println!("7.10 -- {:?}", now.elapsed());

    bitangents_rm = bitangents_rm.select_assign(0, face_indices_expanded, face_bitangents_to_scatter.clone());
    let dot_product = tangents_rm.clone().mul(normals.clone()).sum_dim(1); // Shape: [verts.dims()[0], 1]
                                                                           // println!("7.11 -- {:?}", now.elapsed());

    // Perform Gram-Schmidt orthogonalization: new_tangent = tangent - normal * dot
    let new_tangents = tensor::normalize_tensor(tangents_rm.clone().sub(normals.clone().mul(dot_product))); // Shape: [verts.dims()[0], 3]
    let cross_vec = tensor::cross_product(&tangents_rm, &bitangents_rm);
    let handedness_sign = cross_vec.clone().mul(normals.clone()).sum_dim(1).sign(); //.unsqueeze_dim(1); // Shape: [verts.dims()[0], 1]

    let handness_indices: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::arange(
        0..(i64::try_from(handness_rm.shape().dims[0]).expect("Dimension size exceeds i64 range")),
        &verts.device(),
    );
    handness_rm = handness_rm.select_assign(0, handness_indices.clone(), handedness_sign);
    tangents_rm = tangents_rm.slice_assign([0..num_verts, 0..3], new_tangents);
    // println!("7.12 -- {:?}", now.elapsed());

    // let mut tangents_and_handness_rm =
    //     Tensor::<B, 2, Float>::zeros([num_verts, 4], &verts.device());

    let tangent_x = tangents_rm.clone().slice([0..num_verts, 0..1]);
    let tangent_y = tangents_rm.clone().slice([0..num_verts, 1..2]);
    let tangent_z = tangents_rm.clone().slice([0..num_verts, 2..3]);

    let handness = handness_rm.clone().slice([0..num_verts, 0..1]);
    // println!("7.1 -- {:?}", now.elapsed());

    let tangents_and_handness_rm: Tensor<B, 2, Float> = Tensor::<B, 1>::stack(
        vec![tangent_x.squeeze(1), tangent_y.squeeze(1), tangent_z.squeeze(1), handness.squeeze(1)],
        1,
    );
    tangents_and_handness_rm
}

pub fn transform_verts(verts: &na::DMatrix<f32>, model_matrix: &na::SimilarityMatrix3<f32>) -> na::DMatrix<f32> {
    let mut verts_transformed = verts.clone();
    for (vert, mut vert_transformed) in verts.row_iter().zip(verts_transformed.row_iter_mut()) {
        //Need to transform it to points since vectors don't get affected by the
        // translation part
        let v_modif = model_matrix * na::Point3::from(vert.fixed_columns::<3>(0).transpose());
        vert_transformed.copy_from_slice(v_modif.coords.as_slice());
    }
    verts_transformed
}

pub fn transform_vectors(verts: &na::DMatrix<f32>, model_matrix: &na::SimilarityMatrix3<f32>) -> na::DMatrix<f32> {
    let mut verts_transformed = verts.clone();
    for (vert, mut vert_transformed) in verts.row_iter().zip(verts_transformed.row_iter_mut()) {
        //vectors don't get affected by the translation part
        let v_modif = model_matrix * vert.fixed_columns::<3>(0).transpose();
        vert_transformed.copy_from_slice(v_modif.data.as_slice());
    }
    verts_transformed
}

pub fn get_bounding_points(verts: &na::DMatrix<f32>, model_matrix: Option<na::SimilarityMatrix3<f32>>) -> (na::Point3<f32>, na::Point3<f32>) {
    let mut min_point_global = na::Point3::<f32>::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_point_global = na::Point3::<f32>::new(f32::MIN, f32::MIN, f32::MIN);

    //get min and max vertex in obj coords
    let min_coord_vec: Vec<f32> = verts.column_iter().map(|c| c.min()).collect();
    let max_coord_vec: Vec<f32> = verts.column_iter().map(|c| c.max()).collect();
    let min_point = na::Point3::<f32>::from_slice(&min_coord_vec);
    let max_point = na::Point3::<f32>::from_slice(&max_coord_vec);

    //get the points to world coords
    let (min_point_w, max_point_w) = if let Some(model_matrix) = model_matrix {
        (model_matrix * min_point, model_matrix * max_point)
    } else {
        (min_point, max_point)
    };

    //get the min/max between these points of this mesh and the global one
    min_point_global = min_point_global.inf(&min_point_w);
    max_point_global = max_point_global.sup(&max_point_w);

    (min_point_global, max_point_global)
}

pub fn get_centroid(verts: &na::DMatrix<f32>, model_matrix: Option<na::SimilarityMatrix3<f32>>) -> na::Point3<f32> {
    let (min_point_global, max_point_global) = get_bounding_points(verts, model_matrix);

    //exactly the miggle between min and max
    min_point_global.lerp(&max_point_global, 0.5)
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
pub fn sample_img_with_uvs(img: &DynImage, uvs: &na::DMatrix<f32>, is_srgb: bool) -> DMatrix<f32> {
    let mut sampled_pixels =
        unsafe { DMatrix::<core::mem::MaybeUninit<f32>>::uninit(na::Dyn(uvs.nrows()), na::Dyn(img.channels() as usize)).assume_init() };

    for (i, uv) in uvs.row_iter().enumerate() {
        let x = (uv[0] * img.width() as f32) - 0.5;
        let y = ((1.0 - uv[1]) * img.height() as f32) - 0.5;

        //TODO add also bilinear interpolation, for now we only have nearest
        let x = x.floor() as u32;
        let y = y.floor() as u32;
        let mut sample = img.get_pixel(x, y);

        if is_srgb {
            sample.apply(|v| (v / 255.0).powf(2.2) * 255.0);
        }

        sampled_pixels.row_mut(i).copy_from_slice(&sample.0[0..img.channels() as usize]);
    }

    //if the image is srgb, we first convert to linear space before we store it or
    // before we do any blending sampled_pixels = sampled_pixels.map(|v|
    // v.powf(2.2));

    sampled_pixels
}

/// returns the rows of mat where mask==keep
/// returns the filtered rows and also a matrix of orig2filtered and
/// filtered2orig which maps from original row indices to filtered ones and
/// viceversa (-1 denotes a invalid index)
pub fn filter_rows<T: na::Scalar>(mat: &na::DMatrix<T>, mask: &[bool], keep: bool) -> (DMatrix<T>, Vec<i32>, Vec<i32>) {
    assert_eq!(
        mat.nrows(),
        mask.len(),
        "Mat and mask need to have the same nr of rows. Mat has nr rows{} while mask have length {}",
        mat.nrows(),
        mask.len()
    );

    //figure out how many rows we need
    let nr_filtered_rows: u32 = mask.iter().map(|v| u32::from(*v == keep)).sum();

    let mut selected =
        unsafe { DMatrix::<core::mem::MaybeUninit<T>>::uninit(na::Dyn(nr_filtered_rows as usize), na::Dyn(mat.ncols())).assume_init() };

    //indirections
    let mut orig2filtered: Vec<i32> = vec![-1; mat.nrows()];
    let mut filtered2orig: Vec<i32> = vec![-1; nr_filtered_rows as usize];

    let mut idx_filtered = 0;
    for (idx_orig, (val, mask_val)) in izip!(mat.row_iter(), mask.iter()).enumerate() {
        if *mask_val == keep {
            selected.row_mut(idx_filtered).copy_from(&val);

            //indirections
            orig2filtered[idx_orig] = i32::try_from(idx_filtered).unwrap();
            filtered2orig[idx_filtered] = i32::try_from(idx_orig).unwrap();

            idx_filtered += 1;
        }
    }

    (selected, orig2filtered, filtered2orig)
}

/// returns the cols of mat where mask==keep
/// returns the filtered cols and also a matrix of orig2filtered and
/// filtered2orig which maps from original row indices to filtered ones and
/// viceversa (-1 denotes a invalid index)
pub fn filter_cols<T: na::Scalar>(mat: &na::DMatrix<T>, mask: &[bool], keep: bool) -> (DMatrix<T>, Vec<i32>, Vec<i32>) {
    assert_eq!(
        mat.ncols(),
        mask.len(),
        "Mat and mask need to have the same nr of cols. Mat has nr cols{} while mask have length {}",
        mat.ncols(),
        mask.len()
    );

    //figure out how many rows we need
    let nr_filtered_cols: u32 = mask.iter().map(|v| u32::from(*v == keep)).sum();

    let mut selected =
        unsafe { DMatrix::<core::mem::MaybeUninit<T>>::uninit(na::Dyn(mat.nrows()), na::Dyn(nr_filtered_cols as usize)).assume_init() };

    //indirections
    let mut orig2filtered: Vec<i32> = vec![-1; mat.ncols()];
    let mut filtered2orig: Vec<i32> = vec![-1; nr_filtered_cols as usize];

    let mut idx_filtered = 0;
    for (idx_orig, (val, mask_val)) in izip!(mat.column_iter(), mask.iter()).enumerate() {
        if *mask_val == keep {
            selected.column_mut(idx_filtered).copy_from(&val);

            //indirections
            orig2filtered[idx_orig] = i32::try_from(idx_filtered).unwrap();
            filtered2orig[idx_filtered] = i32::try_from(idx_orig).unwrap();

            idx_filtered += 1;
        }
    }

    (selected, orig2filtered, filtered2orig)
}

/// Gets rows of mat and splats them according to `indices_orig2splatted`
/// such that:
/// ```ignore
/// vec = mat[i];
/// idx_destination = indices_orig2splatted[i];
/// splatted.row(idx_destination) += vec
/// ```
pub fn splat_rows(mat: &na::DMatrix<f32>, indices_orig2splatted: &[u32], splat_type: &SplatType) -> DMatrix<f32> {
    assert_eq!(
        mat.nrows(),
        indices_orig2splatted.len(),
        "Mat and indices need to have the same nr of rows. Mat has nr rows{} while indices have length {}",
        mat.nrows(),
        indices_orig2splatted.len()
    );

    // let mut indices_sorted = indices_orig2splatted.to_vec();
    // indices_sorted.sort_unstable();
    // indices_sorted.dedup();
    // let nr_splatted_rows = indices_sorted.len();
    let max_idx_splatted = *indices_orig2splatted.iter().max().unwrap() as usize;

    let mut splatted = na::DMatrix::zeros(max_idx_splatted + 1, mat.ncols());
    let mut normalization: Vec<f32> = vec![0.0; max_idx_splatted + 1];

    for (val, idx_destin) in izip!(mat.row_iter(), indices_orig2splatted.iter()) {
        let mut splatted_row = splatted.row_mut(*idx_destin as usize);
        splatted_row.add_assign(val);

        #[allow(clippy::single_match)]
        match splat_type {
            SplatType::Avg => {
                normalization[*idx_destin as usize] += 1.0;
            }
            SplatType::Sum => {}
        }
    }

    //renormalize
    match splat_type {
        SplatType::Avg => {
            for (mut splatted_row, normalization_row) in izip!(splatted.row_iter_mut(), normalization.iter()) {
                for e in splatted_row.iter_mut() {
                    *e /= normalization_row;
                }
            }
        }
        SplatType::Sum => {}
    }

    splatted
}

pub fn apply_indirection(mat: &na::DMatrix<u32>, indices_orig2destin: &[i32], removal_policy: &IndirRemovalPolicy) -> (DMatrix<u32>, Vec<bool>) {
    //applies the indices_orig2destin to mat. The values that are invalid are
    // mapped to -1
    let reindexed: DMatrix<i32> = DMatrix::from_iterator(
        mat.nrows(),
        mat.ncols(),
        mat.iter().map(|v| indices_orig2destin.get(*v as usize).map_or(-1, |v| *v)),
    );

    //remove rows or cols that have any -1
    let (reindexed_only_valid, mask) = match removal_policy {
        IndirRemovalPolicy::RemoveInvalidRows => {
            let mut valid_rows = vec![true; mat.nrows()];
            reindexed
                .row_iter()
                .enumerate()
                .filter(|(_, r)| r.iter().any(|x| *x == -1))
                .for_each(|(idx, _)| valid_rows[idx] = false);
            let (filtered, _, _) = filter_rows(&reindexed, &valid_rows, true);
            (filtered, valid_rows)
        }
        IndirRemovalPolicy::RemoveInvalidCols => {
            let mut valid_cols = vec![true; mat.ncols()];
            reindexed
                .column_iter()
                .enumerate()
                .filter(|(_, c)| c.iter().any(|x| *x == -1))
                .for_each(|(idx, _)| valid_cols[idx] = false);
            let (filtered, _, _) = filter_cols(&reindexed, &valid_cols, true);
            (filtered, valid_cols)
        }
    };

    let reindexed_filtered = reindexed_only_valid.try_cast::<u32>().unwrap();

    (reindexed_filtered, mask)
}

pub fn compute_dummy_uvs(nr_verts: usize) -> DMatrix<f32> {
    DMatrix::<f32>::zeros(nr_verts, 2)
}
pub fn compute_dummy_colors(nr_verts: usize) -> DMatrix<f32> {
    DMatrix::<f32>::zeros(nr_verts, 3)
}
pub fn compute_dummy_tangents(nr_verts: usize) -> DMatrix<f32> {
    DMatrix::<f32>::zeros(nr_verts, 4) //make it 4 because it's both the tangent and the handness as the last element
}

pub fn create_frustum_verts_and_edges(
    extrinsics: &na::Matrix4<f32>,
    fovy: f32,
    aspect_ratio: f32,
    near: f32, // this is for  the camera frustum not viewing frustum
    far: f32,
) -> (DMatrix<f32>, DMatrix<u32>) {
    // Extract the rotation and translation from the extrinsics matrix
    let rot = extrinsics.fixed_view::<3, 3>(0, 0);
    let trans = extrinsics.fixed_view::<3, 1>(0, 3);

    // Calculate the camera's position (lookfrom) in world space
    let lookfrom = -rot.transpose() * trans;

    let forward = -rot.column(2);
    let right = rot.column(0);
    let up = rot.column(1);

    // Frustum dimensions at near and far planes
    let near_height = 2.0 * (fovy / 2.0).tan() * near;
    let near_width = near_height * aspect_ratio;

    let far_height = 2.0 * (fovy / 2.0).tan() * far;
    let far_width = far_height * aspect_ratio;

    // Calculate the corners of the near and far planes
    let near_center = lookfrom + forward * near;
    let near_top_left = near_center + (up * (near_height / 2.0)) - (right * (near_width / 2.0));
    let near_top_right = near_center + (up * (near_height / 2.0)) + (right * (near_width / 2.0));
    let near_bottom_left = near_center - (up * (near_height / 2.0)) - (right * (near_width / 2.0));
    let near_bottom_right = near_center - (up * (near_height / 2.0)) + (right * (near_width / 2.0));

    let far_center = lookfrom + forward * far;
    let far_top_left = far_center + (up * (far_height / 2.0)) - (right * (far_width / 2.0));
    let far_top_right = far_center + (up * (far_height / 2.0)) + (right * (far_width / 2.0));
    let far_bottom_left = far_center - (up * (far_height / 2.0)) - (right * (far_width / 2.0));
    let far_bottom_right = far_center - (up * (far_height / 2.0)) + (right * (far_width / 2.0));

    // Create the vertices and edges matrices
    let verts = DMatrix::from_row_slice(
        8,
        3,
        &[
            near_top_left.x,
            near_top_left.y,
            near_top_left.z,
            near_top_right.x,
            near_top_right.y,
            near_top_right.z,
            near_bottom_left.x,
            near_bottom_left.y,
            near_bottom_left.z,
            near_bottom_right.x,
            near_bottom_right.y,
            near_bottom_right.z,
            far_top_left.x,
            far_top_left.y,
            far_top_left.z,
            far_top_right.x,
            far_top_right.y,
            far_top_right.z,
            far_bottom_left.x,
            far_bottom_left.y,
            far_bottom_left.z,
            far_bottom_right.x,
            far_bottom_right.y,
            far_bottom_right.z,
        ],
    );

    let edges = DMatrix::from_row_slice(12, 2, &[0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7]);

    (verts, edges)
}
