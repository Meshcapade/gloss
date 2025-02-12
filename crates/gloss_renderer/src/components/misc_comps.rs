/// Each entity is identifies with an unique name. This component stores this
/// name.
pub struct Name(pub String);

/// Faces from the obj file are reindexed when reading them
/// For example if the first two faces are [4,2,6] and [4,3,7]
/// they will be stored inside the mesh.indices as [0,1,2] and [0,2,3]
/// To obtain back the original index we use this `face_reindex` which maps from
/// final_index->original_index Therefore `faces_original_index`[0]=4 and
/// `faces_original_index`[1]=2
pub struct FacesOriginalIndex(pub Vec<u32>);
