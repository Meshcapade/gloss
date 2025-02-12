pub fn align(size: u32, alignment: u32) -> u32 {
    ((size + alignment - 1) / alignment) * alignment
}

pub fn align_usz(size: usize, alignment: usize) -> usize {
    ((size + alignment - 1) / alignment) * alignment
}

// https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
//transforms a reverse-z depth buffer to linear depth
pub fn linearize_depth_reverse_z(d: f32, near: f32, far: f32) -> f32 {
    if d <= 0.0 {
        0.0
    } else {
        (near * far) / (d * (far - near) + near)
    }
}

//https://stackoverflow.com/a/77388975
pub fn u8_to_f32_vec(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(TryInto::try_into)
        .map(Result::unwrap)
        .map(f32::from_le_bytes)
        .collect()
}
