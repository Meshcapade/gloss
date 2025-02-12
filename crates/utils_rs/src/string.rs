//formats f32 into string with specified precision
// https://stackoverflow.com/a/61954174
pub fn float2string(val: f32, precision: usize) -> String {
    format!("{val:.precision$}")
}
