//map a value from the range [inMin, inMax] to [outMin, outMax]
fn map(value: f32, inMin: f32, inMax: f32, outMin: f32, outMax: f32) -> f32 {
    let value_clamped=clamp(value, inMin, inMax); //so the value doesn't get modified by the clamping, because glsl may pass this by referece
    return outMin + (outMax - outMin) * (value_clamped - inMin) / (inMax - inMin);
}

//https://stackoverflow.com/a/51137756
fn linearize_depth(d: f32, z_near: f32, z_far: f32) -> f32{
    return z_near * z_far / (z_far + d * (z_near - z_far));
}

fn lineardepth_to_nonlinear(lin_depth: f32, z_near: f32, z_far: f32) -> f32{
    return (z_near * z_far / lin_depth  - z_far) / (z_near - z_far);
}

//for reverse Z case
//https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
//transforms a reverse-z depth buffer to linear depth
fn linearize_depth_reverse_z(d: f32, z_near: f32, z_far: f32) -> f32{
    return z_near * z_far / (z_near + d * (z_far - z_near));
}
fn lineardepth_to_nonlinear_reverse_z(lin_depth: f32, z_near: f32, z_far: f32) -> f32{
    return (z_near * z_far / lin_depth  - z_near) / (z_far - z_near);
}


// Ken Perlin suggests an improved version of the smoothstep() function, 
// which has zero 1st- and 2nd-order derivatives at x = 0 and x = 1.
fn smootherstep( low: f32, high: f32, val: f32 ) -> f32{
    let t = map(val, low , high, 0.0, 1.0);
    return t * t * t * ( t * ( t * 6.0 - 15.0 ) + 10.0 );
}

