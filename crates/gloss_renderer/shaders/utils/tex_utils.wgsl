//some textures on wgpu (like 32float textures) you cannot sample bilinearly so we do 4 samples with nearest sampling and blend them manually
// https://community.khronos.org/t/manual-bilinear-filter/58504/8
// sampler should be one of type NEAREST
fn texture_sample_bilinear(tex: texture_2d<f32>, uv_in: vec2<f32>, sampler_nearest_obj: sampler) -> vec4<f32>{
    let textureSize = vec2<f32>(textureDimensions(tex));
    let texelSize = 1.0/textureSize;
    let f = fract( uv_in * textureSize );
    let uv = uv_in+ ( .5 - f ) * texelSize;    // move uv to texel centre
    let tl = textureSample(tex, sampler_nearest_obj, uv);
    let tr = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(texelSize.x, 0.0));
    let bl = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(0.0, texelSize.y));
    let br = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(texelSize.x, texelSize.y));
    let tA = mix( tl, tr, f.x );
    let tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}


//depth textures cannot be sampled bilinearly so for pcf we can do 4 samples with nearest sample and compare the 4 samples with the reference z value
// sampler should be one of type NEAREST
fn texture_compare_bilinear(tex: texture_2d<f32>, uv_in: vec2<f32>, reference_z: f32, sampler_nearest_obj: sampler) -> f32{
    let textureSize = vec2<f32>(textureDimensions(tex));
    let texelSize = 1.0/textureSize;
    let f = fract( uv_in * textureSize );
    let uv = uv_in+ ( .5 - f ) * texelSize;    // move uv to texel centre
    var tl = textureSample(tex, sampler_nearest_obj, uv).x;
    var tr = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(texelSize.x, 0.0)).x;
    var bl = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(0.0, texelSize.y)).x;
    var br = textureSample(tex, sampler_nearest_obj, uv + vec2<f32>(texelSize.x, texelSize.y)).x;
    //compare
    tl = select(0.0, 1.0, reference_z<tl);
    tr = select(0.0, 1.0, reference_z<tr);
    bl = select(0.0, 1.0, reference_z<bl);
    br = select(0.0, 1.0, reference_z<br);
    
    let tA = mix( tl, tr, f.x );
    let tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}


