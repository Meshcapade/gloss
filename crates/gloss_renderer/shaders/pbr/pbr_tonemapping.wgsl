fn tone_mapping(in: vec4<f32>, color_grading: ColorGrading) -> vec4<f32> {
    var color = max(in.rgb, vec3(0.0));

    // Possible future grading:

    // highlight gain gamma: 0..
    // let luma = powsafe(vec3(tonemapping_luminance(color)), 1.0); 

    // highlight gain: 0.. 
    // color += color * luma.xxx * 1.0; 

    // Linear pre tonemapping grading
    color = saturation(color, color_grading.pre_saturation);
    color = powsafe(color, color_grading.gamma);
    color = color * powsafe(vec3(2.0), color_grading.exposure);
    color = max(color, vec3(0.0));

    // tone_mapping
    color = ACESFitted(color.rgb);

    // Perceptual post tonemapping grading
    color = saturation(color, color_grading.post_saturation);
    
    return vec4(color, in.a);
}

// This is an **incredibly crude** approximation of the inverse of the tone mapping function.
// We assume here that there's a simple linear relationship between the input and output
// which is not true at all, but useful to at least preserve the overall luminance of colors
// when sampling from an already tonemapped image. (e.g. for transmissive materials when HDR is off)
fn approximate_inverse_tone_mapping(in: vec4<f32>, color_grading: ColorGrading) -> vec4<f32> {
    let out = tone_mapping(in, color_grading);
    let approximate_ratio = length(in.rgb) / length(out.rgb);
    return vec4(in.rgb * approximate_ratio, in.a);
}