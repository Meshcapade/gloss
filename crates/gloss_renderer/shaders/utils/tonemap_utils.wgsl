// ---------------------------------
// ---------- ACES Fitted ----------
// ---------------------------------
// https://github.com/bevyengine/bevy/blob/1a7fc57a8c8f6611353048a3031c97bcfafa1efa/crates/bevy_core_pipeline/src/tonemapping/tonemapping_shared.wgsl#L305

// Same base implementation that Godot 4.0 uses for Tonemap ACES.

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

// The code in this file was originally written by Stephen Hill (@self_shadow), who deserves all
// credit for coming up with this fit and implementing it. Buy him a beer next time you see him. :)

fn RRTAndODTFit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn ACESFitted(color: vec3<f32>) -> vec3<f32> {    
    var fitted_color = color;

    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    let rgb_to_rrt = mat3x3<f32>(
        vec3(0.59719, 0.35458, 0.04823),
        vec3(0.07600, 0.90834, 0.01566),
        vec3(0.02840, 0.13383, 0.83777)    
    );

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    let odt_to_rgb = mat3x3<f32>(
        vec3(1.60475, -0.53108, -0.07367),
        vec3(-0.10208, 1.10813, -0.00605),
        vec3(-0.00327, -0.07276, 1.07602)
    );

    fitted_color *= rgb_to_rrt;

    // Apply RRT and ODT
    fitted_color = RRTAndODTFit(fitted_color);

    fitted_color *= odt_to_rgb;

    // Clamp to [0, 1]
    fitted_color = saturate(fitted_color);

    return fitted_color;
}


// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn tonemapping_luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// pow() but safe for NaNs/negatives
fn powsafe(color: vec3<f32>, power: f32) -> vec3<f32> {
    return pow(abs(color), vec3(power)) * sign(color);
}



/*
    Increase color saturation of the given color data.
    :param color: expected sRGB primaries input
    :param saturationAmount: expected 0-1 range with 1=neutral, 0=no saturation.
    -- ref[2] [4]
*/
fn saturation(color: vec3<f32>, saturationAmount: f32) -> vec3<f32> {
    let luma = tonemapping_luminance(color);
    return mix(vec3(luma), color, vec3(saturationAmount));
}