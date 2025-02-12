// Do the lookup, using HW 2x2 PCF and comparison
fn sample_shadow_map_hardware(light_local: vec2<f32>, shadow_map: texture_depth_2d, depth: f32, sampler_compare_obj: sampler_comparison) -> f32 {
    return textureSampleCompare(
        shadow_map,
        sampler_compare_obj,
        light_local,
        depth,
    );
}

//does pcf kernel of 3x3
// fn fetch_shadow_pcf_3x3(pos_light_space: vec4<f32>, shadow_map: texture_depth_2d, sampler_comparison_obj: sampler_comparison) -> f32 {
    
//     // compensate for the Y-flip difference between the NDC and texture coordinates
//     let flip_correction = vec2<f32>(0.5, -0.5);
//     // compute texture coordinates for shadow lookup
//     let proj_correction = 1.0 / pos_light_space.w;
//     let shadow_map_coords = pos_light_space.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
//     // do the lookup, using HW PCF and comparison
//     // return textureSampleCompareLevel(t_shadow, sampler_shadow, shadow_map_coords, i32(light_id), homogeneous_coords.z * proj_correction);
//     let epsilon = 2e-6;

//     let cur_frag_z = pos_light_space.z * proj_correction - epsilon;

//     //PCF
//     var visibility = 0.0;
//     let shadowDepthTextureSize =  f32(textureDimensions(shadow_map).x); //TODO do not hardcode this
//     let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
//     for (var y = -1; y <= 1; y++) {
//         for (var x = -1; x <= 1; x++) {
//             let offset = vec2<f32>(vec2(x, y)) * oneOverShadowDepthTextureSize;
//             // visibility+=TexUtils::texture_compare_bilinear(shadow_map, shadow_map_coords+offset, cur_frag_z, sampler_nearest_obj);
//             visibility+=sample_shadow_map_hardware(shadow_map, shadow_map_coords+offset, cur_frag_z, sampler_comparison_obj);
//         }
//     }
//     visibility /= 9.0;


//     //if we are behind the light we are visible 
//     if (pos_light_space.w <= 0.0) {
//         visibility=1.0;
//     }

//     //if we are outside the bounds of the lights
//     if shadow_map_coords.x < 0.0 || shadow_map_coords.y < 0.0 || shadow_map_coords.x > 1.0 || shadow_map_coords.y > 1.0 {
//         visibility=1.0;
//     }

//     return visibility;
// }

fn sample_shadow_map_pcf_3x3(light_local: vec2<f32>, shadow_map: texture_depth_2d, depth: f32, sampler_compare_obj: sampler_comparison) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let inv_shadow_map_size = 1.0 / shadow_map_size;

    var visibility = 0.0;
    let shadowDepthTextureSize =  f32(textureDimensions(shadow_map).x); //TODO do not hardcode this
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(vec2(x, y)) * inv_shadow_map_size;
            visibility+=sample_shadow_map_hardware(light_local+offset, shadow_map, depth, sampler_compare_obj);
        }
    }
    visibility /= 9.0;
  
    return visibility;
}


// https://web.archive.org/web/20230210095515/http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1
fn sample_shadow_map_castano_thirteen(light_local: vec2<f32>, shadow_map: texture_depth_2d, depth: f32, sampler_compare_obj: sampler_comparison) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let inv_shadow_map_size = 1.0 / shadow_map_size;

    let uv = light_local * shadow_map_size;
    var base_uv = floor(uv + 0.5);
    let s = (uv.x + 0.5 - base_uv.x);
    let t = (uv.y + 0.5 - base_uv.y);
    base_uv -= 0.5;
    base_uv *= inv_shadow_map_size;

    let uw0 = (4.0 - 3.0 * s);
    let uw1 = 7.0;
    let uw2 = (1.0 + 3.0 * s);

    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;

    let vw0 = (4.0 - 3.0 * t);
    let vw1 = 7.0;
    let vw2 = (1.0 + 3.0 * t);

    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    var sum = 0.0;

    sum += uw0 * vw0 * sample_shadow_map_hardware(base_uv + (vec2(u0, v0) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw1 * vw0 * sample_shadow_map_hardware(base_uv + (vec2(u1, v0) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw2 * vw0 * sample_shadow_map_hardware(base_uv + (vec2(u2, v0) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);

    sum += uw0 * vw1 * sample_shadow_map_hardware(base_uv + (vec2(u0, v1) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw1 * vw1 * sample_shadow_map_hardware(base_uv + (vec2(u1, v1) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw2 * vw1 * sample_shadow_map_hardware(base_uv + (vec2(u2, v1) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);

    sum += uw0 * vw2 * sample_shadow_map_hardware(base_uv + (vec2(u0, v2) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw1 * vw2 * sample_shadow_map_hardware(base_uv + (vec2(u1, v2) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);
    sum += uw2 * vw2 * sample_shadow_map_hardware(base_uv + (vec2(u2, v2) * inv_shadow_map_size), shadow_map, depth, sampler_compare_obj);

    // //if we are outside the bounds of the lights
    // if base_uv.x < 0.0 || base_uv.y < 0.0 || base_uv.x > 1.0 || base_uv.y > 1.0 {
    //     sum=144.0;
    // }

    return sum * (1.0 / 144.0);
}