#import ../types/pbr_types.wgsl as PbrTypes
#import ../bindings/global_binds.wgsl as GlobalBinds

// // Creates the deferred gbuffer from a PbrInput.
// fn deferred_gbuffer_from_pbr_input(in: PbrInput) -> vec4<u32> {
//      // Only monochrome occlusion supported. May not be worth including at all.
//      // Some models have baked occlusion, GLTF only supports monochrome. 
//      // Real time occlusion is applied in the deferred lighting pass.
//      // Deriving luminance via Rec. 709. coefficients
//      // https://en.wikipedia.org/wiki/Rec._709
//     let occlusion = dot(in.occlusion, vec3<f32>(0.2126, 0.7152, 0.0722));
// #ifdef WEBGL2 // More crunched for webgl so we can also fit depth.
//     var props = deferred_types::pack_unorm3x4_plus_unorm_20_(vec4(
//         in.material.reflectance,
//         in.material.metallic,
//         occlusion, 
//         in.frag_coord.z));
// #else
//     var props = deferred_types::pack_unorm4x8_(vec4(
//         in.material.reflectance, // could be fewer bits
//         in.material.metallic, // could be fewer bits
//         occlusion, // is this worth including?
//         0.0)); // spare
// #endif // WEBGL2
//     let flags = deferred_types::deferred_flags_from_mesh_material_flags(in.flags, in.material.flags);
//     let octahedral_normal = octahedral_encode(normalize(in.N));
//     var base_color_srgb = vec3(0.0);
//     var emissive = in.material.emissive.rgb;
//     if ((in.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) {
//         // Material is unlit, use emissive component of gbuffer for color data.
//         // Unlit materials are effectively emissive.
//         emissive = in.material.base_color.rgb;
//     } else {
//         base_color_srgb = pow(in.material.base_color.rgb, vec3(1.0 / 2.2));
//     }
//     let deferred = vec4(
//         deferred_types::pack_unorm4x8_(vec4(base_color_srgb, in.material.perceptual_roughness)),
//         rgb9e5::vec3_to_rgb9e5_(emissive),
//         props,
//         deferred_types::pack_24bit_normal_and_flags(octahedral_normal, flags),
//     );
//     return deferred;
// }

// Creates a PbrInput from the deferred gbuffer.
fn pbr_input_from_deferred_gbuffer(uv: vec2<f32>, g_albedo: texture_2d<f32>,  g_position: texture_2d<f32>, g_normal: texture_2d<f32>, g_metalness_roughness: texture_2d<f32>, g_depth: texture_2d<f32>) -> PbrTypes::PbrInput {
    var pbr: PbrTypes::PbrInput;
    pbr.material = PbrTypes::standard_material_new();

    //sample gbuffer
    let albedo = textureSample(g_albedo, GlobalBinds::sampler_linear, uv).xyz;
    let pos_world = textureSample(g_position, GlobalBinds::sampler_nearest, uv).xyz;
    let n_world = normalize(textureSample(g_normal, GlobalBinds::sampler_nearest, uv).xyz);
    let metalness_perceptual_roughness = textureSample(g_metalness_roughness, GlobalBinds::sampler_nearest, uv).xy;
    let metalness = metalness_perceptual_roughness.x;
    let perceptual_roughness = metalness_perceptual_roughness.y;
    let depth = textureSample(g_depth, GlobalBinds::sampler_nearest, uv).x;

    let V = normalize(GlobalBinds::camera.pos_world - pos_world); //TOOD this would need to change if the projection is orthographic

    pbr.world_position=vec4<f32>(pos_world, 1.0);

//     let flags = deferred_types::unpack_flags(gbuffer.a);
//     let deferred_flags = deferred_types::mesh_material_flags_from_deferred_flags(flags);
    pbr.flags = 0u;
    pbr.material.flags = 0u;

//     let base_rough = deferred_types::unpack_unorm4x8_(gbuffer.r);
    pbr.material.perceptual_roughness = perceptual_roughness;
//     let emissive = rgb9e5::rgb9e5_to_vec3_(gbuffer.g);
//     if ((pbr.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) {
    pbr.material.base_color = vec4(albedo, 1.0);
    pbr.material.emissive = vec4(vec3(0.0), 1.0);
//     } else {
//         pbr.material.base_color = vec4(pow(base_rough.rgb, vec3(2.2)), 1.0);
//         pbr.material.emissive = vec4(emissive, 1.0);
//     }
// #ifdef WEBGL2 // More crunched for webgl so we can also fit depth.
//     let props = deferred_types::unpack_unorm3x4_plus_unorm_20_(gbuffer.b);
//     // Bias to 0.5 since that's the value for almost all materials.
    pbr.material.reflectance = 0.5; 
// #else
//     let props = deferred_types::unpack_unorm4x8_(gbuffer.b);
//     pbr.material.reflectance = props.r;
// #endif // WEBGL2
    pbr.material.metallic = metalness;
    pbr.occlusion = vec3(1.0);
//     let octahedral_normal = deferred_types::unpack_24bit_normal(gbuffer.a);
//     let N = octahedral_decode(octahedral_normal);

//     let world_position = vec4(position_ndc_to_world(frag_coord_to_ndc(frag_coord)), 1.0);
//     let is_orthographic = view.projection[3].w == 1.0;
//     let V = pbr_functions::calculate_view(world_position, is_orthographic);
    
    // pbr.frag_coord = frag_coord;
//     pbr.world_normal = N; 
//     pbr.world_position = world_position;
    pbr.N = n_world;
    pbr.V = V;
    pbr.is_orthographic = false;

    return pbr;
}

