//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 1, binding 0 = albedo 
//group 1, binding 1 = position
//group 1, binding 2 = normal
//group 1, binding 3 = depth
//group 1, binding 4 = shadow_map_0
//group 1, binding 5 = shadow_map_1
#import ./types/global_types.wgsl as GlobalTypes
#import ./bindings/global_binds.wgsl as GlobalBinds
#import ./utils/num_utils.wgsl as NumUtils
#import ./utils/tex_utils.wgsl as TexUtils
#import ./utils/shadows.wgsl as Shadows
#import ./utils/tonemap_utils.wgsl as TonemapUtils
#import ./utils/full_screen_tri_utils.wgsl as Tri
#import ./pbr/pbr_deferred_functions.wgsl as PbrDefferedFunc
#import ./pbr/pbr_functions.wgsl as PbrFunc


@group(1) @binding(0) var g_albedo: texture_2d<f32>;
@group(1) @binding(1) var g_position: texture_2d<f32>;
@group(1) @binding(2) var g_normal: texture_2d<f32>;
@group(1) @binding(3) var g_metalness_roughness: texture_2d<f32>;
@group(1) @binding(4) var g_depth: texture_2d<f32>;
#import ./bindings/compose_binds.wgsl as ComposeBinds //contains the shadow maps textures
// @group(1) @binding(5) var shadow_map_0: texture_depth_2d;
// @group(1) @binding(6) var shadow_map_1: texture_depth_2d;
// @group(1) @binding(7) var shadow_map_2: texture_depth_2d;
// @group(1) @binding(8) var shadow_map_3: texture_depth_2d;
// @group(1) @binding(9) var shadow_map_4: texture_depth_2d;
// @group(1) @binding(10) var shadow_map_5: texture_depth_2d;
// @group(1) @binding(11) var shadow_map_6: texture_depth_2d;
// @group(1) @binding(12) var shadow_map_7: texture_depth_2d;



struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) view_dir_world: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let pos_uv = Tri::full_screen_tri(vertex_index);

    //attempt 2 based on https://community.khronos.org/t/compute-fragments-ray-direction-worlds-coordinates/68424/2
    var view_ray = vec4<f32>(pos_uv.pos.xy, 0.0, 1.0);
    view_ray = GlobalBinds::camera.proj_inv * view_ray;
    let view_ray_out = view_ray.xyz;
    //view ray that rotates when the camera rotates. This view rays are in the world coordinates and not in the cam_coords like the view_ray_out
    let v_inv = GlobalBinds::camera.view_inv;
    let v_inv_rot = mat3x3<f32> ( v_inv[0].xyz, v_inv[1].xyz, v_inv[2].xyz);
    let view_dir_world = v_inv_rot * view_ray.xyz;
    out.view_dir_world=view_dir_world;

    out.position=pos_uv.pos;
    out.tex_coords=pos_uv.uv;
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(g_albedo, GlobalBinds::sampler_linear, in.tex_coords);
    let pos_world = textureSample(g_position, GlobalBinds::sampler_nearest, in.tex_coords).xyz;
    let n_world = normalize(textureSample(g_normal, GlobalBinds::sampler_nearest, in.tex_coords).xyz);
    let metalness_perceptual_roughness = textureSample(g_metalness_roughness, GlobalBinds::sampler_nearest, in.tex_coords).xy;
    let metalness = metalness_perceptual_roughness.x;
    let perceptual_roughness = metalness_perceptual_roughness.y;
    let depth = textureSample(g_depth, GlobalBinds::sampler_nearest, in.tex_coords).x;

    //new pbr
    // var pbr_input = pbr_input_from_standard_material(in, is_front); //pbr_fragment::pbr_input_from_standard_material
    // alpha discard
    // pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    // out.color = apply_pbr_lighting(pbr_input);  //pbr_functions::{apply_pbr_lighting, 

    //https://webgpu.github.io/webgpu-samples/samples/deferredRendering#fragmentDeferredRendering.wgsl
//     fn world_from_screen_coord(coord : vec2<f32>, depth_sample: f32) -> vec3<f32> {
//   // reconstruct world-space position from the screen coordinate.
//   let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
//   let posWorldW = camera.invViewProjectionMatrix * posClip;
//   let posWorld = posWorldW.xyz / posWorldW.www;
//   return posWorld;
// }


    //attempt 2 pbr
    var color_linear = vec4<f32>(0.0);
    // let irradiance = textureSample(ComposeBinds::environment_map_specular, GlobalBinds::sampler_linear, vec3(in.view_dir_world.xy, -in.view_dir_world.z)).rgb;
    if depth==1.0{
        color_linear = GlobalBinds::params.bg_color;
        // color_linear = vec4<f32>(irradiance, 1.0);
    }else{
        var pbr_input = PbrDefferedFunc::pbr_input_from_deferred_gbuffer(in.tex_coords, g_albedo, g_position, g_normal, g_metalness_roughness, g_depth);
        color_linear = PbrFunc::apply_pbr_lighting(pbr_input);
    }
    

    if(GlobalBinds::params.enable_distance_fade>0u){
        //get fog factor
        let dist_center = length(pos_world.xz); //only take the horizontal plane so xz
        var fog_factor = NumUtils::map(dist_center, GlobalBinds::params.distance_fade_start, GlobalBinds::params.distance_fade_end, 0.0, 1.0);
        fog_factor = num_utils::smootherstep(0.0, 1.0, fog_factor);
        //blend
        color_linear = color_linear*(1.0-fog_factor) + GlobalBinds::params.bg_color*fog_factor;
    }


    // Linear pre tonemapping grading
    var color = max(color_linear.rgb, vec3(0.0));
    color = TonemapUtils::saturation(color, GlobalBinds::params.saturation);
    color = TonemapUtils::powsafe(color, GlobalBinds::params.gamma);
    color = color * TonemapUtils::powsafe(vec3(2.0), GlobalBinds::params.exposure);
    color = max(color, vec3(0.0));
    color_linear = vec4<f32>(color, color_linear.a);

    //tonemap as the last step!
    var color_tonemapped = TonemapUtils::ACESFitted(color_linear.rgb);

    // Perceptual post tonemapping grading
    // color_tonemapped = TonemapUtils::saturation(color_tonemapped.rgb, GlobalBinds::params.post_saturation).rgb;

    let color_tonemapped_gamma = pow(color_tonemapped.xyz, vec3<f32>(1.0/2.2)); //gamma correction
    let color_tonemapped_gamma_rgba = vec4<f32>(color_tonemapped_gamma,1.0);

    return color_tonemapped_gamma_rgba;




    // var color_linear = vec3<f32>(0.0);

    // //if depth==1.0 there is no mesh or anything covering this pixel, so it will show whatever the background was set to
    // if depth==1.0{
    //     color_linear = GlobalBinds::params.bg_color.rgb;
    // }else{
    //     //ambient
    //     let ambient = albedo.xyz*GlobalBinds::params.ambient_factor;

    //     let view_dir = normalize(GlobalBinds::camera.pos_world - pos_world.xyz);
    //     color_linear += ambient; 
    //     for (var l_idx: u32 = 0u; l_idx < GlobalBinds::scene.nr_lights; l_idx++) {
    //         //get data for this light
    //         let light = GlobalBinds::lights[l_idx];
    //         //calculate shadow
    //         // let pos_light_space = light.proj * light.view * vec4<f32>(pos_world.xyz, 1.0);
    
    //         var shadow = 1.0;
    //         let is_shadow_caster: bool = light.is_shadow_caster > 0u;
    //         if l_idx==0u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_0, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==1u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_1, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==2u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_2, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==3u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_3, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==4u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_4, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==5u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_5, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==6u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_6, GlobalBinds::sampler_shadow_map);}
    //         else if l_idx==7u && is_shadow_caster { shadow=Shadows::fetch_shadow(light, pos_world, shadow_map_7, GlobalBinds::sampler_shadow_map);}
        
    //         //diffuse
    //         let light_dir = normalize(light.pos_world - pos_world.xyz);
    //         var diffuse = max(dot(n_world, light_dir), 0.0) * albedo.xyz * light.color * light.intensity * shadow;
    //         // specular
    //         let halfway_dir = normalize(light_dir + view_dir);    
    //         let spec = pow(max(dot(n_world, halfway_dir), 0.0), 64.0);
    //         let spec_intensity = 0.5;
    //         var specular = light.color * spec * spec_intensity;
    //         // attenuation
    //         let distance = length(light.pos_world - pos_world.xyz);
    //         let attenuation = 1.0 / (1.0 + distance * distance);
    //         diffuse *= attenuation;
    //         specular *= attenuation;
    //         //accumulate
    //         color_linear += diffuse + specular;   
    //     }
        
    //     //tonemap  
    //     // let color_tonemapped = TonemapUtils::ACESFitted(color_linear.rgb);
    //     // var color_tonemapped_gamma = vec4<f32>(pow(final_color_rgb.xyz, vec3<f32>(1.0/2.2)), 1.0 ); //gamma correction
    // }

    // if(GlobalBinds::params.enable_distance_fade>0u){
    //     //get fog factor
    //     let dist_center = length(pos_world.xz); //only take the horizontal plane so xz
    //     var fog_factor = NumUtils::map(dist_center, GlobalBinds::params.distance_fade_start, GlobalBinds::params.distance_fade_end, 0.0, 1.0);
    //     fog_factor = num_utils::smootherstep(0.0, 1.0, fog_factor);
    //     //blend
    //     color_linear = color_linear*(1.0-fog_factor) + GlobalBinds::params.bg_color.rgb*fog_factor;
    // }



    // //tonemap as the last step!
    // let color_tonemapped = TonemapUtils::ACESFitted(color_linear.rgb);
    // let color_tonemapped_gamma = pow(color_tonemapped.xyz, vec3<f32>(1.0/2.2)); //gamma correction
    // let color_tonemapped_gamma_rgba = vec4<f32>(color_tonemapped_gamma,1.0);


    // return color_tonemapped_gamma_rgba;
}