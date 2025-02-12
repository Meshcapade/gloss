// Vertex shader
//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 2, binding 0 Locals

#import ./types/global_types.wgsl as GlobalTypes
#import ./bindings/global_binds.wgsl as GlobalBinds
#import ./utils/tonemap_utils.wgsl as TonemapUtils

//basically the idea from https://webgpufundamentals.org/webgpu/lessons/webgpu-points.html
//where we create a quad for every vertex by drawing indexed

struct Locals {
  model_matrix : mat4x4<f32>,
  color_type: i32,
  point_color: vec4<f32>,
  point_size: f32,
  is_point_size_in_world_space: u32,
  zbuffer: u32,
//   pad_d: f32
};

//group 2
@group(1) @binding(0) var<uniform> locals : Locals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) colors: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(vertex_index) v_idx: u32,
) -> VertexOutput {
    var points = array(
        vec2f(-1, -1),
        vec2f( 1, -1),
        vec2f(-1,  1),
        vec2f(-1,  1),
        vec2f( 1, -1),
        vec2f( 1,  1),
    );
    var out: VertexOutput;

    let clip_pos = GlobalBinds::camera.proj * GlobalBinds::camera.view* locals.model_matrix * vec4<f32>(model.position, 1.0);
    var clip_pos_ndc = clip_pos / clip_pos.w; 

    if locals.zbuffer == 0 {
        clip_pos_ndc = vec4f(clip_pos_ndc.xy, 1, 1);
    }
    let pos = points[v_idx];
    let resolution = vec2f(GlobalBinds::camera.width, GlobalBinds::camera.height);
    // let clip_offset = vec4f(pos * locals.point_size / resolution / clip_pos.w, 0, 0);

    var w_coord=1.0;
    if (locals.is_point_size_in_world_space>0u){
        w_coord=clip_pos.w;
    }
    let clip_offset = vec4f(pos * locals.point_size / resolution /w_coord, 0, 0);
    out.clip_position = clip_pos_ndc + clip_offset;
    out.color = model.colors;
    
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color_linear = vec4<f32>(0.0);
    if locals.color_type==0 {
        color_linear = locals.point_color;
    } else if locals.color_type==1 {
        color_linear = vec4<f32>(in.color,1.0);
    }

    // Linear pre tonemapping grading
    var color = max(color_linear.rgb, vec3(0.0));
    color = TonemapUtils::saturation(color, GlobalBinds::params.saturation);
    color = TonemapUtils::powsafe(color, GlobalBinds::params.gamma);
    color = color * TonemapUtils::powsafe(vec3(2.0), GlobalBinds::params.exposure);
    color = max(color, vec3(0.0));
    color_linear = vec4<f32>(color, color_linear.a);

    // Tonemap as the last step!
    var color_tonemapped = TonemapUtils::ACESFitted(color_linear.rgb);
    let color_tonemapped_gamma = pow(color_tonemapped.xyz, vec3<f32>(1.0/2.2)); //gamma correction
    var color_tonemapped_gamma_rgba = vec4<f32>(color_tonemapped_gamma, color_linear.a);

    //just return directly the albedo
    if !(GlobalBinds::params.apply_lighting >0u){
        color_tonemapped_gamma_rgba = color_linear;
    }

    return color_tonemapped_gamma_rgba;
}