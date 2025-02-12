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
#import ./utils/num_utils.wgsl as NumUtils

//basically the idea from https://webgpufundamentals.org/webgpu/lessons/webgpu-points.html
//where we create a quad for every vertex by drawing indexed

struct Locals {
  model_matrix : mat4x4<f32>,
  color_type: i32,
  line_color: vec4<f32>,
  line_width: f32,
  zbuffer: u32,
  antialias_edges: u32,
  is_floor: u32,
  pad_b: f32,
  pad_c: f32,
  pad_d: f32,
};

//group 2
@group(1) @binding(0) var<uniform> locals : Locals;

struct VertexInput {
    @location(0) ev1: vec3<f32>,
    @location(1) ev2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position_world: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) offset_from_edge: f32,
    @location(3) offset_max: f32,
}

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(vertex_index) v_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let clip_pos_ev1 = GlobalBinds::camera.proj * GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.ev1, 1.0);
    let clip_pos_ev2 = GlobalBinds::camera.proj * GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.ev2, 1.0);

    let clip_pos_ev1_ndc = clip_pos_ev1 / clip_pos_ev1.w; 
    let clip_pos_ev2_ndc = clip_pos_ev2 / clip_pos_ev2.w; 
    let w1=clip_pos_ev1.w;
    let w2=clip_pos_ev2.w;

    // let w1=clip_pos_ev1.w;
    // let w2=clip_pos_ev2.w;
    

    let resolution = vec2f(GlobalBinds::camera.width, GlobalBinds::camera.height);
    let thickness = locals.line_width / resolution;
    let direction = clip_pos_ev2_ndc - clip_pos_ev1_ndc; 
    let perpendicular = normalize(vec2<f32>(direction.y, -direction.x));
    let offset = perpendicular * thickness; 

    //don't divide by w, we let the gpu do that by writing the clip_position with the correct w instead of using w=1
    //IF you want to make the line go smaller in world coordinates, you cam multiply the offset.x and offset.y by 1.0 instead of by w1 and w2
    var points = array(
        vec4f(clip_pos_ev1.x + offset.x*w1, clip_pos_ev1.y + offset.y*w1, clip_pos_ev1.z, w1),
        vec4f(clip_pos_ev1.x - offset.x*w1, clip_pos_ev1.y - offset.y*w1, clip_pos_ev1.z, w1),
        vec4f(clip_pos_ev2.x + offset.x*w2, clip_pos_ev2.y + offset.y*w2, clip_pos_ev2.z, w2),
        vec4f(clip_pos_ev2.x + offset.x*w2, clip_pos_ev2.y + offset.y*w2, clip_pos_ev2.z, w2),
        vec4f(clip_pos_ev2.x - offset.x*w2, clip_pos_ev2.y - offset.y*w2, clip_pos_ev2.z, w2),
        vec4f(clip_pos_ev1.x - offset.x*w1, clip_pos_ev1.y - offset.y*w1, clip_pos_ev1.z, w1),
    );

    if locals.zbuffer == 0 {
        points = array(
            vec4f(clip_pos_ev1.x + offset.x*w1, clip_pos_ev1.y + offset.y*w1, 0.999, w1),
            vec4f(clip_pos_ev1.x - offset.x*w1, clip_pos_ev1.y - offset.y*w1, 0.999, w1),
            vec4f(clip_pos_ev2.x + offset.x*w2, clip_pos_ev2.y + offset.y*w2, 0.999, w2),
            vec4f(clip_pos_ev2.x + offset.x*w2, clip_pos_ev2.y + offset.y*w2, 0.999, w2),
            vec4f(clip_pos_ev2.x - offset.x*w2, clip_pos_ev2.y - offset.y*w2, 0.999, w2),
            vec4f(clip_pos_ev1.x - offset.x*w1, clip_pos_ev1.y - offset.y*w1, 0.999, w1),
        );
    }

    let p1_world = (locals.model_matrix * vec4<f32>(model.ev1, 1.0)).xyz;
    let p2_world = (locals.model_matrix * vec4<f32>(model.ev2, 1.0)).xyz;
    var points_world = array(
        p1_world,
        p1_world,
        p2_world,
        p2_world,
        p2_world,
        p1_world,
    );

    let offset_x_plus=length(offset);
    let offset_x_minus=-length(offset);
    // //normalize the offse to be in range [0,1] and flipped at the median
    var offset_from_edge_array = array(
        offset_x_plus,
        offset_x_minus,
        offset_x_plus,
        offset_x_plus,
        offset_x_minus,
        offset_x_minus,
    );

    out.position_world = vec3f(points_world[v_idx]); 
    out.clip_position = points[v_idx]; 
    // out.offset_from_median = length(offset)*100.0; 
    out.offset_from_edge = offset_from_edge_array[v_idx]; 
    out.offset_max = length(offset);
    out.color = locals.line_color.xyz;

    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color_linear = vec4<f32>(0.0);
    if locals.color_type==0 {
        color_linear = locals.line_color;
    } else if locals.color_type==1 {
        color_linear = vec4<f32>(in.color,1.0);
    }

    //modify the color towards background when it gets towards the edges of the line
    if locals.antialias_edges>0u{
        let offset_from_median=abs(in.offset_from_edge/in.offset_max); //0 at the center of the line, 1.0 at the edge of it
        let line_smoothness_factor=smoothstep(0.0, 1.0, offset_from_median); 
        color_linear = color_linear*(1.0-line_smoothness_factor) + GlobalBinds::params.bg_color*line_smoothness_factor;
    }

    var fog_factor = 0.0; 
    if(locals.is_floor>0u && GlobalBinds::params.enable_distance_fade>0u){
        //get fog factor
        let dist_center = length(in.position_world.xz - GlobalBinds::params.distance_fade_center.xz); //only take the horizontal plane so xz
        fog_factor = NumUtils::map(dist_center, GlobalBinds::params.distance_fade_start, GlobalBinds::params.distance_fade_end, 0.0, 1.0);
        fog_factor = num_utils::smootherstep(0.0, 1.0, fog_factor);
    }
    color_linear = color_linear*(1.0-fog_factor) + GlobalBinds::params.bg_color*fog_factor;
    

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
    let color_tonemapped_gamma_rgba = vec4<f32>(color_tonemapped_gamma,1.0);

    //debug
    // let color_tonemapped_gamma_rgba = vec4<f32>(in.position_world.xyz,1.0);

    return color_tonemapped_gamma_rgba;
}