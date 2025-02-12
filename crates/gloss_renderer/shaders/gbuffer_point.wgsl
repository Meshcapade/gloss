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


struct Locals {
  model_matrix : mat4x4<f32>,
  color_type: i32,
  point_color: vec4<f32>,
  point_size: f32,
};

//group 2
@group(2) @binding(0) var<uniform> locals : Locals;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
}



@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = model.position;
    out.clip_position = GlobalBinds::camera.proj * GlobalBinds::camera.view* locals.model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

struct FragOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) position: vec4<f32>,
    @location(2) normals: vec4<f32>,
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    //process some of the input
    let pos_world = locals.model_matrix * vec4<f32>(in.position, 1.0);
    let normal_world = vec3(0.0, 0.0, 0.0); 

    var color = vec4<f32>(0.0);
    if locals.color_type==0{
        color = locals.point_color;
    }
    // }else if locals.color_type==2{
    //     color = textureSample(t_diffuse, sampler_linear, vec2<f32>(in.tex_coords.x, 1.0-in.tex_coords.y));
    // }else if locals.color_type==4{
    //     color = vec4<f32>(in.normals*0.5f+0.5f, 1.0);
    // }



    var out: FragOutput;
    out.albedo = color;
    out.position = vec4<f32>(pos_world.xyz, 1.0);
    out.normals = vec4<f32>(normal_world, 0.0);
    return out;
}