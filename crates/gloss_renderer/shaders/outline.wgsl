// Outline shader
//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 1, binding 0 Locals

#import ./types/global_types.wgsl as GlobalTypes
#import ./bindings/global_binds.wgsl as GlobalBinds

struct Locals {
  model_matrix : mat4x4<f32>,
  outline_color: vec4<f32>,
  outline_width: f32,
  is_floor: u32,
  pad_c: f32,
  pad_d: f32
};

@group(1) @binding(0) var<uniform> locals : Locals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normals: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
        
    let clip_position = GlobalBinds::camera.proj * GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.position, 1.0);
    let clip_normal = GlobalBinds::camera.proj * GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.normals, 0.0);
    let resolution = vec2f(GlobalBinds::camera.width, GlobalBinds::camera.height);
    let normalised_normal = normalize(clip_normal);

    out.clip_position = clip_position;

    // This should represent the outline width in pixels (https://ameye.dev/notes/rendering-outlines/#extrusion-space)
    out.clip_position.x += (normalised_normal.x / resolution.x) * clip_position.w * locals.outline_width * 2.0;
    out.clip_position.y += (normalised_normal.y / resolution.y) * clip_position.w * locals.outline_width * 2.0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return locals.outline_color;
} 