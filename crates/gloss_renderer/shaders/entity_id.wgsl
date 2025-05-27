// Outline shader
//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 1, binding 0 Locals

// entity_id.wgsl
// Entity ID shader
#import ./types/global_types.wgsl as GlobalTypes
#import ./bindings/global_binds.wgsl as GlobalBinds

struct Locals {
  model_matrix: mat4x4<f32>,
  entity_id: u32,
  pad_a: f32,
  pad_b: f32,
  pad_c: f32
};

@group(1) @binding(0) var<uniform> locals: Locals;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = GlobalBinds::camera.proj * GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// @fragment
// fn fs_main() -> @location(0) vec4<u32> {
//     return vec4<u32>(locals.entity_id, 0u, 0u, 0u);
// }

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(locals.entity_id) / 255.0, 0.0, 0.0, 1.0);
}