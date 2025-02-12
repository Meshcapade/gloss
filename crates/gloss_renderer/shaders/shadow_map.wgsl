// Vertex shader
//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 1, binding 0 Locals (for lights)
//group 2, binding 0 Locals (for meshes)
//group 2, binding 1 diffuse_tex
#import ./types/global_types.wgsl as GlobalTypes
#import ./bindings/global_binds.wgsl as GlobalBinds

struct IteratorLight {
    light_idx: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}
struct Locals {
  model_matrix : mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> iterator_light: IteratorLight;
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
    let light_idx = iterator_light.light_idx;
    let light = GlobalBinds::lights[light_idx];
    out.clip_position = light.proj * light.view* locals.model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader--------------------------------------------------
struct FragOutput {
    @builtin(frag_depth) depth: f32,
    // @location(0) depth_moments: vec2<f32>,
}
@fragment
fn fs_main(
    in: VertexOutput, 
    ) -> FragOutput {
    var out: FragOutput;


    let depth=in.clip_position.z;

    // let dx=dpdx(depth);
    // let dy=dpdy(depth);
    // let moment2 = depth*depth + 0.25 * (dx*dx+dy*dy);

    out.depth = in.clip_position.z;
    // out.depth_moments=vec2<f32>(depth, moment2);

    return out;
}



