struct Scene {
    nr_lights: u32,
    environment_map_smallest_specular_mip_level: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_b: u32,
    pad_c: u32,
}
struct Camera {
  view : mat4x4<f32>,
  view_inv : mat4x4<f32>,
  proj : mat4x4<f32>,
  proj_inv : mat4x4<f32>,
  vp : mat4x4<f32>,
  pos_world: vec3<f32>,
  near: f32,
  far: f32,
  aspect_ratio: f32,
  width: f32,
  height: f32,
};
struct Light {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  vp : mat4x4<f32>,
  pos_world: vec3<f32>,
  lookat_dir_world: vec3<f32>,
  color: vec3<f32>,
  intensity: f32,
  range: f32,
  inverse_square_range: f32,
  radius: f32,
  outer_angle: f32,
  inner_angle: f32,
  near: f32,
  far: f32,
  is_shadow_caster: u32, //should be bool but that is not host-sharable: https://www.w3.org/TR/WGSL/#host-shareable-types
  shadow_bias_fixed: f32,
  shadow_bias: f32,
  shadow_bias_normal: f32,
  //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
  pad_b: f32,
  pad_c: f32,
  pad_d: f32
};
struct Params {
    ambient_factor: f32,
    environment_factor: f32,
    bg_color: vec4<f32>,
    enable_distance_fade: u32, //should be bool but that is not host-sharable: https://www.w3.org/TR/WGSL/#host-shareable-types
    distance_fade_center: vec3<f32>,
    distance_fade_start: f32,
    distance_fade_end: f32,
    //color grading, applied before tonemapping
    apply_lighting: u32,
    saturation: f32, 
    gamma: f32,
    exposure: f32,
    shadow_filter_method: i32,
    // post_saturation: f32, //applied after tonemapping
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_b: f32,
    pad_c: f32,
    pad_d: f32
};