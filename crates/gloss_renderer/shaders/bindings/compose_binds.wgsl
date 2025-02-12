
//for deferres we use 4 textures for gbuffer so the indices start at 5
// @group(1) @binding(5) var environment_map_diffuse: texture_cube<f32>;
// @group(1) @binding(6) var environment_map_specular: texture_cube<f32>;
// @group(1) @binding(7) var shadow_map_0: texture_depth_2d;
// @group(1) @binding(8) var shadow_map_1: texture_depth_2d;
// @group(1) @binding(9) var shadow_map_2: texture_depth_2d;
// @group(1) @binding(10) var shadow_map_3: texture_depth_2d;
// @group(1) @binding(11) var shadow_map_4: texture_depth_2d;
// @group(1) @binding(12) var shadow_map_5: texture_depth_2d;
// @group(1) @binding(13) var shadow_map_6: texture_depth_2d;
// @group(1) @binding(14) var shadow_map_7: texture_depth_2d;

//for forward pass we dont have a gbuffer so the indices start at 0
@group(1) @binding(0) var environment_map_diffuse: texture_cube<f32>;
@group(1) @binding(1) var environment_map_specular: texture_cube<f32>;
@group(1) @binding(2) var shadow_map_0: texture_depth_2d;
@group(1) @binding(3) var shadow_map_1: texture_depth_2d;
@group(1) @binding(4) var shadow_map_2: texture_depth_2d;
// @group(1) @binding(5) var shadow_map_3: texture_depth_2d;
// @group(1) @binding(6) var shadow_map_4: texture_depth_2d;
// @group(1) @binding(7) var shadow_map_5: texture_depth_2d;
// @group(1) @binding(8) var shadow_map_6: texture_depth_2d;
// @group(1) @binding(9) var shadow_map_7: texture_depth_2d;