#import ../types/global_types.wgsl as GlobalTypes

@group(0) @binding(0) var<uniform> scene : GlobalTypes::Scene;
@group(0) @binding(1) var<uniform> camera : GlobalTypes::Camera;
// @group(0) @binding(2) var<storage,read> lights : array<Light>;
@group(0) @binding(2) var<uniform> lights : array<GlobalTypes::Light,20>;
@group(0) @binding(3) var<uniform> params : GlobalTypes::Params;
@group(0) @binding(4) var sampler_nearest: sampler;
@group(0) @binding(5) var sampler_linear: sampler;
@group(0) @binding(6) var sampler_shadow_map: sampler_comparison;