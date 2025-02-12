// use gloss_hecs::Bundle;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

/// Component usually added on lights. Defines properties of the light emitter.
pub struct LightEmit {
    pub color: na::Vector3<f32>,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32, //Each spotlight is represented as a circle of this radius
    //from Bevy https://github.com/bevyengine/bevy/blob/0cc11791b9a55f5d310ee754c9f7254081e7138a/crates/bevy_pbr/src/light.rs#L108
    // More explained on filament in the SpotLight section https://google.github.io/filament/Filament.md.html
    /// Angle defining the distance from the spot light direction to the outer
    /// limit of the light's cone of effect.
    /// `outer_angle` should be < `PI / 2.0`.
    /// `PI / 2.0` defines a hemispherical spot light, but shadows become very
    /// blocky as the angle approaches this limit.
    pub outer_angle: f32,
    /// Angle defining the distance from the spot light direction to the inner
    /// limit of the light's cone of effect.
    /// Light is attenuated from `inner_angle` to `outer_angle` to give a smooth
    /// falloff. `inner_angle` should be <= `outer_angle`
    pub inner_angle: f32,
    // pub spot_scale: f32, //the spotlight can be made tigher or wider with this
    //TODO width and height for area lights
}
impl Default for LightEmit {
    fn default() -> LightEmit {
        Self {
            color: na::Vector3::<f32>::new(1.0, 1.0, 1.0),
            intensity: 30.0,
            range: 100.0,
            radius: 0.1,
            outer_angle: std::f32::consts::PI / 2.0, //correspond to a 180 hemisphere
            inner_angle: 0.0,
        }
    }
}

/// Component added to a Light to indicate that it will cast a shadow with a
/// certain resolution
pub struct ShadowCaster {
    /// Resolution of the shadow map. Shadow map is always a square texture.
    pub shadow_res: u32,
    pub shadow_bias_fixed: f32,
    pub shadow_bias: f32,
    pub shadow_bias_normal: f32,
}
impl Default for ShadowCaster {
    fn default() -> Self {
        Self {
            shadow_res: 2048,
            shadow_bias_fixed: 2e-6,
            shadow_bias: 2e-6,
            shadow_bias_normal: 2e-6,
        }
    }
}

/// Component that is usually automatically added by the renderer on all
/// entities that have [`ShadowCaster`]
pub struct ShadowMap {
    pub tex_depth: easy_wgpu::texture::Texture,
    // pub tex_depth_moments: easy_wgpu::texture::Texture,
}

// /so we can use the Components inside the Mutex<Hashmap> in the scene and wasm
// https://stackoverflow.com/a/73773940/22166964
// shenanigans
//verts
#[cfg(target_arch = "wasm32")]
unsafe impl Send for ShadowMap {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for ShadowMap {}
