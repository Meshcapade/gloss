#import ../bindings/global_binds.wgsl as GlobalBinds
#import ../bindings/compose_binds.wgsl as ComposeBinds

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn environment_map_light(
    perceptual_roughness: f32,
    roughness: f32,
    diffuse_color: vec3<f32>,
    NdotV: f32,
    f_ab: vec2<f32>,
    N: vec3<f32>,
    R: vec3<f32>,
    F0: vec3<f32>,
) -> EnvironmentMapLight {

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    // Technically we could use textureNumLevels(environment_map_specular) - 1 here, but we use a uniform
    // because textureNumLevels() does not work on WebGL2
    // let radiance_level = perceptual_roughness * f32(bindings::lights.environment_map_smallest_specular_mip_level);
    let radiance_level = perceptual_roughness * f32(GlobalBinds::scene.environment_map_smallest_specular_mip_level);
    let irradiance = textureSampleLevel(ComposeBinds::environment_map_diffuse, GlobalBinds::sampler_linear, vec3(N.xy, -N.z), 0.0).rgb;
    let radiance = textureSampleLevel(ComposeBinds::environment_map_specular, GlobalBinds::sampler_linear, vec3(R.xy, -R.z), radiance_level).rgb;

    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.html#specularocclusion
    let specular_occlusion = saturate(dot(F0, vec3(50.0 * 0.33)));

    // Multiscattering approximation: https://www.jcgt.org/published/0008/01/03/paper.pdf
    // Useful reference: https://bruop.github.io/ibl
    let Fr = max(vec3(1.0 - roughness), F0) - F0;
    let kS = F0 + Fr * pow(1.0 - NdotV, 5.0);
    let Ess = f_ab.x + f_ab.y;
    let FssEss = kS * Ess * specular_occlusion;
    let Ems = 1.0 - Ess;
    let Favg = F0 + (1.0 - F0) / 21.0;
    let Fms = FssEss * Favg / (1.0 - Ems * Favg);
    let FmsEms = Fms * Ems;
    let Edss = 1.0 - (FssEss + FmsEms);
    let kD = diffuse_color * Edss;

    var out: EnvironmentMapLight;
    out.diffuse = (FmsEms + kD) * irradiance;
    out.specular = FssEss * radiance;
    return out;
}