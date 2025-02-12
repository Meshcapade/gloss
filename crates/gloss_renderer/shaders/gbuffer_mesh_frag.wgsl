// Vertex shader
//group 0, binding 0 = scene
//group 0, binding 1 = camera
//group 0, binding 2 = lights 
//group 0, binding 3 = params
//group 0, binding 4 = sampler_nearest
//group 0, binding 5 = sampler_linear
//group 2, binding 0 Locals
//group 2, binding 1 diffuse_tex

#import ./types/global_types.wgsl as GlobalTypes
#import ./utils/normal_utils.wgsl as NormalUtils
#import gbuffer_mesh_vert.wgsl as VertShader
#import ./utils/num_utils.wgsl as NumUtils
#import ./utils/tex_utils.wgsl as TexUtils
#import ./utils/tonemap_utils.wgsl as TonemapUtils
#import ./utils/noise_utils.wgsl as NoiseUtils
#import ./types/pbr_types.wgsl as PbrTypes
#import ./pbr/pbr_functions.wgsl as PbrFunc

//group 0
#import ./bindings/global_binds.wgsl as GlobalBinds

//group 1
#import ./bindings/compose_binds.wgsl as ComposeBinds //contains the shadow maps textures

//group 2
@group(2) @binding(0) var<uniform> locals : VertShader::Locals;
@group(2) @binding(1) var t_diffuse: texture_2d<f32>;
@group(2) @binding(2) var t_normal: texture_2d<f32>;
@group(2) @binding(3) var t_roughness: texture_2d<f32>;

@fragment
fn fs_main(in: VertShader::VertexOutput) -> @location(0) vec4<f32> {
    //process some of the input
    var normal_world = normalize(in.normal_world); //need to normalize because interpolation across triangle might mess things up
    var tangent_world = normalize(in.tangent_world); //need to normalize because interpolation across triangle might mess things up
    var bitangent_world = normalize(in.bitangent_world); //need to normalize because interpolation across triangle might mess things up

    var albedo = vec4<f32>(0.0);
    if locals.color_type==0{
        albedo = locals.solid_color;
    }else if locals.color_type==1{
        albedo = vec4<f32>(in.color,1.0);
    }else if locals.color_type==2{
        let dims_diffuse = vec2<f32>(textureDimensions(t_diffuse));
        albedo = textureSample(t_diffuse, GlobalBinds::sampler_linear, in.tex_coords);
        if dims_diffuse.x<=4.0 &&dims_diffuse.y<=4.0 {
            albedo = locals.solid_color;
        }
    }else if locals.color_type==3{
        albedo = vec4<f32>(in.tex_coords, 0.0, 1.0);
    }else if locals.color_type==4{
        albedo = vec4<f32>(normal_world*0.5f+0.5f, 1.0);
    }


    //normal mapping as explained here: 
    // http://www.thetenthplanet.de/archives/1180
    //https://www.geeks3d.com/20130122/normal-mapping-without-precomputed-tangent-space-vectors/
    let dims_n = vec2<f32>(textureDimensions(t_normal));
    let tangent_finite = tangent_world.x==tangent_world.x; //nans are not equal to any other nan
    let normal_world_not_perturbed=normal_world;
    if dims_n.x>4.0 &&dims_n.y>4.0 && tangent_finite{ //makes it so we don't run this for the dummy normalmap which is 4x4. TODO However this is a bit of a hacky solution, ideally we would pass a boolean flag
        normal_world = NormalUtils::apply_tbn( normal_world, tangent_world, bitangent_world, t_normal, in.tex_coords, GlobalBinds::sampler_linear );
    }

    //roughness
    let dims_roughness = vec2<f32>(textureDimensions(t_roughness));
    var roughness = textureSample(t_roughness, GlobalBinds::sampler_linear, in.tex_coords).x;
    if dims_roughness.x<=4.0 &&dims_roughness.y<=4.0 {
        roughness = locals.perceptual_roughness;
    }
    roughness = NumUtils::map(roughness, locals.roughness_black_lvl, 1.0, 0.0, 1.0);

    // var pbr_input
    var pbr: PbrTypes::PbrInput;
    pbr.material = PbrTypes::standard_material_new();
    let V = normalize(GlobalBinds::camera.pos_world - in.pos_world); //TOOD this would need to change if the projection is orthographic
    pbr.world_position=vec4<f32>(in.pos_world, 1.0);
    pbr.material.perceptual_roughness = roughness;
    pbr.material.base_color = vec4(albedo.xyz, 1.0);
    pbr.material.metallic = locals.metalness;
    pbr.N = normal_world;
    pbr.world_normal=normal_world_not_perturbed;
    pbr.V = V;

    //run pbg coloring----------------------
    var color_linear = vec4<f32>(0.0);

    var fog_factor = 0.0; 
    if(locals.is_floor>0u && GlobalBinds::params.enable_distance_fade>0u){
        //get fog factor
        let dist_center = length(in.pos_world.xz - GlobalBinds::params.distance_fade_center.xz); //only take the horizontal plane so xz
        fog_factor = NumUtils::map(dist_center, GlobalBinds::params.distance_fade_start, GlobalBinds::params.distance_fade_end, 0.0, 1.0);
        fog_factor = num_utils::smootherstep(0.0, 1.0, fog_factor);
    }
    if(locals.is_floor>0u){
        //if fog is very high there is no reason to compute pbr color
        if fog_factor>0.999{
            color_linear=GlobalBinds::params.bg_color;
        }else{
            color_linear = PbrFunc::apply_pbr_lighting(pbr);
            //blend
            color_linear = color_linear*(1.0-fog_factor) + GlobalBinds::params.bg_color*fog_factor;
            //debanding https://blog.frost.kiwi/GLSL-noise-and-radial-gradient/
            color_linear += (1.0 / 255.0) * NoiseUtils::gradient_noise(in.clip_position.xy) - (0.5 / 255.0);
        }
    }else{
        color_linear = PbrFunc::apply_pbr_lighting(pbr);
    }

    // Linear pre tonemapping grading
    var color = max(color_linear.rgb, vec3(0.0));
    color = TonemapUtils::saturation(color, GlobalBinds::params.saturation);
    color = TonemapUtils::powsafe(color, GlobalBinds::params.gamma);
    color = color * TonemapUtils::powsafe(vec3(2.0), GlobalBinds::params.exposure);
    color = max(color, vec3(0.0));
    color_linear = vec4<f32>(color, color_linear.a);

    //tonemap as the last step!
    // var color_tonemapped_gamma_rgba = color_linear;
    // if GlobalBinds::params.apply_tonemapping >0u{
    var color_tonemapped = TonemapUtils::ACESFitted(color_linear.rgb);
    let color_tonemapped_gamma = pow(color_tonemapped.xyz, vec3<f32>(1.0/2.2)); //gamma correction
    var color_tonemapped_gamma_rgba = vec4<f32>(color_tonemapped_gamma, color_linear.a);
    // }

    //just return directly the albedo
    if !(GlobalBinds::params.apply_lighting >0u){
        color_tonemapped_gamma_rgba = albedo;
    }
    

    return color_tonemapped_gamma_rgba;
}