#import ../types/global_types.wgsl as GlobalTypes
#import ../utils/shadow_sampling.wgsl as ShadowSampling
#import ../utils/num_utils.wgsl as NumUtils


// http://www.ludicon.com/castano/blog/articles/shadow-mapping-summary-part-1/
fn get_shadow_offsets(N: vec3<f32>, L: vec3<f32>) -> vec2<f32> {
    let cos_alpha = saturate(dot(N, L));
    let offset_scale_N = sqrt(1.0 - cos_alpha*cos_alpha); // sin(acos(L·N))
    let offset_scale_L = offset_scale_N / cos_alpha;    // tan(acos(L·N))
    // let offset_scale_N = sin(acos(cos_alpha)); // sin(acos(L·N))
    // let offset_scale_L = tan(acos(cos_alpha));    // tan(acos(L·N))
    return vec2<f32>(offset_scale_N, min(2.0, offset_scale_L));
}

fn fetch_shadow(light: GlobalTypes::Light,  pos_world: vec3<f32>, normal_world: vec3<f32>, shadow_map: texture_depth_2d, sampler_shadow_map: sampler_comparison, params: GlobalTypes::Params) -> f32 {
    // return ShadowUtils::fetch_shadow_pcf_3x3(pos_light_space, shadow_map, sampler_nearest);
    // return ShadowSampling::fetch_shadow_pcf_3x3(pos_light_space, shadow_map, sampler_shadow_map);

    var pos_light_space = light.proj * light.view * vec4<f32>(pos_world.xyz, 1.0);

    //move the pos in world along the normal 
    //using https://www.ludicon.com/castano/blog/articles/shadow-mapping-summary-part-1/
    //https://ndotl.wordpress.com/2014/12/19/notes-on-shadow-bias/
    //from normal offset shadows by Daniel Holbert GDC 2011
    let L = normalize(light.pos_world-pos_world);
    let offsets_scale = get_shadow_offsets(normal_world, L);
    let shadowDepthTextureSize =  f32(textureDimensions(shadow_map).x); 
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    let bias_along_normal_dir=light.shadow_bias_normal*oneOverShadowDepthTextureSize;
    //TODO scale the bias also by the distance as mention by "normal offset shadows" by Daniel Holbert GDC 2011
    let pos_view = light.view * vec4<f32>(pos_world, 1.0);
    let distance_scale= abs(pos_view.z);
    let fov_factor = 1.0/min(light.proj[0].x, light.proj[1].y);//the higher the fov, the larger the texels are so we need to scale more
    // let pos_world_biased = pos_world.xyz + light.shadow_bias_normal*saturate(1.0-cos_angle)*normal_world;
    let pos_world_biased = pos_world.xyz + bias_along_normal_dir*offsets_scale.x*distance_scale*fov_factor*normal_world;
    let pos_light_space_biased = light.proj * light.view * vec4<f32>(pos_world_biased, 1.0);
    pos_light_space=vec4(pos_light_space_biased.xy, pos_light_space.zw);

    // compensate for the Y-flip difference between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    let proj_correction = 1.0 / pos_light_space.w;
    let shadow_map_coords = pos_light_space.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    var cur_frag_z = pos_light_space.z * proj_correction;

    //bias using the approach of http://www.jp.square-enix.com/tech/library/pdf/2023_FFXVIShadowTechPaper.pdf
    //DOES NOT WORK for normal mapped meshes because the normal in a deferred renderer is already normal mapped so it's perturbed away from the actual surface.
    // let light_dir = normalize(pos_world - light.pos_world);
    // let linear_depth=NumUtils::linearize_depth_reverse_z(cur_frag_z, light.near, light.far);
    // let is_facing_light = dot ( normal_world , light_dir ) > 0;
    // let bias_towards_light = select(-light.shadow_bias,light.shadow_bias, is_facing_light); 
    // cur_frag_z= NumUtils::lineardepth_to_nonlinear_reverse_z(linear_depth+bias_towards_light, light.near, light.far);

    //using https://www.ludicon.com/castano/blog/articles/shadow-mapping-summary-part-1/
    // let shadowDepthTextureSize =  f32(textureDimensions(shadow_map).x); 
    // let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    // let bias_towards_light=light.shadow_bias*oneOverShadowDepthTextureSize*distance_scale*fov_factor;
    let bias_towards_light=light.shadow_bias*oneOverShadowDepthTextureSize*fov_factor;
    //apply bias in nonlinear depth
    cur_frag_z = cur_frag_z + light.shadow_bias_fixed*oneOverShadowDepthTextureSize*fov_factor + bias_towards_light*offsets_scale.y;
    

    //usign fixed bias
    // cur_frag_z = cur_frag_z + light.shadow_bias; //don't change it for now because it depends on feste having a concrete near and far values.



    var visibility=0.0; 
    if (pos_light_space.w <= 0.0) { //if we are behind the light we are visible 
        visibility=1.0;
    }else if shadow_map_coords.x < 0.0 || shadow_map_coords.y < 0.0 || shadow_map_coords.x > 1.0 || shadow_map_coords.y > 1.0 {
        //if we are outside the bounds of the lights
        visibility=1.0;
    }else{
        if params.shadow_filter_method==0{
            visibility=ShadowSampling::sample_shadow_map_hardware(shadow_map_coords, shadow_map, cur_frag_z, sampler_shadow_map);
        }else if (params.shadow_filter_method==1){
            visibility=ShadowSampling::sample_shadow_map_castano_thirteen(shadow_map_coords, shadow_map, cur_frag_z, sampler_shadow_map);
        }
        // visibility=ShadowSampling::sample_shadow_map_pcf_3x3(shadow_map_coords, shadow_map, cur_frag_z, sampler_shadow_map);
        // visibility=ShadowSampling::sample_shadow_map_castano_thirteen(shadow_map_coords, shadow_map, cur_frag_z, sampler_shadow_map);
    }


    return visibility;
}
