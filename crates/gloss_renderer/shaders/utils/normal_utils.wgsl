//normal mapping as explained here: 
// http://www.thetenthplanet.de/archives/1180
//https://www.geeks3d.com/20130122/normal-mapping-without-precomputed-tangent-space-vectors/
fn cotangent_frame( N: vec3<f32>, p: vec3<f32>, uv: vec2<f32> ) -> mat3x3<f32> {
    // get edge vectors of the pixel triangle
    let dp1 = dpdxFine( p ).xyz;
    let dp2 = dpdyFine( p ).xyz;
    let duv1 = dpdxFine( uv ).xy;
    let duv2 = dpdyFine( uv ).xy;

    //webgl needs individual derivatives
    //webgl has precision issues when the p or uv are very small so we scale them
    // let ps = p*vec3<f32>(100.0);
    // let uvs = uv*vec2<f32>(100.0);
    // var dp1 = vec3<f32>(dpdxFine(p.x), dpdxFine(p.y), dpdxFine(p.z));
    // var dp2 = vec3<f32>(dpdyFine(p.x), dpdyFine(p.y), dpdyFine(p.z));
    // var duv1 = vec2<f32>(dpdxFine(uv.x), dpdxFine(uv.y));
    // var duv2 = vec2<f32>(dpdyFine(uv.x), dpdyFine(uv.y));

    // dp1*=vec3<f32>(100.0);
    // dp2*=vec3<f32>(100.0);
    // duv1*=vec2<f32>(100.0);
    // duv2*=vec2<f32>(100.0);

    // solve the linear system
    let dp2perp = cross( dp2, N );
    let dp1perp = cross( N, dp1 );
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame 
    let invmax = inverseSqrt( max( dot(T,T), dot(B,B) ) );
    return mat3x3( T * invmax, B * invmax, N );
}
//perturbs the per_vertex interpolated normal N according to the normal map t_normal
fn perturb_normal( N: vec3<f32>, V: vec3<f32>, t_normal: texture_2d<f32>, texcoord: vec2<f32>, sampler_linear_obj: sampler) -> vec3<f32>{
    // assume N, the interpolated vertex normal and // V, the view vector (vertex to eye)
    // vec3 map = texture2D( mapBump, texcoord ).xyz;
    var map_01 = textureSample(t_normal, sampler_linear_obj, texcoord).xyz;

    //for some reason some of the textures 
    // let zero_lvl_x = 0.47843137254;
    // let zero_lvl_y = 0.47843137254;
    // let zero_lvl_z = 1.0;
    // let diffx=abs(map.x-zero_lvl_x);
    // let diffy=abs(map.y-zero_lvl_y);
    // let diffz=abs(map.z-zero_lvl_z);
    // if abs(diffx)<1e-3 && abs(diffy)<1e-3{
    //     return vec3<f32>(1.0, 0.0, 0.0);
    // }
    // let map_u = vec3<u32>(u32(map.x*255.0), u32(map.y*255.0), u32(map.z*255.0));
    // if (map_u.x==127u || map_u.x==128u) && (map_u.y==128u || map_u.y==128u){
    //     // return vec3<f32>(1.0, 0.0, 0.0);
    //     return normalize(N);
    // }

    var map_11 = map_01 * 2.0 - 1.0;
    map_11.y= -map_11.y;

    // let diffx=map.x;
    // let diffy=map.y;
    // let diffz=map.z-1.0;
    // if abs(diffx)<1e-3 && abs(diffy)<1e-3{
    //     return vec3<f32>(1.0, 0.0, 0.0);
    // }
    let TBN = cotangent_frame( N, -V, texcoord );

    //attempt 2 
    //https://irrlicht.sourceforge.io/forum/viewtopic.php?t=52284
    // let denormTangent = dpdx(texcoord.y)*dpdy(V)-dpdx(V)*dpdy(texcoord.y);
    // let tangent = normalize(denormTangent-N*dot(N,denormTangent));
    // let normal = normalize(N);
    // let bitangent = cross(normal,tangent);
    // let TBN = mat3x3( tangent , bitangent , normal );

    // let map_u = vec3<u32>(u32(map_01.x*255.0), u32(map_01.y*255.0), u32(map_01.z*255.0));
    // if map_u.x==122u  && map_u.y==122u{
    //     return normalize(N);
    // }else{
        return normalize( TBN * map_11 );
    // }

   


} 


fn apply_tbn( N: vec3<f32>, T: vec3<f32>, B: vec3<f32>, t_normal: texture_2d<f32>, texcoord: vec2<f32>, sampler_linear_obj: sampler) -> vec3<f32>{
    // assume N, the interpolated vertex normal and // V, the view vector (vertex to eye)
    // vec3 map = texture2D( mapBump, texcoord ).xyz;
    var map_01 = textureSample(t_normal, sampler_linear_obj, texcoord).xyz;

    //for some reason some of the textures 
    // let zero_lvl_x = 0.47843137254;
    // let zero_lvl_y = 0.47843137254;
    // let zero_lvl_z = 1.0;
    // let diffx=abs(map.x-zero_lvl_x);
    // let diffy=abs(map.y-zero_lvl_y);
    // let diffz=abs(map.z-zero_lvl_z);
    // if abs(diffx)<1e-3 && abs(diffy)<1e-3{
    //     return vec3<f32>(1.0, 0.0, 0.0);
    // }
    // let map_u = vec3<u32>(u32(map.x*255.0), u32(map.y*255.0), u32(map.z*255.0));
    // if (map_u.x==127u || map_u.x==128u) && (map_u.y==128u || map_u.y==128u){
    //     // return vec3<f32>(1.0, 0.0, 0.0);
    //     return normalize(N);
    // }

    var map_11 = map_01 * 2.0 - 1.0;
    // map_11.y= -map_11.y;
    // map_11.x= -map_11.x;

    // let diffx=map.x;
    // let diffy=map.y;
    // let diffz=map.z-1.0;
    // if abs(diffx)<1e-3 && abs(diffy)<1e-3{
    //     return vec3<f32>(1.0, 0.0, 0.0);
    // }
    let TBN = mat3x3( T, B , N );

    //attempt 2 
    //https://irrlicht.sourceforge.io/forum/viewtopic.php?t=52284
    // let denormTangent = dpdx(texcoord.y)*dpdy(V)-dpdx(V)*dpdy(texcoord.y);
    // let tangent = normalize(denormTangent-N*dot(N,denormTangent));
    // let normal = normalize(N);
    // let bitangent = cross(normal,tangent);
    // let TBN = mat3x3( tangent , bitangent , normal );

    // let map_u = vec3<u32>(u32(map_01.x*255.0), u32(map_01.y*255.0), u32(map_01.z*255.0));
    // if map_u.x==122u  && map_u.y==122u{
    //     return normalize(N);
    // }else{
        return normalize( TBN * map_11 );
    // }

   


} 
