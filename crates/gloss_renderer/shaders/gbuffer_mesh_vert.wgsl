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

//group 0
#import ./bindings/global_binds.wgsl as GlobalBinds

//group 1
// #import ./bindings/compose_binds.wgsl as ComposeBinds //contains the shadow maps textures


struct Locals {
  model_matrix : mat4x4<f32>,
  color_type: i32,
  solid_color: vec4<f32>,
  metalness: f32,
  perceptual_roughness: f32,
  roughness_black_lvl: f32,
  uv_scale: f32,
  is_floor: u32,
  pad_b: f32,
  pad_c: f32
};


//group 2
@group(2) @binding(0) var<uniform> locals : Locals;
@group(2) @binding(1) var t_diffuse: texture_2d<f32>;
@group(2) @binding(2) var t_normal: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normals: vec3<f32>,
    @location(3) tangents: vec4<f32>,
    @location(4) colors: vec3<f32>,
}

struct VertexOutput {
    // This is `clip position` when the struct is used as a vertex stage output
    // and `frag coord` when used as a fragment stage input
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos_world: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal_world: vec3<f32>,
    // @location(3) tbn_world: mat3x3<f32>,
    @location(3) tangent_world: vec3<f32>,
    @location(4) bitangent_world: vec3<f32>,
    @location(5) color: vec3<f32>,
    // @location(6) view_vector: vec3<f32>, // camera pos - vertex pos
    // @location(4) pos_view: vec3<f32>,
    // @location(5) eye_vec: vec3<f32>,
    // @location(6) normal_view: vec3<f32>,
}



@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let pos_world=(locals.model_matrix * vec4<f32>(model.position, 1.0)).xyz;
    // let pos_view=(GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.position, 1.0)).xyz;
    out.pos_world = pos_world;
    // out.pos_view = pos_view;
    // out.eye_vec = -pos_view;
    out.tex_coords = vec2<f32>(model.tex_coords.x, 1.0-model.tex_coords.y) * locals.uv_scale;
    out.normal_world = normalize((locals.model_matrix * vec4<f32>(model.normals, 0.0)).xyz); //will not get affected by translation but it will be affected by scale;
    out.color = model.colors;
    // out.normal_view = normalize((GlobalBinds::camera.view * locals.model_matrix * vec4<f32>(model.normals, 0.0)).xyz); //will not get affected by translation but it will be affected by scale;
    out.clip_position = GlobalBinds::camera.proj * GlobalBinds::camera.view* locals.model_matrix * vec4<f32>(model.position, 1.0);
    // out.view_vector = GlobalBinds::camera.pos_world - pos_world;

    // out.tbn_world = mat3x3<f32>(vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0));
    // out.tangent_world = normalize((locals.model_matrix * vec4<f32>(model.tangents, 0.0)).xyz);
    // re-orthogonalize T with respect to N
    // out.tangent_world = normalize(out.tangent_world - dot(out.tangent_world, out.normal_world) * out.normal_world);
    // out.bitangent_world = normalize(cross(out.normal_world, out.tangent_world));

    //attempt 2
    // https://stackoverflow.com/questions/5255806/how-to-calculate-tangent-and-binormal
    // var t=normalize((locals.model_matrix * vec4<f32>(model.tangents, 0.0)).xyz);
    // var n=normalize((locals.model_matrix * vec4<f32>(model.normals, 0.0)).xyz);
    // var b=cross(n, t);
    var t = model.tangents.xyz;
    var h = model.tangents.w;
    var n = model.normals;
    //   re-orthogonalize T with respect to N. This is important if even if you orthgonalize on cpu, because the normals might changle slightly due to things like smpl pose-correctives
    t = normalize(t - dot(t, n) * n);
    var b = cross(n, t) * vec3<f32>(h);
    // t = t - n * dot( t, n ); // orthonormalization ot the tangent vectors
    // b = b - n * dot( b, n ); // orthonormalization of the binormal vectors to the normal vector 
    // b = b - t * dot( b, t ); // orthonormalization of the binormal vectors to the tangent vector

    // re-orthogonalize T with respect to N
    // t = normalize(t - dot(t, n) * n);

    // //handness
    // let cross_vec = cross(n,t);
    // if dot(cross_vec,b)< 0.0{
    //     b= b* -1.0;
    // }


    //attempt 3
    // https://foundationsofgameenginedev.com/FGED2-sample.pdf
    // t=t - n * dot(t, n);
    // if dot(cross(t, b), n) < 0.0{
    //     b=-b;
    // }


    // out.tangent_world=t;
    // out.normal_world=n;
    // out.bitangent_world=b;

    out.tangent_world=normalize((locals.model_matrix * vec4<f32>(t, 0.0)).xyz);
    out.normal_world=normalize((locals.model_matrix * vec4<f32>(n, 0.0)).xyz);
    out.bitangent_world=normalize((locals.model_matrix * vec4<f32>(b, 0.0)).xyz);

    // out.bitangent_world = normalize(cross(out.tangent_world, out.normal_world ));

    return out;
}
