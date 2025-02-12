//from https://github.com/gfx-rs/wgpu-rs/blob/master/examples/mipmap/blit.wgsl

#import ./utils/full_screen_tri_utils.wgsl as Tri

@group(0) @binding(0)var r_color: texture_2d<f32>;
@group(0) @binding(1)var r_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    let pos_uv = Tri::full_screen_tri(vertex_index);
    out.position=pos_uv.pos;
    out.tex_coords=pos_uv.uv;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(r_color, r_sampler, in.tex_coords);
}