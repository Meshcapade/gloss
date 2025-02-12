struct PosUV {
    pos: vec4<f32>,
    uv: vec2<f32>,
};

//creates the position and texture coords of a full screen triangle that can be used in a vertex shader
//the vertex index can be obtained in the vertex shader using @builtin(vertex_index) vertex_index: u32
fn full_screen_tri(vertex_index: u32) -> PosUV{
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    let position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0, 1.0
    );

    var pos_uv: PosUV;
    pos_uv.pos=position;
    pos_uv.uv=tc;

    return pos_uv;
}