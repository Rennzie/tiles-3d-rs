// Vertex shader

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
};


// Variables defined with var can be modified, but must specify their type. 
// Variables created with let can have their types inferred, but their value cannot be changed during the shader.
[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;

    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// Fragment shader
[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.clip_position.x, in.clip_position.y, in.clip_position.z, 1.0);
}
 
