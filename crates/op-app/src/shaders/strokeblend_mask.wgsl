// open-painter: stroke opacity for masks. The buffer holds the stroke's coverage (it paints 1 with
// the usual blend); the layer mask moves from its pre-stroke snapshot toward the painted value by
// coverage * opacity.

struct Params {
    v: vec4<f32>,   // x: stroke opacity, y: mask value being painted
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var mask_snap: texture_2d<f32>;
@group(0) @binding(2) var mask_buf: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_mask(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(pos.xy);
    let was = textureLoad(mask_snap, xy, 0).r;
    let cover = textureLoad(mask_buf, xy, 0).r * p.v.x;
    return vec4<f32>(mix(was, p.v.y, cover), 0.0, 0.0, 1.0);
}
