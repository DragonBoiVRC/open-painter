// open-painter: anchor masks. Turns one channel of a recorded anchor (the stack as it was at some
// layer) into a mask: brightness of the colour, metallic, roughness, height, or coverage.

struct Params {
    chan: vec4<u32>,   // x: 0 colour brightness, 1 metallic, 2 roughness, 3 height, 4 coverage
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var a_base: texture_2d<f32>;
@group(0) @binding(2) var a_metal: texture_2d<f32>;
@group(0) @binding(3) var a_rough: texture_2d<f32>;
@group(0) @binding(4) var a_height: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

// Premultiplied (value * coverage, coverage) back to the value; `empty` where nothing was laid down.
fn value(t: vec4<f32>, empty: f32) -> f32 {
    return select(empty, t.r / t.g, t.g > 1e-4);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(pos.xy);
    var v = 0.0;
    switch (p.chan.x) {
        case 0u: {
            let b = textureLoad(a_base, xy, 0);
            v = select(0.0, dot(b.rgb / b.a, vec3<f32>(0.2126, 0.7152, 0.0722)), b.a > 1e-4);
        }
        case 1u: { v = value(textureLoad(a_metal, xy, 0), 0.0); }
        case 2u: { v = value(textureLoad(a_rough, xy, 0), 0.0); }
        case 3u: { v = value(textureLoad(a_height, xy, 0), 0.5); }
        default: { v = textureLoad(a_base, xy, 0).a; }
    }
    return vec4<f32>(clamp(v, 0.0, 1.0), 0.0, 0.0, 1.0);
}
