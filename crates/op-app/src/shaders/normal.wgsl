// Tangent-space normal map from composited height, combined with a baked normal map.
// Same math as op-core/src/normal.rs (a headless test compares them).

struct Params {
    strength: f32,
    size: f32,
    has_baked: u32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var height_tex: texture_2d<f32>;   // (height * coverage, coverage)
@group(0) @binding(2) var baked_tex: texture_2d<f32>;    // packed tangent-space normal, RGBA8

const SLOPE_PER_UV: f32 = 0.01;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

fn height_at(xy: vec2<i32>) -> f32 {
    let d = vec2<i32>(textureDimensions(height_tex));
    let t = textureLoad(height_tex, clamp(xy, vec2<i32>(0), d - 1), 0);
    // Uncovered texels are flat.
    return t.r + 0.5 * (1.0 - t.g);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(i32(pos.x), i32(pos.y));
    let dx = (height_at(xy + vec2<i32>(1, 0)) - height_at(xy - vec2<i32>(1, 0))) * 0.5 * p.size;
    let drow = (height_at(xy + vec2<i32>(0, 1)) - height_at(xy - vec2<i32>(0, 1))) * 0.5 * p.size;
    // Image-up is decreasing row, so the slope along +Y is minus the row gradient.
    let slope = vec2<f32>(-dx, drow) * p.strength * SLOPE_PER_UV;

    var base = vec3<f32>(0.0, 0.0, 1.0);
    if (p.has_baked == 1u) {
        let b = textureLoad(baked_tex, xy, 0).rgb * 2.0 - 1.0;
        let nb = b / max(length(b), 1e-6);
        base = vec3<f32>(nb.xy, max(nb.z, 1e-3));
    }
    let n = normalize(vec3<f32>(base.xy + slope, base.z));
    return vec4<f32>(n * 0.5 + 0.5, 1.0);
}
