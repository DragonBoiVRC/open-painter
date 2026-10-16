// open-painter: layer filters. One pass reads a layer's channels and writes the filtered layer.
//
// All textures are premultiplied: (value * coverage, coverage).
//   kind 0: blur along `dir` (a separable gaussian; run once per axis)
//   kind 1: sharpen: `a` is the layer, `b` its blurred copy; out = a + amount * (a - b)
//   kind 2: invert
//   kind 3: levels

struct Params {
    kind: vec4<u32>,   // x: kind
    v: vec4<f32>,      // blur: radius, dir.x, dir.y | sharpen: amount | levels: black, white, gamma
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var a_base: texture_2d<f32>;
@group(0) @binding(2) var a_metal: texture_2d<f32>;
@group(0) @binding(3) var a_rough: texture_2d<f32>;
@group(0) @binding(4) var a_height: texture_2d<f32>;
@group(0) @binding(5) var a_normal: texture_2d<f32>;
@group(0) @binding(6) var b_base: texture_2d<f32>;
@group(0) @binding(7) var b_metal: texture_2d<f32>;
@group(0) @binding(8) var b_rough: texture_2d<f32>;
@group(0) @binding(9) var b_height: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

struct Out {
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
    @location(4) normal: vec4<f32>,
};

fn load(t: texture_2d<f32>, xy: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t));
    return textureLoad(t, clamp(xy, vec2<i32>(0), dims - 1), 0);
}

// Gaussian blur of one texture along `dir`. Wide blurs stride over texels (up to 16 taps each side).
fn blur(t: texture_2d<f32>, xy: vec2<i32>) -> vec4<f32> {
    let radius = max(p.v.x, 0.0);
    if (radius < 0.5) { return load(t, xy); }
    let taps = i32(min(ceil(radius), 16.0));
    let stride = max(radius / 16.0, 1.0);
    let sigma = max(radius * 0.5, 0.5);
    let dir = vec2<f32>(p.v.y, p.v.z);
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var k = -taps; k <= taps; k++) {
        let d = f32(k) * stride;
        let w = exp(-(d * d) / (2.0 * sigma * sigma));
        sum += w * load(t, xy + vec2<i32>(round(dir * d)));
        wsum += w;
    }
    return sum / wsum;
}

fn levels(v: f32) -> f32 {
    let span = max(p.v.y - p.v.x, 1e-4);
    return pow(clamp((v - p.v.x) / span, 0.0, 1.0), 1.0 / max(p.v.z, 0.1));
}

// Apply `f` to the straight (un-premultiplied) value of a premultiplied sample.
fn map_scalar(s: vec4<f32>, kind: u32) -> vec4<f32> {
    let cov = s.g;
    if (cov <= 0.0) { return s; }
    var v = s.r / cov;
    if (kind == 2u) { v = 1.0 - v; } else { v = levels(v); }
    return vec4<f32>(v * cov, cov, s.b, s.a);
}

fn map_color(s: vec4<f32>, kind: u32) -> vec4<f32> {
    let cov = s.a;
    if (cov <= 0.0) { return s; }
    var c = s.rgb / cov;
    if (kind == 2u) {
        c = vec3<f32>(1.0) - c;
    } else {
        c = vec3<f32>(levels(c.r), levels(c.g), levels(c.b));
    }
    return vec4<f32>(c * cov, cov);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let xy = vec2<i32>(pos.xy);
    var o: Out;
    let kind = p.kind.x;
    if (kind == 0u) {
        o.base = blur(a_base, xy);
        o.metal = blur(a_metal, xy);
        o.rough = blur(a_rough, xy);
        o.height = blur(a_height, xy);
    } else if (kind == 1u) {
        let amount = p.v.x;
        o.base = max(load(a_base, xy) + amount * (load(a_base, xy) - load(b_base, xy)), vec4<f32>(0.0));
        o.base = vec4<f32>(min(o.base.rgb, vec3<f32>(o.base.a)), min(o.base.a, 1.0));
        let sm = max(load(a_metal, xy) + amount * (load(a_metal, xy) - load(b_metal, xy)), vec4<f32>(0.0));
        o.metal = vec4<f32>(min(sm.r, sm.g), min(sm.g, 1.0), 0.0, min(sm.g, 1.0));
        let sr = max(load(a_rough, xy) + amount * (load(a_rough, xy) - load(b_rough, xy)), vec4<f32>(0.0));
        o.rough = vec4<f32>(min(sr.r, sr.g), min(sr.g, 1.0), 0.0, min(sr.g, 1.0));
        let sh = max(load(a_height, xy) + amount * (load(a_height, xy) - load(b_height, xy)), vec4<f32>(0.0));
        o.height = vec4<f32>(min(sh.r, sh.g), min(sh.g, 1.0), 0.0, min(sh.g, 1.0));
    } else {
        o.base = map_color(load(a_base, xy), kind);
        o.metal = map_scalar(load(a_metal, xy), kind);
        o.rough = map_scalar(load(a_rough, xy), kind);
        o.height = map_scalar(load(a_height, xy), kind);
    }
    o.normal = load(a_normal, xy);
    return o;
}
