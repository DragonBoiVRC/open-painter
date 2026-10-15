// open-painter: layer compositor. One pass composites one layer over a backdrop.
// Same math as op-core/src/blend.rs (W3C compositing, separable blend modes);
// a headless test cross-checks the two.
//
// All textures are premultiplied. Pass dst = over(backdrop, source * opacity * mask).

struct Params {
    color: vec4<f32>,   // fill base color, linear rgb
    vals: vec4<f32>,    // opacity, fill metallic, fill roughness, fill height
    mode: vec4<u32>,    // x: source is fill, y: has mask, z: blend mode, w: fill enable bits (1 base, 2 metal, 4 rough, 8 height)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var bk_base: texture_2d<f32>;
@group(0) @binding(2) var bk_metal: texture_2d<f32>;
@group(0) @binding(3) var bk_rough: texture_2d<f32>;
@group(0) @binding(4) var src_base: texture_2d<f32>;
@group(0) @binding(5) var src_metal: texture_2d<f32>;
@group(0) @binding(6) var src_rough: texture_2d<f32>;
@group(0) @binding(7) var mask_tex: texture_2d<f32>;
@group(0) @binding(8) var bk_height: texture_2d<f32>;
@group(0) @binding(9) var src_height: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

fn blend_mode(cb: vec3<f32>, cs: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: { return cb * cs; }
        case 2u: { return cb + cs - cb * cs; }
        case 3u: {
            return select(1.0 - 2.0 * (1.0 - cb) * (1.0 - cs), 2.0 * cb * cs, cb <= vec3<f32>(0.5));
        }
        case 4u: { return min(cb + cs, vec3<f32>(1.0)); }
        case 5u: { return max(cb - cs, vec3<f32>(0.0)); }
        case 6u: { return min(cb, cs); }
        case 7u: { return max(cb, cs); }
        case 8u: { return abs(cb - cs); }
        default: { return cs; }
    }
}

// Premultiplied backdrop, straight source color, effective source alpha.
fn over(bk: vec4<f32>, cs: vec3<f32>, a_s: f32, mode: u32) -> vec4<f32> {
    let ab = bk.a;
    let cb = select(vec3<f32>(0.0), bk.rgb / max(ab, 1e-6), ab > 0.0);
    let b = blend_mode(cb, cs, mode);
    let rgb = (1.0 - a_s) * bk.rgb + a_s * ((1.0 - ab) * cs + ab * b);
    return vec4<f32>(rgb, a_s + ab * (1.0 - a_s));
}

fn unpremul(v: f32, a: f32) -> f32 {
    return select(0.0, v / a, a > 0.0);
}

struct Out {
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let xy = vec2<i32>(i32(pos.x), i32(pos.y));

    var cov = vec4<f32>(0.0);          // coverage of base, metal, rough, height
    var color = vec3<f32>(0.0);        // straight base color
    var metal = 0.0;
    var rough = 0.0;
    var height = 0.0;
    if (p.mode.x == 1u) {
        color = p.color.rgb;
        metal = p.vals.y;
        rough = p.vals.z;
        height = p.vals.w;
        cov = vec4<f32>(f32(p.mode.w & 1u), f32((p.mode.w >> 1u) & 1u), f32((p.mode.w >> 2u) & 1u), f32((p.mode.w >> 3u) & 1u));
    } else {
        let sb = textureLoad(src_base, xy, 0);
        let sm = textureLoad(src_metal, xy, 0);
        let sr = textureLoad(src_rough, xy, 0);
        let sh = textureLoad(src_height, xy, 0);
        cov = vec4<f32>(sb.a, sm.g, sr.g, sh.g);
        height = unpremul(sh.r, sh.g);
        color = select(vec3<f32>(0.0), sb.rgb / sb.a, sb.a > 0.0);
        metal = unpremul(sm.r, sm.g);
        rough = unpremul(sr.r, sr.g);
    }

    var mask = 1.0;
    if (p.mode.y == 1u) {
        mask = textureLoad(mask_tex, xy, 0).r;
    }
    let k = p.vals.x * mask;

    let b = over(textureLoad(bk_base, xy, 0), color, cov.x * k, p.mode.z);
    let bm = textureLoad(bk_metal, xy, 0);
    let br = textureLoad(bk_rough, xy, 0);
    let m = over(vec4<f32>(vec3<f32>(bm.r), bm.g), vec3<f32>(metal), cov.y * k, p.mode.z);
    let r = over(vec4<f32>(vec3<f32>(br.r), br.g), vec3<f32>(rough), cov.z * k, p.mode.z);
    let bh = textureLoad(bk_height, xy, 0);
    let h = over(vec4<f32>(vec3<f32>(bh.r), bh.g), vec3<f32>(height), cov.w * k, p.mode.z);

    var o: Out;
    o.base = b;
    o.metal = vec4<f32>(m.x, m.a, 0.0, 1.0);
    o.rough = vec4<f32>(r.x, r.a, 0.0, 1.0);
    o.height = vec4<f32>(h.x, h.a, 0.0, 1.0);
    return o;
}
