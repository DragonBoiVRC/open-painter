// open-painter: layer compositor. One pass composites one layer over a backdrop.
// Same math as op-core/src/blend.rs (W3C compositing, separable blend modes);
// a headless test cross-checks the two.
//
// All textures are premultiplied. Pass dst = over(backdrop, source * opacity * mask).

struct Params {
    color: vec4<f32>,   // fill base color, linear rgb
    vals: vec4<f32>,    // opacity, fill metallic, fill roughness, fill height
    mode: vec4<u32>,    // x: source is fill, y: has mask, z: blend mode, w: fill enable bits (1 base, 2 metal, 4 rough, 8 height)
    tex: vec4<u32>,     // x: channels read from an image (1 base, 2 metal, 4 rough, 8 height, 16 normal),
                        // y: inverted scalar channels (same bits), z: bit 0 flips the normal map's green channel
    xf: vec4<f32>,      // texture tiling x, y and offset x, y
    xf2: vec4<f32>,     // cos and sin of the texture rotation, then the UV offset of this tile (u, v)
    dims: vec4<f32>,    // texture size in pixels
    lv: vec4<f32>,      // mask levels: black, white, gamma, invert
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
@group(0) @binding(10) var bk_normal: texture_2d<f32>;
@group(0) @binding(11) var src_normal: texture_2d<f32>;
// Images feeding a fill layer's channels (sampled with mipmaps; base colour through an sRGB view).
@group(0) @binding(12) var img_base: texture_2d<f32>;
@group(0) @binding(13) var img_metal: texture_2d<f32>;
@group(0) @binding(14) var img_rough: texture_2d<f32>;
@group(0) @binding(15) var img_height: texture_2d<f32>;
@group(0) @binding(16) var img_normal: texture_2d<f32>;
@group(0) @binding(17) var img_samp: sampler;

struct FsIn {
    @builtin(position) pos: vec4<f32>,
    // The pixel's position, as a plain varying so that texture coordinates made from it are affine.
    @location(0) pos_xy: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> FsIn {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FsIn;
    o.pos = vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
    // Pixel coordinates with the origin at the top left, matching the framebuffer.
    o.pos_xy = vec2<f32>(q.x, 1.0 - q.y) * p.dims.x;
    return o;
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
        case 9u: {
            let d = select(sqrt(cb), ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb <= vec3<f32>(0.25));
            let dark = cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
            let light = cb + (2.0 * cs - 1.0) * (d - cb);
            return select(light, dark, cs <= vec3<f32>(0.5));
        }
        case 10u: {
            let light = cb + (2.0 * cs - 1.0) - cb * (2.0 * cs - 1.0);
            return select(light, 2.0 * cb * cs, cs <= vec3<f32>(0.5));
        }
        case 11u: {
            let v = min(cb / max(1.0 - cs, vec3<f32>(1e-6)), vec3<f32>(1.0));
            return select(select(v, vec3<f32>(1.0), cs >= vec3<f32>(1.0)), vec3<f32>(0.0), cb <= vec3<f32>(0.0));
        }
        case 12u: {
            let v = vec3<f32>(1.0) - min((vec3<f32>(1.0) - cb) / max(cs, vec3<f32>(1e-6)), vec3<f32>(1.0));
            return select(select(v, vec3<f32>(0.0), cs <= vec3<f32>(0.0)), vec3<f32>(1.0), cb >= vec3<f32>(1.0));
        }
        case 13u: { return max(cb + cs - vec3<f32>(1.0), vec3<f32>(0.0)); }
        case 14u: { return cb + cs - 2.0 * cb * cs; }
        case 15u: {
            return select(min(cb / max(cs, vec3<f32>(1e-6)), vec3<f32>(1.0)), vec3<f32>(1.0), cs <= vec3<f32>(0.0));
        }
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
    @location(4) normal: vec4<f32>,
};

// The texture coordinate of this pixel: UV position (with the tile offset), turned about the middle
// of the texture, tiled and shifted.
fn tex_uv(frag: vec2<f32>) -> vec2<f32> {
    let uv = frag / p.dims.x + p.xf2.zw;
    let c = uv - vec2<f32>(0.5);
    let r = vec2<f32>(c.x * p.xf2.x - c.y * p.xf2.y, c.x * p.xf2.y + c.y * p.xf2.x);
    return (r + vec2<f32>(0.5)) * p.xf.xy + p.xf.zw;
}

fn decode_normal(v: vec3<f32>, flip_green: bool) -> vec3<f32> {
    var n = v * 2.0 - vec3<f32>(1.0);
    if (flip_green) { n.y = -n.y; }
    return normalize(n);
}

@fragment
fn fs_main(i: FsIn) -> Out {
    let xy = vec2<i32>(i32(i.pos.x), i32(i.pos.y));

    var cov = vec4<f32>(0.0);          // coverage of base, metal, rough, height
    var src_n = vec3<f32>(0.0, 0.0, 1.0);   // the source's normal
    var n_cov = 0.0;                   // its coverage
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
        if (p.tex.x != 0u) {
            let t = tex_uv(i.pos_xy);
            if ((p.tex.x & 1u) != 0u) {
                let s = textureSample(img_base, img_samp, t);
                color = s.rgb;
                cov.x = s.a;
            }
            if ((p.tex.x & 2u) != 0u) {
                let v = textureSample(img_metal, img_samp, t).r;
                metal = select(v, 1.0 - v, (p.tex.y & 2u) != 0u);
                cov.y = 1.0;
            }
            if ((p.tex.x & 4u) != 0u) {
                let v = textureSample(img_rough, img_samp, t).r;
                rough = select(v, 1.0 - v, (p.tex.y & 4u) != 0u);
                cov.z = 1.0;
            }
            if ((p.tex.x & 8u) != 0u) {
                let v = textureSample(img_height, img_samp, t).r;
                height = select(v, 1.0 - v, (p.tex.y & 8u) != 0u);
                cov.w = 1.0;
            }
            if ((p.tex.x & 16u) != 0u) {
                let s = textureSample(img_normal, img_samp, t);
                src_n = decode_normal(s.rgb, (p.tex.z & 1u) != 0u);
                n_cov = 1.0;
            }
        }
    } else {
        let sb = textureLoad(src_base, xy, 0);
        let sm = textureLoad(src_metal, xy, 0);
        let sr = textureLoad(src_rough, xy, 0);
        let sh = textureLoad(src_height, xy, 0);
        cov = vec4<f32>(sb.a, sm.g, sr.g, sh.g);
        height = unpremul(sh.r, sh.g);
        let sn = textureLoad(src_normal, xy, 0);
        src_n = decode_normal(sn.rgb, false);
        n_cov = sn.a;
        color = select(vec3<f32>(0.0), sb.rgb / sb.a, sb.a > 0.0);
        metal = unpremul(sm.r, sm.g);
        rough = unpremul(sr.r, sr.g);
    }

    var mask = 1.0;
    if (p.mode.y == 1u) {
        mask = textureLoad(mask_tex, xy, 0).r;
        // Levels: stretch black..white to 0..1, apply gamma, optionally invert (see MaskAdjust::apply).
        let stretched = clamp((mask - p.lv.x) / max(p.lv.y - p.lv.x, 1e-4), 0.0, 1.0);
        let lifted = pow(stretched, 1.0 / p.lv.z);
        mask = select(lifted, 1.0 - lifted, p.lv.w > 0.5);
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

    // Normals combine by replacing: where the source covers, its normal takes over in proportion to its
    // coverage and the layer's opacity.
    let bn = textureLoad(bk_normal, xy, 0);
    let a_n = n_cov * k;
    let cov_out = a_n + bn.a * (1.0 - a_n);
    var n_out = vec3<f32>(0.0, 0.0, 1.0);
    if (cov_out > 0.0) {
        let mixed = src_n * a_n + (bn.rgb * 2.0 - vec3<f32>(1.0)) * bn.a * (1.0 - a_n);
        n_out = normalize(mixed / cov_out);
    }
    o.normal = vec4<f32>(n_out * 0.5 + vec3<f32>(0.5), cov_out);
    return o;
}
