// open-painter: projected fill layers. Renders a fill layer whose textures are projected onto the model
// (triplanar or along one axis) instead of laid over the UVs, using the baked Position and World space
// normal maps. The result is an ordinary layer that the compositor then blends like any other.
//
// All outputs are premultiplied: (value * coverage, coverage).

struct Params {
    color: vec4<f32>,      // constant base colour, linear
    vals: vec4<f32>,       // _, metallic, roughness, height constants
    emissive: vec4<f32>,   // constant emissive colour, linear; w is 1 if it is written
    opac: vec4<f32>,       // constant opacity; y is 1 if it is written
    bits: vec4<u32>,       // x: constant channels written (1 base, 2 metal, 4 rough, 8 height),
                           // y: channels read from an image (bit per slot: 1 base, 2 metal, 4 rough, 8 height, 32 emissive, 64 opacity),
                           // z: inverted scalar channels (same bits), w: projection (1 triplanar, 2 X, 3 Y, 4 Z)
    xf: vec4<f32>,         // tiling x, y, offset x, y
    xf2: vec4<f32>,        // cos and sin of the rotation, triplanar sharpness, _
    extent: vec4<f32>,     // the model's size along x, y, z with the largest axis as 1
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var pos_tex: texture_2d<f32>;
@group(0) @binding(2) var nrm_tex: texture_2d<f32>;
@group(0) @binding(3) var img_base: texture_2d<f32>;
@group(0) @binding(4) var img_metal: texture_2d<f32>;
@group(0) @binding(5) var img_rough: texture_2d<f32>;
@group(0) @binding(6) var img_height: texture_2d<f32>;
@group(0) @binding(7) var img_emissive: texture_2d<f32>;
@group(0) @binding(8) var img_samp: sampler;
@group(0) @binding(9) var img_opacity: texture_2d<f32>;

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
    @location(4) emissive: vec4<f32>,
    @location(5) opacity: vec4<f32>,
    @location(6) normal: vec4<f32>,
};

// The picture's coordinate for a point on the plane: turned about the middle, tiled and shifted
// (the same placement the UV mode uses).
fn place(uv: vec2<f32>) -> vec2<f32> {
    let c = uv - vec2<f32>(0.5);
    let r = vec2<f32>(c.x * p.xf2.x - c.y * p.xf2.y, c.x * p.xf2.y + c.y * p.xf2.x);
    return (r + vec2<f32>(0.5)) * p.xf.xy + p.xf.zw;
}

// A point of the model projected onto one of the three planes, upright when seen from outside.
fn plane_uv(axis: u32, q: vec3<f32>) -> vec2<f32> {
    if (axis == 0u) { return vec2<f32>(q.z, 1.0 - q.y); }
    if (axis == 1u) { return vec2<f32>(q.x, q.z); }
    return vec2<f32>(q.x, 1.0 - q.y);
}

// One slot's picture at a point, projected: a single plane, or three blended by `w`.
fn project(t: texture_2d<f32>, q: vec3<f32>, w: vec3<f32>) -> vec4<f32> {
    // Roughly one texel of the baked maps spans this many texels of the picture.
    let density = f32(textureDimensions(t).x) * max(abs(p.xf.x), abs(p.xf.y)) / f32(textureDimensions(pos_tex).x);
    let lod = max(log2(max(density, 1e-3)), 0.0);
    let mode = p.bits.w;
    if (mode >= 2u) {
        return textureSampleLevel(t, img_samp, place(plane_uv(mode - 2u, q)), lod);
    }
    return textureSampleLevel(t, img_samp, place(plane_uv(0u, q)), lod) * w.x
        + textureSampleLevel(t, img_samp, place(plane_uv(1u, q)), lod) * w.y
        + textureSampleLevel(t, img_samp, place(plane_uv(2u, q)), lod) * w.z;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let xy = vec2<i32>(pos.xy);
    let q = textureLoad(pos_tex, xy, 0).rgb * p.extent.xyz;
    let n = textureLoad(nrm_tex, xy, 0).rgb * 2.0 - vec3<f32>(1.0);
    var w = pow(abs(n), vec3<f32>(max(p.xf2.z, 0.1)));
    w = w / max(w.x + w.y + w.z, 1e-6);

    var color = p.color.rgb;
    var cov = vec4<f32>(f32(p.bits.x & 1u), f32((p.bits.x >> 1u) & 1u), f32((p.bits.x >> 2u) & 1u), f32((p.bits.x >> 3u) & 1u));
    var metal = p.vals.y;
    var rough = p.vals.z;
    var height = p.vals.w;
    var e_color = p.emissive.rgb;
    var e_cov = p.emissive.w;
    var o_val = p.opac.x;
    var o_cov = p.opac.y;
    if ((p.bits.y & 1u) != 0u) {
        let s = project(img_base, q, w);
        color = s.rgb;
        cov.x = s.a;
    }
    if ((p.bits.y & 2u) != 0u) {
        let v = project(img_metal, q, w).r;
        metal = select(v, 1.0 - v, (p.bits.z & 2u) != 0u);
        cov.y = 1.0;
    }
    if ((p.bits.y & 4u) != 0u) {
        let v = project(img_rough, q, w).r;
        rough = select(v, 1.0 - v, (p.bits.z & 4u) != 0u);
        cov.z = 1.0;
    }
    if ((p.bits.y & 8u) != 0u) {
        let v = project(img_height, q, w).r;
        height = select(v, 1.0 - v, (p.bits.z & 8u) != 0u);
        cov.w = 1.0;
    }
    if ((p.bits.y & 32u) != 0u) {
        let s = project(img_emissive, q, w);
        e_color = s.rgb;
        e_cov = s.a;
    }
    if ((p.bits.y & 64u) != 0u) {
        let v = project(img_opacity, q, w).r;
        o_val = select(v, 1.0 - v, (p.bits.z & 64u) != 0u);
        o_cov = 1.0;
    }
    var o: Out;
    o.base = vec4<f32>(color * cov.x, cov.x);
    o.metal = vec4<f32>(metal * cov.y, cov.y, 0.0, cov.y);
    o.rough = vec4<f32>(rough * cov.z, cov.z, 0.0, cov.z);
    o.height = vec4<f32>(height * cov.w, cov.w, 0.0, cov.w);
    o.emissive = vec4<f32>(e_color * e_cov, e_cov);
    o.opacity = vec4<f32>(o_val * o_cov, o_cov, 0.0, o_cov);
    // No normal map comes out of a projection; flat, with no coverage.
    o.normal = vec4<f32>(0.5, 0.5, 1.0, 0.0);
    return o;
}
