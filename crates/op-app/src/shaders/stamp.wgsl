// open-painter: projective brush stamping.
//
// The mesh is rasterised in UV space, so every fragment is one texel of the paint
// layer. Each texel knows its world position; we project it into the camera view
// and test it against the brush circle (screen space), then reject texels that are
// back-facing or hidden behind nearer geometry using the viewport depth buffer.
// Instancing = one instance per stamp, drawn in order so overlaps blend sequentially.

struct StampGlobals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    viewport: vec4<f32>,   // width px, height px, near, far
    tile: vec4<f32>,       // UV offset of the tile being painted
};

@group(0) @binding(0) var<uniform> sg: StampGlobals;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) brush: vec4<f32>,    // center px (xy), radius px, hardness
    @location(4) color: vec4<f32>,    // linear rgb
    @location(5) values: vec4<f32>,   // metallic, roughness, opacity, height
    @location(6) enables: vec4<f32>,  // base, metallic, roughness, height (0 or 1)
    @location(7) shape: vec4<f32>,    // angle, kind, seed
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) brush: vec4<f32>,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) values: vec4<f32>,
    @location(5) @interpolate(flat) enables: vec4<f32>,
    @location(6) @interpolate(flat) shape: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    // Texture space has v pointing down; clip space has y pointing up.
    let uv = v.uv - sg.tile.xy;
    o.clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.world_pos = v.position;
    o.normal = v.normal;
    o.brush = v.brush;
    o.color = v.color;
    o.values = v.values;
    o.enables = v.enables;
    o.shape = v.shape;
    return o;
}


fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// Distance from the brush centre, in brush radii, for the stamp's shape. `off` is the offset
// from the centre in radii (screen or UV axes); `shape` is (angle, kind, seed, _).
//   0 round, 1 square, 2 diamond, 3 chalk (noisy edge), 4 streaks (stretched noise)
fn shape_dist(off: vec2<f32>, shape: vec4<f32>) -> f32 {
    let c = cos(shape.x);
    let s = sin(shape.x);
    let l = vec2<f32>(c * off.x + s * off.y, -s * off.x + c * off.y);
    let kind = i32(shape.y + 0.5);
    if (kind == 1) { return max(abs(l.x), abs(l.y)); }
    if (kind == 2) { return abs(l.x) + abs(l.y); }
    let r = length(l);
    if (kind == 3) { return r + (vnoise(l * 5.0 + shape.z) - 0.5) * 0.7; }
    if (kind == 4) { return r + (vnoise(vec2<f32>(l.x * 1.5, l.y * 14.0) + shape.z) - 0.5) * 0.9; }
    return r;
}

struct FsOut {
    @location(0) base: vec4<f32>,    // premultiplied rgb + coverage
    @location(1) metal: vec4<f32>,   // (value * coverage, coverage)
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
};

fn linear_depth(z: f32) -> f32 {
    let n = sg.viewport.z;
    let f = sg.viewport.w;
    return n * f / (f - z * (f - n));
}

// The mirror image of a model-space point (or direction) across the stamp's symmetry plane:
// `shape.w` is the axis (1 x, 2 y, 3 z; 0 for none) and `color.w` the plane's position on it.
fn mirror_point(p: vec3<f32>, axis: i32, c: f32) -> vec3<f32> {
    if (axis == 1) { return vec3<f32>(2.0 * c - p.x, p.y, p.z); }
    if (axis == 2) { return vec3<f32>(p.x, 2.0 * c - p.y, p.z); }
    return vec3<f32>(p.x, p.y, 2.0 * c - p.z);
}

fn mirror_dir(n: vec3<f32>, axis: i32) -> vec3<f32> {
    if (axis == 1) { return vec3<f32>(-n.x, n.y, n.z); }
    if (axis == 2) { return vec3<f32>(n.x, -n.y, n.z); }
    return vec3<f32>(n.x, n.y, -n.z);
}

// The stamp's alpha for a surface point, or -1.0 if it is rejected (outside the brush,
// back-facing, or hidden behind nearer geometry).
fn alpha_at(i: VsOut, world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let clip = sg.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) { return -1.0; }
    let ndc = clip.xyz / clip.w;
    if (ndc.z < 0.0 || ndc.z > 1.0) { return -1.0; }

    let size = sg.viewport.xy;
    let px = vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
    if (px.x < 0.0 || px.y < 0.0 || px.x >= size.x || px.y >= size.y) { return -1.0; }

    let d = shape_dist((px - i.brush.xy) / max(i.brush.z, 1e-3), i.shape);
    if (d >= 1.0) { return -1.0; }

    // Back-facing surfaces never take paint.
    let to_cam = normalize(sg.camera_pos.xyz - world_pos);
    if (dot(normalize(normal), to_cam) <= 0.0) { return -1.0; }

    // Occlusion against what the camera actually sees (previous viewport depth).
    let seen = textureLoad(depth_tex, vec2<i32>(px), 0);
    let ours = linear_depth(ndc.z);
    let theirs = linear_depth(seen);
    if (ours > theirs + theirs * 0.004) { return -1.0; }

    let t = clamp((1.0 - d) / max(1.0 - i.brush.w, 1e-3), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t) * i.values.z;
}

// With symmetry on, a texel is painted when either it or its mirror image is under the brush
// and visible, which paints the mirrored stroke on a symmetric model.
fn stamp_alpha(i: VsOut) -> f32 {
    let a = alpha_at(i, i.world_pos, i.normal);
    let axis = i32(i.shape.w + 0.5);
    if (axis == 0) { return a; }
    let b = alpha_at(i, mirror_point(i.world_pos, axis, i.color.w), mirror_dir(i.normal, axis));
    return max(a, b);
}

// Content stamp: writes premultiplied base/metal/rough with "over" blending.
@fragment
fn fs_main(i: VsOut) -> FsOut {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    let ab = a * i.enables.x;
    let am = a * i.enables.y;
    let ar = a * i.enables.z;
    let ah = a * i.enables.w;
    var o: FsOut;
    o.base = vec4<f32>(i.color.rgb * ab, ab);
    o.metal = vec4<f32>(i.values.x * am, am, 0.0, am);
    o.rough = vec4<f32>(i.values.y * ar, ar, 0.0, ar);
    o.height = vec4<f32>(i.values.w * ah, ah, 0.0, ah);
    return o;
}

// Mask stamp: values.x is the mask value to paint (1 reveals, 0 hides); the
// pipeline blends with SrcAlpha / OneMinusSrcAlpha.
@fragment
fn fs_mask(i: VsOut) -> @location(0) vec4<f32> {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    return vec4<f32>(i.values.x, 0.0, 0.0, a);
}

// Smudge and clone stamps copy paint from another place on the surface: the point `color.xy`
// pixels away on screen. `src_uv` is the UV of what the camera saw at each pixel (-1 where
// nothing was drawn) and the other textures are the layer as it was before this batch of stamps.
@group(1) @binding(0) var src_uv: texture_2d<f32>;
@group(1) @binding(1) var src_base: texture_2d<f32>;
@group(1) @binding(2) var src_metal: texture_2d<f32>;
@group(1) @binding(3) var src_rough: texture_2d<f32>;
@group(1) @binding(4) var src_height: texture_2d<f32>;
@group(1) @binding(5) var src_samp: sampler;

@fragment
fn fs_smudge(i: VsOut) -> FsOut {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    let clip = sg.view_proj * vec4<f32>(i.world_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let size = sg.viewport.xy;
    let px = vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
    let sp = px + i.color.xy;
    if (sp.x < 0.0 || sp.y < 0.0 || sp.x >= size.x || sp.y >= size.y) { discard; }
    let uv_dims = vec2<f32>(textureDimensions(src_uv));
    let suv = textureLoad(src_uv, vec2<i32>(sp * uv_dims / size), 0).xy;
    if (suv.x < -0.5) { discard; }
    let t = suv - sg.tile.xy;
    if (t.x < 0.0 || t.y < 0.0 || t.x > 1.0 || t.y > 1.0) { discard; }
    let b = textureSampleLevel(src_base, src_samp, t, 0.0);
    let m = textureSampleLevel(src_metal, src_samp, t, 0.0).rg;
    let r = textureSampleLevel(src_rough, src_samp, t, 0.0).rg;
    let h = textureSampleLevel(src_height, src_samp, t, 0.0).rg;
    var o: FsOut;
    o.base = b * (a * i.enables.x);
    o.metal = vec4<f32>(m * (a * i.enables.y), 0.0, m.y * (a * i.enables.y));
    o.rough = vec4<f32>(r * (a * i.enables.z), 0.0, r.y * (a * i.enables.z));
    o.height = vec4<f32>(h * (a * i.enables.w), 0.0, h.y * (a * i.enables.w));
    return o;
}
