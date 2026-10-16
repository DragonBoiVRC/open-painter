// open-painter: brush stamping directly in UV space (the 2D view).
//
// One instance per stamp: a quad around the dab, in texture space. `brush` holds the centre
// (u, v), the radius in UV units and the hardness. Blending and outputs match stamp.wgsl.

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(3) brush: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) values: vec4<f32>,
    @location(6) enables: vec4<f32>,
    @location(7) shape: vec4<f32>,
    @location(8) emissive: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) brush: vec4<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) values: vec4<f32>,
    @location(4) @interpolate(flat) enables: vec4<f32>,
    @location(5) @interpolate(flat) shape: vec4<f32>,
    @location(6) @interpolate(flat) emissive: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    // 1.5 radii: a rotated square reaches 1.42 radii from the centre.
    let p = v.brush.xy + corners[v.vi] * v.brush.z * 1.5;
    var o: VsOut;
    // Texture space has v pointing down; clip space has y pointing up.
    o.clip = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = p;
    o.brush = v.brush;
    o.color = v.color;
    o.values = v.values;
    o.enables = v.enables;
    o.shape = v.shape;
    o.emissive = v.emissive;
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
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
    @location(4) emissive: vec4<f32>,
};

fn stamp_alpha(i: VsOut) -> f32 {
    let d = shape_dist((i.uv - i.brush.xy) / max(i.brush.z, 1e-6), i.shape);
    if (d >= 1.0) { return -1.0; }
    let t = clamp((1.0 - d) / max(1.0 - i.brush.w, 1e-3), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t) * i.values.z;
}

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
    let ae = a * i.emissive.w;
    o.emissive = vec4<f32>(i.emissive.rgb * ae, ae);
    return o;
}

@fragment
fn fs_mask(i: VsOut) -> @location(0) vec4<f32> {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    return vec4<f32>(i.values.x, 0.0, 0.0, a);
}

// Eraser: the pipeline multiplies what is there by (1 - alpha), so only the alpha matters.
// `enables` picks the channels that lose paint.
@fragment
fn fs_erase(i: VsOut) -> FsOut {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    var o: FsOut;
    o.base = vec4<f32>(0.0, 0.0, 0.0, a * i.enables.x);
    o.metal = vec4<f32>(0.0, 0.0, 0.0, a * i.enables.y);
    o.rough = vec4<f32>(0.0, 0.0, 0.0, a * i.enables.z);
    o.height = vec4<f32>(0.0, 0.0, 0.0, a * i.enables.w);
    o.emissive = vec4<f32>(0.0, 0.0, 0.0, a * i.emissive.w);
    return o;
}
