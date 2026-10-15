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
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) brush: vec4<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) values: vec4<f32>,
    @location(4) @interpolate(flat) enables: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let p = v.brush.xy + corners[v.vi] * v.brush.z;
    var o: VsOut;
    // Texture space has v pointing down; clip space has y pointing up.
    o.clip = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = p;
    o.brush = v.brush;
    o.color = v.color;
    o.values = v.values;
    o.enables = v.enables;
    return o;
}

struct FsOut {
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
};

fn stamp_alpha(i: VsOut) -> f32 {
    let d = distance(i.uv, i.brush.xy) / max(i.brush.z, 1e-6);
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
    return o;
}

@fragment
fn fs_mask(i: VsOut) -> @location(0) vec4<f32> {
    let a = stamp_alpha(i);
    if (a < 0.0) { discard; }
    return vec4<f32>(i.values.x, 0.0, 0.0, a);
}
