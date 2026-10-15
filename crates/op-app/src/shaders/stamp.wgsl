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
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) brush: vec4<f32>,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) values: vec4<f32>,
    @location(5) @interpolate(flat) enables: vec4<f32>,
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
    return o;
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

// Returns the stamp's alpha at this texel, or -1.0 if the texel is rejected
// (outside the brush, back-facing, or hidden behind nearer geometry).
fn stamp_alpha(i: VsOut) -> f32 {
    let clip = sg.view_proj * vec4<f32>(i.world_pos, 1.0);
    if (clip.w <= 0.0) { return -1.0; }
    let ndc = clip.xyz / clip.w;
    if (ndc.z < 0.0 || ndc.z > 1.0) { return -1.0; }

    let size = sg.viewport.xy;
    let px = vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
    if (px.x < 0.0 || px.y < 0.0 || px.x >= size.x || px.y >= size.y) { return -1.0; }

    let d = distance(px, i.brush.xy) / max(i.brush.z, 1e-3);
    if (d >= 1.0) { return -1.0; }

    // Back-facing surfaces never take paint.
    let to_cam = normalize(sg.camera_pos.xyz - i.world_pos);
    if (dot(normalize(i.normal), to_cam) <= 0.0) { return -1.0; }

    // Occlusion against what the camera actually sees (previous viewport depth).
    let seen = textureLoad(depth_tex, vec2<i32>(px), 0);
    let ours = linear_depth(ndc.z);
    let theirs = linear_depth(seen);
    if (ours > theirs + theirs * 0.004) { return -1.0; }

    let t = clamp((1.0 - d) / max(1.0 - i.brush.w, 1e-3), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t) * i.values.z;
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
