// open-painter: quick mask. After a stroke's marks land, every texel the quick mask protects (0) goes
// back to how the pre-stroke snapshot had it; texels it leaves free (1) keep the new paint.

@group(0) @binding(1) var snap_base: texture_2d<f32>;
@group(0) @binding(2) var snap_metal: texture_2d<f32>;
@group(0) @binding(3) var snap_rough: texture_2d<f32>;
@group(0) @binding(4) var snap_height: texture_2d<f32>;
@group(0) @binding(5) var snap_emissive: texture_2d<f32>;
@group(0) @binding(6) var snap_opacity: texture_2d<f32>;
@group(0) @binding(7) var cur_base: texture_2d<f32>;
@group(0) @binding(8) var cur_metal: texture_2d<f32>;
@group(0) @binding(9) var cur_rough: texture_2d<f32>;
@group(0) @binding(10) var cur_height: texture_2d<f32>;
@group(0) @binding(11) var cur_emissive: texture_2d<f32>;
@group(0) @binding(12) var cur_opacity: texture_2d<f32>;
@group(0) @binding(13) var quick: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

struct FsOut {
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
    @location(4) emissive: vec4<f32>,
    @location(5) opacity: vec4<f32>,
};

@fragment
fn fs_content(@builtin(position) pos: vec4<f32>) -> FsOut {
    let xy = vec2<i32>(pos.xy);
    let q = textureLoad(quick, xy, 0).r;
    var o: FsOut;
    o.base = mix(textureLoad(snap_base, xy, 0), textureLoad(cur_base, xy, 0), q);
    o.metal = mix(textureLoad(snap_metal, xy, 0), textureLoad(cur_metal, xy, 0), q);
    o.rough = mix(textureLoad(snap_rough, xy, 0), textureLoad(cur_rough, xy, 0), q);
    o.height = mix(textureLoad(snap_height, xy, 0), textureLoad(cur_height, xy, 0), q);
    o.emissive = mix(textureLoad(snap_emissive, xy, 0), textureLoad(cur_emissive, xy, 0), q);
    o.opacity = mix(textureLoad(snap_opacity, xy, 0), textureLoad(cur_opacity, xy, 0), q);
    return o;
}

// Masks: binding 1 snapshot, 7 current (so the same layout numbering serves both).
@fragment
fn fs_mask(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(pos.xy);
    let q = textureLoad(quick, xy, 0).r;
    return vec4<f32>(mix(textureLoad(snap_base, xy, 0).r, textureLoad(cur_base, xy, 0).r, q), 0.0, 0.0, 1.0);
}
