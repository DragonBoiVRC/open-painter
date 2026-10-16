// open-painter: stroke opacity. While a stroke is limited to an opacity, its stamps accumulate in a
// separate buffer and every frame the layer is rebuilt as the pre-stroke snapshot with that buffer
// laid over it at the stroke opacity, so overlapping dabs never add up beyond the limit.

struct Params {
    v: vec4<f32>,   // x: stroke opacity, y: mask value being painted
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var snap_base: texture_2d<f32>;
@group(0) @binding(2) var snap_metal: texture_2d<f32>;
@group(0) @binding(3) var snap_rough: texture_2d<f32>;
@group(0) @binding(4) var snap_height: texture_2d<f32>;
@group(0) @binding(5) var buf_base: texture_2d<f32>;
@group(0) @binding(6) var buf_metal: texture_2d<f32>;
@group(0) @binding(7) var buf_rough: texture_2d<f32>;
@group(0) @binding(8) var buf_height: texture_2d<f32>;

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
};

// Premultiplied colour: stroke * k over snapshot.
fn over_color(snap: vec4<f32>, buf: vec4<f32>) -> vec4<f32> {
    let k = p.v.x;
    return buf * k + snap * (1.0 - buf.a * k);
}

// Premultiplied value (value * coverage, coverage): coverage lives in the green channel.
fn over_value(snap: vec4<f32>, buf: vec4<f32>) -> vec4<f32> {
    let k = p.v.x;
    return vec4<f32>(buf.rg * k + snap.rg * (1.0 - buf.g * k), 0.0, 0.0);
}

@fragment
fn fs_content(@builtin(position) pos: vec4<f32>) -> FsOut {
    let xy = vec2<i32>(pos.xy);
    var o: FsOut;
    o.base = over_color(textureLoad(snap_base, xy, 0), textureLoad(buf_base, xy, 0));
    o.metal = over_value(textureLoad(snap_metal, xy, 0), textureLoad(buf_metal, xy, 0));
    o.rough = over_value(textureLoad(snap_rough, xy, 0), textureLoad(buf_rough, xy, 0));
    o.height = over_value(textureLoad(snap_height, xy, 0), textureLoad(buf_height, xy, 0));
    return o;
}
