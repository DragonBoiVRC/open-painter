// open-painter: geometry fill. Draws chosen triangles in UV space with one flat colour, for filling
// a UV island, a piece of the mesh or a whole texture set.

struct FillParams {
    color: vec4<f32>,     // linear rgb
    values: vec4<f32>,    // metallic, roughness, opacity, height (masks: x is the value to paint)
    enables: vec4<f32>,   // base, metallic, roughness, height (0 or 1)
    tile: vec4<f32>,      // UV offset of the tile being painted
};

@group(0) @binding(0) var<uniform> fp: FillParams;

@vertex
fn vs_main(@location(0) uv: vec2<f32>) -> @builtin(position) vec4<f32> {
    let t = uv - fp.tile.xy;
    return vec4<f32>(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0);
}

struct FsOut {
    @location(0) base: vec4<f32>,
    @location(1) metal: vec4<f32>,
    @location(2) rough: vec4<f32>,
    @location(3) height: vec4<f32>,
};

@fragment
fn fs_main() -> FsOut {
    let a = fp.values.z;
    let ab = a * fp.enables.x;
    let am = a * fp.enables.y;
    let ar = a * fp.enables.z;
    let ah = a * fp.enables.w;
    var o: FsOut;
    o.base = vec4<f32>(fp.color.rgb * ab, ab);
    o.metal = vec4<f32>(fp.values.x * am, am, 0.0, am);
    o.rough = vec4<f32>(fp.values.y * ar, ar, 0.0, ar);
    o.height = vec4<f32>(fp.values.w * ah, ah, 0.0, ah);
    return o;
}

@fragment
fn fs_mask() -> @location(0) vec4<f32> {
    return vec4<f32>(fp.values.x, 0.0, 0.0, fp.values.z);
}
