// open-painter: change a texture's resolution. Enlarging reads the source with bilinear filtering;
// shrinking averages the source texels each output pixel covers (a small grid of bilinear taps).
// Layer textures are premultiplied, so averaging them is exact.

struct Params {
    dst: vec4<f32>,   // x: the output size in texels
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_dims = vec2<f32>(textureDimensions(src));
    let cell = src_dims / p.dst.x;
    let taps = i32(clamp(ceil(max(cell.x, cell.y)), 1.0, 8.0));
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < taps; y++) {
        for (var x = 0; x < taps; x++) {
            let offset = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(taps);
            let uv = (floor(pos.xy) + offset) / p.dst.x;
            sum += textureSampleLevel(src, samp, uv, 0.0);
        }
    }
    return sum / f32(taps * taps);
}
