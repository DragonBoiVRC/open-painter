// Box-filtered downscale of a layer texture to a small preview, converted for display.
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

const TAPS: i32 = 8;

fn average(frag: vec2<f32>, out_size: f32) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src));
    // This output pixel covers a square of source texels; sample it on an even grid.
    let cell = dims / out_size;
    let origin = floor(frag) * cell;
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < TAPS; y++) {
        for (var x = 0; x < TAPS; x++) {
            let at = origin + (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(TAPS) * cell;
            sum += textureLoad(src, vec2<i32>(min(at, dims - 1.0)), 0);
        }
    }
    return sum / f32(TAPS * TAPS);
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// Premultiplied linear color in, premultiplied sRGB out.
@fragment
fn fs_base(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let c = average(p.xy, 32.0);
    let rgb = select(vec3<f32>(0.0), c.rgb / c.a, c.a > 1e-4);
    return vec4<f32>(to_srgb(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0))) * c.a, c.a);
}

@fragment
fn fs_mask(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let v = average(p.xy, 32.0).r;
    return vec4<f32>(v, v, v, 1.0);
}
