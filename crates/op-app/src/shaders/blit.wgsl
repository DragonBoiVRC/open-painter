// Fullscreen triangle that copies the offscreen viewport into egui's pass.
// egui sets the pass viewport to the callback rect, so clip space maps to the panel.
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> post: vec4<f32>;   // vignette, grain, bloom, sharpen

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

// The finished picture's look: bloom around bright areas, sharpening, a darker rim, film grain.
fn finish(c: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    var o = c;
    let texel = 1.0 / vec2<f32>(textureDimensions(tex));
    if (post.z > 0.0) {
        var glow = vec3<f32>(0.0);
        for (var k = 0; k < 16; k = k + 1) {
            let a = f32(k) * 2.399963;
            let r = sqrt((f32(k) + 0.5) / 16.0) * 18.0 * texel * (0.5 + post.z);
            let s = textureSampleLevel(tex, samp, uv + vec2<f32>(cos(a), sin(a)) * r, 0.0).rgb;
            glow = glow + s * max(dot(s, vec3<f32>(0.299, 0.587, 0.114)) - 0.7, 0.0);
        }
        o = o + glow / 16.0 * post.z * 3.0;
    }
    if (post.w > 0.0) {
        let avg = 0.25 * (textureSampleLevel(tex, samp, uv + vec2<f32>(texel.x, 0.0), 0.0).rgb
            + textureSampleLevel(tex, samp, uv - vec2<f32>(texel.x, 0.0), 0.0).rgb
            + textureSampleLevel(tex, samp, uv + vec2<f32>(0.0, texel.y), 0.0).rgb
            + textureSampleLevel(tex, samp, uv - vec2<f32>(0.0, texel.y), 0.0).rgb);
        o = o + (c - avg) * post.w * 2.0;
    }
    if (post.x > 0.0) {
        let d = length(uv - vec2<f32>(0.5)) * 1.4142;
        o = o * (1.0 - post.x * smoothstep(0.35, 1.0, d));
    }
    if (post.y > 0.0) {
        let n = fract(sin(dot(uv / texel, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        o = o + (n - 0.5) * post.y * 0.25;
    }
    return max(o, vec3<f32>(0.0));
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, i.uv);
}

// The same copy with FXAA (a compact take on Lottes' FXAA 3.11): edges found from the luma of the
// four diagonal neighbours are blurred along their own direction.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_fxaa(i: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(tex));
    let m = textureSampleLevel(tex, samp, i.uv, 0.0).rgb;
    let nw = textureSampleLevel(tex, samp, i.uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb;
    let ne = textureSampleLevel(tex, samp, i.uv + vec2<f32>(1.0, -1.0) * texel, 0.0).rgb;
    let sw = textureSampleLevel(tex, samp, i.uv + vec2<f32>(-1.0, 1.0) * texel, 0.0).rgb;
    let se = textureSampleLevel(tex, samp, i.uv + vec2<f32>(1.0, 1.0) * texel, 0.0).rgb;
    let l_m = luma(m);
    let l_nw = luma(nw);
    let l_ne = luma(ne);
    let l_sw = luma(sw);
    let l_se = luma(se);
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    // Flat areas are left alone.
    if (l_max - l_min < max(0.0312, l_max * 0.125)) {
        return vec4<f32>(finish(m, i.uv), 1.0);
    }
    var dir = vec2<f32>(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.25 * 0.125, 1.0 / 128.0);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;
    let a = 0.5 * (textureSampleLevel(tex, samp, i.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
        + textureSampleLevel(tex, samp, i.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    let b = a * 0.5 + 0.25 * (textureSampleLevel(tex, samp, i.uv + dir * -0.5, 0.0).rgb
        + textureSampleLevel(tex, samp, i.uv + dir * 0.5, 0.0).rgb);
    let l_b = luma(b);
    if (l_b < l_min || l_b > l_max) {
        return vec4<f32>(finish(a, i.uv), 1.0);
    }
    return vec4<f32>(finish(b, i.uv), 1.0);
}
