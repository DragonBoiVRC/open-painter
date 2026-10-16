// Fullscreen triangle that copies the offscreen viewport into egui's pass.
// egui sets the pass viewport to the callback rect, so clip space maps to the panel.
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
        return vec4<f32>(m, 1.0);
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
        return vec4<f32>(a, 1.0);
    }
    return vec4<f32>(b, 1.0);
}
