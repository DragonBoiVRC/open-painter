// open-painter: original metallic/roughness PBR (Cook-Torrance GGX).
// M0: one directional light + analytic hemisphere ambient. IBL comes later.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,      // direction *towards* the light, xyz
    base_color: vec4<f32>,     // linear rgb
    params: vec4<f32>,         // x = metallic, y = roughness
};

@group(0) @binding(0) var<uniform> g: Globals;

// Paint layer, premultiplied. final = painted + fill * (1 - coverage).
@group(1) @binding(0) var paint_base: texture_2d<f32>;
@group(1) @binding(1) var paint_metal: texture_2d<f32>;
@group(1) @binding(2) var paint_rough: texture_2d<f32>;
@group(1) @binding(3) var paint_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(v.position, 1.0);
    o.world_pos = v.position;
    o.normal = v.normal;
    o.uv = v.uv;
    return o;
}

const PI: f32 = 3.14159265358979;

fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn g_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

// Narkowicz ACES approximation, then linear -> sRGB.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = c * 0.6;
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(i.normal);
    let v = normalize(g.camera_pos.xyz - i.world_pos);
    let l = normalize(g.light_dir.xyz);
    let h = normalize(v + l);

    let pb = textureSample(paint_base, paint_samp, i.uv);
    let pm = textureSample(paint_metal, paint_samp, i.uv);
    let pr = textureSample(paint_rough, paint_samp, i.uv);
    let metallic = pm.r + g.params.x * (1.0 - pm.g);
    let roughness = clamp(pr.r + g.params.y * (1.0 - pr.g), 0.04, 1.0);
    let a = roughness * roughness;
    let albedo = pb.rgb + g.base_color.rgb * (1.0 - pb.a);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);

    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let spec = d_ggx(n_dot_h, a) * g_smith(n_dot_v, n_dot_l, roughness) * f / (4.0 * n_dot_v * max(n_dot_l, 1e-4));
    let kd = (1.0 - f) * (1.0 - metallic);
    let radiance = vec3<f32>(3.0);
    var color = (kd * albedo / PI + spec) * radiance * n_dot_l;

    // Hemisphere ambient with a crude Fresnel-weighted specular term.
    let sky = vec3<f32>(0.45, 0.55, 0.70);
    let ground = vec3<f32>(0.12, 0.10, 0.09);
    let hemi = mix(ground, sky, n.y * 0.5 + 0.5);
    let r = reflect(-v, n);
    let env_spec = mix(ground, sky, r.y * 0.5 + 0.5);
    let f_amb = fresnel_schlick(n_dot_v, f0) * (1.0 - roughness * 0.7);
    color += kd * albedo * hemi + env_spec * f_amb;

    return vec4<f32>(to_srgb(tonemap(color)), 1.0);
}
