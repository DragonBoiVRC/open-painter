// open-painter: original metallic/roughness PBR (Cook-Torrance GGX).
// M0: one directional light + analytic hemisphere ambient. IBL comes later.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,      // direction *towards* the light, xyz
    base_color: vec4<f32>,     // linear rgb
    params: vec4<f32>,         // x = metallic, y = roughness
    env: vec4<f32>,            // rotation (rad), exposure scale, direct light intensity, preset
    inv_view_proj: mat4x4<f32>,
    bg: vec4<f32>,             // x: show the environment, y: blur 0..1, z: tone mapping (0 filmic, 1 Reinhard, 2 none)
    clip: vec4<f32>,           // section plane: cut where dot(position, xyz) > w (all zero: none)
    forward: vec4<f32>,        // where the camera looks (parallel rays when camera_pos.w is 0)
};

@group(0) @binding(0) var<uniform> g: Globals;

// Paint layer, premultiplied. final = painted + fill * (1 - coverage).
@group(1) @binding(0) var paint_base: texture_2d<f32>;
@group(1) @binding(1) var paint_metal: texture_2d<f32>;
@group(1) @binding(2) var paint_rough: texture_2d<f32>;
@group(1) @binding(3) var paint_samp: sampler;
// Tangent-space normal map from painted height and the baked normal, packed in 0..1.
@group(1) @binding(4) var paint_normal: texture_2d<f32>;
// UV offset of the tile this draw shows (0, 0 unless UV tiles are in use).
@group(1) @binding(5) var<uniform> tile: vec4<f32>;
// Emitted light, premultiplied linear colour (black where nothing is painted).
@group(1) @binding(6) var paint_emissive: texture_2d<f32>;
// (opacity * coverage, coverage); the surface is opaque where nothing is painted.
@group(1) @binding(7) var paint_opacity: texture_2d<f32>;
// 1 where the user hid that part of the model.
@group(1) @binding(8) var paint_hidden: texture_2d<f32>;

// Baked map preview. light_dir.w selects the mode: 0 shaded, 1 show the map as-is,
// 2 shaded with the map (ambient occlusion) multiplied into the ambient light, 3 the painted
// normal map as colour.
@group(2) @binding(0) var preview_tex: texture_2d<f32>;
@group(2) @binding(1) var preview_samp: sampler;

// The HDR environment: a lat-long picture with mips (rougher reflections read blurrier mips) and
// diffuse spherical harmonics. info.y is 1 when a picture is loaded.
struct EnvParams {
    sh: array<vec4<f32>, 9>,
    info: vec4<f32>,
};
@group(3) @binding(0) var<uniform> envp: EnvParams;
@group(3) @binding(1) var env_tex: texture_2d<f32>;
@group(3) @binding(2) var env_samp: sampler;

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
    let mode = i32(g.bg.z + 0.5);
    if (mode == 1) {
        let r = c * 0.8;
        return clamp(r / (r + vec3<f32>(1.0)) * 1.1, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (mode == 2) {
        return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let x = c * 0.6;
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}


// A lobe of light around `dir`: sharp for mirrors, broad (and dimmer at the centre) when blurred.
fn lobe(d: vec3<f32>, dir: vec3<f32>, color: vec3<f32>, sharp: f32, blur: f32) -> vec3<f32> {
    let k = mix(sharp, 2.0, blur);
    // A broader lobe carries the same energy over more of the sphere, so its peak is lower.
    return color * exp((dot(d, normalize(dir)) - 1.0) * k) * (k / sharp);
}

// The procedural studio environments. `blur` runs from 0 (mirror) to 1 (diffuse).
//   0 studio softboxes, 1 outdoor, 2 sunset, 3 neutral grey
fn environment(dir: vec3<f32>, blur: f32) -> vec3<f32> {
    let c = cos(g.env.x);
    let s = sin(g.env.x);
    let d = vec3<f32>(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    if (envp.info.y > 0.5 && i32(g.env.w + 0.5) == 4) {
        if (blur >= 0.999) {
            // The diffuse light: spherical harmonics of the whole picture.
            let sh = envp.sh;
            let e = sh[0].xyz * 0.282095
                + sh[1].xyz * 0.488603 * d.y + sh[2].xyz * 0.488603 * d.z + sh[3].xyz * 0.488603 * d.x
                + sh[4].xyz * 1.092548 * d.x * d.y + sh[5].xyz * 1.092548 * d.y * d.z
                + sh[6].xyz * 0.315392 * (3.0 * d.z * d.z - 1.0)
                + sh[7].xyz * 1.092548 * d.x * d.z + sh[8].xyz * 0.546274 * (d.x * d.x - d.y * d.y);
            return max(e, vec3<f32>(0.0));
        }
        let uv = vec2<f32>(atan2(d.z, d.x) / 6.2831853 + 0.5, acos(clamp(d.y, -1.0, 1.0)) / 3.1415927);
        let lod = clamp(blur, 0.0, 1.0) * max(envp.info.x - 1.0, 0.0);
        return textureSampleLevel(env_tex, env_samp, uv, lod).rgb;
    }
    let preset = i32(g.env.w + 0.5);
    let up = d.y * 0.5 + 0.5;
    if (preset == 1) {
        let sky = mix(vec3<f32>(0.55, 0.60, 0.65), vec3<f32>(0.25, 0.45, 0.85), pow(up, 0.7));
        let ground = vec3<f32>(0.20, 0.17, 0.13);
        let base = mix(ground, sky, smoothstep(0.42, 0.58, up));
        return base * 1.2 + lobe(d, vec3<f32>(0.5, 0.8, 0.3), vec3<f32>(9.0, 8.5, 7.5), 60.0, blur);
    }
    if (preset == 2) {
        let sky = mix(vec3<f32>(1.0, 0.45, 0.18), vec3<f32>(0.18, 0.22, 0.45), pow(up, 0.8));
        let ground = vec3<f32>(0.10, 0.07, 0.06);
        let base = mix(ground, sky, smoothstep(0.40, 0.56, up)) * 0.9;
        return base + lobe(d, vec3<f32>(-0.7, 0.18, 0.5), vec3<f32>(12.0, 5.0, 1.6), 50.0, blur);
    }
    if (preset == 3) {
        return vec3<f32>(0.55);
    }
    let base = mix(vec3<f32>(0.10, 0.10, 0.11), vec3<f32>(0.30, 0.31, 0.33), up);
    return base
        + lobe(d, vec3<f32>(-0.8, 0.7, 0.6), vec3<f32>(5.0, 4.9, 4.6), 14.0, blur)
        + lobe(d, vec3<f32>(0.9, 0.3, -0.4), vec3<f32>(2.2, 2.4, 3.0), 10.0, blur)
        + lobe(d, vec3<f32>(0.0, 0.2, -1.0), vec3<f32>(2.6, 2.4, 2.2), 12.0, blur);
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let uv_t = i.uv - tile.xy;
    // Back faces are dropped (as culling would), except when a section plane opens the model up: then
    // the inside is shown, lit as the surface it is.
    let section_on = dot(g.clip.xyz, g.clip.xyz) > 0.5;
    var n = normalize(i.normal);
    if (!front) { n = -n; }
    // The direction to the camera: from the surface point, or against the view direction when the
    // camera is orthographic.
    let v = select(-normalize(g.forward.xyz), normalize(g.camera_pos.xyz - i.world_pos), g.camera_pos.w > 0.5);
    let l = normalize(g.light_dir.xyz);
    let h = normalize(v + l);

    let pv = textureSample(preview_tex, preview_samp, uv_t);
    let nmap = textureSample(paint_normal, paint_samp, uv_t).rgb;
    // The selected layer's mask, as grey (white reveals).
    if (g.light_dir.w > 19.5) {
        return vec4<f32>(vec3<f32>(pv.r), 1.0);
    }
    if (g.light_dir.w > 2.5 && g.light_dir.w < 3.5) {
        return vec4<f32>(nmap, 1.0);
    }
    // Cotangent frame from screen-space derivatives. The image's +Y is up, and v grows downwards.
    let dp1 = dpdx(i.world_pos);
    let dp2 = dpdy(i.world_pos);
    let duv1 = dpdx(uv_t);
    let duv2 = dpdy(uv_t);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let tan = dp2perp * duv1.x + dp1perp * duv2.x;
    let bit = -(dp2perp * duv1.y + dp1perp * duv2.y);
    let longest = max(dot(tan, tan), dot(bit, bit));
    if (longest > 1e-20) {
        let s = inverseSqrt(longest);
        let tn = nmap * 2.0 - 1.0;
        n = normalize(tan * s * tn.x + bit * s * tn.y + n * tn.z);
    }
    if (g.light_dir.w > 0.5 && g.light_dir.w < 1.5) {
        return vec4<f32>(pv.rgb, 1.0);
    }
    let ao = select(1.0, pv.r, g.light_dir.w >= 1.5 && g.light_dir.w < 2.5);
    let pb = textureSample(paint_base, paint_samp, uv_t);
    let pe = textureSample(paint_emissive, paint_samp, uv_t);
    let po = textureSample(paint_opacity, paint_samp, uv_t).rg;
    let opacity = clamp(po.r + (1.0 - po.g), 0.0, 1.0);
    let pm = textureSample(paint_metal, paint_samp, uv_t);
    let pr = textureSample(paint_rough, paint_samp, uv_t);
    let metallic = pm.r + g.params.x * (1.0 - pm.g);
    let roughness = clamp(pr.r + g.params.y * (1.0 - pr.g), 0.04, 1.0);
    let a = roughness * roughness;
    let albedo = pb.rgb + g.base_color.rgb * (1.0 - pb.a);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    // Single-channel views: the painted value, unlit.
    if (g.light_dir.w > 3.5) {
        let mode = i32(g.light_dir.w + 0.5);
        if (mode == 4) { return vec4<f32>(to_srgb(albedo), 1.0); }
        if (mode == 5) { return vec4<f32>(vec3<f32>(roughness), 1.0); }
        if (mode == 7) { return vec4<f32>(to_srgb(pe.rgb), 1.0); }
        if (mode == 8) { return vec4<f32>(vec3<f32>(opacity), 1.0); }
        return vec4<f32>(vec3<f32>(metallic), 1.0);
    }

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);

    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let spec = d_ggx(n_dot_h, a) * g_smith(n_dot_v, n_dot_l, roughness) * f / (4.0 * n_dot_v * max(n_dot_l, 1e-4));
    let kd = (1.0 - f) * (1.0 - metallic);
    let radiance = vec3<f32>(3.0 * g.env.z);
    var color = (kd * albedo / PI + spec) * radiance * n_dot_l;

    // Image-based light from the procedural environment: broad lobes for diffuse, sharper ones
    // for the mirror direction as roughness drops.
    let r = reflect(-v, n);
    let diffuse_env = environment(n, 1.0);
    let spec_env = environment(r, roughness);
    let f_amb = fresnel_schlick(n_dot_v, f0) * (1.0 - roughness * 0.7);
    color += (kd * albedo * diffuse_env + spec_env * f_amb) * ao;
    color += pe.rgb;

    // See-through paint is shown dithered: the surface is dropped from the picture (and its depth) in
    // proportion to how transparent it is. Switched off with bg.w = 0.
    let hidden = textureSample(paint_hidden, paint_samp, uv_t).r > 0.5;
    if (hidden || (!front && !section_on) || (section_on && dot(i.world_pos, g.clip.xyz) > g.clip.w)) {
        discard;
    }
    if (g.bg.w > 0.5 && opacity < 0.999) {
        let noise = fract(sin(dot(i.clip.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        if (opacity <= noise) { discard; }
    }

    return vec4<f32>(to_srgb(tonemap(color * g.env.y)), 1.0);
}
