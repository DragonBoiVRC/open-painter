// open-painter: the HDR environment as the backdrop of the 3D view. Drawn first, with no depth, so
// the model covers it. Same exposure and tone mapping as the model.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
    env: vec4<f32>,            // rotation (rad), exposure scale, direct light intensity, preset
    inv_view_proj: mat4x4<f32>,
    bg: vec4<f32>,             // x: show the environment, y: blur 0..1, z: tone mapping
};

struct EnvParams {
    sh: array<vec4<f32>, 9>,
    info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var<uniform> envp: EnvParams;
@group(1) @binding(1) var env_tex: texture_2d<f32>;
@group(1) @binding(2) var env_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.clip = vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
    o.ndc = q * 2.0 - 1.0;
    return o;
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let mode = i32(g.bg.z + 0.5);
    if (mode == 1) {
        let x = c * 0.8;
        return clamp(x / (x + vec3<f32>(1.0)) * 1.1, vec3<f32>(0.0), vec3<f32>(1.0));
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

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - g.camera_pos.xyz);
    let c = cos(g.env.x);
    let s = sin(g.env.x);
    let d = vec3<f32>(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    let uv = vec2<f32>(atan2(d.z, d.x) / 6.2831853 + 0.5, acos(clamp(d.y, -1.0, 1.0)) / 3.1415927);
    let lod = clamp(g.bg.y, 0.0, 1.0) * max(envp.info.x - 1.0, 0.0);
    let rgb = textureSampleLevel(env_tex, env_samp, uv, lod).rgb;
    return vec4<f32>(to_srgb(tonemap(rgb * g.env.y)), 1.0);
}
