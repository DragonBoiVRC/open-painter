// open-painter: screen-space ambient occlusion. A fullscreen pass over the finished picture that
// darkens it where the depth buffer shows a crease or two parts meeting: each pixel looks at
// neighbours on a spiral and counts those that lie in front of the plane its own surface extends
// into. The result multiplies the picture (see the blend state), so nothing else needs to know.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,         // _, _, near, far
    env: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
    bg: vec4<f32>,
    clip: vec4<f32>,
    forward: vec4<f32>,
    ssao: vec4<f32>,           // x: strength, y: radius in pixels
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var depth_tex: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

const TAPS: i32 = 14;

// Depth along the view direction, and whether the pixel holds anything (the far plane is empty).
fn view_depth(xy: vec2<i32>) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let c = clamp(xy, vec2<i32>(0), dims - 1);
    let d = textureLoad(depth_tex, c, 0);
    if (d >= 0.99999) { return vec2<f32>(0.0, 0.0); }
    let n = g.params.z;
    let f = g.params.w;
    if (g.camera_pos.w < 0.5) { return vec2<f32>(n + d * (f - n), 1.0); }
    return vec2<f32>(n * f / (f - d * (f - n)), 1.0);
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if (g.ssao.x <= 0.0) { return vec4<f32>(1.0); }
    let xy = vec2<i32>(pos.xy);
    let here = view_depth(xy);
    if (here.y < 0.5) { return vec4<f32>(1.0); }
    let z = here.x;

    // The plane this surface extends into: depth change per pixel, from the gentler side of each axis.
    let zl = view_depth(xy + vec2<i32>(-1, 0)); let zr = view_depth(xy + vec2<i32>(1, 0));
    let zu = view_depth(xy + vec2<i32>(0, -1)); let zd = view_depth(xy + vec2<i32>(0, 1));
    var dzdx = 0.0;
    if (zl.y > 0.5 && zr.y > 0.5) {
        let a = z - zl.x; let b = zr.x - z;
        dzdx = select(b, a, abs(a) < abs(b));
    } else if (zr.y > 0.5) { dzdx = zr.x - z; } else if (zl.y > 0.5) { dzdx = z - zl.x; }
    var dzdy = 0.0;
    if (zu.y > 0.5 && zd.y > 0.5) {
        let a = z - zu.x; let b = zd.x - z;
        dzdy = select(b, a, abs(a) < abs(b));
    } else if (zd.y > 0.5) { dzdy = zd.x - z; } else if (zu.y > 0.5) { dzdy = z - zu.x; }

    let radius = max(g.ssao.y, 2.0);
    let range = z * 0.3;
    let turn = hash(pos.xy) * 6.2831853;
    var occluded = 0.0;
    for (var i = 0; i < TAPS; i++) {
        let t = (f32(i) + 0.5) / f32(TAPS);
        let a = turn + f32(i) * 2.399963;
        let off = vec2<f32>(cos(a), sin(a)) * sqrt(t) * radius;
        let s = view_depth(xy + vec2<i32>(round(off)));
        if (s.y < 0.5) { continue; }
        // How much nearer the neighbour is than the plane would put it.
        let expected = z + dzdx * off.x + dzdy * off.y;
        let nearer = expected - s.x - z * 0.004;
        if (nearer > 0.0 && nearer < range) {
            occluded += 1.0 - nearer / range;
        }
    }
    let ao = 1.0 - clamp(g.ssao.x * occluded / f32(TAPS) * 1.6, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
