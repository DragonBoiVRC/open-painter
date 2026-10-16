// open-painter: the mesh's edges over the shaded model. The lines are pulled a little toward the
// camera so they are not eaten by the surface they lie on, and hidden behind nearer geometry.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
    env: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
    bg: vec4<f32>,
    clip: vec4<f32>,           // section plane: cut where dot(position, xyz) > w (all zero: none)
    forward: vec4<f32>,        // where the camera looks (parallel rays when camera_pos.w is 0)
    ssao: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
// The paint layer's material group, for the parts the user hid and the UV tile offset.
@group(1) @binding(5) var<uniform> tile: vec4<f32>;
@group(1) @binding(8) var hidden_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    var c = g.view_proj * vec4<f32>(position, 1.0);
    c.z = c.z - 0.0008 * c.w;
    o.clip = c;
    o.world_pos = position;
    o.uv = uv;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let local = i.uv - tile.xy;
    let hdims = vec2<f32>(textureDimensions(hidden_tex));
    let hidden = textureLoad(hidden_tex, vec2<i32>(clamp(local * hdims, vec2<f32>(0.0), hdims - vec2<f32>(1.0))), 0).r > 0.5;
    if (hidden || (dot(g.clip.xyz, g.clip.xyz) > 0.5 && dot(i.world_pos, g.clip.xyz) > g.clip.w)) {
        discard;
    }
    return vec4<f32>(0.02, 0.02, 0.03, 0.55);
}
