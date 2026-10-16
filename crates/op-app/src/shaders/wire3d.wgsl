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
};

@group(0) @binding(0) var<uniform> g: Globals;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    var c = g.view_proj * vec4<f32>(position, 1.0);
    c.z = c.z - 0.0008 * c.w;
    o.clip = c;
    o.world_pos = position;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    if (dot(g.clip.xyz, g.clip.xyz) > 0.5 && dot(i.world_pos, g.clip.xyz) > g.clip.w) {
        discard;
    }
    return vec4<f32>(0.02, 0.02, 0.03, 0.55);
}
