// open-painter: UV buffer. Draws the model with each pixel holding the UV of the surface there,
// so smudge and clone strokes can find where a screen position lands on the texture.

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
// The paint layer's material group, for the parts the user hid and the UV tile offset.
@group(1) @binding(5) var<uniform> tile: vec4<f32>;
@group(1) @binding(8) var hidden_tex: texture_2d<f32>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(v.position, 1.0);
    o.uv = v.uv;
    o.world_pos = v.position;
    return o;
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec2<f32> {
    let section_on = dot(g.clip.xyz, g.clip.xyz) > 0.5;
    // Culled back faces, unless a section plane shows the inside; cut-away surfaces are skipped.
    let local = i.uv - tile.xy;
    let hdims = vec2<f32>(textureDimensions(hidden_tex));
    let hidden = textureLoad(hidden_tex, vec2<i32>(clamp(local * hdims, vec2<f32>(0.0), hdims - vec2<f32>(1.0))), 0).r > 0.5;
    if (hidden || (!front && !section_on) || (section_on && dot(i.world_pos, g.clip.xyz) > g.clip.w)) {
        discard;
    }
    return i.uv;
}
