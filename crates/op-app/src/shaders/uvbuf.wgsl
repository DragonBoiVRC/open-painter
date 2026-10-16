// open-painter: UV buffer. Draws the model with each pixel holding the UV of the surface there,
// so smudge and clone strokes can find where a screen position lands on the texture.

struct Globals {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(v.position, 1.0);
    o.uv = v.uv;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec2<f32> {
    return i.uv;
}
