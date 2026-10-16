// open-painter: the mesh's edges over the shaded model. The lines are pulled a little toward the
// camera so they are not eaten by the surface they lie on, and hidden behind nearer geometry.

struct Globals {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> {
    var c = g.view_proj * vec4<f32>(position, 1.0);
    c.z = c.z - 0.0008 * c.w;
    return c;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.02, 0.02, 0.03, 0.55);
}
