// Marks the 64x64 tiles in which any of up to six texture pairs differ.
@group(0) @binding(0) var a0: texture_2d<f32>;
@group(0) @binding(1) var a1: texture_2d<f32>;
@group(0) @binding(2) var a2: texture_2d<f32>;
@group(0) @binding(3) var a3: texture_2d<f32>;
@group(0) @binding(4) var a4: texture_2d<f32>;
@group(0) @binding(5) var a5: texture_2d<f32>;
@group(0) @binding(6) var b0: texture_2d<f32>;
@group(0) @binding(7) var b1: texture_2d<f32>;
@group(0) @binding(8) var b2: texture_2d<f32>;
@group(0) @binding(9) var b3: texture_2d<f32>;
@group(0) @binding(10) var b4: texture_2d<f32>;
@group(0) @binding(11) var b5: texture_2d<f32>;
@group(0) @binding(12) var<storage, read_write> flags: array<atomic<u32>>;
// x: tiles per row.
@group(0) @binding(13) var<uniform> params: vec4<u32>;

fn differs(a: texture_2d<f32>, b: texture_2d<f32>, p: vec2<u32>) -> bool {
    let q = vec2<i32>(min(p, textureDimensions(a) - vec2<u32>(1u)));
    return any(textureLoad(a, q, 0) != textureLoad(b, q, 0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    for (var j = 0u; j < 8u; j++) {
        for (var i = 0u; i < 8u; i++) {
            let p = wg.xy * 64u + li.xy + vec2<u32>(i, j) * 8u;
            if (differs(a0, b0, p) || differs(a1, b1, p) || differs(a2, b2, p) || differs(a3, b3, p) || differs(a4, b4, p) || differs(a5, b5, p)) {
                atomicOr(&flags[wg.y * params.x + wg.x], 1u);
                return;
            }
        }
    }
}
