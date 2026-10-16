// The 2D UV view: a set's tile textures laid out in UV space with pan and zoom, a tile grid
// (UDIM numbering: tile 1001 at u 0..1, v 0..1) and a wireframe of the set's UVs.
//
// Drawn in four steps: a dark background, one textured quad per tile, the grid lines on top and
// the wireframe.

struct Params {
    view: vec4<f32>,    // centre u, centre v, pixels per UV unit, channel (0 base, 1 metal, 2 rough, 3 normal, 4 emissive)
    rect: vec4<f32>,    // view width px, height px, show wireframe, _
    wire: vec4<f32>,    // wireframe colour
};

@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var paint_base: texture_2d<f32>;
@group(1) @binding(1) var paint_metal: texture_2d<f32>;
@group(1) @binding(2) var paint_rough: texture_2d<f32>;
@group(1) @binding(3) var paint_samp: sampler;
@group(1) @binding(4) var paint_normal: texture_2d<f32>;
@group(1) @binding(5) var<uniform> tile: vec4<f32>;
@group(1) @binding(6) var paint_emissive: texture_2d<f32>;

fn uv_to_clip(uv: vec2<f32>) -> vec4<f32> {
    let px = (uv - p.view.xy) * p.view.z + p.rect.xy * 0.5;
    return vec4<f32>(px.x / p.rect.x * 2.0 - 1.0, 1.0 - px.y / p.rect.y * 2.0, 0.0, 1.0);
}

struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FsIn {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FsIn;
    o.clip = vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
    // Pixel position inside the view, 0,0 at the top left.
    o.local = vec2<f32>(q.x, 1.0 - q.y) * p.rect.xy;
    return o;
}

// A quad covering this draw's tile.
@vertex
fn vs_tile(@builtin(vertex_index) i: u32) -> FsIn {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[i];
    var o: FsIn;
    o.clip = uv_to_clip(tile.xy + c);
    o.local = c;
    return o;
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_bg(i: FsIn) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(0.13), 1.0);
}

@fragment
fn fs_tile(i: FsIn) -> @location(0) vec4<f32> {
    let uv = i.local;
    // A faint checker so empty areas read as empty.
    let cell = floor(i.clip.xy / 12.0);
    var color = vec3<f32>(select(0.17, 0.2, (i32(cell.x) + i32(cell.y)) % 2 == 0));
    let ch = i32(p.view.w + 0.5);
    if (ch == 0) {
        let b = textureSampleLevel(paint_base, paint_samp, uv, 0.0);
        // Premultiplied linear colour: un-premultiply, encode, and lay it over the checker.
        let straight = clamp(b.rgb / max(b.a, 1e-4), vec3<f32>(0.0), vec3<f32>(1.0));
        color = mix(color, to_srgb(straight), b.a);
    } else if (ch == 1) {
        let m = textureSampleLevel(paint_metal, paint_samp, uv, 0.0);
        color = vec3<f32>(m.r);
    } else if (ch == 2) {
        let r = textureSampleLevel(paint_rough, paint_samp, uv, 0.0);
        color = vec3<f32>(r.r + 0.5 * (1.0 - r.g));
    } else if (ch == 4) {
        let e = textureSampleLevel(paint_emissive, paint_samp, uv, 0.0);
        let straight = clamp(e.rgb / max(e.a, 1e-4), vec3<f32>(0.0), vec3<f32>(1.0));
        color = mix(color, to_srgb(straight), e.a);
    } else {
        color = textureSampleLevel(paint_normal, paint_samp, uv, 0.0).rgb;
    }
    return vec4<f32>(color, 1.0);
}

// Tile borders: a line wherever u or v crosses an integer.
@fragment
fn fs_grid(i: FsIn) -> @location(0) vec4<f32> {
    let uv = p.view.xy + (i.local - p.rect.xy * 0.5) / p.view.z;
    let to_line = abs(uv - round(uv)) * p.view.z;
    if (min(to_line.x, to_line.y) < 1.0) {
        return vec4<f32>(0.9, 0.55, 0.2, 0.85);
    }
    return vec4<f32>(0.0);
}

// ---- wireframe: the set's triangle edges, drawn with the mesh's own UVs ----

struct WireIn {
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_wire(v: WireIn) -> @builtin(position) vec4<f32> {
    return uv_to_clip(v.uv);
}

@fragment
fn fs_wire() -> @location(0) vec4<f32> {
    return p.wire;
}
