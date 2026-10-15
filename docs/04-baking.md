# Baking

`op-bake` renders mesh maps into the UV layout of the low-poly mesh. Every texel covered by a
triangle's UV footprint is shaded by casting rays against the *target* geometry: the optional
high-poly mesh, or the low-poly mesh itself.

| Map | How it is computed |
|---|---|
| Position | world position, normalised to the low-poly bounding box per axis |
| World space normal | interpolated vertex normal, `n * 0.5 + 0.5` |
| Ambient occlusion | cosine-weighted hemisphere rays (default 32) up to *AO distance*; fraction not blocked |
| Bent normals | average unblocked direction, in tangent space, encoded like a normal map |
| Thickness | cosine-weighted rays into the surface; mean hit distance / *thickness distance* (misses count as the maximum) |
| Curvature | signed angle between adjacent faces divided by the distance between their centres (about `1/radius`); 0.5 is flat, brighter is convex, darker is concave. Hard edges register. |
| Normal (tangent space) | normal of the high-poly surface hit by a ray along `-N` from `front` above the low-poly surface, in the low-poly tangent frame. Flat `(0.5, 0.5, 1)` without a high-poly. |
| Height | signed offset of that hit along the normal, mapped around 0.5 over `max(front, rear)` |
| Opacity | 1 where the high-poly was found (always 1 without a high-poly) |
| ID | one hue per connected piece of the mesh |

## Conventions

- Texel `(x, y)` samples `uv = ((x + 0.5)/size, (y + 0.5)/size)`; `v` grows downward, like image rows.
- Tangent space: `T` = direction of increasing `u`, `B` = toward the top of the image (decreasing `v`),
  `N` = surface normal; the encoded map is `(n.T, n.B, n.N) * 0.5 + 0.5` (OpenGL convention, +Y up).
  DirectX output flips green at export time.
- All maps are stored as 16-bit values.
- Sampling patterns are seeded per texel: a bake is bit-for-bit reproducible whatever the thread count.
- Distances are fractions of the low-poly bounding-box diagonal.

## Known limitations

- Tangents come from UV derivatives, not MikkTSpace. Normal maps are self-consistent with this
  application's own viewport and exports, but a game engine that recomputes MikkTSpace tangents may show small
  shading differences on meshes with strongly varying UV density.
- Where UV footprints overlap (mirrored UVs) the last triangle wins; there is no per-island dilation inside the bake
  (padding is applied at export time).
- On a faceted convex surface a few grazing AO rays can hit a neighbouring facet, so AO reads slightly below 1.
- The high-poly mesh is not saved in the project file (only the baked results are).
- Baked maps are tied to the texture resolution they were baked at; changing the resolution discards them.
