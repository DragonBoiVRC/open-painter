# Architecture and roadmap (Rust + wgpu + egui)

Working name: **open-painter** (rename freely). License suggestion: MIT or Apache-2.0 (assets CC0).

## Workspace layout
```
crates/
  op-core/      data model: Project, TextureSet, Channel, LayerStack, Layer, Mask, Effect, BlendMode (no GPU)
  op-gpu/       wgpu context, texture pools, layer compositor, brush stamping, PBR viewport pass
  op-mesh/      load glTF/OBJ/FBX/USD (gltf, tobj, ...), tangents (MikkTSpace), UV islands, BVH
  op-bake/      CPU/GPU ray baking: normal, AO, curvature, position, thickness, ID (bvh + rayon)
  op-export/    export presets (own TOML/JSON schema, preset import from .spexp optional), PNG/EXR/TGA writers
  op-project/   .opproj container (zip or HDF5-free: zip + JSON + PNG/EXR tiles), autosave, undo journal
  op-script/    plugin API (Python via pyo3, later) + event bus
  op-app/       egui UI: viewport, layers, properties, shelf, texture-set list
assets/         CC0 only
```

## Rendering design
- Textures are tiled (e.g. 256^2 sparse tiles) so 4K-8K sets stay editable.
- Per layer: content (RGBA per channel) + optional mask; the compositor blends bottom-to-top per channel into a per-channel result texture; dirty-rect invalidation.
- Brush: stamp quads into the UV-space layer texture using 3D->UV projection (render mesh with UV as position, sample the depth/position buffer from the camera).
- Viewport: own metallic-roughness PBR (GGX, split-sum IBL, bent-normal AO), then optional path tracer later (e.g. `hiqbvh`/wgpu compute).

## Roadmap
| Milestone | Deliverable |
|---|---|
| M0 | Window, wgpu viewport, orbit camera, load glTF, lit PBR with a hard-coded material |
| M1 | Texture set per material; one paint layer; brush stamping with projection; metal/rough/base color channels; undo |
| M2 | Layer stack (layers, groups, masks, blend modes, opacity), fill layers, GPU compositor |
| M3 | Export presets (PBR MR, Unity, Unreal packed, glTF) PNG/EXR/TGA; project save/load; autosave |
| M4 | Baking: world normal, AO, curvature, thickness, position, ID (embree-like BVH on CPU, then GPU) |
| M5 | Smart-material/mask system (own graph-based procedural nodes: noise, gradients, curvature-driven edge wear) |
| M6 | Plugin API (events + scripting), shelves/resources, environment/IBL library |
| M7 | UDIM, text, stencils/alphas, particle brushes, import-only `.spp` best effort |

## Honest scope
Painter is ~15 years and hundreds of engineer-years of work. A credible solo/AI-assisted
target is M0-M3 (a usable PBR painter with layers + export). The Substance procedural
engine (`.sbsar`) and Iray have no open equivalents; M5 builds a different, original system.
