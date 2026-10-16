# open-painter

A cross-platform, open-source 3D texture painter for PBR materials, written in Rust (wgpu + egui).
Paint directly on a model in the 3D view or on its UV layout, with layers, masks, smart materials,
baked mesh maps, height-to-normal painting, UDIM tiles and export presets.

## Features

- Import FBX (binary, multi-material), OBJ and glTF; each material becomes a texture set.
- Layer stack with groups, blend modes, fill and paint layers, masks and generated masks.
- Brush painting in 3D (projective) and in the 2D UV view, with Base color, Metallic, Roughness,
  Height, Emissive and Opacity channels; height becomes a tangent-space normal map.
- Paint, erase, smudge, clone, projection and fill tools; image-tip brushes, stencils, pen pressure,
  stroke opacity vs flow, lazy mouse and saved brush presets.
- Per-layer colour adjustment, filters (blur, sharpen, levels, warp, ...), per-channel blend modes and
  fill layers that project textures triplanar or along an axis.
- Show, hide or isolate individual materials in the 3D view.
- Optional UV tiles (UDIM): one texture per tile, baked and exported separately.
- CPU baking (AO, curvature, thickness, normals, position, ID and more), including from a high-poly mesh.
- Export presets (PBR, Unity, Unreal, glTF, and import of Substance Painter `.spexp` presets).
- Imported textures (Base Color, Normal, Height, ...) dragged from the shelf onto a fill layer's channel slots.
- Brush shapes, rotation, jitter and mirror symmetry; HDRI or procedural environment lighting, wireframe,
  anti-aliasing and single-channel views.
- Dockable interface; the layout is remembered.

Coming from Substance Painter? See [docs/07-coming-from-substance-painter.md](docs/07-coming-from-substance-painter.md).

## Run it

```sh
cargo run --release -p op-app                     # start empty
cargo run --release -p op-app -- model.fbx        # open a model or a .opproj project
```

You need a recent stable Rust toolchain and a GPU with Vulkan, Metal or DirectX 12 support.

## Install a launcher (Linux)

```sh
packaging/install.sh            # builds, then adds open-painter to your application menu
packaging/install.sh --remove   # removes it again
```

This installs for the current user only (`~/.local`), registers `.opproj` files, and turns on
automatic updates: each time you start it, the launcher checks this repository for new commits
and, if there are any (and your checkout has no local changes), pulls and rebuilds in the
background. The new version is used the next time you start the program. To turn that off, put
`off` in `~/.config/open-painter/auto-update`, or set `OPEN_PAINTER_NO_UPDATE=1`. The update log is
`~/.local/state/open-painter/update.log`.

## Layout

| Crate | What it does |
|---|---|
| `op-core` | document model, layer stack, blending reference, normals from height |
| `op-mesh` | mesh formats (FBX, OBJ, glTF), texture sets, UDIM tiles |
| `op-bake` | CPU ray baker |
| `op-proc` | procedural mask graphs |
| `op-export` | export presets and texture writers |
| `op-project` | `.opproj` project files |
| `op-app` | the application |

See `docs/` for the design notes and file-format write-ups.

## Licensing

Dual licensed under MIT or Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
Bundled third-party code and fonts are listed in `THIRD_PARTY.md`.
open-painter contains no code, shaders or assets from any commercial painting application.
