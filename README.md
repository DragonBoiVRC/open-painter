# open-painter

A cross-platform, open-source 3D texture painter for PBR materials, written in Rust (wgpu + egui).
Paint directly on a model in the 3D view or on its UV layout, with layers, masks, smart materials,
baked mesh maps, height-to-normal painting, UDIM tiles and export presets.

## Features

- Import FBX (binary, multi-material), OBJ, glTF, PLY and STL; each material becomes a texture set (meshes without UVs get a rough layout).
- Layer stack with groups, blend modes, fill and paint layers, masks and generated masks.
- Brush painting in 3D (projective) and in the 2D UV view, with Base color, Metallic, Roughness,
  Height, Emissive and Opacity channels; height becomes a tangent-space normal map.
- Paint, erase, smudge, clone, projection, fill, lasso, path, spray and decal tools, quick mask, geometry masks and anchor-driven masks; image-tip brushes, stencils, pen pressure,
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
- Agent control: a JSON channel (`op-app --control`) and an MCP server (`op-cli mcp`); see [docs/08-agent-control.md](docs/08-agent-control.md).

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

## Windows

Each push builds a Windows x64 zip in GitHub Actions (`.github/workflows/windows.yml`); download it
from the workflow run's artifacts, or from the release page for tagged versions. Unzip and run
`open-painter.exe`. To build it yourself, `cargo build --release -p op-app` works on Windows too.

## Memory

Projects with many texture sets or large textures are kept inside your video memory (VRAM) and RAM
without you managing it:

- **Only the set you edit is full size on the GPU.** The other sets keep a 512 px picture of their
  finished look, made in the background after a project opens, so the whole model stays painted in
  the viewport. Picking a set loads its layers at full size; sets you leave unchanged go back to a
  picture.
- **Source textures upload when needed.** A set's image textures go to the GPU only while that set is
  open or being exported, and are freed afterwards.
- **A GPU budget** (half your video memory) moves the least recently used sets out when it is
  exceeded. Their paint stays in RAM.
- **A RAM budget** (half the free memory) moves the oldest of that paint to files in
  `<data folder>/spill` (`~/.local/share/open-painter/spill` on Linux). Saving, exporting and
  reopening a set read it back; the files are deleted when it is loaded or the program closes.
- **Exceptions:** the set being edited and sets with undo history are never moved out. If the active
  set alone is larger than the budget, the GPU can still run out of memory; lower its resolution in
  Texture Set Settings.

The status bar and `info` show the GPU memory the texture sets use. Set `OPEN_PAINTER_GPU_BUDGET_MB`
or `OPEN_PAINTER_RAM_BUDGET_MB` to change either budget. Video memory is detected on Linux (NVIDIA
via `nvidia-smi`, AMD via the kernel); elsewhere the GPU budget defaults to 2 GB unless you set it.

## Layout

| Crate | What it does |
|---|---|
| `op-core` | document model, layer stack, blending reference, normals from height |
| `op-mesh` | mesh formats (FBX, OBJ, glTF), texture sets, UDIM tiles |
| `op-bake` | CPU ray baker and path-traced render preview |
| `op-proc` | procedural mask graphs |
| `op-export` | export presets and texture writers |
| `op-project` | `.opproj` project files |
| `op-app` | the application |
| `op-cli` | agent control: `call` and an MCP server for a running `op-app --control` |

See `docs/` for the design notes and file-format write-ups.

## Licensing

Dual licensed under MIT or Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
Bundled third-party code and fonts are listed in `THIRD_PARTY.md`.
open-painter contains no code, shaders or assets from any commercial painting application.
