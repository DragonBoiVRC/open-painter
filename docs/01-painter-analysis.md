# Substance 3D Painter: analysis for a clean-room open-source alternative

Method: static inspection of the shipped install folder (public SDK docs, plugin
sources, file containers, asset inventory). The 171 MB main executable has **not**
been decompiled. REA's PE tool caps at 64 MB, and nothing needed so far required it.
Everything below is *observed* in shipped files unless marked *(inferred)*.

## Legal / clean-room rules for this project
- Bundled GLSL shaders, `.sbsar` materials and the `starter_assets` are marked
  "ADOBE CONFIDENTIAL ... reproduction strictly forbidden". **Do not copy them.**
  Write our own PBR shaders and ship CC0 assets (e.g. Poly Haven, ambientCG).
- Use the documented public API (Python/JS/QML docs) as a *feature checklist*,
  not as code to copy. Do not decompile the binary to copy implementation.
- Interop with Adobe formats (`.spp`, `.spexp`, `.sbsar`) should be import-only,
  built from observed structure, with no redistribution of Adobe data.

## Architecture (observed)
| Layer | Evidence |
|---|---|
| C++ core + Qt 6 (Widgets + QML/Quick) UI | Qt6*.dll, `qml/`, `qml-doc/` (AlgWidgets QML component set) |
| Embedded Python 3.13 + PySide6 plugin API | `python313.dll`, `pythonsdk/`, `python-doc/substance_painter/*` |
| Legacy JavaScript plugin API (`alg.*`) | `javascript-doc/` (baking, display, mapexport, project, resources, settings, shaders, texturesets, ui) |
| Substance engine (procedural textures) | `substance_linker.dll`, `substance_sse2_blend.dll`, `substance_vk_sal.dll`, `.sbsar` assets |
| Viewport: OpenGL/Vulkan PBR + Iray path tracer | `PlugIns/iray`, `ngl/`, `shader-doc/`, `mdl/` (NVIDIA MDL) |
| Mesh IO | `assimp.dll`, `Alembic.dll`, `usd_ms.dll` (USD), glTF settings in `project.GltfSettings` |
| Ray tracing / baking | `embree3.dll`, Intel MKL, TBB |
| Color management | `OpenColorIO_2_5.dll`, `ocio/` (ACES 1.0.3 / 1.2 / 2.0) |
| UV tooling | `anorigami_sdk.dll`, `AutoUnwrapSettings` |
| Plugins = QML + plugin.json | `application_plugins/{autosave,photoshop-export,resources-updater}` (MIT, readable) |

## Project file format (observed)
`.spp` and `.spt` (templates) are **HDF5** containers:
- `/projectsettings.ini`, `/editor/*.ini|json` (camera, color profile, exporter, Iray, post FX, viewer)
- `/baking/baking.ini`
- `/paint/document.bin` (layer stacks), `/paint/scene.assbin` (mesh, assimp-style binary), `/paint/default_material.bin`
- `/resources/<sha1>.{image,sbsar,shader}` + `/resources/.alg_meta/*` (content-addressed embedded assets)
- `/plugins/plugins.toc`
Binary blobs (`*.bin`, `*.assbin`) are proprietary and **unparsed**: unknown.
`.spexp` (export presets) is a binary serialized `DataExportPreset` -> `DataExportMap` -> `DataExportChannel`
(fields: bitdepth, channels, dithering, fileformat, path template, channelSrc, mapId).
Path templates use `$mesh_$textureSet_BaseColor(_$colorSpace)(.$udim)`.

## Feature inventory (from public Python API)
- **Texture sets & channels**: per-set stacks, channel add/remove (BaseColor, Roughness, Metallic, Normal, Height, Emissive...), per-channel format and bit depth, per-set/UDIM-tile resolution, mesh-map resources.
- **Layer stack**: layers, groups, folders, instances, masks (+background), geometry masks (by mesh / UV tile), blend modes, opacity, and content/mask *effects*: fill, paint, filter, generator, levels, color selection, compare mask, smart mask, smart material, anchor point.
- **Painting**: brushes (`.sppr`, alphas), projection/stencil/tiling, particle emitters (`.pkfx`), fonts/text.
- **Baking**: per-texture-set, UV-tile aware, linked parameter groups, curvature method; maps incl. AO, curvature, position, world normal, thickness, ID.
- **Export**: preset-driven channel -> file mapping (72 presets: Unity, Unreal, Blender, Arnold, Roblox...), bit depth, dither, file format.
- **Display**: env maps (EXR), color LUTs, tone mapping, camera (persp/ortho, DOF), post FX.
- **Resources/shelves**: searchable, tagged, application + user shelves; project-embedded copies.
- **Events/automation**: project, bake, export, layer-stack events; plugin UI (docks, menus, toolbars).

## Gaps / unknowns
- Brush engine internals, layer compositing order/GPU pipeline: not documented, **design our own**.
- `.sbsar` evaluation needs Adobe's closed Substance engine: **no open equivalent** (see roadmap: node-based procedural system of our own, later).
- Exact `.spp` blob encodings: unknown; best-effort import is a stretch goal.
