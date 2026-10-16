# File formats

The project file and the preset JSON are our own design. Reading Adobe `.spexp` presets (import only) is based on the
observed structure of sample files; no Adobe data is copied into this repository or redistributed.

## Project file (`.opproj`)

A zip archive (deflate). Version 2 (version 1 files, which have no baked maps, still load).

| Entry | Content |
|---|---|
| `manifest.json` | `format` = `"open-painter-project"`, `version`, `texture_size`, `mesh_name`, `document` (the layer tree), `blobs` (index of `{node, kind}`) |
| `mesh.bin` | `OPMESH01` magic, vertex count, index count (u32 LE), vertices (position, normal, uv as f32), indices (u32) |
| `textures/<node>_<kind>.raw` | raw GPU texels, row-major, top row first, no header |
| `bakes/<Map>.raw` | one baked mesh map: 16-bit little-endian values, row-major, 1 channel (scalar maps) or 3 (vector maps). The manifest's `bakes` list gives `map`, `size`, `channels`. |

Texture kinds:

| kind | format | meaning |
|---|---|---|
| `base`, `emissive` | RGBA16F | premultiplied linear color + coverage (`emissive` is the light the surface gives off) |
| `metal`, `rough`, `height`, `opacity` | RG8 | (value x coverage, coverage); height 0.5 is flat; opacity is 1 where nothing is painted |
| `mask` | R8 | 1 reveals, 0 hides |

Guarantees and rules:

- **Lossless**: the bytes are the GPU texels, so save then load reproduces the composited result bit for bit (tested).
- **Atomic saves**: written to a temp file, fsynced, then renamed over the target. A failed save or crash never damages the previous file.
- **Untrusted input**: sizes are checked against the manifest *before* reading, entry names are derived (never read from the file), blobs must match a real layer of the right kind, and a newer `version` is refused with a clear message.
- **Autosave** writes `<project>.opproj.autosave` next to the project (or `untitled.opproj.autosave` in the data directory) and never touches the project file itself. A successful explicit save deletes it. "Recover autosave" loads it and marks the project modified.

## Export presets (JSON)

Built-in presets live in `op-export`. Put extra `*.json` files in `<data dir>/open-painter/presets/`
(Linux: `~/.local/share/open-painter/presets/`). Invalid files are skipped and reported in the Export window.
Presets imported from Substance Painter (see below) are saved there automatically.

```json
{
  "name": "My engine",
  "maps": [
    { "file": "$mesh_$textureSet_BaseColor(_$colorSpace)(.$udim)",
      "channels": [ { "Color": { "map": "Base", "comp": "R" } },
                    { "Color": { "map": "Base", "comp": "G" } },
                    { "Color": { "map": "Base", "comp": "B" } } ],
      "format": "Png", "bit_depth": "U16", "srgb": true },
    { "file": "$project_$textureSet_ORM",
      "channels": [ "AoMixed", "Roughness", "Metallic" ],
      "format": "Png", "bit_depth": "U8", "srgb": false }
  ]
}
```

- `channels`: 1 (gray), 3 (RGB) or 4 (RGBA) sources. A source is one of:
  `Metallic`, `Roughness`, `Smoothness` (1 - roughness), `Height`, `Opacity` (1 where nothing is painted),
  `AoMixed` (baked AO), `{"Constant": v}`,
  `{"Color": {"map": "Base"|"Diffuse"|"Specular"|"Emissive", "comp": "R"|"G"|"B"|"L"}}`,
  `{"Normal": {"directx": bool, "comp": ...}}` (baked tangent-space normal; `directx` flips green),
  `{"Mesh": {"map": <MeshMap>, "comp": ...}}` (any baked map). `L` is Rec. 709 luminance.
  Diffuse = `base * (1 - metallic)`, Specular = `lerp(0.04, base, metallic)`.
- `format` / `bit_depth`: `Png` + `U8`/`U16`, `Tga` + `U8`, `Exr` + `F32`. EXR is always linear.
- `srgb`: encode color-like sources as sRGB (ignored for EXR). Data channels are never encoded.
- `file`: a plain name, no path separators or `..`. Variables `$project`, `$textureSet`, `$mesh`
  (`$colorSpace` and `$udim` expand to nothing). A parenthesised group such as `(_$colorSpace)` is dropped
  when any variable inside it is empty. Values are sanitised; two maps that would write the same file are refused.
- Edge padding fills empty texels around UV islands (baked maps included) from covered neighbours.
- Uncovered texels use a fixed fallback material (gray 0.18, metallic 0, roughness 0.5), the same one the viewport shows.
- A source that needs a baked map which has not been baked falls back to that map's neutral value
  (AO 1, flat normal, ...) and the export reports it.

## Importing Substance Painter presets (`.spexp`)

`op_export::import_spexp` reads Painter export presets and converts them to the schema above. The
file format was worked out from sample files; the reader is strict and fails on anything it does not
understand. Both container versions seen in practice are supported: version 0 (self-describing tagged
values) and version 1 (class table plus schema). All 34 presets that ship with the Painter
installation this was developed against, and a third-party preset (Unity Poiyomi v10.0), decode completely.

Developer tool: `cargo run -p op-export --example spexp -- preset.spexp [--json]`.

How a preset channel `(mapIdType, mapId, channelSrc)` is translated:

| mapIdType | meaning | handled |
|---|---|---|
| 1 | document channel (version 1 stores the bit index, version 0 the bit flag `1 << index`) | BaseColor (0), Height (1), Opacity (3), Emissive (4), Glossiness (6), Roughness (7), Metallic (13), Diffuse (24) and Specular (2) derived from base/metallic |
| 2 | named virtual map | `Normal_OpenGL`, `Normal_DirectX`, `AO_Mixed`, `Glossiness`, `Diffuse`, `Specular` |
| 3 | baked mesh map: 0 ID, 1 AO, 2 curvature, 3 world normal, 4 position, 5 thickness, 6 normal, 7 height, 8 bent normals, 9 opacity | all of them |
| 0 | constant (0 black, 1 white) | yes |

`channelSrc`: 0 R, 1 G, 2 B, 3 A (opaque), 4 L. Bit depth setting 0 = 8-bit, 1 = 16-bit.

Everything that is not an exact translation is **reported as a warning and written as a constant**, never
guessed: document channels this project does not have (Anisotropy, Transmissive, Scattering, user
channels), channel ids that no sample preset identifies (for example 14, 15, 17 and 23 in the Poiyomi
preset), unsupported file formats, folders inside file names, unknown `$variables`, dithering.
The id table comes from the sample presets (ids appear next to the file names Painter gives them);
ids with no evidence stay unidentified on purpose.

Maps made only of emissive light or opacity are left out of an export when nothing glows or nothing is
see-through, so the built-in presets can list them without littering every export with black or white files.
