# File formats

Both formats are our own design. Nothing here is derived from Adobe's `.spp` / `.spexp`.

## Project file (`.opproj`)

A zip archive (deflate). Version 1.

| Entry | Content |
|---|---|
| `manifest.json` | `format` = `"open-painter-project"`, `version`, `texture_size`, `mesh_name`, `document` (the layer tree), `blobs` (index of `{node, kind}`) |
| `mesh.bin` | `OPMESH01` magic, vertex count, index count (u32 LE), vertices (position, normal, uv as f32), indices (u32) |
| `textures/<node>_<kind>.raw` | raw GPU texels, row-major, top row first, no header |

Texture kinds:

| kind | format | meaning |
|---|---|---|
| `base` | RGBA16F | premultiplied linear color + coverage |
| `metal`, `rough` | RG8 | (value x coverage, coverage) |
| `mask` | R8 | 1 reveals, 0 hides |

Guarantees and rules:

- **Lossless**: the bytes are the GPU texels, so save then load reproduces the composited result bit for bit (tested).
- **Atomic saves**: written to a temp file, fsynced, then renamed over the target. A failed save or crash never damages the previous file.
- **Untrusted input**: sizes are checked against the manifest *before* reading, entry names are derived (never read from the file), blobs must match a real layer of the right kind, and a newer `version` is refused with a clear message.
- **Autosave** writes `<project>.opproj.autosave` next to the project (or `untitled.opproj.autosave` in the data directory) and never touches the project file itself. A successful explicit save deletes it. "Recover autosave" loads it and marks the project modified.

## Export presets (JSON)

Built-in presets live in `op-export`. Put extra `*.json` files in `<data dir>/open-painter/presets/`
(Linux: `~/.local/share/open-painter/presets/`). Invalid files are skipped and reported in the Export window.

```json
{
  "name": "My engine",
  "maps": [
    { "file": "$project_$textureSet_BaseColor",
      "channels": ["BaseR", "BaseG", "BaseB"],
      "format": "Png", "bit_depth": "U8", "srgb": true },
    { "file": "$project_$textureSet_ORM",
      "channels": [{ "Constant": 1.0 }, "Roughness", "Metallic"],
      "format": "Png", "bit_depth": "U8", "srgb": false }
  ]
}
```

- `channels`: 1 (gray), 3 (RGB) or 4 (RGBA) entries from `BaseR`, `BaseG`, `BaseB`, `Metallic`, `Roughness`, `Smoothness` (1 - roughness), `{"Constant": v}`.
- `format` / `bit_depth`: `Png` + `U8`/`U16`, `Tga` + `U8`, `Exr` + `F32`. EXR is always linear.
- `srgb`: encode the `Base*` sources as sRGB (ignored for EXR). Data channels are never encoded.
- `file`: a plain name, no path separators or `..`. `$project` and `$textureSet` are substituted and sanitized.
- Edge padding: empty texels around UV islands are filled from covered neighbours for the requested number of texels, so filtering in other applications does not pull in background color.
- Uncovered texels use a fixed fallback material (gray 0.18, metallic 0, roughness 0.5), the same one the viewport shows.
