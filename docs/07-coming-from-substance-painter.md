# Coming from Substance Painter

open-painter keeps the same working model: texture sets, a layer stack of paint and fill layers with
masks, a shelf of materials and brushes, and PBR channels painted straight onto the model. This page
maps what you know onto what is here, and says plainly what is not here yet.

## Where things are

| Substance Painter | open-painter |
|---|---|
| Texture Set List | **Texture Sets** panel (Ctrl+PageUp / PageDown to switch, Ctrl+click on the model to pick, H to hide) |
| Layers | **Layers** panel: paint, fill and folder layers, drag to reorder, right-click for the menu |
| Properties - Paint | **Properties - Paint**: brush, symmetry, stamp shape, paint channels |
| Properties - Fill | **Properties - Layer**: channel slots accept textures dragged from the shelf; tiling, offset, rotation |
| Shelf | **Shelf**: smart materials, smart masks, brushes, imported textures |
| Texture Set Settings / Bake Mesh Maps | **Texture Set Settings** and **Bake Mesh Maps** tabs |
| Display Settings | **Display Settings**: environment, rotation, exposure, key light, shading views |
| Export Textures | File > Export textures... with presets (`.spexp` presets can be imported) |
| 2D / 3D view | **Viewport** and **UV View**, side by side by default |

## Keys and mouse

| Action | Keys |
|---|---|
| Orbit / pan / zoom | Alt + left drag / middle drag / Alt + right drag or scroll |
| Brush size and hardness | Ctrl + right drag, or `[` and `]` for size |
| Turn the environment | Shift + right drag |
| Paint / erase | `P` / `E` or `1` / `2` |
| Straight line | Shift + click after a stroke |
| Symmetry | `L` toggles it; pick the axis in Properties - Paint |
| Pick colour from the model | `I` |
| Cycle base colour / roughness / metallic views | `C` |
| Hide all panels / show them again | Tab |
| 3D + 2D / 3D only / show 2D | F1 / F3 / F4 |
| Frame the model | `F` |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y |
| Duplicate / group layer | Ctrl+D / Ctrl+G |
| Rename / delete layer | F2 / Delete |
| Save / save as / open / export | Ctrl+S / Ctrl+Shift+S / Ctrl+O / Ctrl+E |

## Differences to know about

- Texture sets are the model's materials. Hide or isolate a set with the eye icon in the Texture Sets panel.
- Lighting uses built-in procedural environments rather than HDRI files.
- Mask levels (black, white, gamma, invert) live on the mask in Properties - Layer.
- Not here yet: the clone, smudge and projection tools, layer filters and effects, anchor points,
  pen pressure, and path tracing.
