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
- Lighting has four built-in procedural environments, and you can load your own HDRI (Radiance .hdr or OpenEXR, lat-long) in Display Settings - Environment. Loaded files are kept in the app's data folder (`environments/`) and listed in the HDRI menu. Reflections use a mip chain and diffuse light uses spherical harmonics, a fast approximation of Painter's prefiltered maps.
- Mask levels (black, white, gamma, invert) live on the mask in Properties - Layer.
- Brush Opacity is now called Flow (the opacity of each dab, so overlaps build up) and there is a separate Stroke opacity that caps a whole stroke, like Painter's. Stroke opacity applies to painting, projection and masks.
- Fill (G) fills a UV island, a piece of the mesh or a whole texture set with a click, using the paint colour and channels (or white on a mask; Shift+click for black).
- Camera: view presets (front, back, left, right, top, bottom) and a field-of-view slider are in the viewport bar and Display Settings.
- Per-channel blending: under Properties - Layer, Channels, each of base colour, metallic, roughness, height and (for fill layers) normal can be switched off, blended differently from the layer, or made more transparent.
- Material picker: I picks the colour under the cursor, Shift+I picks colour, metallic, roughness and height.
- Emissive is a real channel: paint it with the brush (Properties - Brush, Paint channels), give fill layers an emissive colour or texture, view it with C, and it glows in the viewport. It is exported as its own map (Emissive/Emission) in the built-in presets when something glows, and Substance export presets that ask for Emissive now read it.
- Fill layers can project their textures: triplanar or planar X/Y/Z (Properties - Layer, Texture placement). The projection reads the baked Position and World space normal maps, so bake those first; without them the UV layout is used. Tiling 1 spans the model's longest side, and normal maps are not projected.
- Wireframe: W (or the polygon button in the viewport bar) draws the mesh edges over the model.
- The loaded HDRI can be shown behind the model (Display Settings - Environment), with a blur slider, and the tone mapping can be filmic, Reinhard or off.
- Lazy mouse (Properties - Brush, "Lazy mouse") makes the brush trail the cursor on a string for smooth strokes.
- Pen pressure drives brush size and/or opacity (Brush panel) when the system reports it as touch force.
- Smudge (S) and clone (K, Ctrl+click sets the source) work in the 3D view on layer content; they copy from a screen offset and ignore symmetry.
- Brush alpha and stencils: pick an image from the texture library in Properties - Brush ("Image and stencil"). The Image stamp shapes each dab; a stencil is laid over the screen and paint only lands where it is bright. Tick "Move it in the viewport" to drag and scroll it into place.
- Projection (J) paints the stencil picture onto the model through the brush, using the stencil placement above.
- Filters (blur, sharpen, invert, levels) are listed under Properties - Layer for paint layers and groups; they reshape what the layer puts down without touching the paint. Fill layers have none, as they are constants.
- Not here yet: further filters and effects (warp, directional blur, ...), anchor points,
  and path tracing.
