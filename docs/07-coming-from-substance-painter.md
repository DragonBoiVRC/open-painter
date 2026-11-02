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
| Smudge / clone / projection / fill | `S` / `K` / `J` / `G` or `3` / `4` / `5` / `6` (Ctrl+click sets the clone source) |
| Straight line | Shift + click after a stroke |
| Symmetry | `L` toggles it; pick the axis in Properties - Paint |
| Pick colour from the model / every channel | `I` / `Shift+I` |
| Cycle base colour / roughness / metallic / emissive / opacity views | `C` |
| Show the mesh edges | `W` |
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
- Fill (G) fills one flat face, a UV island, a piece of the mesh or a whole texture set with a click, using the paint colour and channels (or white on a mask; Shift+click for black).
- Camera: view presets (front, back, left, right, top, bottom), a field-of-view slider and a perspective/orthographic switch are in the viewport bar and Display Settings.
- Per-channel blending: under Properties - Layer, Channels, each of base colour, metallic, roughness, height, emissive, opacity and (for fill layers) normal can be switched off, blended differently from the layer, or made more transparent.
- Material picker: I picks the colour under the cursor, Shift+I picks colour, metallic, roughness, height, emissive colour and opacity.
- Emissive is a real channel: paint it with the brush (Properties - Brush, Paint channels), give fill layers an emissive colour or texture, view it with C, and it glows in the viewport. It is exported as its own map (Emissive/Emission) in the built-in presets when something glows, and Substance export presets that ask for Emissive now read it.
- Fill layers can project their textures: triplanar or planar X/Y/Z (Properties - Layer, Texture placement). The projection reads the baked Position and World space normal maps, so bake those first; without them the UV layout is used. Tiling 1 spans the model's longest side, and normal maps are not projected.
- Wireframe: W (or the polygon button in the viewport bar) draws the mesh edges over the model.
- The loaded HDRI can be shown behind the model (Display Settings - Environment), with a blur slider, and the tone mapping can be filmic, Reinhard or off.
- Show back faces (Display Settings, on by default): both sides of every surface are drawn and the side facing you is lit, so parts whose normals or winding are flipped (mirrored copies, inside-out shells) look right, and you can paint on them. Switch it off to see them as a one-sided material would.
- Opacity is a channel too: paint it, give fill layers an opacity value or texture, and it is exported as the alpha of the Unity preset (and as optional Opacity maps when something is see-through). The viewport shows see-through paint as dithered holes (Display Settings can switch that off).
- Edges in the 3D view are smoothed with FXAA (Display Settings can switch it off).
- Brush presets: "Save the current brush" in the Shelf (Brushes) keeps the whole brush, including flow, stroke opacity, lazy mouse and pressure settings, in the data folder; right-click a saved brush to delete it.
- Reach: "Paint through the model" and "Paint back faces" (Properties - Brush) let strokes land on hidden or turned-away surfaces.
- Colour-selection masks: a generated mask called "Color selection" picks one piece of the model by its baked ID colour; "Pick the piece from the model" lets you click it.
- Section plane (Display Settings): cuts the model open along X, Y or Z so you can see and paint the inside.
- A strip of recently used colours sits under the colour swatch in Properties - Brush.
- File > Save viewport image (F12) writes the 3D view as it is to a PNG next to the project.
- Changing the resolution in Texture Set Settings resamples the paint and masks instead of clearing them (undo history and baked maps are cleared).
- Hide parts (7): click a face, UV island, mesh part or texture set to hide it so you can paint what is behind (Shift+click shows it, "Show all parts" resets). Hidden parts are a view setting and are not saved with the project. They are kept in UV space, so parts that share UV space (mirrored UVs) hide together.
- Mask view: the backslash key shows the selected layer's mask on the model as grey (white reveals, black hides); press it again to go back.
- Your lighting and display choices, including the HDRI in use, are remembered between sessions.
- Ambient occlusion (screen space) in Display Settings darkens creases and where parts meet; it is a viewport look only.
- Lazy mouse (Properties - Brush, "Lazy mouse") makes the brush trail the cursor on a string for smooth strokes.
- Pen pressure drives brush size and/or opacity (Brush panel) when the system reports it as touch force.
- Smudge (S) and clone (K, Ctrl+click sets the source) work in the 3D view on layer content; they copy from a screen offset and ignore symmetry.
- Brush alpha and stencils: pick an image from the texture library in Properties - Brush ("Image and stencil"). The Image stamp shapes each dab; a stencil is laid over the screen and paint only lands where it is bright. Tick "Move it in the viewport" to drag and scroll it into place.
- Projection (J) paints the stencil picture onto the model through the brush, using the stencil placement above.
- Filters (blur, sharpen, invert, levels, directional blur, warp, threshold, pixelate, hue/saturation) are listed under Properties - Layer for paint layers and groups; they reshape what the layer puts down without touching the paint. Fill layers have none, as they are constants.
- Quick mask (Y): protect areas so paint only lands where the quick mask is free; U edits it with the brush (paint protects, the eraser frees). It covers strokes, erase, smudge and fills on layer content and masks. It is a working aid: it is not saved with the project.
- Geometry mask (8): the Fill tool on the selected layer's mask over mesh parts, so the layer shows only on the parts you click (a layer without a mask gets a black one).
- Lasso, Path and Spray: Lasso (drag an outline) and Path (click corners, Enter or double-click closes) fill everything inside; in the UV view the outline is filled exactly, in the 3D view every region it covers is filled (coarse on dense meshes). Spray scatters small dots around the brush.
- Decals: a fill layer with the Decal projection puts its picture through a frame on the surface (Width, Height, Turn, Depth; "Place on model" then click). It needs the Position and World space normal maps baked. The Text section of Properties - Brush renders text (built-in or installed fonts, plain left-to-right layout) as a brush stamp, a stencil or a decal.
- Anchors: give a layer an Anchor name and a mask higher in the stack can follow one channel of the stack as it was there (colour brightness, metallic, roughness, height, coverage): Mask > Follows anchor.
- More filters: Posterize, Brightness / contrast, Grayscale and Edge detect.
- More content: 46 base materials, 30 smart materials (with the fabrics), 22 brush presets and 12 brush shapes (triangle, star, cross, ring, hexagon and blot are new).
- Models: PLY and STL import; a model without UVs gets a rough box layout; File > Reimport model reads the file again keeping all paint (the texture sets must match).
- View: bloom, sharpen, vignette and grain (Display Settings, Look), saved camera positions per model, and a path-traced render preview (View > Render preview, F10; CPU, refines in the background, can be saved as PNG).
- Interface: four brightness levels and seven accent colours (Window > Appearance), a start window, and an estimate of GPU memory in the status bar. Empty paint layers take no GPU memory until you paint on them.
- Agents: `op-app --control <port>` and `op-cli mcp` let scripts and AI agents drive the program; see docs/08-agent-control.md.
- Not here yet: more PBR channels (clear coat, sheen, anisotropy, subsurface, transmission, ambient occlusion and specular level as paintable channels), depth of field and displacement in the viewport, a shortcut editor, translations, COLLADA, USD and Alembic import, slope blur and gradient map filters, and a scripting API beyond the control channel.
