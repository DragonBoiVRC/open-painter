# Procedural masks and smart materials

A layer's mask can be **generated** from the baked mesh maps instead of painted: edge wear from
curvature, dirt from occlusion, dust from the world normal, and so on. Generated masks recompute
automatically when their settings, the baked maps or the resolution change.

## Generators

| Generator | Reads | What it does |
|---|---|---|
| Edge wear | curvature | convex edges, softened, broken up with noise |
| Crevice dirt | ambient occlusion, curvature | occluded and concave areas, broken up with noise |
| Top dust | world-space normal | up-facing surfaces, broken up with noise |
| Height gradient | position | a ramp along the object's up axis |
| Noise | position | fractal Perlin noise with adjustable contrast |
| Custom | whatever its graph reads | any graph loaded from a file |

Every generator has **Invert** and the parameters listed in `op-proc/src/templates.rs`
(amount, softness, breakup, scale, seed, ...), all range-checked. A generator that needs a map you
have not baked still runs: it uses the map's neutral value and the layer panel says what is missing.

Noise is evaluated in **object space** from the baked position map, so patterns do not stretch
or break at UV seams; without a position bake it falls back to UV space and says so.

A generated mask cannot be painted. **Freeze** removes the generator and keeps the generated pixels,
so the mask becomes an ordinary paintable one. Replacing an already painted mask with a generated one
is undoable (Ctrl+Z restores the painted pixels).

## The graph engine (`op-core::graph`, `op-proc`)

A graph is a validated DAG of scalar-field nodes producing one grayscale image:

- sources: `Const`, `Baked { map, comp }`, `Noise { kind: Value|Perlin|Cellular, scale, octaves, persistence, seed, uv_space }`, `Uv { axis }`
- one input: `Invert`, `Levels`, `Contrast`, `Threshold`, `Blur` (radius is a fraction of the texture size)
- two inputs: `Blend { op: Add|Multiply|Subtract|Min|Max|Screen|Overlay|Difference, opacity }`
- three inputs: `Lerp` (the third input is the per-texel blend factor)

Results of each node are clamped to 0..1. Evaluation order follows dependencies, intermediate
images are freed as soon as nothing needs them, and baked maps at another resolution are resampled
bilinearly. Graphs are checked before use: arity, missing nodes, cycles (iterative, no recursion),
parameter ranges, at most 256 nodes. Evaluation is on the CPU with rayon and is deterministic: the same
inputs give the same bits whatever the thread count (and across sessions).

### Custom generator files

Put `*.json` files in `<data dir>/open-painter/generators/` (Linux: `~/.local/share/open-painter/generators/`):

```json
{ "name": "left to right",
  "graph": {
    "nodes": [ { "id": 0, "op": { "Uv": { "axis": 0 } }, "inputs": [] } ],
    "output": 0 } }
```

They appear in the generator menus. Invalid files are skipped and listed under Smart materials.
There is no visual node editor yet: graphs are written as JSON.

## Smart materials

A smart material is a saved layer subtree (fills, groups, masks with generators). Three ship with the
application ("Worn painted metal", "Rusty iron", "Dusty stone"); all are original. **Add to layer stack**
inserts a copy above the selected layer with fresh ids. **Save selection** stores the selected layer or group
as `<data dir>/open-painter/smart-materials/<name>.json`; paint layers are saved empty (their pixels are not part of
the file). Files are validated on load (names, value ranges, nesting depth, node count, generator graphs).

## Limitations

- Mask generation uses the CPU. Measured on a 16-thread machine (`cargo run --release -p op-proc --example bench`):
  each built-in generator takes about 60-80 ms at 2048x2048 and 230-350 ms at 4096x4096. Expect proportionally
  more on machines with fewer cores, and in debug builds. Generation runs on a worker thread and the layer panel
  shows "Generating..." while it works.
- Blur works in UV space, so it does not cross UV seams; the baker pads islands (16 texels by default) to keep
  dead space out of the result.
- Curvature-driven generators look as good as the mesh's tessellation allows: curvature is measured from
  the geometry, so a coarse mesh gives coarse edge detail.
- Adobe `.sbsar` materials cannot be loaded (that needs Adobe's closed engine); these generators are a
  different, original system.
- No visual node editor, and a generated mask cannot be combined with painted strokes without freezing it first.
