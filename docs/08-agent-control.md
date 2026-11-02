# Agent control

`op-app --control 7878` listens on 127.0.0.1:7878 (this machine only, no password: leave it off
unless you want it). One JSON object per line in, one per line out:

    {"id": 1, "cmd": "paint_uv", "params": {"points": [[0.2,0.5],[0.8,0.5]], "radius": 0.04}}
    {"id": 1, "ok": true, "result": true}

`op-cli` wraps it:

    op-cli call info                     # layers, texture sets, export presets
    op-cli call add_layer '{"kind":"paint","name":"Dirt"}'
    op-cli call bake '{"maps":["ao","curvature"]}'    # then poll: op-cli call status
    op-cli mcp                           # MCP server on stdio

MCP config (Claude Code: `claude mcp add open-painter -- /path/to/op-cli mcp`). `op-cli` starts
`op-app --control` itself if nothing is listening; it looks for `op-app` beside itself, then on PATH.

`op-cli call commands` lists every command with its parameters. Paint, bake and export go through
the same code as the UI, so undo, thumbnails and the modified flag behave the same. Strokes are
painted in UV space (`paint_uv`); there is no 3D-view painting over the channel yet. Layers are
edited as JSON: `get_layer` shows a layer, `set_layer` merges a partial one into it.

Commands include layers (`add_layer`, `set_layer`, `get_layer`), painting (`paint_uv`, `fill_uv`,
`lasso_uv`, `set_brush`, `quick_mask`, `geometry_mask`), images (`import_image`, `text_image`,
`set_texture`, `add_decal`), bake and export, `reimport_mesh`, the look (`camera`, `set_display`,
`appearance`) and pictures (`screenshot` of the 3D view, `render` a path-traced preview).
`info` reports an estimate of GPU memory in use as `gpu_bytes`. Commands run a few frames apart, so
an `export` or `screenshot` always sees the edits before it.
