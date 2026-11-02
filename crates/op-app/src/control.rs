//! A JSON control channel: `op-app --control <port>` listens on 127.0.0.1 and runs one command per
//! line, so scripts and agents (see the `op-cli` crate's MCP server) can drive the running app.
//!
//! Request: `{"id": 1, "cmd": "paint_uv", "params": {...}}`. Reply: `{"id": 1, "ok": true, "result": ...}`
//! or `{"id": 1, "ok": false, "error": "..."}`. Anything on this machine can connect, so it is off
//! unless asked for.

use crate::paint::{PaintOp, Target, MAX_STAMPS};
use crate::viewport::ViewportGpu;
use crate::{App, FillScope, Pending, Tool};
use eframe::egui;
use op_core::{FillParams, Mask, MeshMap, NodeId, NodeKind};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpListener};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

pub struct Request {
    cmd: String,
    params: Value,
    reply: Sender<Result<Value, String>>,
}

/// How long a connection waits for the app to answer (it runs commands between frames).
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);

pub const COMMANDS: &[(&str, &str)] = &[
    ("info", "Project, texture sets, layer tree, tool and what is running."),
    ("status", "The status-bar message and whether a bake or export is running."),
    ("open", "{path}: open a .opproj project or a model (FBX/OBJ/glTF), discarding unsaved changes."),
    ("save", "{path?}: save the project (to path, or where it was opened from)."),
    ("set_active", "{set}: make a texture set (index) the active one."),
    ("add_layer", "{kind: paint|fill|group, name?, mask?: 'white'|'black', patch?}: add a layer above the selection; returns its id. patch is a partial layer (see get_layer)."),
    ("get_layer", "{id}: the layer as JSON."),
    ("set_layer", "{id, patch}: merge a partial layer into a layer, e.g. {\"opacity\":0.5,\"kind\":{\"Fill\":{\"base\":[1,0,0]}}}."),
    ("select_layer", "{id}: select a layer to paint on."),
    ("delete_layer", "{id}: delete a layer."),
    ("set_brush", "{tool?: paint|erase|spray, target?: content|mask, color?: [r,g,b] sRGB, radius?, hardness?, opacity?, spacing?, shape?: 0 round, 1 square, 2 diamond, 3 chalk, 4 streaks, 6 triangle, 7 star, 8 cross, 9 ring, 10 hexagon, 11 blot, metallic?, roughness?, mask_value?}"),
    ("paint_uv", "{points: [[u,v],...], radius?: UV units}: one stroke through the points on the selected layer, as one undo step."),
    ("screenshot", "{path}: save a PNG of the 3D view as it is drawn now."),
    ("quick_mask", "{on?, edit?, clear?}: protect areas: while on, paint lands only where the quick mask is free; with edit, paint_uv strokes mark protected areas (tool erase frees them); clear frees all."),
    ("geometry_mask", "{}: select the Fill tool on the selected layer's mask (adding a black one if missing) over mesh parts, so fill_uv shows the layer only where it fills."),
    ("fill_uv", "{points: [[u,v],...], scope?: triangle|face|island|part|set, hide?: bool}: fill the region of the mesh under each UV point on the selected layer (content, or its mask after geometry_mask)."),
    ("lasso_uv", "{points: [[u,v],...], hide?}: fill the polygon with these corners (in UV space) on the selected layer, as one undo step."),
    ("import_image", "{path}: add an image file to the texture library; returns its id (see `info` images)."),
    ("set_texture", "{id, image?: image id or null, slot?: base|metallic|roughness|height|normal|emissive|opacity}: put an image in a fill layer's channel (default base)."),
    ("add_decal", "{image, origin: [x,y,z], dir: [x,y,z], width?, height?, name?}: a fill layer with the image projected onto the model where the ray from origin along dir hits it (world units). Needs the Position and World space normal maps baked (`bake`)."),
    ("text_image", "{text, size?: px per line (default 128), font?: path to a .ttf/.otf}: render text as white letters on transparent and add it to the library; returns the image id (use it with add_decal or set_texture)."),
    ("reimport_mesh", "{path?}: read the model file again (default: the one it came from) keeping all paint and layers; the texture sets must match."),
    ("set_display", "{bloom?, sharpen?, vignette?, grain?: 0..1, exposure?: -3..3, fxaa?: bool}: the 3D view's look (not exported)."),
    ("camera", "{view?: front|back|left|right|top|bottom, yaw?, pitch?, distance? (radians / model units), frame?: true}: move the 3D camera, then use `screenshot`."),
    ("undo", "Undo the last edit or stroke."),
    ("redo", "Redo."),
    ("bake", "{maps?: [ao, curvature, thickness, normal, position, id, world_normal, height, bent_normals, opacity], samples?, size?}: bake the active set; poll `status`."),
    ("export", "{dir, preset?: name, padding?}: export the textures of every set; poll `status`."),
];

pub fn serve(port: u16, ctx: egui::Context) -> std::io::Result<Receiver<Request>> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let (tx, rx) = mpsc::channel::<Request>();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (tx, ctx) = (tx.clone(), ctx.clone());
            std::thread::spawn(move || {
                let Ok(read) = stream.try_clone() else { return };
                let mut out = stream;
                for line in BufReader::new(read).lines() {
                    let Ok(line) = line else { break };
                    if line.trim().is_empty() {
                        continue;
                    }
                    let reply = match serde_json::from_str::<Value>(&line) {
                        Ok(v) => {
                            let id = v.get("id").cloned().unwrap_or(Value::Null);
                            let (rtx, rrx) = mpsc::channel();
                            let req = Request {
                                cmd: v.get("cmd").and_then(Value::as_str).unwrap_or_default().to_string(),
                                params: v.get("params").cloned().unwrap_or(Value::Null),
                                reply: rtx,
                            };
                            let result = if tx.send(req).is_err() {
                                Err("the app is closing".to_string())
                            } else {
                                ctx.request_repaint();
                                rrx.recv_timeout(REPLY_TIMEOUT).unwrap_or_else(|_| Err("timed out waiting for the app".into()))
                            };
                            match result {
                                Ok(r) => json!({"id": id, "ok": true, "result": r}),
                                Err(e) => json!({"id": id, "ok": false, "error": e}),
                            }
                        }
                        Err(e) => json!({"id": null, "ok": false, "error": format!("not JSON: {e}")}),
                    };
                    if writeln!(out, "{reply}").is_err() {
                        break;
                    }
                }
                let _ = out.shutdown(Shutdown::Both);
            });
        }
    });
    Ok(rx)
}

fn str_param<'a>(p: &'a Value, key: &str) -> Result<&'a str, String> {
    p.get(key).and_then(Value::as_str).ok_or_else(|| format!("'{key}' (a string) is required"))
}

fn id_param(p: &Value) -> Result<NodeId, String> {
    p.get("id").and_then(Value::as_u64).ok_or_else(|| "'id' (a layer id from `info`) is required".to_string())
}

fn f32_param(p: &Value, key: &str) -> Option<f32> {
    p.get(key).and_then(Value::as_f64).map(|v| v as f32)
}

fn rgb_param(p: &Value, key: &str) -> Result<Option<[f32; 3]>, String> {
    let Some(v) = p.get(key) else { return Ok(None) };
    let a: Vec<f32> = v.as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default();
    match a[..] {
        [r, g, b] if [r, g, b].iter().all(|c| (0.0..=1.0).contains(c)) => Ok(Some([r, g, b])),
        _ => Err(format!("'{key}' must be [r, g, b], each 0..1")),
    }
}

/// `patch` laid over `base`: objects merge key by key, everything else replaces.
fn merge(base: &mut Value, patch: &Value) {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                merge(b.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (b, p) => *b = p.clone(),
    }
}

fn map_from_name(name: &str) -> Option<MeshMap> {
    Some(match name {
        "ao" => MeshMap::AmbientOcclusion,
        "curvature" => MeshMap::Curvature,
        "thickness" => MeshMap::Thickness,
        "normal" => MeshMap::NormalBase,
        "position" => MeshMap::Position,
        "id" => MeshMap::Id,
        "world_normal" => MeshMap::WorldSpaceNormal,
        "height" => MeshMap::HeightBase,
        "bent_normals" => MeshMap::BentNormals,
        "opacity" => MeshMap::OpacityBase,
        _ => return None,
    })
}

fn node_json(n: &op_core::Node, selected: Option<NodeId>) -> Value {
    let (kind, children) = match &n.kind {
        NodeKind::Paint => ("paint", Vec::new()),
        NodeKind::Fill(_) => ("fill", Vec::new()),
        NodeKind::Group { children } => ("group", children.iter().map(|c| node_json(c, selected)).collect()),
    };
    json!({"id": n.id, "name": n.name, "kind": kind, "visible": n.visible, "opacity": n.opacity, "blend": n.blend.name(),
           "mask": n.mask.is_some(), "selected": selected == Some(n.id), "children": children})
}

impl App {
    /// Runs the commands that arrived since the last frame.
    pub(crate) fn poll_control(&mut self, ctx: &egui::Context) {
        // A few frames between commands, so a read (export, save, screenshot) never sees the state from
        // before the edit ahead of it: strokes, new layers and images take a frame or two to reach the
        // composite.
        if self.control_wait > 0 {
            self.control_wait -= 1;
            ctx.request_repaint();
            return;
        }
        if let Some(req) = self.control.as_ref().and_then(|rx| rx.try_recv().ok()) {
            let result = self.run_command(&req.cmd, &req.params, ctx);
            let _ = req.reply.send(result);
            self.control_wait = 4;
            ctx.request_repaint();
        }
    }

    fn run_command(&mut self, cmd: &str, p: &Value, ctx: &egui::Context) -> Result<Value, String> {
        match cmd {
            "commands" => Ok(COMMANDS.iter().map(|(n, d)| json!({"name": n, "help": d})).collect()),
            "info" => Ok(json!({
                "project": self.project_name(), "path": self.project_path, "modified": self.modified, "mesh": self.mesh_name,
                "triangles": self.mesh.indices.len() / 3, "texture_size": self.tex_size, "active_set": self.active,
                "sets": (0..self.set_ranges_len()).map(|i| self.set_name(i)).collect::<Vec<_>>(),
                "layers": self.doc.nodes.iter().map(|n| node_json(n, self.selected)).collect::<Vec<_>>(),
                "selected": self.selected, "tool": if self.tool == Tool::Erase { "erase" } else { "paint" },
                "paint_on": if self.target == Target::Mask { "mask" } else { "content" },
                "images": self.assets.iter().map(|a| json!({"id": a.id, "name": a.name})).collect::<Vec<_>>(),
                "presets": self.presets.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
                "status": self.status,
            })),
            "status" => Ok(json!({"status": self.status, "baking": self.bake.job.is_some(), "exporting": self.export.running.is_some()})),
            "open" => {
                let path = PathBuf::from(str_param(p, "path")?);
                if !path.exists() {
                    return Err(format!("{} does not exist", path.display()));
                }
                self.perform(if crate::is_project(&path) { Pending::Open(path) } else { Pending::Mesh(path) }, ctx);
                Ok(json!(self.status))
            }
            "save" => {
                let path = match p.get("path").and_then(Value::as_str) {
                    Some(s) => PathBuf::from(s),
                    None => self.project_path.clone().ok_or("the project has no path yet: pass 'path'")?,
                };
                self.do_save(path, ctx);
                Ok(json!(self.status))
            }
            "set_active" => {
                let i = p.get("set").and_then(Value::as_u64).ok_or("'set' (an index) is required")? as usize;
                if i >= self.set_ranges_len() {
                    return Err(format!("there are {} texture sets", self.set_ranges_len()));
                }
                self.switch_set(i);
                Ok(json!(self.active))
            }
            "get_layer" => {
                let node = self.doc.find(id_param(p)?).ok_or("no such layer")?;
                serde_json::to_value(node).map_err(|e| e.to_string())
            }
            "add_layer" => {
                let kind = match str_param(p, "kind")? {
                    "paint" => NodeKind::Paint,
                    "fill" => NodeKind::Fill(FillParams::default()),
                    "group" => NodeKind::Group { children: Vec::new() },
                    other => return Err(format!("unknown kind '{other}': paint, fill or group")),
                };
                self.add_node(p.get("name").and_then(Value::as_str).unwrap_or("Layer"), kind);
                let id = self.selected.ok_or("could not add the layer")?;
                let mut patch = p.get("patch").cloned().unwrap_or(json!({}));
                if let Some(m) = p.get("mask").and_then(Value::as_str) {
                    patch["mask"] = serde_json::to_value(Mask::new(m != "black")).map_err(|e| e.to_string())?;
                }
                if let Some(n) = p.get("name") {
                    patch["name"] = n.clone();
                }
                self.patch_layer(id, &patch)?;
                Ok(json!(id))
            }
            "set_layer" => {
                let id = id_param(p)?;
                self.patch_layer(id, p.get("patch").ok_or("'patch' is required")?)?;
                Ok(json!(id))
            }
            "select_layer" => {
                let id = id_param(p)?;
                self.doc.find(id).ok_or("no such layer")?;
                self.selected = Some(id);
                Ok(json!(id))
            }
            "delete_layer" => {
                let id = id_param(p)?;
                self.doc.find(id).ok_or("no such layer")?;
                self.edit(|d| {
                    d.remove(id);
                });
                self.fix_selection();
                Ok(json!(true))
            }
            "set_brush" => {
                match p.get("tool").and_then(Value::as_str) {
                    Some("paint") => self.tool = Tool::Paint,
                    Some("erase") => self.tool = Tool::Erase,
                    Some("spray") => self.tool = Tool::Spray,
                    Some(o) => return Err(format!("unknown tool '{o}': paint or erase")),
                    None => {}
                }
                match p.get("target").and_then(Value::as_str) {
                    Some("content") => self.target = Target::Content,
                    Some("mask") => self.target = Target::Mask,
                    Some(o) => return Err(format!("unknown target '{o}': content or mask")),
                    None => {}
                }
                if let Some(c) = rgb_param(p, "color")? {
                    self.channels.base_color = c;
                }
                if let Some(v) = f32_param(p, "hardness") {
                    self.brush.hardness = v.clamp(0.0, 1.0);
                }
                if let Some(v) = f32_param(p, "opacity") {
                    self.brush.opacity = v.clamp(0.0, 1.0);
                }
                if let Some(v) = p.get("shape").and_then(Value::as_u64) {
                    self.brush.shape = (v as u32).min(11);
                }
                if let Some(v) = f32_param(p, "spacing") {
                    self.brush.spacing = v.clamp(0.01, 2.0);
                }
                if let Some(v) = f32_param(p, "metallic") {
                    self.channels.metallic = v.clamp(0.0, 1.0);
                }
                if let Some(v) = f32_param(p, "roughness") {
                    self.channels.roughness = v.clamp(0.0, 1.0);
                }
                if let Some(v) = f32_param(p, "mask_value") {
                    self.mask_value = Some(v.clamp(0.0, 1.0));
                }
                Ok(json!(true))
            }
            "screenshot" => {
                let path = PathBuf::from(str_param(p, "path")?);
                let shot = self.render_state.renderer.read().callback_resources.get::<ViewportGpu>().and_then(|g| g.screenshot(&self.render_state.device, &self.render_state.queue));
                let (w, h, mut rgba) = shot.ok_or("the 3D view has not been drawn yet")?;
                rgba.chunks_exact_mut(4).for_each(|px| px[3] = 255);
                let png = op_image::encode_png(w, h, &rgba).map_err(|e| e.to_string())?;
                std::fs::write(&path, png).map_err(|e| e.to_string())?;
                Ok(json!(path))
            }
            "quick_mask" => {
                if let Some(on) = p.get("on").and_then(Value::as_bool) {
                    self.quick_on = on;
                    self.quick_edit &= on;
                }
                if let Some(edit) = p.get("edit").and_then(Value::as_bool) {
                    self.quick_edit = edit && self.quick_on;
                }
                if p.get("clear").and_then(Value::as_bool) == Some(true) {
                    self.pending.push(PaintOp::QuickClear);
                }
                Ok(json!({"on": self.quick_on, "edit": self.quick_edit}))
            }
            "geometry_mask" => {
                self.use_geometry_mask();
                Ok(json!(self.status))
            }
            "fill_uv" => {
                let points: Vec<[f32; 2]> = p
                    .get("points")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|pt| Some([pt.get(0)?.as_f64()? as f32, pt.get(1)?.as_f64()? as f32])).collect())
                    .unwrap_or_default();
                if points.is_empty() {
                    return Err("'points' must be a list of [u, v]".into());
                }
                if let Some(scope) = p.get("scope").and_then(Value::as_str) {
                    self.fill_scope = match scope {
                        "triangle" => FillScope::Triangle,
                        "face" => FillScope::Face,
                        "island" => FillScope::UvIsland,
                        "part" => FillScope::MeshPart,
                        "set" => FillScope::TextureSet,
                        other => return Err(format!("unknown scope '{other}': triangle, face, island, part or set")),
                    };
                }
                let hide = p.get("hide").and_then(Value::as_bool).unwrap_or(false);
                self.paint_target().map_err(str::to_string)?;
                for pt in points {
                    let picked = self.picked_triangles_uv(pt);
                    self.fill_picked(picked, hide);
                }
                Ok(json!(self.status))
            }
            "lasso_uv" => {
                let points: Vec<[f32; 2]> = p
                    .get("points")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|pt| Some([pt.get(0)?.as_f64()? as f32, pt.get(1)?.as_f64()? as f32])).collect())
                    .unwrap_or_default();
                if points.len() < 3 {
                    return Err("'points' needs at least three [u, v] corners".into());
                }
                self.lasso_fill_uv(&points, p.get("hide").and_then(Value::as_bool).unwrap_or(false));
                Ok(json!(self.status))
            }
            "import_image" => self.import_texture_file(std::path::Path::new(str_param(p, "path")?)).map(|id| json!(id)),
            "set_texture" => {
                let id = id_param(p)?;
                let slot = match p.get("slot").and_then(Value::as_str).unwrap_or("base") {
                    "base" => op_core::Slot::BaseColor,
                    "metallic" => op_core::Slot::Metallic,
                    "roughness" => op_core::Slot::Roughness,
                    "height" => op_core::Slot::Height,
                    "normal" => op_core::Slot::Normal,
                    "emissive" => op_core::Slot::Emissive,
                    "opacity" => op_core::Slot::Opacity,
                    other => return Err(format!("unknown slot '{other}'")),
                };
                let tex = match p.get("image").and_then(Value::as_str) {
                    Some(img) => Some(self.texture(img).map(|_| op_core::TexRef::new(img)).ok_or("no such image: see `info`")?),
                    None => None,
                };
                if !matches!(self.doc.find(id).map(|n| &n.kind), Some(NodeKind::Fill(_))) {
                    return Err("that is not a fill layer".into());
                }
                self.set_slot_texture(id, slot, tex);
                Ok(json!(id))
            }
            "add_decal" => {
                let vec3 = |key: &str| -> Result<glam::Vec3, String> {
                    let a: Vec<f32> = p.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default();
                    match a[..] {
                        [x, y, z] if [x, y, z].iter().all(|c| c.is_finite()) => Ok(glam::Vec3::new(x, y, z)),
                        _ => Err(format!("'{key}' must be [x, y, z]")),
                    }
                };
                let (origin, dir) = (vec3("origin")?, vec3("dir")?.normalize_or_zero());
                let image = str_param(p, "image")?;
                self.texture(image).ok_or("no such image: see `info`")?;
                let before = self.selected;
                self.add_node(p.get("name").and_then(Value::as_str).unwrap_or("Decal"), NodeKind::Fill(FillParams::default()));
                let id = self.selected.ok_or("could not add the layer")?;
                self.set_slot_texture(id, op_core::Slot::BaseColor, Some(op_core::TexRef::new(image)));
                let (w, h) = (f32_param(p, "width").unwrap_or(0.25), f32_param(p, "height").unwrap_or(0.25));
                self.edit(|d| {
                    if let Some(NodeKind::Fill(f)) = d.find_mut(id).map(|n| &mut n.kind) {
                        f.decal.u = [w, 0.0, 0.0];
                        f.decal.v = [0.0, h, 0.0];
                    }
                });
                if let Err(e) = self.place_decal(id, origin, dir) {
                    self.edit(|d| {
                        d.remove(id);
                    });
                    self.selected = before;
                    return Err(e);
                }
                Ok(json!(id))
            }
            "text_image" => {
                let font = p.get("font").and_then(Value::as_str).map(PathBuf::from);
                self.make_text_image(str_param(p, "text")?, font.as_deref(), f32_param(p, "size").unwrap_or(128.0)).map(|id| json!(id))
            }
            "reimport_mesh" => {
                let path = match p.get("path").and_then(Value::as_str) {
                    Some(s) => PathBuf::from(s),
                    None => self.mesh_source.clone().ok_or("this model was not loaded from a file: pass 'path'")?,
                };
                self.reimport_mesh(&path).map(|s| json!(s))
            }
            "set_display" => {
                let e = &mut self.env;
                for (key, slot) in [("bloom", &mut e.bloom), ("sharpen", &mut e.sharpen), ("vignette", &mut e.vignette), ("grain", &mut e.grain)] {
                    if let Some(v) = f32_param(p, key) {
                        *slot = v.clamp(0.0, 1.0);
                    }
                }
                if let Some(v) = f32_param(p, "exposure") {
                    e.exposure = v.clamp(-3.0, 3.0);
                }
                if let Some(b) = p.get("fxaa").and_then(Value::as_bool) {
                    e.fxaa = b;
                }
                Ok(json!(true))
            }
            "camera" => {
                if let Some(v) = p.get("view").and_then(Value::as_str) {
                    let preset = crate::camera::ViewPreset::ALL.into_iter().find(|x| x.label().eq_ignore_ascii_case(v)).ok_or("view must be front, back, left, right, top or bottom")?;
                    self.camera.set_view(preset);
                }
                if p.get("frame").and_then(Value::as_bool) == Some(true) {
                    self.frame_model();
                }
                if let Some(v) = f32_param(p, "yaw") {
                    self.camera.yaw = v;
                }
                if let Some(v) = f32_param(p, "pitch") {
                    self.camera.pitch = v.clamp(-1.55, 1.55);
                }
                if let Some(v) = f32_param(p, "distance") {
                    self.camera.distance = v.clamp(1e-3, 1e6);
                }
                Ok(json!({"yaw": self.camera.yaw, "pitch": self.camera.pitch, "distance": self.camera.distance}))
            }
            "paint_uv" => self.control_paint_uv(p),
            "undo" => {
                self.do_undo();
                Ok(json!(self.status))
            }
            "redo" => {
                self.do_redo();
                Ok(json!(self.status))
            }
            "bake" => {
                if self.bake.job.is_some() {
                    return Err("a bake is already running".into());
                }
                if let Some(names) = p.get("maps").and_then(Value::as_array) {
                    self.bake.maps = names
                        .iter()
                        .map(|n| n.as_str().and_then(map_from_name).ok_or_else(|| format!("unknown map {n}")))
                        .collect::<Result<_, _>>()?;
                }
                if let Some(s) = p.get("samples").and_then(Value::as_u64) {
                    self.bake.samples = (s as u32).clamp(1, 256);
                }
                self.start_bake();
                Ok(json!(self.status))
            }
            "export" => {
                if self.export.running.is_some() {
                    return Err("an export is already running".into());
                }
                self.export.dir = str_param(p, "dir")?.to_string();
                if let Some(name) = p.get("preset").and_then(Value::as_str) {
                    self.export.preset = self.presets.iter().position(|x| x.name == name).ok_or_else(|| format!("no preset '{name}': see `info`"))?;
                }
                if let Some(n) = p.get("padding").and_then(Value::as_u64) {
                    self.export.padding = (n as u32).min(256);
                }
                self.export.skip_sets.clear();
                self.start_export();
                Ok(json!(self.status))
            }
            other => Err(format!("unknown command '{other}': see `commands`")),
        }
    }

    fn patch_layer(&mut self, id: NodeId, patch: &Value) -> Result<(), String> {
        let node = self.doc.find(id).ok_or("no such layer")?;
        let mut v = serde_json::to_value(node).map_err(|e| e.to_string())?;
        merge(&mut v, patch);
        v["id"] = json!(id);
        let mut new: op_core::Node = serde_json::from_value(v).map_err(|e| format!("bad patch: {e}"))?;
        new.opacity = new.opacity.clamp(0.0, 1.0);
        let before = self.doc.clone();
        *self.doc.find_mut(id).ok_or("no such layer")? = new;
        self.record(before);
        Ok(())
    }

    fn control_paint_uv(&mut self, p: &Value) -> Result<Value, String> {
        let points: Vec<[f32; 2]> = p
            .get("points")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|pt| {
                        let pt = pt.as_array()?;
                        Some([pt.first()?.as_f64()? as f32, pt.get(1)?.as_f64()? as f32])
                    })
                    .collect()
            })
            .unwrap_or_default();
        if points.is_empty() {
            return Err("'points' must be a list of [u, v]".into());
        }
        let radius = f32_param(p, "radius").unwrap_or(0.02).clamp(1e-4, 1.0);
        let (layer, target) = self.paint_target().map_err(str::to_string)?;
        let length: f32 = points.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum();
        if length / (radius * self.brush.spacing).max(1e-5) >= MAX_STAMPS as f32 {
            return Err("that stroke is too long for one call: use a bigger radius or spacing, or split it".into());
        }
        let mut stamps = Vec::new();
        self.stroke.active = false;
        for &pt in &points {
            self.stroke_to_uv(pt, radius, layer, target, &mut stamps);
        }
        self.stroke.active = false;
        self.stroke_was_active = false;
        let erase = self.tool == Tool::Erase && target == Target::Content;
        self.pending.push(if erase { PaintOp::Erase { layer, stamps, uv: true } } else { PaintOp::Stamps2d { layer, target, stamps } });
        self.pending.push(PaintOp::EndStroke);
        self.thumbs_dirty();
        self.modified = true;
        Ok(json!(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_merges_into_nested_objects_and_replaces_the_rest() {
        let mut base = json!({"opacity": 1.0, "kind": {"Fill": {"base": [1, 0, 0], "metallic": 0.2}}});
        merge(&mut base, &json!({"kind": {"Fill": {"base": [0, 0, 1]}}, "opacity": 0.5}));
        assert_eq!(base, json!({"opacity": 0.5, "kind": {"Fill": {"base": [0, 0, 1], "metallic": 0.2}}}));
    }

    #[test]
    fn colours_must_be_three_numbers_in_range() {
        assert_eq!(rgb_param(&json!({"c": [0, 0.5, 1]}), "c"), Ok(Some([0.0, 0.5, 1.0])));
        assert!(rgb_param(&json!({"c": [0, 2, 1]}), "c").is_err());
        assert!(rgb_param(&json!({"c": [1, 1]}), "c").is_err());
        assert_eq!(rgb_param(&json!({}), "c"), Ok(None));
    }
}
