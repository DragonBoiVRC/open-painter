//! The texture library: images you import into the shelf and drag onto fill-layer slots.
//!
//! A library entry is the original file's bytes under a content id. Images are decoded when first
//! needed (previews, upload to the GPU). Imported files are kept in the user's data folder so the
//! library is there next session, and the ones a project uses are embedded in the project.

use crate::{files, viewport::ViewportGpu, App};
use op_core::{Document, FillParams, NodeKind, Slot, TexRef};
use op_image::{Decoded, Guess};
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// How many files the on-disk library may hold; more are ignored when loading.
const MAX_LIBRARY_FILES: usize = 500;
/// Separates the id from the original name in a library file name.
const NAME_SEPARATOR: &str = "--";

/// File extensions the import accepts.
pub const IMAGE_EXTENSIONS: [&str; 9] = ["png", "jpg", "jpeg", "tga", "bmp", "tif", "tiff", "webp", "exr"];

pub fn is_image_path(p: &Path) -> bool {
    p.extension().is_some_and(|e| IMAGE_EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

pub struct TextureAsset {
    pub id: String,
    pub name: String,
    pub bytes: Arc<Vec<u8>>,
    image: OnceLock<Option<Arc<Decoded>>>,
}

impl TextureAsset {
    pub fn new(id: String, name: String, bytes: Vec<u8>) -> Self {
        Self { id, name, bytes: Arc::new(bytes), image: OnceLock::new() }
    }

    /// The decoded pixels, decoding on first use. `None` if the bytes are not a usable image.
    pub fn decoded(&self) -> Option<&Arc<Decoded>> {
        self.image.get_or_init(|| op_image::decode(&self.bytes).ok().map(Arc::new)).as_ref()
    }

    pub fn guess(&self) -> Option<Guess> {
        op_image::guess_slot(&self.name)
    }
}

/// A file name that is safe to store: no directory parts and only plain characters.
pub fn clean_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let cleaned: String = base.chars().map(|c| if c.is_alphanumeric() || "._- ()".contains(c) { c } else { '_' }).collect();
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == ' ').to_string();
    if cleaned.is_empty() { "texture".into() } else { cleaned.chars().take(120).collect() }
}

/// The library file name for an asset, and the way back.
pub fn library_file_name(id: &str, name: &str) -> String {
    format!("{id}{NAME_SEPARATOR}{}", clean_name(name))
}

pub fn parse_library_file_name(file: &str) -> Option<(String, String)> {
    let (id, name) = file.split_once(NAME_SEPARATOR)?;
    op_core::is_valid_texture_id(id).then(|| (id.to_string(), name.to_string()))
}

impl App {
    pub(crate) fn texture(&self, id: &str) -> Option<&TextureAsset> {
        self.assets.iter().find(|a| a.id == id)
    }

    /// Add an image to the library from its original bytes. Returns its id; an image already in the
    /// library (same content) is not added twice.
    pub(crate) fn add_texture(&mut self, name: &str, bytes: Vec<u8>) -> Result<String, String> {
        let id = op_image::content_id(&bytes);
        if self.texture(&id).is_some() {
            return Ok(id);
        }
        let asset = TextureAsset::new(id.clone(), clean_name(name), bytes);
        if asset.decoded().is_none() {
            // Decode now so a bad file is reported to the user at import, not later.
            return Err(match op_image::decode(&asset.bytes) {
                Err(e) => format!("{name}: {e}"),
                Ok(_) => format!("{name}: could not be read"),
            });
        }
        self.assets.push(asset);
        Ok(id)
    }

    /// Import an image file: add it to the library and keep a copy in the user's library folder.
    pub(crate) fn import_texture_file(&mut self, path: &Path) -> Result<String, String> {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "texture".into());
        let meta = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?;
        if meta.len() as usize > op_image::MAX_FILE_BYTES {
            return Err(format!("{name}: the file is larger than {} MB", op_image::MAX_FILE_BYTES >> 20));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
        let id = self.add_texture(&name, bytes)?;
        if let Some(a) = self.texture(&id) {
            let dir = files::textures_dir();
            let target = dir.join(library_file_name(&a.id, &a.name));
            if !target.exists() && std::fs::create_dir_all(&dir).is_ok() {
                let _ = std::fs::write(&target, &*a.bytes);
            }
        }
        Ok(id)
    }

    /// Put a texture on a brand new fill layer, in the slot its name suggests.
    pub(crate) fn use_texture_on_new_layer(&mut self, id: &str) {
        let Some(asset) = self.texture(id) else { return };
        let guess = asset.guess();
        let slot = guess.map_or(Slot::BaseColor, |g| g.slot);
        let mut f = FillParams::default();
        f.maps.set(slot, Some(TexRef { flip_green: guess.is_some_and(|g| g.directx), ..TexRef::new(id) }));
        self.add_node("Textured fill", NodeKind::Fill(f));
    }

    /// Use a texture for one specific channel of the selected fill layer (or a new fill layer).
    pub(crate) fn use_texture_as(&mut self, id: &str, slot: Slot) {
        let Some(asset) = self.texture(id) else { return };
        let directx = asset.guess().is_some_and(|g| g.directx);
        let tex = TexRef { flip_green: slot == Slot::Normal && directx, ..TexRef::new(id) };
        match self.selected.filter(|s| matches!(self.doc.find(*s).map(|n| &n.kind), Some(NodeKind::Fill(_)))) {
            Some(layer) => self.set_slot_texture(layer, slot, Some(tex)),
            None => {
                let mut f = FillParams::default();
                f.maps.set(slot, Some(tex));
                self.add_node("Textured fill", NodeKind::Fill(f));
            }
        }
    }

    /// Open the file dialog for importing textures; the chosen files arrive in `poll_texture_import`.
    pub(crate) fn start_texture_import(&mut self) {
        if self.import_rx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let files = rfd::FileDialog::new().add_filter("Images", &IMAGE_EXTENSIONS).set_title("Import textures").pick_files();
            let _ = tx.send(files.unwrap_or_default());
        });
        self.import_rx = Some(rx);
    }

    pub(crate) fn poll_texture_import(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.import_rx else { return };
        match rx.try_recv() {
            Ok(files) => {
                self.import_rx = None;
                if !files.is_empty() {
                    self.import_texture_files(&files);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(150)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.import_rx = None,
        }
    }

    /// Import several files and say how it went.
    pub(crate) fn import_texture_files(&mut self, paths: &[std::path::PathBuf]) {
        let (mut ok, mut errors) = (0, Vec::new());
        for p in paths {
            match self.import_texture_file(p) {
                Ok(_) => ok += 1,
                Err(e) => errors.push(e),
            }
        }
        self.status = match (ok, errors.first()) {
            (0, Some(e)) => format!("Import failed: {e}"),
            (n, None) => format!("Imported {n} texture(s)."),
            (n, Some(e)) => format!("Imported {n} texture(s); {} failed (first: {e})", errors.len()),
        };
        if ok > 0 {
            self.shelf.category = crate::ui::panels::shelf_textures();
        }
    }

    /// Read the library folder at startup. Files that do not parse as `<id>--<name>` are ignored.
    pub(crate) fn load_texture_library(&mut self) {
        let Ok(rd) = std::fs::read_dir(files::textures_dir()) else { return };
        for entry in rd.flatten().take(MAX_LIBRARY_FILES) {
            let Some((id, name)) = entry.file_name().to_str().and_then(parse_library_file_name) else { continue };
            if self.texture(&id).is_some() {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() || meta.len() as usize > op_image::MAX_FILE_BYTES {
                continue;
            }
            if let Ok(bytes) = std::fs::read(entry.path()) {
                // The file name claims an id; only trust it if the content agrees.
                if op_image::content_id(&bytes) == id {
                    self.assets.push(TextureAsset::new(id, name, bytes));
                }
            }
        }
        self.assets.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    }

    /// Remove an image from the library (and from disk). Layers that use it fall back to their constants.
    pub(crate) fn remove_texture(&mut self, id: &str) {
        if let Some(i) = self.assets.iter().position(|a| a.id == id) {
            let a = self.assets.remove(i);
            let _ = std::fs::remove_file(files::textures_dir().join(library_file_name(&a.id, &a.name)));
            self.tex_thumbs.remove(id);
        }
    }

    /// Make sure every texture `doc` uses is on the GPU. Safe to call every frame: it only works for
    /// images the GPU does not have yet.
    pub(crate) fn ensure_doc_images(&self, doc: &Document) {
        let ids = doc.texture_ids();
        if ids.is_empty() {
            return;
        }
        let mut renderer = self.render_state.renderer.write();
        let Some(gpu) = renderer.callback_resources.get_mut::<ViewportGpu>() else { return };
        for id in ids {
            if gpu.has_image(&id) {
                continue;
            }
            if let Some(img) = self.texture(&id).and_then(|a| a.decoded()) {
                gpu.upload_image(&self.render_state.device, &self.render_state.queue, &id, img);
            }
        }
    }

    /// Send the brush's alpha and stencil pictures to the GPU when they change, uploading any that
    /// it does not have yet. A picture that has left the library is dropped from the brush.
    pub(crate) fn sync_brush_pictures(&mut self) {
        if self.brush_pics.alpha.as_ref().is_some_and(|id| self.texture(id).is_none()) {
            self.brush_pics.alpha = None;
        }
        if self.brush_pics.stencil.as_ref().is_some_and(|id| self.texture(id).is_none()) {
            self.brush_pics.stencil = None;
        }
        if self.brush_pics_sent.as_ref() == Some(&self.brush_pics) {
            return;
        }
        let mut renderer = self.render_state.renderer.write();
        let Some(gpu) = renderer.callback_resources.get_mut::<ViewportGpu>() else { return };
        for id in [&self.brush_pics.alpha, &self.brush_pics.stencil].into_iter().flatten() {
            if !gpu.has_image(id) {
                if let Some(img) = self.texture(id).and_then(|a| a.decoded()) {
                    gpu.upload_image(&self.render_state.device, &self.render_state.queue, id, img);
                }
            }
        }
        gpu.set_brush_pictures(self.brush_pics.clone());
        drop(renderer);
        self.brush_pics_sent = Some(self.brush_pics.clone());
    }

    /// Small egui textures for the shelf and layer rows, made on demand.
    pub(crate) fn ensure_tex_thumbs(&mut self, ctx: &egui::Context, ids: &[String]) {
        // Decoding is the slow part, so only a few per frame; the rest arrive on the next ones.
        let mut budget = 3;
        for id in ids {
            if self.tex_thumbs.contains_key(id) {
                continue;
            }
            if budget == 0 {
                ctx.request_repaint();
                break;
            }
            budget -= 1;
            let Some(small) = self.texture(id).and_then(|a| a.decoded()).map(|d| op_image::thumbnail(d, 96)) else { continue };
            let image = egui::ColorImage::from_rgba_unmultiplied([small.width as usize, small.height as usize], &small.rgba8);
            let handle = ctx.load_texture(format!("tex-{id}"), image, egui::TextureOptions::LINEAR);
            self.tex_thumbs.insert(id.clone(), handle);
        }
    }

    /// Set `slot` of fill layer `layer` to `tex`, as one undoable edit.
    pub(crate) fn set_slot_texture(&mut self, layer: op_core::NodeId, slot: Slot, tex: Option<TexRef>) {
        self.edit(|d| {
            if let Some(NodeKind::Fill(f)) = d.find_mut(layer).map(|n| &mut n.kind) {
                f.maps.set(slot, tex);
            }
        });
    }

    /// Use a texture where its name says it belongs: on the selected fill layer, or on a new fill
    /// layer if the selection is not one. A name that says nothing means base color.
    pub(crate) fn use_texture(&mut self, id: &str) {
        let Some(asset) = self.texture(id) else { return };
        let guess = asset.guess();
        let slot = guess.map_or(Slot::BaseColor, |g| g.slot);
        let tex = TexRef { flip_green: guess.is_some_and(|g| g.directx), ..TexRef::new(id) };
        let label = asset.name.clone();
        match self.selected.filter(|s| matches!(self.doc.find(*s).map(|n| &n.kind), Some(NodeKind::Fill(_)))) {
            Some(layer) => self.set_slot_texture(layer, slot, Some(tex)),
            None => {
                let mut f = FillParams::default();
                f.maps.set(slot, Some(tex));
                self.add_node("Textured fill", NodeKind::Fill(f));
            }
        }
        self.status = format!("'{label}' is now the {} of the layer.", slot.label().to_lowercase());
    }
}

/// File extensions an HDR environment can come in.
const HDRI_EXTENSIONS: [&str; 3] = ["hdr", "exr", "png"];

/// A file name that is safe to keep in the environments folder.
fn hdri_file_name(name: &str) -> String {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' ') { c } else { '_' }).collect();
    let clean = clean.trim_matches(|c| c == '.' || c == ' ').to_string();
    if clean.is_empty() { "environment.hdr".into() } else { clean.chars().take(96).collect() }
}

impl App {
    /// The environments saved in the library folder, by file name. The folder is only re-read every
    /// few seconds (and after an import), not on every frame.
    pub(crate) fn hdri_library(&mut self) -> Vec<String> {
        let fresh = self.hdri_files.as_ref().is_some_and(|(at, _)| at.elapsed() < std::time::Duration::from_secs(3));
        if !fresh {
            self.hdri_files = Some((std::time::Instant::now(), self.scan_hdri_library()));
        }
        self.hdri_files.as_ref().map(|(_, v)| v.clone()).unwrap_or_default()
    }

    fn scan_hdri_library(&self) -> Vec<String> {
        let Ok(rd) = std::fs::read_dir(files::environments_dir()) else { return Vec::new() };
        let mut names: Vec<String> = rd
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()).is_some_and(|x| HDRI_EXTENSIONS.contains(&x.to_ascii_lowercase().as_str())))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort_by_key(|n| n.to_lowercase());
        names
    }

    /// Ask for an HDR environment file, keep a copy in the library and use it.
    pub(crate) fn start_hdri_import(&mut self) {
        if self.hdri_rx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Some(path) = rfd::FileDialog::new().add_filter("HDR environments", &HDRI_EXTENSIONS).set_title("Load an HDR environment").pick_file() else {
                let _ = tx.send(Err(String::new()));
                return;
            };
            let _ = tx.send(read_hdri(&path, true));
        });
        self.hdri_rx = Some(rx);
        self.status = "Reading the environment...".into();
    }

    /// Use an environment from the library folder.
    pub(crate) fn use_hdri_from_library(&mut self, name: &str) {
        if self.hdri_rx.is_some() {
            return;
        }
        let path = files::environments_dir().join(hdri_file_name(name));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(read_hdri(&path, false));
        });
        self.hdri_rx = Some(rx);
        self.status = "Reading the environment...".into();
    }

    pub(crate) fn poll_hdri(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.hdri_rx else { return };
        match rx.try_recv() {
            Ok(result) => {
                self.hdri_rx = None;
                match result {
                    Ok((name, img)) => {
                        self.hdri_files = None; // a new file may have joined the library
                        let mut renderer = self.render_state.renderer.write();
                        if let Some(gpu) = renderer.callback_resources.get_mut::<ViewportGpu>() {
                            gpu.set_environment(&self.render_state.device, &self.render_state.queue, Some(&img));
                        }
                        drop(renderer);
                        self.env.preset = 4;
                        self.status = format!("Lighting from {name}");
                        self.hdri_name = Some(name);
                    }
                    Err(e) if e.is_empty() => self.status.clear(),
                    Err(e) => {
                        // The remembered HDRI is gone or unreadable: go back to a built-in environment.
                        if self.hdri_name.is_none() && self.env.preset == 4 {
                            self.env.preset = 0;
                        }
                        self.status = format!("Could not use that environment: {e}");
                    }
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(150)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.hdri_rx = None,
        }
    }
}

/// Read and decode an environment on a worker thread; with `save`, also copy it into the library.
fn read_hdri(path: &Path, save: bool) -> Result<(String, op_image::hdr::HdrImage), String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() as usize > op_image::MAX_FILE_BYTES {
        return Err("the file is too large".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let img = op_image::hdr::decode_hdr(&bytes).map_err(|e| e.to_string())?;
    let name = hdri_file_name(&path.file_name().and_then(|n| n.to_str()).unwrap_or("environment.hdr").to_string());
    if save {
        let dir = files::environments_dir();
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = std::fs::write(dir.join(&name), &bytes);
        }
    }
    Ok((name, img))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_file_names_round_trip_and_reject_impostors() {
        let id = "0123456789abcdef0123456789abcdef";
        let file = library_file_name(id, "wood/planks BaseColor.png");
        assert_eq!(parse_library_file_name(&file), Some((id.to_string(), "planks BaseColor.png".to_string())));
        for bad in ["wood.png", "../x--y.png", "0123--x.png", &format!("{}--x", "G".repeat(32)), ""] {
            assert!(parse_library_file_name(bad).is_none(), "{bad}");
        }
        // A name can contain the separator without confusing the id.
        let tricky = library_file_name(id, "a--b.png");
        assert_eq!(parse_library_file_name(&tricky).unwrap().0, id);
    }

    #[test]
    fn names_are_cleaned_for_storage() {
        assert_eq!(clean_name("../../etc/passwd"), "passwd");
        assert_eq!(clean_name("C:\\tex\\wood.png"), "wood.png");
        assert_eq!(clean_name(".."), "texture");
        assert_eq!(clean_name("  "), "texture");
        assert_eq!(clean_name("naïve:tex*.png"), "naïve_tex_.png");
        assert!(clean_name(&"a".repeat(500)).len() <= 120);
    }

    #[test]
    fn only_image_files_are_recognised() {
        for ok in ["a.png", "A.JPG", "x.tga", "y.EXR", "z.tiff", "w.webp"] {
            assert!(is_image_path(Path::new(ok)), "{ok}");
        }
        for no in ["a.fbx", "b.opproj", "c", "d.png.txt", "e.glb"] {
            assert!(!is_image_path(Path::new(no)), "{no}");
        }
    }
}
