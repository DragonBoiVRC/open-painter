//! Native file dialogs. They block, so each runs on its own thread and the app polls for the answer.

use crate::App;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};

/// Which text field a chosen path goes into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    OpenPath,
    SavePath,
    ExportDir,
    PresetImport,
    HighPoly,
}

pub struct Picker {
    field: Field,
    rx: Receiver<Option<PathBuf>>,
}

/// File types offered for each kind of path.
fn filters(field: Field) -> (&'static str, &'static [&'static str]) {
    match field {
        Field::OpenPath => ("Projects and models", &["opproj", "fbx", "obj", "glb", "gltf"]),
        Field::SavePath => ("Project", &["opproj"]),
        Field::PresetImport => ("Substance Painter export preset", &["spexp"]),
        Field::HighPoly => ("Model", &["glb", "gltf"]),
        Field::ExportDir => ("", &[]),
    }
}

impl App {
    /// Open a native dialog for `field`, starting at the field's current text if it is a path.
    pub(crate) fn browse(&mut self, field: Field, current: &str) {
        if self.picker.is_none() {
            self.browse_req = Some((field, current.to_string()));
        }
    }

    /// Start the dialog `browse` asked for. It needs the window, so the dialog opens on top of it.
    pub(crate) fn start_browse(&mut self, frame: &eframe::Frame) {
        let Some((field, current)) = self.browse_req.take() else { return };
        let current = current.as_str();
        let (label, exts) = filters(field);
        let mut dialog = rfd::FileDialog::new().set_parent(frame);
        if !exts.is_empty() {
            dialog = dialog.add_filter(label, exts);
        }
        let start = std::path::Path::new(current.trim());
        let dir = if start.is_dir() { Some(start) } else { start.parent().filter(|p| p.is_dir()) };
        if let Some(d) = dir {
            dialog = dialog.set_directory(d);
        }
        if field == Field::SavePath {
            if let Some(name) = start.file_name() {
                dialog = dialog.set_file_name(name.to_string_lossy());
            }
        }
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let picked = match field {
                Field::ExportDir => dialog.pick_folder(),
                Field::SavePath => dialog.save_file(),
                _ => dialog.pick_file(),
            };
            let _ = tx.send(picked);
        });
        self.picker = Some(Picker { field, rx });
    }

    /// Put a finished dialog's answer into its field.
    pub(crate) fn poll_picker(&mut self, ctx: &egui::Context) {
        let Some(p) = &self.picker else { return };
        match p.rx.try_recv() {
            Ok(path) => {
                let field = p.field;
                self.picker = None;
                if let Some(path) = path {
                    let text = path.display().to_string();
                    match field {
                        Field::OpenPath | Field::SavePath => {
                            if let Some(d) = &mut self.dialog {
                                d.path = text;
                            }
                        }
                        Field::ExportDir => self.export.dir = text,
                        Field::PresetImport => self.export.import_path = text,
                        Field::HighPoly => self.bake.high_path = text,
                    }
                }
                ctx.request_repaint();
            }
            Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(150)),
            Err(mpsc::TryRecvError::Disconnected) => self.picker = None,
        }
    }
}

/// A "Browse..." button that is disabled while another dialog is open.
pub fn browse_button(ui: &mut egui::Ui, busy: bool) -> bool {
    ui.add_enabled(!busy, egui::Button::new(format!("{}  Browse", egui_phosphor::regular::FOLDER_OPEN))).clicked()
}
