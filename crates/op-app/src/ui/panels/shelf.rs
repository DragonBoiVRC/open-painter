//! "Shelf": reusable assets. Phase 1 shows the smart materials; the full browser comes next.

use crate::App;

impl App {
    pub(crate) fn shelf_panel(&mut self, ui: &mut egui::Ui) {
        self.smart_panel(ui);
    }
}
