# Third-party code included in this repository

Dependencies fetched by Cargo from crates.io keep their own licenses and are not copied here.
Two crates are **vendored** (copied in and modified) because their published versions target an older egui
than the one this application uses:

| Crate | Location | License | Changes |
|---|---|---|---|
| [`egui_dock`](https://github.com/Adanos020/egui_dock) 0.21.1 | `crates/egui_dock` | MIT (see `crates/egui_dock/LICENSE`) | Ported to egui 0.37 (one `Id` construction changed); `#![allow(deprecated)]` added; examples, dev-dependencies and upstream's repository-contribution/agent-instruction files removed |
| [`egui-phosphor`](https://github.com/amPerl/egui-phosphor) 0.14.0 | `crates/egui-phosphor` | MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`) | Dependency raised to egui 0.37; examples, dev-dependencies and tests removed |

The Phosphor icon font bundled with `egui-phosphor` is [Phosphor Icons](https://phosphoricons.com), MIT licensed.

`egui_dock`'s authors ask that AI-assisted *contributions to their repository* be disclosed and human-reviewed.
These vendored copies are modified for this project only; no changes have been (or should be) submitted upstream
without that review. If you want to help upstream support egui 0.37, the port is small and described above.
