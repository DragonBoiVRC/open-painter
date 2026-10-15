//! Developer tool: convert a Substance Painter `.spexp` preset and show what happened.
//!
//!   cargo run -p op-export --example spexp -- "path/to/preset.spexp" [--json]

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: spexp <file.spexp> [--json]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let name = Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match op_export::import_spexp(&bytes, &name) {
        Ok(imp) => {
            if args.iter().any(|a| a == "--json") {
                println!("{}", serde_json::to_string_pretty(&imp.preset).unwrap());
                return;
            }
            println!("preset '{}': {} map(s)", imp.preset.name, imp.preset.maps.len());
            for m in &imp.preset.maps {
                println!("  {}.{}  {:?} {:?} srgb={}", m.file, m.format.extension(), m.format, m.bit_depth, m.srgb);
                for (i, c) in m.channels.iter().enumerate() {
                    println!("      [{i}] {c:?}");
                }
            }
            println!("{} warning(s):", imp.warnings.len());
            for w in &imp.warnings {
                println!("  - {w}");
            }
        }
        Err(e) => {
            eprintln!("import failed: {e}");
            std::process::exit(1);
        }
    }
}
