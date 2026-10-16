//! Per-user data locations and user-supplied export presets.

use op_core::smart::SmartMaterial;
use op_core::Generator;
use op_export::ExportPreset;
use std::path::{Path, PathBuf};

/// Directory for autosaves and user presets. Platform conventions without extra crates.
pub fn data_dir() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from).filter(|p| !p.as_os_str().is_empty());
    let base = if cfg!(windows) {
        var("APPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library").join("Application Support"))
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|h| h.join(".local").join("share")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("open-painter")
}

pub fn layout_path() -> PathBuf {
    data_dir().join("layout.json")
}

/// Where imported texture images are kept between sessions.
pub fn textures_dir() -> PathBuf {
    data_dir().join("textures")
}

pub fn presets_dir() -> PathBuf {
    data_dir().join("presets")
}

pub fn smart_materials_dir() -> PathBuf {
    data_dir().join("smart-materials")
}

pub fn generators_dir() -> PathBuf {
    data_dir().join("generators")
}

/// Largest file we will read as a smart material or generator definition.
const MAX_DEFINITION_BYTES: u64 = 4 << 20;

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> =
        entries.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    paths.sort();
    paths
}

fn read_limited(path: &Path) -> Result<String, String> {
    let len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len > MAX_DEFINITION_BYTES {
        return Err(format!("file is {len} bytes; the limit is {MAX_DEFINITION_BYTES}"));
    }
    std::fs::read_to_string(path).map_err(|e| e.to_string())
}

/// Load the user's smart materials from `dir`; bad files become warnings.
pub fn load_smart_materials(dir: &Path) -> (Vec<SmartMaterial>, Vec<String>) {
    let (mut out, mut warnings) = (Vec::new(), Vec::new());
    for path in json_files(dir) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match read_limited(&path).and_then(|t| SmartMaterial::from_json(&t)) {
            Ok(m) => out.push(m),
            Err(e) => warnings.push(format!("{name}: {e}")),
        }
    }
    (out, warnings)
}

/// Load custom mask generators (graph files) from `dir`; bad files become warnings.
pub fn load_custom_generators(dir: &Path) -> (Vec<Generator>, Vec<String>) {
    let (mut out, mut warnings) = (Vec::new(), Vec::new());
    for path in json_files(dir) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match read_limited(&path).and_then(|t| op_proc::custom_generator_from_json(&t)) {
            Ok(g) => out.push(g),
            Err(e) => warnings.push(format!("{name}: {e}")),
        }
    }
    (out, warnings)
}

/// Load every `*.json` export preset in `dir`. Files that fail to parse or validate are
/// skipped and described in the returned warnings instead of aborting the whole load.
pub fn load_user_presets(dir: &Path) -> (Vec<ExportPreset>, Vec<String>) {
    let (mut presets, mut warnings) = (Vec::new(), Vec::new());
    let Ok(entries) = std::fs::read_dir(dir) else { return (presets, warnings) };
    let mut paths: Vec<PathBuf> =
        entries.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match std::fs::read_to_string(&path) {
            Err(e) => warnings.push(format!("{name}: {e}")),
            Ok(text) => match serde_json::from_str::<ExportPreset>(&text) {
                Err(e) => warnings.push(format!("{name}: {e}")),
                Ok(p) => match p.validate() {
                    Ok(()) => presets.push(p),
                    Err(e) => warnings.push(format!("{name}: {e}")),
                },
            },
        }
    }
    (presets, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_presets_load_and_bad_files_become_warnings() {
        let dir = tempfile::tempdir().unwrap();
        let good = op_export::builtin_presets().remove(0);
        std::fs::write(dir.path().join("a_good.json"), serde_json::to_string(&good).unwrap()).unwrap();
        std::fs::write(dir.path().join("b_broken.json"), "{ not json").unwrap();
        let mut escaping = good.clone();
        escaping.name = "escaping".into();
        escaping.maps[0].file = "../../etc/passwd".into();
        std::fs::write(dir.path().join("c_escape.json"), serde_json::to_string(&escaping).unwrap()).unwrap();
        std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();

        let (presets, warnings) = load_user_presets(dir.path());
        assert_eq!(presets, vec![good]);
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings.iter().any(|w| w.starts_with("b_broken.json")));
        assert!(warnings.iter().any(|w| w.starts_with("c_escape.json")));
    }

    #[test]
    fn user_smart_materials_load_and_hostile_files_become_warnings() {
        let dir = tempfile::tempdir().unwrap();
        let good = op_core::smart::builtin_smart_materials().remove(0);
        std::fs::write(dir.path().join("a.json"), good.to_json().unwrap()).unwrap();
        std::fs::write(dir.path().join("b.json"), "not json").unwrap();
        let mut bad = good.clone();
        bad.nodes[0].opacity = 9.0;
        std::fs::write(dir.path().join("c.json"), serde_json::to_string(&bad).unwrap()).unwrap();
        std::fs::write(dir.path().join("big.json"), vec![b' '; (MAX_DEFINITION_BYTES + 1) as usize]).unwrap();
        let (mats, warnings) = load_smart_materials(dir.path());
        assert_eq!(mats, vec![good]);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(warnings.iter().any(|w| w.starts_with("big.json") && w.contains("limit")));
        assert_eq!(load_smart_materials(Path::new("/nowhere")), (vec![], vec![]));
    }

    #[test]
    fn custom_generators_load_from_graph_files() {
        use op_core::graph::{GraphBuilder, Op};
        let dir = tempfile::tempdir().unwrap();
        let mut b = GraphBuilder::new();
        let u = b.add(Op::Uv { axis: 0 }, &[]);
        let graph = b.finish(u);
        std::fs::write(
            dir.path().join("ramp.json"),
            format!(r#"{{"name":"ramp","graph":{}}}"#, serde_json::to_string(&graph).unwrap()),
        )
        .unwrap();
        std::fs::write(dir.path().join("broken.json"), r#"{"name":"x","graph":{"nodes":[],"output":0}}"#).unwrap();
        let (gens, warnings) = load_custom_generators(dir.path());
        assert_eq!(gens.len(), 1);
        assert_eq!(op_proc::label(&gens[0].kind), "ramp");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("broken.json"));
    }

    #[test]
    fn missing_presets_dir_is_not_an_error() {
        let (p, w) = load_user_presets(Path::new("/definitely/not/here"));
        assert!(p.is_empty() && w.is_empty());
    }
}
