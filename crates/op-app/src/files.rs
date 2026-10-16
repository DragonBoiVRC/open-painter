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

const MAX_RECENT: usize = 10;

/// Where the list of recently used projects is kept.
pub fn recent_path() -> PathBuf {
    data_dir().join("recent.json")
}

/// `path` goes to the front of `list`; repeats are dropped and the list stays short.
pub fn push_recent(list: &mut Vec<PathBuf>, path: &Path) {
    list.retain(|p| p != path);
    list.insert(0, path.to_path_buf());
    list.truncate(MAX_RECENT);
}

pub fn load_recent() -> Vec<PathBuf> {
    let Ok(text) = std::fs::read_to_string(recent_path()) else { return Vec::new() };
    let mut list: Vec<PathBuf> = serde_json::from_str(&text).unwrap_or_default();
    list.truncate(MAX_RECENT);
    list
}

pub fn save_recent(list: &[PathBuf]) {
    if let Ok(text) = serde_json::to_string(list) {
        let _ = std::fs::create_dir_all(data_dir());
        let _ = std::fs::write(recent_path(), text);
    }
}

/// Where imported texture images are kept between sessions.
pub fn textures_dir() -> PathBuf {
    data_dir().join("textures")
}

pub fn environments_dir() -> PathBuf {
    data_dir().join("environments")
}

pub fn brushes_dir() -> PathBuf {
    data_dir().join("brushes")
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
    #[test]
    fn recent_projects_are_newest_first_without_repeats_and_bounded() {
        let mut list = Vec::new();
        for i in 0..14 {
            super::push_recent(&mut list, std::path::Path::new(&format!("/p/{i}.opproj")));
        }
        super::push_recent(&mut list, std::path::Path::new("/p/5.opproj"));
        assert_eq!(list.len(), 10);
        assert_eq!(list[0], std::path::PathBuf::from("/p/5.opproj"));
        assert_eq!(list.iter().filter(|p| p.ends_with("5.opproj")).count(), 1);
        assert_eq!(list[1], std::path::PathBuf::from("/p/13.opproj"));
    }

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

/// What a brush file holds.
#[derive(serde::Serialize, serde::Deserialize)]
struct BrushFile {
    name: String,
    brush: crate::ui::panels::BrushPreset,
}

/// The file name a brush is kept under: its name with anything odd replaced.
fn brush_file_name(name: &str) -> String {
    let stem: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect();
    format!("{}.json", stem.trim())
}

/// Save a user brush preset into `dir`; returns where it went.
pub fn save_user_brush(dir: &Path, name: &str, brush: &crate::ui::panels::BrushPreset) -> Result<PathBuf, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 64 {
        return Err("a brush name needs 1 to 64 characters".into());
    }
    let json = serde_json::to_string_pretty(&BrushFile { name: name.to_string(), brush: *brush }).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(brush_file_name(name));
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(path)
}

pub fn delete_user_brush(dir: &Path, name: &str) -> std::io::Result<()> {
    std::fs::remove_file(dir.join(brush_file_name(name.trim())))
}

/// Load the user's brush presets from `dir`; bad files become warnings.
pub fn load_user_brushes(dir: &Path) -> (Vec<(String, crate::ui::panels::BrushPreset)>, Vec<String>) {
    let (mut out, mut warnings) = (Vec::new(), Vec::new());
    for path in json_files(dir) {
        let file = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let parsed = read_limited(&path).and_then(|t| serde_json::from_str::<BrushFile>(&t).map_err(|e| e.to_string()));
        match parsed {
            Ok(b) if b.brush.is_sane() && !b.name.trim().is_empty() && b.name.chars().count() <= 64 => out.push((b.name.trim().to_string(), b.brush)),
            Ok(_) => warnings.push(format!("{file}: the brush settings are out of range")),
            Err(e) => warnings.push(format!("{file}: {e}")),
        }
    }
    (out, warnings)
}

#[cfg(test)]
mod brush_tests {
    use super::*;
    use crate::ui::panels::BrushPreset;

    #[test]
    fn a_saved_brush_loads_back_and_old_files_without_new_fields_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = BrushPreset { radius_pts: 55.0, hardness: 0.25, stroke_opacity: 0.4, lazy_radius: 30.0, ..BrushPreset::BASE };
        b.pressure_opacity = true;
        save_user_brush(dir.path(), "My / brush", &b).unwrap();
        let (got, warnings) = load_user_brushes(dir.path());
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(got, vec![("My / brush".to_string(), b)]);

        // A file written before stroke opacity, lazy mouse and pressure existed.
        std::fs::write(
            dir.path().join("Old.json"),
            r#"{"name":"Old","brush":{"radius_pts":20.0,"hardness":0.5,"opacity":1.0,"spacing":0.15,"shape":0,"angle":0.0,"follow_path":false,"angle_jitter":0.0,"size_jitter":0.0,"position_jitter":0.0,"opacity_jitter":0.0}}"#,
        )
        .unwrap();
        let (got, warnings) = load_user_brushes(dir.path());
        assert!(warnings.is_empty(), "{warnings:?}");
        let old = got.iter().find(|(n, _)| n == "Old").unwrap().1;
        assert_eq!((old.stroke_opacity, old.lazy_radius, old.pressure_size, old.pressure_opacity), (1.0, 0.0, true, false));
        delete_user_brush(dir.path(), "Old").unwrap();
        assert_eq!(load_user_brushes(dir.path()).0.len(), 1);
    }

    #[test]
    fn hostile_brush_files_are_warnings_not_brushes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.json"), r#"{"name":"Huge","brush":{"radius_pts":1e30,"hardness":0.5,"opacity":1.0,"spacing":0.15,"shape":0,"angle":0.0,"follow_path":false,"angle_jitter":0.0,"size_jitter":0.0,"position_jitter":0.0,"opacity_jitter":0.0}}"#).unwrap();
        std::fs::write(dir.path().join("b.json"), "not json").unwrap();
        std::fs::write(dir.path().join("c.json"), r#"{"name":"","brush":{"radius_pts":10.0,"hardness":0.5,"opacity":1.0,"spacing":0.15,"shape":9,"angle":0.0,"follow_path":false,"angle_jitter":0.0,"size_jitter":0.0,"position_jitter":0.0,"opacity_jitter":0.0}}"#).unwrap();
        let (got, warnings) = load_user_brushes(dir.path());
        assert!(got.is_empty());
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(save_user_brush(dir.path(), "   ", &BrushPreset::BASE).is_err());
    }
}
