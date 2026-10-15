//! Per-user data locations and user-supplied export presets.

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

pub fn presets_dir() -> PathBuf {
    data_dir().join("presets")
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
    fn missing_presets_dir_is_not_an_error() {
        let (p, w) = load_user_presets(Path::new("/definitely/not/here"));
        assert!(p.is_empty() && w.is_empty());
    }
}
