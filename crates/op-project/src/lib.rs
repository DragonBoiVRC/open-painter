//! Project files (`.opproj`): a zip container holding the layer document, the mesh and
//! the raw paint textures. Lossless: textures are the GPU texel bytes, unchanged.
//!
//! Layout:
//!   manifest.json                     format id, version, texture size, document, blob index
//!   mesh.bin                          `op_mesh::Mesh::to_bytes`
//!   textures/<node>_<kind>.raw        one entry per paint channel / mask
//!
//! Saving is atomic (temp file + rename) so a crash or full disk while saving, or
//! autosaving, never destroys the previous good file. Loading treats the file as
//! untrusted: every size is checked against what the manifest implies before any
//! allocation, and entry names are derived by us, never taken from the file.

use op_core::{Document, Node, NodeId, NodeKind};
use op_mesh::Mesh;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const FORMAT_ID: &str = "open-painter-project";
pub const FORMAT_VERSION: u32 = 1;
pub const EXTENSION: &str = "opproj";
const MAX_TEXTURE_SIZE: u32 = 8192;
const MAX_MANIFEST_BYTES: u64 = 64 << 20;
const MAX_MESH_BYTES: u64 = 1 << 30;

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a valid project file: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("invalid manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("invalid mesh: {0}")]
    Mesh(#[from] op_mesh::MeshError),
    #[error("{0}")]
    Invalid(String),
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, ProjectError> {
    Err(ProjectError::Invalid(msg.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlobKind {
    /// RGBA16F, premultiplied.
    Base,
    /// RG8: (value * coverage, coverage).
    Metal,
    Rough,
    /// R8 mask.
    Mask,
}

impl BlobKind {
    pub fn bytes_per_texel(self) -> usize {
        match self {
            BlobKind::Base => 8,
            BlobKind::Metal | BlobKind::Rough => 2,
            BlobKind::Mask => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            BlobKind::Base => "base",
            BlobKind::Metal => "metal",
            BlobKind::Rough => "rough",
            BlobKind::Mask => "mask",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureBlob {
    pub node: NodeId,
    pub kind: BlobKind,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub texture_size: u32,
    pub document: Document,
    pub mesh_name: String,
    pub mesh: Mesh,
    pub blobs: Vec<TextureBlob>,
}

#[derive(Serialize, Deserialize)]
struct BlobEntry {
    node: NodeId,
    kind: BlobKind,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    texture_size: u32,
    mesh_name: String,
    document: Document,
    blobs: Vec<BlobEntry>,
}

fn blob_path(node: NodeId, kind: BlobKind) -> String {
    format!("textures/{node}_{}.raw", kind.name())
}

fn expected_len(size: u32, kind: BlobKind) -> usize {
    size as usize * size as usize * kind.bytes_per_texel()
}

fn find_node(doc: &Document, id: NodeId) -> Option<&Node> {
    doc.find(id)
}

/// Check that a (node, kind) pair makes sense for this document.
fn check_blob_target(doc: &Document, node: NodeId, kind: BlobKind) -> Result<(), ProjectError> {
    let Some(n) = find_node(doc, node) else {
        return invalid(format!("texture refers to unknown layer {node}"));
    };
    let ok = match kind {
        BlobKind::Mask => n.mask.is_some(),
        _ => matches!(n.kind, NodeKind::Paint),
    };
    if ok { Ok(()) } else { invalid(format!("layer {node} cannot have a {} texture", kind.name())) }
}

fn check_size(size: u32) -> Result<(), ProjectError> {
    if (1..=MAX_TEXTURE_SIZE).contains(&size) { Ok(()) } else { invalid(format!("unsupported texture size {size}")) }
}

impl Project {
    /// Everything `save` relies on; also usable before attempting an expensive readback.
    pub fn validate(&self) -> Result<(), ProjectError> {
        check_size(self.texture_size)?;
        let mut seen = HashSet::new();
        for b in &self.blobs {
            check_blob_target(&self.document, b.node, b.kind)?;
            if !seen.insert((b.node, b.kind)) {
                return invalid(format!("duplicate {} texture for layer {}", b.kind.name(), b.node));
            }
            let want = expected_len(self.texture_size, b.kind);
            if b.bytes.len() != want {
                return invalid(format!("{} texture of layer {} has {} bytes, expected {want}", b.kind.name(), b.node, b.bytes.len()));
            }
        }
        Ok(())
    }
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(format!(".tmp{}", std::process::id()));
    path.with_file_name(name)
}

/// Write `project` to `path` atomically.
pub fn save(path: &Path, project: &Project) -> Result<(), ProjectError> {
    project.validate()?;
    let tmp = temp_sibling(path);
    let result = write_zip(&tmp, project).and_then(|()| std::fs::rename(&tmp, path).map_err(Into::into));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn write_zip(path: &Path, project: &Project) -> Result<(), ProjectError> {
    use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    // Fast deflate: paint textures are mostly empty and compress well even at level 1.
    let deflate = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(1));

    let manifest = Manifest {
        format: FORMAT_ID.into(),
        version: FORMAT_VERSION,
        texture_size: project.texture_size,
        mesh_name: project.mesh_name.clone(),
        document: project.document.clone(),
        blobs: project.blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
    };
    zip.start_file("manifest.json", deflate)?;
    zip.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    zip.start_file("mesh.bin", deflate)?;
    zip.write_all(&project.mesh.to_bytes())?;
    for b in &project.blobs {
        zip.start_file(blob_path(b.node, b.kind), deflate)?;
        zip.write_all(&b.bytes)?;
    }
    let file = zip.finish()?;
    file.sync_all()?;
    Ok(())
}

/// Read up to `limit` bytes from an entry; error if the entry is larger.
fn read_limited(entry: &mut impl Read, limit: u64, what: &str) -> Result<Vec<u8>, ProjectError> {
    let mut buf = Vec::new();
    entry.take(limit + 1).read_to_end(&mut buf)?;
    if buf.len() as u64 > limit {
        return invalid(format!("{what} is larger than the allowed {limit} bytes"));
    }
    Ok(buf)
}

pub fn load(path: &Path) -> Result<Project, ProjectError> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;

    let manifest: Manifest = {
        let mut entry = zip.by_name("manifest.json")?;
        serde_json::from_slice(&read_limited(&mut entry, MAX_MANIFEST_BYTES, "manifest")?)?
    };
    if manifest.format != FORMAT_ID {
        return invalid("this is not an open-painter project");
    }
    if manifest.version > FORMAT_VERSION {
        return invalid(format!(
            "project was saved by a newer version (format {}, this build reads up to {FORMAT_VERSION})",
            manifest.version
        ));
    }
    check_size(manifest.texture_size)?;

    let mesh = {
        let mut entry = zip.by_name("mesh.bin")?;
        Mesh::from_bytes(&read_limited(&mut entry, MAX_MESH_BYTES, "mesh")?)?
    };

    let mut blobs = Vec::with_capacity(manifest.blobs.len());
    let mut seen = HashSet::new();
    for b in &manifest.blobs {
        check_blob_target(&manifest.document, b.node, b.kind)?;
        if !seen.insert((b.node, b.kind)) {
            return invalid(format!("duplicate {} texture for layer {}", b.kind.name(), b.node));
        }
        let want = expected_len(manifest.texture_size, b.kind);
        let mut entry = zip.by_name(&blob_path(b.node, b.kind))?;
        let bytes = read_limited(&mut entry, want as u64, "texture")?;
        if bytes.len() != want {
            return invalid(format!("{} texture of layer {} is truncated ({} of {want} bytes)", b.kind.name(), b.node, bytes.len()));
        }
        blobs.push(TextureBlob { node: b.node, kind: b.kind, bytes });
    }

    Ok(Project {
        texture_size: manifest.texture_size,
        document: manifest.document,
        mesh_name: manifest.mesh_name,
        mesh,
        blobs,
    })
}

/// Where to autosave: next to a saved project, or in `fallback_dir` for an untitled one.
/// Never the project file itself, so autosave cannot overwrite the user's last explicit save.
pub fn autosave_path(project: Option<&Path>, fallback_dir: &Path) -> PathBuf {
    match project {
        Some(p) => {
            let mut name = p.file_name().map(|n| n.to_os_string()).unwrap_or_else(|| "untitled".into());
            name.push(".autosave");
            p.with_file_name(name)
        }
        None => fallback_dir.join(format!("untitled.{EXTENSION}.autosave")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{FillParams, Mask};

    const SIZE: u32 = 8;

    fn sample() -> Project {
        let mut doc = Document::with_default_material();
        let paint = doc.add(None, None, "Paint", NodeKind::Paint);
        doc.find_mut(paint).unwrap().mask = Some(Mask { enabled: true, init_white: false });
        let g = doc.add(None, None, "Group", NodeKind::Group { children: vec![] });
        doc.add(Some(g), None, "inner", NodeKind::Fill(FillParams { base: None, metallic: Some(0.5), roughness: None }));
        let pattern = |len: usize, seed: u8| (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect::<Vec<u8>>();
        let blobs = vec![
            TextureBlob { node: paint, kind: BlobKind::Base, bytes: pattern(expected_len(SIZE, BlobKind::Base), 1) },
            TextureBlob { node: paint, kind: BlobKind::Metal, bytes: pattern(expected_len(SIZE, BlobKind::Metal), 2) },
            TextureBlob { node: paint, kind: BlobKind::Rough, bytes: pattern(expected_len(SIZE, BlobKind::Rough), 3) },
            TextureBlob { node: paint, kind: BlobKind::Mask, bytes: pattern(expected_len(SIZE, BlobKind::Mask), 4) },
        ];
        Project { texture_size: SIZE, document: doc, mesh_name: "sphere".into(), mesh: Mesh::uv_sphere(1.0, 12, 6), blobs }
    }

    #[test]
    fn save_then_load_reproduces_everything_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.opproj");
        let p = sample();
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        assert_eq!(q.document, p.document);
        assert_eq!((q.texture_size, q.mesh_name.as_str()), (SIZE, "sphere"));
        assert_eq!(q.mesh.to_bytes(), p.mesh.to_bytes());
        assert_eq!(q.blobs, p.blobs);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(leftovers.len(), 1, "temp file left behind: {leftovers:?}");
    }

    #[test]
    fn invalid_projects_are_refused_without_touching_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keep.opproj");
        save(&path, &sample()).unwrap();
        let before = std::fs::read(&path).unwrap();

        let mut short = sample();
        short.blobs[0].bytes.pop();
        assert!(save(&path, &short).is_err(), "wrong blob length");
        let mut unknown = sample();
        unknown.blobs[0].node = 9999;
        assert!(save(&path, &unknown).is_err(), "unknown layer");
        let mut wrong_kind = sample();
        let fill = wrong_kind.document.nodes[0].id; // fill layer cannot own paint textures
        wrong_kind.blobs[0].node = fill;
        assert!(save(&path, &wrong_kind).is_err(), "texture on fill layer");
        let mut dup = sample();
        dup.blobs.push(dup.blobs[0].clone());
        assert!(save(&path, &dup).is_err(), "duplicate blob");
        let mut huge = sample();
        huge.texture_size = 1 << 20;
        assert!(save(&path, &huge).is_err(), "absurd size");

        assert_eq!(std::fs::read(&path).unwrap(), before, "previous file was modified");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn failed_write_cleans_up_and_keeps_the_old_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keep.opproj");
        save(&path, &sample()).unwrap();
        let before = std::fs::read(&path).unwrap();
        // Saving into a directory that does not exist fails at File::create.
        assert!(save(&dir.path().join("missing").join("x.opproj"), &sample()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn corrupt_and_foreign_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.opproj");
        save(&good, &sample()).unwrap();
        let bytes = std::fs::read(&good).unwrap();

        let truncated = dir.path().join("t.opproj");
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        assert!(load(&truncated).is_err(), "truncated zip");

        let garbage = dir.path().join("g.opproj");
        std::fs::write(&garbage, b"definitely not a zip file").unwrap();
        assert!(load(&garbage).is_err());
        assert!(load(&dir.path().join("nope.opproj")).is_err(), "missing file");
    }

    /// Build a zip by hand to exercise the loader's checks.
    fn handmade(path: &Path, edit: impl FnOnce(&mut Manifest, &mut Vec<(String, Vec<u8>)>)) {
        let p = sample();
        let mut manifest = Manifest {
            format: FORMAT_ID.into(),
            version: FORMAT_VERSION,
            texture_size: p.texture_size,
            mesh_name: "m".into(),
            document: p.document.clone(),
            blobs: p.blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
        };
        let mut entries: Vec<(String, Vec<u8>)> = vec![("mesh.bin".into(), p.mesh.to_bytes())];
        for b in &p.blobs {
            entries.push((blob_path(b.node, b.kind), b.bytes.clone()));
        }
        edit(&mut manifest, &mut entries);
        let mut zip = zip::ZipWriter::new(File::create(path).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file("manifest.json", opts).unwrap();
        zip.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        for (name, data) in entries {
            zip.start_file(name, opts).unwrap();
            zip.write_all(&data).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn loader_enforces_manifest_invariants() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("h.opproj");

        handmade(&path, |_, _| {});
        assert!(load(&path).is_ok(), "control: handmade file should load");

        handmade(&path, |m, _| m.version = FORMAT_VERSION + 1);
        let e = load(&path).unwrap_err().to_string();
        assert!(e.contains("newer version"), "{e}");

        handmade(&path, |m, _| m.format = "something-else".into());
        assert!(load(&path).is_err());

        handmade(&path, |m, _| m.texture_size = 0);
        assert!(load(&path).is_err(), "zero size");
        handmade(&path, |m, _| m.texture_size = 1 << 20);
        assert!(load(&path).is_err(), "absurd size must fail before allocating");

        // A blob entry bigger than the manifest implies must be rejected (zip-bomb guard).
        handmade(&path, |_, e| e[1].1.extend(std::iter::repeat_n(0u8, 1 << 20)));
        assert!(load(&path).unwrap_err().to_string().contains("larger"), "oversize blob");

        // Truncated blob.
        handmade(&path, |_, e| {
            e[1].1.pop();
        });
        assert!(load(&path).unwrap_err().to_string().contains("truncated"));

        // Missing blob entry.
        handmade(&path, |_, e| {
            e.remove(1);
        });
        assert!(load(&path).is_err());

        // Manifest names a layer that does not exist.
        handmade(&path, |m, _| m.blobs[0].node = 4242);
        assert!(load(&path).is_err());

        // Corrupt mesh.
        handmade(&path, |_, e| e[0].1.truncate(20));
        assert!(load(&path).is_err());
    }

    #[test]
    fn autosave_never_targets_the_project_itself() {
        let dir = Path::new("/data");
        let saved = autosave_path(Some(Path::new("/work/foo.opproj")), dir);
        assert_eq!(saved, Path::new("/work/foo.opproj.autosave"));
        assert_eq!(autosave_path(None, dir), Path::new("/data/untitled.opproj.autosave"));
    }
}
