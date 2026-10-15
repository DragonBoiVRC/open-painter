//! Project files (`.opproj`): a zip container holding the mesh and, for each texture set, its layer
//! document and raw paint textures. Lossless: textures are the GPU texel bytes, unchanged.
//!
//! Layout (version 3):
//!   manifest.json                          format id, version, mesh name, one entry per texture set
//!   mesh.bin                               `op_mesh::Mesh::to_bytes` (carries the set names and ranges)
//!   sets/<set>/textures/<node>_<kind>.raw  one entry per paint channel / mask of that set
//!   sets/<set>/bakes/<map>.raw             one entry per baked mesh map (u16 little endian)
//!
//! Versions 1 and 2 held a single set at the top level (`textures/`, `bakes/`); they still load.
//!
//! Saving is atomic (temp file + rename) so a crash or full disk while saving, or
//! autosaving, never destroys the previous good file. Loading treats the file as
//! untrusted: every size is checked against what the manifest implies before any
//! allocation, and entry names are derived by us, never taken from the file.

use op_core::{BakedMap, Document, MeshMap, Node, NodeId, NodeKind};
use op_mesh::Mesh;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const FORMAT_ID: &str = "open-painter-project";
/// Version 3 holds several texture sets; versions 1 (no baked maps) and 2 (one set) remain readable.
pub const FORMAT_VERSION: u32 = 3;
pub const EXTENSION: &str = "opproj";
const MAX_TEXTURE_SIZE: u32 = 8192;
const MAX_BAKE_BYTES: usize = 8192 * 8192 * 3 * 2;
const MAX_MANIFEST_BYTES: u64 = 64 << 20;
const MAX_MESH_BYTES: u64 = 1 << 30;
const MAX_SETS: usize = 4096;

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

/// One texture set: its layer stack, the paint inside it and the maps baked for it.
#[derive(Clone, Debug)]
pub struct SetData {
    pub texture_size: u32,
    pub document: Document,
    pub blobs: Vec<TextureBlob>,
    /// Baked mesh maps; each carries its own resolution.
    pub baked: HashMap<MeshMap, BakedMap>,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub mesh_name: String,
    pub mesh: Mesh,
    /// One entry per texture set of the mesh, in set order.
    pub sets: Vec<SetData>,
    /// The set that was being edited.
    pub active: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BlobEntry {
    node: NodeId,
    kind: BlobKind,
}

#[derive(Serialize, Deserialize)]
struct BakeEntry {
    map: MeshMap,
    size: u32,
    channels: u8,
}

#[derive(Serialize, Deserialize)]
struct SetEntry {
    texture_size: u32,
    document: Document,
    blobs: Vec<BlobEntry>,
    #[serde(default)]
    bakes: Vec<BakeEntry>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    mesh_name: String,
    #[serde(default)]
    active: usize,
    sets: Vec<SetEntry>,
}

/// Versions 1 and 2: a single set at the top level.
#[derive(Serialize, Deserialize)]
struct LegacyManifest {
    format: String,
    version: u32,
    texture_size: u32,
    mesh_name: String,
    document: Document,
    blobs: Vec<BlobEntry>,
    /// Absent in version 1 files.
    #[serde(default)]
    bakes: Vec<BakeEntry>,
}

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

/// Where a set's entries live in the zip: `Some(i)` is set `i` of a version 3 file, `None` the
/// top level of a version 1 or 2 file.
fn blob_path(set: Option<usize>, node: NodeId, kind: BlobKind) -> String {
    match set {
        Some(i) => format!("sets/{i}/textures/{node}_{}.raw", kind.name()),
        None => format!("textures/{node}_{}.raw", kind.name()),
    }
}

fn bake_path(set: Option<usize>, map: MeshMap) -> String {
    match set {
        Some(i) => format!("sets/{i}/bakes/{map:?}.raw"),
        None => format!("bakes/{map:?}.raw"),
    }
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

impl SetData {
    fn validate(&self) -> Result<(), ProjectError> {
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
        for (map, b) in &self.baked {
            check_size(b.size)?;
            if b.channels != map.channels() {
                return invalid(format!("{} must have {} channel(s), not {}", map.label(), map.channels(), b.channels));
            }
            if b.data.len() != b.texels() * b.channels as usize {
                return invalid(format!("baked {} has {} values, expected {}", map.label(), b.data.len(), b.texels() * b.channels as usize));
            }
        }
        Ok(())
    }
}

impl Project {
    /// Everything `save` relies on; also usable before attempting an expensive readback.
    pub fn validate(&self) -> Result<(), ProjectError> {
        if !self.mesh.sets_are_valid() {
            return invalid("the mesh's texture sets are inconsistent");
        }
        let want = self.mesh.set_ranges().len();
        if self.sets.len() != want || self.sets.len() > MAX_SETS {
            return invalid(format!("the mesh has {want} texture set(s) but the project holds {}", self.sets.len()));
        }
        if self.active >= self.sets.len() {
            return invalid("the active texture set does not exist");
        }
        for (i, s) in self.sets.iter().enumerate() {
            s.validate().map_err(|e| ProjectError::Invalid(format!("texture set {i}: {e}")))?;
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
        mesh_name: project.mesh_name.clone(),
        active: project.active,
        sets: project
            .sets
            .iter()
            .map(|s| SetEntry {
                texture_size: s.texture_size,
                document: s.document.clone(),
                blobs: s.blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
                bakes: baked_sorted(&s.baked).iter().map(|(m, b)| BakeEntry { map: *m, size: b.size, channels: b.channels }).collect(),
            })
            .collect(),
    };
    zip.start_file("manifest.json", deflate)?;
    zip.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    zip.start_file("mesh.bin", deflate)?;
    zip.write_all(&project.mesh.to_bytes())?;
    for (i, set) in project.sets.iter().enumerate() {
        for b in &set.blobs {
            zip.start_file(blob_path(Some(i), b.node, b.kind), deflate)?;
            zip.write_all(&b.bytes)?;
        }
        for (map, b) in baked_sorted(&set.baked) {
            zip.start_file(bake_path(Some(i), map), deflate)?;
            let bytes: Vec<u8> = b.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            zip.write_all(&bytes)?;
        }
    }
    let file = zip.finish()?;
    file.sync_all()?;
    Ok(())
}

/// Baked maps in a stable order so identical projects produce identical files.
fn baked_sorted(baked: &HashMap<MeshMap, BakedMap>) -> Vec<(MeshMap, &BakedMap)> {
    let mut v: Vec<_> = baked.iter().map(|(m, b)| (*m, b)).collect();
    v.sort_by_key(|(m, _)| *m);
    v
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

/// Read one set's textures and baked maps, checking everything against the manifest.
fn load_set(zip: &mut zip::ZipArchive<File>, which: Option<usize>, e: SetEntry) -> Result<SetData, ProjectError> {
    check_size(e.texture_size)?;
    let mut blobs = Vec::with_capacity(e.blobs.len());
    let mut seen = HashSet::new();
    for b in &e.blobs {
        check_blob_target(&e.document, b.node, b.kind)?;
        if !seen.insert((b.node, b.kind)) {
            return invalid(format!("duplicate {} texture for layer {}", b.kind.name(), b.node));
        }
        let want = expected_len(e.texture_size, b.kind);
        let mut entry = zip.by_name(&blob_path(which, b.node, b.kind))?;
        let bytes = read_limited(&mut entry, want as u64, "texture")?;
        if bytes.len() != want {
            return invalid(format!("{} texture of layer {} is truncated ({} of {want} bytes)", b.kind.name(), b.node, bytes.len()));
        }
        blobs.push(TextureBlob { node: b.node, kind: b.kind, bytes });
    }

    let mut baked = HashMap::new();
    for be in &e.bakes {
        check_size(be.size)?;
        if be.channels != be.map.channels() {
            return invalid(format!("{} must have {} channel(s)", be.map.label(), be.map.channels()));
        }
        if baked.contains_key(&be.map) {
            return invalid(format!("duplicate baked {}", be.map.label()));
        }
        let want = be.size as usize * be.size as usize * be.channels as usize * 2;
        if want > MAX_BAKE_BYTES {
            return invalid("baked map is implausibly large");
        }
        let mut entry = zip.by_name(&bake_path(which, be.map))?;
        let bytes = read_limited(&mut entry, want as u64, "baked map")?;
        if bytes.len() != want {
            return invalid(format!("baked {} is truncated ({} of {want} bytes)", be.map.label(), bytes.len()));
        }
        let data: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        baked.insert(be.map, BakedMap::new(be.size, be.channels, data).map_err(ProjectError::Invalid)?);
    }
    Ok(SetData { texture_size: e.texture_size, document: e.document, blobs, baked })
}

pub fn load(path: &Path) -> Result<Project, ProjectError> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;

    let manifest_bytes = {
        let mut entry = zip.by_name("manifest.json")?;
        read_limited(&mut entry, MAX_MANIFEST_BYTES, "manifest")?
    };
    let header: Header = serde_json::from_slice(&manifest_bytes)?;
    if header.format != FORMAT_ID {
        return invalid("this is not an open-painter project");
    }
    if header.version > FORMAT_VERSION {
        return invalid(format!(
            "project was saved by a newer version (format {}, this build reads up to {FORMAT_VERSION})",
            header.version
        ));
    }

    let mesh = {
        let mut entry = zip.by_name("mesh.bin")?;
        Mesh::from_bytes(&read_limited(&mut entry, MAX_MESH_BYTES, "mesh")?)?
    };

    let (mesh_name, entries, active, legacy) = if header.version >= 3 {
        let m: Manifest = serde_json::from_slice(&manifest_bytes)?;
        (m.mesh_name, m.sets, m.active, false)
    } else {
        let m: LegacyManifest = serde_json::from_slice(&manifest_bytes)?;
        let set = SetEntry { texture_size: m.texture_size, document: m.document, blobs: m.blobs, bakes: m.bakes };
        (m.mesh_name, vec![set], 0, true)
    };
    let want = mesh.set_ranges().len();
    if entries.len() != want || entries.len() > MAX_SETS {
        return invalid(format!("the mesh has {want} texture set(s) but the manifest lists {}", entries.len()));
    }
    if active >= entries.len() {
        return invalid("the active texture set does not exist");
    }
    let mut sets = Vec::with_capacity(entries.len());
    for (i, e) in entries.into_iter().enumerate() {
        sets.push(load_set(&mut zip, if legacy { None } else { Some(i) }, e)?);
    }
    Ok(Project { mesh_name, mesh, sets, active })
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
        doc.find_mut(paint).unwrap().mask = Some(Mask::new(false));
        let g = doc.add(None, None, "Group", NodeKind::Group { children: vec![] });
        doc.add(Some(g), None, "inner", NodeKind::Fill(FillParams { base: None, metallic: Some(0.5), roughness: None }));
        let pattern = |len: usize, seed: u8| (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect::<Vec<u8>>();
        let blobs = vec![
            TextureBlob { node: paint, kind: BlobKind::Base, bytes: pattern(expected_len(SIZE, BlobKind::Base), 1) },
            TextureBlob { node: paint, kind: BlobKind::Metal, bytes: pattern(expected_len(SIZE, BlobKind::Metal), 2) },
            TextureBlob { node: paint, kind: BlobKind::Rough, bytes: pattern(expected_len(SIZE, BlobKind::Rough), 3) },
            TextureBlob { node: paint, kind: BlobKind::Mask, bytes: pattern(expected_len(SIZE, BlobKind::Mask), 4) },
        ];
        let mut baked = HashMap::new();
        baked.insert(MeshMap::AmbientOcclusion, BakedMap::new(4, 1, (0..16).map(|i| i * 4000).collect()).unwrap());
        baked.insert(MeshMap::NormalBase, BakedMap::new(2, 3, (0..12).map(|i| 65535 - i * 5000).collect()).unwrap());
        Project {
            mesh_name: "sphere".into(),
            mesh: Mesh::uv_sphere(1.0, 12, 6),
            sets: vec![SetData { texture_size: SIZE, document: doc, blobs, baked }],
            active: 0,
        }
    }

    /// Two texture sets with different documents, sizes, paint and bakes.
    fn two_sets() -> Project {
        let mut p = sample();
        let mut m = Mesh::uv_sphere(1.0, 12, 6);
        let half = (m.indices.len() / 6 * 3) as u32;
        let total = m.indices.len() as u32;
        m.sets = vec![
            op_mesh::SetRange { name: "Body".into(), first_index: 0, index_count: half },
            op_mesh::SetRange { name: "Trim".into(), first_index: half, index_count: total - half },
        ];
        p.mesh = m;
        let mut doc = Document::with_default_material();
        let paint = doc.add(None, None, "Dents", NodeKind::Paint);
        let size = 4;
        let blobs = vec![
            TextureBlob { node: paint, kind: BlobKind::Base, bytes: vec![9; expected_len(size, BlobKind::Base)] },
            TextureBlob { node: paint, kind: BlobKind::Metal, bytes: vec![8; expected_len(size, BlobKind::Metal)] },
            TextureBlob { node: paint, kind: BlobKind::Rough, bytes: vec![7; expected_len(size, BlobKind::Rough)] },
        ];
        let mut baked = HashMap::new();
        baked.insert(MeshMap::Thickness, BakedMap::new(2, 1, vec![1, 2, 3, 4]).unwrap());
        p.sets.push(SetData { texture_size: size, document: doc, blobs, baked });
        p.active = 1;
        p
    }

    #[test]
    fn save_then_load_reproduces_everything_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.opproj");
        let p = sample();
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        let (qs, ps) = (&q.sets[0], &p.sets[0]);
        assert_eq!(qs.document, ps.document);
        assert_eq!((qs.texture_size, q.mesh_name.as_str()), (SIZE, "sphere"));
        assert_eq!(q.mesh.to_bytes(), p.mesh.to_bytes());
        assert_eq!(qs.blobs, ps.blobs);
        assert_eq!(qs.baked, ps.baked, "baked maps come back bit for bit, each at its own size");
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
        short.sets[0].blobs[0].bytes.pop();
        assert!(save(&path, &short).is_err(), "wrong blob length");
        let mut unknown = sample();
        unknown.sets[0].blobs[0].node = 9999;
        assert!(save(&path, &unknown).is_err(), "unknown layer");
        let mut wrong_kind = sample();
        let fill = wrong_kind.sets[0].document.nodes[0].id; // fill layer cannot own paint textures
        wrong_kind.sets[0].blobs[0].node = fill;
        assert!(save(&path, &wrong_kind).is_err(), "texture on fill layer");
        let mut dup = sample();
        let first = dup.sets[0].blobs[0].clone();
        dup.sets[0].blobs.push(first);
        assert!(save(&path, &dup).is_err(), "duplicate blob");
        let mut huge = sample();
        huge.sets[0].texture_size = 1 << 20;
        assert!(save(&path, &huge).is_err(), "absurd size");
        let mut wrong = sample();
        wrong.sets[0].baked.insert(MeshMap::Curvature, BakedMap::new(2, 3, vec![0; 12]).unwrap()); // curvature is scalar
        assert!(save(&path, &wrong).is_err(), "channel count must match the map kind");

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

    /// Build a version 2 (single set) zip by hand to exercise the loader's checks.
    fn handmade(path: &Path, edit: impl FnOnce(&mut LegacyManifest, &mut Vec<(String, Vec<u8>)>)) {
        let p = sample();
        let s = &p.sets[0];
        let mut manifest = LegacyManifest {
            format: FORMAT_ID.into(),
            version: 2,
            texture_size: s.texture_size,
            mesh_name: "m".into(),
            document: s.document.clone(),
            blobs: s.blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
            bakes: Vec::new(),
        };
        let mut entries: Vec<(String, Vec<u8>)> = vec![("mesh.bin".into(), p.mesh.to_bytes())];
        for b in &s.blobs {
            entries.push((blob_path(None, b.node, b.kind), b.bytes.clone()));
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
    fn version_1_files_without_bakes_still_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.opproj");
        handmade(&path, |m, _| m.version = 1);
        let p = load(&path).unwrap();
        assert!(p.sets[0].baked.is_empty());
    }

    #[test]
    fn loader_validates_baked_maps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.opproj");
        let entry = |map, size, channels| BakeEntry { map, size, channels };
        let ao_bytes = |n: usize| vec![7u8; n * 2];

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(None, MeshMap::AmbientOcclusion), ao_bytes(16)));
        });
        assert_eq!(load(&path).unwrap().sets[0].baked[&MeshMap::AmbientOcclusion].texels(), 16);

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 3)); // AO is scalar
            e.push((bake_path(None, MeshMap::AmbientOcclusion), ao_bytes(48)));
        });
        assert!(load(&path).is_err(), "wrong channel count");

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(None, MeshMap::AmbientOcclusion), ao_bytes(15)));
        });
        assert!(load(&path).unwrap_err().to_string().contains("truncated"));

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(None, MeshMap::AmbientOcclusion), ao_bytes(1 << 20))); // far larger than declared
        });
        assert!(load(&path).unwrap_err().to_string().contains("larger"), "zip-bomb guard");

        handmade(&path, |m, _| m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1)));
        assert!(load(&path).is_err(), "declared but missing entry");

        handmade(&path, |m, e| {
            for _ in 0..2 {
                m.bakes.push(entry(MeshMap::AmbientOcclusion, 2, 1));
            }
            e.push((bake_path(None, MeshMap::AmbientOcclusion), ao_bytes(4)));
        });
        assert!(load(&path).is_err(), "duplicate bake");

        handmade(&path, |m, _| m.bakes.push(entry(MeshMap::AmbientOcclusion, 1 << 20, 1)));
        assert!(load(&path).is_err(), "absurd size must fail before allocating");
    }

    #[test]
    fn several_sets_round_trip_and_keep_their_own_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two.opproj");
        let p = two_sets();
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        assert_eq!(q.active, 1);
        assert_eq!(q.mesh.sets, p.mesh.sets, "set names and ranges are stored with the mesh");
        assert_eq!(q.sets.len(), 2);
        for (a, b) in q.sets.iter().zip(&p.sets) {
            assert_eq!((a.texture_size, &a.document, &a.blobs, &a.baked), (b.texture_size, &b.document, &b.blobs, &b.baked));
        }
        // The two sets really are different (not both reading the first set's entries).
        assert_ne!(q.sets[0].texture_size, q.sets[1].texture_size);
        assert!(q.sets[0].baked.contains_key(&MeshMap::AmbientOcclusion) && q.sets[1].baked.contains_key(&MeshMap::Thickness));
    }

    #[test]
    fn set_count_and_active_set_must_match_the_mesh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.opproj");
        let mut missing = two_sets();
        missing.sets.pop();
        assert!(save(&path, &missing).is_err(), "a set per material is required");
        let mut extra = sample();
        extra.sets.push(extra.sets[0].clone());
        assert!(save(&path, &extra).is_err(), "no extra sets");
        let mut bad_active = two_sets();
        bad_active.active = 2;
        assert!(save(&path, &bad_active).is_err());
        let mut bad_inner = two_sets();
        bad_inner.sets[1].blobs[0].bytes.pop();
        let e = save(&path, &bad_inner).unwrap_err().to_string();
        assert!(e.contains("texture set 1"), "the error names the set: {e}");
        assert!(!path.exists());
    }

    #[test]
    fn a_v3_file_with_a_mismatched_manifest_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (good, bad) = (dir.path().join("g.opproj"), dir.path().join("b.opproj"));
        save(&good, &two_sets()).unwrap();
        // Rewrite the zip with the second set's entry dropped from the manifest.
        let mut zin = zip::ZipArchive::new(File::open(&good).unwrap()).unwrap();
        let mut manifest: Manifest = serde_json::from_slice(&read_limited(&mut zin.by_name("manifest.json").unwrap(), 1 << 20, "m").unwrap()).unwrap();
        manifest.sets.pop();
        let mut zout = zip::ZipWriter::new(File::create(&bad).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zout.start_file("manifest.json", opts).unwrap();
        zout.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        zout.start_file("mesh.bin", opts).unwrap();
        zout.write_all(&two_sets().mesh.to_bytes()).unwrap();
        zout.finish().unwrap();
        assert!(load(&bad).unwrap_err().to_string().contains("texture set"));
    }

    #[test]
    fn single_set_files_of_version_2_load_as_one_set() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.opproj");
        handmade(&path, |_, _| {});
        let p = load(&path).unwrap();
        assert_eq!((p.sets.len(), p.active), (1, 0));
        assert_eq!(p.sets[0].blobs.len(), 4);
    }

    #[test]
    fn autosave_never_targets_the_project_itself() {
        let dir = Path::new("/data");
        let saved = autosave_path(Some(Path::new("/work/foo.opproj")), dir);
        assert_eq!(saved, Path::new("/work/foo.opproj.autosave"));
        assert_eq!(autosave_path(None, dir), Path::new("/data/untitled.opproj.autosave"));
    }
}
