//! Project files (`.opproj`): a zip container holding the mesh and, for each texture set, its layer
//! document and raw paint textures. Lossless: textures are the GPU texel bytes, unchanged.
//!
//! Layout (version 4):
//!   manifest.json                          format id, version, mesh name, one entry per texture set
//!   mesh.bin                               `op_mesh::Mesh::to_bytes` (carries the set names and ranges)
//!   sets/<set>/tiles/<tile>/textures/<node>_<kind>.raw  one entry per paint channel / mask of that tile
//!   sets/<set>/tiles/<tile>/bakes/<map>.raw             one entry per baked mesh map (u16 little endian)
//!
//! Version 3 had one tile per set (`sets/<set>/textures/`); versions 1 and 2 held a single set at the top
//! level (`textures/`, `bakes/`). All still load.
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
/// Version 4 stores each texture set's paint per UV tile; versions 1 (no baked maps), 2 (one set) and 3
/// (several sets, one tile each) remain readable.
pub const FORMAT_VERSION: u32 = 4;
pub const EXTENSION: &str = "opproj";
const MAX_TEXTURE_SIZE: u32 = 8192;
const MAX_BAKE_BYTES: usize = 8192 * 8192 * 3 * 2;
const MAX_MANIFEST_BYTES: u64 = 64 << 20;
const MAX_MESH_BYTES: u64 = 1 << 30;
const MAX_SETS: usize = 4096;
const MAX_IMAGES: usize = 4096;
const MAX_IMAGE_BYTES: u64 = 256 << 20;

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
    /// RG8 like the others: (height * coverage, coverage); 0.5 is flat.
    Height,
    /// R8 mask.
    Mask,
}

impl BlobKind {
    pub fn bytes_per_texel(self) -> usize {
        match self {
            BlobKind::Base => 8,
            BlobKind::Metal | BlobKind::Rough | BlobKind::Height => 2,
            BlobKind::Mask => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            BlobKind::Base => "base",
            BlobKind::Metal => "metal",
            BlobKind::Rough => "rough",
            BlobKind::Height => "height",
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

/// The paint and baked maps of one UV tile of a texture set.
#[derive(Clone, Debug, Default)]
pub struct TileData {
    pub blobs: Vec<TextureBlob>,
    /// Baked mesh maps; each carries its own resolution.
    pub baked: HashMap<MeshMap, BakedMap>,
}

/// One texture set: its layer stack and, for each UV tile, the paint inside it. Without UV tiles a
/// set has exactly one tile.
#[derive(Clone, Debug)]
pub struct SetData {
    pub texture_size: u32,
    pub document: Document,
    pub tiles: Vec<TileData>,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub mesh_name: String,
    pub mesh: Mesh,
    /// One entry per texture set of the mesh, in set order.
    pub sets: Vec<SetData>,
    /// The set that was being edited.
    pub active: usize,
    /// Whether UV tiles are in use: then each set has a tile for every UDIM tile its UVs occupy.
    pub udim: bool,
    /// Texture sets hidden in the 3D view (a view setting, remembered with the project).
    pub hidden_sets: Vec<usize>,
    /// The texture images the layers use, as the original files, so the project is self-contained.
    pub images: Vec<ImageData>,
}

/// One embedded texture image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageData {
    /// The library id: the content hash of `bytes`.
    pub id: String,
    /// The original file name, shown in the shelf.
    pub name: String,
    /// The file as it was imported (PNG, JPEG, ...).
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ImageEntry {
    id: String,
    name: String,
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
struct TileEntry {
    blobs: Vec<BlobEntry>,
    #[serde(default)]
    bakes: Vec<BakeEntry>,
}

#[derive(Serialize, Deserialize)]
struct SetEntry {
    texture_size: u32,
    document: Document,
    tiles: Vec<TileEntry>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    mesh_name: String,
    #[serde(default)]
    active: usize,
    #[serde(default)]
    udim: bool,
    #[serde(default)]
    hidden: Vec<usize>,
    #[serde(default)]
    images: Vec<ImageEntry>,
    sets: Vec<SetEntry>,
}

/// Version 3: one tile per set, stored at the set level.
#[derive(Serialize, Deserialize)]
struct SetEntryV3 {
    texture_size: u32,
    document: Document,
    blobs: Vec<BlobEntry>,
    #[serde(default)]
    bakes: Vec<BakeEntry>,
}

#[derive(Serialize, Deserialize)]
struct ManifestV3 {
    format: String,
    version: u32,
    mesh_name: String,
    #[serde(default)]
    active: usize,
    sets: Vec<SetEntryV3>,
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

/// Where a tile's entries live in the zip.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// Versions 1 and 2: the top level.
    Legacy,
    /// Version 3: `sets/<set>/`.
    Set(usize),
    /// Version 4: `sets/<set>/tiles/<tile>/`.
    Tile(usize, usize),
}

impl Place {
    fn dir(self) -> String {
        match self {
            Place::Legacy => String::new(),
            Place::Set(s) => format!("sets/{s}/"),
            Place::Tile(s, t) => format!("sets/{s}/tiles/{t}/"),
        }
    }
}

fn blob_path(at: Place, node: NodeId, kind: BlobKind) -> String {
    format!("{}textures/{node}_{}.raw", at.dir(), kind.name())
}

fn image_path(id: &str) -> String {
    format!("images/{id}.bin")
}

fn bake_path(at: Place, map: MeshMap) -> String {
    format!("{}bakes/{map:?}.raw", at.dir())
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
        if self.tiles.is_empty() {
            return invalid("a texture set has no tiles");
        }
        for (t, tile) in self.tiles.iter().enumerate() {
            let mut seen = HashSet::new();
            for b in &tile.blobs {
                check_blob_target(&self.document, b.node, b.kind)?;
                if !seen.insert((b.node, b.kind)) {
                    return invalid(format!("duplicate {} texture for layer {}", b.kind.name(), b.node));
                }
                let want = expected_len(self.texture_size, b.kind);
                if b.bytes.len() != want {
                    return invalid(format!("{} texture of layer {} has {} bytes, expected {want}", b.kind.name(), b.node, b.bytes.len()));
                }
            }
            for (map, b) in &tile.baked {
                check_size(b.size)?;
                if b.channels != map.channels() {
                    return invalid(format!("{} must have {} channel(s), not {}", map.label(), map.channels(), b.channels));
                }
                if b.data.len() != b.texels() * b.channels as usize {
                    return invalid(format!("baked {} has {} values, expected {}", map.label(), b.data.len(), b.texels() * b.channels as usize));
                }
            }
            let _ = t;
        }
        Ok(())
    }
}

/// How many tiles set `set` of `mesh` has: one per UDIM tile its UVs occupy with UV tiles, else one.
fn expected_tiles(mesh: &Mesh, set: usize, udim: bool) -> usize {
    if udim { mesh.set_tiles(set).len() } else { 1 }
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
        if self.images.len() > MAX_IMAGES {
            return invalid("too many texture images");
        }
        let mut seen = HashSet::new();
        for img in &self.images {
            if !op_core::is_valid_texture_id(&img.id) || !seen.insert(img.id.as_str()) {
                return invalid(format!("texture image '{}' has an unusable or repeated id", img.name));
            }
        }
        for (i, s) in self.sets.iter().enumerate() {
            s.validate().map_err(|e| ProjectError::Invalid(format!("texture set {i}: {e}")))?;
            let tiles = expected_tiles(&self.mesh, i, self.udim);
            if s.tiles.len() != tiles {
                return invalid(format!("texture set {i} has {} tile(s) but its UVs occupy {tiles}", s.tiles.len()));
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
        mesh_name: project.mesh_name.clone(),
        active: project.active,
        udim: project.udim,
        hidden: project.hidden_sets.clone(),
        images: project.images.iter().map(|i| ImageEntry { id: i.id.clone(), name: i.name.clone() }).collect(),
        sets: project
            .sets
            .iter()
            .map(|s| SetEntry {
                texture_size: s.texture_size,
                document: s.document.clone(),
                tiles: s
                    .tiles
                    .iter()
                    .map(|t| TileEntry {
                        blobs: t.blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
                        bakes: baked_sorted(&t.baked).iter().map(|(m, b)| BakeEntry { map: *m, size: b.size, channels: b.channels }).collect(),
                    })
                    .collect(),
            })
            .collect(),
    };
    zip.start_file("manifest.json", deflate)?;
    zip.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    zip.start_file("mesh.bin", deflate)?;
    zip.write_all(&project.mesh.to_bytes())?;
    for img in &project.images {
        zip.start_file(image_path(&img.id), deflate)?;
        zip.write_all(&img.bytes)?;
    }
    for (i, set) in project.sets.iter().enumerate() {
        for (t, tile) in set.tiles.iter().enumerate() {
            let at = Place::Tile(i, t);
            for b in &tile.blobs {
                zip.start_file(blob_path(at, b.node, b.kind), deflate)?;
                zip.write_all(&b.bytes)?;
            }
            for (map, b) in baked_sorted(&tile.baked) {
                zip.start_file(bake_path(at, map), deflate)?;
                let bytes: Vec<u8> = b.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                zip.write_all(&bytes)?;
            }
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

/// Read one tile's textures and baked maps, checking everything against the manifest.
fn load_tile(zip: &mut zip::ZipArchive<File>, at: Place, doc: &Document, size: u32, blobs: &[BlobEntry], bakes: &[BakeEntry]) -> Result<TileData, ProjectError> {
    let mut out_blobs = Vec::with_capacity(blobs.len());
    let mut seen = HashSet::new();
    for b in blobs {
        check_blob_target(doc, b.node, b.kind)?;
        if !seen.insert((b.node, b.kind)) {
            return invalid(format!("duplicate {} texture for layer {}", b.kind.name(), b.node));
        }
        let want = expected_len(size, b.kind);
        let mut entry = zip.by_name(&blob_path(at, b.node, b.kind))?;
        let bytes = read_limited(&mut entry, want as u64, "texture")?;
        if bytes.len() != want {
            return invalid(format!("{} texture of layer {} is truncated ({} of {want} bytes)", b.kind.name(), b.node, bytes.len()));
        }
        out_blobs.push(TextureBlob { node: b.node, kind: b.kind, bytes });
    }

    let mut baked = HashMap::new();
    for be in bakes {
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
        let mut entry = zip.by_name(&bake_path(at, be.map))?;
        let bytes = read_limited(&mut entry, want as u64, "baked map")?;
        if bytes.len() != want {
            return invalid(format!("baked {} is truncated ({} of {want} bytes)", be.map.label(), bytes.len()));
        }
        let data: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        baked.insert(be.map, BakedMap::new(be.size, be.channels, data).map_err(ProjectError::Invalid)?);
    }
    Ok(TileData { blobs: out_blobs, baked })
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

    /// A set as the manifest describes it, before its entries are read.
    struct Pending {
        texture_size: u32,
        document: Document,
        tiles: Vec<(Place, Vec<BlobEntry>, Vec<BakeEntry>)>,
    }
    let mut hidden_sets = Vec::new();
    let mut image_entries = Vec::new();
    let (mesh_name, entries, active, udim): (String, Vec<Pending>, usize, bool) = if header.version >= 4 {
        let m: Manifest = serde_json::from_slice(&manifest_bytes)?;
        let sets = m
            .sets
            .into_iter()
            .enumerate()
            .map(|(i, s)| Pending {
                texture_size: s.texture_size,
                document: s.document,
                tiles: s.tiles.into_iter().enumerate().map(|(t, e)| (Place::Tile(i, t), e.blobs, e.bakes)).collect(),
            })
            .collect();
        hidden_sets = m.hidden;
        image_entries = m.images;
        (m.mesh_name, sets, m.active, m.udim)
    } else if header.version == 3 {
        let m: ManifestV3 = serde_json::from_slice(&manifest_bytes)?;
        let sets = m
            .sets
            .into_iter()
            .enumerate()
            .map(|(i, s)| Pending { texture_size: s.texture_size, document: s.document, tiles: vec![(Place::Set(i), s.blobs, s.bakes)] })
            .collect();
        (m.mesh_name, sets, m.active, false)
    } else {
        let m: LegacyManifest = serde_json::from_slice(&manifest_bytes)?;
        let one = Pending { texture_size: m.texture_size, document: m.document, tiles: vec![(Place::Legacy, m.blobs, m.bakes)] };
        (m.mesh_name, vec![one], 0, false)
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
        check_size(e.texture_size)?;
        let tiles = expected_tiles(&mesh, i, udim);
        if e.tiles.len() != tiles {
            return invalid(format!("texture set {i} lists {} tile(s) but its UVs occupy {tiles}", e.tiles.len()));
        }
        let mut data = Vec::with_capacity(tiles);
        for (at, blobs, bakes) in &e.tiles {
            data.push(load_tile(&mut zip, *at, &e.document, e.texture_size, blobs, bakes)?);
        }
        sets.push(SetData { texture_size: e.texture_size, document: e.document, tiles: data });
    }
    let mut images = Vec::with_capacity(image_entries.len());
    if image_entries.len() > MAX_IMAGES {
        return invalid("too many texture images");
    }
    for e in image_entries {
        if !op_core::is_valid_texture_id(&e.id) {
            return invalid("a texture image has an unusable id");
        }
        let mut entry = zip.by_name(&image_path(&e.id))?;
        let bytes = read_limited(&mut entry, MAX_IMAGE_BYTES, "texture image")?;
        images.push(ImageData { id: e.id, name: e.name.chars().take(200).collect(), bytes });
    }
    hidden_sets.retain(|s| *s < sets.len());
    hidden_sets.sort_unstable();
    hidden_sets.dedup();
    Ok(Project { mesh_name, mesh, sets, active, udim, hidden_sets, images })
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
        doc.add(Some(g), None, "inner", NodeKind::Fill(FillParams { metallic: Some(0.5), ..Default::default() }));
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
            sets: vec![SetData { texture_size: SIZE, document: doc, tiles: vec![TileData { blobs, baked }] }],
            active: 0,
            udim: false,
            hidden_sets: Vec::new(),
            images: Vec::new(),
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
        p.sets.push(SetData { texture_size: size, document: doc, tiles: vec![TileData { blobs, baked }] });
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
        assert_eq!(qs.tiles[0].blobs, ps.tiles[0].blobs);
        assert_eq!(qs.tiles[0].baked, ps.tiles[0].baked, "baked maps come back bit for bit, each at its own size");
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
        short.sets[0].tiles[0].blobs[0].bytes.pop();
        assert!(save(&path, &short).is_err(), "wrong blob length");
        let mut unknown = sample();
        unknown.sets[0].tiles[0].blobs[0].node = 9999;
        assert!(save(&path, &unknown).is_err(), "unknown layer");
        let mut wrong_kind = sample();
        let fill = wrong_kind.sets[0].document.nodes[0].id; // fill layer cannot own paint textures
        wrong_kind.sets[0].tiles[0].blobs[0].node = fill;
        assert!(save(&path, &wrong_kind).is_err(), "texture on fill layer");
        let mut dup = sample();
        let first = dup.sets[0].tiles[0].blobs[0].clone();
        dup.sets[0].tiles[0].blobs.push(first);
        assert!(save(&path, &dup).is_err(), "duplicate blob");
        let mut huge = sample();
        huge.sets[0].texture_size = 1 << 20;
        assert!(save(&path, &huge).is_err(), "absurd size");
        let mut wrong = sample();
        wrong.sets[0].tiles[0].baked.insert(MeshMap::Curvature, BakedMap::new(2, 3, vec![0; 12]).unwrap()); // curvature is scalar
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
            blobs: s.tiles[0].blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
            bakes: Vec::new(),
        };
        let mut entries: Vec<(String, Vec<u8>)> = vec![("mesh.bin".into(), p.mesh.to_bytes())];
        for b in &s.tiles[0].blobs {
            entries.push((blob_path(Place::Legacy, b.node, b.kind), b.bytes.clone()));
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
        assert!(p.sets[0].tiles[0].baked.is_empty());
    }

    #[test]
    fn loader_validates_baked_maps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.opproj");
        let entry = |map, size, channels| BakeEntry { map, size, channels };
        let ao_bytes = |n: usize| vec![7u8; n * 2];

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(Place::Legacy, MeshMap::AmbientOcclusion), ao_bytes(16)));
        });
        assert_eq!(load(&path).unwrap().sets[0].tiles[0].baked[&MeshMap::AmbientOcclusion].texels(), 16);

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 3)); // AO is scalar
            e.push((bake_path(Place::Legacy, MeshMap::AmbientOcclusion), ao_bytes(48)));
        });
        assert!(load(&path).is_err(), "wrong channel count");

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(Place::Legacy, MeshMap::AmbientOcclusion), ao_bytes(15)));
        });
        assert!(load(&path).unwrap_err().to_string().contains("truncated"));

        handmade(&path, |m, e| {
            m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1));
            e.push((bake_path(Place::Legacy, MeshMap::AmbientOcclusion), ao_bytes(1 << 20))); // far larger than declared
        });
        assert!(load(&path).unwrap_err().to_string().contains("larger"), "zip-bomb guard");

        handmade(&path, |m, _| m.bakes.push(entry(MeshMap::AmbientOcclusion, 4, 1)));
        assert!(load(&path).is_err(), "declared but missing entry");

        handmade(&path, |m, e| {
            for _ in 0..2 {
                m.bakes.push(entry(MeshMap::AmbientOcclusion, 2, 1));
            }
            e.push((bake_path(Place::Legacy, MeshMap::AmbientOcclusion), ao_bytes(4)));
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
            assert_eq!((a.texture_size, &a.document, &a.tiles[0].blobs, &a.tiles[0].baked), (b.texture_size, &b.document, &b.tiles[0].blobs, &b.tiles[0].baked));
        }
        // The two sets really are different (not both reading the first set's entries).
        assert_ne!(q.sets[0].texture_size, q.sets[1].texture_size);
        assert!(q.sets[0].tiles[0].baked.contains_key(&MeshMap::AmbientOcclusion) && q.sets[1].tiles[0].baked.contains_key(&MeshMap::Thickness));
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
        bad_inner.sets[1].tiles[0].blobs[0].bytes.pop();
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
        assert_eq!(p.sets[0].tiles[0].blobs.len(), 4);
    }

    /// A mesh with one set whose UVs fill tiles 1001 and 1002.
    fn two_tile_project() -> Project {
        let mut mesh = Mesh::uv_sphere(1.0, 12, 6);
        let half = mesh.indices.len() / 6 * 3;
        let mut moved = HashMap::new();
        for k in half..mesh.indices.len() {
            let old = mesh.indices[k];
            let new = *moved.entry(old).or_insert_with(|| {
                let mut v = mesh.vertices[old as usize];
                v.uv[0] += 1.0;
                mesh.vertices.push(v);
                (mesh.vertices.len() - 1) as u32
            });
            mesh.indices[k] = new;
        }
        mesh.partition_tiles();
        let mut doc = Document::with_default_material();
        let paint = doc.add(None, None, "Paint", NodeKind::Paint);
        let tile = |seed: u8| {
            let blobs = [BlobKind::Base, BlobKind::Metal, BlobKind::Rough, BlobKind::Height]
                .map(|kind| TextureBlob { node: paint, kind, bytes: vec![seed; expected_len(SIZE, kind)] })
                .to_vec();
            let mut baked = HashMap::new();
            baked.insert(MeshMap::Thickness, BakedMap::new(2, 1, vec![seed as u16; 4]).unwrap());
            TileData { blobs, baked }
        };
        Project {
            mesh_name: "tiled".into(),
            mesh,
            sets: vec![SetData { texture_size: SIZE, document: doc, tiles: vec![tile(10), tile(20)] }],
            active: 0,
            udim: true,
            hidden_sets: Vec::new(),
            images: Vec::new(),
        }
    }

    #[test]
    fn uv_tiles_round_trip_each_with_its_own_paint_and_bakes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("udim.opproj");
        let p = two_tile_project();
        assert_eq!(p.mesh.set_tiles(0).len(), 2);
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        assert!(q.udim);
        assert_eq!(q.sets[0].tiles.len(), 2);
        for t in 0..2 {
            assert_eq!(q.sets[0].tiles[t].blobs, p.sets[0].tiles[t].blobs, "tile {t}");
            assert_eq!(q.sets[0].tiles[t].baked, p.sets[0].tiles[t].baked, "tile {t}");
        }
        assert_ne!(q.sets[0].tiles[0].blobs[0].bytes, q.sets[0].tiles[1].blobs[0].bytes, "the tiles hold different paint");
    }

    #[test]
    fn embedded_texture_images_round_trip_and_hostile_ids_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("img.opproj");
        let mut p = sample();
        let id = "0123456789abcdef0123456789abcdef".to_string();
        p.images = vec![ImageData { id: id.clone(), name: "wood_BaseColor.png".into(), bytes: (0..=255u8).cycle().take(5000).collect() }];
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        assert_eq!(q.images, p.images);
        // Ids that are not hashes could name paths: refused when saving...
        for bad in ["../../etc/passwd", "short", "0123456789ABCDEF0123456789ABCDEF"] {
            let mut evil = sample();
            evil.images = vec![ImageData { id: bad.into(), name: "x".into(), bytes: vec![1] }];
            assert!(save(&path, &evil).is_err(), "{bad}");
        }
        let mut twice = sample();
        twice.images = vec![p.images[0].clone(), p.images[0].clone()];
        assert!(save(&path, &twice).is_err(), "the same id twice");
        // ...and when loading a manifest that was edited by hand.
        save(&path, &p).unwrap();
        let mut zin = zip::ZipArchive::new(File::open(&path).unwrap()).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&read_limited(&mut zin.by_name("manifest.json").unwrap(), 1 << 22, "m").unwrap()).unwrap();
        value["images"][0]["id"] = "../../x".into();
        let out = dir.path().join("evil.opproj");
        let mut zout = zip::ZipWriter::new(File::create(&out).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zout.start_file("manifest.json", opts).unwrap();
        zout.write_all(&serde_json::to_vec(&value).unwrap()).unwrap();
        zout.start_file("mesh.bin", opts).unwrap();
        zout.write_all(&p.mesh.to_bytes()).unwrap();
        zout.finish().unwrap();
        assert!(load(&out).is_err(), "a manifest naming a path as an image id");
        // A project from before textures has none.
        assert!(load_without_hidden_field(&path).images.len() == 1, "images are kept when only `hidden` is absent");
    }

    #[test]
    fn hidden_sets_are_remembered_and_stale_entries_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("h.opproj");
        let mut p = two_sets();
        p.hidden_sets = vec![1];
        save(&path, &p).unwrap();
        assert_eq!(load(&path).unwrap().hidden_sets, [1]);
        // A hand-edited list pointing at sets that do not exist is cleaned up on load.
        p.hidden_sets = vec![1, 7, 1, 0];
        save(&path, &p).unwrap();
        assert_eq!(load(&path).unwrap().hidden_sets, [0, 1]);
        assert!(load_without_hidden_field(&path).hidden_sets.is_empty(), "older files have no such field");
    }

    /// Rewrite a saved project's manifest without the `hidden` field, as an older build wrote it.
    fn load_without_hidden_field(path: &Path) -> Project {
        let mut zin = zip::ZipArchive::new(File::open(path).unwrap()).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&read_limited(&mut zin.by_name("manifest.json").unwrap(), 1 << 22, "m").unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("hidden");
        let out = path.with_extension("old");
        let mut zout = zip::ZipWriter::new(File::create(&out).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zout.start_file("manifest.json", opts).unwrap();
        zout.write_all(&serde_json::to_vec(&value).unwrap()).unwrap();
        for i in 0..zin.len() {
            let mut f = zin.by_index(i).unwrap();
            let name = f.name().unwrap().to_string();
            if name == "manifest.json" {
                continue;
            }
            let mut data = Vec::new();
            f.read_to_end(&mut data).unwrap();
            zout.start_file(name, opts).unwrap();
            zout.write_all(&data).unwrap();
        }
        zout.finish().unwrap();
        load(&out).unwrap()
    }

    #[test]
    fn the_tile_count_must_match_the_uvs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.opproj");
        let mut one = two_tile_project();
        one.sets[0].tiles.pop();
        assert!(save(&path, &one).unwrap_err().to_string().contains("tile"), "a missing tile");
        let mut extra = sample();
        let copy = extra.sets[0].tiles[0].clone();
        extra.sets[0].tiles.push(copy);
        assert!(save(&path, &extra).is_err(), "no extra tiles without UV tiles");
        let mut none = sample();
        none.sets[0].tiles.clear();
        assert!(save(&path, &none).is_err());
        let mut off = two_tile_project();
        off.udim = false; // two tiles stored but UV tiles not in use
        assert!(save(&path, &off).is_err());
    }

    /// A version 3 file keeps its paint at the set level; it loads as a single tile.
    #[test]
    fn version_3_files_load_as_one_tile_per_set() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v3.opproj");
        let p = sample();
        let s = &p.sets[0];
        let manifest = ManifestV3 {
            format: FORMAT_ID.into(),
            version: 3,
            mesh_name: "m".into(),
            active: 0,
            sets: vec![SetEntryV3 {
                texture_size: s.texture_size,
                document: s.document.clone(),
                blobs: s.tiles[0].blobs.iter().map(|b| BlobEntry { node: b.node, kind: b.kind }).collect(),
                bakes: Vec::new(),
            }],
        };
        let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file("manifest.json", opts).unwrap();
        zip.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        zip.start_file("mesh.bin", opts).unwrap();
        zip.write_all(&p.mesh.to_bytes()).unwrap();
        for b in &s.tiles[0].blobs {
            zip.start_file(blob_path(Place::Set(0), b.node, b.kind), opts).unwrap();
            zip.write_all(&b.bytes).unwrap();
        }
        zip.finish().unwrap();
        let q = load(&path).unwrap();
        assert!(!q.udim);
        assert_eq!(q.sets[0].tiles.len(), 1);
        assert_eq!(q.sets[0].tiles[0].blobs, s.tiles[0].blobs);
    }

    #[test]
    fn autosave_never_targets_the_project_itself() {
        let dir = Path::new("/data");
        let saved = autosave_path(Some(Path::new("/work/foo.opproj")), dir);
        assert_eq!(saved, Path::new("/work/foo.opproj.autosave"));
        assert_eq!(autosave_path(None, dir), Path::new("/data/untitled.opproj.autosave"));
    }
}
