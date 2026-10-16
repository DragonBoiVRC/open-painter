//! Texture export: turn the composited channels of a texture set into image files
//! according to an export preset. Pure CPU, no GPU or UI.

mod import;
mod preset;
mod spexp;
mod template;
mod uv;

pub use import::{import_spexp, ImportError, ImportedPreset};
pub use preset::{builtin_presets, BitDepth, ChannelSrc, ColorMap, Comp, ExportPreset, FileFormat, MapSpec};
pub use spexp::{decode_spexp, MapId, SpexpChannel, SpexpError, SpexpMap, SpexpPreset};
pub use template::{expand_template, variables as template_variables, Vars, KNOWN_VARS};
pub use uv::uv_coverage;

use op_core::{BakedMap, MeshMap, FALLBACK_BASE, FALLBACK_METALLIC, FALLBACK_ROUGHNESS};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("invalid preset: {0}")]
    Preset(String),
    #[error("invalid texture data: {0}")]
    Data(String),
    #[error("could not write {path}: {source}")]
    Write { path: PathBuf, source: image::ImageError },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Final per-texel channel values of one texture set, linear, fully resolved
/// (uncovered texels already replaced by the fallback material).
#[derive(Clone, Debug)]
pub struct TextureSetData {
    pub size: u32,
    pub base: Vec<[f32; 3]>,
    pub metallic: Vec<f32>,
    pub roughness: Vec<f32>,
    /// Baked mesh maps, each exactly `size` x `size`.
    pub baked: HashMap<MeshMap, BakedMap>,
    /// Painted height, 0..1 with 0.5 flat. Empty when the document has no height channel data.
    pub height: Vec<f32>,
    /// Painted emissive colour, linear 0..1. Empty when the document gives off no light.
    pub emissive: Vec<[f32; 3]>,
    /// Painted opacity, 0..1. Empty when the document is fully opaque.
    pub opacity: Vec<f32>,
    /// Normal maps of fill layers, composited: unit xyz and coverage per texel. Empty when no layer has one.
    pub layer_normal: Vec<[f32; 4]>,
    /// Tangent-space normal derived from the height and the baked normal. `None` until
    /// [`TextureSetData::compute_normal`] ran and there was something to derive it from.
    pub normal: Option<Vec<[f32; 3]>>,
}

impl TextureSetData {
    /// Build from the compositor's premultiplied output:
    /// `base_rgba16f` is RGBA half floats (8 bytes/texel); `metal_rg8` and `rough_rg8`
    /// are (value * coverage, coverage) bytes. Uncovered parts show the fallback
    /// material, exactly as in the viewport.
    pub fn from_composite(size: u32, base_rgba16f: &[u8], metal_rg8: &[u8], rough_rg8: &[u8]) -> Result<Self, ExportError> {
        let n = (size as usize) * (size as usize);
        if size == 0 || base_rgba16f.len() != n * 8 || metal_rg8.len() != n * 2 || rough_rg8.len() != n * 2 {
            return Err(ExportError::Data(format!(
                "buffer sizes do not match a {size}x{size} texture set ({} / {} / {} bytes)",
                base_rgba16f.len(),
                metal_rg8.len(),
                rough_rg8.len()
            )));
        }
        let mut base = Vec::with_capacity(n);
        let half = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        for t in base_rgba16f.chunks_exact(8) {
            let a = half(&t[6..8]).clamp(0.0, 1.0);
            let c = |i: usize| (half(&t[i * 2..i * 2 + 2]) + FALLBACK_BASE[i] * (1.0 - a)).clamp(0.0, 1.0);
            base.push([c(0), c(1), c(2)]);
        }
        let resolve = |rg: &[u8], fallback: f32| -> Vec<f32> {
            rg.chunks_exact(2)
                .map(|p| (p[0] as f32 / 255.0 + fallback * (1.0 - p[1] as f32 / 255.0)).clamp(0.0, 1.0))
                .collect()
        };
        Ok(Self {
            size,
            base,
            metallic: resolve(metal_rg8, FALLBACK_METALLIC),
            roughness: resolve(rough_rg8, FALLBACK_ROUGHNESS),
            baked: HashMap::new(),
            height: Vec::new(),
            emissive: Vec::new(),
            opacity: Vec::new(),
            layer_normal: Vec::new(),
            normal: None,
        })
    }

    /// Attach the composited emissive channel: RGBA half floats (8 bytes per texel), premultiplied.
    /// Where nothing glows the colour is black.
    pub fn set_emissive(&mut self, rgba16f: &[u8]) -> Result<(), ExportError> {
        let n = (self.size as usize).pow(2);
        if rgba16f.len() != n * 8 {
            return Err(ExportError::Data(format!("emissive has {} bytes, expected {}", rgba16f.len(), n * 8)));
        }
        let half = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        let all: Vec<[f32; 3]> = rgba16f
            .chunks_exact(8)
            .map(|t| [0, 1, 2].map(|c| {
                let v = half(&t[c * 2..c * 2 + 2]);
                if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
            }))
            .collect();
        // Keep nothing for a document that glows nowhere, so exporters can skip the map.
        self.emissive = if all.iter().any(|c| c.iter().any(|v| *v > 0.0)) { all } else { Vec::new() };
        Ok(())
    }

    /// Attach the composited opacity channel: `(opacity * coverage, coverage)` bytes. Uncovered texels
    /// are opaque (1). A document that is opaque everywhere keeps nothing, so exporters can skip the map.
    pub fn set_opacity(&mut self, rg8: &[u8]) -> Result<(), ExportError> {
        let n = (self.size as usize).pow(2);
        if rg8.len() != n * 2 {
            return Err(ExportError::Data(format!("opacity has {} bytes, expected {}", rg8.len(), n * 2)));
        }
        let all: Vec<f32> = rg8.chunks_exact(2).map(|p| (p[0] as f32 / 255.0 + (1.0 - p[1] as f32 / 255.0)).clamp(0.0, 1.0)).collect();
        self.opacity = if all.iter().any(|v| *v < 0.999) { all } else { Vec::new() };
        Ok(())
    }

    /// Attach the composited fill-layer normals: RGBA8 with the packed normal in rgb and the coverage in alpha.
    pub fn set_layer_normal(&mut self, rgba8: &[u8]) -> Result<(), ExportError> {
        let n = (self.size as usize).pow(2);
        if rgba8.len() != n * 4 {
            return Err(ExportError::Data(format!("layer normal has {} bytes, expected {}", rgba8.len(), n * 4)));
        }
        self.layer_normal = rgba8
            .chunks_exact(4)
            .map(|p| {
                let c = op_core::normal::unpack([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
                [c[0], c[1], c[2], p[3] as f32 / 255.0]
            })
            .collect();
        Ok(())
    }

    /// Attach the composited height channel: `(height * coverage, coverage)` bytes. Uncovered texels
    /// are flat (0.5).
    pub fn set_height(&mut self, height_rg8: &[u8]) -> Result<(), ExportError> {
        let n = (self.size as usize).pow(2);
        if height_rg8.len() != n * 2 {
            return Err(ExportError::Data(format!("height has {} bytes, expected {}", height_rg8.len(), n * 2)));
        }
        self.height = height_rg8
            .chunks_exact(2)
            .map(|p| (p[0] as f32 / 255.0 + 0.5 * (1.0 - p[1] as f32 / 255.0)).clamp(0.0, 1.0))
            .collect();
        Ok(())
    }

    /// Derive the normal map from the height (if any was painted) and the baked normal (if baked).
    /// `strength` is the document's height strength. With neither, no normal is produced and the
    /// `Normal` source falls back to the baked map or a flat default.
    pub fn compute_normal(&mut self, strength: f32) {
        let painted = self.height.iter().any(|h| (*h - 0.5).abs() > 1e-4);
        let layered = self.layer_normal.iter().any(|l| l[3] > 0.0);
        let baked = self.baked.get(&MeshMap::NormalBase);
        if !painted && !layered && baked.is_none() {
            self.normal = None;
            return;
        }
        let n = (self.size as usize).pow(2);
        let flat;
        let height: &[f32] = if self.height.len() == n {
            &self.height
        } else {
            flat = vec![0.5; n];
            &flat
        };
        let base: Option<Vec<[f32; 3]>> = baked.map(|b| (0..n).map(|i| op_core::normal::unpack(b.texel(i))).collect());
        let layer = (layered && self.layer_normal.len() == n).then_some(self.layer_normal.as_slice());
        self.normal = Some(op_core::normal::height_to_normal(height, self.size, strength, base.as_deref(), layer));
    }

    /// Attach a baked map. Rejected if its size does not match the texture set.
    pub fn set_baked(&mut self, map: MeshMap, baked: BakedMap) -> Result<(), ExportError> {
        if baked.size != self.size {
            return Err(ExportError::Data(format!(
                "{} was baked at {}x{} but the texture set is {}x{}",
                map.label(),
                baked.size,
                baked.size,
                self.size,
                self.size
            )));
        }
        self.baked.insert(map, baked);
        Ok(())
    }

    /// Bleed colors outward from covered texels into the empty space around UV islands,
    /// so bilinear filtering and mip-mapping in other applications don't pull in
    /// background color at seams. `iterations` is the padding width in texels.
    pub fn dilate(&mut self, covered: &[bool], iterations: u32) {
        let s = self.size as i64;
        assert_eq!(covered.len(), (s * s) as usize, "coverage mask size mismatch");
        let mut covered = covered.to_vec();
        for _ in 0..iterations {
            let mut grown = covered.clone();
            let mut any = false;
            for y in 0..s {
                for x in 0..s {
                    let i = (y * s + x) as usize;
                    if covered[i] {
                        continue;
                    }
                    let mut nbrs = [0usize; 8];
                    let mut n = 0;
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let (nx, ny) = (x + dx, y + dy);
                            if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= s || ny >= s {
                                continue;
                            }
                            let j = (ny * s + nx) as usize;
                            if covered[j] {
                                nbrs[n] = j;
                                n += 1;
                            }
                        }
                    }
                    if n > 0 {
                        // Only previously-covered texels are read, so writing in place is safe.
                        let nf = n as f32;
                        let avg3 = |v: &[[f32; 3]]| {
                            let mut a = [0.0f32; 3];
                            for &j in &nbrs[..n] {
                                for c in 0..3 {
                                    a[c] += v[j][c];
                                }
                            }
                            [a[0] / nf, a[1] / nf, a[2] / nf]
                        };
                        let avg1 = |v: &[f32]| nbrs[..n].iter().map(|&j| v[j]).sum::<f32>() / nf;
                        self.base[i] = avg3(&self.base);
                        self.metallic[i] = avg1(&self.metallic);
                        self.roughness[i] = avg1(&self.roughness);
                        if self.height.len() == self.base.len() {
                            self.height[i] = avg1(&self.height);
                        }
                        if self.emissive.len() == self.base.len() {
                            self.emissive[i] = avg3(&self.emissive);
                        }
                        if self.opacity.len() == self.base.len() {
                            self.opacity[i] = avg1(&self.opacity);
                        }
                        if let Some(nm) = &mut self.normal {
                            let a = avg3(nm);
                            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
                            nm[i] = [a[0] / len, a[1] / len, a[2] / len];
                        }
                        for map in self.baked.values_mut() {
                            let mut a = [0.0f32; 3];
                            for &j in &nbrs[..n] {
                                let t = map.texel(j);
                                for c in 0..3 {
                                    a[c] += t[c];
                                }
                            }
                            map.set_texel(i, [a[0] / nf, a[1] / nf, a[2] / nf]);
                        }
                        grown[i] = true;
                        any = true;
                    }
                }
            }
            covered = grown;
            if !any {
                break;
            }
        }
    }
}

pub struct ExportContext<'a> {
    pub project: &'a str,
    pub texture_set: &'a str,
    pub mesh: &'a str,
    pub out_dir: &'a Path,
    /// The UDIM number of the tile being exported, when the set has UV tiles; fills `$udim`.
    pub udim: Option<u32>,
}

/// What an export wrote, and anything the user should know about how it was produced.
#[derive(Debug, Default)]
pub struct ExportOutcome {
    pub files: Vec<PathBuf>,
    /// For example "Ambient occlusion was not baked: used the neutral default."
    pub notes: Vec<String>,
}

fn srgb_encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn comp_of(c: [f32; 3], comp: Comp) -> f32 {
    match comp.index() {
        Some(i) => c[i],
        None => luma(c),
    }
}

/// Value of one source at texel `i`. Missing baked maps fall back to their neutral default
/// and are recorded in `missing`.
fn sample(src: ChannelSrc, i: usize, data: &TextureSetData, encode: bool, missing: &mut BTreeSet<MeshMap>) -> f32 {
    let enc = |v: f32| if encode { srgb_encode(v) } else { v };
    let mut baked = |map: MeshMap| -> [f32; 3] {
        match data.baked.get(&map) {
            Some(t) => t.texel(i),
            None => {
                missing.insert(map);
                map.default_value()
            }
        }
    };
    match src {
        ChannelSrc::Color { map, comp } => {
            let (b, m) = (data.base[i], data.metallic[i]);
            let c = match map {
                ColorMap::Base => b,
                ColorMap::Diffuse => [b[0] * (1.0 - m), b[1] * (1.0 - m), b[2] * (1.0 - m)],
                ColorMap::Specular => [0, 1, 2].map(|k| 0.04 + (b[k] - 0.04) * m),
                ColorMap::Emissive => data.emissive.get(i).copied().unwrap_or([0.0; 3]),
            };
            enc(comp_of(c, comp))
        }
        ChannelSrc::Metallic => data.metallic[i],
        ChannelSrc::Roughness => data.roughness[i],
        ChannelSrc::Smoothness => 1.0 - data.roughness[i],
        ChannelSrc::Mesh { map, comp } => comp_of(baked(map), if map.is_scalar() { Comp::R } else { comp }),
        ChannelSrc::Height => if data.height.len() == data.base.len() { data.height[i] } else { 0.5 },
        ChannelSrc::Opacity => data.opacity.get(i).copied().unwrap_or(1.0),
        ChannelSrc::Normal { directx, comp } => {
            let mut n = match &data.normal {
                Some(nm) => op_core::normal::pack(nm[i]),
                None => baked(MeshMap::NormalBase),
            };
            if directx {
                n[1] = 1.0 - n[1];
            }
            comp_of(n, comp)
        }
        ChannelSrc::AoMixed => baked(MeshMap::AmbientOcclusion)[0],
        ChannelSrc::Constant(v) => v,
    }
}

fn write_map(
    path: &Path,
    spec: &MapSpec,
    data: &TextureSetData,
    missing: &mut BTreeSet<MeshMap>,
) -> Result<(), ExportError> {
    use image::{ColorType, ImageFormat};
    let n = data.size as usize * data.size as usize;
    let ch = spec.channels.len();
    let encode = spec.srgb && spec.format != FileFormat::Exr;
    let format = match spec.format {
        FileFormat::Png => ImageFormat::Png,
        FileFormat::Tga => ImageFormat::Tga,
        FileFormat::Exr => ImageFormat::OpenExr,
    };
    let wrap = |source| ExportError::Write { path: path.to_path_buf(), source };
    // sRGB applies per source: only color-like sources are encoded.
    let mut value = |i: usize, c: usize, linear_only: bool| {
        let src = spec.channels[c];
        sample(src, i, data, encode && src.is_color() && !linear_only, missing)
    };

    match spec.bit_depth {
        BitDepth::U8 => {
            let mut buf = Vec::with_capacity(n * ch);
            for i in 0..n {
                for c in 0..ch {
                    buf.push((value(i, c, false).clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            let color = match ch {
                1 => ColorType::L8,
                3 => ColorType::Rgb8,
                _ => ColorType::Rgba8,
            };
            image::save_buffer_with_format(path, &buf, data.size, data.size, color, format).map_err(wrap)
        }
        BitDepth::U16 => {
            let mut buf: Vec<u16> = Vec::with_capacity(n * ch);
            for i in 0..n {
                for c in 0..ch {
                    buf.push((value(i, c, false).clamp(0.0, 1.0) * 65535.0).round() as u16);
                }
            }
            let color = match ch {
                1 => ColorType::L16,
                3 => ColorType::Rgb16,
                _ => ColorType::Rgba16,
            };
            // image expects native-endian bytes for 16-bit buffers.
            image::save_buffer_with_format(path, bytemuck::cast_slice(&buf), data.size, data.size, color, format)
                .map_err(wrap)
        }
        BitDepth::F32 => {
            // OpenEXR output is RGB(A); single channels are written as gray RGB.
            let out_ch = if ch == 4 { 4 } else { 3 };
            let mut buf: Vec<f32> = Vec::with_capacity(n * out_ch);
            for i in 0..n {
                for c in 0..out_ch {
                    buf.push(value(i, if ch == 1 { 0 } else { c }, true));
                }
            }
            let color = if out_ch == 4 { ColorType::Rgba32F } else { ColorType::Rgb32F };
            image::save_buffer_with_format(path, bytemuck::cast_slice(&buf), data.size, data.size, color, format)
                .map_err(wrap)
        }
    }
}

/// Write every map of `preset` into `ctx.out_dir`.
pub fn export(preset: &ExportPreset, data: &TextureSetData, ctx: &ExportContext) -> Result<ExportOutcome, ExportError> {
    preset.validate()?;
    let n = data.size as usize * data.size as usize;
    if data.base.len() != n || data.metallic.len() != n || data.roughness.len() != n {
        return Err(ExportError::Data("channel lengths do not match size".into()));
    }
    // Expand every file name first so a bad template fails before anything is written.
    let vars = Vars { project: ctx.project, texture_set: ctx.texture_set, mesh: ctx.mesh, udim: ctx.udim };
    let mut names = Vec::new();
    for spec in &preset.maps {
        let stem = expand_template(&spec.file, &vars).map_err(ExportError::Preset)?;
        names.push(format!("{stem}.{}", spec.format.extension()));
    }
    let mut seen = BTreeSet::new();
    for name in &names {
        if !seen.insert(name.to_lowercase()) {
            return Err(ExportError::Preset(format!("two maps would both be written to '{name}'")));
        }
    }
    std::fs::create_dir_all(ctx.out_dir)?;
    let mut outcome = ExportOutcome::default();
    let mut missing = BTreeSet::new();
    for (spec, name) in preset.maps.iter().zip(&names) {
        // A map made only of emissive light or opacity is pointless when nothing glows or is
        // see-through: leave it out.
        if (data.emissive.is_empty() && spec.channels.iter().all(|c| c.is_emissive())) || (data.opacity.is_empty() && spec.channels.iter().all(|c| c.is_opacity())) {
            continue;
        }
        let path = ctx.out_dir.join(name);
        write_map(&path, spec, data, &mut missing)?;
        outcome.files.push(path);
    }
    for map in missing {
        outcome.notes.push(format!("{} has not been baked: the neutral default value was used.", map.label()));
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn data(size: u32, base: [f32; 3], metallic: f32, roughness: f32) -> TextureSetData {
        let n = (size * size) as usize;
        TextureSetData {
            size,
            base: vec![base; n],
            metallic: vec![metallic; n],
            roughness: vec![roughness; n],
            baked: HashMap::new(),
            height: Vec::new(),
            emissive: Vec::new(),
            opacity: Vec::new(),
            layer_normal: Vec::new(),
            normal: None,
        }
    }

    fn ctx<'a>(dir: &'a Path) -> ExportContext<'a> {
        ExportContext { project: "proj", texture_set: "set", mesh: "mesh", out_dir: dir, udim: None }
    }

    fn preset(name: &str) -> ExportPreset {
        builtin_presets().into_iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no preset {name}"))
    }

    fn f16_bytes(v: [f32; 4]) -> Vec<u8> {
        v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()
    }

    #[test]
    fn from_composite_resolves_premultiplied_coverage_against_the_fallback() {
        // Half-covered red, quarter-covered metal 0.5, uncovered roughness.
        let base = f16_bytes([0.5, 0.0, 0.0, 0.5]);
        let metal = [(0.125 * 255.0f32).round() as u8, (0.25 * 255.0f32).round() as u8];
        let d = TextureSetData::from_composite(1, &base, &metal, &[0, 0]).unwrap();
        assert!((d.base[0][0] - (0.5 + FALLBACK_BASE[0] * 0.5)).abs() < 1e-3);
        assert!((d.base[0][1] - FALLBACK_BASE[1] * 0.5).abs() < 1e-3);
        assert!((d.metallic[0] - (0.125 + FALLBACK_METALLIC * 0.75)).abs() < 2e-3);
        assert_eq!(d.roughness[0], FALLBACK_ROUGHNESS, "uncovered texel shows the fallback");
    }

    #[test]
    fn from_composite_rejects_wrong_buffer_sizes() {
        assert!(TextureSetData::from_composite(2, &[0; 8], &[0; 8], &[0; 8]).is_err());
        assert!(TextureSetData::from_composite(0, &[], &[], &[]).is_err());
    }

    #[test]
    fn png8_basecolor_is_srgb_encoded_and_data_maps_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let d = data(4, [0.2, 0.5, 0.05], 0.3, 0.8);
        let out = export(&preset("PBR Metallic Roughness"), &d, &ctx(dir.path())).unwrap();
        assert_eq!(out.files.len(), 3);
        assert!(out.notes.is_empty(), "{:?}", out.notes);
        let base = image::open(dir.path().join("proj_set_BaseColor.png")).unwrap().to_rgb8();
        for (got, lin) in base.get_pixel(1, 1).0.iter().zip([0.2f32, 0.5, 0.05]) {
            assert_eq!(*got, (srgb_encode(lin) * 255.0).round() as u8);
        }
        let rough = image::open(dir.path().join("proj_set_Roughness.png")).unwrap().to_luma8();
        assert_eq!(rough.get_pixel(2, 3).0[0], (0.8f32 * 255.0).round() as u8, "roughness stays linear");
        let metal = image::open(dir.path().join("proj_set_Metallic.png")).unwrap().to_luma8();
        assert_eq!(metal.get_pixel(0, 0).0[0], (0.3f32 * 255.0).round() as u8);
    }

    #[test]
    fn the_emissive_map_is_written_only_when_something_glows() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = data(2, [0.5; 3], 0.0, 0.5);
        let plain = export(&preset("PBR Metallic Roughness"), &d, &ctx(dir.path())).unwrap();
        assert_eq!(plain.files.len(), 3);
        assert!(!dir.path().join("proj_set_Emissive.png").exists());

        // One texel glows orange (half floats, premultiplied, alpha ignored).
        let mut bytes = Vec::new();
        for t in 0..4 {
            let rgba = if t == 3 { [0.8, 0.4, 0.0, 1.0] } else { [0.0; 4] };
            bytes.extend(f16_bytes(rgba));
        }
        d.set_emissive(&bytes).unwrap();
        let glowing = export(&preset("PBR Metallic Roughness"), &d, &ctx(dir.path())).unwrap();
        assert_eq!(glowing.files.len(), 4);
        let e = image::open(dir.path().join("proj_set_Emissive.png")).unwrap().to_rgb8();
        assert_eq!(e.get_pixel(1, 1).0, [0.8f32, 0.4, 0.0].map(|v| (srgb_encode(v) * 255.0).round() as u8), "emissive is colour: sRGB encoded");
        assert_eq!(e.get_pixel(0, 0).0, [0, 0, 0]);
        assert!(d.set_emissive(&[0; 7]).is_err());
    }

    #[test]
    fn unity_alpha_carries_the_painted_opacity() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = data(2, [0.5; 3], 0.0, 0.5);
        export(&preset("Unity Standard (Metallic Smoothness)"), &d, &ctx(dir.path())).unwrap();
        let opaque = image::open(dir.path().join("proj_set_AlbedoTransparency.png")).unwrap().to_rgba8();
        assert_eq!(opaque.get_pixel(0, 0).0[3], 255, "nothing painted: fully opaque");

        // One texel half see-through (value * coverage, coverage) as the compositor stores it.
        let half = (0.5f32 * 255.0).round() as u8;
        d.set_opacity(&[255, 255, 255, 255, 255, 255, half, 255]).unwrap();
        export(&preset("Unity Standard (Metallic Smoothness)"), &d, &ctx(dir.path())).unwrap();
        let img = image::open(dir.path().join("proj_set_AlbedoTransparency.png")).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(0, 0).0[3], 255);
        assert_eq!(img.get_pixel(1, 1).0[3], half, "alpha is the opacity, linear");
        // Opacity-only maps are skipped while the document is solid.
        assert!(d.set_opacity(&[255; 8]).is_ok() && d.opacity.is_empty());
        assert!(d.set_opacity(&[0; 3]).is_err());
    }

    #[test]
    fn packed_orm_uses_baked_ao_when_present_and_reports_when_not() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = data(2, [0.5; 3], 0.25, 0.75);
        let out = export(&preset("glTF Metallic-Roughness"), &d, &ctx(dir.path())).unwrap();
        assert_eq!(out.notes.len(), 1, "AO was not baked, the user must be told: {:?}", out.notes);
        let mr = image::open(dir.path().join("proj_set_MetallicRoughness.png")).unwrap().to_rgb8();
        // glTF: R = occlusion (white default), G = roughness, B = metalness.
        assert_eq!(mr.get_pixel(0, 0).0, [255, (0.75f32 * 255.0).round() as u8, (0.25f32 * 255.0).round() as u8]);

        d.set_baked(MeshMap::AmbientOcclusion, BakedMap::filled(2, 1, [0.5; 3])).unwrap();
        let out = export(&preset("glTF Metallic-Roughness"), &d, &ctx(dir.path())).unwrap();
        assert!(out.notes.is_empty());
        let mr = image::open(dir.path().join("proj_set_MetallicRoughness.png")).unwrap().to_rgb8();
        assert_eq!(mr.get_pixel(1, 1).0[0], 128, "baked AO lands in the red channel");
    }

    #[test]
    fn normal_maps_follow_the_requested_convention() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = data(2, [0.5; 3], 0.0, 0.5);
        d.set_baked(MeshMap::NormalBase, BakedMap::filled(2, 3, [0.25, 0.75, 1.0])).unwrap();
        export(&preset("Unreal Engine (ORM packed)"), &d, &ctx(dir.path())).unwrap(); // DirectX
        let dx = image::open(dir.path().join("proj_set_Normal.png")).unwrap().to_rgb8().get_pixel(0, 0).0;
        assert_eq!(dx, [64, 64, 255], "DirectX flips green");
        export(&preset("Unity Standard (Metallic Smoothness)"), &d, &ctx(dir.path())).unwrap(); // OpenGL
        let gl = image::open(dir.path().join("proj_set_Normal.png")).unwrap().to_rgb8().get_pixel(0, 0).0;
        assert_eq!(gl, [64, 191, 255], "OpenGL keeps green");
    }

    #[test]
    fn unity_metallic_map_carries_smoothness_in_alpha() {
        let dir = tempfile::tempdir().unwrap();
        export(&preset("Unity Standard (Metallic Smoothness)"), &data(2, [0.5; 3], 0.4, 0.7), &ctx(dir.path())).unwrap();
        let img = image::open(dir.path().join("proj_set_MetallicSmoothness.png")).unwrap().to_rgba8();
        let p = img.get_pixel(1, 0).0;
        assert_eq!(p[0], (0.4f32 * 255.0).round() as u8, "R = metallic");
        assert_eq!(p[3], ((1.0f32 - 0.7) * 255.0).round() as u8, "A = smoothness = 1 - roughness");
    }

    #[test]
    fn derived_diffuse_and_specular_follow_the_standard_conversion() {
        let d = data(1, [0.8, 0.4, 0.2], 0.5, 0.5);
        let mut missing = BTreeSet::new();
        let mut at = |src| sample(src, 0, &d, false, &mut missing);
        let diffuse = |c| ChannelSrc::Color { map: ColorMap::Diffuse, comp: c };
        let specular = |c| ChannelSrc::Color { map: ColorMap::Specular, comp: c };
        assert!((at(diffuse(Comp::R)) - 0.4).abs() < 1e-6, "base * (1 - metallic)");
        assert!((at(specular(Comp::R)) - (0.04 + (0.8 - 0.04) * 0.5)).abs() < 1e-6, "lerp(0.04, base, metallic)");
        let l = at(ChannelSrc::base(Comp::L));
        assert!((l - (0.2126 * 0.8 + 0.7152 * 0.4 + 0.0722 * 0.2)).abs() < 1e-6, "Rec.709 luminance");
    }

    #[test]
    fn exr_preserves_float_values_exactly_and_without_srgb() {
        let dir = tempfile::tempdir().unwrap();
        let d = data(2, [0.2, 0.4, 0.6], 0.123_456, 0.654_321);
        export(&preset("OpenEXR 32-bit (linear)"), &d, &ctx(dir.path())).unwrap();
        let base = image::open(dir.path().join("proj_set_BaseColor.exr")).unwrap().to_rgb32f();
        assert_eq!(base.get_pixel(1, 1).0, [0.2, 0.4, 0.6]);
        let rough = image::open(dir.path().join("proj_set_Roughness.exr")).unwrap().to_rgb32f();
        assert_eq!(rough.get_pixel(0, 1).0, [0.654_321; 3]);
    }

    #[test]
    fn sixteen_bit_png_and_tga_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let d = data(2, [0.5; 3], 0.5, 0.25);
        let mut p = preset("PBR Metallic Roughness");
        p.maps[1].bit_depth = BitDepth::U16;
        p.maps[2].format = FileFormat::Tga;
        export(&p, &d, &ctx(dir.path())).unwrap();
        let r = image::open(dir.path().join("proj_set_Roughness.png")).unwrap().to_luma16();
        assert_eq!(r.get_pixel(0, 0).0[0], (0.25f32 * 65535.0).round() as u16);
        let t = image::open(dir.path().join("proj_set_Metallic.tga")).unwrap().to_luma8();
        assert_eq!((t.width(), t.get_pixel(1, 1).0[0]), (2, 128));
    }

    #[test]
    fn project_names_cannot_escape_the_output_directory() {
        let dir = tempfile::tempdir().unwrap();
        let inner = dir.path().join("out");
        let evil = ExportContext { project: "../../evil/..\\x", texture_set: "a/b", mesh: "m", out_dir: &inner, udim: None };
        let out = export(&preset("PBR Metallic Roughness"), &data(2, [0.5; 3], 0.5, 0.5), &evil).unwrap();
        for f in out.files {
            assert_eq!(f.parent().unwrap(), inner, "{f:?} escaped");
            assert!(f.exists());
        }
        assert!(!dir.path().join("evil").exists());
    }

    #[test]
    fn colliding_output_names_are_refused_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = preset("PBR Metallic Roughness");
        p.maps[1].file = p.maps[0].file.clone();
        assert!(export(&p, &data(2, [0.5; 3], 0.5, 0.5), &ctx(dir.path())).is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn dilation_fills_exactly_the_requested_width_and_keeps_covered_texels() {
        // 8x8, only the left column covered with distinct values.
        let size = 8u32;
        let mut d = data(size, [0.0; 3], 0.0, 0.0);
        d.set_baked(MeshMap::AmbientOcclusion, BakedMap::filled(8, 1, [0.0; 3])).unwrap();
        let mut covered = vec![false; 64];
        for y in 0..8usize {
            covered[y * 8] = true;
            d.metallic[y * 8] = 1.0;
            d.base[y * 8] = [1.0, 0.5, 0.25];
            d.baked.get_mut(&MeshMap::AmbientOcclusion).unwrap().set_texel(y * 8, [0.9; 3]);
        }
        d.dilate(&covered, 3);
        for y in 0..8usize {
            assert_eq!(d.metallic[y * 8], 1.0, "covered texel changed");
            for x in 1..=3usize {
                assert!(d.metallic[y * 8 + x] > 0.99 && d.base[y * 8 + x][1] > 0.49, "({x},{y}) not padded");
                assert!(d.baked[&MeshMap::AmbientOcclusion].texel(y * 8 + x)[0] > 0.89, "baked maps are padded too");
            }
            assert_eq!(d.metallic[y * 8 + 4], 0.0, "padding went further than requested");
        }
    }

    #[test]
    fn sphere_uvs_cover_almost_the_whole_texture() {
        let m = op_mesh::Mesh::uv_sphere(1.0, 48, 24);
        let cov = uv_coverage(&m, 128);
        let frac = cov.iter().filter(|c| **c).count() as f32 / cov.len() as f32;
        assert!(frac > 0.99, "coverage {frac}");
    }

    #[test]
    fn partial_uv_layout_is_detected() {
        use op_mesh::{Mesh, Vertex};
        let v = |u: f32, w: f32| Vertex { position: [0.0; 3], normal: [0.0, 1.0, 0.0], uv: [u, w] };
        // One triangle covering the lower-left half of UV space (v grows downward in images).
        let m = Mesh { vertices: vec![v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)], indices: vec![0, 1, 2], ..Default::default() };
        let cov = uv_coverage(&m, 32);
        let at = |x: usize, y: usize| cov[y * 32 + x];
        assert!(at(2, 28) && at(10, 30), "inside texels missing");
        assert!(!at(28, 2) && !at(30, 8), "outside texels covered");
        let frac = cov.iter().filter(|c| **c).count() as f32 / cov.len() as f32;
        assert!((0.45..0.6).contains(&frac), "half-ish coverage expected, got {frac}");
    }

    #[test]
    fn invalid_presets_are_rejected_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let d = data(2, [0.5; 3], 0.5, 0.5);
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].channels.pop(); // 2 channels
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].file = "../escape".into();
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].format = FileFormat::Exr; // exr needs f32
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].channels[0] = ChannelSrc::Constant(f32::NAN);
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0, "nothing should be written");
    }

    #[test]
    fn presets_roundtrip_through_json_and_builtins_are_valid() {
        for p in builtin_presets() {
            p.validate().unwrap_or_else(|e| panic!("{}: {e}", p.name));
            let json = serde_json::to_string_pretty(&p).unwrap();
            let back: ExportPreset = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p);
        }
    }

    /// Painted height becomes a tangent-space normal map, and the Height source writes it as is.
    #[test]
    fn height_and_normal_maps_are_exported() {
        let size = 8u32;
        let n = (size * size) as usize;
        // A ramp rising to the right, as the compositor would store it: (height * coverage, coverage).
        let ramp: Vec<u8> = (0..n).flat_map(|i| [(((i as u32 % size) * 255 / (size - 1)) as u8), 255]).collect();
        let mut d = data(size, [0.5; 3], 0.0, 0.5);
        d.set_height(&ramp).unwrap();
        assert!(d.height[0] < 0.01 && d.height[7] > 0.99);
        d.compute_normal(50.0);
        let nm = d.normal.as_ref().expect("a painted ramp produces a normal map");
        assert!(nm[3 * 8 + 4][0] < -0.05, "the surface faces left of a rise to the right: {:?}", nm[3 * 8 + 4]);

        let dir = tempfile::tempdir().unwrap();
        let spec = |file: &str, channels: Vec<ChannelSrc>| MapSpec {
            file: file.into(),
            channels,
            format: FileFormat::Png,
            bit_depth: BitDepth::U8,
            srgb: false,
        };
        let preset = ExportPreset {
            name: "t".into(),
            maps: vec![
                spec("n", vec![ChannelSrc::Normal { directx: false, comp: Comp::R }, ChannelSrc::Normal { directx: false, comp: Comp::G }, ChannelSrc::Normal { directx: false, comp: Comp::B }]),
                spec("nd", vec![ChannelSrc::Normal { directx: true, comp: Comp::G }]),
                spec("h", vec![ChannelSrc::Height]),
            ],
        };
        let out = export(&preset, &d, &ExportContext { project: "p", texture_set: "s", mesh: "m", out_dir: dir.path(), udim: None }).unwrap();
        assert!(out.notes.is_empty(), "{:?}", out.notes);
        let normal = image::open(dir.path().join("n.png")).unwrap().to_rgb8();
        let px = normal.get_pixel(4, 3).0;
        assert!(px[0] < 120 && px[2] > 200, "tilted left, mostly facing out: {px:?}");
        let (gl, dx) = (normal.get_pixel(4, 3).0[1], image::open(dir.path().join("nd.png")).unwrap().to_luma8().get_pixel(4, 3).0[0]);
        assert!((gl as i32 + dx as i32 - 255).abs() <= 1, "DirectX flips green: {gl} {dx}");
        let height = image::open(dir.path().join("h.png")).unwrap().to_luma8();
        assert!(height.get_pixel(0, 0).0[0] < 5 && height.get_pixel(7, 0).0[0] > 250);
    }

    #[test]
    fn a_fill_layer_normal_map_reaches_the_exported_normal() {
        let mut d = data(2, [0.5; 3], 0.0, 0.5);
        d.set_height(&[0; 8]).unwrap();
        // Every texel: a normal tilted towards +x (packed), fully covered.
        let tilt = op_core::normal::pack([0.6, 0.0, 0.8]).map(|c| (c * 255.0).round() as u8);
        let px = [tilt[0], tilt[1], tilt[2], 255];
        d.set_layer_normal(&px.repeat(4)).unwrap();
        d.compute_normal(1.0);
        let n = d.normal.as_ref().expect("a layer normal map alone is enough").clone();
        assert!(n[0][0] > 0.3 && n[0][1].abs() < 0.02, "{:?}", n[0]);
        assert!(d.set_layer_normal(&[0; 3]).is_err(), "wrong size is refused");
        // Nothing covered: no normal map is made.
        let mut e = data(2, [0.5; 3], 0.0, 0.5);
        e.set_layer_normal(&[128, 128, 255, 0].repeat(4)).unwrap();
        e.compute_normal(1.0);
        assert!(e.normal.is_none());
    }

    #[test]
    fn with_nothing_painted_there_is_no_normal_and_the_baked_one_is_used() {
        let mut d = data(2, [0.5; 3], 0.0, 0.5);
        d.set_height(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap(); // coverage 0 everywhere: flat
        d.compute_normal(1.0);
        assert!(d.normal.is_none(), "nothing painted and nothing baked");
        let tilted = BakedMap::new(2, 3, [0.8f32, 0.5, 0.9].iter().cycle().take(12).map(|v| (v * 65535.0) as u16).collect()).unwrap();
        d.set_baked(MeshMap::NormalBase, tilted).unwrap();
        d.compute_normal(1.0);
        let n = d.normal.as_ref().unwrap()[0];
        assert!(n[0] > 0.2 && (n.iter().map(|c| c * c).sum::<f32>() - 1.0).abs() < 1e-4, "{n:?}");
        assert!(d.set_height(&[0; 3]).is_err(), "wrong size is refused");
    }
}
