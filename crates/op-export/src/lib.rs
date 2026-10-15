//! Texture export: turn the composited channels of a texture set into image files
//! according to an export preset. Pure CPU, no GPU or UI.

mod preset;
mod uv;

pub use preset::{builtin_presets, BitDepth, ChannelSrc, ExportPreset, FileFormat, MapSpec};
pub use uv::uv_coverage;

use op_core::{FALLBACK_BASE, FALLBACK_METALLIC, FALLBACK_ROUGHNESS};
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
        })
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
                    let (mut n, mut b, mut m, mut r) = (0.0f32, [0.0f32; 3], 0.0f32, 0.0f32);
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let (nx, ny) = (x + dx, y + dy);
                            if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= s || ny >= s {
                                continue;
                            }
                            let j = (ny * s + nx) as usize;
                            if covered[j] {
                                n += 1.0;
                                for c in 0..3 {
                                    b[c] += self.base[j][c];
                                }
                                m += self.metallic[j];
                                r += self.roughness[j];
                            }
                        }
                    }
                    if n > 0.0 {
                        // Only previously-covered texels are read, so writing in place is safe.
                        self.base[i] = [b[0] / n, b[1] / n, b[2] / n];
                        self.metallic[i] = m / n;
                        self.roughness[i] = r / n;
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
    pub out_dir: &'a Path,
}

/// Keep only characters that are safe in a file name; everything else becomes `_`.
fn sanitize(s: &str) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() { "untitled".into() } else { cleaned }
}

fn expand(template: &str, ctx: &ExportContext) -> String {
    template.replace("$project", &sanitize(ctx.project)).replace("$textureSet", &sanitize(ctx.texture_set))
}

fn srgb_encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn sample(src: ChannelSrc, i: usize, data: &TextureSetData, encode: bool) -> f32 {
    let base = |c: usize| {
        let v = data.base[i][c];
        if encode { srgb_encode(v) } else { v }
    };
    match src {
        ChannelSrc::BaseR => base(0),
        ChannelSrc::BaseG => base(1),
        ChannelSrc::BaseB => base(2),
        ChannelSrc::Metallic => data.metallic[i],
        ChannelSrc::Roughness => data.roughness[i],
        ChannelSrc::Smoothness => 1.0 - data.roughness[i],
        ChannelSrc::Constant(v) => v,
    }
}

fn write_map(path: &Path, spec: &MapSpec, data: &TextureSetData) -> Result<(), ExportError> {
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
    let values = |i: usize, c: usize| sample(spec.channels[c], i, data, encode).clamp(0.0, 1.0);

    match spec.bit_depth {
        BitDepth::U8 => {
            let mut buf = Vec::with_capacity(n * ch);
            for i in 0..n {
                for c in 0..ch {
                    buf.push((values(i, c) * 255.0).round() as u8);
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
                    buf.push((values(i, c) * 65535.0).round() as u16);
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
                    let src = if ch == 1 { 0 } else { c };
                    buf.push(sample(spec.channels[src], i, data, false));
                }
            }
            let color = if out_ch == 4 { ColorType::Rgba32F } else { ColorType::Rgb32F };
            image::save_buffer_with_format(path, bytemuck::cast_slice(&buf), data.size, data.size, color, format)
                .map_err(wrap)
        }
    }
}

/// Write every map of `preset` into `ctx.out_dir`. Returns the files written.
pub fn export(preset: &ExportPreset, data: &TextureSetData, ctx: &ExportContext) -> Result<Vec<PathBuf>, ExportError> {
    preset.validate()?;
    let n = data.size as usize * data.size as usize;
    if data.base.len() != n || data.metallic.len() != n || data.roughness.len() != n {
        return Err(ExportError::Data("channel lengths do not match size".into()));
    }
    std::fs::create_dir_all(ctx.out_dir)?;
    let mut written = Vec::new();
    for spec in &preset.maps {
        let name = format!("{}.{}", expand(&spec.file, ctx), spec.format.extension());
        let path = ctx.out_dir.join(name);
        write_map(&path, spec, data)?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn data(size: u32, base: [f32; 3], metallic: f32, roughness: f32) -> TextureSetData {
        let n = (size * size) as usize;
        TextureSetData { size, base: vec![base; n], metallic: vec![metallic; n], roughness: vec![roughness; n] }
    }

    fn ctx<'a>(dir: &'a Path) -> ExportContext<'a> {
        ExportContext { project: "proj", texture_set: "set", out_dir: dir }
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
        let files = export(&preset("PBR Metallic Roughness"), &d, &ctx(dir.path())).unwrap();
        assert_eq!(files.len(), 3);
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
    fn packed_orm_uses_the_documented_channel_order() {
        let dir = tempfile::tempdir().unwrap();
        let d = data(2, [0.5; 3], 0.25, 0.75);
        export(&preset("glTF Metallic-Roughness"), &d, &ctx(dir.path())).unwrap();
        let mr = image::open(dir.path().join("proj_set_MetallicRoughness.png")).unwrap().to_rgb8();
        // glTF: R = occlusion (white placeholder), G = roughness, B = metalness.
        assert_eq!(mr.get_pixel(0, 0).0, [255, (0.75f32 * 255.0).round() as u8, (0.25f32 * 255.0).round() as u8]);
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
        let evil = ExportContext { project: "../../evil/..\\x", texture_set: "a/b", out_dir: &inner };
        let files = export(&preset("PBR Metallic Roughness"), &data(2, [0.5; 3], 0.5, 0.5), &evil).unwrap();
        for f in files {
            assert_eq!(f.parent().unwrap(), inner, "{f:?} escaped");
            assert!(f.exists());
        }
        assert!(!dir.path().join("evil").exists());
    }

    #[test]
    fn dilation_fills_exactly_the_requested_width_and_keeps_covered_texels() {
        // 8x8, only the left column covered with distinct values.
        let size = 8u32;
        let mut d = data(size, [0.0; 3], 0.0, 0.0);
        let mut covered = vec![false; 64];
        for y in 0..8usize {
            covered[y * 8] = true;
            d.metallic[y * 8] = 1.0;
            d.base[y * 8] = [1.0, 0.5, 0.25];
        }
        d.dilate(&covered, 3);
        for y in 0..8usize {
            assert_eq!(d.metallic[y * 8], 1.0, "covered texel changed");
            for x in 1..=3usize {
                assert!(d.metallic[y * 8 + x] > 0.99 && d.base[y * 8 + x][1] > 0.49, "({x},{y}) not padded");
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
        let m = Mesh { vertices: vec![v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)], indices: vec![0, 1, 2] };
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
        p.maps[0].channels = vec![ChannelSrc::BaseR, ChannelSrc::BaseG]; // 2 channels
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].file = "../escape".into();
        assert!(export(&p, &d, &ctx(dir.path())).is_err());
        let mut p = preset("PBR Metallic Roughness");
        p.maps[0].format = FileFormat::Exr; // exr needs f32
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
}
