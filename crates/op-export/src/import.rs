//! Convert a decoded Substance Painter export preset into our own [`ExportPreset`].
//!
//! The conversion is deliberately honest: whatever cannot be reproduced exactly (channels
//! this application does not have yet, unknown ids, file formats we cannot write) is
//! replaced by a constant and **reported as a warning**, never silently dropped or guessed.
//!
//! The id tables come from reading the presets that ship with Painter (the numbers are
//! bit positions of the document channels; baked maps and named maps have their own
//! numbering) and from the file names those presets give to the maps. Ids that no
//! sample explains stay "unknown".

use crate::preset::{BitDepth, ChannelSrc, ColorMap, Comp, ExportPreset, FileFormat, MapSpec};
use crate::spexp::{decode_spexp, MapId, SpexpChannel, SpexpError, SpexpMap};
use crate::template;
use op_core::MeshMap;

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error(transparent)]
    Decode(#[from] SpexpError),
    #[error("none of the preset's maps can be used here:\n{}", .0.join("\n"))]
    NoUsableMaps(Vec<String>),
}

#[derive(Debug)]
pub struct ImportedPreset {
    pub preset: ExportPreset,
    /// Everything that is not an exact translation, in file order.
    pub warnings: Vec<String>,
}

#[derive(Default)]
struct Warnings(Vec<String>);

impl Warnings {
    fn add(&mut self, msg: String) {
        if !self.0.contains(&msg) {
            self.0.push(msg);
        }
    }
}

/// Document-channel indices (bit positions) that are known from sample presets.
fn document_channel(index: u32) -> Option<&'static str> {
    Some(match index {
        0 => "BaseColor",
        1 => "Height",
        2 => "Specular",
        3 => "Opacity",
        4 => "Emissive",
        6 => "Glossiness",
        7 => "Roughness",
        8 => "Anisotropy level",
        9 => "Anisotropy angle",
        10 => "Transmissive",
        13 => "Metallic",
        24 => "Diffuse",
        25 => "Specular level",
        27 => "Scattering",
        64 => "User channel 0",
        65 => "User channel 1",
        66 => "User channel 2",
        67 => "User channel 3",
        _ => return None,
    })
}

fn mesh_map(index: i64) -> Option<MeshMap> {
    Some(match index {
        0 => MeshMap::Id,
        1 => MeshMap::AmbientOcclusion,
        2 => MeshMap::Curvature,
        3 => MeshMap::WorldSpaceNormal,
        4 => MeshMap::Position,
        5 => MeshMap::Thickness,
        6 => MeshMap::NormalBase,
        7 => MeshMap::HeightBase,
        8 => MeshMap::BentNormals,
        9 => MeshMap::OpacityBase,
        _ => return None,
    })
}

/// What a source map looks like to a channel that reads from it.
enum Source {
    /// An RGB map; `R`/`G`/`B`/`L` pick a component, `A` is opaque.
    Color(ColorMap),
    Normal { directx: bool },
    MeshVector(MeshMap),
    /// A single-valued map; every component reads the same value.
    Gray(ChannelSrc),
    /// Not available here; reads as `value` and is reported.
    Missing { what: String, value: f32 },
}

fn comp_of(src: i64) -> Option<Comp> {
    match src {
        0 => Some(Comp::R),
        1 => Some(Comp::G),
        2 => Some(Comp::B),
        4 => Some(Comp::L),
        _ => None, // 3 = alpha
    }
}

fn classify(ch: &SpexpChannel, version0: bool, w: &mut Warnings) -> Source {
    match (ch.map_type, &ch.map) {
        // Constants: the sample presets only use 0 (black) and 1 (white).
        (0, MapId::Int(v @ (0 | 1))) => Source::Gray(ChannelSrc::Constant(*v as f32)),
        (0, other) => Source::Missing { what: format!("constant {other:?}"), value: 0.0 },
        (1, MapId::Int(id)) => {
            // Version 0 stores the document channel as a single bit flag, version 1 as the bit index.
            let index = if version0 {
                if *id > 0 && (*id as u64).is_power_of_two() { Some((*id as u64).trailing_zeros()) } else { None }
            } else {
                u32::try_from(*id).ok()
            };
            let Some(index) = index else {
                return Source::Missing { what: format!("document channel mask {id}"), value: 0.0 };
            };
            match index {
                0 => Source::Color(ColorMap::Base),
                1 => Source::Gray(ChannelSrc::Height),
                2 => {
                    w.add("Specular color is derived from base color and metallic (this project has no painted specular channel).".into());
                    Source::Color(ColorMap::Specular)
                }
                24 => {
                    w.add("Diffuse color is derived as base color x (1 - metallic) (this project has no painted diffuse channel).".into());
                    Source::Color(ColorMap::Diffuse)
                }
                3 => Source::Gray(ChannelSrc::Opacity),
                6 => Source::Gray(ChannelSrc::Smoothness),
                7 => Source::Gray(ChannelSrc::Roughness),
                13 => Source::Gray(ChannelSrc::Metallic),
                other => Source::Missing {
                    what: match document_channel(other) {
                        Some(name) => format!("the {name} channel"),
                        None => format!("document channel #{other} (not identified)"),
                    },
                    value: 0.0,
                },
            }
        }
        (2, MapId::Str(name)) => match name.as_str() {
            "Normal_OpenGL" => Source::Normal { directx: false },
            "Normal_DirectX" => Source::Normal { directx: true },
            "AO_Mixed" => Source::Gray(ChannelSrc::AoMixed),
            "Glossiness" => Source::Gray(ChannelSrc::Smoothness),
            "Diffuse" => Source::Color(ColorMap::Diffuse),
            "Specular" => Source::Color(ColorMap::Specular),
            other => Source::Missing { what: format!("the virtual map '{other}'"), value: 0.0 },
        },
        (3, MapId::Int(id)) => match mesh_map(*id) {
            Some(m) if m.is_scalar() => Source::Gray(ChannelSrc::Mesh { map: m, comp: Comp::L }),
            Some(m) => Source::MeshVector(m),
            None => Source::Missing { what: format!("baked map #{id}"), value: 0.0 },
        },
        (t, id) => Source::Missing { what: format!("map {id:?} of unknown type {t}"), value: 0.0 },
    }
}

/// The document channel Substance calls Emissive.
fn is_emissive(ch: &SpexpChannel, version0: bool) -> bool {
    let id = match &ch.map {
        MapId::Int(i) if ch.map_type == 1 => *i,
        _ => return false,
    };
    if version0 { id == 16 } else { id == 4 }
}

fn resolve(
    ch: &SpexpChannel,
    version0: bool,
    dest: char,
    file: &str,
    w: &mut Warnings,
) -> ChannelSrc {
    let comp = comp_of(ch.src);
    if is_emissive(ch, version0) {
        return match comp {
            Some(c) => ChannelSrc::Color { map: ColorMap::Emissive, comp: c },
            None => ChannelSrc::Constant(1.0),
        };
    }
    match classify(ch, version0, w) {
        Source::Color(map) => match comp {
            Some(c) => ChannelSrc::Color { map, comp: c },
            None => ChannelSrc::Constant(1.0),
        },
        Source::Normal { directx } => match comp {
            Some(c) => ChannelSrc::Normal { directx, comp: c },
            None => ChannelSrc::Constant(1.0),
        },
        Source::MeshVector(map) => match comp {
            Some(c) => ChannelSrc::Mesh { map, comp: c },
            None => ChannelSrc::Constant(1.0),
        },
        Source::Gray(src) => src,
        Source::Missing { what, value } => {
            w.add(format!("{file}: channel {dest} reads {what}, which is not available here; it is written as {value}."));
            ChannelSrc::Constant(value)
        }
    }
}

fn convert_map(m: &SpexpMap, version0: bool, w: &mut Warnings) -> Result<MapSpec, String> {
    let (format, depth) = match m.file_format.to_ascii_lowercase().as_str() {
        "png" => (FileFormat::Png, if m.bitdepth == 1 { BitDepth::U16 } else { BitDepth::U8 }),
        "tga" => {
            if m.bitdepth == 1 {
                w.add(format!("{}: TGA is written as 8-bit.", m.path));
            }
            (FileFormat::Tga, BitDepth::U8)
        }
        "exr" => (FileFormat::Exr, BitDepth::F32),
        other => return Err(format!("{}: the '{other}' file format cannot be written yet", m.path)),
    };
    if !matches!(m.bitdepth, 0 | 1) && format == FileFormat::Png {
        w.add(format!("{}: unknown bit depth setting {}; written as 8-bit.", m.path, m.bitdepth));
    }
    if m.dithering {
        w.add(format!("{}: dithering is not applied.", m.path));
    }
    let dests: &[char] = match m.channels.len() {
        1 => &['L'],
        3 => &['R', 'G', 'B'],
        4 => &['R', 'G', 'B', 'A'],
        n => return Err(format!("{}: {n} channels is not a supported layout", m.path)),
    };
    let channels: Vec<ChannelSrc> =
        m.channels.iter().zip(dests).map(|(c, d)| resolve(c, version0, *d, &m.path, w)).collect();
    for var in template::variables(&m.path) {
        if !template::KNOWN_VARS.contains(&var.as_str()) {
            w.add(format!("{}: the variable ${var} is not supported and expands to nothing.", m.path));
        }
    }
    let spec = MapSpec { srgb: channels.iter().any(|c| c.is_color()), file: m.path.clone(), channels, format, bit_depth: depth };
    spec.validate().map_err(|e| format!("{}: {e}", m.path))?;
    Ok(spec)
}

/// Import a `.spexp` file's bytes. `name` becomes the preset's name (usually the file stem).
pub fn import_spexp(bytes: &[u8], name: &str) -> Result<ImportedPreset, ImportError> {
    let decoded = decode_spexp(bytes)?;
    let version0 = decoded.container_version == 0;
    let mut w = Warnings::default();
    let mut maps = Vec::new();
    let mut skipped = Vec::new();
    for m in &decoded.maps {
        match convert_map(m, version0, &mut w) {
            Ok(spec) => maps.push(spec),
            Err(why) => skipped.push(format!("Skipped {why}")),
        }
    }
    if maps.is_empty() {
        return Err(ImportError::NoUsableMaps(skipped));
    }
    let mut warnings = w.0;
    warnings.extend(skipped);
    Ok(ImportedPreset { preset: ExportPreset { name: name.to_string(), maps }, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spexp::fixtures::*;

    fn map<'a>(path: &'a str, fmt: &'a str, bitdepth: i64, channels: Vec<Ch>) -> Map<'a> {
        Map { bitdepth, dithering: false, format: fmt, path, channels }
    }

    fn import(maps: &[Map], v0: bool) -> Result<ImportedPreset, ImportError> {
        let bytes = if v0 { encode_v0(maps) } else { encode_v1(maps) };
        import_spexp(&bytes, "test")
    }

    fn rgb(id: Vec<u8>, ty: i64) -> Vec<Ch> {
        (0..3).map(|s| Ch { src: s, id: id.clone(), ty }).collect()
    }

    #[test]
    fn base_color_with_opacity_alpha_matches_in_both_container_versions() {
        // v0 uses bit flags (1 = bit 0, 8 = bit 3); v1 uses bit indices (0, 3).
        let v0 = vec![map("c_$textureSet(.$udim)", "png", 1, {
            let mut c = rgb(variant_u64(1), 1);
            c.push(Ch { src: 4, id: variant_u64(8), ty: 1 });
            c
        })];
        let v1 = vec![map("c_$textureSet(.$udim)", "png", 1, {
            let mut c = rgb(variant_i32(0), 1);
            c.push(Ch { src: 4, id: variant_i32(3), ty: 1 });
            c
        })];
        let a = import(&v0, true).unwrap();
        let b = import(&v1, false).unwrap();
        assert_eq!(a.preset.maps, b.preset.maps, "same preset in either container");
        let m = &a.preset.maps[0];
        assert_eq!(m.channels[..3], [ChannelSrc::base(Comp::R), ChannelSrc::base(Comp::G), ChannelSrc::base(Comp::B)]);
        assert_eq!(m.channels[3], ChannelSrc::Opacity, "alpha reads the opacity channel");
        assert!(m.srgb && m.format == FileFormat::Png && m.bit_depth == BitDepth::U16);
        assert!(a.warnings.is_empty(), "{:?}", a.warnings);
    }

    #[test]
    fn named_baked_and_data_maps_translate_exactly() {
        let maps = vec![
            map("n", "png", 0, rgb(variant_str("Normal_DirectX"), 2)),
            map("ao", "png", 0, vec![Ch { src: 4, id: variant_str("AO_Mixed"), ty: 2 }]),
            map("rough", "png", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }]),
            map("metal", "png", 0, vec![Ch { src: 4, id: variant_i32(13), ty: 1 }]),
            map("curv", "png", 0, vec![Ch { src: 4, id: variant_i32(2), ty: 3 }]),
            map("wsn", "png", 0, rgb(variant_i32(3), 3)),
            map("gloss", "png", 0, vec![Ch { src: 4, id: variant_str("Glossiness"), ty: 2 }]),
        ];
        let got = import(&maps, false).unwrap();
        assert!(got.warnings.is_empty(), "exact translations should not warn: {:?}", got.warnings);
        let ch = |i: usize| got.preset.maps[i].channels.clone();
        assert_eq!(ch(0)[1], ChannelSrc::Normal { directx: true, comp: Comp::G });
        assert_eq!(ch(1), [ChannelSrc::AoMixed]);
        assert_eq!(ch(2), [ChannelSrc::Roughness]);
        assert_eq!(ch(3), [ChannelSrc::Metallic]);
        assert_eq!(ch(4), [ChannelSrc::Mesh { map: MeshMap::Curvature, comp: Comp::L }]);
        assert_eq!(ch(5)[2], ChannelSrc::Mesh { map: MeshMap::WorldSpaceNormal, comp: Comp::B });
        assert_eq!(ch(6), [ChannelSrc::Smoothness]);
        assert!(!got.preset.maps[0].srgb && !got.preset.maps[2].srgb, "data maps are never sRGB");
    }

    #[test]
    fn unsupported_and_unknown_channels_become_reported_constants() {
        let maps = vec![map(
            "mask",
            "png",
            0,
            vec![
                Ch { src: 4, id: variant_u64(1 << 13), ty: 1 },  // metallic: exact
                Ch { src: 4, id: variant_str("Glossiness"), ty: 2 },
                Ch { src: 4, id: variant_u64(1 << 14), ty: 1 },  // not identified
                Ch { src: 4, id: variant_u64(1 << 25), ty: 1 },  // specular level: known but unsupported
            ],
        )];
        let got = import(&maps, true).unwrap();
        let c = &got.preset.maps[0].channels;
        assert_eq!((c[0], c[1]), (ChannelSrc::Metallic, ChannelSrc::Smoothness));
        assert_eq!((c[2], c[3]), (ChannelSrc::Constant(0.0), ChannelSrc::Constant(0.0)));
        let joined = got.warnings.join("\n");
        assert!(joined.contains("channel B") && joined.contains("#14"), "{joined}");
        assert!(joined.contains("channel A") && joined.contains("Specular level"), "{joined}");
        // Height, on the other hand, is a real channel now.
        let h = import(&[map("h", "png", 0, vec![Ch { src: 4, id: variant_u64(1 << 1), ty: 1 }])], true).unwrap();
        assert_eq!(h.preset.maps[0].channels[0], ChannelSrc::Height);
        assert!(h.warnings.is_empty(), "{:?}", h.warnings);
    }

    #[test]
    fn emissive_maps_read_the_emissive_channel_in_both_versions() {
        let v0 = import(&[map("e", "png", 0, rgb(variant_u64(16), 1))], true).unwrap();
        let v1 = import(&[map("e", "png", 0, rgb(variant_i32(4), 1))], false).unwrap();
        for got in [v0, v1] {
            assert!(got.preset.maps[0].channels.iter().all(|c| c.is_emissive()), "{:?}", got.preset.maps[0].channels);
            assert!(got.warnings.is_empty(), "{:?}", got.warnings);
        }
    }

    #[test]
    fn version_0_masks_that_are_not_a_single_bit_are_not_misread() {
        // 3 = bits 0 and 1 together: not a document channel we can name.
        let got = import(&[map("m", "png", 0, vec![Ch { src: 4, id: variant_u64(3), ty: 1 }])], true).unwrap();
        assert_eq!(got.preset.maps[0].channels[0], ChannelSrc::Constant(0.0));
        assert!(got.warnings[0].contains("mask 3"), "{:?}", got.warnings);
    }

    #[test]
    fn unusable_maps_are_skipped_with_a_reason_and_all_unusable_is_an_error() {
        let ok = map("fine", "png", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }]);
        let folders = map("dir/name", "png", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }]);
        let psd = map("layers", "psd", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }]);
        let two = map("two", "png", 0, vec![Ch { src: 0, id: variant_i32(7), ty: 1 }, Ch { src: 1, id: variant_i32(7), ty: 1 }]);
        let got = import(&[ok, folders, psd, two], false).unwrap();
        assert_eq!(got.preset.maps.len(), 1);
        let joined = got.warnings.join("\n");
        assert!(joined.contains("dir/name") && joined.contains("psd") && joined.contains("2 channels"), "{joined}");

        let only_bad = [map("x", "bmp", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }])];
        match import(&only_bad, false) {
            Err(ImportError::NoUsableMaps(why)) => assert!(why[0].contains("bmp")),
            other => panic!("expected NoUsableMaps, got {other:?}"),
        }
    }

    #[test]
    fn formats_and_depths_map_sensibly() {
        let ch = || vec![Ch { src: 4, id: variant_i32(7), ty: 1 }];
        let got = import(&[map("a", "tga", 1, ch()), map("b", "exr", 0, ch()), map("c", "PNG", 1, ch())], false).unwrap();
        let m = &got.preset.maps;
        assert_eq!((m[0].format, m[0].bit_depth), (FileFormat::Tga, BitDepth::U8));
        assert_eq!((m[1].format, m[1].bit_depth), (FileFormat::Exr, BitDepth::F32));
        assert_eq!((m[2].format, m[2].bit_depth), (FileFormat::Png, BitDepth::U16));
        assert!(got.warnings.iter().any(|w| w.contains("TGA")));
    }

    #[test]
    fn unsupported_template_variables_are_reported() {
        let got = import(&[map("$weird_$textureSet", "png", 0, vec![Ch { src: 4, id: variant_i32(7), ty: 1 }])], false).unwrap();
        assert!(got.warnings.iter().any(|w| w.contains("$weird")), "{:?}", got.warnings);
    }

    #[test]
    fn imported_presets_export_real_files() {
        use crate::{export, ExportContext, TextureSetData};
        let maps = vec![
            map("t_$textureSet_Base(.$udim)", "png", 0, {
                let mut c = rgb(variant_i32(0), 1);
                c.push(Ch { src: 4, id: variant_i32(3), ty: 1 });
                c
            }),
            map("t_$textureSet_Packed", "png", 0, vec![
                Ch { src: 4, id: variant_i32(13), ty: 1 },
                Ch { src: 4, id: variant_str("Glossiness"), ty: 2 },
                Ch { src: 4, id: variant_i32(1), ty: 0 },
            ]),
        ];
        let got = import(&maps, false).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let n = 4usize;
        let data = TextureSetData {
            size: 2,
            base: vec![[0.5; 3]; n],
            metallic: vec![0.25; n],
            roughness: vec![0.75; n],
            baked: Default::default(),
            height: Vec::new(),
            emissive: Vec::new(),
            opacity: Vec::new(),
            layer_normal: Vec::new(),
            normal: None,
        };
        let out = export(&got.preset, &data, &ExportContext { project: "p", texture_set: "body", mesh: "m", out_dir: dir.path(), udim: None });
        // The packed map has a 3-channel layout: R metallic, G smoothness, B constant white.
        let out = out.unwrap();
        assert_eq!(out.files.len(), 2);
        let base = image::open(dir.path().join("t_body_Base.png")).unwrap().to_rgba8();
        assert_eq!(base.get_pixel(0, 0).0[3], 255, "opacity written as opaque");
        let packed = image::open(dir.path().join("t_body_Packed.png")).unwrap().to_rgb8();
        assert_eq!(packed.get_pixel(1, 1).0, [64, 64, 255]);
    }
}
