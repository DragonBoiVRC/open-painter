//! Export presets: which files to write and how channels are packed into them.
//! This is our own schema (JSON-serializable). Adobe `.spexp` files are converted into it
//! by [`crate::import`].

use crate::ExportError;
use op_core::MeshMap;
use serde::{Deserialize, Serialize};

/// A component of a color-like map. `L` is luminance (Rec. 709 weights).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comp {
    R,
    G,
    B,
    L,
}

impl Comp {
    pub fn index(self) -> Option<usize> {
        match self {
            Comp::R => Some(0),
            Comp::G => Some(1),
            Comp::B => Some(2),
            Comp::L => None,
        }
    }
}

/// Color maps derived from the document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMap {
    Base,
    /// Diffuse color of the specular/glossiness workflow: `base * (1 - metallic)`.
    Diffuse,
    /// Specular color (F0) of the specular/glossiness workflow: `lerp(0.04, base, metallic)`.
    Specular,
    /// The painted emissive colour (black where nothing glows).
    Emissive,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelSrc {
    Color { map: ColorMap, comp: Comp },
    Metallic,
    Roughness,
    /// `1 - roughness`, for engines that store smoothness / glossiness.
    Smoothness,
    /// A baked mesh map. Scalar maps ignore the component.
    Mesh { map: MeshMap, comp: Comp },
    /// Tangent-space normal map: painted height combined with the baked normal map when there
    /// is one. `directx` flips the green channel.
    Normal { directx: bool, comp: Comp },
    /// The painted height channel (0.5 where nothing is painted).
    Height,
    /// Ambient occlusion. There is no painted AO channel yet, so this is the baked AO.
    AoMixed,
    /// A fixed value, e.g. a white placeholder or an unsupported source replaced by black.
    Constant(f32),
}

impl ChannelSrc {
    /// True if this source reads the emissive channel.
    pub fn is_emissive(self) -> bool {
        matches!(self, ChannelSrc::Color { map: ColorMap::Emissive, .. })
    }

    pub fn base(comp: Comp) -> Self {
        ChannelSrc::Color { map: ColorMap::Base, comp }
    }

    /// Color-like sources are sRGB-encoded in 8/16-bit files when the map asks for it.
    pub fn is_color(self) -> bool {
        matches!(self, ChannelSrc::Color { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileFormat {
    Png,
    Tga,
    Exr,
}

impl FileFormat {
    pub fn extension(self) -> &'static str {
        match self {
            FileFormat::Png => "png",
            FileFormat::Tga => "tga",
            FileFormat::Exr => "exr",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDepth {
    U8,
    U16,
    F32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapSpec {
    /// File name template without extension. See [`crate::expand_template`].
    pub file: String,
    /// 1 (gray), 3 (RGB) or 4 (RGBA) sources, in output channel order.
    pub channels: Vec<ChannelSrc>,
    pub format: FileFormat,
    pub bit_depth: BitDepth,
    /// Encode color-like sources as sRGB (ignored for EXR, which is always linear).
    pub srgb: bool,
}

impl MapSpec {
    pub fn validate(&self) -> Result<(), ExportError> {
        let bad = |why: String| Err(ExportError::Preset(format!("map '{}': {why}", self.file)));
        if !matches!(self.channels.len(), 1 | 3 | 4) {
            return bad(format!("needs 1, 3 or 4 channels, has {}", self.channels.len()));
        }
        let fits = match self.format {
            FileFormat::Png => matches!(self.bit_depth, BitDepth::U8 | BitDepth::U16),
            FileFormat::Tga => self.bit_depth == BitDepth::U8,
            FileFormat::Exr => self.bit_depth == BitDepth::F32,
        };
        if !fits {
            return bad(format!("{:?} does not support {:?}", self.format, self.bit_depth));
        }
        // Presets can come from user files: never let one write outside the export folder.
        let f = &self.file;
        if f.is_empty() || f.contains(['/', '\\', ':', '\0']) || f.contains("..") || f.starts_with('.') {
            return bad("file name must be a plain name (no path separators or '..')".into());
        }
        crate::template::check_template(f).or_else(|e| bad(e))?;
        for c in &self.channels {
            if let ChannelSrc::Constant(v) = c {
                if !v.is_finite() {
                    return bad("constant channel is not a finite number".into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportPreset {
    pub name: String,
    pub maps: Vec<MapSpec>,
}

impl ExportPreset {
    pub fn validate(&self) -> Result<(), ExportError> {
        if self.maps.is_empty() {
            return Err(ExportError::Preset(format!("preset '{}' has no maps", self.name)));
        }
        self.maps.iter().try_for_each(MapSpec::validate)
    }

    /// True if every file name contains `$textureSet`, so exporting several texture sets writes
    /// separate files instead of each set overwriting the last.
    pub fn separates_texture_sets(&self) -> bool {
        self.maps.iter().all(|m| m.file.contains("$textureSet"))
    }

    /// True if every file name contains `$udim`, so the tiles of a set go to separate files.
    pub fn separates_udim_tiles(&self) -> bool {
        self.maps.iter().all(|m| m.file.contains("$udim"))
    }
}

fn map(file: &str, channels: &[ChannelSrc], format: FileFormat, bit_depth: BitDepth, srgb: bool) -> MapSpec {
    MapSpec { file: file.to_string(), channels: channels.to_vec(), format, bit_depth, srgb }
}

pub fn builtin_presets() -> Vec<ExportPreset> {
    use ChannelSrc::*;
    let png8 = |file: &str, ch: &[ChannelSrc], srgb: bool| map(file, ch, FileFormat::Png, BitDepth::U8, srgb);
    let color = [ChannelSrc::base(Comp::R), ChannelSrc::base(Comp::G), ChannelSrc::base(Comp::B)];
    let glow = |c| ChannelSrc::Color { map: ColorMap::Emissive, comp: c };
    let emissive = [glow(Comp::R), glow(Comp::G), glow(Comp::B)];
    // Occlusion has no painted channel yet, so packed maps carry the baked AO (white if not baked).
    let orm = [AoMixed, Roughness, Metallic];
    vec![
        ExportPreset {
            name: "PBR Metallic Roughness".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor(.$udim)", &color, true),
                png8("$project_$textureSet_Roughness(.$udim)", &[Roughness], false),
                png8("$project_$textureSet_Metallic(.$udim)", &[Metallic], false),
                png8("$project_$textureSet_Emissive(.$udim)", &emissive, true),
            ],
        },
        ExportPreset {
            name: "glTF Metallic-Roughness".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor(.$udim)", &color, true),
                png8("$project_$textureSet_MetallicRoughness(.$udim)", &orm, false),
                png8("$project_$textureSet_Emissive(.$udim)", &emissive, true),
            ],
        },
        ExportPreset {
            name: "Unreal Engine (ORM packed)".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor(.$udim)", &color, true),
                png8("$project_$textureSet_OcclusionRoughnessMetallic(.$udim)", &orm, false),
                png8(
                    "$project_$textureSet_Normal(.$udim)",
                    &[Normal { directx: true, comp: Comp::R }, Normal { directx: true, comp: Comp::G }, Normal { directx: true, comp: Comp::B }],
                    false,
                ),
                png8("$project_$textureSet_Emissive(.$udim)", &emissive, true),
            ],
        },
        ExportPreset {
            name: "Unity Standard (Metallic Smoothness)".into(),
            maps: vec![
                png8(
                    "$project_$textureSet_AlbedoTransparency(.$udim)",
                    &[ChannelSrc::base(Comp::R), ChannelSrc::base(Comp::G), ChannelSrc::base(Comp::B), Constant(1.0)],
                    true,
                ),
                png8("$project_$textureSet_MetallicSmoothness(.$udim)", &[Metallic, Constant(0.0), Constant(0.0), Smoothness], false),
                png8(
                    "$project_$textureSet_Normal(.$udim)",
                    &[Normal { directx: false, comp: Comp::R }, Normal { directx: false, comp: Comp::G }, Normal { directx: false, comp: Comp::B }],
                    false,
                ),
                png8("$project_$textureSet_Emission(.$udim)", &emissive, true),
            ],
        },
        ExportPreset {
            name: "OpenEXR 32-bit (linear)".into(),
            maps: vec![
                map("$project_$textureSet_BaseColor(.$udim)", &color, FileFormat::Exr, BitDepth::F32, false),
                map("$project_$textureSet_Roughness(.$udim)", &[Roughness], FileFormat::Exr, BitDepth::F32, false),
                map("$project_$textureSet_Metallic(.$udim)", &[Metallic], FileFormat::Exr, BitDepth::F32, false),
                map("$project_$textureSet_Emissive(.$udim)", &emissive, FileFormat::Exr, BitDepth::F32, false),
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn built_in_presets_keep_texture_sets_apart_and_a_bare_name_does_not() {
        for preset in crate::builtin_presets() {
            assert!(preset.separates_texture_sets(), "{} would overwrite files between sets", preset.name);
        }
        let mut p = crate::builtin_presets().remove(0);
        p.maps[0].file = "diffuse".into();
        assert!(!p.separates_texture_sets());
        assert!(!p.separates_udim_tiles());
        for preset in crate::builtin_presets() {
            assert!(preset.separates_udim_tiles(), "{} would overwrite files between UV tiles", preset.name);
        }
    }
}
