//! Export presets: which files to write and how channels are packed into them.
//! This is our own schema (JSON-serializable); it is not an Adobe `.spexp`.

use crate::ExportError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelSrc {
    /// Base color channels (linear in the document; sRGB-encoded on output if `srgb`).
    BaseR,
    BaseG,
    BaseB,
    Metallic,
    Roughness,
    /// `1 - roughness`, for engines that store smoothness.
    Smoothness,
    /// A fixed value, e.g. a white ambient-occlusion placeholder (there is no AO channel yet).
    Constant(f32),
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
    /// File name without extension. May contain `$project` and `$textureSet`.
    pub file: String,
    /// 1 (gray), 3 (RGB) or 4 (RGBA) sources, in output channel order.
    pub channels: Vec<ChannelSrc>,
    pub format: FileFormat,
    pub bit_depth: BitDepth,
    /// Encode base-color sources as sRGB (ignored for EXR, which is always linear).
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
}

fn map(file: &str, channels: &[ChannelSrc], format: FileFormat, bit_depth: BitDepth, srgb: bool) -> MapSpec {
    MapSpec { file: file.to_string(), channels: channels.to_vec(), format, bit_depth, srgb }
}

pub fn builtin_presets() -> Vec<ExportPreset> {
    use ChannelSrc::*;
    let png8 = |file: &str, ch: &[ChannelSrc], srgb: bool| map(file, ch, FileFormat::Png, BitDepth::U8, srgb);
    let color = [BaseR, BaseG, BaseB];
    // Occlusion has no channel yet, so packed maps carry a white placeholder.
    let orm = [Constant(1.0), Roughness, Metallic];
    vec![
        ExportPreset {
            name: "PBR Metallic Roughness".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor", &color, true),
                png8("$project_$textureSet_Roughness", &[Roughness], false),
                png8("$project_$textureSet_Metallic", &[Metallic], false),
            ],
        },
        ExportPreset {
            name: "glTF Metallic-Roughness".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor", &color, true),
                png8("$project_$textureSet_MetallicRoughness", &orm, false),
            ],
        },
        ExportPreset {
            name: "Unreal Engine (ORM packed)".into(),
            maps: vec![
                png8("$project_$textureSet_BaseColor", &color, true),
                png8("$project_$textureSet_OcclusionRoughnessMetallic", &orm, false),
            ],
        },
        ExportPreset {
            name: "Unity Standard (Metallic Smoothness)".into(),
            maps: vec![
                png8("$project_$textureSet_AlbedoTransparency", &[BaseR, BaseG, BaseB, Constant(1.0)], true),
                png8("$project_$textureSet_MetallicSmoothness", &[Metallic, Constant(0.0), Constant(0.0), Smoothness], false),
            ],
        },
        ExportPreset {
            name: "OpenEXR 32-bit (linear)".into(),
            maps: vec![
                map("$project_$textureSet_BaseColor", &color, FileFormat::Exr, BitDepth::F32, false),
                map("$project_$textureSet_Roughness", &[Roughness], FileFormat::Exr, BitDepth::F32, false),
                map("$project_$textureSet_Metallic", &[Metallic], FileFormat::Exr, BitDepth::F32, false),
            ],
        },
    ]
}
