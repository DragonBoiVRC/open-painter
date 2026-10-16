//! Document model for open-painter: the layer stack. No GPU, no UI.
//!
//! `nodes` lists are ordered **bottom to top** (index 0 is composited first).

pub mod blend;
pub mod graph;
pub mod normal;
pub mod smart;

use graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type NodeId = u64;

/// One baked map: 16-bit normalised texels, either 1 channel (scalar) or 3 (vector/color).
/// Compact on purpose: baked maps are the largest data in a project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakedMap {
    pub size: u32,
    pub channels: u8,
    pub data: Vec<u16>,
}

impl BakedMap {
    pub fn new(size: u32, channels: u8, data: Vec<u16>) -> Result<Self, String> {
        if !matches!(channels, 1 | 3) {
            return Err(format!("a baked map has 1 or 3 channels, not {channels}"));
        }
        let want = size as usize * size as usize * channels as usize;
        if size == 0 || data.len() != want {
            return Err(format!("{size}x{size}x{channels} needs {want} values, got {}", data.len()));
        }
        Ok(Self { size, channels, data })
    }

    /// A map filled with one value, for tests and neutral defaults.
    pub fn filled(size: u32, channels: u8, value: [f32; 3]) -> Self {
        let n = size as usize * size as usize;
        let mut data = Vec::with_capacity(n * channels as usize);
        for _ in 0..n {
            for c in 0..channels as usize {
                data.push(Self::quantize(value[c]));
            }
        }
        Self { size, channels, data }
    }

    pub fn quantize(v: f32) -> u16 {
        (v.clamp(0.0, 1.0) * 65535.0).round() as u16
    }

    pub fn texels(&self) -> usize {
        self.size as usize * self.size as usize
    }

    /// Texel `i` as RGB; scalar maps repeat their value.
    pub fn texel(&self, i: usize) -> [f32; 3] {
        let f = |v: u16| v as f32 / 65535.0;
        match self.channels {
            1 => {
                let v = f(self.data[i]);
                [v, v, v]
            }
            _ => [f(self.data[i * 3]), f(self.data[i * 3 + 1]), f(self.data[i * 3 + 2])],
        }
    }

    pub fn set_texel(&mut self, i: usize, v: [f32; 3]) {
        match self.channels {
            1 => self.data[i] = Self::quantize(v[0]),
            _ => {
                for c in 0..3 {
                    self.data[i * 3 + c] = Self::quantize(v[c]);
                }
            }
        }
    }
}

/// Maps that are baked from the mesh (and optionally a high-poly source) rather than painted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MeshMap {
    Id,
    AmbientOcclusion,
    Curvature,
    WorldSpaceNormal,
    Position,
    Thickness,
    /// Tangent-space normal map (OpenGL convention, +Y up), encoded 0..1.
    NormalBase,
    HeightBase,
    BentNormals,
    OpacityBase,
}

impl MeshMap {
    pub const ALL: [MeshMap; 10] = [
        MeshMap::Id,
        MeshMap::AmbientOcclusion,
        MeshMap::Curvature,
        MeshMap::WorldSpaceNormal,
        MeshMap::Position,
        MeshMap::Thickness,
        MeshMap::NormalBase,
        MeshMap::HeightBase,
        MeshMap::BentNormals,
        MeshMap::OpacityBase,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MeshMap::Id => "ID",
            MeshMap::AmbientOcclusion => "Ambient occlusion",
            MeshMap::Curvature => "Curvature",
            MeshMap::WorldSpaceNormal => "World space normal",
            MeshMap::Position => "Position",
            MeshMap::Thickness => "Thickness",
            MeshMap::NormalBase => "Normal (tangent space)",
            MeshMap::HeightBase => "Height",
            MeshMap::BentNormals => "Bent normals",
            MeshMap::OpacityBase => "Opacity",
        }
    }

    /// Value used where nothing has been baked: a neutral, "no detail" result.
    pub fn default_value(self) -> [f32; 3] {
        match self {
            MeshMap::Id | MeshMap::Position | MeshMap::Thickness => [0.0; 3],
            MeshMap::AmbientOcclusion | MeshMap::OpacityBase => [1.0; 3],
            MeshMap::Curvature | MeshMap::HeightBase | MeshMap::WorldSpaceNormal => [0.5; 3],
            MeshMap::NormalBase | MeshMap::BentNormals => [0.5, 0.5, 1.0],
        }
    }

    /// Channel count of the baked map.
    pub fn channels(self) -> u8 {
        if self.is_scalar() { 1 } else { 3 }
    }

    /// True for maps that are a single value per texel.
    pub fn is_scalar(self) -> bool {
        matches!(
            self,
            MeshMap::AmbientOcclusion | MeshMap::Curvature | MeshMap::Thickness | MeshMap::HeightBase | MeshMap::OpacityBase
        )
    }
}

/// What the viewport and exporters show where the layer stack leaves a texel uncovered.
pub const FALLBACK_BASE: [f32; 3] = [0.18, 0.18, 0.18];
pub const FALLBACK_METALLIC: f32 = 0.0;
pub const FALLBACK_ROUGHNESS: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Add,
    Subtract,
    Darken,
    Lighten,
    Difference,
    SoftLight,
    HardLight,
    ColorDodge,
    ColorBurn,
    LinearBurn,
    Exclusion,
    Divide,
}

impl BlendMode {
    pub const ALL: [BlendMode; 16] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Difference,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::LinearBurn,
        BlendMode::Exclusion,
        BlendMode::Divide,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Add => "Add",
            BlendMode::Subtract => "Subtract",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::Difference => "Difference",
            BlendMode::SoftLight => "Soft light",
            BlendMode::HardLight => "Hard light",
            BlendMode::ColorDodge => "Color dodge",
            BlendMode::ColorBurn => "Color burn",
            BlendMode::LinearBurn => "Linear burn",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Divide => "Divide",
        }
    }

    /// Index shared with the compositor shader.
    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|m| *m == self).unwrap() as u32
    }
}

/// A texture used as the source of one channel of a fill layer. `id` names an image in the
/// project's texture library (a content hash); the flags adjust how its pixels are read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TexRef {
    pub id: String,
    /// Use `1 - value` (for scalar channels).
    #[serde(default)]
    pub invert: bool,
    /// The normal map's green channel points down (DirectX convention) and must be flipped.
    #[serde(default)]
    pub flip_green: bool,
}

/// True for ids of the form the texture library makes (32 lowercase hex digits). A project file may
/// only name textures like this, so it can never smuggle in a path.
pub fn is_valid_texture_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl TexRef {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), invert: false, flip_green: false }
    }
}

/// The channels of a fill layer that read a texture instead of a constant. A channel with a texture
/// is active; the texture's pixels replace the constant.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FillMaps {
    #[serde(default)]
    pub base: Option<TexRef>,
    #[serde(default)]
    pub metallic: Option<TexRef>,
    #[serde(default)]
    pub roughness: Option<TexRef>,
    #[serde(default)]
    pub height: Option<TexRef>,
    /// Tangent-space normal map; combined with painted height and the baked normal map.
    #[serde(default)]
    pub normal: Option<TexRef>,
}

/// The channels a texture can feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    BaseColor,
    Metallic,
    Roughness,
    Height,
    Normal,
}

impl Slot {
    pub const ALL: [Slot; 5] = [Slot::BaseColor, Slot::Metallic, Slot::Roughness, Slot::Height, Slot::Normal];

    pub fn label(self) -> &'static str {
        match self {
            Slot::BaseColor => "Base color",
            Slot::Metallic => "Metallic",
            Slot::Roughness => "Roughness",
            Slot::Height => "Height",
            Slot::Normal => "Normal",
        }
    }
}

impl FillMaps {
    pub fn get(&self, slot: Slot) -> Option<&TexRef> {
        match slot {
            Slot::BaseColor => self.base.as_ref(),
            Slot::Metallic => self.metallic.as_ref(),
            Slot::Roughness => self.roughness.as_ref(),
            Slot::Height => self.height.as_ref(),
            Slot::Normal => self.normal.as_ref(),
        }
    }

    pub fn set(&mut self, slot: Slot, tex: Option<TexRef>) {
        match slot {
            Slot::BaseColor => self.base = tex,
            Slot::Metallic => self.metallic = tex,
            Slot::Roughness => self.roughness = tex,
            Slot::Height => self.height = tex,
            Slot::Normal => self.normal = tex,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Slot, &TexRef)> {
        Slot::ALL.into_iter().filter_map(|s| self.get(s).map(|t| (s, t)))
    }

    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

/// How a fill layer's textures are laid over the UV layout: scaled, shifted and turned.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UvTransform {
    /// How many times the texture repeats across the UV square.
    pub tiling: [f32; 2],
    pub offset: [f32; 2],
    /// Counter-clockwise, in degrees, around the middle of the texture.
    pub rotation: f32,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self { tiling: [1.0, 1.0], offset: [0.0, 0.0], rotation: 0.0 }
    }
}

impl UvTransform {
    /// Tiling between 1/1000 and 1000 times, finite offset and rotation.
    pub fn is_sane(&self) -> bool {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && (lo..=hi).contains(&v);
        self.tiling.iter().all(|t| ok(t.abs(), 0.001, 1000.0)) && self.offset.iter().all(|o| ok(*o, -1000.0, 1000.0)) && ok(self.rotation, -3600.0, 3600.0)
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// The texture coordinate for a UV position: turn around the centre, scale, then shift.
    pub fn apply(&self, uv: [f32; 2]) -> [f32; 2] {
        let (s, c) = self.rotation.to_radians().sin_cos();
        let (x, y) = (uv[0] - 0.5, uv[1] - 0.5);
        let (rx, ry) = (x * c - y * s, x * s + y * c);
        [(rx + 0.5) * self.tiling[0] + self.offset[0], (ry + 0.5) * self.tiling[1] + self.offset[1]]
    }
}

/// A fill layer writes constants or textures. `None` leaves that channel untouched.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FillParams {
    /// Textures that replace the constants below, channel by channel.
    #[serde(default)]
    pub maps: FillMaps,
    /// Placement of those textures.
    #[serde(default)]
    pub uv: UvTransform,
    /// sRGB color.
    pub base: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    /// Height in 0..1 where 0.5 is flat; the normal map is derived from the composited height.
    #[serde(default)]
    pub height: Option<f32>,
}

/// A procedural source for a mask. The template's parameters live in `params` by name;
/// anything missing takes the template's default (see `op-proc`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GeneratorKind {
    /// Convex edges, from the baked curvature, broken up with noise.
    EdgeWear,
    /// Occluded crevices, from baked AO and concave curvature, broken up with noise.
    CreviceDirt,
    /// Up-facing surfaces, from the baked world normal.
    TopDust,
    /// A gradient along the object's up axis, from the baked position.
    HeightGradient,
    /// Plain fractal noise.
    Noise,
    /// An arbitrary graph, usually loaded from a file.
    Custom { name: String, graph: Graph },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Generator {
    pub kind: GeneratorKind,
    pub params: BTreeMap<String, f32>,
    pub invert: bool,
}

impl Generator {
    pub fn new(kind: GeneratorKind) -> Self {
        Self { kind, params: BTreeMap::new(), invert: false }
    }

    pub fn with(mut self, key: &str, value: f32) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    pub fn inverted(mut self) -> Self {
        self.invert = true;
        self
    }

    /// Reject anything that did not come from our own code before it is evaluated.
    pub fn validate(&self) -> Result<(), String> {
        if self.params.len() > 32 {
            return Err("too many generator parameters".into());
        }
        for (k, v) in &self.params {
            if k.is_empty() || k.len() > 32 {
                return Err(format!("parameter name '{k}' has an unusable length"));
            }
            if !v.is_finite() {
                return Err(format!("parameter '{k}' is not a finite number"));
            }
        }
        if let GeneratorKind::Custom { name, graph } = &self.kind {
            if name.len() > 128 {
                return Err("generator name is too long".into());
            }
            graph.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub enabled: bool,
    /// Initial content: white reveals everything, black hides everything.
    pub init_white: bool,
    /// When set the mask content is computed from the generator and cannot be painted
    /// until it is frozen (the generator removed).
    #[serde(default)]
    pub generator: Option<Generator>,
    /// Levels applied to the mask's value where it is used (painting still edits the raw mask).
    #[serde(default)]
    pub adjust: MaskAdjust,
}

impl Mask {
    pub fn new(init_white: bool) -> Self {
        Self { enabled: true, init_white, generator: None, adjust: MaskAdjust::default() }
    }

    pub fn generated(generator: Generator) -> Self {
        Self { enabled: true, init_white: true, generator: Some(generator), adjust: MaskAdjust::default() }
    }
}

/// Hue, saturation, brightness and contrast of a layer's base colour, applied in sRGB space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorAdjust {
    /// Degrees around the colour wheel, -180..180.
    pub hue: f32,
    /// Multiplier: 0 is grey, 1 unchanged, 2 doubles the colourfulness.
    pub saturation: f32,
    /// Added to every channel, -1..1.
    pub brightness: f32,
    /// Multiplier around mid grey: 1 unchanged.
    pub contrast: f32,
}

impl Default for ColorAdjust {
    fn default() -> Self {
        Self { hue: 0.0, saturation: 1.0, brightness: 0.0, contrast: 1.0 }
    }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

impl ColorAdjust {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// The same value with anything non-finite replaced and everything clamped to its range.
    pub fn sanitized(self) -> Self {
        let f = |v: f32, d: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        Self { hue: f(self.hue, 0.0, -360.0, 360.0), saturation: f(self.saturation, 1.0, 0.0, 4.0), brightness: f(self.brightness, 0.0, -1.0, 1.0), contrast: f(self.contrast, 1.0, 0.0, 4.0) }
    }

    /// Adjust a linear RGB colour. The CPU twin of the compositor's `adjust_color`.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let a = self.sanitized();
        let s = rgb.map(|c| linear_to_srgb(c.clamp(0.0, 1.0)));
        let (max, min) = (s[0].max(s[1]).max(s[2]), s[0].min(s[1]).min(s[2]));
        let d = max - min;
        let mut h = if d <= 1e-6 {
            0.0
        } else if max == s[0] {
            ((s[1] - s[2]) / d).rem_euclid(6.0)
        } else if max == s[1] {
            (s[2] - s[0]) / d + 2.0
        } else {
            (s[0] - s[1]) / d + 4.0
        };
        let sat = if max <= 1e-6 { 0.0 } else { d / max };
        h = (h + a.hue / 60.0).rem_euclid(6.0);
        let sat = (sat * a.saturation).clamp(0.0, 1.0);
        let k = |n: f32| {
            let k = (n + h).rem_euclid(6.0);
            max - max * sat * k.min(4.0 - k).clamp(0.0, 1.0)
        };
        let hsv = [k(5.0), k(3.0), k(1.0)];
        hsv.map(|c| srgb_to_linear(((c - 0.5) * a.contrast + 0.5 + a.brightness).clamp(0.0, 1.0)))
    }
}

/// Levels and invert for a mask: the input range `black..white` is stretched to 0..1, `gamma` > 1
/// brightens the midtones, and `invert` flips the result.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskAdjust {
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
    pub invert: bool,
}

impl Default for MaskAdjust {
    fn default() -> Self {
        Self { black: 0.0, white: 1.0, gamma: 1.0, invert: false }
    }
}

impl MaskAdjust {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// The adjusted value of a mask sample in 0..1. Nonsense settings (NaN, an empty range) fall
    /// back to something harmless instead of poisoning the render.
    pub fn apply(&self, v: f32) -> f32 {
        let black = if self.black.is_finite() { self.black.clamp(0.0, 1.0) } else { 0.0 };
        let white = if self.white.is_finite() { self.white.clamp(0.0, 1.0) } else { 1.0 };
        let gamma = if self.gamma.is_finite() { self.gamma.clamp(0.1, 10.0) } else { 1.0 };
        let span = (white - black).max(1e-4);
        let stretched = ((v - black) / span).clamp(0.0, 1.0);
        let out = stretched.powf(1.0 / gamma);
        if self.invert { 1.0 - out } else { out }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Owns paintable base/metal/rough textures.
    Paint,
    Fill(FillParams),
    /// Children are composited in isolation, then blended as one layer.
    Group { children: Vec<Node> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub mask: Option<Mask>,
    pub kind: NodeKind,
    /// Colour correction of what this layer (or group) puts down.
    #[serde(default)]
    pub adjust: ColorAdjust,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub nodes: Vec<Node>,
    next_id: NodeId,
    /// How strongly painted height bends the normal map (1.0 is the default look).
    #[serde(default = "default_height_strength")]
    pub height_strength: f32,
}

fn default_height_strength() -> f32 {
    1.0
}

impl Default for Document {
    fn default() -> Self {
        Self { nodes: Vec::new(), next_id: 1, height_strength: 1.0 }
    }
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn find_in(nodes: &[Node], id: NodeId) -> Option<&Node> {
    for n in nodes {
        if n.id == id {
            return Some(n);
        }
        if let NodeKind::Group { children } = &n.kind {
            if let Some(f) = find_in(children, id) {
                return Some(f);
            }
        }
    }
    None
}

fn find_in_mut(nodes: &mut [Node], id: NodeId) -> Option<&mut Node> {
    for n in nodes {
        if n.id == id {
            return Some(n);
        }
        if let NodeKind::Group { children } = &mut n.kind {
            if let Some(f) = find_in_mut(children, id) {
                return Some(f);
            }
        }
    }
    None
}

/// The sibling list that contains `id`, plus its index in it.
fn list_of_mut(nodes: &mut Vec<Node>, id: NodeId) -> Option<(&mut Vec<Node>, usize)> {
    if let Some(i) = nodes.iter().position(|n| n.id == id) {
        return Some((nodes, i));
    }
    for n in nodes.iter_mut() {
        if let NodeKind::Group { children } = &mut n.kind {
            if let Some(r) = list_of_mut(children, id) {
                return Some(r);
            }
        }
    }
    None
}

fn find_in_node(node: &Node, id: NodeId) -> bool {
    match &node.kind {
        NodeKind::Group { children } => children.iter().any(|c| c.id == id || find_in_node(c, id)),
        _ => false,
    }
}

fn collect_preorder(node: &Node, out: &mut Vec<NodeId>) {
    out.push(node.id);
    if let NodeKind::Group { children } = &node.kind {
        for c in children {
            collect_preorder(c, out);
        }
    }
}

fn collect_ids(nodes: &[Node], out: &mut Vec<NodeId>) {
    for n in nodes {
        out.push(n.id);
        if let NodeKind::Group { children } = &n.kind {
            collect_ids(children, out);
        }
    }
}

impl Document {
    /// A fresh document: one orange fill layer, like a default material.
    pub fn with_default_material() -> Self {
        let mut d = Self::default();
        d.add(
            None,
            None,
            "Base material",
            NodeKind::Fill(FillParams { base: Some([0.80, 0.55, 0.30]), metallic: Some(0.0), roughness: Some(0.45), ..Default::default() }),
        );
        d
    }

    pub fn find(&self, id: NodeId) -> Option<&Node> {
        find_in(&self.nodes, id)
    }

    pub fn find_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        find_in_mut(&mut self.nodes, id)
    }

    /// Ids of every texture any fill layer uses, without repeats, in layer order.
    pub fn texture_ids(&self) -> Vec<String> {
        fn walk(nodes: &[Node], out: &mut Vec<String>) {
            for n in nodes {
                match &n.kind {
                    NodeKind::Fill(f) => {
                        for (_, t) in f.maps.iter() {
                            if !out.contains(&t.id) {
                                out.push(t.id.clone());
                            }
                        }
                    }
                    NodeKind::Group { children } => walk(children, out),
                    NodeKind::Paint => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.nodes, &mut out);
        out
    }

    pub fn ids(&self) -> Vec<NodeId> {
        let mut v = Vec::new();
        collect_ids(&self.nodes, &mut v);
        v
    }

    /// Parent group (None = top level) and index within its sibling list.
    pub fn locate(&self, id: NodeId) -> Option<(Option<NodeId>, usize)> {
        fn go(nodes: &[Node], parent: Option<NodeId>, id: NodeId) -> Option<(Option<NodeId>, usize)> {
            for (i, n) in nodes.iter().enumerate() {
                if n.id == id {
                    return Some((parent, i));
                }
                if let NodeKind::Group { children } = &n.kind {
                    if let Some(r) = go(children, Some(n.id), id) {
                        return Some(r);
                    }
                }
            }
            None
        }
        go(&self.nodes, None, id)
    }

    /// Add a node into `parent` (None = top level) at `index` (None = on top).
    pub fn add(&mut self, parent: Option<NodeId>, index: Option<usize>, name: &str, kind: NodeKind) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = Node {
            id,
            name: name.to_string(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            mask: None,
            kind,
            adjust: ColorAdjust::default(),
        };
        let list = match parent.and_then(|p| self.find_mut(p)) {
            Some(Node { kind: NodeKind::Group { children }, .. }) => children,
            _ => &mut self.nodes,
        };
        let at = index.unwrap_or(list.len()).min(list.len());
        list.insert(at, node);
        id
    }

    /// A copy of the node and everything under it.
    pub fn extract(&self, id: NodeId) -> Option<Node> {
        self.find(id).cloned()
    }

    fn reassign_ids(&mut self, node: &mut Node) {
        node.id = self.next_id;
        self.next_id += 1;
        if let NodeKind::Group { children } = &mut node.kind {
            for c in children {
                self.reassign_ids(c);
            }
        }
    }

    /// Insert copies of `nodes` (bottom to top) with fresh ids, as `add` would. Returns the ids
    /// of the inserted top-level nodes.
    pub fn insert_nodes(&mut self, parent: Option<NodeId>, index: Option<usize>, mut nodes: Vec<Node>) -> Vec<NodeId> {
        for n in &mut nodes {
            self.reassign_ids(n);
        }
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let list = match parent.and_then(|p| self.find_mut(p)) {
            Some(Node { kind: NodeKind::Group { children }, .. }) => children,
            _ => &mut self.nodes,
        };
        let at = index.unwrap_or(list.len()).min(list.len());
        list.splice(at..at, nodes);
        ids
    }

    pub fn remove(&mut self, id: NodeId) -> Option<Node> {
        let (list, i) = list_of_mut(&mut self.nodes, id)?;
        Some(list.remove(i))
    }

    /// True if `id` is `ancestor` itself or lies anywhere beneath it.
    pub fn is_within(&self, ancestor: NodeId, id: NodeId) -> bool {
        match self.find(ancestor) {
            Some(n) => ancestor == id || find_in_node(n, id),
            None => false,
        }
    }

    /// Move a node (with everything under it) to a drop slot: into `parent`'s children (`None` = top
    /// level), *before the element currently at `index`* (`index == len` appends, i.e. the top of
    /// the stack). Returns false, changing nothing, if the move is impossible: unknown ids, a
    /// parent that is not a group, or a parent inside the moved node.
    pub fn move_to(&mut self, id: NodeId, parent: Option<NodeId>, index: usize) -> bool {
        if self.find(id).is_none() {
            return false;
        }
        if let Some(p) = parent {
            match self.find(p) {
                Some(Node { kind: NodeKind::Group { .. }, .. }) => {}
                _ => return false,
            }
            if self.is_within(id, p) {
                return false; // would put a group inside itself
            }
        }
        let Some((old_parent, old_index)) = self.locate(id) else { return false };
        let mut at = index;
        if old_parent == parent && old_index < index {
            at -= 1; // removing the node first shifts everything after it down by one
        }
        let node = self.remove(id).expect("located above");
        let list = match parent.and_then(|p| self.find_mut(p)) {
            Some(Node { kind: NodeKind::Group { children }, .. }) => children,
            _ => &mut self.nodes,
        };
        list.insert(at.min(list.len()), node);
        true
    }

    /// Copy a node and everything under it, placed just above the original. Returns the copy's id
    /// and the (original, copy) id pairs of every node in the subtree, in pre-order, so callers can
    /// copy per-node data (such as paint) too.
    pub fn duplicate(&mut self, id: NodeId) -> Option<(NodeId, Vec<(NodeId, NodeId)>)> {
        let (parent, index) = self.locate(id)?;
        let mut copy = self.find(id)?.clone();
        let mut old_ids = Vec::new();
        collect_preorder(&copy, &mut old_ids);
        self.reassign_ids(&mut copy);
        let mut new_ids = Vec::new();
        collect_preorder(&copy, &mut new_ids);
        copy.name = format!("{} copy", copy.name);
        let top = copy.id;
        self.insert_nodes_raw(parent, index + 1, copy);
        Some((top, old_ids.into_iter().zip(new_ids).collect()))
    }

    /// Replace a node by a new group containing it, in the same position.
    pub fn wrap_in_group(&mut self, id: NodeId) -> Option<NodeId> {
        let (parent, index) = self.locate(id)?;
        let node = self.remove(id)?;
        let group = self.add(parent, Some(index), "Group", NodeKind::Group { children: vec![node] });
        Some(group)
    }

    fn insert_nodes_raw(&mut self, parent: Option<NodeId>, index: usize, node: Node) {
        let list = match parent.and_then(|p| self.find_mut(p)) {
            Some(Node { kind: NodeKind::Group { children }, .. }) => children,
            _ => &mut self.nodes,
        };
        list.insert(index.min(list.len()), node);
    }

    /// Move one step toward the top of its sibling list.
    pub fn move_up(&mut self, id: NodeId) -> bool {
        match list_of_mut(&mut self.nodes, id) {
            Some((list, i)) if i + 1 < list.len() => {
                list.swap(i, i + 1);
                true
            }
            _ => false,
        }
    }

    pub fn move_down(&mut self, id: NodeId) -> bool {
        match list_of_mut(&mut self.nodes, id) {
            Some((list, i)) if i > 0 => {
                list.swap(i, i - 1);
                true
            }
            _ => false,
        }
    }

    /// Move into the group directly above it in the sibling list (as that group's bottom child).
    pub fn indent(&mut self, id: NodeId) -> bool {
        let Some((list, i)) = list_of_mut(&mut self.nodes, id) else { return false };
        if i + 1 >= list.len() || !matches!(list[i + 1].kind, NodeKind::Group { .. }) {
            return false;
        }
        let node = list.remove(i);
        if let NodeKind::Group { children } = &mut list[i].kind {
            children.insert(0, node);
        }
        true
    }

    /// Move out of its parent group, to just below that group.
    pub fn outdent(&mut self, id: NodeId) -> bool {
        let Some((Some(parent), _)) = self.locate(id) else { return false };
        let Some(node) = self.remove(id) else { return false };
        let (list, pi) = list_of_mut(&mut self.nodes, parent).expect("parent exists");
        list.insert(pi, node);
        true
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn colour_adjust_leaves_identity_alone_and_moves_hue_saturation_and_brightness() {
        use super::{linear_to_srgb, ColorAdjust};
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-3);
        let c = [0.6, 0.2, 0.1];
        assert!(near(ColorAdjust::default().apply(c), c), "identity round-trips");
        let red = [1.0, 0.0, 0.0];
        assert!(near(ColorAdjust { hue: 120.0, ..Default::default() }.apply(red), [0.0, 1.0, 0.0]), "red turned a third of the wheel is green");
        assert!(near(ColorAdjust { hue: 360.0, ..Default::default() }.apply(c), c), "a full turn is no change");
        let grey = ColorAdjust { saturation: 0.0, ..Default::default() }.apply(c);
        assert!((grey[0] - grey[1]).abs() < 1e-4 && (grey[1] - grey[2]).abs() < 1e-4, "no saturation is grey: {grey:?}");
        let up = ColorAdjust { brightness: 0.2, ..Default::default() }.apply([0.2, 0.2, 0.2]);
        assert!(linear_to_srgb(up[0]) > linear_to_srgb(0.2) + 0.19, "brightness lifts in sRGB: {up:?}");
        let flat = ColorAdjust { contrast: 0.0, ..Default::default() }.apply(c);
        assert!(near(flat.map(linear_to_srgb), [0.5; 3]), "no contrast is mid grey");
        let junk = ColorAdjust { hue: f32::NAN, saturation: f32::INFINITY, brightness: f32::NAN, contrast: -3.0 };
        assert!(junk.apply(c).iter().all(|v| v.is_finite()), "nonsense settings stay finite");
    }

    use super::*;

    #[test]
    fn baked_map_validates_and_roundtrips_values() {
        assert!(BakedMap::new(2, 2, vec![0; 8]).is_err(), "channel count");
        assert!(BakedMap::new(2, 1, vec![0; 3]).is_err(), "length");
        assert!(BakedMap::new(0, 1, vec![]).is_err(), "size");
        let mut m = BakedMap::filled(2, 3, [0.25, 0.5, 1.0]);
        let t = m.texel(3);
        assert!((t[0] - 0.25).abs() < 1e-4 && (t[1] - 0.5).abs() < 1e-4 && t[2] == 1.0);
        m.set_texel(1, [2.0, -1.0, 0.5]); // out of range values clamp
        assert_eq!(m.texel(1)[..2], [1.0, 0.0]);
        let s = BakedMap::filled(2, 1, [0.5; 3]);
        assert_eq!(s.texel(2)[0], s.texel(2)[2], "scalar maps replicate");
        assert_eq!(MeshMap::AmbientOcclusion.channels(), 1);
        assert_eq!(MeshMap::NormalBase.channels(), 3);
    }

    fn names(nodes: &[Node]) -> Vec<&str> {
        nodes.iter().map(|n| n.name.as_str()).collect()
    }

    fn doc_abc() -> (Document, NodeId, NodeId, NodeId) {
        let mut d = Document::default();
        let a = d.add(None, None, "a", NodeKind::Paint);
        let b = d.add(None, None, "b", NodeKind::Paint);
        let c = d.add(None, None, "c", NodeKind::Group { children: vec![] });
        (d, a, b, c)
    }

    #[test]
    fn add_orders_bottom_to_top_and_ids_are_unique() {
        let (d, a, b, c) = doc_abc();
        assert_eq!(names(&d.nodes), ["a", "b", "c"]);
        assert!(a != b && b != c && a != c);
        assert_eq!(d.ids().len(), 3);
    }

    #[test]
    fn move_up_down_stay_within_bounds() {
        let (mut d, a, _b, c) = doc_abc();
        assert!(d.move_up(a));
        assert_eq!(names(&d.nodes), ["b", "a", "c"]);
        assert!(!d.move_up(c), "already on top");
        assert!(d.move_down(c));
        assert_eq!(names(&d.nodes), ["b", "c", "a"]);
        assert!(d.move_down(a) && !d.move_down(d.nodes[0].id));
    }

    #[test]
    fn indent_requires_group_above_and_outdent_reverses_it() {
        let (mut d, a, b, c) = doc_abc();
        assert!(!d.indent(a), "b is not a group");
        assert!(d.indent(b), "c (group) is directly above b");
        assert_eq!(names(&d.nodes), ["a", "c"]);
        assert_eq!(d.locate(b), Some((Some(c), 0)));
        assert!(d.outdent(b));
        assert_eq!(names(&d.nodes), ["a", "b", "c"]);
        assert!(!d.outdent(b), "already top level");
    }

    #[test]
    fn nested_find_remove_and_locate() {
        let (mut d, _a, b, c) = doc_abc();
        d.indent(b);
        let inner = d.add(Some(c), None, "inner", NodeKind::Paint);
        assert_eq!(d.find(inner).unwrap().name, "inner");
        assert_eq!(d.locate(inner), Some((Some(c), 1)));
        assert_eq!(d.remove(inner).unwrap().id, inner);
        assert!(d.find(inner).is_none());
        assert_eq!(d.ids().len(), 3);
    }

    fn order(d: &Document) -> Vec<String> {
        d.nodes.iter().map(|n| n.name.clone()).collect()
    }

    fn four() -> (Document, [NodeId; 4]) {
        let mut d = Document::default();
        let ids = ["a", "b", "c", "d"].map(|n| d.add(None, None, n, NodeKind::Paint));
        (d, ids)
    }

    #[test]
    fn move_to_a_slot_within_one_list_uses_pre_removal_indices() {
        // slot k means "before the element currently at k"
        let (mut d, [a, _b, _c, dd]) = four();
        assert!(d.move_to(a, None, 3)); // before d
        assert_eq!(order(&d), ["b", "c", "a", "d"]);
        assert!(d.move_to(dd, None, 0)); // to the very bottom
        assert_eq!(order(&d), ["d", "b", "c", "a"]);
        assert!(d.move_to(dd, None, 4)); // append: the very top
        assert_eq!(order(&d), ["b", "c", "a", "d"]);
        // Dropping where it already is changes nothing, whichever side of itself the slot is on.
        let before = order(&d);
        let c = d.nodes[1].id;
        assert!(d.move_to(c, None, 1) && d.move_to(c, None, 2));
        assert_eq!(order(&d), before);
    }

    #[test]
    fn move_to_into_and_out_of_groups() {
        let (mut d, [a, b, c, _d]) = four();
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        assert!(d.move_to(a, Some(g), 0));
        assert!(d.move_to(b, Some(g), 1)); // append above a
        assert_eq!(d.locate(a), Some((Some(g), 0)));
        assert_eq!(d.locate(b), Some((Some(g), 1)));
        assert!(d.move_to(a, Some(g), 2)); // within the group, to the top
        assert_eq!(d.locate(b), Some((Some(g), 0)));
        assert_eq!(d.locate(a), Some((Some(g), 1)));
        assert!(d.move_to(b, None, 0)); // out of the group, to the very bottom
        assert_eq!(d.locate(b), Some((None, 0)));
        assert_eq!(d.find(c).unwrap().name, "c");
        assert_eq!(d.ids().len(), 5, "no node was lost or duplicated");
    }

    #[test]
    fn impossible_moves_change_nothing() {
        let (mut d, [a, b, ..]) = four();
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        let inner = d.add(Some(g), None, "inner", NodeKind::Group { children: vec![] });
        let before = d.clone();
        assert!(!d.move_to(g, Some(g), 0), "a group into itself");
        assert!(!d.move_to(g, Some(inner), 0), "a group into its own descendant");
        assert!(!d.move_to(a, Some(b), 0), "a paint layer is not a container");
        assert!(!d.move_to(999, None, 0), "unknown node");
        assert!(!d.move_to(a, Some(999), 0), "unknown parent");
        assert_eq!(d, before);
        assert!(d.move_to(inner, None, 0), "but a child may leave its group");
    }

    #[test]
    fn move_to_clamps_absurd_indices_and_keeps_ids_and_contents() {
        let (mut d, [a, ..]) = four();
        d.find_mut(a).unwrap().opacity = 0.25;
        assert!(d.move_to(a, None, usize::MAX / 2));
        assert_eq!(d.nodes.last().unwrap().id, a);
        assert_eq!(d.find(a).unwrap().opacity, 0.25);
    }

    #[test]
    fn is_within_follows_the_tree() {
        let mut d = Document::default();
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        let h = d.add(Some(g), None, "h", NodeKind::Group { children: vec![] });
        let leaf = d.add(Some(h), None, "leaf", NodeKind::Paint);
        let other = d.add(None, None, "other", NodeKind::Paint);
        assert!(d.is_within(g, leaf) && d.is_within(g, h) && d.is_within(g, g));
        assert!(!d.is_within(leaf, g) && !d.is_within(g, other) && !d.is_within(h, other));
        assert!(!d.is_within(999, leaf));
    }

    #[test]
    fn duplicate_copies_a_subtree_above_the_original_with_fresh_ids() {
        let mut d = Document::default();
        let base = d.add(None, None, "base", NodeKind::Paint);
        let g = d.add(None, None, "grp", NodeKind::Group { children: vec![] });
        let x = d.add(Some(g), None, "x", NodeKind::Paint);
        let y = d.add(Some(g), None, "y", NodeKind::Fill(FillParams { base: Some([0.1, 0.2, 0.3]), ..Default::default() }));
        d.find_mut(y).unwrap().mask = Some(Mask::generated(Generator::new(GeneratorKind::EdgeWear).with("amount", 0.9)));
        let (copy, pairs) = d.duplicate(g).unwrap();
        assert_eq!(d.locate(copy), Some((None, 2)), "directly above the original");
        assert_eq!(d.find(copy).unwrap().name, "grp copy");
        assert_eq!(pairs.len(), 3, "group and both children");
        assert_eq!(pairs[0].0, g);
        let ids = d.ids();
        assert_eq!(ids.len(), 7);
        assert_eq!(ids.iter().collect::<std::collections::HashSet<_>>().len(), 7, "ids are unique");
        // children are copied in order with their settings, masks and generators
        let NodeKind::Group { children } = &d.find(copy).unwrap().kind else { panic!() };
        assert_eq!(children.len(), 2);
        assert_eq!((pairs[1].0, pairs[2].0), (x, y));
        assert_eq!((children[0].id, children[1].id), (pairs[1].1, pairs[2].1));
        assert_eq!(children[1].mask, d.find(y).unwrap().mask);
        assert!(d.duplicate(999).is_none());
        let _ = base;
    }

    #[test]
    fn duplicating_inside_a_group_stays_in_the_group() {
        let mut d = Document::default();
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        let a = d.add(Some(g), None, "a", NodeKind::Paint);
        let top = d.add(Some(g), None, "top", NodeKind::Paint);
        let (copy, _) = d.duplicate(a).unwrap();
        assert_eq!(d.locate(copy), Some((Some(g), 1)));
        assert_eq!(d.locate(top), Some((Some(g), 2)), "the layer above is pushed up, not overwritten");
    }

    #[test]
    fn wrap_in_group_keeps_the_position() {
        let (mut d, [_a, b, ..]) = four();
        let g = d.wrap_in_group(b).unwrap();
        assert_eq!(d.locate(g), Some((None, 1)));
        assert_eq!(d.locate(b), Some((Some(g), 0)));
        assert_eq!(order(&d), ["a", "Group", "c", "d"]);
        assert!(d.wrap_in_group(999).is_none());
    }

    #[test]
    fn uv_transforms_scale_shift_and_turn_around_the_middle() {
        let id = UvTransform::default();
        assert!(id.is_identity() && id.is_sane());
        assert_eq!(id.apply([0.25, 0.75]), [0.25, 0.75]);
        let tiled = UvTransform { tiling: [4.0, 2.0], offset: [0.5, 0.0], rotation: 0.0 };
        assert_eq!(tiled.apply([0.5, 0.5]), [2.5, 1.0]);
        // A quarter turn about the centre sends the point right of centre to above it.
        let quarter = UvTransform { rotation: 90.0, ..Default::default() };
        let p = quarter.apply([1.0, 0.5]);
        assert!((p[0] - 0.5).abs() < 1e-5 && (p[1] - 1.0).abs() < 1e-5, "{p:?}");
        // The centre never moves under rotation alone.
        assert!({ let c = quarter.apply([0.5, 0.5]); (c[0] - 0.5).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6 });
        for bad in [
            UvTransform { tiling: [0.0, 1.0], ..Default::default() },
            UvTransform { tiling: [f32::NAN, 1.0], ..Default::default() },
            UvTransform { offset: [1e9, 0.0], ..Default::default() },
            UvTransform { rotation: f32::INFINITY, ..Default::default() },
        ] {
            assert!(!bad.is_sane(), "{bad:?}");
        }
        assert!(UvTransform { tiling: [-2.0, 3.0], ..Default::default() }.is_sane(), "negative tiling mirrors");
    }

    #[test]
    fn documents_without_textures_still_load_and_texture_ids_are_listed_once() {
        // A fill layer written before textures existed has no `maps` or `uv`.
        let old = r#"{"base":[0.5,0.5,0.5],"metallic":null,"roughness":0.4}"#;
        let f: FillParams = serde_json::from_str(old).unwrap();
        assert!(f.maps.is_empty() && f.uv.is_identity());
        let mut d = Document::default();
        let id = "0123456789abcdef0123456789abcdef";
        let mut a = FillParams::default();
        a.maps.base = Some(TexRef::new(id));
        a.maps.normal = Some(TexRef { flip_green: true, ..TexRef::new("ffffffffffffffffffffffffffffffff") });
        d.add(None, None, "a", NodeKind::Fill(a.clone()));
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        d.add(Some(g), None, "b", NodeKind::Fill(a));
        d.add(None, None, "p", NodeKind::Paint);
        assert_eq!(d.texture_ids(), [id, "ffffffffffffffffffffffffffffffff"]);
        assert!(is_valid_texture_id(id) && !is_valid_texture_id("../x") && !is_valid_texture_id(&id.to_uppercase()));
        // The whole thing survives a round trip.
        let back: Document = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn slots_map_to_their_texture() {
        let mut m = FillMaps::default();
        for (i, slot) in Slot::ALL.into_iter().enumerate() {
            assert!(m.get(slot).is_none());
            m.set(slot, Some(TexRef::new(format!("{i:032x}"))));
        }
        assert_eq!(m.iter().count(), 5);
        assert_eq!(m.get(Slot::Height).unwrap().id, format!("{:032x}", 3));
        m.set(Slot::Height, None);
        assert_eq!(m.iter().count(), 4);
        assert!(!m.is_empty());
    }

    #[test]
    fn clone_equality_detects_edits() {
        let (mut d, a, ..) = doc_abc();
        let before = d.clone();
        assert_eq!(d, before);
        d.find_mut(a).unwrap().opacity = 0.5;
        assert_ne!(d, before);
    }
}
