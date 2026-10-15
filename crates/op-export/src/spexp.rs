//! Reader for Substance Painter export-preset files (`.spexp`).
//!
//! The format was worked out from sample files, not from documentation, so the reader is
//! strict: anything it does not understand is an error, never a guess.
//!
//! Container (little endian unless noted). Magic `DD 2F 7C 1B`, then two words
//! (`version`, `revision`). Two layouts exist:
//!
//! * **version 0** is self-describing. Values carry a tag byte: `09` 8-byte int,
//!   `0C` 8-byte uid, `0A` bool, `10` length-prefixed bytes, `11`/`13` list
//!   (`size`, `count`, items), `12`/`14` object (`00`, `size`, class name, u16 field count,
//!   then `name, value` pairs).
//! * **version 1** stores a class table: an object is `[size][class ref]`; the reference
//!   `FFFFFFFF` introduces a class (name, field count, then `name, type` pairs) and any
//!   other value indexes the table (0-based, the root class first). Values are untyped
//!   and follow the schema: `09` 4-byte int, `0A` 1-byte bool, `0C` 8-byte uid, `10`
//!   length-prefixed bytes, `13` list of sized objects, `11` list of unsized objects. In
//!   a `11` list only the first item has a class reference.
//!
//! A channel's `mapId` is a Qt `QVariant` serialised big-endian with `QDataStream`
//! (an integer, or a UTF-16 string for named maps).

use serde::{Deserialize, Serialize};

const MAGIC: [u8; 4] = [0xdd, 0x2f, 0x7c, 0x1b];
const MAX_NODES: usize = 100_000;
const MAX_DEPTH: usize = 8;
const MAX_NAME: usize = 256;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpexpError {
    #[error("not a Substance Painter export preset (bad signature)")]
    BadMagic,
    #[error("unsupported preset container version {0}")]
    UnsupportedVersion(u32),
    #[error("file ends unexpectedly")]
    Truncated,
    #[error("malformed preset: {0}")]
    Malformed(String),
}

fn malformed<T>(msg: impl Into<String>) -> Result<T, SpexpError> {
    Err(SpexpError::Malformed(msg.into()))
}

/// Id of the map a channel reads from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MapId {
    Int(i64),
    Str(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpexpChannel {
    /// 0 R, 1 G, 2 B, 3 A, 4 L (luminance / gray).
    pub src: i64,
    pub map: MapId,
    /// 0 constant, 1 document channel, 2 named (virtual) map, 3 baked mesh map.
    pub map_type: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpexpMap {
    pub bitdepth: i64,
    pub dithering: bool,
    pub file_format: String,
    pub path: String,
    pub channels: Vec<SpexpChannel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpexpPreset {
    pub container_version: u32,
    pub maps: Vec<SpexpMap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Val {
    Int(i64),
    Uid(u64),
    Bool(bool),
    Bytes(Vec<u8>),
    List(Vec<Obj>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Obj {
    class: String,
    fields: Vec<(String, Val)>,
}

impl Obj {
    fn get(&self, name: &str) -> Result<&Val, SpexpError> {
        self.fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| SpexpError::Malformed(format!("{} has no field '{name}'", self.class)))
    }

    fn int(&self, name: &str) -> Result<i64, SpexpError> {
        match self.get(name)? {
            Val::Int(v) => Ok(*v),
            _ => malformed(format!("field '{name}' is not an integer")),
        }
    }

    fn text(&self, name: &str) -> Result<String, SpexpError> {
        match self.get(name)? {
            Val::Bytes(b) => String::from_utf8(b.clone()).or_else(|_| malformed(format!("field '{name}' is not valid text"))),
            _ => malformed(format!("field '{name}' is not text")),
        }
    }

    fn list(&self, name: &str) -> Result<&[Obj], SpexpError> {
        match self.get(name)? {
            Val::List(l) => Ok(l),
            _ => malformed(format!("field '{name}' is not a list")),
        }
    }
}

struct Reader<'a> {
    d: &'a [u8],
    i: usize,
    nodes: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.d.len() - self.i
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SpexpError> {
        if n > self.remaining() {
            return Err(SpexpError::Truncated);
        }
        let s = &self.d[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, SpexpError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SpexpError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, SpexpError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, SpexpError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn name(&mut self) -> Result<String, SpexpError> {
        let n = self.u32()? as usize;
        if n > MAX_NAME {
            return malformed("identifier is implausibly long");
        }
        String::from_utf8(self.take(n)?.to_vec()).or_else(|_| malformed("identifier is not valid text"))
    }

    fn bytes(&mut self) -> Result<Vec<u8>, SpexpError> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }

    fn node(&mut self) -> Result<(), SpexpError> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return malformed("preset is implausibly large");
        }
        Ok(())
    }

    fn count(&mut self) -> Result<usize, SpexpError> {
        let n = self.u32()? as usize;
        if n > MAX_NODES {
            return malformed("list is implausibly long");
        }
        Ok(n)
    }
}

// ---- version 0 ---------------------------------------------------------------------------

fn v0_value(r: &mut Reader, depth: usize) -> Result<Val, SpexpError> {
    if depth > MAX_DEPTH {
        return malformed("nesting is too deep");
    }
    r.node()?;
    match r.u8()? {
        0x09 => Ok(Val::Int(r.u64()? as i64)),
        0x0c => Ok(Val::Uid(r.u64()?)),
        0x0a => Ok(Val::Bool(r.u8()? != 0)),
        0x10 => Ok(Val::Bytes(r.bytes()?)),
        0x11 | 0x13 => {
            let _size = r.u32()?;
            let n = r.count()?;
            let mut items = Vec::new();
            for _ in 0..n {
                r.node()?;
                match r.u8()? {
                    0x12 | 0x14 => items.push(v0_object(r, depth + 1)?),
                    t => return malformed(format!("list item has tag {t:#04x}, expected an object")),
                }
            }
            Ok(Val::List(items))
        }
        t => malformed(format!("unknown value tag {t:#04x}")),
    }
}

/// Reads an object after its tag byte.
fn v0_object(r: &mut Reader, depth: usize) -> Result<Obj, SpexpError> {
    if depth > MAX_DEPTH {
        return malformed("nesting is too deep");
    }
    if r.u8()? != 0 {
        return malformed("object header is not recognised");
    }
    let _size = r.u32()?;
    let class = r.name()?;
    let n = r.u16()? as usize;
    let mut fields = Vec::with_capacity(n.min(64));
    for _ in 0..n {
        let name = r.name()?;
        fields.push((name, v0_value(r, depth + 1)?));
    }
    Ok(Obj { class, fields })
}

fn decode_v0(r: &mut Reader) -> Result<Obj, SpexpError> {
    r.node()?;
    match r.u8()? {
        0x12 => v0_object(r, 0),
        _ => malformed("the file does not start with an object"),
    }
}

// ---- version 1 ---------------------------------------------------------------------------

struct Class {
    name: String,
    fields: Vec<(String, u32)>,
}

struct V1<'a, 'b> {
    r: &'b mut Reader<'a>,
    classes: Vec<std::rc::Rc<Class>>,
}

impl V1<'_, '_> {
    fn value(&mut self, ty: u32, depth: usize) -> Result<Val, SpexpError> {
        if depth > MAX_DEPTH {
            return malformed("nesting is too deep");
        }
        self.r.node()?;
        match ty {
            0x09 => Ok(Val::Int(i32::from_le_bytes(self.r.take(4)?.try_into().unwrap()) as i64)),
            0x0a => Ok(Val::Bool(self.r.u8()? != 0)),
            0x0c => Ok(Val::Uid(self.r.u64()?)),
            0x10 => Ok(Val::Bytes(self.r.bytes()?)),
            0x11 | 0x13 => {
                let n = self.r.count()?;
                let mut items = Vec::new();
                let mut class: Option<std::rc::Rc<Class>> = None;
                for k in 0..n {
                    if ty == 0x13 {
                        let _size = self.r.u32()?;
                        let c = self.class_ref()?;
                        items.push(self.body(&c, depth)?);
                    } else if k == 0 {
                        let c = self.class_ref()?;
                        items.push(self.body(&c, depth)?);
                        class = Some(c);
                    } else {
                        let c = class.clone().ok_or_else(|| SpexpError::Malformed("list without a class".into()))?;
                        items.push(self.body(&c, depth)?);
                    }
                }
                Ok(Val::List(items))
            }
            t => malformed(format!("unknown value type {t:#04x}")),
        }
    }

    /// Read a class reference; a new definition is added to the table.
    fn class_ref(&mut self) -> Result<std::rc::Rc<Class>, SpexpError> {
        let reference = self.r.u32()?;
        if reference == 0xffff_ffff {
            let name = self.r.name()?;
            let n = self.r.u32()? as usize;
            if n > 1024 {
                return malformed("class has too many fields");
            }
            let mut fields = Vec::with_capacity(n);
            for _ in 0..n {
                let fname = self.r.name()?;
                let ty = self.r.u32()?;
                fields.push((fname, ty));
            }
            let class = std::rc::Rc::new(Class { name, fields });
            self.classes.push(class.clone());
            Ok(class)
        } else {
            self.classes
                .get(reference as usize)
                .cloned()
                .ok_or_else(|| SpexpError::Malformed(format!("reference to undefined class {reference}")))
        }
    }

    fn body(&mut self, class: &Class, depth: usize) -> Result<Obj, SpexpError> {
        let mut fields = Vec::with_capacity(class.fields.len());
        for (name, ty) in &class.fields {
            fields.push((name.clone(), self.value(*ty, depth + 1)?));
        }
        Ok(Obj { class: class.name.clone(), fields })
    }
}

fn decode_v1(r: &mut Reader) -> Result<Obj, SpexpError> {
    let marker = r.u32()?;
    let _word = r.u32()?;
    if marker != 0x12 {
        return malformed("the file does not start with an object");
    }
    let _size = r.u32()?;
    let mut v1 = V1 { r, classes: Vec::new() };
    let class = v1.class_ref()?;
    v1.body(&class, 0)
}

// ---- QVariant ------------------------------------------------------------------------------

fn decode_map_id(b: &[u8]) -> Result<MapId, SpexpError> {
    let be32 = |s: &[u8]| u32::from_be_bytes(s.try_into().unwrap());
    if b.len() < 5 {
        return malformed("map id is truncated");
    }
    let ty = be32(&b[0..4]);
    let payload = &b[5..];
    let int = |v: i64, len: usize| {
        if payload.len() == len { Ok(MapId::Int(v)) } else { malformed("map id has trailing bytes") }
    };
    match ty {
        // QMetaType::Int / UInt
        2 | 3 if payload.len() == 4 => {
            let raw = be32(payload);
            int(if ty == 2 { raw as i32 as i64 } else { raw as i64 }, 4)
        }
        // LongLong / ULongLong
        4 | 5 if payload.len() == 8 => {
            let raw = u64::from_be_bytes(payload.try_into().unwrap());
            if ty == 5 && raw > i64::MAX as u64 {
                return malformed("map id is out of range");
            }
            int(raw as i64, 8)
        }
        // QString: u32 byte length (0xFFFFFFFF = null), UTF-16BE
        10 if payload.len() >= 4 => {
            let len = be32(&payload[0..4]);
            if len == 0xffff_ffff {
                return Ok(MapId::Str(String::new()));
            }
            let len = len as usize;
            if len % 2 != 0 || payload.len() != 4 + len {
                return malformed("map id string has the wrong length");
            }
            let units: Vec<u16> = payload[4..].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
            String::from_utf16(&units).map(MapId::Str).or_else(|_| malformed("map id is not valid UTF-16"))
        }
        t => malformed(format!("unsupported map id type {t}")),
    }
}

fn extract(root: &Obj, container_version: u32) -> Result<SpexpPreset, SpexpError> {
    if root.class != "DataExportPreset" {
        return malformed(format!("expected a DataExportPreset, found {}", root.class));
    }
    let mut maps = Vec::new();
    for m in root.list("maps")? {
        if m.class != "DataExportMap" {
            return malformed(format!("expected a DataExportMap, found {}", m.class));
        }
        let mut channels = Vec::new();
        for c in m.list("channels")? {
            if c.class != "DataExportChannel" {
                return malformed(format!("expected a DataExportChannel, found {}", c.class));
            }
            let map = match c.get("mapId")? {
                Val::Bytes(b) => decode_map_id(b)?,
                _ => return malformed("mapId is not a value blob"),
            };
            channels.push(SpexpChannel { src: c.int("channelSrc")?, map, map_type: c.int("mapIdType")? });
        }
        let dithering = match m.get("dithering")? {
            Val::Bool(b) => *b,
            _ => return malformed("dithering is not a flag"),
        };
        maps.push(SpexpMap {
            bitdepth: m.int("bitdepth")?,
            dithering,
            file_format: m.text("fileformat")?,
            path: m.text("path")?,
            channels,
        });
    }
    Ok(SpexpPreset { container_version, maps })
}

/// Decode a `.spexp` file. Never panics on malformed input.
pub fn decode_spexp(bytes: &[u8]) -> Result<SpexpPreset, SpexpError> {
    let mut r = Reader { d: bytes, i: 0, nodes: 0 };
    if r.take(4).map_err(|_| SpexpError::BadMagic)? != MAGIC {
        return Err(SpexpError::BadMagic);
    }
    let version = r.u32()?;
    let _revision = r.u32()?;
    let root = match version {
        0 => decode_v0(&mut r)?,
        1 => decode_v1(&mut r)?,
        v => return Err(SpexpError::UnsupportedVersion(v)),
    };
    if r.remaining() != 0 {
        return malformed(format!("{} unexpected trailing bytes", r.remaining()));
    }
    extract(&root, version)
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Byte builders that mirror the documented layouts, used to test the reader.
    use super::*;

    pub fn name(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u32).to_le_bytes().to_vec();
        v.extend(s.as_bytes());
        v
    }

    pub fn variant_u64(v: u64) -> Vec<u8> {
        let mut b = 5u32.to_be_bytes().to_vec();
        b.push(0);
        b.extend(v.to_be_bytes());
        b
    }

    pub fn variant_i32(v: i32) -> Vec<u8> {
        let mut b = 2u32.to_be_bytes().to_vec();
        b.push(0);
        b.extend(v.to_be_bytes());
        b
    }

    pub fn variant_str(s: &str) -> Vec<u8> {
        let utf16: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_be_bytes()).collect();
        let mut b = 10u32.to_be_bytes().to_vec();
        b.push(0);
        b.extend((utf16.len() as u32).to_be_bytes());
        b.extend(utf16);
        b
    }

    /// One logical channel for the builders below.
    pub struct Ch {
        pub src: i64,
        pub id: Vec<u8>,
        pub ty: i64,
    }

    pub struct Map<'a> {
        pub bitdepth: i64,
        pub dithering: bool,
        pub format: &'a str,
        pub path: &'a str,
        pub channels: Vec<Ch>,
    }

    // version 0 ----------------------------------------------------------------------------

    fn v0_int(v: i64) -> Vec<u8> {
        let mut b = vec![0x09];
        b.extend(v.to_le_bytes());
        b
    }

    fn v0_bytes(data: &[u8]) -> Vec<u8> {
        let mut b = vec![0x10];
        b.extend((data.len() as u32).to_le_bytes());
        b.extend(data);
        b
    }

    fn v0_object(tag: u8, class: &str, fields: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
        let mut body = name(class);
        body.extend((fields.len() as u16).to_le_bytes());
        for (k, v) in fields {
            body.extend(name(k));
            body.extend(v);
        }
        let mut b = vec![tag, 0];
        b.extend((body.len() as u32).to_le_bytes()); // size is ignored by the reader
        b.extend(body);
        b
    }

    fn v0_list(tag: u8, items: Vec<Vec<u8>>) -> Vec<u8> {
        let payload: Vec<u8> = items.iter().flatten().copied().collect();
        let mut b = vec![tag];
        b.extend((payload.len() as u32 + 4).to_le_bytes());
        b.extend((items.len() as u32).to_le_bytes());
        b.extend(payload);
        b
    }

    pub fn encode_v0(maps: &[Map]) -> Vec<u8> {
        let map_objs = maps
            .iter()
            .map(|m| {
                let chans = m
                    .channels
                    .iter()
                    .map(|c| {
                        v0_object(
                            0x14,
                            "DataExportChannel",
                            vec![("channelSrc", v0_int(c.src)), ("mapId", v0_bytes(&c.id)), ("mapIdType", v0_int(c.ty))],
                        )
                    })
                    .collect();
                v0_object(
                    0x12,
                    "DataExportMap",
                    vec![
                        ("bitdepth", v0_int(m.bitdepth)),
                        ("channels", v0_list(0x11, chans)),
                        ("dithering", vec![0x0a, m.dithering as u8]),
                        ("fileformat", v0_bytes(m.format.as_bytes())),
                        ("path", v0_bytes(m.path.as_bytes())),
                        ("uid", {
                            let mut u = vec![0x0c];
                            u.extend(14u64.to_le_bytes());
                            u
                        }),
                    ],
                )
            })
            .collect();
        let root = v0_object(
            0x12,
            "DataExportPreset",
            vec![("maps", v0_list(0x13, map_objs)), ("uid", {
                let mut u = vec![0x0c];
                u.extend(13u64.to_le_bytes());
                u
            })],
        );
        let mut file = MAGIC.to_vec();
        file.extend(0u32.to_le_bytes());
        file.extend(3u32.to_le_bytes());
        file.extend(root);
        file
    }

    // version 1 ----------------------------------------------------------------------------

    fn schema(class: &str, fields: &[(&str, u32)]) -> Vec<u8> {
        let mut b = name(class);
        b.extend((fields.len() as u32).to_le_bytes());
        for (k, t) in fields {
            b.extend(name(k));
            b.extend(t.to_le_bytes());
        }
        b
    }

    fn v1_bytes(data: &[u8]) -> Vec<u8> {
        let mut b = (data.len() as u32).to_le_bytes().to_vec();
        b.extend(data);
        b
    }

    pub fn encode_v1(maps: &[Map]) -> Vec<u8> {
        const MAP_FIELDS: [(&str, u32); 6] =
            [("bitdepth", 9), ("channels", 0x11), ("dithering", 0x0a), ("fileformat", 0x10), ("path", 0x10), ("uid", 0x0c)];
        const CH_FIELDS: [(&str, u32); 3] = [("channelSrc", 9), ("mapId", 0x10), ("mapIdType", 9)];
        let mut body = Vec::new();
        body.extend(0xffff_ffffu32.to_le_bytes());
        body.extend(schema("DataExportPreset", &[("maps", 0x13), ("uid", 0x0c)]));
        body.extend((maps.len() as u32).to_le_bytes());
        for (mi, m) in maps.iter().enumerate() {
            let mut item = Vec::new();
            // First map defines the class; later ones reference index 1 (0 is the root class).
            if mi == 0 {
                item.extend(0xffff_ffffu32.to_le_bytes());
                item.extend(schema("DataExportMap", &MAP_FIELDS));
            } else {
                item.extend(1u32.to_le_bytes());
            }
            item.extend((m.bitdepth as i32).to_le_bytes());
            item.extend((m.channels.len() as u32).to_le_bytes());
            for (ci, c) in m.channels.iter().enumerate() {
                if ci == 0 {
                    // The channel class is defined once; afterwards a reference to index 2.
                    if mi == 0 {
                        item.extend(0xffff_ffffu32.to_le_bytes());
                        item.extend(schema("DataExportChannel", &CH_FIELDS));
                    } else {
                        item.extend(2u32.to_le_bytes());
                    }
                }
                item.extend((c.src as i32).to_le_bytes());
                item.extend(v1_bytes(&c.id));
                item.extend((c.ty as i32).to_le_bytes());
            }
            item.push(m.dithering as u8);
            item.extend(v1_bytes(m.format.as_bytes()));
            item.extend(v1_bytes(m.path.as_bytes()));
            item.extend(14u64.to_le_bytes());
            body.extend((item.len() as u32).to_le_bytes()); // size is ignored by the reader
            body.extend(item);
        }
        body.extend(13u64.to_le_bytes());

        let mut file = MAGIC.to_vec();
        file.extend(1u32.to_le_bytes());
        file.extend(5u32.to_le_bytes());
        file.extend(0x12u32.to_le_bytes());
        file.extend(3u32.to_le_bytes());
        file.extend((body.len() as u32).to_le_bytes());
        file.extend(body);
        file
    }

    pub fn sample_maps() -> Vec<Map<'static>> {
        vec![
            Map {
                bitdepth: 1,
                dithering: false,
                format: "png",
                path: "t_$textureSet_BaseColor(.$udim)",
                channels: vec![
                    Ch { src: 0, id: variant_u64(1), ty: 1 },
                    Ch { src: 1, id: variant_u64(1), ty: 1 },
                    Ch { src: 2, id: variant_u64(1), ty: 1 },
                    Ch { src: 4, id: variant_u64(8), ty: 1 },
                ],
            },
            Map {
                bitdepth: 0,
                dithering: true,
                format: "png",
                path: "n_$textureSet",
                channels: vec![
                    Ch { src: 0, id: variant_str("Normal_OpenGL"), ty: 2 },
                    Ch { src: 1, id: variant_str("Normal_OpenGL"), ty: 2 },
                    Ch { src: 2, id: variant_str("Normal_OpenGL"), ty: 2 },
                ],
            },
            Map {
                bitdepth: 0,
                dithering: false,
                format: "png",
                path: "ao",
                channels: vec![Ch { src: 4, id: variant_i32(1), ty: 3 }],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn expected() -> SpexpPreset {
        let maps = sample_maps();
        SpexpPreset {
            container_version: 0,
            maps: maps
                .iter()
                .map(|m| SpexpMap {
                    bitdepth: m.bitdepth,
                    dithering: m.dithering,
                    file_format: m.format.into(),
                    path: m.path.into(),
                    channels: m
                        .channels
                        .iter()
                        .map(|c| SpexpChannel { src: c.src, map: decode_map_id(&c.id).unwrap(), map_type: c.ty })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn version_0_layout_decodes() {
        let got = decode_spexp(&encode_v0(&sample_maps())).unwrap();
        assert_eq!(got, expected());
        assert_eq!(got.maps[0].channels[3].map, MapId::Int(8));
        assert_eq!(got.maps[1].channels[0].map, MapId::Str("Normal_OpenGL".into()));
        assert_eq!(got.maps[2].channels[0].map, MapId::Int(1));
    }

    #[test]
    fn version_1_layout_decodes_to_the_same_content() {
        let got = decode_spexp(&encode_v1(&sample_maps())).unwrap();
        let mut want = expected();
        want.container_version = 1;
        assert_eq!(got, want);
    }

    #[test]
    fn rejects_foreign_and_unsupported_files() {
        assert_eq!(decode_spexp(b""), Err(SpexpError::BadMagic));
        assert_eq!(decode_spexp(b"PK\x03\x04 not a preset"), Err(SpexpError::BadMagic));
        let mut v = encode_v0(&sample_maps());
        v[4..8].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(decode_spexp(&v), Err(SpexpError::UnsupportedVersion(7)));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        for mut file in [encode_v0(&sample_maps()), encode_v1(&sample_maps())] {
            file.push(0);
            assert!(matches!(decode_spexp(&file), Err(SpexpError::Malformed(_))));
        }
    }

    /// Every truncation of a valid file must be an error, never a panic.
    #[test]
    fn every_truncation_is_an_error() {
        for file in [encode_v0(&sample_maps()), encode_v1(&sample_maps())] {
            for cut in 0..file.len() {
                assert!(decode_spexp(&file[..cut]).is_err(), "prefix of {cut} bytes was accepted");
            }
        }
    }

    /// Corrupt bytes may give a different preset or an error, but must never panic or hang.
    #[test]
    fn random_corruption_never_panics() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for base in [encode_v0(&sample_maps()), encode_v1(&sample_maps())] {
            for _ in 0..3000 {
                let mut f = base.clone();
                for _ in 0..1 + next() % 4 {
                    let at = (next() as usize) % f.len();
                    f[at] = next() as u8;
                }
                let _ = decode_spexp(&f);
            }
        }
    }

    #[test]
    fn absurd_list_counts_fail_fast_without_allocating() {
        // With no maps the list count is the 4 bytes just before the preset's uid.
        let v1 = encode_v1(&[]);
        let at = v1.len() - 8 - 4;
        assert_eq!(&v1[at..at + 4], &0u32.to_le_bytes(), "test assumes this layout");
        for count in [u32::MAX, 1_000_000, 50_000] {
            let mut f = v1.clone();
            f[at..at + 4].copy_from_slice(&count.to_le_bytes());
            assert!(decode_spexp(&f).is_err(), "v1 count {count} accepted");
        }
        // Version 0: the list is `13 size count`, followed by the uid field (name + tag + 8 bytes).
        let v0 = encode_v0(&[]);
        let at = v0.len() - (4 + 3 + 9) - 4;
        assert_eq!(&v0[at..at + 4], &0u32.to_le_bytes(), "test assumes this layout");
        for count in [u32::MAX, 1_000_000, 50_000] {
            let mut f = v0.clone();
            f[at..at + 4].copy_from_slice(&count.to_le_bytes());
            assert!(decode_spexp(&f).is_err(), "v0 count {count} accepted");
        }
    }

    #[test]
    fn map_ids_decode_the_supported_variant_types() {
        assert_eq!(decode_map_id(&variant_u64(8_388_608)).unwrap(), MapId::Int(8_388_608));
        assert_eq!(decode_map_id(&variant_i32(-3)).unwrap(), MapId::Int(-3));
        assert_eq!(decode_map_id(&variant_str("AO_Mixed")).unwrap(), MapId::Str("AO_Mixed".into()));
        assert!(decode_map_id(&[]).is_err());
        assert!(decode_map_id(&variant_u64(u64::MAX)).is_err(), "beyond i64");
        let mut odd = variant_str("ab");
        let n = odd.len();
        odd[n - 1] ^= 0; // keep valid
        odd.pop();
        assert!(decode_map_id(&odd).is_err(), "odd UTF-16 length");
        let mut trailing = variant_i32(1);
        trailing.push(0);
        assert!(decode_map_id(&trailing).is_err());
    }

    /// Optional: decode a directory of real presets and/or single files, set
    /// `OPEN_PAINTER_SPEXP_PATHS` to a `:`-separated list of files and directories.
    #[test]
    fn real_files_from_the_environment_decode_completely() {
        let Ok(paths) = std::env::var("OPEN_PAINTER_SPEXP_PATHS") else { return };
        let mut count = 0;
        for p in paths.split(':').filter(|p| !p.is_empty()) {
            let path = std::path::Path::new(p);
            let files: Vec<_> = if path.is_dir() {
                std::fs::read_dir(path).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "spexp")).collect()
            } else {
                vec![path.to_path_buf()]
            };
            for f in files {
                let bytes = std::fs::read(&f).unwrap();
                let p = decode_spexp(&bytes).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                assert!(!p.maps.is_empty(), "{} has no maps", f.display());
                count += 1;
            }
        }
        eprintln!("decoded {count} real preset files");
    }
}
