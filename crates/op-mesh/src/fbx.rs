//! FBX import: meshes with their node transforms, normals, UVs and per-polygon materials.
//!
//! Only binary FBX 7.x is read (what Blender, Maya, 3ds Max and Unity write by default).

use crate::{Mesh, MeshError, SetRange, Vertex};
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::object::property::loaders::{PrimitiveLoader, F64Arr3Loader};
use fbxcel_dom::v7400::object::{ObjectHandle, TypedObjectHandle, model::TypedModelHandle};
use fbxcel_dom::v7400::Document;
use glam::{DMat3, DMat4, DVec3, Vec3};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

/// Refuse files that would need absurd amounts of memory.
const MAX_VERTICES: usize = 50_000_000;
const MAX_PARENT_DEPTH: usize = 256;
const DEFAULT_SET: &str = "DefaultMaterial";

fn err(why: impl ToString) -> MeshError {
    MeshError::Fbx(why.to_string())
}

pub fn load(path: &Path) -> Result<Mesh, MeshError> {
    let file = std::fs::File::open(path).map_err(err)?;
    // ASCII FBX starts with a "; FBX" comment instead of the binary magic.
    {
        use std::io::Read;
        let mut head = [0u8; 20];
        let n = (&file).take(20).read(&mut head).map_err(err)?;
        if head[..n].starts_with(b"; FBX") || head[..n].starts_with(b"\xEF\xBB\xBF; FBX") {
            return Err(err("this is an ASCII FBX file; export it as binary FBX (7.x) and try again"));
        }
    }
    let file = std::fs::File::open(path).map_err(err)?;
    match AnyDocument::from_seekable_reader(BufReader::new(file)).map_err(err)? {
        AnyDocument::V7400(_, doc) => load_document(&doc),
        _ => Err(err("only FBX 7.x files are supported")),
    }
}

/// Euler rotation order as the FBX `RotationOrder` enum: which axis is applied first, second, third.
fn rotation_order(code: i32) -> [usize; 3] {
    match code {
        1 => [0, 2, 1], // XZY
        2 => [1, 2, 0], // YZX
        3 => [1, 0, 2], // YXZ
        4 => [2, 0, 1], // ZXY
        5 => [2, 1, 0], // ZYX
        _ => [0, 1, 2], // XYZ (also the spheric order, which is not otherwise supported)
    }
}

/// Rotation from Euler angles in degrees, applying the axes in `order`.
fn euler(deg: DVec3, order: [usize; 3]) -> DMat4 {
    let axis = |i: usize| match i {
        0 => DMat4::from_rotation_x(deg.x.to_radians()),
        1 => DMat4::from_rotation_y(deg.y.to_radians()),
        _ => DMat4::from_rotation_z(deg.z.to_radians()),
    };
    axis(order[2]) * axis(order[1]) * axis(order[0])
}

struct Props<'a> {
    obj: ObjectHandle<'a>,
}

impl Props<'_> {
    fn vec3(&self, name: &str, default: DVec3) -> DVec3 {
        let props = self.obj.properties_by_native_typename("FbxNode");
        props
            .get_property(name)
            .and_then(|p| p.load_value(F64Arr3Loader::new()).ok())
            .map_or(default, |v| DVec3::from_array(v))
    }

    fn int(&self, name: &str, default: i32) -> i32 {
        let props = self.obj.properties_by_native_typename("FbxNode");
        props.get_property(name).and_then(|p| p.load_value(PrimitiveLoader::<i32>::new()).ok()).unwrap_or(default)
    }
}

/// A node's transform relative to its parent, following the FBX transform pipeline:
/// `T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1`.
fn local_transform(obj: ObjectHandle<'_>) -> DMat4 {
    let p = Props { obj };
    let t = p.vec3("Lcl Translation", DVec3::ZERO);
    let r = p.vec3("Lcl Rotation", DVec3::ZERO);
    let s = p.vec3("Lcl Scaling", DVec3::ONE);
    let pre = p.vec3("PreRotation", DVec3::ZERO);
    let post = p.vec3("PostRotation", DVec3::ZERO);
    let roff = p.vec3("RotationOffset", DVec3::ZERO);
    let rp = p.vec3("RotationPivot", DVec3::ZERO);
    let soff = p.vec3("ScalingOffset", DVec3::ZERO);
    let sp = p.vec3("ScalingPivot", DVec3::ZERO);
    let order = rotation_order(p.int("RotationOrder", 0));
    let tr = DMat4::from_translation;
    tr(t) * tr(roff) * tr(rp) * euler(pre, [0, 1, 2]) * euler(r, order) * euler(post, [0, 1, 2]).inverse() * tr(-rp)
        * tr(soff)
        * tr(sp)
        * DMat4::from_scale(s)
        * tr(-sp)
}

/// Extra transform that applies to a node's own geometry but not to its children.
fn geometric_transform(obj: ObjectHandle<'_>) -> DMat4 {
    let p = Props { obj };
    let t = p.vec3("GeometricTranslation", DVec3::ZERO);
    let r = p.vec3("GeometricRotation", DVec3::ZERO);
    let s = p.vec3("GeometricScaling", DVec3::ONE);
    DMat4::from_translation(t) * euler(r, [0, 1, 2]) * DMat4::from_scale(s)
}

fn world_transform(model: &TypedModelHandle<'_>) -> DMat4 {
    let mut chain = vec![local_transform(***model)];
    let mut cur = model.parent_model();
    while let Some(parent) = cur {
        if chain.len() >= MAX_PARENT_DEPTH {
            break;
        }
        chain.push(local_transform(***&parent));
        cur = parent.parent_model();
    }
    chain.iter().rev().fold(DMat4::IDENTITY, |acc, m| acc * *m)
}

/// A matrix taking file units to metres and the file's up axis to +Y (the viewer is Y-up).
fn scene_fix(doc: &Document) -> DMat4 {
    let get = |name: &str, default: i32| {
        doc.global_settings()
            .and_then(|g| g.raw_properties().get_property(name))
            .and_then(|p| p.load_value(PrimitiveLoader::<i32>::new()).ok())
            .unwrap_or(default)
    };
    // FBX coordinates are in units of `UnitScaleFactor` centimetres; the viewer works in metres.
    let unit = doc
        .global_settings()
        .and_then(|g| g.raw_properties().get_property("UnitScaleFactor"))
        .and_then(|p| p.load_value(PrimitiveLoader::<f64>::new()).ok())
        .filter(|u| u.is_finite() && *u > 0.0)
        .unwrap_or(1.0);
    let scale = DMat4::from_scale(DVec3::splat(unit * 0.01));
    let sign = if get("UpAxisSign", 1) < 0 { -1.0 } else { 1.0 };
    scale * match get("UpAxis", 1) {
        // Z-up: rotate -90 degrees about X so +Z becomes +Y.
        2 => DMat4::from_rotation_x(-sign * std::f64::consts::FRAC_PI_2),
        // X-up: rotate about Z so +X becomes +Y.
        0 => DMat4::from_rotation_z(sign * std::f64::consts::FRAC_PI_2),
        _ if sign < 0.0 => DMat4::from_rotation_x(std::f64::consts::PI),
        _ => DMat4::IDENTITY,
    }
}

/// Ear-clipping triangulation, so concave n-gons keep their shape.
fn ear_clip(
    pv: &fbxcel_dom::v7400::data::mesh::PolygonVertices<'_>,
    poly: &[fbxcel_dom::v7400::data::mesh::PolygonVertexIndex],
    out: &mut Vec<[fbxcel_dom::v7400::data::mesh::PolygonVertexIndex; 3]>,
) -> anyhow::Result<()> {
    let mut pts = Vec::with_capacity(poly.len());
    for &i in poly {
        let p = pv.control_point(i).ok_or_else(|| anyhow::anyhow!("a polygon refers to a missing control point"))?;
        pts.push(DVec3::new(p.x, p.y, p.z));
    }
    for t in crate::triangulate::triangulate(&pts) {
        out.push(t.map(|k| poly[k]));
    }
    Ok(())
}

#[derive(Default)]
struct Builder {
    vertices: Vec<Vertex>,
    /// Triangle indices per set, in first-seen order of set names.
    sets: Vec<(String, Vec<u32>)>,
}

impl Builder {
    fn set(&mut self, name: &str) -> usize {
        match self.sets.iter().position(|(n, _)| n == name) {
            Some(i) => i,
            None => {
                self.sets.push((name.to_string(), Vec::new()));
                self.sets.len() - 1
            }
        }
    }
}

fn load_document(doc: &Document) -> Result<Mesh, MeshError> {
    let fix = scene_fix(doc);
    let mut b = Builder::default();
    // Object order in the file's index is not stable between loads; the object id is.
    let mut objects: Vec<_> = doc.objects().collect();
    objects.sort_by_key(|o| o.object_id());
    for obj in objects {
        let TypedObjectHandle::Model(TypedModelHandle::Mesh(model)) = obj.get_typed() else { continue };
        let Ok(geometry) = model.geometry() else { continue };
        let typed = TypedModelHandle::Mesh(model.clone());
        let world = fix * world_transform(&typed) * geometric_transform(***&typed);
        add_geometry(&mut b, &model, &geometry, world)?;
    }
    finish(b)
}

fn add_geometry(
    b: &mut Builder,
    model: &fbxcel_dom::v7400::object::model::MeshHandle<'_>,
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    world: DMat4,
) -> Result<(), MeshError> {
    let polys = geometry.polygon_vertices().map_err(err)?;
    let tris = polys.triangulate_each(ear_clip).map_err(err)?;

    let (mut normals, mut uvs, mut mats) = (None, None, None);
    for layer in geometry.layers() {
        for entry in layer.layer_element_entries() {
            match entry.typed_layer_element() {
                Ok(TypedLayerElementHandle::Normal(h)) if normals.is_none() => normals = h.normals().ok(),
                Ok(TypedLayerElementHandle::Uv(h)) if uvs.is_none() => uvs = h.uv().ok(),
                Ok(TypedLayerElementHandle::Material(h)) if mats.is_none() => mats = h.materials().ok(),
                _ => {}
            }
        }
    }
    let material_names: Vec<String> = model.materials().map(|m| m.name().unwrap_or(DEFAULT_SET).to_string()).collect();
    let normal_matrix = DMat3::from_mat4(world).inverse().transpose();
    // Mirroring transforms flip the winding, so restore it.
    let flip = world.determinant() < 0.0;

    let mut dedup: HashMap<(u32, [u32; 3], [u32; 2]), u32> = HashMap::new();
    let mut corner = Vec::with_capacity(3);
    for tri_vi in tris.triangle_vertex_indices() {
        let Some(cp_index) = tris.control_point_index(tri_vi) else { return Err(err("a polygon refers to a missing control point")) };
        let cp = tris.control_point(tri_vi).ok_or_else(|| err("a polygon refers to a missing control point"))?;
        let position = world.transform_point3(DVec3::new(cp.x, cp.y, cp.z));
        let normal = match &normals {
            Some(n) => n.normal(&tris, tri_vi).ok().map(|n| (normal_matrix * DVec3::new(n.x, n.y, n.z)).normalize_or_zero()),
            None => None,
        };
        let uv = uvs.as_ref().and_then(|u| u.uv(&tris, tri_vi).ok()).map_or([0.0f32; 2], |p| [p.x as f32, p.y as f32]);
        let nrm = normal.map_or([0.0f32; 3], |n| [n.x as f32, n.y as f32, n.z as f32]);
        let key = (cp_index.to_u32(), nrm.map(f32::to_bits), uv.map(f32::to_bits));
        let index = match dedup.get(&key) {
            Some(&i) => i,
            None => {
                if b.vertices.len() >= MAX_VERTICES {
                    return Err(err("the mesh has too many vertices"));
                }
                let i = b.vertices.len() as u32;
                b.vertices.push(Vertex { position: Vec3::new(position.x as f32, position.y as f32, position.z as f32).into(), normal: nrm, uv });
                dedup.insert(key, i);
                i
            }
        };
        corner.push(index);
        if corner.len() == 3 {
            // The material of a triangle is that of its first corner's polygon.
            let slot = match &mats {
                Some(m) => m.material_index(&tris, tri_vi).map(|i| i.to_u32() as usize).unwrap_or(0),
                None => 0,
            };
            let name = material_names.get(slot).map(String::as_str).unwrap_or(DEFAULT_SET);
            let set = b.set(name);
            if flip {
                corner.swap(1, 2);
            }
            b.sets[set].1.extend_from_slice(&corner);
            corner.clear();
        }
    }
    Ok(())
}

fn finish(b: Builder) -> Result<Mesh, MeshError> {
    let mut mesh = Mesh { vertices: b.vertices, ..Default::default() };
    for (name, indices) in b.sets {
        if indices.is_empty() {
            continue;
        }
        mesh.sets.push(SetRange { name, first_index: mesh.indices.len() as u32, index_count: indices.len() as u32 });
        mesh.indices.extend(indices);
    }
    if mesh.indices.is_empty() {
        return Err(MeshError::Empty);
    }
    // Vertices without a stored normal get smooth normals from their triangles.
    if mesh.vertices.iter().any(|v| v.normal == [0.0; 3]) {
        let missing: Vec<bool> = mesh.vertices.iter().map(|v| v.normal == [0.0; 3]).collect();
        let old: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.normal).collect();
        mesh.recompute_normals();
        for (i, v) in mesh.vertices.iter_mut().enumerate() {
            if !missing[i] {
                v.normal = old[i];
            }
        }
    }
    debug_assert!(mesh.sets_are_valid());
    Ok(mesh)
}
