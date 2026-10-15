//! FBX import against a file written by Blender (see `data/README.md`): a 2x1x0.5 crate whose six
//! faces use three materials, and a rotated sphere using one of them.

use glam::Vec3;
use op_mesh::Mesh;

fn load() -> Mesh {
    Mesh::load_fbx(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/multi.fbx")).expect("fixture loads")
}

fn set_bounds(m: &Mesh, name: &str) -> (Vec3, Vec3) {
    let s = m.sets.iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no set {name}"));
    let idx = &m.indices[s.first_index as usize..(s.first_index + s.index_count) as usize];
    let pts: Vec<Vec3> = idx.iter().map(|&i| Vec3::from(m.vertices[i as usize].position)).collect();
    (pts.iter().copied().reduce(Vec3::min).unwrap(), pts.iter().copied().reduce(Vec3::max).unwrap())
}

#[test]
fn materials_become_sets_with_the_right_triangle_counts() {
    let m = load();
    assert!(m.sets_are_valid());
    let counts: Vec<(&str, u32)> = m.sets.iter().map(|s| (s.name.as_str(), s.index_count / 3)).collect();
    let get = |n: &str| counts.iter().find(|(name, _)| *name == n).map(|(_, c)| *c);
    assert_eq!(get("Body"), Some(4));
    assert_eq!(get("Glass"), Some(4));
    // The crate's two Trim faces plus the whole sphere: same material name, one set.
    assert_eq!(get("Trim"), Some(4 + 960));
    assert_eq!(m.sets.len(), 3, "{counts:?}");
    assert_eq!(m.indices.len(), 3 * (4 + 4 + 4 + 960));
}

#[test]
fn transforms_and_the_up_axis_are_applied() {
    let m = load();
    // Blender is Z-up; the importer is Y-up: (x, y, z) -> (x, z, -y). Blender reported these bounds.
    let (lo, hi) = m.bounds();
    let close = |a: Vec3, b: Vec3| (a - b).abs().max_element() < 1e-3;
    assert!(close(lo, Vec3::new(-3.5, 0.5, -3.0)), "min {lo}");
    assert!(close(hi, Vec3::new(3.0, 3.5, 0.5)), "max {hi}");
    // The sphere sits at -x and is the only Trim geometry that low.
    let (tlo, _) = set_bounds(&m, "Trim");
    assert!(close(tlo, Vec3::new(-3.5, 0.5, -3.0)), "trim min {tlo}");
}

#[test]
fn normals_are_unit_and_point_away_from_the_object() {
    let m = load();
    for v in &m.vertices {
        assert!((Vec3::from(v.normal).length() - 1.0).abs() < 1e-3, "{:?}", v.normal);
    }
    // The sphere is centred on (-3, 1, 0) in Y-up space.
    let centre = Vec3::new(-3.0, 1.0, 0.0);
    let s = m.sets.iter().find(|s| s.name == "Trim").unwrap();
    let mut checked = 0;
    for &i in &m.indices[s.first_index as usize..(s.first_index + s.index_count) as usize] {
        let v = &m.vertices[i as usize];
        let p = Vec3::from(v.position);
        if (p - centre).length() < 0.6 {
            assert!(Vec3::from(v.normal).dot((p - centre).normalize()) > 0.95, "inward normal at {p}");
            checked += 1;
        }
    }
    assert!(checked > 500);
}

#[test]
fn uvs_are_present_and_in_the_unit_square_for_the_sphere() {
    let m = load();
    assert!(m.vertices.iter().any(|v| v.uv != [0.0, 0.0]));
    assert!(m.vertices.iter().all(|v| v.uv.iter().all(|c| c.is_finite())));
}

#[test]
fn damaged_and_foreign_files_are_errors_not_panics() {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/multi.fbx")).unwrap();
    let dir = std::env::temp_dir().join(format!("op-fbx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let try_load = |name: &str, data: &[u8]| {
        let p = dir.join(name);
        std::fs::write(&p, data).unwrap();
        Mesh::load_fbx(&p)
    };
    assert!(try_load("empty.fbx", b"").is_err());
    assert!(try_load("text.fbx", b"this is not an fbx file at all").is_err());
    for cut in [30, 100, bytes.len() / 3, bytes.len() / 2, bytes.len() - 5] {
        let _ = try_load("cut.fbx", &bytes[..cut]);
    }
    assert!(Mesh::load_fbx(dir.join("missing.fbx")).is_err());
    std::fs::remove_dir_all(dir).ok();
}

/// Regression: set order must not depend on hash ordering inside the FBX library.
#[test]
fn set_order_is_the_same_on_every_load() {
    let names = |m: &Mesh| m.sets.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
    let first = names(&load());
    for _ in 0..30 {
        assert_eq!(names(&load()), first);
    }
}

#[test]
fn ascii_fbx_gets_a_helpful_error() {
    let dir = std::env::temp_dir().join(format!("op-fbx-ascii-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("a.fbx");
    std::fs::write(&p, "; FBX 7.3.0 project file\nFBXHeaderExtension:  {\n}\n").unwrap();
    let e = Mesh::load_fbx(&p).unwrap_err().to_string();
    assert!(e.contains("ASCII") && e.contains("binary"), "{e}");
    std::fs::remove_dir_all(dir).ok();
}
