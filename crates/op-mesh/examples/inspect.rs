fn main() {
    let p = std::env::args().nth(1).expect("path");
    let t = std::time::Instant::now();
    match op_mesh::Mesh::load_fbx(&p) {
        Ok(m) => {
            let (lo, hi) = m.bounds();
            println!("loaded in {:?}: {} verts, {} tris, bounds {lo:?}..{hi:?}", t.elapsed(), m.vertices.len(), m.indices.len() / 3);
            for s in &m.sets { println!("  {:30} {:>8} tris", s.name, s.index_count / 3); }
        }
        Err(e) => println!("ERROR: {e}"),
    }
}
