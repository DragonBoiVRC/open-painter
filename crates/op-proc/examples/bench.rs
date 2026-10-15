//! Times every built-in generator on a baked, subdivided cube.
//!
//!   cargo run --release -p op-proc --example bench -- [size]

use op_bake::{bake, BakeSettings, Progress};
use op_core::{Generator, GeneratorKind, MeshMap};
use op_mesh::{Mesh, Vertex};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn cube(n: usize) -> Mesh {
    let faces = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for (f, (nrm, u, v)) in faces.iter().enumerate() {
        let (tx, ty) = ((f % 3) as f32 / 3.0, (f / 3) as f32 / 2.0);
        let base = vertices.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (a, b) = (i as f32 / n as f32, j as f32 / n as f32);
                let (s, t) = (a * 2.0 - 1.0, b * 2.0 - 1.0);
                let p = [0, 1, 2].map(|k| nrm[k] + u[k] * s + v[k] * t);
                vertices.push(Vertex { position: p, normal: *nrm, uv: [tx + a / 3.0, ty + b / 2.0] });
            }
        }
        for j in 0..n {
            for i in 0..n {
                let k = base + (j * (n + 1) + i) as u32;
                indices.extend([k, k + 1, k + n as u32 + 2, k, k + n as u32 + 2, k + n as u32 + 1]);
            }
        }
    }
    Mesh { vertices, indices }
}

fn main() {
    let size: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    println!("threads: {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let t = Instant::now();
    let maps = bake(
        &cube(10),
        None,
        &BakeSettings {
            size,
            maps: vec![MeshMap::Curvature, MeshMap::AmbientOcclusion, MeshMap::WorldSpaceNormal, MeshMap::Position],
            samples: 16,
            ..Default::default()
        },
        &Progress::default(),
    )
    .unwrap()
    .maps;
    println!("bake {size}x{size} (4 maps, 16 rays): {:.2} s", t.elapsed().as_secs_f32());
    let cancel = AtomicBool::new(false);
    for kind in [GeneratorKind::EdgeWear, GeneratorKind::CreviceDirt, GeneratorKind::TopDust, GeneratorKind::HeightGradient, GeneratorKind::Noise] {
        let label = op_proc::label(&kind);
        let g = Generator::new(kind);
        let t = Instant::now();
        let e = op_proc::generate(&g, size, &maps, &cancel).unwrap();
        println!("{label:16} {:>7.0} ms  ({} warnings)", t.elapsed().as_secs_f32() * 1000.0, e.warnings.len());
    }
}
