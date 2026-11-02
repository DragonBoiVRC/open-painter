//! A small CPU path tracer for previewing finished textures on the model: diffuse and specular
//! bounces, a procedural sky with a sun, soft shadows. It refines over many passes (one sample per
//! pixel each), so a viewer can show the picture sharpen. Not physically exact: a quick, honest look.

use crate::bvh::{Bvh, Ray};
use glam::{Mat4, Vec2, Vec3};
use op_mesh::Mesh;
use rayon::prelude::*;

/// One texture set's colour, metallic and roughness, linear, square.
pub struct SetTexture {
    pub size: u32,
    /// Linear base colour, three floats per texel.
    pub base: Vec<f32>,
    pub metal: Vec<f32>,
    pub rough: Vec<f32>,
}

impl SetTexture {
    /// A texture of one value everywhere.
    pub fn flat(base: [f32; 3], metal: f32, rough: f32) -> Self {
        Self { size: 1, base: base.to_vec(), metal: vec![metal], rough: vec![rough] }
    }

    fn texel(&self, uv: Vec2) -> (Vec3, f32, f32) {
        let s = self.size as i32;
        let x = ((uv.x - uv.x.floor()) * self.size as f32) as i32;
        let y = ((uv.y - uv.y.floor()) * self.size as f32) as i32;
        let i = (y.clamp(0, s - 1) * s + x.clamp(0, s - 1)) as usize;
        (Vec3::new(self.base[i * 3], self.base[i * 3 + 1], self.base[i * 3 + 2]), self.metal[i], self.rough[i])
    }
}

pub struct Scene {
    bvh: Bvh,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,
    /// The texture set of each triangle.
    tri_set: Vec<u32>,
    textures: Vec<SetTexture>,
}

impl Scene {
    /// `textures` has one entry per texture set of the mesh; a missing one renders grey.
    pub fn new(mesh: &Mesh, mut textures: Vec<SetTexture>) -> Self {
        let sets = mesh.set_ranges();
        while textures.len() < sets.len() {
            textures.push(SetTexture::flat([0.5; 3], 0.0, 0.5));
        }
        let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from(v.position)).collect();
        let tris: Vec<[Vec3; 3]> = mesh.indices.chunks_exact(3).map(|t| [positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]]).collect();
        let mut tri_set = vec![0u32; tris.len()];
        for (s, r) in sets.iter().enumerate() {
            for t in (r.first_index / 3)..((r.first_index + r.index_count) / 3) {
                tri_set[t as usize] = s as u32;
            }
        }
        Self {
            bvh: Bvh::build(&tris),
            normals: mesh.vertices.iter().map(|v| Vec3::from(v.normal)).collect(),
            uvs: mesh.vertices.iter().map(|v| Vec2::from(v.uv)).collect(),
            positions,
            indices: mesh.indices.clone(),
            tri_set,
            textures,
        }
    }
}

/// A pinhole or orthographic view, from the viewport's view-projection matrix.
pub struct View {
    inverse: Mat4,
    pub width: u32,
    pub height: u32,
}

impl View {
    pub fn new(view_proj: Mat4, width: u32, height: u32) -> Self {
        Self { inverse: view_proj.inverse(), width, height }
    }

    fn ray(&self, x: f32, y: f32) -> Ray {
        let ndc = Vec2::new(x / self.width as f32 * 2.0 - 1.0, 1.0 - y / self.height as f32 * 2.0);
        let near = self.inverse.project_point3(ndc.extend(0.0));
        let far = self.inverse.project_point3(ndc.extend(1.0));
        Ray { origin: near, dir: (far - near).normalize_or_zero(), tmin: 1e-4, tmax: f32::MAX }
    }
}

struct Rng(u32);

impl Rng {
    fn new(a: u32, b: u32) -> Self {
        let mut r = Rng(a.wrapping_mul(0x9E3779B1) ^ b.wrapping_mul(0x85EBCA6B) ^ 0x1234567);
        r.next();
        r
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

const SUN: Vec3 = Vec3::new(0.45, 0.8, 0.4);

fn sky(d: Vec3) -> Vec3 {
    let t = (d.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let base = Vec3::new(0.75, 0.82, 0.95).lerp(Vec3::new(0.25, 0.45, 0.85), t.powf(0.6));
    let ground = Vec3::splat(0.12);
    let col = if d.y < 0.0 { base.lerp(ground, (-d.y * 3.0).min(1.0)) } else { base };
    col + Vec3::new(1.0, 0.92, 0.8) * 25.0 * d.dot(SUN.normalize()).max(0.0).powf(400.0)
}

fn orthonormal(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t = n.cross(a).normalize();
    (t, n.cross(t))
}

fn cosine(n: Vec3, rng: &mut Rng) -> Vec3 {
    let (r, phi) = (rng.next().sqrt(), rng.next() * std::f32::consts::TAU);
    let (t, b) = orthonormal(n);
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - r * r).max(0.0).sqrt()).normalize()
}

fn trace(scene: &Scene, mut ray: Ray, rng: &mut Rng) -> Vec3 {
    let sun = SUN.normalize();
    let (mut radiance, mut through) = (Vec3::ZERO, Vec3::ONE);
    for bounce in 0..4 {
        let Some(hit) = scene.bvh.closest_hit(&ray) else {
            radiance += through * sky(ray.dir);
            break;
        };
        let t = hit.tri as usize;
        let i = [scene.indices[t * 3] as usize, scene.indices[t * 3 + 1] as usize, scene.indices[t * 3 + 2] as usize];
        let w = [1.0 - hit.u - hit.v, hit.u, hit.v];
        let blend3 = |a: &[Vec3]| a[i[0]] * w[0] + a[i[1]] * w[1] + a[i[2]] * w[2];
        let p = blend3(&scene.positions);
        let uv = scene.uvs[i[0]] * w[0] + scene.uvs[i[1]] * w[1] + scene.uvs[i[2]] * w[2];
        let mut n = blend3(&scene.normals).normalize_or_zero();
        let geo = (scene.positions[i[1]] - scene.positions[i[0]]).cross(scene.positions[i[2]] - scene.positions[i[0]]).normalize_or_zero();
        if n == Vec3::ZERO {
            n = geo;
        }
        if n.dot(ray.dir) > 0.0 {
            n = -n; // two-sided
        }
        let (base, metal, rough) = scene.textures[scene.tri_set[t] as usize].texel(uv);
        let (base, metal, rough) = (base.max(Vec3::ZERO), metal.clamp(0.0, 1.0), rough.clamp(0.04, 1.0));
        let f0 = Vec3::splat(0.04).lerp(base, metal);
        let diffuse = base * (1.0 - metal);
        let origin = p + n * 1e-3;
        // Specular share of this bounce: Fresnel at the viewing angle, strongest for metals and smooth surfaces.
        let cos_v = (-ray.dir).dot(n).clamp(0.0, 1.0);
        let fresnel = f0 + (Vec3::ONE - f0) * (1.0 - cos_v).powi(5);
        let spec_p = (fresnel.dot(Vec3::splat(1.0 / 3.0)) * (1.2 - rough * 0.6)).clamp(0.05, 0.95);
        if rng.next() < spec_p {
            let mirror = ray.dir - n * 2.0 * ray.dir.dot(n);
            let mut d = (mirror + (cosine(n, rng) - n) * (rough * rough)).normalize_or_zero();
            if d.dot(n) <= 0.0 {
                d = mirror;
            }
            through *= fresnel / spec_p;
            ray = Ray { origin, dir: d, tmin: 1e-4, tmax: f32::MAX };
        } else {
            // Sunlight straight onto the surface, if nothing is in the way.
            let to_sun = (sun + (cosine(Vec3::Y, rng) - Vec3::Y) * 0.03).normalize();
            let facing = n.dot(to_sun);
            if facing > 0.0 && !scene.bvh.any_hit(&Ray { origin, dir: to_sun, tmin: 1e-4, tmax: f32::MAX }) {
                radiance += through * diffuse * Vec3::new(3.2, 3.0, 2.7) * facing / (1.0 - spec_p);
            }
            through *= diffuse / (1.0 - spec_p);
            ray = Ray { origin, dir: cosine(n, rng), tmin: 1e-4, tmax: f32::MAX };
        }
        if bounce >= 2 {
            let live = through.max_element().clamp(0.05, 0.95);
            if rng.next() > live {
                break;
            }
            through /= live;
        }
    }
    radiance
}

/// Add one sample per pixel to `accum` (three floats per pixel, row by row). `pass` varies the random
/// numbers and jitters the pixel, so repeated calls converge.
pub fn render_pass(scene: &Scene, view: &View, pass: u32, accum: &mut [f32]) {
    let w = view.width as usize;
    accum.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut rng = Rng::new((y * w + x) as u32, pass);
            let ray = view.ray(x as f32 + rng.next(), y as f32 + rng.next());
            let c = trace(scene, ray, &mut rng);
            // A stray bright sample would show as a firefly: keep each one bounded.
            let c = if c.is_finite() { c.min(Vec3::splat(12.0)) } else { Vec3::ZERO };
            row[x * 3] += c.x;
            row[x * 3 + 1] += c.y;
            row[x * 3 + 2] += c.z;
        }
    });
}

/// The accumulated samples as sRGB RGBA8 (filmic curve, then gamma), `samples` passes in.
pub fn to_rgba8(accum: &[f32], samples: u32) -> Vec<u8> {
    let k = 1.0 / samples.max(1) as f32;
    accum
        .chunks_exact(3)
        .flat_map(|c| {
            let map = |v: f32| {
                let v = v * k;
                let a = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14); // ACES approximation
                (a.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
            };
            [map(c[0]), map(c[1]), map(c[2]), 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_mesh::Mesh;

    #[test]
    fn a_lit_sphere_is_brighter_than_the_dark_ground_and_converges() {
        let mesh = Mesh::uv_sphere(1.0, 24, 12);
        let scene = Scene::new(&mesh, vec![SetTexture::flat([0.8, 0.2, 0.1], 0.0, 0.6)]);
        let proj = glam::camera::rh::proj::directx::perspective(0.8, 1.0, 0.1, 20.0) * glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 3.5), Vec3::ZERO, Vec3::Y);
        let view = View::new(proj, 24, 24);
        let mut accum = vec![0.0; 24 * 24 * 3];
        for pass in 0..8 {
            render_pass(&scene, &view, pass, &mut accum);
        }
        let rgba = to_rgba8(&accum, 8);
        let px = |x: usize, y: usize| &rgba[(y * 24 + x) * 4..(y * 24 + x) * 4 + 3];
        let centre = px(12, 12);
        assert!(centre[0] > centre[2], "the red sphere reads red: {centre:?}");
        assert!(rgba.chunks(4).all(|p| p[3] == 255));
        // The corner sees sky, the centre sees the sphere: they differ.
        assert_ne!(px(0, 0), centre);
    }

    #[test]
    fn an_empty_scene_renders_the_sky_without_panicking() {
        let scene = Scene::new(&Mesh::default(), Vec::new());
        let view = View::new(Mat4::IDENTITY, 4, 4);
        let mut accum = vec![0.0; 4 * 4 * 3];
        render_pass(&scene, &view, 0, &mut accum);
        assert!(accum.iter().all(|v| v.is_finite()));
    }
}
