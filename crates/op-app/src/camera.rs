use glam::{Mat4, Vec3};
// wgpu NDC: depth in [0, 1], Y up == glam's `directx` projection convention.

/// Turntable camera orbiting a target point.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    /// Orthographic projection: no perspective, objects keep their size however far they are.
    pub ortho: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vec3::ZERO, distance: 4.0, yaw: 0.6, pitch: 0.35, fov_y: 45f32.to_radians(), ortho: false }
    }
}

/// Looking at the model along one of its axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPreset {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

impl ViewPreset {
    pub const ALL: [ViewPreset; 6] = [ViewPreset::Front, ViewPreset::Back, ViewPreset::Left, ViewPreset::Right, ViewPreset::Top, ViewPreset::Bottom];

    pub fn label(self) -> &'static str {
        match self {
            ViewPreset::Front => "Front",
            ViewPreset::Back => "Back",
            ViewPreset::Left => "Left",
            ViewPreset::Right => "Right",
            ViewPreset::Top => "Top",
            ViewPreset::Bottom => "Bottom",
        }
    }
}

/// The orbit never reaches straight up or down, so the top and bottom views stop just short.
const POLE_PITCH: f32 = 1.5;

impl OrbitCamera {
    /// Turn to look along an axis, keeping the target and distance. Front is the +Z side.
    pub fn set_view(&mut self, view: ViewPreset) {
        use std::f32::consts::{FRAC_PI_2, PI};
        let (yaw, pitch) = match view {
            ViewPreset::Front => (FRAC_PI_2, 0.0),
            ViewPreset::Back => (-FRAC_PI_2, 0.0),
            ViewPreset::Right => (0.0, 0.0),
            ViewPreset::Left => (PI, 0.0),
            ViewPreset::Top => (FRAC_PI_2, POLE_PITCH),
            ViewPreset::Bottom => (FRAC_PI_2, -POLE_PITCH),
        };
        self.yaw = yaw;
        self.pitch = pitch;
    }

    pub fn position(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cy * cp, sp, sy * cp)
    }

    /// The camera's orthonormal axes in world space: (right, up, forward).
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.target - self.position()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        (right, up, forward)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.0015)).clamp(0.05, 1000.0);
    }

    /// Pan in the camera plane; `dx`/`dy` are in screen pixels.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let world_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let fwd = (self.target - self.position()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        self.target += (-right * dx + up * dy) * world_per_px;
    }

    /// Frame a bounding box so it fills the view.
    pub fn frame(&mut self, min: Vec3, max: Vec3) {
        self.target = (min + max) * 0.5;
        let radius = ((max - min) * 0.5).length().max(1e-3);
        self.distance = radius / (self.fov_y * 0.5).sin() * 1.1;
    }

    pub fn near_far(&self) -> (f32, f32) {
        ((self.distance * 0.01).max(0.001), self.distance * 100.0)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.position(), self.target, Vec3::Y);
        let (near, far) = self.near_far();
        let aspect = aspect.max(1e-3);
        let proj = if self.ortho {
            // The same height of the scene is visible as in perspective at the target's distance.
            let half_h = self.distance * (self.fov_y * 0.5).tan();
            glam::camera::rh::proj::directx::orthographic(-half_h * aspect, half_h * aspect, -half_h, half_h, near, far)
        } else {
            glam::camera::rh::proj::directx::perspective(self.fov_y, aspect, near, far)
        };
        proj * view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_centers_target_and_keeps_box_visible() {
        let mut c = OrbitCamera::default();
        c.frame(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(c.target.length() < 1e-6);
        let clip = c.view_proj(1.0) * Vec3::splat(1.0).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0, "corner outside frustum: {ndc}");
    }

    #[test]
    fn view_presets_look_along_their_axis() {
        let mut c = OrbitCamera::default();
        for (view, axis) in [
            (ViewPreset::Front, Vec3::Z),
            (ViewPreset::Back, -Vec3::Z),
            (ViewPreset::Right, Vec3::X),
            (ViewPreset::Left, -Vec3::X),
        ] {
            c.set_view(view);
            let dir = (c.position() - c.target).normalize();
            assert!((dir - axis).length() < 1e-5, "{view:?}: camera at {dir}");
        }
        c.set_view(ViewPreset::Top);
        assert!((c.position() - c.target).normalize().y > 0.99);
        c.set_view(ViewPreset::Bottom);
        assert!((c.position() - c.target).normalize().y < -0.99);
    }

    #[test]
    fn orthographic_keeps_sizes_constant_with_distance() {
        let mut c = OrbitCamera { ortho: true, ..Default::default() };
        let size_at = |c: &OrbitCamera, depth_offset: f32| {
            // Two points one unit apart across the view, at a given depth along the view direction.
            let (right, _, forward) = c.basis();
            let base = c.target + forward * depth_offset;
            let (a, b) = (c.view_proj(1.0) * base.extend(1.0), c.view_proj(1.0) * (base + right).extend(1.0));
            ((b.x / b.w) - (a.x / a.w)).abs()
        };
        let (near_size, far_size) = (size_at(&c, -1.0), size_at(&c, 1.0));
        assert!((near_size - far_size).abs() < 1e-5, "orthographic: {near_size} vs {far_size}");
        c.ortho = false;
        assert!(size_at(&c, -1.0) > size_at(&c, 1.0) * 1.1, "perspective shrinks with distance");
        // The target stays in the middle of the view either way.
        for ortho in [false, true] {
            c.ortho = ortho;
            let clip = c.view_proj(1.5) * c.target.extend(1.0);
            assert!((clip.x / clip.w).abs() < 1e-4 && (clip.y / clip.w).abs() < 1e-4);
        }
    }

    #[test]
    fn position_is_at_orbit_distance() {
        let c = OrbitCamera { distance: 7.0, ..Default::default() };
        assert!((c.position() - c.target).length() - 7.0 < 1e-4);
    }

    #[test]
    fn basis_is_orthonormal_and_points_along_the_view() {
        for (yaw, pitch) in [(0.0, 0.0), (0.6, 0.35), (2.5, -1.0), (-1.2, 1.4)] {
            let c = OrbitCamera { yaw, pitch, ..Default::default() };
            let (r, u, f) = c.basis();
            for v in [r, u, f] {
                assert!((v.length() - 1.0).abs() < 1e-5, "unit length at {yaw},{pitch}");
            }
            assert!(r.dot(u).abs() < 1e-5 && r.dot(f).abs() < 1e-5 && u.dot(f).abs() < 1e-5);
            assert!((f - (c.target - c.position()).normalize()).length() < 1e-5);
            assert!(u.dot(Vec3::Y) > 0.0, "up stays on the upper side of the horizon");
        }
        // At yaw 0 the camera sits on +X looking back at the origin.
        let (_, _, f) = OrbitCamera { yaw: 0.0, pitch: 0.0, ..Default::default() }.basis();
        assert!((f + Vec3::X).length() < 1e-5);
    }
}
