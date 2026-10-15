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
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vec3::ZERO, distance: 4.0, yaw: 0.6, pitch: 0.35, fov_y: 45f32.to_radians() }
    }
}

impl OrbitCamera {
    pub fn position(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cy * cp, sp, sy * cp)
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
        let proj = glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, far);
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
    fn position_is_at_orbit_distance() {
        let c = OrbitCamera { distance: 7.0, ..Default::default() };
        assert!((c.position() - c.target).length() - 7.0 < 1e-4);
    }
}
