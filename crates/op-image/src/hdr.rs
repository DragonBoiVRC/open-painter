//! High dynamic range environment images: decoding, mip chains for glossy reflections, and
//! spherical harmonics for diffuse lighting. No GPU code here.

use crate::{ImageError, MAX_FILE_BYTES, MAX_SIDE};
use image::{DynamicImage, ImageReader};
use std::io::Cursor;

/// The brightest radiance kept. Suns are orders of magnitude brighter than the sky and would
/// otherwise overflow half floats and dominate the diffuse light.
pub const MAX_RADIANCE: f32 = 5000.0;

/// A latitude-longitude (equirectangular) environment: linear RGB radiance, top row first.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    /// Three floats per pixel.
    pub rgb: Vec<f32>,
}

/// One level of a mip chain: RGBA floats (alpha 1).
#[derive(Clone, Debug, PartialEq)]
pub struct HdrMip {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f32>,
}

/// Decode a Radiance `.hdr` or OpenEXR file (any other image format also works, as low dynamic range).
pub fn decode_hdr(bytes: &[u8]) -> Result<HdrImage, ImageError> {
    if bytes.is_empty() {
        return Err(ImageError::Empty);
    }
    if bytes.len() > MAX_FILE_BYTES {
        return Err(ImageError::TooLarge);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| ImageError::Decode(e.to_string()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(1 << 30);
    reader.limits(limits);
    let img: DynamicImage = reader.decode().map_err(|e| ImageError::Decode(e.to_string()))?;
    let f = img.to_rgb32f();
    let (width, height) = f.dimensions();
    if width < 2 || height < 2 {
        return Err(ImageError::Decode("the image is too small to be an environment".into()));
    }
    let rgb = f.into_raw().into_iter().map(|v| if v.is_finite() { v.clamp(0.0, MAX_RADIANCE) } else { 0.0 }).collect();
    Ok(HdrImage { width, height, rgb })
}

impl HdrImage {
    /// Box-filter down to at most `max_width` pixels wide, keeping the aspect ratio. Smaller images
    /// are returned unchanged.
    pub fn downsampled(&self, max_width: u32) -> HdrImage {
        if self.width <= max_width.max(2) {
            return self.clone();
        }
        let nw = max_width.max(2);
        let nh = ((self.height as u64 * nw as u64) / self.width as u64).max(1) as u32;
        let mut out = vec![0.0f32; (nw * nh * 3) as usize];
        for y in 0..nh {
            let (y0, y1) = (y * self.height / nh, ((y + 1) * self.height / nh).max(y * self.height / nh + 1));
            for x in 0..nw {
                let (x0, x1) = (x * self.width / nw, ((x + 1) * self.width / nw).max(x * self.width / nw + 1));
                let mut sum = [0.0f32; 3];
                for yy in y0..y1.min(self.height) {
                    for xx in x0..x1.min(self.width) {
                        let i = ((yy * self.width + xx) * 3) as usize;
                        for c in 0..3 {
                            sum[c] += self.rgb[i + c];
                        }
                    }
                }
                let n = ((y1.min(self.height) - y0) * (x1.min(self.width) - x0)) as f32;
                let o = ((y * nw + x) * 3) as usize;
                for c in 0..3 {
                    out[o + c] = sum[c] / n;
                }
            }
        }
        HdrImage { width: nw, height: nh, rgb: out }
    }

    /// The full mip chain down to one pixel high, level 0 first. Each level halves the previous one,
    /// so lower levels stand in for rougher reflections.
    pub fn mips(&self) -> Vec<HdrMip> {
        let to_rgba = |im: &HdrImage| HdrMip {
            width: im.width,
            height: im.height,
            rgba: im.rgb.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 1.0]).collect(),
        };
        let mut levels = vec![to_rgba(self)];
        let mut cur = self.clone();
        while cur.height > 1 && cur.width > 1 {
            let (nw, nh) = ((cur.width / 2).max(1), (cur.height / 2).max(1));
            let mut out = vec![0.0f32; (nw * nh * 3) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    let mut sum = [0.0f32; 3];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((x * 2 + dx).min(cur.width - 1), (y * 2 + dy).min(cur.height - 1));
                        let i = ((sy * cur.width + sx) * 3) as usize;
                        for c in 0..3 {
                            sum[c] += cur.rgb[i + c];
                        }
                    }
                    let o = ((y * nw + x) * 3) as usize;
                    for c in 0..3 {
                        out[o + c] = sum[c] * 0.25;
                    }
                }
            }
            cur = HdrImage { width: nw, height: nh, rgb: out };
            levels.push(to_rgba(&cur));
        }
        levels
    }

    /// Second-order spherical harmonics of the light arriving from every direction, already scaled
    /// so that evaluating them at a surface normal gives the cosine-weighted average radiance (the
    /// irradiance divided by pi). Directions follow the shader: `u = atan2(z, x) / 2pi + 0.5`,
    /// `v = acos(y) / pi`.
    pub fn diffuse_sh(&self) -> [[f32; 3]; 9] {
        // Work from a small copy: the lowest frequencies do not need every pixel.
        let src = self.downsampled(128);
        let mut sh = [[0.0f64; 3]; 9];
        for y in 0..src.height {
            let theta = (y as f64 + 0.5) / src.height as f64 * std::f64::consts::PI;
            let solid = theta.sin() * (std::f64::consts::PI / src.height as f64) * (std::f64::consts::TAU / src.width as f64);
            for x in 0..src.width {
                let phi = ((x as f64 + 0.5) / src.width as f64 - 0.5) * std::f64::consts::TAU;
                let d = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
                let basis = sh_basis(d);
                let i = ((y * src.width + x) * 3) as usize;
                for (k, b) in basis.iter().enumerate() {
                    for c in 0..3 {
                        sh[k][c] += src.rgb[i + c] as f64 * b * solid;
                    }
                }
            }
        }
        // Convolve with the cosine lobe (pi, 2pi/3, pi/4 per band) and divide by pi.
        let band = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
        let mut out = [[0.0f32; 3]; 9];
        for k in 0..9 {
            for c in 0..3 {
                out[k][c] = (sh[k][c] * band[k]) as f32;
            }
        }
        out
    }
}

fn sh_basis(d: [f64; 3]) -> [f64; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Evaluate [`HdrImage::diffuse_sh`] coefficients at a unit direction. The CPU twin of the shader.
pub fn eval_sh(sh: &[[f32; 3]; 9], n: [f32; 3]) -> [f32; 3] {
    let b = sh_basis([n[0] as f64, n[1] as f64, n[2] as f64]);
    let mut out = [0.0f32; 3];
    for k in 0..9 {
        for c in 0..3 {
            out[c] += sh[k][c] * b[k] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: u32, h: u32, v: [f32; 3]) -> HdrImage {
        HdrImage { width: w, height: h, rgb: (0..w * h).flat_map(|_| v).collect() }
    }

    #[test]
    fn a_uniform_environment_lights_every_direction_equally() {
        let sh = flat(64, 32, [0.5, 1.0, 2.0]).diffuse_sh();
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.577, 0.577, 0.577]] {
            let e = eval_sh(&sh, n);
            assert!((e[0] - 0.5).abs() < 0.02 && (e[1] - 1.0).abs() < 0.03 && (e[2] - 2.0).abs() < 0.06, "{n:?}: {e:?}");
        }
    }

    #[test]
    fn a_bright_sky_lights_up_facing_surfaces_more_than_down_facing_ones() {
        // Top half bright, bottom half dark.
        let (w, h) = (64u32, 32u32);
        let rgb = (0..h).flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [1.0; 3] } else { [0.0; 3] })).collect();
        let sh = HdrImage { width: w, height: h, rgb }.diffuse_sh();
        let up = eval_sh(&sh, [0.0, 1.0, 0.0])[0];
        let down = eval_sh(&sh, [0.0, -1.0, 0.0])[0];
        let side = eval_sh(&sh, [1.0, 0.0, 0.0])[0];
        assert!(up > 0.8 && down < 0.2 && side > 0.3 && side < 0.7, "up {up} down {down} side {side}");
    }

    #[test]
    fn mip_chain_halves_down_to_one_pixel_and_keeps_the_average() {
        let (w, h) = (16u32, 8u32);
        let rgb: Vec<f32> = (0..w * h).flat_map(|i| [i as f32, 0.0, 1.0]).collect();
        let mean = (0..w * h).sum::<u32>() as f32 / (w * h) as f32;
        let mips = HdrImage { width: w, height: h, rgb }.mips();
        assert_eq!(mips.len(), 4);
        assert_eq!((mips[1].width, mips[1].height), (8, 4));
        let last = mips.last().unwrap();
        assert_eq!((last.width, last.height), (2, 1));
        let avg = (last.rgba[0] + last.rgba[4]) / 2.0;
        assert!((avg - mean).abs() < 1e-3, "{avg} vs {mean}");
    }

    #[test]
    fn downsampling_keeps_the_aspect_and_the_average() {
        let im = flat(64, 32, [2.0, 3.0, 4.0]).downsampled(16);
        assert_eq!((im.width, im.height), (16, 8));
        assert!(im.rgb.chunks_exact(3).all(|p| (p[0] - 2.0).abs() < 1e-5 && (p[2] - 4.0).abs() < 1e-5));
        assert_eq!(flat(8, 4, [1.0; 3]).downsampled(16).width, 8);
    }

    #[test]
    fn a_radiance_file_decodes_to_linear_radiance() {
        // A 2x2 uncompressed Radiance picture: RGBE (128, 128, 128, 129) is 1.0 on every channel,
        // (128, 64, 0, 130) is (2, 1, 0).
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n".to_vec();
        bytes.extend([128, 128, 128, 129, 128, 64, 0, 130, 128, 128, 128, 129, 128, 128, 128, 129]);
        let img = decode_hdr(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        let px = |i: usize| [img.rgb[i * 3], img.rgb[i * 3 + 1], img.rgb[i * 3 + 2]];
        for (got, want) in px(0).into_iter().zip([1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < 0.02, "{got} vs {want}");
        }
        for (got, want) in px(1).into_iter().zip([2.0, 1.0, 0.0]) {
            assert!((got - want).abs() < 0.03, "{got} vs {want}");
        }
        // Absurd radiance is clamped rather than passed on.
        assert!(img.rgb.iter().all(|v| *v <= MAX_RADIANCE));
    }

    #[test]
    fn junk_is_an_error_not_a_panic() {
        assert!(decode_hdr(&[]).is_err());
        assert!(decode_hdr(b"definitely not an image").is_err());
        assert!(decode_hdr(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 99999999 +X 99999999\n").is_err());
    }
}
