//! Texture images for fill layers: decoding with limits, content ids, mip chains, thumbnails and
//! guessing which channel an image is meant for from its file name. No GPU code here.

pub mod hdr;

use image::{imageops::FilterType, DynamicImage, ImageReader, RgbaImage};
use op_core::Slot;
use std::io::Cursor;

/// The largest image side we accept. Larger textures would not fit the GPU limits we rely on.
pub const MAX_SIDE: u32 = 8192;
/// Refuse files larger than this before decoding.
pub const MAX_FILE_BYTES: usize = 256 << 20;

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("the file is empty")]
    Empty,
    #[error("the file is larger than {} MB", MAX_FILE_BYTES >> 20)]
    TooLarge,
    #[error("the image is {0} x {1}; the largest supported side is {MAX_SIDE}")]
    TooBig(u32, u32),
    #[error("not a supported image: {0}")]
    Decode(String),
}

/// An image as 8-bit RGBA, top row first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Decode PNG, JPEG, TGA, BMP, TIFF, WebP or OpenEXR bytes. 16-bit and floating point images are
/// reduced to 8 bits per channel.
pub fn decode(bytes: &[u8]) -> Result<Decoded, ImageError> {
    if bytes.is_empty() {
        return Err(ImageError::Empty);
    }
    if bytes.len() > MAX_FILE_BYTES {
        return Err(ImageError::TooLarge);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| ImageError::Decode(e.to_string()))?;
    // The decoder checks these before it allocates, so a hostile header cannot ask for gigabytes.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(1 << 30);
    reader.limits(limits);
    let img: DynamicImage = reader.decode().map_err(|e| ImageError::Decode(e.to_string()))?;
    let rgba: RgbaImage = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Decode("the image has no pixels".into()));
    }
    Ok(Decoded { width, height, rgba8: rgba.into_raw() })
}

/// A stable id for an image: two FNV-1a hashes of its bytes, as 32 hex digits. Identical files get
/// the same id, so importing a texture twice does not duplicate it.
pub fn content_id(bytes: &[u8]) -> String {
    fn fnv(bytes: &[u8], seed: u64) -> u64 {
        let mut h = seed;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
    format!("{:016x}{:016x}", fnv(bytes, 0xcbf2_9ce4_8422_2325), fnv(bytes, 0x8422_2325_cbf2_9ce4 ^ bytes.len() as u64))
}

/// True for strings that look like ids made by [`content_id`]: what a project file may name.
pub fn is_valid_id(id: &str) -> bool {
    op_core::is_valid_texture_id(id)
}

/// One level of a mip chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mip {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// The image and every smaller level down to 1 x 1, each averaged from the one above. Averaging
/// premultiplies by alpha so transparent texels do not bleed their colour into opaque ones.
pub fn mip_chain(img: &Decoded) -> Vec<Mip> {
    let mut levels = vec![Mip { width: img.width, height: img.height, rgba8: img.rgba8.clone() }];
    while let Some(last) = levels.last() {
        if last.width == 1 && last.height == 1 {
            break;
        }
        let (w, h) = ((last.width / 2).max(1), (last.height / 2).max(1));
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                // The (up to) four source texels that this one covers.
                let xs = [(x * 2).min(last.width - 1), (x * 2 + 1).min(last.width - 1)];
                let ys = [(y * 2).min(last.height - 1), (y * 2 + 1).min(last.height - 1)];
                let (mut rgb, mut a) = ([0.0f32; 3], 0.0f32);
                for &sy in &ys {
                    for &sx in &xs {
                        let i = ((sy * last.width + sx) * 4) as usize;
                        let alpha = last.rgba8[i + 3] as f32;
                        for c in 0..3 {
                            rgb[c] += last.rgba8[i + c] as f32 * alpha;
                        }
                        a += alpha;
                    }
                }
                let o = ((y * w + x) * 4) as usize;
                for c in 0..3 {
                    data[o + c] = if a > 0.0 { (rgb[c] / a).round() as u8 } else { 0 };
                }
                data[o + 3] = (a / 4.0).round() as u8;
            }
        }
        levels.push(Mip { width: w, height: h, rgba8: data });
    }
    levels
}

/// A `size` x `size` (at most) copy that keeps the aspect ratio, for picking and previews.
pub fn thumbnail(img: &Decoded, size: u32) -> Decoded {
    let src = RgbaImage::from_raw(img.width, img.height, img.rgba8.clone()).expect("a decoded image has consistent dimensions");
    let scale = (size as f32 / img.width.max(img.height) as f32).min(1.0);
    let (w, h) = (((img.width as f32 * scale).round() as u32).max(1), ((img.height as f32 * scale).round() as u32).max(1));
    let out = image::imageops::resize(&src, w, h, FilterType::Triangle);
    Decoded { width: w, height: h, rgba8: out.into_raw() }
}

/// Which channel a file is probably for, and whether a normal map follows the DirectX convention
/// (green pointing down) rather than OpenGL's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guess {
    pub slot: Slot,
    pub directx: bool,
}

/// Look at the words in a file name (`wood_planks_Normal_GL_4k.png`) and guess the channel. The
/// last recognisable word wins, since the channel normally ends the name.
pub fn guess_slot(file_name: &str) -> Option<Guess> {
    let stem = file_name.rsplit_once('.').map_or(file_name, |(s, _)| s);
    // Split on anything that is not a letter or digit, and at lower-to-upper case changes.
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in stem.chars() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        prev_lower = c.is_lowercase();
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    let directx = stem.to_lowercase().contains("directx") || words.iter().any(|w| matches!(w.as_str(), "dx" | "d3d"));
    let mut found = None;
    for (i, w) in words.iter().enumerate() {
        let slot = match w.as_str() {
            "basecolor" | "basecolour" | "albedo" | "diffuse" | "diff" | "color" | "colour" | "col" | "base" | "alb" => Some(Slot::BaseColor),
            "normal" | "normals" | "nrm" | "norm" | "nor" | "nml" | "normalmap" => Some(Slot::Normal),
            "height" | "displacement" | "disp" | "bump" | "heightmap" | "displace" => Some(Slot::Height),
            "roughness" | "rough" | "rgh" | "roughnessmap" => Some(Slot::Roughness),
            "metallic" | "metalness" | "metal" | "mtl" | "metallicmap" => Some(Slot::Metallic),
            "emissive" | "emission" | "emit" | "glow" | "emissivemap" => Some(Slot::Emissive),
            // "n" alone is a normal map only at the end ("brick_n.png").
            "n" if i + 1 == words.len() && i > 0 => Some(Slot::Normal),
            _ => None,
        };
        // "base" followed by "color" is one word to a human; either way it is the base color.
        if slot.is_some() {
            found = slot;
        }
    }
    found.map(|slot| Guess { slot, directx: directx && slot == Slot::Normal })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut img = RgbaImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba(f(x, y));
        }
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png).unwrap();
        out
    }

    #[test]
    fn png_round_trips_with_its_pixels_in_row_order() {
        let bytes = png(3, 2, |x, y| [x as u8 * 50, y as u8 * 100, 7, 255]);
        let d = decode(&bytes).unwrap();
        assert_eq!((d.width, d.height), (3, 2));
        assert_eq!(&d.rgba8[..4], &[0, 0, 7, 255], "top-left first");
        assert_eq!(&d.rgba8[(3 * 4)..(3 * 4 + 4)], &[0, 100, 7, 255], "the second row starts after the first");
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(matches!(decode(&[]), Err(ImageError::Empty)));
        assert!(matches!(decode(b"this is not an image"), Err(ImageError::Decode(_))));
        let good = png(4, 4, |_, _| [1, 2, 3, 255]);
        for cut in [4, 20, good.len() / 2] {
            assert!(decode(&good[..cut]).is_err(), "truncated at {cut}");
        }
        let mut corrupt = good.clone();
        for b in corrupt.iter_mut().skip(40).step_by(3) {
            *b ^= 0xA5;
        }
        let _ = decode(&corrupt); // may fail or decode garbage, but must not panic
    }

    #[test]
    fn oversized_images_are_refused() {
        // A PNG header that claims 20000 x 20000 pixels, without the pixels.
        let mut bytes = png(1, 1, |_, _| [0, 0, 0, 255]);
        // IHDR width/height live at offsets 16..24 (big endian); the CRC will not match, so a decoder that
        // checks it fails first - either way the result must be an error, never a huge allocation.
        bytes[16..20].copy_from_slice(&20000u32.to_be_bytes());
        bytes[20..24].copy_from_slice(&20000u32.to_be_bytes());
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn ids_follow_the_content() {
        let a = content_id(b"hello");
        assert_eq!(a, content_id(b"hello"));
        assert_ne!(a, content_id(b"hellp"));
        assert_ne!(content_id(b""), content_id(b"\0"));
        assert!(is_valid_id(&a));
        for bad in ["", "abc", &"g".repeat(32), &"A".repeat(32), "../../etc/passwd/../../../../../x"] {
            assert!(!is_valid_id(bad), "{bad}");
        }
    }

    #[test]
    fn mip_chains_halve_down_to_one_pixel_and_average() {
        let d = Decoded { width: 4, height: 2, rgba8: (0..8).flat_map(|i| [if i % 2 == 0 { 0 } else { 200 }, 100, 50, 255]).collect() };
        let chain = mip_chain(&d);
        let sizes: Vec<_> = chain.iter().map(|m| (m.width, m.height)).collect();
        assert_eq!(sizes, [(4, 2), (2, 1), (1, 1)]);
        assert_eq!(chain[1].rgba8[0], 100, "two black and two 200 texels average to 100");
        assert_eq!(chain.last().unwrap().rgba8, [100, 100, 50, 255]);
        // Odd and one-wide sizes still terminate.
        assert_eq!(mip_chain(&Decoded { width: 5, height: 3, rgba8: vec![9; 60] }).len(), 3);
        assert_eq!(mip_chain(&Decoded { width: 1, height: 1, rgba8: vec![1, 2, 3, 4] }).len(), 1);
        assert_eq!(mip_chain(&Decoded { width: 1, height: 7, rgba8: vec![9; 28] }).len(), 3, "7 -> 3 -> 1");
    }

    #[test]
    fn transparent_texels_do_not_bleed_colour_into_mips() {
        // One opaque red texel next to a transparent green one.
        let d = Decoded { width: 2, height: 1, rgba8: vec![255, 0, 0, 255, 0, 255, 0, 0] };
        let top = &mip_chain(&d)[1];
        assert_eq!(&top.rgba8[..3], &[255, 0, 0], "the colour is that of the visible texel");
        assert_eq!(top.rgba8[3], 128);
    }

    #[test]
    fn thumbnails_keep_the_aspect_ratio_and_never_enlarge() {
        let big = Decoded { width: 200, height: 100, rgba8: vec![128; 200 * 100 * 4] };
        let t = thumbnail(&big, 64);
        assert_eq!((t.width, t.height), (64, 32));
        let small = Decoded { width: 10, height: 10, rgba8: vec![1; 400] };
        assert_eq!((thumbnail(&small, 64).width, thumbnail(&small, 64).height), (10, 10));
    }

    #[test]
    fn file_names_say_which_channel_they_are_for() {
        let g = |n: &str| guess_slot(n).map(|g| (g.slot, g.directx));
        assert_eq!(g("wood_planks_BaseColor.png"), Some((Slot::BaseColor, false)));
        assert_eq!(g("Brick-Albedo.jpg"), Some((Slot::BaseColor, false)));
        assert_eq!(g("metal_diffuse_4k.png"), Some((Slot::BaseColor, false)));
        assert_eq!(g("rock_Normal_GL.png"), Some((Slot::Normal, false)));
        assert_eq!(g("rock_Normal_DX.png"), Some((Slot::Normal, true)));
        assert_eq!(g("rock_normalDirectX.png"), Some((Slot::Normal, true)));
        assert_eq!(g("tile_n.png"), Some((Slot::Normal, false)));
        assert_eq!(g("tile_height.exr"), Some((Slot::Height, false)));
        assert_eq!(g("sand_Displacement.png"), Some((Slot::Height, false)));
        assert_eq!(g("sand_Roughness.png"), Some((Slot::Roughness, false)));
        assert_eq!(g("sand_Metalness.png"), Some((Slot::Metallic, false)));
        assert_eq!(g("lamp_Emissive.png"), Some((Slot::Emissive, false)));
        assert_eq!(g("neon_glow.jpg"), Some((Slot::Emissive, false)));
        // The last word that names a channel wins.
        assert_eq!(g("normal_wall_roughness.png"), Some((Slot::Roughness, false)));
        // A bare "n", an unknown name and an AO map give no guess.
        assert_eq!(g("n.png"), None);
        assert_eq!(g("photo_0042.jpg"), None);
        assert_eq!(g("wall_ambientOcclusion.png"), None);
        // DirectX only matters for normal maps.
        assert_eq!(g("wall_roughness_dx.png"), Some((Slot::Roughness, false)));
    }
}
