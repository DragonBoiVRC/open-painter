//! Texture images on the GPU: the library fill layers sample from.

use eframe::egui_wgpu::wgpu;
use op_image::Decoded;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One uploaded image with its full mip chain, readable as sRGB (colour) or linear (data) texels.
pub struct GpuImage {
    pub texture: wgpu::Texture,
    pub srgb: wgpu::TextureView,
    pub linear: wgpu::TextureView,
}

fn make(device: &wgpu::Device, w: u32, h: u32, mips: u32) -> GpuImage {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("library_image"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let srgb = texture.create_view(&wgpu::TextureViewDescriptor { format: Some(wgpu::TextureFormat::Rgba8UnormSrgb), ..Default::default() });
    let linear = texture.create_view(&Default::default());
    GpuImage { texture, srgb, linear }
}

/// A 1x1 stand-in bound where a pass has no image for a slot.
pub fn dummy_image(device: &wgpu::Device) -> GpuImage {
    make(device, 1, 1, 1)
}

/// How an upload went.
pub enum Upload {
    Full(GpuImage),
    /// The GPU was nearly full, so the picture was kept at half its width and height.
    Reduced(GpuImage),
    /// Not even the smaller copy fitted.
    Failed,
}

/// Upload `img` with a generated mip chain. A GPU that is out of memory does not crash the app: the
/// picture is tried at half size (a quarter of the memory), and then given up on.
pub fn upload_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &Decoded) -> Upload {
    let mips = op_image::mip_chain(img);
    for skip in 0..2 {
        if skip >= mips.len() {
            break;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let gpu = write_levels(device, queue, &mips[skip..]);
        if pollster::block_on(scope.pop()).is_none() {
            return if skip == 0 { Upload::Full(gpu) } else { Upload::Reduced(gpu) };
        }
    }
    Upload::Failed
}

fn write_levels(device: &wgpu::Device, queue: &wgpu::Queue, mips: &[op_image::Mip]) -> GpuImage {
    let gpu = make(device, mips[0].width, mips[0].height, mips.len() as u32);
    for (level, m) in mips.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &m.rgba8,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(m.width * 4), rows_per_image: Some(m.height) },
            wgpu::Extent3d { width: m.width, height: m.height, depth_or_array_layers: 1 },
        );
    }
    gpu
}

/// The pictures a brush stroke reads: an alpha that shapes the stamp, and a stencil laid over the
/// screen that the stroke paints through.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BrushPictures {
    /// Library id of the stamp's alpha picture (the brush shape "Image").
    pub alpha: Option<String>,
    /// Library id of the stencil picture.
    pub stencil: Option<String>,
    /// Where the stencil sits in the viewport: centre (fractions of width/height), height as a
    /// fraction of the viewport height, rotation in radians, mirrored horizontally.
    pub stencil_place: [f32; 4],
    pub stencil_flip: bool,
}

/// Images by id, shared by every paint store.
#[derive(Default)]
pub struct ImageLibrary {
    map: HashMap<String, Arc<GpuImage>>,
    /// Pictures the GPU had no room for; they are not retried every frame.
    failed: std::collections::HashSet<String>,
    pub brush: BrushPictures,
}

pub type SharedImages = Arc<Mutex<ImageLibrary>>;

impl ImageLibrary {
    pub fn contains(&self, id: &str) -> bool {
        self.map.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<Arc<GpuImage>> {
        self.map.get(id).cloned()
    }

    pub fn mark_failed(&mut self, id: &str) {
        self.failed.insert(id.to_string());
    }

    pub fn has_failed(&self, id: &str) -> bool {
        self.failed.contains(id)
    }

    pub fn insert(&mut self, id: &str, image: GpuImage) {
        self.map.insert(id.to_string(), Arc::new(image));
    }
}
