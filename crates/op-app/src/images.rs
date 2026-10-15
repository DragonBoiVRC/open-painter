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
    pub size: (u32, u32),
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
    GpuImage { texture, srgb, linear, size: (w, h) }
}

/// A 1x1 stand-in bound where a pass has no image for a slot.
pub fn dummy_image(device: &wgpu::Device) -> GpuImage {
    make(device, 1, 1, 1)
}

/// Upload `img` with a generated mip chain.
pub fn upload_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &Decoded) -> GpuImage {
    let mips = op_image::mip_chain(img);
    let gpu = make(device, img.width, img.height, mips.len() as u32);
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

/// Images by id, shared by every paint store.
#[derive(Default)]
pub struct ImageLibrary {
    map: HashMap<String, Arc<GpuImage>>,
}

pub type SharedImages = Arc<Mutex<ImageLibrary>>;

impl ImageLibrary {
    pub fn contains(&self, id: &str) -> bool {
        self.map.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<Arc<GpuImage>> {
        self.map.get(id).cloned()
    }

    pub fn insert(&mut self, id: &str, image: GpuImage) {
        self.map.insert(id.to_string(), Arc::new(image));
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}
