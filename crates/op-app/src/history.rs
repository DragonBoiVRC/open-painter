//! Compact undo snapshots. While a stroke is being painted its snapshot is a full copy of the layer;
//! once the stroke is over, the GPU compares that copy with the result, and only the 64x64 tiles
//! that changed are kept (packed into a small atlas). A stroke usually touches a small part of the
//! texture, so this is a fraction of the memory.

use crate::paint::{Layer, BASE_FORMAT, PAINT_TARGETS, VALUE_FORMAT};
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const TILE: u32 = 64;
const ATLAS_COLS: u32 = 32;
/// Formats of the six paint channels in a layer (base, metal, rough, height, emissive, opacity).
const FORMATS: [wgpu::TextureFormat; PAINT_TARGETS] = [BASE_FORMAT, VALUE_FORMAT, VALUE_FORMAT, VALUE_FORMAT, BASE_FORMAT, VALUE_FORMAT];
/// Bytes per texel of each channel.
const BYTES: [u64; PAINT_TARGETS] = [8, 2, 2, 2, 8, 2];

/// The changed tiles of a layer's six paint channels, before or after a stroke.
pub struct TileSet {
    tiles: Vec<[u32; 2]>,
    atlas: Vec<wgpu::Texture>,
}

fn channels(layer: &Layer) -> [&wgpu::Texture; PAINT_TARGETS] {
    [&layer.base, &layer.metal, &layer.rough, &layer.height, &layer.emissive, &layer.opacity]
}

fn atlas_origin(i: usize) -> wgpu::Origin3d {
    wgpu::Origin3d { x: (i as u32 % ATLAS_COLS) * TILE, y: (i as u32 / ATLAS_COLS) * TILE, z: 0 }
}

impl TileSet {
    pub fn bytes(&self) -> u64 {
        self.tiles.len() as u64 * (TILE * TILE) as u64 * BYTES.iter().sum::<u64>()
    }

    fn new(device: &wgpu::Device, tiles: Vec<[u32; 2]>) -> Self {
        let n = (tiles.len() as u32).max(1);
        let width = n.min(ATLAS_COLS) * TILE;
        let height = n.div_ceil(ATLAS_COLS) * TILE;
        let atlas = FORMATS
            .iter()
            .map(|&format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("undo_tiles"),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            })
            .collect();
        Self { tiles, atlas }
    }

    fn copy(&self, encoder: &mut wgpu::CommandEncoder, layer: &Layer, to_atlas: bool) {
        let extent = wgpu::Extent3d { width: TILE, height: TILE, depth_or_array_layers: 1 };
        for (live, atlas) in channels(layer).into_iter().zip(&self.atlas) {
            for (i, t) in self.tiles.iter().enumerate() {
                let tile = wgpu::TexelCopyTextureInfo { texture: live, mip_level: 0, origin: wgpu::Origin3d { x: t[0] * TILE, y: t[1] * TILE, z: 0 }, aspect: wgpu::TextureAspect::All };
                let packed = wgpu::TexelCopyTextureInfo { texture: atlas, mip_level: 0, origin: atlas_origin(i), aspect: wgpu::TextureAspect::All };
                if to_atlas {
                    encoder.copy_texture_to_texture(tile, packed, extent);
                } else {
                    encoder.copy_texture_to_texture(packed, tile, extent);
                }
            }
        }
    }

    /// Tiles of `layer` at the same places as these.
    pub fn capture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, layer: &Layer) -> TileSet {
        let now = TileSet::new(device, self.tiles.clone());
        now.copy(encoder, layer, true);
        now
    }

    /// Write the tiles back into `layer`.
    pub fn restore(&self, encoder: &mut wgpu::CommandEncoder, layer: &Layer) {
        self.copy(encoder, layer, false);
    }
}

/// The pipeline that finds changed tiles.
pub struct TileDiff {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// A comparison on its way back from the GPU.
pub struct DiffJob {
    pub id: u64,
    flags: wgpu::Buffer,
    readback: wgpu::Buffer,
    tiles_x: u32,
    mapped: bool,
    ready: Arc<AtomicBool>,
}

impl TileDiff {
    pub fn new(device: &wgpu::Device) -> Self {
        let float_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let mut entries: Vec<_> = (0..12).map(float_tex).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("tilediff_layout"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tilediff_pl"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/tilediff.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tilediff"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Whether a layer of this size can be kept as tiles.
    pub fn supports(size: u32) -> bool {
        size >= 2 * TILE && size % TILE == 0
    }

    /// Record a comparison of `before` (the snapshot) with `now` (the live layer).
    pub fn start(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, id: u64, before: &Layer, now: &Layer, size: u32) -> DiffJob {
        let tiles_x = size / TILE;
        let bytes = (tiles_x * tiles_x * 4) as u64;
        let flags = device.create_buffer(&wgpu::BufferDescriptor { label: Some("tilediff_flags"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: Some("tilediff_readback"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tilediff_params"), contents: bytemuck::cast_slice(&[tiles_x, 0u32, 0, 0]), usage: wgpu::BufferUsages::UNIFORM });
        let views: Vec<wgpu::TextureView> = channels(before).into_iter().chain(channels(now)).map(|t| t.create_view(&Default::default())).collect();
        let mut group: Vec<_> = views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(v) }).collect();
        group.push(wgpu::BindGroupEntry { binding: 12, resource: flags.as_entire_binding() });
        group.push(wgpu::BindGroupEntry { binding: 13, resource: params.as_entire_binding() });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("tilediff_bind"), layout: &self.layout, entries: &group });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tilediff_pass"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(tiles_x, tiles_x, 1);
        }
        encoder.copy_buffer_to_buffer(&flags, 0, &readback, 0, bytes);
        DiffJob { id, flags, readback, tiles_x, mapped: false, ready: Arc::new(AtomicBool::new(false)) }
    }
}

impl DiffJob {
    /// Advance the job. Call once per frame, after the frame that recorded it was submitted. Returns
    /// the changed tiles (row-major) once the GPU has answered.
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<Vec<[u32; 2]>> {
        if !self.mapped {
            let ready = self.ready.clone();
            self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| ready.store(r.is_ok(), Ordering::Release));
            self.mapped = true;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        if !self.ready.load(Ordering::Acquire) {
            return None;
        }
        let tiles = {
            let data = self.readback.slice(..).get_mapped_range().ok()?;
            let words: &[u32] = bytemuck::cast_slice(&data);
            words.iter().enumerate().filter(|(_, &w)| w != 0).map(|(i, _)| [i as u32 % self.tiles_x, i as u32 / self.tiles_x]).collect()
        };
        self.readback.unmap();
        let _ = &self.flags;
        Some(tiles)
    }
}

/// Keep the changed `tiles` of `before` as a compact snapshot, copied from the snapshot itself.
pub fn compact(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, before: &Layer, tiles: Vec<[u32; 2]>) -> TileSet {
    let set = TileSet::new(device, tiles);
    set.copy(encoder, before, true);
    set
}
