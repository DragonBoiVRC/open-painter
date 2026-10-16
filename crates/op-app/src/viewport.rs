//! 3D viewport: renders the model into an offscreen target, then blits it into
//! the egui panel. egui's own pass has no depth buffer, hence the offscreen step.

use eframe::egui_wgpu::{self, wgpu, wgpu::util::DeviceExt};
use crate::paint::{MeshBuffers, PaintGpu, PaintOp, StampView};
use op_core::Document;
use op_mesh::{Mesh, Vertex};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub base_color: [f32; 4],
    pub params: [f32; 4],
    /// Environment: rotation (radians), exposure scale, direct light intensity, preset.
    pub env: [f32; 4],
}

struct Offscreen {
    size: [u32; 2],
    color: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blit_bind: wgpu::BindGroup,
}

fn preview_texture_desc(size: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("preview"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

fn make_preview_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("preview_bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// GPU objects shared between frames; lives in egui's callback resources.
/// The paint stores of one texture set: one per UV tile (a single one unless UV tiles are in use).
/// Dereferences to the first tile, so code that only knows single-tile sets keeps working.
pub struct SetGpu {
    pub tiles: Vec<PaintGpu>,
    /// The index-buffer run (first index, count) of each tile's triangles.
    pub tile_ranges: Vec<(u32, u32)>,
}

impl std::ops::Deref for SetGpu {
    type Target = PaintGpu;
    fn deref(&self) -> &PaintGpu {
        &self.tiles[0]
    }
}

impl std::ops::DerefMut for SetGpu {
    fn deref_mut(&mut self) -> &mut PaintGpu {
        &mut self.tiles[0]
    }
}

/// 2D stamps are given in global UV space; a tile's store works in tile-local UVs.
fn ops_for_tile(ops: &[PaintOp], tile: op_mesh::Tile) -> std::borrow::Cow<'_, [PaintOp]> {
    if (tile.u == 0 && tile.v == 0) || !ops.iter().any(|o| matches!(o, PaintOp::Stamps2d { .. })) {
        return std::borrow::Cow::Borrowed(ops);
    }
    let shifted = ops
        .iter()
        .map(|op| match op {
            PaintOp::Stamps2d { layer, target, stamps } => {
                let moved = stamps
                    .iter()
                    .map(|s| {
                        let mut s = *s;
                        s.brush[0] -= tile.u as f32;
                        s.brush[1] -= tile.v as f32;
                        s
                    })
                    .collect();
                PaintOp::Stamps2d { layer: *layer, target: *target, stamps: moved }
            }
            other => other.clone(),
        })
        .collect();
    std::borrow::Cow::Owned(shifted)
}

/// Apply one batch to the tiles of its set. `view` is the 3D camera state (a dummy when only the
/// 2D view is drawn, in which case `mesh_bufs` is `None` and 3D strokes are skipped).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_batch(
    sets: &mut [SetGpu],
    mesh_bufs: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    b: &SetBatch,
    view: &StampView<'_>,
) {
    let Some(set) = sets.get_mut(b.set) else { return };
    let ranges = set.tile_ranges.clone();
    for (ti, paint) in set.tiles.iter_mut().enumerate() {
        if b.tile.is_some_and(|t| t != ti) {
            continue;
        }
        let (first_index, index_count) = ranges.get(ti).copied().unwrap_or((0, 0));
        let mesh = mesh_bufs.map(|(vertices, indices)| MeshBuffers { vertices, indices, first_index, index_count });
        let ops = ops_for_tile(&b.ops, paint.tile);
        paint.apply(device, queue, encoder, &b.doc, &ops, mesh, view);
    }
}

pub struct ViewportGpu {
    pbr: wgpu::RenderPipeline,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    pub(crate) vbuf: Option<wgpu::Buffer>,
    ibuf: Option<wgpu::Buffer>,
    index_count: u32,
    /// Draw ranges (first index, index count) of each texture set, in set order.
    pub(crate) ranges: Vec<(u32, u32)>,
    /// A copy of the globals with the baked-map preview switched off, for sets that are not being viewed.
    plain_buf: wgpu::Buffer,
    plain_bind: wgpu::BindGroup,
    offscreen: Option<Offscreen>,
    /// One paint store per texture set.
    pub sets: Vec<SetGpu>,
    /// The set being edited and previewed.
    pub active: usize,
    /// Texture sets left out of the 3D view (indexed by set; missing entries are shown).
    pub hidden: Vec<bool>,
    /// Batches waiting for the next rendered frame (set up by the app outside of frames).
    pub(crate) queued: Vec<SetBatch>,
    thumbnailer: crate::thumbs::Thumbnailer,
    pub(crate) uv: crate::uvview::UvGpu,
    /// Textures that fill layers read, shared by every paint store.
    pub(crate) images: crate::images::SharedImages,
    /// CPU copy of the mesh indices, for building the UV wireframe of a set.
    pub(crate) cpu_indices: Vec<u32>,
    /// A 1x1 depth texture for applying paint when the 3D view is not drawn.
    pub(crate) dummy_depth: wgpu::Texture,
    preview_layout: wgpu::BindGroupLayout,
    preview_sampler: wgpu::Sampler,
    /// The neutral preview (a white texel), shown for everything that has no baked map to show.
    preview_bind: wgpu::BindGroup,
    /// Baked-map previews of the active set, one per tile.
    preview_tiles: Vec<wgpu::BindGroup>,
}

impl ViewportGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        let images = crate::images::SharedImages::default();
        let mut paint = PaintGpu::new(device, crate::paint::DEFAULT_TEXTURE_SIZE);
        paint.set_images(images.clone());
        let paint_layout = paint.material_layout.clone();
        let plain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_plain"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plain_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_plain_bind"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: plain_buf.as_entire_binding() }],
        });
        let preview_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let preview_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        // Until a map is chosen the preview is a 1x1 white texel (also what AO mode multiplies by).
        let white_view = device
            .create_texture_with_data(
                queue,
                &preview_texture_desc(1),
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            )
            .create_view(&Default::default());
        let preview_bind = make_preview_bind(device, &preview_layout, &white_view, &preview_sampler);
        let pbr_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/pbr.wgsl"));
        let pbr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_pl"),
            bind_group_layouts: &[Some(&globals_layout), Some(&paint.material_layout), Some(&preview_layout)],
            immediate_size: 0,
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        let pbr = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr"),
            layout: Some(&pbr_pl),
            vertex: wgpu::VertexState {
                module: &pbr_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &pbr_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(COLOR_FORMAT.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/blit.wgsl"));
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pbr,
            blit,
            blit_layout,
            sampler,
            globals_buf,
            globals_bind,
            vbuf: None,
            ibuf: None,
            index_count: 0,
            ranges: Vec::new(),
            plain_buf,
            plain_bind,
            offscreen: None,
            sets: vec![SetGpu { tiles: vec![paint], tile_ranges: vec![(0, 0)] }],
            active: 0,
            hidden: Vec::new(),
            queued: Vec::new(),
            thumbnailer: crate::thumbs::Thumbnailer::new(device),
            uv: crate::uvview::UvGpu::new(device, target_format, &paint_layout),
            images,
            cpu_indices: Vec::new(),
            dummy_depth: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy_depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }),
            preview_layout,
            preview_sampler,
            preview_bind,
            preview_tiles: Vec::new(),
        }
    }

    fn preview_bind_for(&self, device: &wgpu::Device, queue: &wgpu::Queue, image: Option<(u32, &[u8])>) -> wgpu::BindGroup {
        let tex = match image {
            Some((size, rgba8)) if rgba8.len() == (size * size * 4) as usize => device.create_texture_with_data(
                queue,
                &preview_texture_desc(size),
                wgpu::util::TextureDataOrder::LayerMajor,
                rgba8,
            ),
            _ => device.create_texture_with_data(
                queue,
                &preview_texture_desc(1),
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            ),
        };
        make_preview_bind(device, &self.preview_layout, &tex.create_view(&Default::default()), &self.preview_sampler)
    }

    /// Show `rgba8` (a `size` x `size` image) on the model, or go back to the neutral preview.
    #[cfg(test)]
    pub fn set_preview(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: Option<(u32, &[u8])>) {
        self.set_previews(device, queue, &[image]);
    }

    /// Baked-map previews of the active set, one per tile (tiles beyond the list show nothing).
    pub fn set_previews(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, images: &[Option<(u32, &[u8])>]) {
        let binds: Vec<_> = images.iter().map(|img| self.preview_bind_for(device, queue, *img)).collect();
        self.preview_bind = self.preview_bind_for(device, queue, None);
        self.preview_tiles = binds;
    }

    #[cfg(test)]
    pub fn set_mesh(&mut self, device: &wgpu::Device, mesh: &Mesh) {
        self.set_mesh_tiles(device, mesh, false);
    }

    /// Upload a mesh. With `udim` each texture set gets one paint store per UV tile it occupies and
    /// `mesh` must have been through `partition_tiles`; without it each set is a single tile.
    pub fn set_mesh_tiles(&mut self, device: &wgpu::Device, mesh: &Mesh, udim: bool) {
        self.vbuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.ibuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.index_count = mesh.indices.len() as u32;
        self.cpu_indices = mesh.indices.clone();
        self.uv.generation += 1;
        self.ranges = mesh.set_ranges().iter().map(|s| (s.first_index, s.index_count)).collect();
        let layout = mesh.unit_ranges(udim);
        // Keep the stores that are still wanted (a store is identified by its position), add the
        // missing ones at the default size (a `Reset`/`Load` op gives the real size) and drop the rest.
        let mut old: Vec<Vec<PaintGpu>> = std::mem::take(&mut self.sets).into_iter().map(|s| s.tiles).collect();
        old.reverse();
        for set in 0..self.ranges.len() {
            let wanted: Vec<_> = layout.iter().filter(|r| r.set == set).collect();
            let mut have = old.pop().unwrap_or_default();
            have.reverse();
            let mut tiles = Vec::new();
            for r in &wanted {
                let mut paint = have.pop().unwrap_or_else(|| PaintGpu::new(device, crate::paint::DEFAULT_TEXTURE_SIZE));
                paint.set_images(self.images.clone());
                paint.set_tile(device, r.tile);
                tiles.push(paint);
            }
            self.sets.push(SetGpu { tile_ranges: wanted.iter().map(|r| (r.first_index, r.index_count)).collect(), tiles });
        }
        self.active = self.active.min(self.sets.len().saturating_sub(1));
        self.preview_tiles.clear();
    }

    /// 32x32 previews of the wanted layer textures of `set`: `(node, is_mask)` -> RGBA8 bytes.
    pub fn thumbnails(&self, device: &wgpu::Device, queue: &wgpu::Queue, set: usize, wanted: &[(u64, bool)]) -> Vec<((u64, bool), Vec<u8>)> {
        let Some(paint) = self.sets.get(set) else { return Vec::new() };
        wanted
            .iter()
            .filter_map(|&(node, mask)| {
                let g = paint.store.get(&node)?;
                let tex = if mask { g.mask.as_ref()? } else { &g.content.as_ref()?.base };
                Some(((node, mask), self.thumbnailer.render(device, queue, tex, mask)))
            })
            .collect()
    }

    /// Reallocate one set's paint store at `size` and composite `doc` immediately.
    pub fn rebuild_set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, set: usize, doc: &Document, size: u32) {
        if let Some(s) = self.sets.get_mut(set) {
            for p in &mut s.tiles {
                p.rebuild(device, queue, doc, size);
            }
        }
    }

    /// Make `image` available to fill layers under `id`. Uploading the same id again keeps the first.
    pub fn upload_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: &str, image: &op_image::Decoded) {
        let mut lib = self.images.lock().expect("image library lock");
        if !lib.contains(id) {
            lib.insert(id, crate::images::upload_image(device, queue, image));
            drop(lib);
            // Layers that were composited without it must be composited again.
            for set in &mut self.sets {
                for tile in &mut set.tiles {
                    tile.invalidate();
                }
            }
        }
    }

    pub fn has_image(&self, id: &str) -> bool {
        self.images.lock().expect("image library lock").contains(id)
    }

    /// Hold `batch` until the next frame is rendered; queued batches run before that frame's own.
    pub fn enqueue(&mut self, batch: SetBatch) {
        self.queued.push(batch);
    }

    /// Forget queued batches (they belong to a mesh that was replaced).
    pub fn clear_queue(&mut self) {
        self.queued.clear();
    }

    /// The paint store of the set being edited.
    #[cfg(test)]
    pub fn paint(&self) -> &PaintGpu {
        &self.sets[self.active]
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.offscreen.as_ref().is_some_and(|o| o.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            // COPY_SRC lets tests and screenshots read the rendered frame back.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let blit_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.offscreen =
            Some(Offscreen { size, color, depth_view: depth.create_view(&Default::default()), blit_bind });
    }


    /// Single-set convenience used by tests: apply `ops` to set 0 and draw.
    #[cfg(test)]
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, globals: &Globals, size: [u32; 2], doc: &Document, ops: &[PaintOp]) {
        let batch = [SetBatch { tile: None, set: 0, doc: doc.clone(), ops: ops.to_vec() }];
        self.render_sets(device, queue, encoder, globals, size, &batch);
    }

    /// Apply each batch to its set's paint store, then draw every set with its own material.
    pub fn render_sets(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, globals: &Globals, size: [u32; 2], batches: &[SetBatch]) {
        self.ensure_offscreen(device, size);
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(globals));
        let mut plain = *globals;
        plain.light_dir[3] = 0.0;
        queue.write_buffer(&self.plain_buf, 0, bytemuck::bytes_of(&plain));
        let off = self.offscreen.as_ref().unwrap();

        // Paint first, using last frame's depth (the camera is static while painting),
        // so this frame's scene pass already shows the new strokes.
        let depth = off.depth_view.clone();
        let queued = std::mem::take(&mut self.queued);
        let view = StampView {
            view_proj: globals.view_proj,
            camera_pos: globals.camera_pos,
            size_px: size,
            near_far: [globals.params[2], globals.params[3]],
            depth: &depth,
        };
        let bufs = match (&self.vbuf, &self.ibuf) {
            (Some(v), Some(i)) => Some((v, i)),
            _ => None,
        };
        for b in queued.iter().chain(batches) {
            apply_batch(&mut self.sets, bufs, device, queue, encoder, b, &view);
        }

        let color_view = off.color.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.09, g: 0.09, b: 0.10, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &off.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let (Some(v), Some(i)) = (&self.vbuf, &self.ibuf) {
            pass.set_pipeline(&self.pbr);
            pass.set_vertex_buffer(0, v.slice(..));
            pass.set_index_buffer(i.slice(..), wgpu::IndexFormat::Uint32);
            for (set, unit) in self.sets.iter().enumerate() {
                if self.hidden.get(set).copied().unwrap_or(false) {
                    continue;
                }
                // The baked-map preview belongs to the set being edited; the others are shaded normally.
                let globals = if set == self.active { &self.globals_bind } else { &self.plain_bind };
                pass.set_bind_group(0, globals, &[]);
                for (ti, paint) in unit.tiles.iter().enumerate() {
                    let Some(&(first, count)) = unit.tile_ranges.get(ti) else { continue };
                    if count == 0 {
                        continue;
                    }
                    let preview = if set == self.active { self.preview_tiles.get(ti).unwrap_or(&self.preview_bind) } else { &self.preview_bind };
                    pass.set_bind_group(1, paint.material_bind(), &[]);
                    pass.set_bind_group(2, preview, &[]);
                    pass.draw_indexed(first..first + count, 0, 0..1);
                }
            }
        }
    }
}

/// Paint operations for one texture set, with that set's document as of when they were issued.
#[derive(Clone)]
pub struct SetBatch {
    pub set: usize,
    /// Only this tile of the set (`None`: every tile).
    pub tile: Option<usize>,
    pub doc: Document,
    pub ops: Vec<PaintOp>,
}

/// Per-frame paint callback handed to egui.
pub struct ViewportCallback {
    pub globals: Globals,
    pub size_px: [u32; 2],
    pub batches: Vec<SetBatch>,
    /// The set shown with the baked-map preview.
    pub active: usize,
    pub hidden: Vec<bool>,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = resources.get_mut::<ViewportGpu>() {
            gpu.active = self.active.min(gpu.sets.len().saturating_sub(1));
            gpu.hidden = self.hidden.clone();
            gpu.render_sets(device, queue, encoder, &self.globals, self.size_px, &self.batches);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<ViewportGpu>() else { return };
        let Some(off) = &gpu.offscreen else { return };
        pass.set_pipeline(&gpu.blit);
        pass.set_bind_group(0, &off.blit_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::paint::{Stamp, Target};
    use glam::Vec3;
    use op_core::{BlendMode, FillParams, Mask, NodeKind};

    const SIZE: u32 = 256;

    fn gpu_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Everything a test needs to drive the viewport headlessly.
    struct Rig {
        device: wgpu::Device,
        queue: wgpu::Queue,
        gpu: ViewportGpu,
        globals: Globals,
        front: Vec3,
    }

    impl Rig {
        fn new() -> Option<Self> {
            let (device, queue) = gpu_device()?;
            let mut gpu = ViewportGpu::new(&device, &queue, COLOR_FORMAT);
            gpu.set_mesh(&device, &Mesh::uv_sphere(1.0, 96, 48));
            let mut cam = OrbitCamera::default();
            cam.frame(Vec3::splat(-1.0), Vec3::splat(1.0));
            let (near, far) = cam.near_far();
            let cp = cam.position();
            let globals = Globals {
                view_proj: cam.view_proj(1.0).to_cols_array_2d(),
                camera_pos: [cp.x, cp.y, cp.z, 1.0],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                base_color: [0.18; 4],
                params: [0.0, 0.5, near, far],
                env: [0.0, 1.0, 1.0, 0.0],
            };
            Some(Self { device, queue, gpu, globals, front: cp.normalize() })
        }

        fn run(&mut self, doc: &Document, ops: &[PaintOp]) {
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.gpu.render(&self.device, &self.queue, &mut enc, &self.globals, [SIZE, SIZE], doc, ops);
            self.queue.submit([enc.finish()]);
        }

        /// Raw bytes of a SIZE x SIZE texture with `bpp` bytes per texel.
        fn read(&self, tex: &wgpu::Texture, bpp: u32) -> Vec<u8> {
            crate::gpu_io::read_texture(&self.device, &self.queue, tex, SIZE, bpp)
        }

        fn metal(&self) -> Vec<u8> {
            self.read(&self.gpu.paint().output.metal, 2)
        }
    }

    fn texel_rg(data: &[u8], uv: (f32, f32)) -> (u8, u8) {
        let x = ((uv.0 * SIZE as f32) as u32).min(SIZE - 1);
        let y = ((uv.1 * SIZE as f32) as u32).min(SIZE - 1);
        let i = ((y * SIZE + x) * 2) as usize;
        (data[i], data[i + 1])
    }

    /// Inverse of `Mesh::uv_sphere`'s parameterisation.
    fn sphere_uv(n: Vec3) -> (f32, f32) {
        let v = n.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
        let u = n.z.atan2(n.x).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
        (u, v)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let (sign, exp, frac) = ((h >> 15) & 1, (h >> 10) & 0x1f, (h & 0x3ff) as f32);
        let v = match exp {
            0 => frac / 1024.0 * 2f32.powi(-14),
            31 => f32::INFINITY,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e as i32 - 15),
        };
        if sign == 1 { -v } else { v }
    }

    fn dab(center: f32, radius: f32, metal_value: f32, enables: [f32; 4]) -> Stamp {
        Stamp {
            brush: [center, center, radius, 1.0],
            color: [1.0, 0.0, 0.0, 0.0],
            values: [metal_value, 0.5, 1.0, 0.0],
            enables,
            shape: [0.0; 4],
        }
    }

    fn fill(base: Option<[f32; 3]>, metallic: Option<f32>, roughness: Option<f32>) -> NodeKind {
        NodeKind::Fill(FillParams { base, metallic, roughness, ..Default::default() })
    }

    #[test]
    fn projective_brush_paints_visible_side_only_and_undoes() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]); // also renders the scene so depth exists

        let stamp = dab(SIZE as f32 / 2.0, 20.0, 1.0, [0.0, 1.0, 0.0, 0.0]); // metallic only
        rig.run(
            &doc,
            &[
                PaintOp::BeginStroke { layer, target: Target::Content },
                PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] },
            ],
        );

        let front = rig.front;
        let metal = rig.metal();
        let (value, cover) = texel_rg(&metal, sphere_uv(front));
        assert!(cover > 240 && value > 240, "front texel not painted: value={value} cover={cover}");
        assert_eq!(texel_rg(&metal, sphere_uv(-front)).1, 0, "back-facing texel must not be painted");
        let painted = metal.chunks_exact(2).filter(|t| t[1] > 0).count();
        assert!(painted > 0 && painted < (SIZE * SIZE) as usize / 10, "implausible painted area: {painted}");
        let rough = rig.read(&rig.gpu.paint().output.rough, 2);
        assert!(rough.iter().all(|&b| b == 0), "roughness written while disabled");

        rig.run(&doc, &[PaintOp::UndoStroke]);
        assert_eq!(texel_rg(&rig.metal(), sphere_uv(front)).1, 0, "undo did not clear the stroke");
        assert_eq!(rig.gpu.paint().undo_depth(), (0, 1));
        rig.run(&doc, &[PaintOp::RedoStroke]);
        assert!(texel_rg(&rig.metal(), sphere_uv(front)).1 > 240, "redo did not restore the stroke");
    }

    /// With symmetry on, the stroke also paints the mirror image of the visible surface, even on
    /// the side the camera cannot see; with it off, only the visible side is painted.
    #[test]
    fn symmetry_paints_the_mirrored_texels() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let front = rig.front;
        let mirrored = Vec3::new(-front.x, front.y, front.z);
        let mut painted = |symmetry: f32| {
            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            let mut stamp = dab(SIZE as f32 / 2.0, 20.0, 1.0, [0.0, 1.0, 0.0, 0.0]);
            stamp.shape[3] = symmetry; // axis 1 = x, plane through the origin
            rig.run(&doc, &[PaintOp::BeginStroke { layer, target: Target::Content }, PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] }]);
            let metal = rig.metal();
            (texel_rg(&metal, sphere_uv(front)).1 > 240, texel_rg(&metal, sphere_uv(mirrored)).1 > 240)
        };
        assert_eq!(painted(0.0), (true, false), "without symmetry only the visible side is painted");
        assert_eq!(painted(1.0), (true, true), "with symmetry the mirror image is painted too");
    }

    /// A sphere cut into a northern set (first half of the indices) and a southern one.
    fn two_set_sphere() -> Mesh {
        let mut m = Mesh::uv_sphere(1.0, 96, 48);
        let half = (m.indices.len() / 6 * 3) as u32;
        let total = m.indices.len() as u32;
        m.sets = vec![
            op_mesh::SetRange { name: "North".into(), first_index: 0, index_count: half },
            op_mesh::SetRange { name: "South".into(), first_index: half, index_count: total - half },
        ];
        m
    }

    impl Rig {
        fn run_sets(&mut self, batches: &[SetBatch]) {
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.gpu.render_sets(&self.device, &self.queue, &mut enc, &self.globals, [SIZE, SIZE], batches);
            self.queue.submit([enc.finish()]);
        }

        fn set_metal(&self, set: usize) -> Vec<u8> {
            self.read(&self.gpu.sets[set].output.metal, 2)
        }
    }

    /// Each texture set has its own paint store and material, and a stroke only lands on the
    /// triangles of the set it was made in.
    #[test]
    fn texture_sets_keep_separate_paint_and_only_stamp_their_own_triangles() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        rig.gpu.set_mesh(&rig.device, &two_set_sphere());
        assert_eq!(rig.gpu.sets.len(), 2);
        let (mut a, mut b) = (Document::default(), Document::default());
        a.add(None, None, "metal", fill(None, Some(1.0), None));
        b.add(None, None, "metal half", fill(None, Some(0.5), None));
        let layer_b = b.add(None, None, "paint", NodeKind::Paint);
        let reset = |set, doc: &Document| SetBatch { tile: None, set, doc: doc.clone(), ops: vec![PaintOp::Reset(SIZE)] };
        rig.run_sets(&[reset(0, &a), reset(1, &b)]);

        // The camera looks at the northern half: set 0's fill is everywhere in set 0's texture, and
        // set 1's texture is still empty.
        let front = rig.front;
        assert!(front.y > 0.0, "the test needs the camera above the equator");
        let uv = sphere_uv(front);
        assert!(texel_rg(&rig.set_metal(0), uv).1 > 240, "set 0 should hold its own fill");
        let (m1, c1) = texel_rg(&rig.set_metal(1), uv);
        assert!(c1 > 240 && (m1 as i32 - 128).abs() < 4, "set 1 should hold its own, different fill: {m1} {c1}");

        // A stroke in set 1 aimed at northern geometry paints nothing: its triangles are not drawn.
        let stamp = dab(SIZE as f32 / 2.0, 20.0, 1.0, [0.0, 1.0, 0.0, 0.0]);
        let stroke = |set, doc: &Document, layer| SetBatch {
            tile: None,
            set,
            doc: doc.clone(),
            ops: vec![
                PaintOp::BeginStroke { layer, target: Target::Content },
                PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] },
            ],
        };
        rig.run_sets(&[stroke(1, &b, layer_b)]);
        assert!(
            rig.set_metal(1).chunks_exact(2).all(|t| t[0] == 128 || t[0] == 127),
            "a stroke leaked into another set's triangles"
        );
    }

    /// The GPU compositor must agree with the CPU reference in op-core, including
    /// blend modes, opacity, hidden layers and isolated groups.
    #[test]
    fn gpu_compositor_matches_cpu_reference() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        doc.add(None, None, "bg", fill(Some([0.9, 0.4, 0.2]), Some(0.8), Some(0.3)));
        let mul = doc.add(None, None, "mul", fill(Some([0.5, 0.7, 0.9]), Some(0.5), None));
        doc.find_mut(mul).unwrap().blend = BlendMode::Multiply;
        doc.find_mut(mul).unwrap().opacity = 0.5;
        let hidden = doc.add(None, None, "hidden", fill(Some([0.0; 3]), Some(0.0), Some(0.0)));
        doc.find_mut(hidden).unwrap().visible = false;
        let g = doc.add(None, None, "group", NodeKind::Group { children: vec![] });
        doc.find_mut(g).unwrap().blend = BlendMode::Screen;
        doc.find_mut(g).unwrap().opacity = 0.75;
        let a = doc.add(Some(g), None, "a", fill(Some([0.2, 0.2, 0.6]), None, Some(0.9)));
        let b = doc.add(Some(g), None, "b", fill(Some([0.9, 0.1, 0.1]), Some(0.25), None));
        doc.find_mut(a).unwrap().blend = BlendMode::Overlay;
        doc.find_mut(b).unwrap().blend = BlendMode::Difference;
        doc.find_mut(b).unwrap().opacity = 0.6;

        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let expect = op_core::blend::eval_fill_only(&doc);

        let base = rig.read(&rig.gpu.paint().output.base, 8);
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 8) as usize;
        let got_base: Vec<f32> =
            (0..4).map(|c| f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]]))).collect();
        for c in 0..4 {
            assert!((got_base[c] - expect.base[c]).abs() < 2e-3, "base[{c}] gpu {} cpu {}", got_base[c], expect.base[c]);
        }
        let m = rig.metal();
        let r = rig.read(&rig.gpu.paint().output.rough, 2);
        let j = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
        for (name, bytes, want) in [("metal", &m, expect.metal), ("rough", &r, expect.rough)] {
            for c in 0..2 {
                let got = bytes[j + c] as f32 / 255.0;
                assert!((got - want[c]).abs() < 3.0 / 255.0, "{name}[{c}] gpu {got} cpu {}", want[c]);
            }
        }
    }

    /// Every blend mode over a non-empty backdrop (over an empty one all modes
    /// degenerate to Normal, which would hide formula errors).
    #[test]
    fn every_blend_mode_matches_cpu_reference() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        for mode in BlendMode::ALL {
            let mut doc = Document::default();
            // Backdrop channels straddle 0.5 so both Overlay branches are exercised.
            doc.add(None, None, "bg", fill(Some([0.9, 0.4, 0.2]), Some(0.8), Some(0.3)));
            let top = doc.add(None, None, "top", fill(Some([0.5, 0.7, 0.9]), Some(0.4), Some(0.6)));
            doc.find_mut(top).unwrap().blend = mode;
            doc.find_mut(top).unwrap().opacity = 0.7;

            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            let expect = op_core::blend::eval_fill_only(&doc);
            let base = rig.read(&rig.gpu.paint().output.base, 8);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 8) as usize;
            for c in 0..3 {
                let got = f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]]));
                assert!((got - expect.base[c]).abs() < 2e-3, "{mode:?} base[{c}] gpu {got} cpu {}", expect.base[c]);
            }
            let (m, r) = (rig.metal(), rig.read(&rig.gpu.paint().output.rough, 2));
            let j = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
            let (gm, gr) = (m[j] as f32 / 255.0, r[j] as f32 / 255.0);
            assert!((gm - expect.metal[0]).abs() < 3.0 / 255.0, "{mode:?} metal gpu {gm} cpu {}", expect.metal[0]);
            assert!((gr - expect.rough[0]).abs() < 3.0 / 255.0, "{mode:?} rough gpu {gr} cpu {}", expect.rough[0]);
        }
    }

    /// A black mask hides a layer; painting white into it reveals only where stamped.
    #[test]
    fn painted_mask_reveals_layer_locally() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "metal", fill(None, Some(1.0), None));
        doc.find_mut(layer).unwrap().mask = Some(Mask::new(false));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(rig.metal().iter().all(|&b| b == 0), "black mask should hide the layer entirely");

        let reveal = Stamp {
            brush: [SIZE as f32 / 2.0, SIZE as f32 / 2.0, 20.0, 1.0],
            color: [0.0; 4],
            values: [1.0, 0.0, 1.0, 0.0],
            enables: [0.0; 4],
            shape: [0.0; 4],
        };
        rig.run(
            &doc,
            &[
                PaintOp::BeginStroke { layer, target: Target::Mask },
                PaintOp::Stamps { layer, target: Target::Mask, stamps: vec![reveal] },
            ],
        );
        let front = rig.front;
        let metal = rig.metal();
        let (v, c) = texel_rg(&metal, sphere_uv(front));
        assert!(c > 240 && v > 240, "revealed texel: value={v} cover={c}");
        assert_eq!(texel_rg(&metal, sphere_uv(-front)).1, 0, "unpainted mask area must stay hidden");

        // Disabling the mask shows the whole layer; undo of the stroke re-hides the dab.
        doc.find_mut(layer).unwrap().mask.as_mut().unwrap().enabled = false;
        rig.run(&doc, &[]);
        assert!(texel_rg(&rig.metal(), sphere_uv(-front)).1 > 240, "disabled mask should not hide anything");
        doc.find_mut(layer).unwrap().mask.as_mut().unwrap().enabled = true;
        rig.run(&doc, &[PaintOp::UndoStroke]);
        assert_eq!(texel_rg(&rig.metal(), sphere_uv(front)).1, 0, "undo should re-hide the dab");
    }

    /// Textures of deleted layers survive while history can still restore them,
    /// and are freed once nothing references them.
    #[test]
    fn gc_keeps_referenced_nodes_and_frees_the_rest() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let p = doc.add(None, None, "p", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(rig.gpu.paint().store.contains_key(&p));

        let mut gone = doc.clone();
        gone.remove(p);
        rig.run(&gone, &[PaintOp::Gc(vec![p])]); // history still references it
        assert!(rig.gpu.paint().store.contains_key(&p), "referenced node was freed");
        rig.run(&gone, &[PaintOp::Gc(vec![])]);
        assert!(!rig.gpu.paint().store.contains_key(&p), "unreferenced node was kept");
    }

    fn output_bytes(rig: &Rig) -> [Vec<u8>; 5] {
        rig.gpu.paint().capture_output(&rig.device, &rig.queue)
    }

    /// Paint, save to disk, load into a fresh GPU context: the composited result and
    /// every stored texture must come back bit-for-bit.
    #[test]
    fn saved_project_reloads_to_an_identical_result() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::with_default_material();
        let paint = doc.add(None, None, "paint", NodeKind::Paint);
        doc.find_mut(paint).unwrap().mask = Some(Mask::new(false));
        doc.find_mut(paint).unwrap().blend = BlendMode::Multiply;
        doc.find_mut(paint).unwrap().opacity = 0.8;
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let blank = output_bytes(&rig);

        let content = Stamp {
            brush: [SIZE as f32 / 2.0, SIZE as f32 / 2.0, 40.0, 0.5],
            color: [0.2, 0.5, 0.9, 0.0],
            values: [0.7, 0.3, 0.9, 0.0],
            enables: [1.0, 1.0, 1.0, 0.0],
            shape: [0.0; 4],
        };
        let reveal = Stamp { brush: [SIZE as f32 / 2.0 + 10.0, SIZE as f32 / 2.0, 30.0, 0.3], color: [0.0; 4], values: [1.0, 0.0, 0.8, 0.0], enables: [0.0; 4], shape: [0.0; 4] };
        rig.run(&doc, &[PaintOp::Stamps { layer: paint, target: Target::Content, stamps: vec![content] }]);
        rig.run(&doc, &[PaintOp::Stamps { layer: paint, target: Target::Mask, stamps: vec![reveal] }]);
        let before = output_bytes(&rig);
        assert_ne!(before[0], blank[0], "painting did not change the composited base color");
        assert_ne!(before[2], blank[2], "painting did not change the composited roughness");

        let blobs = rig.gpu.paint().capture_blobs(&rig.device, &rig.queue, &doc);
        assert_eq!(blobs.len(), 5, "base, metal, rough, height and mask of the paint layer");
        let project = op_project::Project {
            mesh_name: "sphere".into(),
            mesh: Mesh::uv_sphere(1.0, 96, 48),
            sets: vec![op_project::SetData {
                texture_size: SIZE,
                document: doc.clone(),
                tiles: vec![op_project::TileData { blobs: blobs.clone(), baked: Default::default() }],
            }],
            active: 0,
            udim: false,
            hidden_sets: Vec::new(),
            images: Vec::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.opproj");
        op_project::save(&path, &project).unwrap();
        let loaded = op_project::load(&path).unwrap();
        let loaded = loaded.sets.into_iter().next().unwrap();
        assert_eq!(loaded.document, doc);

        let mut fresh = Rig::new().unwrap();
        fresh.run(&loaded.document, &[PaintOp::Load { size: loaded.texture_size, blobs: loaded.tiles[0].blobs.clone() }]);
        // Compare piecewise so a failure says which texture differs, not a wall of bytes.
        let recaptured = fresh.gpu.paint().capture_blobs(&fresh.device, &fresh.queue, &loaded.document);
        assert_eq!(recaptured.len(), blobs.len());
        for (a, b) in recaptured.iter().zip(&blobs) {
            let diff = a.bytes.iter().zip(&b.bytes).filter(|(x, y)| x != y).count();
            assert_eq!(diff, 0, "stored {:?} texture differs in {diff} bytes after reload", b.kind);
        }
        let after = output_bytes(&fresh);
        for (i, name) in ["base", "metal", "rough"].iter().enumerate() {
            let diff = after[i].iter().zip(&before[i]).filter(|(x, y)| x != y).count();
            assert_eq!(diff, 0, "composited {name} differs in {diff} of {} bytes after reload", before[i].len());
        }
    }

    fn fill_h(base: Option<[f32; 3]>, height: Option<f32>) -> NodeKind {
        NodeKind::Fill(FillParams { base, height, ..Default::default() })
    }

    /// The height channel composites exactly like the others, including blend modes and opacity.
    #[test]
    fn height_compositing_matches_the_cpu_reference() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        for mode in BlendMode::ALL {
            let mut doc = Document::default();
            doc.add(None, None, "bg", fill_h(Some([0.5; 3]), Some(0.7)));
            let top = doc.add(None, None, "top", fill_h(None, Some(0.35)));
            doc.find_mut(top).unwrap().blend = mode;
            doc.find_mut(top).unwrap().opacity = 0.6;
            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            let want = op_core::blend::eval_fill_only(&doc).height;
            let h = rig.read(&rig.gpu.paint().output.height, 2);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
            let got = [h[i] as f32 / 255.0, h[i + 1] as f32 / 255.0];
            assert!((got[0] - want[0]).abs() < 3.0 / 255.0 && (got[1] - want[1]).abs() < 3.0 / 255.0, "{mode:?}: gpu {got:?} cpu {want:?}");
        }
    }

    /// A brush stamp with the height channel enabled writes height and nothing else.
    #[test]
    fn a_height_stamp_writes_only_the_height_channel() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let stamp = Stamp {
            brush: [SIZE as f32 / 2.0, SIZE as f32 / 2.0, 20.0, 1.0],
            color: [0.0; 4],
            values: [0.0, 0.0, 1.0, 0.9],
            enables: [0.0, 0.0, 0.0, 1.0],
            shape: [0.0; 4],
        };
        rig.run(&doc, &[PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] }]);
        let uv = sphere_uv(rig.front);
        let (hv, hc) = texel_rg(&rig.read(&rig.gpu.paint().output.height, 2), uv);
        assert!(hc > 240 && (hv as f32 / hc as f32 - 0.9).abs() < 0.02, "height {hv} coverage {hc}");
        assert!(rig.metal().chunks_exact(2).all(|t| t[1] == 0), "metallic must be untouched");
        // Painting raises the surface: the normal map is no longer flat anywhere near the edge of the dab.
        let normal = rig.read(&rig.gpu.paint().normal.target, 4);
        let tilted = normal.chunks_exact(4).filter(|p| (p[0] as i32 - 128).abs() > 8 || (p[1] as i32 - 128).abs() > 8).count();
        assert!(tilted > 50, "only {tilted} texels tilted");
    }

    /// The normal-map shader agrees with the CPU derivation, with and without baked detail.
    #[test]
    fn gpu_normal_map_matches_the_cpu_derivation() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        doc.height_strength = 3.0;
        // A height field with slopes in both directions, stored as the compositor would: (h * cov, cov).
        let n = (SIZE * SIZE) as usize;
        let h: Vec<f32> = (0..n)
            .map(|i| {
                let (x, y) = ((i as u32 % SIZE) as f32 / SIZE as f32, (i as u32 / SIZE) as f32 / SIZE as f32);
                0.5 + 0.3 * (x * 6.0).sin() * (y * 4.0).cos()
            })
            .collect();
        let bytes: Vec<u8> = h.iter().flat_map(|v| [(v * 255.0).round() as u8, 255]).collect();
        let blob = op_project::TextureBlob { node: layer, kind: op_project::BlobKind::Height, bytes };
        rig.run(&doc, &[PaintOp::Load { size: SIZE, blobs: vec![blob] }]);

        // The stored height is quantised to 8 bits; the CPU side must use the same values.
        let hq: Vec<f32> = h.iter().map(|v| (v * 255.0).round() / 255.0).collect();
        let check = |rig: &Rig, base: Option<&[[f32; 3]]>, what: &str| {
            let want = op_core::normal::height_to_normal(&hq, SIZE, doc.height_strength, base, None);
            let got = rig.read(&rig.gpu.paint().normal.target, 4);
            let mut worst = 0.0f32;
            for (i, w) in want.iter().enumerate() {
                let g = op_core::normal::unpack([got[i * 4] as f32 / 255.0, got[i * 4 + 1] as f32 / 255.0, got[i * 4 + 2] as f32 / 255.0]);
                worst = worst.max((0..3).map(|c| (g[c] - w[c]).abs()).fold(0.0, f32::max));
            }
            assert!(worst < 0.03, "{what}: worst component error {worst}");
        };
        check(&rig, None, "painted height only");

        // Add baked detail: a constant tilt.
        let tilt = op_core::normal::unpack([0.7, 0.45, 0.9]);
        let packed: Vec<u8> = (0..n).flat_map(|_| [178u8, 115, 230, 255]).collect();
        let base: Vec<[f32; 3]> = (0..n).map(|_| op_core::normal::unpack([178.0 / 255.0, 115.0 / 255.0, 230.0 / 255.0])).collect();
        let _ = tilt;
        rig.run(&doc, &[PaintOp::SetBakedNormal { size: SIZE, rgba8: Some(packed) }]);
        check(&rig, Some(&base), "painted height over a baked normal");

        // Removing it restores the plain result.
        rig.run(&doc, &[PaintOp::SetBakedNormal { size: SIZE, rgba8: None }]);
        check(&rig, None, "after removing the baked normal");
    }

    /// Stamp shapes cover what their outline says: a square reaches its corners, a round dab does
    /// not, and rotating the square by 45 degrees pulls the corners in and pushes the points out.
    #[test]
    fn stamp_shapes_and_rotation_cover_the_right_texels() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        let covered = |rig: &mut Rig, shape: [f32; 4], at: (f32, f32)| {
            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            let stamp = Stamp {
                brush: [0.5, 0.5, 0.2, 1.0],
                color: [1.0, 0.0, 0.0, 0.0],
                values: [0.0, 0.0, 1.0, 0.0],
                enables: [1.0, 0.0, 0.0, 0.0],
                shape,
            };
            rig.run(&doc, &[PaintOp::Stamps2d { layer, target: Target::Content, stamps: vec![stamp] }]);
            let base = rig.read(&rig.gpu.paint().output.base, 8);
            let (x, y) = ((at.0 * SIZE as f32) as usize, (at.1 * SIZE as f32) as usize);
            let i = (y * SIZE as usize + x) * 8 + 6;
            f16_to_f32(u16::from_le_bytes([base[i], base[i + 1]])) > 0.5
        };
        let corner = (0.5 + 0.17, 0.5 + 0.17); // inside the square's corner, outside the circle
        let tip = (0.5 + 0.19, 0.5);
        assert!(!covered(&mut rig, [0.0, 0.0, 0.0, 0.0], corner), "a round dab does not reach the corner");
        assert!(covered(&mut rig, [0.0, 1.0, 0.0, 0.0], corner), "a square dab does");
        let turned = std::f32::consts::FRAC_PI_4;
        assert!(covered(&mut rig, [turned, 1.0, 0.0, 0.0], tip), "a square turned 45 degrees points along the axis");
        assert!(!covered(&mut rig, [turned, 1.0, 0.0, 0.0], corner), "and its corners moved off the diagonal");
        assert!(!covered(&mut rig, [0.0, 2.0, 0.0, 0.0], corner), "a diamond is smaller than the circle on the diagonal");
        assert!(covered(&mut rig, [0.0, 2.0, 0.0, 0.0], (0.5 + 0.1, 0.5)), "a diamond covers its middle");
    }

    /// 2D stamps land exactly where their UV position says, with the radius in UV units.
    #[test]
    fn uv_space_stamps_land_at_their_uv_position() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let stamp = Stamp {
            brush: [0.25, 0.75, 0.1, 1.0], // u, v (v down), radius, hardness
            color: [1.0, 0.0, 0.0, 0.0],
            values: [0.0, 0.0, 1.0, 0.0],
            enables: [1.0, 0.0, 0.0, 0.0],
            shape: [0.0; 4],
        };
        rig.run(&doc, &[PaintOp::Stamps2d { layer, target: Target::Content, stamps: vec![stamp] }]);
        let base = rig.read(&rig.gpu.paint().output.base, 8);
        let alpha = |u: f32, v: f32| {
            let (x, y) = ((u * SIZE as f32) as usize, (v * SIZE as f32) as usize);
            let i = (y * SIZE as usize + x) * 8 + 6;
            f16_to_f32(u16::from_le_bytes([base[i], base[i + 1]]))
        };
        assert!(alpha(0.25, 0.75) > 0.99, "centre is covered");
        assert!(alpha(0.25 + 0.08, 0.75) > 0.99 && alpha(0.25, 0.75 - 0.08) > 0.99, "inside the radius");
        assert!(alpha(0.25 + 0.13, 0.75) < 0.01, "outside the radius");
        assert!(alpha(0.75, 0.25) < 0.01 && alpha(0.25, 0.25) < 0.01 && alpha(0.75, 0.75) < 0.01, "v points down, not mirrored");
        let covered = base.chunks_exact(8).filter(|t| f16_to_f32(u16::from_le_bytes([t[6], t[7]])) > 0.5).count();
        let expect = std::f32::consts::PI * (0.1 * SIZE as f32).powi(2);
        assert!((covered as f32 - expect).abs() < expect * 0.1, "{covered} texels vs {expect}");
        // Mask stamps work the same way.
        let mut doc2 = Document::default();
        let m = doc2.add(None, None, "m", NodeKind::Paint);
        doc2.find_mut(m).unwrap().mask = Some(Mask::new(false));
        rig.run(&doc2, &[PaintOp::Reset(SIZE)]);
        let reveal = Stamp { brush: [0.5, 0.5, 0.05, 1.0], color: [0.0; 4], values: [1.0, 0.0, 1.0, 0.0], enables: [0.0; 4], shape: [0.0; 4] };
        rig.run(&doc2, &[PaintOp::Stamps2d { layer: m, target: Target::Mask, stamps: vec![reveal] }]);
        let mask = rig.read(rig.gpu.paint().store[&m].mask.as_ref().unwrap(), 1);
        assert_eq!(mask[(SIZE / 2 * SIZE + SIZE / 2) as usize], 255);
        assert_eq!(mask[0], 0);
    }

    /// A sphere (one texture set) whose southern half has its UVs moved into tile 1002.
    fn two_tile_sphere() -> Mesh {
        let mut m = Mesh::uv_sphere(1.0, 96, 48);
        let half = m.indices.len() / 6 * 3;
        let mut moved = std::collections::HashMap::new();
        for k in half..m.indices.len() {
            let old = m.indices[k];
            let new = *moved.entry(old).or_insert_with(|| {
                let mut v = m.vertices[old as usize];
                v.uv[0] += 1.0;
                m.vertices.push(v);
                (m.vertices.len() - 1) as u32
            });
            m.indices[k] = new;
        }
        m.partition_tiles();
        m
    }

    /// With UV tiles, each tile of a set has its own paint store: strokes land only in the tile whose
    /// triangles they touch, 2D stamps are addressed in global UV space, and layers composite per tile.
    #[test]
    fn uv_tiles_have_separate_paint_stores() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mesh = two_tile_sphere();
        rig.gpu.set_mesh_tiles(&rig.device, &mesh, true);
        assert_eq!(rig.gpu.sets.len(), 1);
        let set = &rig.gpu.sets[0];
        assert_eq!(set.tiles.len(), 2);
        assert_eq!((set.tiles[0].tile.udim(), set.tiles[1].tile.udim()), (1001, 1002));

        let mut doc = Document::default();
        doc.add(None, None, "metal", fill(None, Some(1.0), None));
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        let reset = SetBatch { tile: None, set: 0, doc: doc.clone(), ops: vec![PaintOp::Reset(SIZE)] };
        rig.run_sets(&[reset]);
        let tile_metal = |rig: &Rig, t: usize| rig.read(&rig.gpu.sets[0].tiles[t].output.metal, 2);
        // The fill layer reaches every tile.
        for t in 0..2 {
            assert!(tile_metal(&rig, t).chunks_exact(2).all(|p| p[1] == 255), "tile {t} is filled");
        }

        // A 3D stroke aimed at the northern half (tile 1001) paints only tile 1001.
        let stamp = dab(SIZE as f32 / 2.0, 20.0, 0.0, [0.0, 1.0, 0.0, 0.0]); // metallic 0 where it lands
        let stroke = SetBatch {
            tile: None,
            set: 0,
            doc: doc.clone(),
            ops: vec![
                PaintOp::BeginStroke { layer, target: Target::Content },
                PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] },
            ],
        };
        rig.run_sets(&[stroke]);
        let uv = sphere_uv(rig.front);
        assert!(texel_rg(&tile_metal(&rig, 0), uv).0 < 10, "the front texel of tile 1001 was painted to 0");
        assert!(tile_metal(&rig, 1).chunks_exact(2).all(|p| p[0] == 255), "tile 1002 must be untouched");

        // A 2D stamp at global UV (1.5, 0.5) lands in tile 1002 at local (0.5, 0.5), and not in 1001.
        let dab2d = Stamp { brush: [1.5, 0.5, 0.05, 1.0], color: [0.0; 4], values: [0.0, 0.0, 1.0, 0.0], enables: [0.0, 1.0, 0.0, 0.0], shape: [0.0; 4] };
        let paint2d = SetBatch { tile: None, set: 0, doc: doc.clone(), ops: vec![PaintOp::Stamps2d { layer, target: Target::Content, stamps: vec![dab2d] }] };
        rig.run_sets(&[paint2d]);
        let centre = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
        assert!(tile_metal(&rig, 1)[centre] < 10, "tile 1002 centre is now metallic 0");
        let first = tile_metal(&rig, 0);
        assert_eq!(first[centre], 255, "tile 1001 centre is not touched by a stamp in 1002");

        // Undo reverts every tile together.
        rig.run_sets(&[SetBatch { tile: None, set: 0, doc: doc.clone(), ops: vec![PaintOp::UndoStroke] }]);
        assert!(texel_rg(&tile_metal(&rig, 0), uv).0 > 240, "the 3D stroke is undone in tile 1001");
    }

    /// A hidden texture set is not drawn: the view goes through it to whatever is behind.
    #[test]
    fn hidden_texture_sets_are_not_drawn() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        rig.gpu.set_mesh(&rig.device, &two_set_sphere());
        let red = Document::with_default_material();
        let mut red = red;
        red.find_mut(1).unwrap().kind = fill(Some([1.0, 0.0, 0.0]), Some(0.0), Some(0.6));
        let mut green = Document::with_default_material();
        green.find_mut(1).unwrap().kind = fill(Some([0.0, 1.0, 0.0]), Some(0.0), Some(0.6));
        let reset = |set, doc: &Document| SetBatch { tile: None, set, doc: doc.clone(), ops: vec![PaintOp::Reset(SIZE)] };
        rig.run_sets(&[reset(0, &red), reset(1, &green)]);
        let shown = center_pixel(&rig);
        assert!(shown[0] > shown[1], "the camera sees the red northern set: {shown:?}");

        rig.gpu.hidden = vec![true, false];
        rig.run_sets(&[]);
        let hidden = center_pixel(&rig);
        assert!(hidden[0] < shown[0] && hidden[0] <= hidden[1], "with the red set hidden it is gone: {hidden:?}");

        rig.gpu.hidden = vec![false, true];
        rig.run_sets(&[]);
        assert_eq!(center_pixel(&rig), shown, "hiding the other set leaves the view unchanged");
        rig.gpu.hidden = Vec::new();
        rig.run_sets(&[]);
        assert_eq!(center_pixel(&rig), shown, "nothing hidden again");
    }

    const TEX_ID: &str = "00112233445566778899aabbccddeeff";

    /// A 2x2 test image: red and green on top, blue and white below.
    fn quad_image() -> op_image::Decoded {
        op_image::Decoded {
            width: 2,
            height: 2,
            rgba8: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255],
        }
    }

    fn textured(slot: op_core::Slot, tex: op_core::TexRef, uv: op_core::UvTransform) -> Document {
        let mut f = FillParams { uv, ..Default::default() };
        f.maps.set(slot, Some(tex));
        let mut doc = Document::default();
        doc.add(None, None, "textured", NodeKind::Fill(f));
        doc
    }

    /// Linear RGB (as stored) of the base output at a UV position.
    fn base_at(rig: &Rig, uv: (f32, f32)) -> [f32; 4] {
        let base = rig.read(&rig.gpu.paint().output.base, 8);
        let (x, y) = (((uv.0 * SIZE as f32) as u32).min(SIZE - 1), ((uv.1 * SIZE as f32) as u32).min(SIZE - 1));
        let i = ((y * SIZE + x) * 8) as usize;
        [0, 1, 2, 3].map(|c| f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]])))
    }

    fn near(a: [f32; 4], want: [f32; 3]) -> bool {
        (0..3).all(|c| (a[c] - want[c]).abs() < 0.02)
    }

    /// A base colour texture lands on the UV square the right way up, with sRGB decoding.
    #[test]
    fn a_base_colour_texture_is_laid_over_uv_space_the_right_way_up() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &quad_image());
        let doc = textured(op_core::Slot::BaseColor, op_core::TexRef::new(TEX_ID), Default::default());
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(near(base_at(&rig, (0.25, 0.25)), [1.0, 0.0, 0.0]), "top left is red: {:?}", base_at(&rig, (0.25, 0.25)));
        assert!(near(base_at(&rig, (0.75, 0.25)), [0.0, 1.0, 0.0]), "top right is green");
        assert!(near(base_at(&rig, (0.25, 0.75)), [0.0, 0.0, 1.0]), "bottom left is blue");
        assert!(near(base_at(&rig, (0.75, 0.75)), [1.0, 1.0, 1.0]), "bottom right is white");
        assert!(base_at(&rig, (0.25, 0.25))[3] > 0.99, "an opaque image covers");

        // sRGB: a mid-grey texel (128) is about 0.216 in linear light, not 0.5.
        let grey = op_image::Decoded { width: 1, height: 1, rgba8: vec![128, 128, 128, 255] };
        let grey_id = "ffeeddccbbaa99887766554433221100";
        rig.gpu.upload_image(&rig.device, &rig.queue, grey_id, &grey);
        let doc = textured(op_core::Slot::BaseColor, op_core::TexRef::new(grey_id), Default::default());
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let c = base_at(&rig, (0.5, 0.5));
        assert!((c[0] - 0.216).abs() < 0.01, "colour textures are decoded from sRGB: {c:?}");
    }

    /// Tiling and rotation place the texture exactly as `UvTransform::apply` says.
    #[test]
    fn texture_tiling_and_rotation_follow_the_uv_transform() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &quad_image());
        let colour = |t: [f32; 2]| {
            // Which quadrant of the repeating image a texture coordinate is in.
            let (fu, fv) = (t[0].rem_euclid(1.0), t[1].rem_euclid(1.0));
            match (fu < 0.5, fv < 0.5) {
                (true, true) => [1.0, 0.0, 0.0],
                (false, true) => [0.0, 1.0, 0.0],
                (true, false) => [0.0, 0.0, 1.0],
                (false, false) => [1.0, 1.0, 1.0],
            }
        };
        for xf in [
            op_core::UvTransform { tiling: [2.0, 2.0], ..Default::default() },
            op_core::UvTransform { tiling: [1.0, 1.0], offset: [0.5, 0.0], rotation: 0.0 },
            op_core::UvTransform { rotation: 90.0, ..Default::default() },
            op_core::UvTransform { tiling: [3.0, 1.0], offset: [0.0, 0.25], rotation: 180.0 },
            op_core::UvTransform { tiling: [-1.0, 1.0], ..Default::default() },
        ] {
            let doc = textured(op_core::Slot::BaseColor, op_core::TexRef::new(TEX_ID), xf);
            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            // Only the exact centre of an image texel is a pure colour once the image is filtered, so test the
            // pixels whose texture coordinate falls (almost) on one. Read the output once for all of them.
            let out = rig.read(&rig.gpu.paint().output.base, 8);
            let pixel = |x: u32, y: u32| {
                let i = ((y * SIZE + x) * 8) as usize;
                [0, 1, 2, 3].map(|c| f16_to_f32(u16::from_le_bytes([out[i + c * 2], out[i + c * 2 + 1]])))
            };
            let mut tested = 0;
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let uv = [(x as f32 + 0.5) / SIZE as f32, (y as f32 + 0.5) / SIZE as f32];
                    let t = xf.apply(uv);
                    let on_centre = |v: f32| {
                        let f = v.rem_euclid(1.0);
                        (f - 0.25).abs() < 0.006 || (f - 0.75).abs() < 0.006
                    };
                    if !(on_centre(t[0]) && on_centre(t[1])) {
                        continue;
                    }
                    tested += 1;
                    let got = pixel(x, y);
                    // A little blend with the neighbouring texel is expected this close to a centre.
                    assert!((0..3).all(|c| (got[c] - colour(t)[c]).abs() < 0.05), "{xf:?} at {uv:?}: got {got:?}, expected {:?}", colour(t));
                }
            }
            assert!(tested >= 4, "{xf:?}: only {tested} points were checkable");
        }
    }

    /// Scalar channels read the red channel as data (no sRGB decoding) and can be inverted.
    #[test]
    fn scalar_textures_feed_roughness_metallic_and_height() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let img = op_image::Decoded { width: 1, height: 1, rgba8: vec![64, 0, 0, 255] };
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &img);
        let value = |rig: &Rig, tex: &wgpu::Texture| {
            let d = rig.read(tex, 2);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
            (d[i] as f32 / 255.0, d[i + 1] as f32 / 255.0)
        };
        for (slot, invert, want) in [
            (op_core::Slot::Roughness, false, 0.251),
            (op_core::Slot::Roughness, true, 0.749),
            (op_core::Slot::Metallic, false, 0.251),
            (op_core::Slot::Height, false, 0.251),
            (op_core::Slot::Height, true, 0.749),
        ] {
            let tex = op_core::TexRef { invert, ..op_core::TexRef::new(TEX_ID) };
            rig.run(&textured(slot, tex, Default::default()), &[PaintOp::Reset(SIZE)]);
            let out = &rig.gpu.paint().output;
            let (v, cov) = value(
                &rig,
                match slot {
                    op_core::Slot::Roughness => &out.rough,
                    op_core::Slot::Metallic => &out.metal,
                    _ => &out.height,
                },
            );
            assert!((v - want).abs() < 0.01 && cov > 0.99, "{slot:?} invert={invert}: value {v}, coverage {cov}");
        }
    }

    /// A normal map from a fill layer reaches the final normal, and `flip_green` mirrors its y.
    #[test]
    fn a_normal_map_texture_tilts_the_surface_and_flip_green_mirrors_it() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        // Tilted towards +x and +y (OpenGL packing).
        let img = op_image::Decoded { width: 1, height: 1, rgba8: vec![200, 190, 220, 255] };
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &img);
        let layer_normal = |rig: &Rig| {
            let d = rig.read(&rig.gpu.paint().output.normal, 4);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
            op_core::normal::unpack([d[i] as f32 / 255.0, d[i + 1] as f32 / 255.0, d[i + 2] as f32 / 255.0])
        };
        let final_normal = |rig: &Rig| {
            let d = rig.read(&rig.gpu.paint().normal.target, 4);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
            op_core::normal::unpack([d[i] as f32 / 255.0, d[i + 1] as f32 / 255.0, d[i + 2] as f32 / 255.0])
        };
        rig.run(&textured(op_core::Slot::Normal, op_core::TexRef::new(TEX_ID), Default::default()), &[PaintOp::Reset(SIZE)]);
        let (l, f) = (layer_normal(&rig), final_normal(&rig));
        assert!(l[0] > 0.3 && l[1] > 0.2 && l[2] > 0.0, "decoded from the packed texel: {l:?}");
        assert!(f[0] > 0.2 && f[1] > 0.1, "the shaded normal is tilted the same way: {f:?}");

        let flipped = op_core::TexRef { flip_green: true, ..op_core::TexRef::new(TEX_ID) };
        rig.run(&textured(op_core::Slot::Normal, flipped, Default::default()), &[PaintOp::Reset(SIZE)]);
        let (lf, ff) = (layer_normal(&rig), final_normal(&rig));
        assert!((lf[0] - l[0]).abs() < 0.02 && (lf[1] + l[1]).abs() < 0.02, "green is mirrored: {l:?} vs {lf:?}");
        assert!(ff[1] < 0.0 && ff[0] > 0.2, "the surface now tilts down: {ff:?}");
    }

    /// A texture the library does not have is ignored: the channel falls back to its constant.
    #[test]
    fn a_missing_texture_falls_back_to_the_constant() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut f = FillParams { base: Some([1.0, 0.0, 0.0]), ..Default::default() };
        f.maps.base = Some(op_core::TexRef::new("deadbeefdeadbeefdeadbeefdeadbeef"));
        let mut doc = Document::default();
        doc.add(None, None, "x", NodeKind::Fill(f));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(near(base_at(&rig, (0.5, 0.5)), [1.0, 0.0, 0.0]), "{:?}", base_at(&rig, (0.5, 0.5)));
    }

    /// Textures continue across UV tiles instead of restarting in each.
    #[test]
    fn textures_run_on_across_uv_tiles() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        rig.gpu.set_mesh_tiles(&rig.device, &two_tile_sphere(), true);
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &quad_image());
        // Tiling 0.5 stretches the image over two tiles: tile 1001 shows its left half, tile 1002 its right half.
        let uv = op_core::UvTransform { tiling: [0.5, 1.0], ..Default::default() };
        let doc = textured(op_core::Slot::BaseColor, op_core::TexRef::new(TEX_ID), uv);
        rig.run_sets(&[SetBatch { tile: None, set: 0, doc, ops: vec![PaintOp::Reset(SIZE)] }]);
        let at = |t: usize| {
            let base = rig.read(&rig.gpu.sets[0].tiles[t].output.base, 8);
            let i = (((SIZE / 4) * SIZE + SIZE / 2) * 8) as usize; // u = 0.5, v = 0.25
            [0, 1, 2].map(|c| f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]])))
        };
        // Texture coordinate u = 0.25 (tile 1001, u = 0.5) -> left half, red; tile 1002 -> u = 0.75 -> right half, green.
        let (a, b) = (at(0), at(1));
        assert!(a[0] > 0.9 && a[1] < 0.1, "tile 1001 is red at its middle: {a:?}");
        assert!(b[1] > 0.9 && b[0] < 0.1, "tile 1002 continues with green: {b:?}");
    }

    /// Paint in two texture sets survives a save and reload, each set keeping its own textures.
    #[test]
    fn two_sets_of_paint_survive_a_project_round_trip() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mesh = two_set_sphere();
        rig.gpu.set_mesh(&rig.device, &mesh);
        let (mut a, mut b) = (Document::default(), Document::default());
        let pa = a.add(None, None, "paint a", NodeKind::Paint);
        let pb = b.add(None, None, "paint b", NodeKind::Paint);
        let reset = |set, doc: &Document| SetBatch { tile: None, set, doc: doc.clone(), ops: vec![PaintOp::Reset(SIZE)] };
        rig.run_sets(&[reset(0, &a), reset(1, &b)]);

        // Set 0 gets a real stroke; set 1 is restored from stored bytes (the camera only sees set 0).
        let stamp = dab(SIZE as f32 / 2.0, 20.0, 1.0, [0.0, 1.0, 0.0, 0.0]);
        let stroke = SetBatch {
            tile: None,
            set: 0,
            doc: a.clone(),
            ops: vec![PaintOp::Stamps { layer: pa, target: Target::Content, stamps: vec![stamp] }],
        };
        let n = (SIZE * SIZE) as usize;
        let blobs_b = vec![
            op_project::TextureBlob { node: pb, kind: op_project::BlobKind::Base, bytes: (0..n * 8).map(|i| (i % 251) as u8).collect() },
            op_project::TextureBlob { node: pb, kind: op_project::BlobKind::Metal, bytes: (0..n * 2).map(|i| (i % 13) as u8).collect() },
            op_project::TextureBlob { node: pb, kind: op_project::BlobKind::Rough, bytes: (0..n * 2).map(|i| (i % 7) as u8).collect() },
            op_project::TextureBlob { node: pb, kind: op_project::BlobKind::Height, bytes: (0..n * 2).map(|i| (i % 5) as u8).collect() },
        ];
        let load_b = SetBatch { tile: None, set: 1, doc: b.clone(), ops: vec![PaintOp::Load { size: SIZE, blobs: blobs_b.clone() }] };
        rig.run_sets(&[stroke, load_b]);

        let capture = |rig: &Rig, set: usize, doc: &Document| rig.gpu.sets[set].capture_blobs(&rig.device, &rig.queue, doc);
        let (c0, c1) = (capture(&rig, 0, &a), capture(&rig, 1, &b));
        assert!(c0.iter().any(|t| t.bytes.iter().any(|&x| x != 0)), "set 0 holds the stroke");
        assert_eq!(c1, blobs_b, "set 1 holds exactly what was loaded");

        let project = op_project::Project {
            mesh_name: "sphere".into(),
            mesh: mesh.clone(),
            sets: vec![
                op_project::SetData { texture_size: SIZE, document: a.clone(), tiles: vec![op_project::TileData { blobs: c0.clone(), baked: Default::default() }] },
                op_project::SetData { texture_size: SIZE, document: b.clone(), tiles: vec![op_project::TileData { blobs: c1.clone(), baked: Default::default() }] },
            ],
            active: 1,
            udim: false,
            hidden_sets: Vec::new(),
            images: Vec::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two.opproj");
        op_project::save(&path, &project).unwrap();
        let loaded = op_project::load(&path).unwrap();
        assert_eq!(loaded.active, 1);

        let mut fresh = Rig::new().unwrap();
        fresh.gpu.set_mesh(&fresh.device, &loaded.mesh);
        let batches: Vec<SetBatch> = loaded
            .sets
            .iter()
            .enumerate()
            .map(|(i, s)| SetBatch { tile: None, set: i, doc: s.document.clone(), ops: vec![PaintOp::Load { size: s.texture_size, blobs: s.tiles[0].blobs.clone() }] })
            .collect();
        fresh.run_sets(&batches);
        assert_eq!(capture(&fresh, 0, &loaded.sets[0].document), c0, "set 0 after reload");
        assert_eq!(capture(&fresh, 1, &loaded.sets[1].document), c1, "set 1 after reload");
    }

    /// GPU composite -> CPU export -> PNG on disk must reproduce the document's values.
    #[test]
    fn exported_pngs_match_the_document() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        doc.add(None, None, "fill", fill(Some([0.8, 0.55, 0.30]), Some(0.25), Some(0.6)));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let [base, metal, rough, _, _] = output_bytes(&rig);
        let data = op_export::TextureSetData::from_composite(SIZE, &base, &metal, &rough).unwrap();
        let preset = op_export::builtin_presets().remove(0);
        let dir = tempfile::tempdir().unwrap();
        let ctx = op_export::ExportContext { project: "p", texture_set: "s", mesh: "m", out_dir: dir.path(), udim: None };
        op_export::export(&preset, &data, &ctx).unwrap();

        let near = |a: u8, b: f32| (a as f32 - b * 255.0).abs() <= 1.5;
        let img = image::open(dir.path().join("p_s_BaseColor.png")).unwrap().to_rgb8();
        let px = img.get_pixel(SIZE / 2, SIZE / 2).0;
        assert!(near(px[0], 0.8) && near(px[1], 0.55) && near(px[2], 0.30), "base color {px:?}");
        let r = image::open(dir.path().join("p_s_Roughness.png")).unwrap().to_luma8();
        let m = image::open(dir.path().join("p_s_Metallic.png")).unwrap().to_luma8();
        assert!(near(r.get_pixel(3, 3).0[0], 0.6) && near(m.get_pixel(3, 3).0[0], 0.25));
    }

    /// A set painted with imported textures exports what the textures say: the base colour image
    /// (sRGB round trip, right way up), a scalar roughness texture, and a tilted normal map.
    #[test]
    fn textured_set_exports_the_textures_it_uses() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let (rough_id, normal_id) = ("11111111111111111111111111111111", "22222222222222222222222222222222");
        rig.gpu.upload_image(&rig.device, &rig.queue, TEX_ID, &quad_image());
        let one = |rgba8: [u8; 4]| op_image::Decoded { width: 1, height: 1, rgba8: rgba8.to_vec() };
        rig.gpu.upload_image(&rig.device, &rig.queue, rough_id, &one([64, 64, 64, 255]));
        rig.gpu.upload_image(&rig.device, &rig.queue, normal_id, &one([200, 190, 220, 255]));
        let mut f = FillParams::default();
        f.maps.set(op_core::Slot::BaseColor, Some(op_core::TexRef::new(TEX_ID)));
        f.maps.set(op_core::Slot::Roughness, Some(op_core::TexRef::new(rough_id)));
        f.maps.set(op_core::Slot::Normal, Some(op_core::TexRef::new(normal_id)));
        let mut doc = Document::default();
        doc.add(None, None, "bricks", NodeKind::Fill(f));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);

        // The same steps the app's export job takes.
        let [base, metal, rough, height, layer_normal] = output_bytes(&rig);
        let mut data = op_export::TextureSetData::from_composite(SIZE, &base, &metal, &rough).unwrap();
        data.set_height(&height).unwrap();
        data.set_layer_normal(&layer_normal).unwrap();
        data.compute_normal(doc.height_strength);
        let preset = op_export::builtin_presets().into_iter().find(|p| p.name.starts_with("Unreal")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ctx = op_export::ExportContext { project: "p", texture_set: "s", mesh: "m", out_dir: dir.path(), udim: None };
        op_export::export(&preset, &data, &ctx).unwrap();

        let q = SIZE / 4;
        let color = image::open(dir.path().join("p_s_BaseColor.png")).unwrap().to_rgb8();
        let at = |x: u32, y: u32| color.get_pixel(x, y).0;
        let close = |a: [u8; 3], b: [u8; 3]| (0..3).all(|c| (a[c] as i32 - b[c] as i32).abs() <= 40);
        assert!(close(at(q, q), [255, 0, 0]), "top left is red: {:?}", at(q, q));
        assert!(close(at(3 * q, q), [0, 255, 0]), "top right is green: {:?}", at(3 * q, q));
        assert!(close(at(q, 3 * q), [0, 0, 255]), "bottom left is blue: {:?}", at(q, 3 * q));
        assert!(close(at(3 * q, 3 * q), [255, 255, 255]), "bottom right is white");

        // Roughness is a data map: the texel value passes through unchanged (the Unreal preset packs it in G).
        let orm = image::open(dir.path().join("p_s_OcclusionRoughnessMetallic.png")).unwrap().to_rgb8();
        let g = orm.get_pixel(q, q).0[1] as i32;
        assert!((g - 64).abs() <= 2, "roughness texel 64 exported as {g}");

        // The tilted normal map survives export: x and y lean away from flat, z stays up.
        let normal = image::open(dir.path().join("p_s_Normal.png")).unwrap().to_rgb8();
        let n = normal.get_pixel(q, q).0;
        assert!(n[0] > 150 && n[2] > 128, "normal red/blue: {n:?}");
        assert!(n[1] != 128, "the DirectX green channel moved off flat: {n:?}");
    }

    /// The center pixel of the last rendered frame (RGBA8).
    fn center_pixel(rig: &Rig) -> [u8; 4] {
        let tex = &rig.gpu.offscreen.as_ref().expect("a frame was rendered").color;
        let px = crate::gpu_io::read_texture(&rig.device, &rig.queue, tex, SIZE, 4);
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    }

    /// Preview modes: 0 shaded, 1 the map as-is, 2 shaded with the map darkening the ambient light.
    #[test]
    fn baked_map_preview_modes_render_as_documented() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let doc = Document::with_default_material();
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let shaded = center_pixel(&rig);

        // Mode 1 shows the image exactly.
        let red: Vec<u8> = [255, 0, 0, 255].repeat(4);
        rig.gpu.set_preview(&rig.device, &rig.queue, Some((2, &red)));
        rig.globals.light_dir[3] = 1.0;
        rig.run(&doc, &[]);
        assert_eq!(center_pixel(&rig), [255, 0, 0, 255], "mode 1 shows the map unchanged");

        // Mode 2 with a dark map must darken the shaded result; with no map it must not change it.
        let dark: Vec<u8> = [64, 64, 64, 255].repeat(4);
        rig.gpu.set_preview(&rig.device, &rig.queue, Some((2, &dark)));
        rig.globals.light_dir[3] = 2.0;
        rig.run(&doc, &[]);
        let occluded = center_pixel(&rig);
        let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
        assert!(lum(occluded) < lum(shaded), "baked AO must darken: {occluded:?} vs {shaded:?}");

        rig.gpu.set_preview(&rig.device, &rig.queue, None);
        rig.run(&doc, &[]);
        assert_eq!(center_pixel(&rig), shaded, "mode 2 without a map is the plain shaded view");

        // Mode 0 ignores whatever preview is bound.
        rig.gpu.set_preview(&rig.device, &rig.queue, Some((2, &red)));
        rig.globals.light_dir[3] = 0.0;
        rig.run(&doc, &[]);
        assert_eq!(center_pixel(&rig), shaded);
    }

    /// Mask levels reshape the mask where it is used: stretch, gamma and invert.
    #[test]
    fn mask_levels_stretch_gamma_and_invert_the_mask() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "metal", fill(None, Some(1.0), None));
        doc.find_mut(layer).unwrap().mask = Some(Mask::new(true));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        // A flat mask of 0.5 (128).
        let n = (SIZE * SIZE) as usize;
        rig.run(&doc, &[PaintOp::SetMask { layer, size: SIZE, bytes: vec![128; n] }]);
        let coverage = |rig: &Rig| texel_rg(&rig.metal(), (0.5, 0.5)).1 as f32 / 255.0;
        let check = |rig: &mut Rig, adjust: op_core::MaskAdjust| {
            let mut d = doc.clone();
            d.find_mut(layer).unwrap().mask.as_mut().unwrap().adjust = adjust;
            rig.run(&d, &[]);
            (coverage(rig), adjust.apply(128.0 / 255.0))
        };
        for adjust in [
            op_core::MaskAdjust::default(),
            op_core::MaskAdjust { black: 0.25, white: 0.75, ..Default::default() },
            op_core::MaskAdjust { gamma: 2.0, ..Default::default() },
            op_core::MaskAdjust { invert: true, ..Default::default() },
            op_core::MaskAdjust { black: 0.6, ..Default::default() },
        ] {
            let (got, want) = check(&mut rig, adjust);
            assert!((got - want).abs() < 0.02, "{adjust:?}: GPU {got} vs CPU {want}");
        }
        assert!(check(&mut rig, op_core::MaskAdjust { gamma: 2.0, ..Default::default() }).0 > 0.6, "gamma > 1 brightens");
    }

    /// A generated mask uploaded with `SetMask` gates its layer like a painted one.
    #[test]
    fn generated_mask_content_is_uploaded_and_composited() {
        use op_core::{Generator, GeneratorKind};
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "metal", fill(None, Some(1.0), None));
        doc.find_mut(layer).unwrap().mask = Some(Mask::generated(Generator::new(GeneratorKind::Noise)));
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);

        // Left half white, right half black.
        let n = SIZE as usize;
        let bytes: Vec<u8> = (0..n * n).map(|i| if i % n < n / 2 { 255 } else { 0 }).collect();
        rig.run(&doc, &[PaintOp::SetMask { layer, size: SIZE, bytes: bytes.clone() }]);
        let metal = rig.metal();
        let probe = |x: usize, y: usize| metal[(y * n + x) * 2 + 1];
        assert_eq!(probe(10, 40), 255, "left half is revealed");
        assert_eq!(probe(n - 10, 40), 0, "right half stays hidden");

        // A result for another resolution, or with the wrong byte count, is ignored.
        rig.run(&doc, &[PaintOp::SetMask { layer, size: SIZE / 2, bytes: vec![0; (n / 2) * (n / 2)] }]);
        rig.run(&doc, &[PaintOp::SetMask { layer, size: SIZE, bytes: vec![0; 5] }]);
        rig.run(&doc, &[PaintOp::SetMask { layer: 12345, size: SIZE, bytes }]); // no such layer
        assert_eq!(rig.metal(), metal, "mismatched uploads must not change anything");
    }
}
