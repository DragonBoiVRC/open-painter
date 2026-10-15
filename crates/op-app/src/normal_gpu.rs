//! The normal-map pass: composited height (and an optional baked normal) to a tangent-space
//! normal texture the viewport samples.

use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};

pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    strength: f32,
    size: f32,
    has_baked: u32,
    pad: f32,
}

pub struct NormalPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The result, `size` x `size`.
    pub target: wgpu::Texture,
    baked: wgpu::Texture,
    has_baked: bool,
    size: u32,
}

fn target_texture(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("normal_map"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn dummy_baked(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    baked_texture(device, queue, 1, &[128, 128, 255, 255])
}

fn baked_texture(device: &wgpu::Device, queue: &wgpu::Queue, size: u32, rgba8: &[u8]) -> wgpu::Texture {
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("baked_normal"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: NORMAL_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        rgba8,
    )
}

impl NormalPass {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normal_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1),
                tex(2),
                tex(3),
            ],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/normal.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("normal"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState::from(NORMAL_FORMAT))],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // A flat default until a baked normal arrives; created lazily with a queue in `set_baked`.
        let baked = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("baked_normal_placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: NORMAL_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self { pipeline, layout, target: target_texture(device, size), baked, has_baked: false, size }
    }

    /// Reallocate for a new resolution. A baked normal of another size is dropped.
    pub fn resize(&mut self, device: &wgpu::Device, size: u32) {
        self.target = target_texture(device, size);
        if size != self.size {
            self.has_baked = false;
        }
        self.size = size;
    }

    /// Use `rgba8` (`size` x `size`, packed normals) as the baked detail, or remove it.
    pub fn set_baked(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, rgba8: Option<&[u8]>) {
        match rgba8 {
            Some(bytes) if bytes.len() == (self.size as usize).pow(2) * 4 => {
                self.baked = baked_texture(device, queue, self.size, bytes);
                self.has_baked = true;
            }
            _ => {
                self.baked = dummy_baked(device, queue);
                self.has_baked = false;
            }
        }
    }

    /// Record the pass: `height` is the composited (value * coverage, coverage) texture.
    pub fn run(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, height: &wgpu::Texture, layer_normal: &wgpu::Texture, strength: f32) {
        let params = Params { strength, size: self.size as f32, has_baked: self.has_baked as u32, pad: 0.0 };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("normal_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (hv, bv) = (height.create_view(&Default::default()), self.baked.create_view(&Default::default()));
        let lv = layer_normal.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal_bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&hv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&bv) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&lv) },
            ],
        });
        let view = self.target.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("normal_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
