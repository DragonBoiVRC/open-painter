//! The image-based lighting environment: an HDR picture the viewport reflects, plus spherical
//! harmonics for its diffuse light. With no picture loaded a 1x1 stand-in is bound and the shader
//! uses its procedural environments instead.

use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use op_image::hdr::HdrImage;

/// The largest width of the reflection texture. Wider pictures are averaged down.
const MAX_WIDTH: u32 = 2048;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvParams {
    /// Diffuse spherical harmonics (xyz used).
    sh: [[f32; 4]; 9],
    /// x: highest mip level, y: 1 when a picture is loaded.
    info: [f32; 4],
}

pub struct EnvGpu {
    pub layout: wgpu::BindGroupLayout,
    pub bind: wgpu::BindGroup,
    sampler: wgpu::Sampler,
}

impl EnvGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let mut me = Self { bind: placeholder(device), layout, sampler };
        me.set(device, queue, None);
        me
    }

    /// Use `image` as the environment, or go back to the procedural ones with `None`.
    pub fn set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: Option<&HdrImage>) {
        let (mips, sh, loaded) = match image {
            Some(img) => {
                let small = img.downsampled(MAX_WIDTH);
                (small.mips(), img.diffuse_sh(), 1.0)
            }
            None => (vec![op_image::hdr::HdrMip { width: 1, height: 1, rgba: vec![0.0, 0.0, 0.0, 1.0] }], [[0.0; 3]; 9], 0.0),
        };
        let first = &mips[0];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment"),
            size: wgpu::Extent3d { width: first.width, height: first.height, depth_or_array_layers: 1 },
            mip_level_count: mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, m) in mips.iter().enumerate() {
            let halves: Vec<u16> = m.rgba.iter().map(|v| half::f16::from_f32(*v).to_bits()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&halves),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(m.width * 8), rows_per_image: Some(m.height) },
                wgpu::Extent3d { width: m.width, height: m.height, depth_or_array_layers: 1 },
            );
        }
        let mut params = EnvParams { sh: [[0.0; 4]; 9], info: [(mips.len() - 1) as f32, loaded, 0.0, 0.0] };
        for (dst, src) in params.sh.iter_mut().zip(sh) {
            *dst = [src[0], src[1], src[2], 0.0];
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("env_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view = texture.create_view(&Default::default());
        self.bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env_bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
    }
}

/// A bind group that is replaced right away; `new` needs some value for the field first.
fn placeholder(device: &wgpu::Device) -> wgpu::BindGroup {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("env_placeholder_layout"), entries: &[] });
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("env_placeholder"), layout: &layout, entries: &[] })
}
