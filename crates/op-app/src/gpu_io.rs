//! Blocking texture readback and upload, used for saving, loading and exporting.

use eframe::egui_wgpu::wgpu;

/// Copy a square texture to the CPU. Returns exactly `size * size * bpp` bytes,
/// tightly packed (wgpu's 256-byte row alignment is removed).
pub fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, size: u32, bpp: u32) -> Vec<u8> {
    let unpadded = size * bpp;
    let padded = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded as u64 * size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(size) },
        },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("buffer map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");
    let mapped = buf.slice(..).get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((unpadded * size) as usize);
    for row in mapped.chunks_exact(padded as usize) {
        out.extend_from_slice(&row[..unpadded as usize]);
    }
    out
}

/// Overwrite a whole square texture with tightly packed bytes.
///
/// The copy is recorded in `encoder` rather than issued with `queue.write_texture`:
/// queue writes run at the *start* of a submission, before any command already recorded
/// in the encoder, so a clear recorded earlier (e.g. when a mask is created) would erase
/// the uploaded data.
pub fn write_texture(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, tex: &wgpu::Texture, size: u32, bpp: u32, bytes: &[u8]) {
    use wgpu::util::DeviceExt;
    assert_eq!(bytes.len(), (size * size * bpp) as usize, "texture upload size mismatch");
    let unpadded = (size * bpp) as usize;
    let padded = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
    let mut staged = vec![0u8; padded * size as usize];
    for (dst, src) in staged.chunks_exact_mut(padded).zip(bytes.chunks_exact(unpadded)) {
        dst[..unpadded].copy_from_slice(src);
    }
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("texture_upload"),
        contents: &staged,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded as u32), rows_per_image: Some(size) },
        },
        tex.as_image_copy(),
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
}
