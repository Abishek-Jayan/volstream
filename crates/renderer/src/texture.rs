use volume_loader::VolumeData;

use crate::gpu::GpuContext;

/// Holds the wgpu resources for the 3D volume texture.
pub struct VolumeTexture {
    #[allow(dead_code)] // keeps GPU memory alive
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

/// Upload VolumeData to a wgpu 3D texture (R32Float, single channel).
pub fn upload_volume(gpu: &GpuContext, volume: &VolumeData) -> VolumeTexture {
    let [nx, ny, nz] = volume.dims;

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume-3d"),
        size: wgpu::Extent3d {
            width: nx,
            height: ny,
            depth_or_array_layers: nz,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&volume.data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(nx * 4), // 1 float × 4 bytes per texel
            rows_per_image: Some(ny),
        },
        wgpu::Extent3d {
            width: nx,
            height: ny,
            depth_or_array_layers: nz,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });

    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("volume-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    VolumeTexture {
        texture,
        view,
        sampler,
    }
}
