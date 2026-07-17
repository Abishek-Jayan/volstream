use crate::gpu::GpuContext;
use crate::texture::VolumeTexture;

/// Default per-eye resolution used when the headset has not yet reported its
/// native viewport size.
pub const DEFAULT_EYE_WIDTH: u32 = 1024;
pub const DEFAULT_EYE_HEIGHT: u32 = 1024;

pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Camera + volume parameters uploaded to the GPU every frame.
/// Must match the `CameraUniforms` struct in raymarch.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub left_view_inv: [f32; 16],
    pub left_proj_inv: [f32; 16],
    pub right_view_inv: [f32; 16],
    pub right_proj_inv: [f32; 16],
    pub world_to_volume: [f32; 16],
    /// [eye_width, eye_height, step_size, max_steps]
    pub params: [f32; 4],
    /// [sample_density, _unused, _unused, _unused]
    /// sample_density: fraction of ray-march samples to evaluate (0.0–1.0).
    pub params2: [f32; 4],
}

/// All GPU resources for the volume rendering pipeline.
pub struct GpuPipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub volume_bind_group: wgpu::BindGroup,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
}

impl GpuPipeline {
    pub fn new(gpu: &GpuContext, vol_tex: &VolumeTexture, eye_width: u32, eye_height: u32) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raymarch-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/raymarch.wgsl").into()),
            });

        // ── Bind group layouts ─────────────────────────────────────────────────
        let camera_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let volume_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("volume-bgl"),
                entries: &[
                    // binding 0: 3D volume texture (R32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: volume sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // ── Pipeline layout ────────────────────────────────────────────────────
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline-layout"),
                bind_group_layouts: &[&camera_bgl, &volume_bgl],
                push_constant_ranges: &[],
            });

        // ── Render pipeline ────────────────────────────────────────────────────
        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("volume-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: OUTPUT_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // ── Camera uniform buffer ──────────────────────────────────────────────
        let camera_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── Volume bind group ──────────────────────────────────────────────────
        let volume_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume-bg"),
            layout: &volume_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&vol_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&vol_tex.sampler),
                },
            ],
        });

        // ── Output texture (render target) ─────────────────────────────────────
        let (output_texture, output_view) = create_output_texture(gpu, eye_width, eye_height);

        GpuPipeline {
            render_pipeline,
            camera_buffer,
            camera_bind_group,
            volume_bind_group,
            output_texture,
            output_view,
        }
    }

    /// Resize the output render target to a new per-eye resolution.
    /// All other GPU resources (shader, bind groups, pipeline) are reused.
    pub fn resize(&mut self, gpu: &GpuContext, eye_width: u32, eye_height: u32) {
        let (output_texture, output_view) = create_output_texture(gpu, eye_width, eye_height);
        self.output_texture = output_texture;
        self.output_view = output_view;
    }

    /// Upload new camera uniforms for the current frame.
    pub fn update_camera(&self, gpu: &GpuContext, uniforms: &CameraUniforms) {
        gpu.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Execute the render pass into `output_texture`.
    pub fn render(&self, gpu: &GpuContext) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.volume_bind_group, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle
        }

        gpu.queue.submit([encoder.finish()]);
    }
}

/// Create the side-by-side stereo output texture and its default view.
fn create_output_texture(
    gpu: &GpuContext,
    eye_width: u32,
    eye_height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output-texture"),
        size: wgpu::Extent3d {
            width: eye_width * 2,
            height: eye_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}
