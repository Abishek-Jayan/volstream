mod camera;
mod gpu;
mod pipeline;
mod readback;
mod texture;

pub use camera::{HeadPose, DEFAULT_IPD};
pub use gpu::GpuContext;
pub use pipeline::{DEFAULT_EYE_HEIGHT, DEFAULT_EYE_WIDTH};

use glam::{Mat4, Vec3};
use pipeline::CameraUniforms;
use volume_loader::VolumeData;

#[derive(thiserror::Error, Debug)]
pub enum RendererError {
    #[error("GPU initialization failed: {0}")]
    GpuInit(String),
    #[error("Render error: {0}")]
    Render(String),
}

/// A rendered stereo frame, side-by-side RGBA8.
/// Left eye occupies x ∈ [0, eye_width), right eye x ∈ [eye_width, eye_width*2).
pub struct StereoFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Pre-computed volume geometry: maps world space → volume [0,1]³.
struct VolumeGeometry {
    world_to_volume: Mat4,
    step_size: f32,
}

impl VolumeGeometry {
    fn from_volume(volume: &VolumeData) -> Self {
        let [nx, ny, nz] = volume.dims;
        let [sx, sy, sz] = volume.spacing;

        // Physical size per axis in mm
        let phys = [nx as f32 * sx, ny as f32 * sy, nz as f32 * sz];
        let max_phys = phys.iter().cloned().fold(0.0f32, f32::max).max(1e-6);

        // The volume is rendered inside a [-0.5, 0.5]³ world-space cube, scaled
        // non-uniformly to preserve the physical aspect ratio.
        //
        // world_to_volume:  world → [0,1]³ uvw
        //   1. scale world by max_phys / phys_axis  (un-stretch non-isotropic data)
        //   2. translate by +0.5  (shift [-0.5,0.5] → [0,1])
        let scale = Vec3::new(max_phys / phys[0], max_phys / phys[1], max_phys / phys[2]);
        let world_to_volume = Mat4::from_translation(Vec3::splat(0.5)) * Mat4::from_scale(scale);

        // Step size: traverse the unit cube with ~2 samples per voxel along the
        // longest axis.
        let max_dim = nx.max(ny).max(nz) as f32;
        let step_size = 1.0 / (max_dim * 2.0);

        Self {
            world_to_volume,
            step_size,
        }
    }
}

/// The main renderer. Owns all GPU state.
pub struct Renderer {
    gpu: GpuContext,
    pipeline: pipeline::GpuPipeline,
    readback: readback::DoubleReadback,
    geometry: VolumeGeometry,
    /// Per-eye render resolution (headset-reported or default).
    pub eye_width: u32,
    pub eye_height: u32,
    /// Inter-pupillary distance in metres (default: DEFAULT_IPD).
    pub ipd: f32,
    /// Vertical field of view in radians (default: 60°).
    pub fov_y: f32,
    /// Distance in metres from the user to the centre of the volume along -Z
    /// (default: 2.0 m).  Increasing this moves the volume further away.
    pub viewing_distance: f32,
    /// Fraction of ray-march samples to evaluate per ray (0.0–1.0; default 1.0).
    pub sample_density: f32,
}

impl Renderer {
    pub fn new(volume: &VolumeData) -> Result<Self, RendererError> {
        let gpu = GpuContext::new()?;

        let vol_tex = texture::upload_volume(&gpu, volume);

        let eye_width = DEFAULT_EYE_WIDTH;
        let eye_height = DEFAULT_EYE_HEIGHT;

        let pipeline = pipeline::GpuPipeline::new(&gpu, &vol_tex, eye_width, eye_height);
        let readback = readback::DoubleReadback::new(&gpu, eye_width, eye_height);
        let geometry = VolumeGeometry::from_volume(volume);

        Ok(Self {
            gpu,
            pipeline,
            readback,
            geometry,
            eye_width,
            eye_height,
            ipd: DEFAULT_IPD,
            fov_y: 60_f32.to_radians(),
            viewing_distance: 2.0,
            sample_density: 0.60,
        })
    }

    /// Resize the render target and readback buffer to a new per-eye resolution.
    /// Call this when the headset reports a different viewport size.
    pub fn resize(&mut self, eye_width: u32, eye_height: u32) {
        if eye_width == self.eye_width && eye_height == self.eye_height {
            return;
        }
        self.eye_width = eye_width;
        self.eye_height = eye_height;
        self.pipeline.resize(&self.gpu, eye_width, eye_height);
        self.readback.resize(&self.gpu, eye_width, eye_height);
    }

    /// Render one stereo frame from `pose`.
    ///
    /// Uses double-buffered readback: the GPU copy for this frame is submitted
    /// immediately, then the *previous* frame's pixel data is read back from the
    /// other slot.  Returns `None` on the first call (and after `resize`) because
    /// there is no previous frame yet — callers should skip encoding that iteration.
    ///
    /// From the second call onwards this returns `Some` and the GPU wait is
    /// typically near-instant (the previous copy completed during the frame interval).
    pub fn render_frame(&mut self, pose: &HeadPose) -> Result<Option<StereoFrame>, RendererError> {
        let uniforms = self.build_uniforms(pose);
        self.pipeline.update_camera(&self.gpu, &uniforms);
        self.pipeline.render(&self.gpu);

        let eye_width = self.eye_width;
        let eye_height = self.eye_height;
        let result = self
            .readback
            .submit_and_read(&self.gpu, &self.pipeline.output_texture);

        Ok(result.map(|rgba| StereoFrame {
            rgba,
            width: eye_width * 2,
            height: eye_height,
        }))
    }

    // ── Private ────────────────────────────────────────────────────────────────

    fn build_uniforms(&self, pose: &HeadPose) -> CameraUniforms {
        // Prefer the headset's actual per-eye projection matrices when the client
        // sends them.  These encode the correct FOV and asymmetric frustum for the
        // physical optics, which is required for the two eye images to fuse.
        //
        // XRView.projectionMatrix uses OpenGL NDC convention (z ∈ [-1, 1]).  The
        // ray-marching shader only uses the inverse projection to compute ray
        // directions (world_near → world_far), so the z convention difference
        // between GL and WebGPU does not affect the computed ray direction.
        //
        // Fallback: built-in symmetric perspective_rh (used before the client
        // connects or on runtimes that do not expose projections).
        let fallback_proj = Mat4::perspective_rh(
            self.fov_y,
            self.eye_width as f32 / self.eye_height as f32,
            0.01,
            1000.0,
        );
        let proj_left = pose.proj_left.unwrap_or(fallback_proj);
        let proj_right = pose.proj_right.unwrap_or(fallback_proj);

        let (left_view, right_view) = camera::compute_stereo_views(pose, self.ipd);
        let left_view_inv = left_view.inverse();
        let right_view_inv = right_view.inverse();

        let max_steps = (1.0 / self.geometry.step_size) * 1.5; // headroom

        // Shift the volume centre to (0, 0, -viewing_distance) in world space.
        let depth_offset = Mat4::from_translation(Vec3::new(0.0, 0.0, self.viewing_distance));
        let world_to_volume = self.geometry.world_to_volume * depth_offset;

        CameraUniforms {
            left_view_inv: left_view_inv.to_cols_array(),
            left_proj_inv: proj_left.inverse().to_cols_array(),
            right_view_inv: right_view_inv.to_cols_array(),
            right_proj_inv: proj_right.inverse().to_cols_array(),
            world_to_volume: world_to_volume.to_cols_array(),
            params: [
                self.eye_width as f32,
                self.eye_height as f32,
                self.geometry.step_size,
                max_steps,
            ],
            params2: [self.sample_density.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small synthetic volume (16³ gradient) for GPU tests.
    fn synthetic_volume() -> VolumeData {
        let n = 16u32;
        let data: Vec<f32> = (0..n * n * n)
            .map(|i| i as f32 / (n * n * n - 1) as f32)
            .collect();
        VolumeData::new([n, n, n], [1.0, 1.0, 1.0], data, (0.0, 1.0))
    }

    #[test]
    fn renderer_produces_non_empty_frame() {
        let volume = synthetic_volume();
        let mut renderer = match Renderer::new(&volume) {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping GPU render test (no adapter): {e}");
                return;
            }
        };

        // Camera at z=+2, identity quaternion → forward=(0,0,-1), looks through origin/volume
        let pose = HeadPose {
            position: glam::Vec3::new(0.0, 0.0, 2.0),
            orientation: glam::Quat::IDENTITY,
            proj_left: None,
            proj_right: None,
        };

        // Double-buffered readback: first call returns None (warmup), second returns data.
        renderer
            .render_frame(&pose)
            .expect("warmup render_frame failed");
        let frame = renderer
            .render_frame(&pose)
            .expect("render_frame failed")
            .expect("expected Some on second call");
        assert_eq!(frame.width, renderer.eye_width * 2);
        assert_eq!(frame.height, renderer.eye_height);
        assert_eq!(frame.rgba.len(), (frame.width * frame.height * 4) as usize);

        // At least some pixels should be non-black (the volume should be visible)
        let non_black = frame
            .rgba
            .chunks(4)
            .filter(|px| px[0] > 10 || px[1] > 10 || px[2] > 10)
            .count();
        println!(
            "Non-black pixels: {} / {}",
            non_black,
            frame.width * frame.height
        );
        assert!(
            non_black > 100,
            "Expected visible volume pixels, got {non_black}"
        );
    }

    #[test]
    fn stereo_eyes_differ() {
        let volume = synthetic_volume();
        let mut renderer = match Renderer::new(&volume) {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping GPU render test (no adapter): {e}");
                return;
            }
        };

        let pose = HeadPose {
            position: glam::Vec3::new(0.0, 0.0, 2.0),
            orientation: glam::Quat::IDENTITY,
            proj_left: None,
            proj_right: None,
        };

        renderer.render_frame(&pose).expect("warmup render failed");
        let frame = renderer
            .render_frame(&pose)
            .expect("render failed")
            .expect("expected Some on second call");

        let output_width = renderer.eye_width * 2;
        let eye_width = renderer.eye_width;
        let output_height = renderer.eye_height;

        // Compare left and right halves — they should differ (parallax from IPD)
        let row_bytes = (output_width * 4) as usize;
        let eye_bytes = (eye_width * 4) as usize;
        let mut differences = 0usize;
        for row in 0..output_height as usize {
            let row_start = row * row_bytes;
            let left = &frame.rgba[row_start..row_start + eye_bytes];
            let right = &frame.rgba[row_start + eye_bytes..row_start + eye_bytes * 2];
            differences += left.iter().zip(right).filter(|(a, b)| a != b).count();
        }
        println!("Stereo byte differences: {differences}");
        assert!(
            differences > 0,
            "Left and right eye images should differ (IPD={} m)",
            renderer.ipd
        );
    }

    #[test]
    fn resize_changes_frame_dimensions() {
        let volume = synthetic_volume();
        let mut renderer = match Renderer::new(&volume) {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping GPU render test (no adapter): {e}");
                return;
            }
        };

        renderer.resize(512, 512);
        assert_eq!(renderer.eye_width, 512);
        assert_eq!(renderer.eye_height, 512);

        let pose = HeadPose {
            position: glam::Vec3::new(0.0, 0.0, 2.0),
            orientation: glam::Quat::IDENTITY,
            proj_left: None,
            proj_right: None,
        };
        // After resize the double-buffer pipeline resets — need a warmup call again.
        renderer
            .render_frame(&pose)
            .expect("warmup after resize failed");
        let frame = renderer
            .render_frame(&pose)
            .expect("render after resize failed")
            .expect("expected Some on second call after resize");
        assert_eq!(frame.width, 1024);
        assert_eq!(frame.height, 512);
        assert_eq!(frame.rgba.len(), (1024 * 512 * 4) as usize);
    }

    #[test]
    fn world_to_volume_maps_origin_to_center() {
        let volume = synthetic_volume();
        let geom = VolumeGeometry::from_volume(&volume);
        // Origin in world space should map to (0.5, 0.5, 0.5) in volume space
        let center = geom.world_to_volume.transform_point3(glam::Vec3::ZERO);
        assert!((center.x - 0.5).abs() < 1e-5, "x: {}", center.x);
        assert!((center.y - 0.5).abs() < 1e-5, "y: {}", center.y);
        assert!((center.z - 0.5).abs() < 1e-5, "z: {}", center.z);
    }

    #[test]
    fn world_to_volume_maps_half_extent_to_boundary() {
        // Isotropic 16³, spacing 1.0 → physical size 16mm → world extent [-0.5, 0.5]
        let volume = synthetic_volume();
        let geom = VolumeGeometry::from_volume(&volume);
        // Point at (0.5, 0.5, 0.5) in world space should map to (1.0, 1.0, 1.0) in volume
        let corner = geom
            .world_to_volume
            .transform_point3(glam::Vec3::splat(0.5));
        assert!((corner.x - 1.0).abs() < 1e-5, "corner x: {}", corner.x);
    }
}
