use crate::gpu::GpuContext;

fn aligned_bytes_per_row(width: u32) -> u32 {
    let unaligned = width * 4;
    (unaligned + 255) & !255
}

/// A single CPU-visible staging buffer for one frame's worth of pixel data.
struct Slot {
    buffer: wgpu::Buffer,
    output_width: u32,
    output_height: u32,
    bytes_per_row: u32,
}

impl Slot {
    fn new(gpu: &GpuContext, eye_width: u32, eye_height: u32) -> Self {
        let output_width = eye_width * 2;
        let output_height = eye_height;
        let bytes_per_row = aligned_bytes_per_row(output_width);
        let buffer_size = (bytes_per_row * output_height) as u64;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-slot"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            output_width,
            output_height,
            bytes_per_row,
        }
    }

    /// Submit a copy of `src_texture` into this buffer and return the
    /// `SubmissionIndex` that can be waited on later.
    fn submit_copy(&self, gpu: &GpuContext, src: &wgpu::Texture) -> wgpu::SubmissionIndex {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback-copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.output_height),
                },
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()])
    }

    /// Wait for `sub` to complete, then map, read, and unmap this buffer.
    ///
    /// Returns exactly `output_width * output_height * 4` bytes with padding
    /// rows stripped.
    fn wait_and_read(&self, gpu: &GpuContext, sub: wgpu::SubmissionIndex) -> Vec<u8> {
        // Request the mapping first so the callback is queued before we poll.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });

        // Wait only for the submission that wrote this slot, not all GPU work.
        gpu.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub));
        rx.recv()
            .expect("readback channel closed")
            .expect("GPU readback mapping failed");

        let raw = self.buffer.slice(..).get_mapped_range();
        let image_row = (self.output_width * 4) as usize;
        let padded_row = self.bytes_per_row as usize;
        let data = if image_row == padded_row {
            raw.to_vec()
        } else {
            let mut out = Vec::with_capacity(image_row * self.output_height as usize);
            for row in 0..self.output_height as usize {
                out.extend_from_slice(&raw[row * padded_row..row * padded_row + image_row]);
            }
            out
        };
        drop(raw);
        self.buffer.unmap();
        data
    }
}

/// Double-buffered GPU → CPU readback.
///
/// Pipelines the GPU copy latency by keeping two slots and alternating between
/// them each frame:
///
/// - Frame N: submit copy into slot A, return slot B's data (from frame N-1).
/// - Frame N+1: submit copy into slot B, return slot A's data (from frame N).
///
/// By the time we wait for slot B (submitted one full frame interval ago), the
/// GPU has typically already finished writing it — so `poll(WaitForSubmissionIndex)`
/// returns almost immediately instead of blocking for a full GPU-render cycle.
///
/// **Latency trade-off**: the pixel data returned on iteration N was rendered on
/// iteration N-1 (≈14 ms at 72 fps).  This is well within the 40-ms
/// pose-prediction horizon the render thread applies, so perceived latency does
/// not increase in practice.
///
/// **First call** (and first call after `resize`): returns `None` because there
/// is no previous frame to return yet.  The caller should skip encoding that
/// iteration.
pub struct DoubleReadback {
    slots: [Slot; 2],
    /// The outstanding GPU copy: (slot index, submission index to wait for).
    pending: Option<(usize, wgpu::SubmissionIndex)>,
    /// Slot index to write on the *next* `submit_and_read` call.
    cur: usize,
}

impl DoubleReadback {
    pub fn new(gpu: &GpuContext, eye_width: u32, eye_height: u32) -> Self {
        Self {
            slots: [
                Slot::new(gpu, eye_width, eye_height),
                Slot::new(gpu, eye_width, eye_height),
            ],
            pending: None,
            cur: 0,
        }
    }

    /// Recreate both slots for a new eye resolution and reset the pipeline.
    ///
    /// wgpu tracks buffer lifetimes internally, so the old buffers are freed
    /// safely after any in-flight GPU work completes.  The next call to
    /// `submit_and_read` will return `None` (warmup frame).
    pub fn resize(&mut self, gpu: &GpuContext, eye_width: u32, eye_height: u32) {
        *self = Self::new(gpu, eye_width, eye_height);
    }

    /// Submit a copy of `src_texture` into the current slot, then return the
    /// RGBA bytes from the *previous* submission.
    ///
    /// Returns `None` on the first call after construction or `resize` — there
    /// is no previous frame to return yet.
    pub fn submit_and_read(&mut self, gpu: &GpuContext, src: &wgpu::Texture) -> Option<Vec<u8>> {
        let cur = self.cur;
        self.cur = 1 - self.cur; // toggle 0 ↔ 1

        // Submit the current frame's copy.
        let sub = self.slots[cur].submit_copy(gpu, src);

        // Read the previous frame's slot (if any) and store the current as pending.
        let result = self.pending.take().map(|(prev_slot, prev_sub)| {
            // The previous submission was issued ~1 frame ago; the GPU is almost
            // certainly done.  This poll is typically a no-op.
            self.slots[prev_slot].wait_and_read(gpu, prev_sub)
        });

        self.pending = Some((cur, sub));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_gpu() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    /// Create a minimal COPY_SRC texture (side-by-side dimensions: eye_w*2 × eye_h).
    fn make_src_texture(gpu: &GpuContext, eye_w: u32, eye_h: u32) -> wgpu::Texture {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-src"),
            size: wgpu::Extent3d {
                width: eye_w * 2,
                height: eye_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    #[test]
    fn first_call_returns_none() {
        let Some(gpu) = get_gpu() else {
            println!("Skipping: no GPU");
            return;
        };
        let tex = make_src_texture(&gpu, 4, 4);
        let mut rb = DoubleReadback::new(&gpu, 4, 4);
        assert!(rb.submit_and_read(&gpu, &tex).is_none());
    }

    #[test]
    fn second_call_returns_correct_byte_count() {
        let Some(gpu) = get_gpu() else {
            println!("Skipping: no GPU");
            return;
        };
        let eye_w = 8u32;
        let eye_h = 8u32;
        let tex = make_src_texture(&gpu, eye_w, eye_h);
        let mut rb = DoubleReadback::new(&gpu, eye_w, eye_h);
        assert!(rb.submit_and_read(&gpu, &tex).is_none()); // warmup
        let data = rb
            .submit_and_read(&gpu, &tex)
            .expect("expected Some on second call");
        let expected = (eye_w * 2 * eye_h * 4) as usize;
        assert_eq!(data.len(), expected);
    }

    #[test]
    fn resize_resets_pipeline_to_none() {
        let Some(gpu) = get_gpu() else {
            println!("Skipping: no GPU");
            return;
        };
        let tex = make_src_texture(&gpu, 4, 4);
        let mut rb = DoubleReadback::new(&gpu, 4, 4);
        rb.submit_and_read(&gpu, &tex); // frame 0 → None
        rb.submit_and_read(&gpu, &tex); // frame 1 → Some
        rb.resize(&gpu, 4, 4); // reset
        assert!(rb.submit_and_read(&gpu, &tex).is_none()); // warmup again
    }

    #[test]
    fn consecutive_calls_all_return_data() {
        let Some(gpu) = get_gpu() else {
            println!("Skipping: no GPU");
            return;
        };
        let eye_w = 4u32;
        let eye_h = 4u32;
        let tex = make_src_texture(&gpu, eye_w, eye_h);
        let mut rb = DoubleReadback::new(&gpu, eye_w, eye_h);
        let expected = (eye_w * 2 * eye_h * 4) as usize;

        rb.submit_and_read(&gpu, &tex); // frame 0 → None (warmup)
        for _ in 0..4 {
            let data = rb.submit_and_read(&gpu, &tex).expect("expected Some");
            assert_eq!(data.len(), expected);
        }
    }
}
