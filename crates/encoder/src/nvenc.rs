use ffmpeg_next::{self as ffmpeg, codec, format::Pixel, software::scaling, Dictionary, Rational};

use crate::{EncoderError, FrameEncoder};

/// Hardware H.264 encoder using NVIDIA NVENC via FFmpeg's `h264_nvenc` codec.
///
/// Configured for VR streaming with ultra-low-latency settings:
/// - `preset=p4` — balanced quality/speed
/// - `tune=ull`  — ultra-low latency (disables B-frames, lookahead, scene-cut)
/// - `rc=cbr`    — constant bitrate for predictable network behaviour
/// - `delay=0`   — no encoder pipeline delay; every input frame produces output
///
/// RGBA input is converted to NV12 via libswscale before being handed to NVENC.
/// The CPU colour conversion costs ~0.5 ms vs the ~8–20 ms of OpenH264 software
/// encoding at 2048×1024.
pub struct NvencEncoder {
    encoder: codec::encoder::video::Encoder,
    scaler: scaling::Context,
    pts: i64,
}

// SAFETY: NvencEncoder is only ever used from a single thread (the encode loop).
// AVCodecContext and SwsContext are not thread-safe but are safe to *move* across
// threads as long as concurrent access is prevented, which our single-owner
// pattern guarantees.
unsafe impl Send for NvencEncoder {}

impl NvencEncoder {
    pub fn new(width: u32, height: u32, fps: u32, bitrate_kbps: u32) -> Result<Self, EncoderError> {
        ffmpeg::init().map_err(|e| EncoderError::Init(format!("ffmpeg init: {e}")))?;

        let codec = ffmpeg::encoder::find_by_name("h264_nvenc").ok_or_else(|| {
            EncoderError::Init("h264_nvenc not found — NVENC not available".into())
        })?;

        let context = codec::context::Context::new_with_codec(codec);
        let mut enc = context
            .encoder()
            .video()
            .map_err(|e| EncoderError::Init(format!("video encoder context: {e}")))?;

        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(Pixel::NV12);
        enc.set_time_base(Rational::new(1, fps as i32));
        enc.set_frame_rate(Some(Rational::new(fps as i32, 1)));
        enc.set_bit_rate(bitrate_kbps as usize * 1000);
        enc.set_max_b_frames(0);
        enc.set_gop(fps * 2); // 2-second keyframe interval

        let mut opts = Dictionary::new();
        opts.set("preset", "p4"); // balanced latency/quality for VR
        opts.set("tune", "ull"); // ultra-low latency: no lookahead, no scene-cut
        opts.set("rc", "cbr"); // constant bitrate for streaming
        opts.set("bf", "0"); // no B-frames (already implied by ull, explicit for clarity)
        opts.set("delay", "0"); // zero-delay: output packet per input frame

        let encoder = enc
            .open_with(opts)
            .map_err(|e| EncoderError::Init(format!("open h264_nvenc: {e}")))?;

        let scaler = scaling::Context::get(
            Pixel::RGBA,
            width,
            height,
            Pixel::NV12,
            width,
            height,
            scaling::Flags::BILINEAR,
        )
        .map_err(|e| EncoderError::Init(format!("swscale RGBA→NV12: {e}")))?;

        tracing::info!(
            "NVENC encoder opened: {}×{} @ {} fps, {} kbps (preset=p4, tune=ull)",
            width,
            height,
            fps,
            bitrate_kbps,
        );

        Ok(Self {
            encoder,
            scaler,
            pts: 0,
        })
    }
}

impl FrameEncoder for NvencEncoder {
    fn encode(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, EncoderError> {
        // ── 1. Fill RGBA source frame ─────────────────────────────────────────
        let mut src = ffmpeg::frame::Video::new(Pixel::RGBA, width, height);
        {
            let src_stride = width as usize * 4; // packed RGBA, no row padding
            let dst_stride = src.stride(0);
            let dst = src.data_mut(0);
            for y in 0..height as usize {
                let row = &rgba[y * src_stride..(y + 1) * src_stride];
                dst[y * dst_stride..y * dst_stride + src_stride].copy_from_slice(row);
            }
        }

        // ── 2. Convert RGBA → NV12 (preferred NVENC input format) ─────────────
        let mut nv12 = ffmpeg::frame::Video::new(Pixel::NV12, width, height);
        self.scaler
            .run(&src, &mut nv12)
            .map_err(|e| EncoderError::Encode(format!("swscale: {e}")))?;

        nv12.set_pts(Some(self.pts));
        self.pts += 1;

        // ── 3. Send to NVENC ──────────────────────────────────────────────────
        self.encoder
            .send_frame(&nv12)
            .map_err(|e| EncoderError::Encode(format!("send_frame: {e}")))?;

        // ── 4. Collect encoded NAL units ──────────────────────────────────────
        let mut output = Vec::new();
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            if let Some(data) = packet.data() {
                output.extend_from_slice(data);
            }
        }

        Ok(output)
    }
}

impl Drop for NvencEncoder {
    /// Flush any buffered frames out of the NVENC pipeline.
    fn drop(&mut self) {
        let _ = self.encoder.send_eof();
        let mut pkt = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut pkt).is_ok() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameEncoder;

    fn make_rgba(w: usize, h: usize) -> Vec<u8> {
        vec![128u8; w * h * 4]
    }

    /// Returns None on machines without NVENC (CI, non-NVIDIA hosts).
    fn try_new(w: u32, h: u32) -> Option<NvencEncoder> {
        NvencEncoder::new(w, h, 30, 2000).ok()
    }

    #[test]
    fn nvenc_initializes() {
        if NvencEncoder::new(256, 256, 30, 2000).is_err() {
            println!("Skipping: NVENC not available");
        }
    }

    #[test]
    fn encode_first_frame_returns_nalus() {
        let Some(mut enc) = try_new(256, 256) else {
            println!("Skipping: NVENC not available");
            return;
        };
        let rgba = make_rgba(256, 256);
        let nalus = enc.encode(&rgba, 256, 256).expect("encode");
        assert!(
            !nalus.is_empty(),
            "expected non-empty NAL output on first frame"
        );
    }

    #[test]
    fn encode_multiple_frames_succeeds() {
        let Some(mut enc) = try_new(128, 128) else {
            println!("Skipping: NVENC not available");
            return;
        };
        let rgba = make_rgba(128, 128);
        for i in 0..5 {
            enc.encode(&rgba, 128, 128)
                .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
        }
    }

    #[test]
    fn encode_stereo_vr_resolution() {
        let Some(mut enc) = try_new(2048, 1024) else {
            println!("Skipping: NVENC not available");
            return;
        };
        let rgba = make_rgba(2048, 1024);
        let nalus = enc.encode(&rgba, 2048, 1024).expect("stereo encode");
        assert!(!nalus.is_empty());
    }
}
