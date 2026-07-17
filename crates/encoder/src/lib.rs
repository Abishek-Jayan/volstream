/// Trait for frame encoders. Implementations: H.264 (default), VP8 (feature-gated).
pub trait FrameEncoder: Send {
    /// Encode an RGBA8 frame. Returns encoded bytes (NAL units or similar).
    fn encode(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, EncoderError>;
}

#[derive(thiserror::Error, Debug)]
pub enum EncoderError {
    #[error("Encoder initialization failed: {0}")]
    Init(String),
    #[error("Encode failed: {0}")]
    Encode(String),
}

#[cfg(feature = "nvenc")]
pub mod nvenc;

#[cfg(feature = "h264")]
pub mod h264;

/// Create the best available encoder for the given parameters.
///
/// Selection order:
/// 1. **NVENC** (`nvenc` feature) — hardware H.264 via `h264_nvenc` (RTX 3090, <1 ms/frame).
///    Falls back automatically if NVENC is not available at runtime (no NVIDIA GPU, etc.).
/// 2. **OpenH264** (`h264` feature) — software fallback (~8–20 ms/frame at 2048×1024).
pub fn default_encoder(
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<Box<dyn FrameEncoder>, EncoderError> {
    #[cfg(feature = "nvenc")]
    match nvenc::NvencEncoder::new(width, height, fps, bitrate_kbps) {
        Ok(enc) => return Ok(Box::new(enc)),
        Err(e) => {
            tracing::warn!("NVENC unavailable ({e}), falling back to software encoder");
        }
    }

    #[cfg(feature = "h264")]
    {
        return Ok(Box::new(h264::H264Encoder::new(
            width,
            height,
            fps,
            bitrate_kbps,
        )?));
    }

    #[allow(unreachable_code)]
    {
        let _ = (width, height, fps, bitrate_kbps);
        Err(EncoderError::Init("No encoder feature enabled".into()))
    }
}
