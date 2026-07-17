use openh264::encoder::{EncoderConfig, RateControlMode};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use openh264::OpenH264API;

use crate::{EncoderError, FrameEncoder};

pub struct H264Encoder {
    inner: openh264::encoder::Encoder,
}

impl H264Encoder {
    pub fn new(
        _width: u32,
        _height: u32,
        fps: u32,
        bitrate_kbps: u32,
    ) -> Result<Self, EncoderError> {
        let config = EncoderConfig::new()
            .set_bitrate_bps(bitrate_kbps * 1000)
            .max_frame_rate(fps as f32)
            .rate_control_mode(RateControlMode::Bitrate)
            .enable_skip_frame(false)
            .debug(false);

        let api = OpenH264API::from_source();
        let inner = openh264::encoder::Encoder::with_api_config(api, config)
            .map_err(|e| EncoderError::Init(e.to_string()))?;

        Ok(Self { inner })
    }
}

impl FrameEncoder for H264Encoder {
    fn encode(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, EncoderError> {
        let rgba_src = RgbaSliceU8::new(rgba, (width as usize, height as usize));
        let yuv = YUVBuffer::from_rgb_source(rgba_src);

        let bitstream = self
            .inner
            .encode(&yuv)
            .map_err(|e| EncoderError::Encode(e.to_string()))?;

        Ok(bitstream.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba_frame(width: usize, height: usize) -> Vec<u8> {
        // Solid mid-grey frame
        vec![128u8; width * height * 4]
    }

    #[test]
    fn encoder_initializes() {
        let result = H264Encoder::new(256, 256, 30, 2000);
        assert!(result.is_ok(), "encoder init failed: {:?}", result.err());
    }

    #[test]
    fn encode_first_frame_returns_nalus() {
        let mut enc = H264Encoder::new(256, 256, 30, 2000).expect("init");
        let rgba = make_rgba_frame(256, 256);
        let nalus = enc.encode(&rgba, 256, 256).expect("encode");
        assert!(
            !nalus.is_empty(),
            "expected non-empty NAL output on first frame"
        );
    }

    #[test]
    fn encode_multiple_frames_succeeds() {
        let mut enc = H264Encoder::new(128, 128, 60, 1000).expect("init");
        let rgba = make_rgba_frame(128, 128);
        for i in 0..5 {
            let result = enc.encode(&rgba, 128, 128);
            assert!(result.is_ok(), "frame {i} failed: {:?}", result.err());
        }
    }

    #[test]
    fn encode_stereo_resolution() {
        // Side-by-side stereo frame as produced by the renderer (2048x1024)
        let mut enc = H264Encoder::new(2048, 1024, 72, 8000).expect("init");
        let rgba = make_rgba_frame(2048, 1024);
        let nalus = enc.encode(&rgba, 2048, 1024).expect("encode stereo frame");
        assert!(!nalus.is_empty());
    }
}
