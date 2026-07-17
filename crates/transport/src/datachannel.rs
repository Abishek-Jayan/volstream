use serde::{Deserialize, Serialize};

/// Head pose received from the WebXR client over the data channel.
///
/// **Wire formats** (both are accepted):
///
/// *Binary* (preferred — lower CPU overhead):
/// ```text
/// Offset  Size  Field
///  0       1    magic = 0x50 ('P')
///  1       1    version = 0x01
///  2       2    flags (u16 LE): bit0=has_ipd, bit1=has_proj, bit2=has_eye_dims
///  4      12    position  [f32; 3]  (LE)
/// 16      16    orientation [f32; 4] (LE, [x,y,z,w])
/// 32       8    timestamp u64 (LE, milliseconds)
/// -- if bit0: 4   ipd f32 (LE)
/// -- if bit1: 128 proj_left [f32;16] + proj_right [f32;16] (LE, col-major)
/// -- if bit2: 8   eye_width u32 + eye_height u32 (LE)
/// ```
///
/// *JSON* (fallback for older clients):
/// ```json
/// {"position":[x,y,z],"orientation":[x,y,z,w],"timestamp":123456,
///  "ipd":0.063,
///  "proj_left":[16 floats, column-major],
///  "proj_right":[16 floats, column-major],
///  "eye_width":1832,"eye_height":1920}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeadPose {
    /// World-space camera position (metres).
    pub position: [f32; 3],
    /// Orientation quaternion [x, y, z, w].
    pub orientation: [f32; 4],
    /// Client-side timestamp in milliseconds.
    pub timestamp: u64,
    /// Inter-pupillary distance in metres, measured from the headset's XR eye
    /// transforms.  Absent when the runtime does not expose stereo views.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ipd: Option<f32>,
    /// Left-eye projection matrix from `XRView.projectionMatrix` (column-major,
    /// OpenGL NDC convention: z ∈ [-1, 1]).  When present, the server uses this
    /// instead of its built-in perspective matrix so the rendered FOV and
    /// asymmetric frustum exactly match the headset's optics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proj_left: Option<[f32; 16]>,
    /// Right-eye projection matrix, same convention as `proj_left`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proj_right: Option<[f32; 16]>,
    /// Per-eye viewport width in pixels, from `XRWebGLLayer.getViewport()`.
    /// When present the server will resize its render target to match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye_width: Option<u32>,
    /// Per-eye viewport height in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye_height: Option<u32>,
}

/// Binary magic byte — first byte of every binary-format pose message.
const BINARY_MAGIC: u8 = 0x50; // 'P'
const BINARY_VERSION: u8 = 0x01;

const FLAG_HAS_IPD: u16 = 1 << 0;
const FLAG_HAS_PROJ: u16 = 1 << 1;
const FLAG_HAS_EYE_DIMS: u16 = 1 << 2;

/// Parse a `HeadPose` from a data-channel payload.
///
/// Accepts both binary (magic byte `0x50`) and JSON formats.
/// Returns `None` if the payload cannot be parsed in either format.
pub fn try_parse_pose(data: &[u8]) -> Option<HeadPose> {
    if data.first() == Some(&BINARY_MAGIC) {
        parse_binary(data)
    } else {
        serde_json::from_slice(data).ok()
    }
}

/// Encode a `HeadPose` to the compact binary wire format.
pub fn encode_pose_binary(p: &HeadPose) -> Vec<u8> {
    let mut flags: u16 = 0;
    if p.ipd.is_some() {
        flags |= FLAG_HAS_IPD;
    }
    if p.proj_left.is_some() || p.proj_right.is_some() {
        flags |= FLAG_HAS_PROJ;
    }
    if p.eye_width.is_some() || p.eye_height.is_some() {
        flags |= FLAG_HAS_EYE_DIMS;
    }

    // Fixed header: magic(1) + version(1) + flags(2) + position(12) +
    //               orientation(16) + timestamp(8) = 40 bytes.
    let mut buf = Vec::with_capacity(176);
    buf.push(BINARY_MAGIC);
    buf.push(BINARY_VERSION);
    buf.extend_from_slice(&flags.to_le_bytes());
    for v in &p.position {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in &p.orientation {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&p.timestamp.to_le_bytes());

    if flags & FLAG_HAS_IPD != 0 {
        buf.extend_from_slice(&p.ipd.unwrap_or(0.0).to_le_bytes());
    }
    if flags & FLAG_HAS_PROJ != 0 {
        let zeros = [0f32; 16];
        for v in p.proj_left.as_ref().unwrap_or(&zeros) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in p.proj_right.as_ref().unwrap_or(&zeros) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    if flags & FLAG_HAS_EYE_DIMS != 0 {
        buf.extend_from_slice(&p.eye_width.unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&p.eye_height.unwrap_or(0).to_le_bytes());
    }
    buf
}

// ── Internal binary parser ─────────────────────────────────────────────────────

fn read_f32_le(data: &[u8], offset: usize) -> Option<f32> {
    data.get(offset..offset + 4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
}
fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    data.get(offset..offset + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}
fn read_u64_le(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
}
fn read_u16_le(data: &[u8], offset: usize) -> Option<u16> {
    data.get(offset..offset + 2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
}

fn parse_binary(data: &[u8]) -> Option<HeadPose> {
    // Minimum packet: magic(1) + version(1) + flags(2) + pos(12) + orient(16) + ts(8) = 40
    if data.len() < 40 {
        return None;
    }
    if data[0] != BINARY_MAGIC || data[1] != BINARY_VERSION {
        return None;
    }

    let flags = read_u16_le(data, 2)?;

    let position = [
        read_f32_le(data, 4)?,
        read_f32_le(data, 8)?,
        read_f32_le(data, 12)?,
    ];
    let orientation = [
        read_f32_le(data, 16)?,
        read_f32_le(data, 20)?,
        read_f32_le(data, 24)?,
        read_f32_le(data, 28)?,
    ];
    let timestamp = read_u64_le(data, 32)?;

    let mut cursor = 40usize;

    let ipd = if flags & FLAG_HAS_IPD != 0 {
        let v = read_f32_le(data, cursor)?;
        cursor += 4;
        Some(v)
    } else {
        None
    };

    let (proj_left, proj_right) = if flags & FLAG_HAS_PROJ != 0 {
        if data.len() < cursor + 128 {
            return None;
        }
        let mut left = [0f32; 16];
        let mut right = [0f32; 16];
        for i in 0..16 {
            left[i] = read_f32_le(data, cursor + i * 4)?;
            right[i] = read_f32_le(data, cursor + 64 + i * 4)?;
        }
        cursor += 128;
        (Some(left), Some(right))
    } else {
        (None, None)
    };

    let (eye_width, eye_height) = if flags & FLAG_HAS_EYE_DIMS != 0 {
        if data.len() < cursor + 8 {
            return None;
        }
        let w = read_u32_le(data, cursor)?;
        let h = read_u32_le(data, cursor + 4)?;
        (Some(w), Some(h))
    } else {
        (None, None)
    };

    Some(HeadPose {
        position,
        orientation,
        timestamp,
        ipd,
        proj_left,
        proj_right,
        eye_width,
        eye_height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_pose() -> HeadPose {
        HeadPose {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 0,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        }
    }

    // ── JSON ──────────────────────────────────────────────────────────────────

    #[test]
    fn round_trips_through_json() {
        let pose = HeadPose {
            position: [1.0, 2.0, 3.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 42_000,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        let json = serde_json::to_vec(&pose).unwrap();
        let parsed = try_parse_pose(&json).expect("should parse");
        assert_eq!(parsed, pose);
    }

    #[test]
    fn parses_wire_format() {
        let raw =
            br#"{"position":[0.1,-0.2,1.5],"orientation":[0.0,0.0,0.707,0.707],"timestamp":1000}"#;
        let pose = try_parse_pose(raw).expect("valid pose");
        assert!((pose.position[0] - 0.1).abs() < 1e-6);
        assert_eq!(pose.timestamp, 1000);
    }

    #[test]
    fn returns_none_for_garbage() {
        assert!(try_parse_pose(b"not json").is_none());
        assert!(try_parse_pose(b"{}").is_none()); // missing required fields
        assert!(try_parse_pose(b"").is_none());
    }

    #[test]
    fn ignores_extra_fields() {
        let raw =
            br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0,"extra":"ignored"}"#;
        assert!(try_parse_pose(raw).is_some());
    }

    #[test]
    fn parses_ipd_when_present() {
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0,"ipd":0.064}"#;
        let pose = try_parse_pose(raw).expect("valid pose with ipd");
        assert!((pose.ipd.unwrap() - 0.064).abs() < 1e-6);
    }

    #[test]
    fn ipd_defaults_to_none_when_absent() {
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0}"#;
        let pose = try_parse_pose(raw).expect("valid pose without ipd");
        assert!(pose.ipd.is_none());
    }

    #[test]
    fn proj_matrices_default_to_none_when_absent() {
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0}"#;
        let pose = try_parse_pose(raw).expect("valid pose");
        assert!(pose.proj_left.is_none());
        assert!(pose.proj_right.is_none());
    }

    #[test]
    fn round_trips_with_proj_matrices() {
        let identity = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let pose = HeadPose {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 0,
            ipd: None,
            proj_left: Some(identity),
            proj_right: Some(identity),
            eye_width: None,
            eye_height: None,
        };
        let json = serde_json::to_vec(&pose).unwrap();
        let parsed = try_parse_pose(&json).expect("should parse");
        assert_eq!(parsed.proj_left, Some(identity));
        assert_eq!(parsed.proj_right, Some(identity));
    }

    #[test]
    fn round_trips_with_ipd() {
        let pose = HeadPose {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 0,
            ipd: Some(0.063),
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        let json = serde_json::to_vec(&pose).unwrap();
        let parsed = try_parse_pose(&json).expect("should parse");
        assert_eq!(parsed, pose);
    }

    #[test]
    fn eye_dims_default_to_none_when_absent() {
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0}"#;
        let pose = try_parse_pose(raw).expect("valid pose");
        assert!(pose.eye_width.is_none());
        assert!(pose.eye_height.is_none());
    }

    #[test]
    fn parses_eye_dims_when_present() {
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0,"eye_width":1832,"eye_height":1920}"#;
        let pose = try_parse_pose(raw).expect("valid pose with eye dims");
        assert_eq!(pose.eye_width, Some(1832));
        assert_eq!(pose.eye_height, Some(1920));
    }

    // ── Binary ────────────────────────────────────────────────────────────────

    #[test]
    fn binary_round_trip_minimal() {
        let pose = identity_pose();
        let encoded = encode_pose_binary(&pose);
        assert_eq!(encoded[0], BINARY_MAGIC);
        assert_eq!(encoded[1], BINARY_VERSION);
        assert_eq!(encoded.len(), 40);
        let parsed = try_parse_pose(&encoded).expect("binary parse failed");
        assert_eq!(parsed, pose);
    }

    #[test]
    fn binary_round_trip_with_all_fields() {
        let mat = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let pose = HeadPose {
            position: [1.5, -0.3, 2.0],
            orientation: [0.0, 0.707, 0.0, 0.707],
            timestamp: 99_000,
            ipd: Some(0.064),
            proj_left: Some(mat),
            proj_right: Some(mat),
            eye_width: Some(1832),
            eye_height: Some(1920),
        };
        let encoded = encode_pose_binary(&pose);
        // 40 (header) + 4 (ipd) + 128 (proj) + 8 (eye dims) = 180
        assert_eq!(encoded.len(), 180);
        let parsed = try_parse_pose(&encoded).expect("binary parse failed");
        assert!((parsed.position[0] - 1.5).abs() < 1e-6);
        assert!((parsed.orientation[1] - 0.707).abs() < 1e-5);
        assert_eq!(parsed.timestamp, 99_000);
        assert!((parsed.ipd.unwrap() - 0.064).abs() < 1e-6);
        assert_eq!(parsed.proj_left, Some(mat));
        assert_eq!(parsed.proj_right, Some(mat));
        assert_eq!(parsed.eye_width, Some(1832));
        assert_eq!(parsed.eye_height, Some(1920));
    }

    #[test]
    fn binary_round_trip_only_ipd() {
        let mut pose = identity_pose();
        pose.ipd = Some(0.063);
        let encoded = encode_pose_binary(&pose);
        assert_eq!(encoded.len(), 44); // 40 + 4
        let parsed = try_parse_pose(&encoded).unwrap();
        assert!((parsed.ipd.unwrap() - 0.063).abs() < 1e-6);
        assert!(parsed.proj_left.is_none());
        assert!(parsed.eye_width.is_none());
    }

    #[test]
    fn binary_truncated_returns_none() {
        let pose = identity_pose();
        let encoded = encode_pose_binary(&pose);
        assert!(try_parse_pose(&encoded[..20]).is_none());
    }

    #[test]
    fn binary_wrong_magic_falls_through_to_json() {
        // A byte stream starting with anything other than 0x50 goes to JSON path.
        let raw = br#"{"position":[0,0,0],"orientation":[0,0,0,1],"timestamp":0}"#;
        assert_ne!(raw[0], BINARY_MAGIC);
        assert!(try_parse_pose(raw).is_some());
    }

    #[test]
    fn binary_size_minimal_is_40() {
        assert_eq!(encode_pose_binary(&identity_pose()).len(), 40);
    }
}
