use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::{VolumeData, VolumeError};

/// Load an NRRD file (.nrrd) from the given path.
///
/// Supported encodings: raw, gzip
/// Supported types: uint8, int8, uint16, int16, uint32, float, double
pub fn load_nrrd(path: &Path) -> Result<VolumeData, VolumeError> {
    let bytes = std::fs::read(path)?;
    parse_nrrd(&bytes)
}

fn parse_nrrd(bytes: &[u8]) -> Result<VolumeData, VolumeError> {
    // Find the blank line separating header from data.
    // NRRD spec: header ends at the first blank line ("\n\n" or "\r\n\r\n").
    let header_end = find_header_end(bytes).ok_or_else(|| {
        VolumeError::Parse("No blank line found separating NRRD header from data".into())
    })?;

    let header_bytes = &bytes[..header_end];
    let data_start = header_end + blank_line_len(bytes, header_end);

    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| VolumeError::Parse("NRRD header is not valid UTF-8".into()))?;

    // Validate magic line
    let first_line = header_str.lines().next().unwrap_or("");
    if !first_line.starts_with("NRRD") {
        return Err(VolumeError::Parse(format!(
            "Not an NRRD file (magic: {:?})",
            first_line
        )));
    }

    let fields = parse_header_fields(header_str);

    let dimension: usize = fields
        .get("dimension")
        .ok_or_else(|| VolumeError::Parse("Missing 'dimension' field".into()))?
        .parse()
        .map_err(|_| VolumeError::Parse("Invalid 'dimension' value".into()))?;

    if dimension != 3 {
        return Err(VolumeError::Unsupported(format!(
            "Only 3D NRRD files are supported, got dimension={}",
            dimension
        )));
    }

    let sizes = parse_sizes(fields.get("sizes").map(|s| s.as_str()).unwrap_or(""))?;
    if sizes.len() != 3 {
        return Err(VolumeError::Parse(format!(
            "Expected 3 sizes, got {}",
            sizes.len()
        )));
    }

    let spacing = parse_spacing(&fields)?;
    let data_type = fields.get("type").map(|s| s.as_str()).unwrap_or("float");
    let encoding = fields.get("encoding").map(|s| s.as_str()).unwrap_or("raw");
    let little_endian = fields.get("endian").map(|s| s == "little").unwrap_or(true);

    let raw_data = decode_data(&bytes[data_start..], encoding)?;

    let data = convert_to_f32(&raw_data, data_type, little_endian)?;

    let expected = sizes[0] * sizes[1] * sizes[2];
    if data.len() != expected {
        return Err(VolumeError::Parse(format!(
            "Data length mismatch: expected {} voxels, got {}",
            expected,
            data.len()
        )));
    }

    let (normalized, range) = normalize(&data);

    tracing::info!(
        "Loaded NRRD: {}x{}x{}, spacing={:?}, range=({:.2},{:.2})",
        sizes[0],
        sizes[1],
        sizes[2],
        spacing,
        range.0,
        range.1
    );

    Ok(VolumeData::new(
        [sizes[0] as u32, sizes[1] as u32, sizes[2] as u32],
        spacing,
        normalized,
        range,
    ))
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    // Look for \n\n (Unix) or \r\n\r\n (Windows)
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            return Some(i + 1); // position of the second \n
        }
        if i + 3 < bytes.len()
            && bytes[i] == b'\r'
            && bytes[i + 1] == b'\n'
            && bytes[i + 2] == b'\r'
            && bytes[i + 3] == b'\n'
        {
            return Some(i + 2); // position of the second \r\n start
        }
    }
    None
}

fn blank_line_len(bytes: &[u8], pos: usize) -> usize {
    // Skip past the blank line delimiter
    if pos + 3 < bytes.len()
        && bytes[pos] == b'\r'
        && bytes[pos + 1] == b'\n'
        && bytes[pos + 2] == b'\r'
        && bytes[pos + 3] == b'\n'
    {
        2 // already consumed first \r\n in find_header_end, skip second
    } else {
        1 // just the second \n
    }
}

fn parse_header_fields(header: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    for line in header.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() || line.starts_with("NRRD") {
            continue;
        }
        // Fields use ": " separator (with space), old-style uses ":"
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim().to_lowercase().replace(' ', "_");
            let value = line[colon_pos + 1..].trim().to_string();
            fields.insert(key, value);
        }
    }
    fields
}

fn parse_sizes(s: &str) -> Result<Vec<usize>, VolumeError> {
    s.split_whitespace()
        .map(|v| {
            v.parse::<usize>()
                .map_err(|_| VolumeError::Parse(format!("Invalid size value: {}", v)))
        })
        .collect()
}

fn parse_spacing(fields: &HashMap<String, String>) -> Result<[f32; 3], VolumeError> {
    // Prefer 'space directions' over 'spacings'
    if let Some(dirs) = fields.get("space_directions") {
        return parse_space_directions(dirs);
    }
    if let Some(spacings) = fields.get("spacings") {
        let vals: Vec<f32> = spacings
            .split_whitespace()
            .map(|v| v.parse::<f32>().unwrap_or(1.0))
            .collect();
        if vals.len() >= 3 {
            return Ok([vals[0], vals[1], vals[2]]);
        }
    }
    // Default: 1mm isotropic
    Ok([1.0, 1.0, 1.0])
}

/// Parse 'space directions: (sx,0,0) (0,sy,0) (0,0,sz)'
/// The spacing is the magnitude of each direction vector.
fn parse_space_directions(s: &str) -> Result<[f32; 3], VolumeError> {
    let mut spacings = [1.0f32; 3];
    let mut idx = 0;

    let mut chars = s.chars().peekable();
    while idx < 3 {
        // Skip until '('
        while chars.peek().is_some() && *chars.peek().unwrap() != '(' {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        chars.next(); // consume '('

        // Read until ')'
        let mut vec_str = String::new();
        for c in chars.by_ref() {
            if c == ')' {
                break;
            }
            vec_str.push(c);
        }

        // Handle "none" (some NRRD files use this for non-spatial dims)
        if vec_str.trim().eq_ignore_ascii_case("none") {
            idx += 1;
            continue;
        }

        let components: Vec<f32> = vec_str
            .split(',')
            .map(|v| v.trim().parse::<f32>().unwrap_or(0.0))
            .collect();

        if components.len() == 3 {
            let magnitude =
                (components[0].powi(2) + components[1].powi(2) + components[2].powi(2)).sqrt();
            spacings[idx] = if magnitude > 0.0 { magnitude } else { 1.0 };
        }
        idx += 1;
    }

    Ok(spacings)
}

fn decode_data(data: &[u8], encoding: &str) -> Result<Vec<u8>, VolumeError> {
    match encoding.to_lowercase().as_str() {
        "raw" => Ok(data.to_vec()),
        "gzip" | "gz" => {
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            Ok(out)
        }
        other => Err(VolumeError::Unsupported(format!(
            "NRRD encoding '{}' not supported",
            other
        ))),
    }
}

fn convert_to_f32(
    raw: &[u8],
    data_type: &str,
    little_endian: bool,
) -> Result<Vec<f32>, VolumeError> {
    match data_type.to_lowercase().replace(" ", "").as_str() {
        "uint8" | "uchar" | "unsignedchar" => Ok(raw.iter().map(|&v| v as f32).collect()),
        "int8" | "char" | "signedchar" => Ok(raw.iter().map(|&v| v as i8 as f32).collect()),
        "uint16" | "ushort" => {
            read_u16(raw, little_endian).map(|v| v.into_iter().map(|x| x as f32).collect())
        }
        "int16" | "short" => {
            read_u16(raw, little_endian).map(|v| v.into_iter().map(|x| x as i16 as f32).collect())
        }
        "uint32" | "uint" => {
            read_u32(raw, little_endian).map(|v| v.into_iter().map(|x| x as f32).collect())
        }
        "int32" | "int" => {
            read_u32(raw, little_endian).map(|v| v.into_iter().map(|x| x as i32 as f32).collect())
        }
        "float" => read_f32(raw, little_endian),
        "double" => read_f64(raw, little_endian).map(|v| v.into_iter().map(|x| x as f32).collect()),
        other => Err(VolumeError::Unsupported(format!(
            "NRRD data type '{}' not supported",
            other
        ))),
    }
}

fn read_u16(raw: &[u8], little_endian: bool) -> Result<Vec<u16>, VolumeError> {
    if !raw.len().is_multiple_of(2) {
        return Err(VolumeError::Parse("uint16 data has odd byte count".into()));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| {
            let arr = [c[0], c[1]];
            if little_endian {
                u16::from_le_bytes(arr)
            } else {
                u16::from_be_bytes(arr)
            }
        })
        .collect())
}

fn read_u32(raw: &[u8], little_endian: bool) -> Result<Vec<u32>, VolumeError> {
    if !raw.len().is_multiple_of(4) {
        return Err(VolumeError::Parse(
            "uint32 data has byte count not divisible by 4".into(),
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| {
            let arr = [c[0], c[1], c[2], c[3]];
            if little_endian {
                u32::from_le_bytes(arr)
            } else {
                u32::from_be_bytes(arr)
            }
        })
        .collect())
}

fn read_f32(raw: &[u8], little_endian: bool) -> Result<Vec<f32>, VolumeError> {
    if !raw.len().is_multiple_of(4) {
        return Err(VolumeError::Parse(
            "float32 data has byte count not divisible by 4".into(),
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| {
            let arr = [c[0], c[1], c[2], c[3]];
            if little_endian {
                f32::from_le_bytes(arr)
            } else {
                f32::from_be_bytes(arr)
            }
        })
        .collect())
}

fn read_f64(raw: &[u8], little_endian: bool) -> Result<Vec<f64>, VolumeError> {
    if !raw.len().is_multiple_of(8) {
        return Err(VolumeError::Parse(
            "float64 data has byte count not divisible by 8".into(),
        ));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|c| {
            let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            if little_endian {
                f64::from_le_bytes(arr)
            } else {
                f64::from_be_bytes(arr)
            }
        })
        .collect())
}

/// Normalize data to [0.0, 1.0]. Returns (normalized, (min, max)).
pub fn normalize(data: &[f32]) -> (Vec<f32>, (f64, f64)) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let range = max - min;
    let normalized = if range < f64::EPSILON {
        vec![0.0f32; data.len()]
    } else {
        data.iter()
            .map(|&v| ((v as f64 - min) / range) as f32)
            .collect()
    };
    (normalized, (min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nrrd_bytes(
        type_str: &str,
        encoding: &str,
        endian: &str,
        sizes: [usize; 3],
        data: &[u8],
    ) -> Vec<u8> {
        let header = format!(
            "NRRD0004\ntype: {}\ndimension: 3\nsizes: {} {} {}\nencoding: {}\nendian: {}\n\n",
            type_str, sizes[0], sizes[1], sizes[2], encoding, endian
        );
        let mut out = header.into_bytes();
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parse_uint8_raw() {
        // 2x2x2 volume, all 128
        let data: Vec<u8> = vec![128u8; 8];
        let bytes = make_nrrd_bytes("uint8", "raw", "little", [2, 2, 2], &data);
        let vol = parse_nrrd(&bytes).unwrap();
        assert_eq!(vol.dims, [2, 2, 2]);
        // All same value → normalized to 0
        assert!(vol.data.iter().all(|&v| v == 0.0 || v == 1.0 || v == 0.5));
    }

    #[test]
    fn parse_uint16_little_endian() {
        // 2x2x1 volume: values 0, 100, 200, 300
        let vals: &[u16] = &[0, 100, 200, 300];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes = make_nrrd_bytes("uint16", "raw", "little", [2, 2, 1], &data);
        let vol = parse_nrrd(&bytes).unwrap();
        assert_eq!(vol.dims, [2, 2, 1]);
        assert_eq!(vol.data.len(), 4);
        assert!((vol.data[0] - 0.0).abs() < 1e-5);
        assert!((vol.data[3] - 1.0).abs() < 1e-5);
        assert!((vol.original_range.0 - 0.0).abs() < 1e-3);
        assert!((vol.original_range.1 - 300.0).abs() < 1e-3);
    }

    #[test]
    fn parse_space_directions_extracts_spacing() {
        let header = "NRRD0004\ntype: uint8\ndimension: 3\nsizes: 2 2 2\n\
                      space directions: (0.5,0,0) (0,0.5,0) (0,0,1.2)\n\
                      encoding: raw\n\n";
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        let vol = parse_nrrd(&bytes).unwrap();
        assert!((vol.spacing[0] - 0.5).abs() < 1e-5);
        assert!((vol.spacing[1] - 0.5).abs() < 1e-5);
        assert!((vol.spacing[2] - 1.2).abs() < 1e-4);
    }

    #[test]
    fn parse_gzip_encoded() {
        use flate2::write::GzEncoder;
        use std::io::Write;

        let raw: Vec<u8> = (0u8..8).collect();
        let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&raw).unwrap();
        let compressed = encoder.finish().unwrap();

        let bytes = make_nrrd_bytes("uint8", "gzip", "little", [2, 2, 2], &compressed);
        let vol = parse_nrrd(&bytes).unwrap();
        assert_eq!(vol.dims, [2, 2, 2]);
        assert_eq!(vol.data.len(), 8);
        // First voxel should be 0/7 ≈ 0.0, last should be 7/7 = 1.0
        assert!((vol.data[0] - 0.0).abs() < 1e-5);
        assert!((vol.data[7] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_wrong_magic() {
        let bytes = b"NOTANRRD\ntype: uint8\n\n\x00".to_vec();
        assert!(parse_nrrd(&bytes).is_err());
    }

    #[test]
    fn rejects_unsupported_encoding() {
        let bytes =
            b"NRRD0004\ntype: uint8\ndimension: 3\nsizes: 1 1 1\nencoding: bzip2\n\n\x00".to_vec();
        assert!(parse_nrrd(&bytes).is_err());
    }

    #[test]
    fn rejects_non_3d() {
        let bytes = b"NRRD0004\ntype: uint8\ndimension: 2\nsizes: 4 4\nencoding: raw\n\n".to_vec();
        // Data would be missing but the dimension check fires first
        let result = parse_nrrd(&bytes);
        assert!(matches!(result, Err(VolumeError::Unsupported(_))));
    }

    #[test]
    fn normalize_maps_range_to_zero_one() {
        let data = vec![10.0f32, 20.0, 30.0];
        let (norm, range) = normalize(&data);
        assert!((norm[0] - 0.0).abs() < 1e-6);
        assert!((norm[1] - 0.5).abs() < 1e-6);
        assert!((norm[2] - 1.0).abs() < 1e-6);
        assert!((range.0 - 10.0).abs() < 1e-4);
        assert!((range.1 - 30.0).abs() < 1e-4);
    }
}
