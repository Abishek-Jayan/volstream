use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use zarrs::array_subset::ArraySubset;

use crate::{nrrd::normalize, VolumeData, VolumeError};

// ---------------------------------------------------------------------------
// OME-NGFF metadata types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OmeZattrs {
    multiscales: Vec<Multiscale>,
}

#[derive(Debug, Deserialize)]
struct Multiscale {
    #[serde(default)]
    axes: Vec<Axis>,
    datasets: Vec<Dataset>,
}

#[derive(Debug, Deserialize)]
struct Axis {
    name: String,
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    axis_type: String,
    #[serde(default)]
    unit: String,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    path: String,
    #[serde(rename = "coordinateTransformations", default)]
    coordinate_transformations: Vec<CoordinateTransform>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum CoordinateTransform {
    #[serde(rename = "scale")]
    Scale { scale: Vec<f64> },
    #[serde(rename = "translation")]
    #[allow(dead_code)]
    Translation { translation: Vec<f64> },
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Load an OME-Zarr (OME-NGFF) dataset from the given filesystem path.
///
/// Selects the finest resolution level that fits within `max_voxels`.
/// Defaults to 256³ (≈16M voxels). For a full-quality load pass `usize::MAX`.
///
/// Handles N-dimensional arrays (t, c, z, y, x) — selects the first index
/// along any non-spatial (non-z/y/x) axes.
pub fn load_zarr(path: &Path) -> Result<VolumeData, VolumeError> {
    load_zarr_with_limit(path, 256 * 256 * 256 * 4) // ~4× 256³ = up to ~512³ if data fits
}

pub fn load_zarr_with_limit(path: &Path, max_voxels: usize) -> Result<VolumeData, VolumeError> {
    // bioformats2raw layout 3 stores multiscales one level down under the series directory.
    // Detect this and redirect transparently so the rest of the loader sees a normal OME-Zarr.
    if let Some(series_path) = detect_bioformats2raw_series(path)? {
        tracing::info!(
            "bioformats2raw layout detected, loading series from {}",
            series_path.display()
        );
        return load_zarr_with_limit(&series_path, max_voxels);
    }

    // bioformats2raw emits `"fill_value": null` in .zarray files, which zarrs rejects
    // for numeric types. Per the Zarr v2 spec null means "uninitialized" (= 0 for numbers),
    // so patching null → 0 in place is semantically correct.
    fix_null_fill_values(path);

    let ome = read_ome_metadata(path)?;
    let multiscale = ome
        .multiscales
        .into_iter()
        .next()
        .ok_or_else(|| VolumeError::Parse("No 'multiscales' entry in .zattrs".into()))?;

    let (dataset, level_idx) = select_level(&multiscale.datasets, max_voxels, path)?;
    tracing::info!(
        "Selected OME-Zarr level {} (path: {})",
        level_idx,
        dataset.path
    );

    let spacing = extract_spacing(&multiscale.axes, &dataset.coordinate_transformations);
    tracing::info!("Voxel spacing: {:?}", spacing);

    let store = Arc::new(
        zarrs::filesystem::FilesystemStore::new(path)
            .map_err(|e| VolumeError::Zarr(e.to_string()))?,
    );

    let array_path = format!("/{}", dataset.path);
    let array = zarrs::array::Array::open(store, &array_path)
        .map_err(|e| VolumeError::Zarr(format!("Failed to open array at {}: {}", array_path, e)))?;

    let shape = array.shape().to_vec(); // &[u64] → Vec<u64>
    tracing::info!("Array shape: {:?}", shape);

    let (spatial_shape, spatial_offsets) = extract_spatial_dims(&shape, &multiscale.axes)?;
    let [nx, ny, nz] = spatial_shape;

    let expected = nx * ny * nz;

    // Build subset ranges: full range for spatial dims, first index for all others.
    let spatial_set: std::collections::HashSet<usize> = spatial_offsets.iter().copied().collect();
    let ranges: Vec<std::ops::Range<u64>> = (0..shape.len())
        .map(|i| {
            if spatial_set.contains(&i) {
                0..shape[i]
            } else {
                0..1
            }
        })
        .collect();
    let subset = ArraySubset::new_with_ranges(&ranges);
    let elements = retrieve_as_f32(&array, &subset)?;
    if elements.len() != expected {
        return Err(VolumeError::Zarr(format!(
            "Retrieved {} elements but expected {} ({}x{}x{})",
            elements.len(),
            expected,
            nx,
            ny,
            nz
        )));
    }
    let (normalized, range) = normalize(&elements);

    tracing::info!(
        "Loaded OME-Zarr: {}x{}x{}, spacing={:?}, range=({:.2},{:.2})",
        nx,
        ny,
        nz,
        spacing,
        range.0,
        range.1
    );

    Ok(VolumeData::new(
        [nx as u32, ny as u32, nz as u32],
        spacing,
        normalized,
        range,
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Retrieve a zarr array subset as `Vec<f32>`, converting from the array's native dtype.
fn retrieve_as_f32(
    array: &zarrs::array::Array<zarrs::filesystem::FilesystemStore>,
    subset: &ArraySubset,
) -> Result<Vec<f32>, VolumeError> {
    use zarrs::array::DataType;
    macro_rules! retrieve {
        ($t:ty) => {
            array
                .retrieve_array_subset_elements::<$t>(subset)
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .map_err(|e| VolumeError::Zarr(format!("Failed to retrieve array data: {}", e)))
        };
    }

    match array.data_type() {
        DataType::Float32 => array
            .retrieve_array_subset_elements::<f32>(subset)
            .map_err(|e| VolumeError::Zarr(format!("Failed to retrieve array data: {}", e))),
        DataType::Float64 => retrieve!(f64),
        DataType::UInt8 => retrieve!(u8),
        DataType::Int8 => retrieve!(i8),
        DataType::UInt16 => retrieve!(u16),
        DataType::Int16 => retrieve!(i16),
        DataType::UInt32 => retrieve!(u32),
        DataType::Int32 => retrieve!(i32),
        DataType::UInt64 => retrieve!(u64),
        DataType::Int64 => retrieve!(i64),
        other => Err(VolumeError::Unsupported(format!(
            "Unsupported zarr data type: {:?}",
            other
        ))),
    }
}

/// Walk `path` recursively and rewrite any `.zarray` file that has `"fill_value": null`
/// to `"fill_value": 0`. The Zarr v2 spec treats null as "no fill value defined"
/// (effectively 0 for numeric types), but zarrs rejects it. Errors are logged and skipped.
fn fix_null_fill_values(path: &Path) {
    let walker = match std::fs::read_dir(path) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in walker.flatten() {
        let p = entry.path();
        if p.is_dir() {
            fix_null_fill_values(&p);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(".zarray") {
            patch_zarray_fill_value(&p);
        }
    }
}

fn patch_zarray_fill_value(path: &Path) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return;
    };

    let fill = val.get("fill_value");
    if !matches!(fill, Some(serde_json::Value::Null)) {
        return; // already fine
    }

    val["fill_value"] = serde_json::Value::Number(0.into());
    if let Ok(patched) = serde_json::to_vec_pretty(&val) {
        if let Err(e) = std::fs::write(path, patched) {
            tracing::warn!("Could not patch fill_value in {}: {}", path.display(), e);
        } else {
            tracing::debug!("Patched null fill_value in {}", path.display());
        }
    }
}

/// Detect bioformats2raw layout 3 and return the path to the first series directory.
///
/// Layout 3 looks like:
///   <root>/.zattrs          → {"bioformats2raw.layout": 3}
///   <root>/OME/.zattrs      → {"series": ["0", ...]}
///   <root>/0/.zattrs        → {"multiscales": [...]}   ← actual OME-NGFF data
///
/// Returns `Some(series_path)` when detected, `None` for normal OME-Zarr.
fn detect_bioformats2raw_series(path: &Path) -> Result<Option<PathBuf>, VolumeError> {
    let zattrs_path = path.join(".zattrs");
    if !zattrs_path.exists() {
        return Ok(None);
    }

    let bytes = std::fs::read(&zattrs_path)?;
    let root: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| VolumeError::Parse(format!("Failed to parse .zattrs: {}", e)))?;

    if root.get("bioformats2raw.layout").is_none() {
        return Ok(None);
    }

    // Read series list from OME/.zattrs
    let ome_zattrs = path.join("OME").join(".zattrs");
    if ome_zattrs.exists() {
        let bytes = std::fs::read(&ome_zattrs)?;
        let ome: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| VolumeError::Parse(format!("Failed to parse OME/.zattrs: {}", e)))?;
        if let Some(first) = ome
            .get("series")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        {
            return Ok(Some(path.join(first)));
        }
    }

    // Fallback: try series "0" directly
    let series_0 = path.join("0");
    if series_0.join(".zattrs").exists() {
        return Ok(Some(series_0));
    }

    Ok(None)
}

fn read_ome_metadata(path: &Path) -> Result<OmeZattrs, VolumeError> {
    // Try OME-NGFF v0.4 location: .zattrs at the root
    let zattrs_path = path.join(".zattrs");
    if zattrs_path.exists() {
        let bytes = std::fs::read(&zattrs_path)?;
        return serde_json::from_slice(&bytes)
            .map_err(|e| VolumeError::Parse(format!("Failed to parse .zattrs: {}", e)));
    }

    // Try Zarr v3 location: zarr.json at the root
    let zarr_json_path = path.join("zarr.json");
    if zarr_json_path.exists() {
        let bytes = std::fs::read(&zarr_json_path)?;
        // zarr.json has attributes nested under "attributes"
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| VolumeError::Parse(format!("Failed to parse zarr.json: {}", e)))?;
        let attrs = root.get("attributes").cloned().unwrap_or(root);
        return serde_json::from_value(attrs).map_err(|e| {
            VolumeError::Parse(format!(
                "Failed to parse OME metadata from zarr.json: {}",
                e
            ))
        });
    }

    Err(VolumeError::Parse(
        "No .zattrs or zarr.json found — is this an OME-Zarr directory?".into(),
    ))
}

/// Select the finest resolution level that fits within `max_voxels`.
/// Falls back to the coarsest level if everything is too large.
fn select_level<'a>(
    datasets: &'a [Dataset],
    max_voxels: usize,
    base_path: &Path,
) -> Result<(&'a Dataset, usize), VolumeError> {
    if datasets.is_empty() {
        return Err(VolumeError::Parse("No datasets in multiscales".into()));
    }

    // Try each level from finest (index 0) to coarsest
    for (i, dataset) in datasets.iter().enumerate() {
        let array_path = base_path.join(&dataset.path);
        if let Ok(voxels) = estimate_voxels(&array_path) {
            if voxels <= max_voxels {
                return Ok((dataset, i));
            }
        }
    }

    // Nothing fits — use coarsest
    tracing::warn!(
        "No level fits within {} voxels, using coarsest level ({})",
        max_voxels,
        datasets.last().unwrap().path
    );
    Ok((datasets.last().unwrap(), datasets.len() - 1))
}

fn estimate_voxels(array_path: &Path) -> Result<usize, VolumeError> {
    // Read .zarray or zarr.json to get shape without loading data
    let zarray_path = array_path.join(".zarray");
    if zarray_path.exists() {
        let bytes = std::fs::read(&zarray_path)?;
        let meta: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| VolumeError::Parse(e.to_string()))?;
        if let Some(shape) = meta.get("shape").and_then(|s| s.as_array()) {
            let voxels: usize = shape
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .product();
            return Ok(voxels);
        }
    }
    // Zarr v3: zarr.json
    let zarr_json = array_path.join("zarr.json");
    if zarr_json.exists() {
        let bytes = std::fs::read(&zarr_json)?;
        let meta: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| VolumeError::Parse(e.to_string()))?;
        if let Some(shape) = meta.get("shape").and_then(|s| s.as_array()) {
            let voxels: usize = shape
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .product();
            return Ok(voxels);
        }
    }
    Err(VolumeError::Parse(
        "Could not read array shape metadata".into(),
    ))
}

/// Extract spatial (z, y, x) dimensions and their indices in the full shape array.
/// Returns ([nx, ny, nz], [x_idx, y_idx, z_idx]) in the full shape.
fn extract_spatial_dims(
    shape: &[u64],
    axes: &[Axis],
) -> Result<([usize; 3], [usize; 3]), VolumeError> {
    if axes.is_empty() {
        // No axes metadata — assume last 3 dims are z, y, x
        if shape.len() < 3 {
            return Err(VolumeError::Parse(format!(
                "Array has {} dims, need at least 3",
                shape.len()
            )));
        }
        let n = shape.len();
        return Ok((
            [
                shape[n - 1] as usize,
                shape[n - 2] as usize,
                shape[n - 3] as usize,
            ],
            [n - 1, n - 2, n - 3],
        ));
    }

    // Find the x, y, z axes by name (case insensitive)
    let find = |name: &str| -> Option<usize> {
        axes.iter().position(|a| a.name.eq_ignore_ascii_case(name))
    };

    let xi =
        find("x").ok_or_else(|| VolumeError::Parse("No 'x' axis in OME-NGFF metadata".into()))?;
    let yi =
        find("y").ok_or_else(|| VolumeError::Parse("No 'y' axis in OME-NGFF metadata".into()))?;
    let zi =
        find("z").ok_or_else(|| VolumeError::Parse("No 'z' axis in OME-NGFF metadata".into()))?;

    if xi >= shape.len() || yi >= shape.len() || zi >= shape.len() {
        return Err(VolumeError::Parse(
            "Axis indices out of bounds for array shape".into(),
        ));
    }

    Ok((
        [shape[xi] as usize, shape[yi] as usize, shape[zi] as usize],
        [xi, yi, zi],
    ))
}

/// Extract [x_spacing, y_spacing, z_spacing] in mm from OME-NGFF coordinate transforms.
fn extract_spacing(axes: &[Axis], transforms: &[CoordinateTransform]) -> [f32; 3] {
    // Find the scale transform
    let scale_vec = transforms.iter().find_map(|t| match t {
        CoordinateTransform::Scale { scale } => Some(scale.clone()),
        _ => None,
    });

    let Some(scale) = scale_vec else {
        return [1.0, 1.0, 1.0];
    };

    let get_scale = |name: &str| -> f32 {
        axes.iter()
            .position(|a| a.name.eq_ignore_ascii_case(name))
            .and_then(|i| scale.get(i).copied())
            .unwrap_or(1.0) as f32
    };

    // Convert to mm if unit is given
    let to_mm = |v: f32, unit: &str| -> f32 {
        match unit {
            "micrometer" | "micron" | "µm" | "um" => v / 1000.0,
            "nanometer" | "nm" => v / 1_000_000.0,
            "centimeter" | "cm" => v * 10.0,
            "meter" | "m" => v * 1000.0,
            _ => v, // assume mm
        }
    };

    let get_unit = |name: &str| -> &str {
        axes.iter()
            .find(|a| a.name.eq_ignore_ascii_case(name))
            .map(|a| a.unit.as_str())
            .unwrap_or("")
    };

    [
        to_mm(get_scale("x"), get_unit("x")),
        to_mm(get_scale("y"), get_unit("y")),
        to_mm(get_scale("z"), get_unit("z")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_spacing_from_axes_and_transforms() {
        let axes = vec![
            Axis {
                name: "z".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
            Axis {
                name: "y".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
            Axis {
                name: "x".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
        ];
        let transforms = vec![CoordinateTransform::Scale {
            scale: vec![0.5, 0.25, 0.25],
        }];
        let spacing = extract_spacing(&axes, &transforms);
        // z: index 0, scale 0.5 µm → 0.5/1000 mm = 0.0005
        assert!(
            (spacing[2] - 0.0005).abs() < 1e-7,
            "z spacing: {}",
            spacing[2]
        );
        // x: index 2, scale 0.25 µm → 0.00025
        assert!(
            (spacing[0] - 0.00025).abs() < 1e-8,
            "x spacing: {}",
            spacing[0]
        );
    }

    #[test]
    fn extract_spatial_dims_with_axes_metadata() {
        let axes = vec![
            Axis {
                name: "t".into(),
                axis_type: "time".into(),
                unit: String::new(),
            },
            Axis {
                name: "c".into(),
                axis_type: "channel".into(),
                unit: String::new(),
            },
            Axis {
                name: "z".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
            Axis {
                name: "y".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
            Axis {
                name: "x".into(),
                axis_type: "space".into(),
                unit: "micrometer".into(),
            },
        ];
        let shape = vec![2u64, 3, 64, 128, 256]; // t=2, c=3, z=64, y=128, x=256
        let ([nx, ny, nz], [xi, yi, zi]) = extract_spatial_dims(&shape, &axes).unwrap();
        assert_eq!(nx, 256); // x
        assert_eq!(ny, 128); // y
        assert_eq!(nz, 64); // z
        assert_eq!(xi, 4); // index of x in shape
        assert_eq!(yi, 3);
        assert_eq!(zi, 2);
    }

    #[test]
    fn extract_spatial_dims_fallback_no_axes() {
        // When no axes metadata, assume last 3 dims are x, y, z
        let shape = vec![2u64, 64, 128, 256];
        let ([nx, ny, nz], _) = extract_spatial_dims(&shape, &[]).unwrap();
        assert_eq!(nx, 256);
        assert_eq!(ny, 128);
        assert_eq!(nz, 64);
    }
}
