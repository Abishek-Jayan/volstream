mod nrrd;
mod volume;
mod zarr;

pub use nrrd::load_nrrd;
pub use volume::VolumeData;
pub use zarr::{load_zarr, load_zarr_with_limit};

use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum VolumeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported format: {0}")]
    Unsupported(String),
    #[error("Zarr error: {0}")]
    Zarr(String),
}

/// Load a volume from any supported file format, detected by extension.
/// - `.nrrd` → NRRD
/// - Directory (or `.zarr`) → OME-Zarr
pub fn load_volume(path: &Path) -> Result<VolumeData, VolumeError> {
    if path.is_dir() {
        return load_zarr(path);
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("nrrd") => load_nrrd(path),
        Some("zarr") => load_zarr(path),
        other => Err(VolumeError::Unsupported(format!(
            "Unknown file extension: {:?}. Supported: .nrrd, .zarr directory",
            other
        ))),
    }
}
