/// Loaded volumetric data, normalized to [0.0, 1.0].
#[derive(Debug, Clone)]
pub struct VolumeData {
    /// Dimensions in voxels: [x, y, z]
    pub dims: [u32; 3],
    /// Voxel spacing in mm: [x, y, z]
    pub spacing: [f32; 3],
    /// Voxel values normalized to [0.0, 1.0], stored in x-fastest (C) order
    pub data: Vec<f32>,
    /// Original data range before normalization (min, max)
    pub original_range: (f64, f64),
}

impl VolumeData {
    pub fn new(
        dims: [u32; 3],
        spacing: [f32; 3],
        data: Vec<f32>,
        original_range: (f64, f64),
    ) -> Self {
        assert_eq!(
            data.len(),
            (dims[0] * dims[1] * dims[2]) as usize,
            "data length must match dims"
        );
        Self {
            dims,
            spacing,
            data,
            original_range,
        }
    }

    pub fn voxel_count(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize
    }

    /// Index into data at voxel (x, y, z)
    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * self.dims[1] * self.dims[0] + y * self.dims[0] + x) as usize
    }
}
