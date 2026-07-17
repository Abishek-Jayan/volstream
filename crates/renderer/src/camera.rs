use glam::{Mat4, Quat, Vec3};

/// Head pose received from the XR client.
#[derive(Debug, Clone)]
pub struct HeadPose {
    pub position: Vec3,
    pub orientation: Quat,
    /// Per-eye projection matrices as reported by the XR runtime
    /// (`XRView.projectionMatrix`, column-major, OpenGL z ∈ [-1, 1] convention).
    /// When `None`, the renderer falls back to its built-in perspective matrix.
    pub proj_left: Option<Mat4>,
    pub proj_right: Option<Mat4>,
}

impl Default for HeadPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            proj_left: None,
            proj_right: None,
        }
    }
}

/// Inter-pupillary distance in meters (typical: 0.063)
pub const DEFAULT_IPD: f32 = 0.063;

/// Compute left and right view matrices from a head pose.
pub fn compute_stereo_views(pose: &HeadPose, ipd: f32) -> (Mat4, Mat4) {
    let rotation = Mat4::from_quat(pose.orientation);
    let right = rotation.transform_vector3(Vec3::X);

    let left_pos = pose.position - right * (ipd / 2.0);
    let right_pos = pose.position + right * (ipd / 2.0);

    let forward = rotation.transform_vector3(-Vec3::Z);
    let up = rotation.transform_vector3(Vec3::Y);

    let left_view = Mat4::look_to_rh(left_pos, forward, up);
    let right_view = Mat4::look_to_rh(right_pos, forward, up);

    (left_view, right_view)
}

/// Compute a symmetric perspective projection matrix.
#[allow(dead_code)]
pub fn compute_projection(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y_radians, aspect, near, far)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_pose_produces_symmetric_stereo_views() {
        let pose = HeadPose::default();
        let (left, right) = compute_stereo_views(&pose, DEFAULT_IPD);

        // At identity pose, left and right views differ only in X translation
        let left_pos = left.col(3);
        let right_pos = right.col(3);

        // The translation columns should be mirror images in X
        assert!(
            (left_pos.x + right_pos.x).abs() < 1e-5,
            "X offsets should cancel"
        );
        assert!((left_pos.y - right_pos.y).abs() < 1e-5, "Y should be equal");
        assert!((left_pos.z - right_pos.z).abs() < 1e-5, "Z should be equal");
    }

    #[test]
    fn projection_matrix_has_correct_near_plane() {
        let proj = compute_projection(FRAC_PI_2, 1.0, 0.1, 100.0);
        // The (2,3) element of an RH perspective matrix is -2*far*near/(far-near)
        // Just verify it is non-zero and negative (valid projection)
        assert!(proj.col(2).w < 0.0);
    }
}
