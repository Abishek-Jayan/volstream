/// End-to-end pipeline integration test using the real test.nrrd volume.
///
/// Exercises: volume-loader → renderer → encoder in sequence.
/// GPU tests are skipped gracefully when no adapter is available (CI).
use std::path::PathBuf;

fn test_nrrd_path() -> PathBuf {
    // CARGO_MANIFEST_DIR for crates/server → go up two levels to workspace root.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../test.nrrd");
    p
}

// ── Volume loading ───────────────────────────────────────────────────────────

#[test]
fn load_test_nrrd() {
    let path = test_nrrd_path();
    if !path.exists() {
        eprintln!("SKIP: test.nrrd not found at {}", path.display());
        return;
    }

    let volume = volume_loader::load_volume(&path).expect("load_volume failed");

    // Header says: sizes: 818 773 548
    assert_eq!(
        volume.dims,
        [818, 773, 548],
        "unexpected dims: {:?}",
        volume.dims
    );
    assert_eq!(
        volume.data.len(),
        (818u64 * 773 * 548) as usize,
        "data length mismatch"
    );

    let min = volume.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = volume
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(min >= 0.0, "min below 0: {min}");
    assert!(max <= 1.0, "max above 1: {max}");

    println!(
        "Volume OK: {:?} voxels, spacing {:?}, value range [{min:.4}, {max:.4}]",
        volume.dims, volume.spacing
    );
}

// ── Renderer ─────────────────────────────────────────────────────────────────

#[test]
fn render_frame_from_test_nrrd() {
    let path = test_nrrd_path();
    if !path.exists() {
        eprintln!("SKIP: test.nrrd not found");
        return;
    }

    let volume = volume_loader::load_volume(&path).expect("load_volume failed");

    let mut rend = match renderer::Renderer::new(&volume) {
        Ok(r) => r,
        Err(e) => {
            println!("SKIP: no GPU adapter — {e}");
            return;
        }
    };

    let pose = renderer::HeadPose {
        position: glam::Vec3::new(0.0, 0.0, 2.0),
        orientation: glam::Quat::IDENTITY,
        proj_left: None,
        proj_right: None,
    };

    // Double-buffered readback: first call returns None (warmup).
    rend.render_frame(&pose)
        .expect("warmup render_frame failed");
    let frame = rend
        .render_frame(&pose)
        .expect("render_frame failed")
        .expect("expected Some on second call");

    assert_eq!(frame.width, rend.eye_width * 2);
    assert_eq!(frame.height, rend.eye_height);
    assert_eq!(frame.rgba.len(), (frame.width * frame.height * 4) as usize);

    let non_black = frame
        .rgba
        .chunks(4)
        .filter(|px| px[0] > 10 || px[1] > 10 || px[2] > 10)
        .count();
    println!(
        "Render OK: {non_black} non-black pixels / {} total",
        frame.width * frame.height
    );
    assert!(non_black > 0, "Expected visible volume pixels");
}

// ── Encoder ──────────────────────────────────────────────────────────────────

#[test]
fn encode_frame_from_test_nrrd() {
    let path = test_nrrd_path();
    if !path.exists() {
        eprintln!("SKIP: test.nrrd not found");
        return;
    }

    let volume = volume_loader::load_volume(&path).expect("load_volume failed");

    let mut rend = match renderer::Renderer::new(&volume) {
        Ok(r) => r,
        Err(e) => {
            println!("SKIP: no GPU adapter — {e}");
            return;
        }
    };

    let pose = renderer::HeadPose {
        position: glam::Vec3::new(0.0, 0.0, 2.0),
        orientation: glam::Quat::IDENTITY,
        proj_left: None,
        proj_right: None,
    };
    // Double-buffered readback: first call returns None (warmup).
    rend.render_frame(&pose)
        .expect("warmup render_frame failed");
    let frame = rend
        .render_frame(&pose)
        .expect("render_frame failed")
        .expect("expected Some on second call");

    let mut enc =
        encoder::default_encoder(frame.width, frame.height, 72, 8000).expect("encoder init failed");

    let nalus = enc
        .encode(&frame.rgba, frame.width, frame.height)
        .expect("encode failed");

    assert!(!nalus.is_empty(), "Encoder produced no bytes");

    // Annex-B start code: 00 00 00 01 or 00 00 01
    let has_start_code =
        nalus.windows(4).any(|w| w == [0, 0, 0, 1]) || nalus.windows(3).any(|w| w == [0, 0, 1]);
    assert!(has_start_code, "Output missing Annex-B H.264 start code");

    println!("Encode OK: {} bytes of H.264 NAL units", nalus.len());
}
