use std::sync::mpsc::{self, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc as tokio_mpsc, watch};
use transport::HeadPose as TransportPose;
use volume_loader::VolumeData;

use renderer::{HeadPose as RendererPose, Renderer, DEFAULT_EYE_HEIGHT, DEFAULT_EYE_WIDTH};

// ── Pose prediction ──────────────────────────────────────────────────────────

/// Extrapolates head pose forward using constant-velocity kinematics derived
/// from the two most recently received poses.
///
/// Position is extrapolated linearly; orientation uses the angle-axis scaling
/// of the delta quaternion between the two poses (constant angular velocity).
struct PosePredictor {
    older: Option<(TransportPose, Instant)>,
    newer: Option<(TransportPose, Instant)>,
}

impl PosePredictor {
    fn new() -> Self {
        Self {
            older: None,
            newer: None,
        }
    }

    /// Record a newly received pose (call only when the timestamp advances).
    fn update(&mut self, pose: &TransportPose, received_at: Instant) {
        self.older = self.newer.take();
        self.newer = Some((pose.clone(), received_at));
    }

    /// Extrapolate the pose forward by `horizon_secs`.
    ///
    /// Returns `None` when fewer than two poses are available or the time gap
    /// between them is out of the plausible range.
    fn predict(&self, horizon_secs: f32) -> Option<RendererPose> {
        let (prev, prev_t) = self.older.as_ref()?;
        let (curr, curr_t) = self.newer.as_ref()?;

        let dt = curr_t.duration_since(*prev_t).as_secs_f32();
        // Reject duplicate or stale readings: < 1 ms apart or > 500 ms apart.
        if !(0.001..=0.5).contains(&dt) {
            return None;
        }

        // Cap extrapolation factor at 1.5× to prevent overshoot.
        // 3.0 was too aggressive: at 72fps (dt≈14ms, horizon=20ms) it produced
        // t≈1.4, but at the previous 40ms horizon it reached t≈2.9, making the
        // volume swing ~3× farther than the actual head movement.
        let t = (horizon_secs / dt).min(1.5);

        // ── Position (linear velocity) ─────────────────────────────────────
        let curr_pos = glam::Vec3::from(curr.position);
        let prev_pos = glam::Vec3::from(prev.position);
        let vel = (curr_pos - prev_pos) / dt;
        let pred_pos = curr_pos + vel * horizon_secs;

        // ── Orientation (constant angular velocity) ────────────────────────
        // delta_q: incremental rotation from prev to curr.
        // Extrapolate by applying it t× (scale the angle-axis).
        let curr_q = glam::Quat::from_xyzw(
            curr.orientation[0],
            curr.orientation[1],
            curr.orientation[2],
            curr.orientation[3],
        );
        let prev_q = glam::Quat::from_xyzw(
            prev.orientation[0],
            prev.orientation[1],
            prev.orientation[2],
            prev.orientation[3],
        );

        let delta_q = prev_q.inverse() * curr_q;
        let pred_q = if delta_q.w.abs() > 0.9999 {
            // Near-zero rotation delta — extrapolation not meaningful.
            curr_q
        } else {
            let (axis, angle) = delta_q.to_axis_angle();
            let extra = glam::Quat::from_axis_angle(axis, angle * t);
            (curr_q * extra).normalize()
        };

        Some(RendererPose {
            position: pred_pos,
            orientation: pred_q,
            proj_left: curr.proj_left.map(|m| glam::Mat4::from_cols_array(&m)),
            proj_right: curr.proj_right.map(|m| glam::Mat4::from_cols_array(&m)),
        })
    }
}

// ── Adaptive resolution ───────────────────────────────────────────────────────

/// Compute the next render scale based on the drop rate observed over the last
/// FPS reporting window.
///
/// Rules (applied in order):
/// - `drop_rate > DROP_THRESHOLD` → scale down by `STEP_DOWN`, floor `SCALE_MIN`.
/// - `drop_rate == 0.0`           → scale up by `STEP_UP`, cap `SCALE_MAX`.
/// - otherwise                    → no change.
fn adapt_scale(current: f32, drop_rate: f32) -> f32 {
    const STEP_DOWN: f32 = 0.10;
    const STEP_UP: f32 = 0.05;
    const SCALE_MIN: f32 = 0.50;
    const SCALE_MAX: f32 = 1.00;
    const DROP_THRESHOLD: f32 = 0.10; // >10 % drops triggers a scale-down

    if drop_rate > DROP_THRESHOLD {
        (current - STEP_DOWN).max(SCALE_MIN)
    } else if drop_rate == 0.0 {
        (current + STEP_UP).min(SCALE_MAX)
    } else {
        current
    }
}

// ── Pipeline types ────────────────────────────────────────────────────────────

/// Raw RGBA8 frame from the GPU render thread, consumed by the encode thread.
struct RawFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    /// Quaternion [x, y, z, w] of the pose that was used to render this frame.
    /// Forwarded to the client as an ATW pose-tag after encoding.
    rendered_orientation: [f32; 4],
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Convert a wire-format head pose to the renderer's glam-typed pose.
pub fn to_renderer_pose(p: &TransportPose) -> RendererPose {
    RendererPose {
        position: glam::Vec3::new(p.position[0], p.position[1], p.position[2]),
        orientation: glam::Quat::from_xyzw(
            p.orientation[0],
            p.orientation[1],
            p.orientation[2],
            p.orientation[3],
        ),
        proj_left: p.proj_left.map(|m| glam::Mat4::from_cols_array(&m)),
        proj_right: p.proj_right.map(|m| glam::Mat4::from_cols_array(&m)),
    }
}

/// Scale and clamp per-eye dimensions for the given `render_scale` factor.
///
/// - Width rounded down to nearest multiple of 8 (GPU YUV shader alignment).
/// - Height rounded down to even (H.264 YUV420 requirement).
/// - Both clamped to a minimum of 64 (encoder requirement).
fn scale_dims(w: u32, h: u32, scale: f32) -> (u32, u32) {
    let sw = ((w as f32 * scale).round() as u32).max(64) & !7;
    let sh = ((h as f32 * scale).round() as u32).max(64) & !1;
    (sw, sh)
}

/// Blocking entry point — called from `tokio::task::spawn_blocking`.
///
/// Internally spawns a dedicated **render thread** (GPU work + pose prediction)
/// and runs the **encode loop** on the current thread in parallel.
///
/// ```text
///  Render thread  ──[RawFrame, cap=1]──►  Encode loop (this thread)
///  (GPU render, paced to fps)              (RGBA→YUV, H.264 encode)
///                                                │
///                                          pose_tag_tx  ──►  WebRTC → client ATW
///                                                │
///                                           video_tx (cap=2) ──►  WebRTC RTP
/// ```
#[allow(clippy::too_many_arguments)]
pub fn run(
    volume: Arc<VolumeData>,
    fps: u32,
    bitrate_kbps: u32,
    ipd: f32,
    viewing_distance: f32,
    render_scale: f32,
    sample_density: f32,
    prediction_horizon_secs: f32,
    pose_rx: watch::Receiver<Option<TransportPose>>,
    video_tx: tokio_mpsc::Sender<Vec<u8>>,
    pose_tag_tx: tokio_mpsc::Sender<[f32; 4]>,
) {
    let (raw_tx, raw_rx) = mpsc::sync_channel::<RawFrame>(1);

    let render_handle = std::thread::Builder::new()
        .name("webxr-render".into())
        .spawn(move || {
            render_thread(
                volume,
                fps,
                ipd,
                viewing_distance,
                render_scale,
                sample_density,
                prediction_horizon_secs,
                pose_rx,
                raw_tx,
            );
        })
        .expect("failed to spawn render thread");

    encode_loop(fps, bitrate_kbps, raw_rx, video_tx, pose_tag_tx);

    render_handle.join().ok();
}

// ── Render thread ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn render_thread(
    volume: Arc<VolumeData>,
    fps: u32,
    ipd: f32,
    viewing_distance: f32,
    render_scale: f32,
    sample_density: f32,
    prediction_horizon_secs: f32,
    pose_rx: watch::Receiver<Option<TransportPose>>,
    raw_tx: mpsc::SyncSender<RawFrame>,
) {
    let mut renderer = match Renderer::new(&volume) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Renderer init failed: {e}");
            return;
        }
    };
    renderer.ipd = ipd;
    renderer.viewing_distance = viewing_distance;
    renderer.sample_density = sample_density;

    let mut cur_eye_w = DEFAULT_EYE_WIDTH;
    let mut cur_eye_h = DEFAULT_EYE_HEIGHT;

    let mut predictor = PosePredictor::new();
    let mut last_seen_ts: u64 = 0;

    // Make pose_rx mutable (needed for borrow_and_update).
    let mut pose_rx = pose_rx;

    let frame_dur = Duration::from_micros(1_000_000 / fps as u64);
    let mut deadline = Instant::now() + frame_dur;

    const REPORT_INTERVAL: Duration = Duration::from_secs(5);
    let mut fps_window_start = Instant::now();
    let mut sent_count: u32 = 0;
    let mut dropped_count: u32 = 0;

    // Adaptive resolution state.
    let mut current_scale = render_scale;
    // Native (unscaled) per-eye resolution reported by the headset.  Populated
    // once the first pose with eye_width/eye_height arrives.
    let mut native_dims: Option<(u32, u32)> = None;

    // The orientation that was used to render the frame *currently in the readback
    // pipeline* (i.e. frame N-1).  Forwarded as ATW tag so the client can correct
    // for the rotation delta between render time and display time.
    let mut prev_rendered_orientation: Option<[f32; 4]> = None;

    tracing::info!(
        "Render thread started: {} fps, {}×{} per eye (scale {:.2}, pred {:.0} ms)",
        fps,
        cur_eye_w,
        cur_eye_h,
        render_scale,
        prediction_horizon_secs * 1000.0,
    );

    loop {
        // ── Frame pacing ───────────────────────────────────────────────────
        let now = Instant::now();
        if now < deadline {
            std::thread::sleep(deadline - now);
        }
        deadline += frame_dur;

        if deadline < Instant::now() - frame_dur {
            tracing::debug!("render thread: deadline drifted, resetting");
            deadline = Instant::now() + frame_dur;
        }

        // ── Pose update + prediction ───────────────────────────────────────
        let transport_pose = pose_rx.borrow_and_update().clone();
        let rpose = match transport_pose.as_ref() {
            Some(p) => {
                renderer.ipd = p.ipd.unwrap_or(ipd);

                // Update predictor only when a new pose arrives.
                if p.timestamp != last_seen_ts {
                    predictor.update(p, Instant::now());
                    last_seen_ts = p.timestamp;
                }

                // Resize renderer when headset reports a different viewport.
                if let (Some(raw_w), Some(raw_h)) = (p.eye_width, p.eye_height) {
                    native_dims = Some((raw_w, raw_h));
                    let (w, h) = scale_dims(raw_w, raw_h, current_scale);
                    if w != cur_eye_w || h != cur_eye_h {
                        tracing::info!(
                            "Headset viewport: {}×{} native → {}×{} scaled (×{:.2})",
                            raw_w,
                            raw_h,
                            w,
                            h,
                            current_scale,
                        );
                        renderer.resize(w, h);
                        cur_eye_w = w;
                        cur_eye_h = h;
                    }
                }

                // Use predicted pose; fall back to raw if insufficient history.
                predictor
                    .predict(prediction_horizon_secs)
                    .unwrap_or_else(|| to_renderer_pose(p))
            }
            None => RendererPose::default(),
        };

        // Record the orientation for this frame so we can attach it to the ATW tag
        // when the pixels emerge from the readback pipeline on the *next* iteration.
        let rendered_orientation = rpose.orientation.to_array();

        // ── Render ─────────────────────────────────────────────────────────
        // render_frame submits GPU work for `rpose` and returns the *previous*
        // frame's pixels (double-buffered).  Returns None on the first call.
        let maybe_frame = match renderer.render_frame(&rpose) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("render_frame: {e}");
                continue;
            }
        };

        // ── Send to encode thread; drop if encode is still busy ────────────
        if let Some(frame) = maybe_frame {
            // ATW orientation must match the frame whose pixels we are sending,
            // which was rendered on the *previous* iteration.
            let atw_orientation = prev_rendered_orientation.unwrap_or(rendered_orientation);

            match raw_tx.try_send(RawFrame {
                rgba: frame.rgba,
                width: frame.width,
                height: frame.height,
                rendered_orientation: atw_orientation,
            }) {
                Ok(()) => {
                    sent_count += 1;
                }
                Err(TrySendError::Full(_)) => {
                    dropped_count += 1;
                    tracing::debug!("encode busy — dropping rendered frame");
                }
                Err(TrySendError::Disconnected(_)) => {
                    tracing::info!("Render thread: encode channel closed, stopping");
                    break;
                }
            }
        }

        prev_rendered_orientation = Some(rendered_orientation);

        // ── FPS reporting + adaptive resolution ───────────────────────────
        let elapsed = fps_window_start.elapsed();
        if elapsed >= REPORT_INTERVAL {
            let total = sent_count + dropped_count;
            let drop_rate = if total > 0 {
                dropped_count as f32 / total as f32
            } else {
                0.0
            };
            let actual_fps = sent_count as f32 / elapsed.as_secs_f32();
            tracing::info!(
                "Render: {:.1} fps encoded (target {} fps, {}×{} per eye, \
                 {} dropped ({:.1}%), scale {:.2})",
                actual_fps,
                fps,
                cur_eye_w,
                cur_eye_h,
                dropped_count,
                drop_rate * 100.0,
                current_scale,
            );

            // Adjust render scale based on encode throughput.
            let new_scale = adapt_scale(current_scale, drop_rate);
            if (new_scale - current_scale).abs() > 1e-4 {
                if let Some((nw, nh)) = native_dims {
                    let (sw, sh) = scale_dims(nw, nh, new_scale);
                    if sw != cur_eye_w || sh != cur_eye_h {
                        tracing::info!(
                            "Adaptive resolution: scale {:.2} → {:.2} \
                             (drop_rate={:.1}%, {}×{} → {}×{} per eye)",
                            current_scale,
                            new_scale,
                            drop_rate * 100.0,
                            cur_eye_w,
                            cur_eye_h,
                            sw,
                            sh,
                        );
                        renderer.resize(sw, sh);
                        cur_eye_w = sw;
                        cur_eye_h = sh;
                    }
                }
                current_scale = new_scale;
            }

            sent_count = 0;
            dropped_count = 0;
            fps_window_start = Instant::now();
        }
    }
}

// ── Encode loop ───────────────────────────────────────────────────────────────

fn encode_loop(
    fps: u32,
    bitrate_kbps: u32,
    raw_rx: mpsc::Receiver<RawFrame>,
    video_tx: tokio_mpsc::Sender<Vec<u8>>,
    pose_tag_tx: tokio_mpsc::Sender<[f32; 4]>,
) {
    let mut enc: Option<Box<dyn encoder::FrameEncoder>> = None;
    let mut enc_dims: Option<(u32, u32)> = None;

    while let Ok(raw) = raw_rx.recv() {
        let dims = (raw.width, raw.height);

        if enc_dims != Some(dims) {
            match encoder::default_encoder(raw.width, raw.height, fps, bitrate_kbps) {
                Ok(e) => {
                    tracing::info!("Encoder (re)initialized: {}×{}", raw.width, raw.height);
                    enc = Some(e);
                    enc_dims = Some(dims);
                }
                Err(e) => {
                    tracing::error!("Encoder init failed ({}×{}): {}", raw.width, raw.height, e);
                    continue;
                }
            }
        }

        let Some(ref mut enc) = enc else { continue };

        let encoded = match enc.encode(&raw.rgba, raw.width, raw.height) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("encode: {e}");
                continue;
            }
        };

        // Forward rendered orientation as an ATW pose-tag to the WebRTC session.
        // This is best-effort (drop on full) — a slightly stale tag is still useful.
        let _ = pose_tag_tx.try_send(raw.rendered_orientation);

        match video_tx.try_send(encoded) {
            Ok(()) => {}
            Err(tokio_mpsc::error::TrySendError::Full(_)) => {
                tracing::debug!("video channel full — dropping encoded frame");
            }
            Err(tokio_mpsc::error::TrySendError::Closed(_)) => {
                tracing::info!("Encode loop: video channel closed, stopping");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapt_scale_reduces_on_high_drop_rate() {
        let s = adapt_scale(1.0, 0.15);
        assert!((s - 0.9).abs() < 1e-5, "expected 0.9 got {s}");
    }

    #[test]
    fn adapt_scale_increases_on_zero_drops() {
        let s = adapt_scale(0.75, 0.0);
        assert!((s - 0.8).abs() < 1e-5, "expected 0.80 got {s}");
    }

    #[test]
    fn adapt_scale_clamps_at_minimum() {
        let s = adapt_scale(0.55, 0.15);
        assert!((s - 0.5).abs() < 1e-5, "expected 0.50 (floor) got {s}");
    }

    #[test]
    fn adapt_scale_clamps_at_maximum() {
        let s = adapt_scale(0.98, 0.0);
        assert!((s - 1.0).abs() < 1e-5, "expected 1.00 (cap) got {s}");
    }

    #[test]
    fn adapt_scale_unchanged_in_acceptable_range() {
        let s = adapt_scale(0.75, 0.05);
        assert!((s - 0.75).abs() < 1e-5, "expected unchanged 0.75 got {s}");
    }

    #[test]
    fn scale_dims_rounds_to_mult_of_8_and_2() {
        assert_eq!(scale_dims(1832, 1920, 0.75), (1368, 1440));
        assert_eq!(scale_dims(1832, 1920, 1.0), (1832, 1920));
        assert_eq!(scale_dims(101, 101, 1.0), (96, 100));
        assert_eq!(scale_dims(10, 10, 0.1), (64, 64));
    }

    #[test]
    fn converts_identity_pose() {
        let tp = TransportPose {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 0,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        let rp = to_renderer_pose(&tp);
        assert_eq!(rp.position, glam::Vec3::ZERO);
        assert!((rp.orientation.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn converts_arbitrary_pose() {
        let tp = TransportPose {
            position: [1.0, -2.5, 3.0],
            orientation: [0.0, 0.707, 0.0, 0.707],
            timestamp: 999,
            ipd: Some(0.064),
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        let rp = to_renderer_pose(&tp);
        assert!((rp.position.x - 1.0).abs() < 1e-6);
        assert!((rp.position.y - (-2.5)).abs() < 1e-6);
        assert!((rp.position.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn predictor_returns_none_with_single_pose() {
        let mut pred = PosePredictor::new();
        let pose = TransportPose {
            position: [0.0, 1.6, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 1000,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        pred.update(&pose, Instant::now());
        assert!(pred.predict(0.040).is_none(), "need at least 2 poses");
    }

    #[test]
    fn predictor_extrapolates_linear_position() {
        let mut pred = PosePredictor::new();
        let t0 = Instant::now();

        let p0 = TransportPose {
            position: [0.0, 1.6, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 0,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };
        let p1 = TransportPose {
            position: [0.1, 1.6, 0.0], // moving +X at ~0.1/dt m/s
            orientation: [0.0, 0.0, 0.0, 1.0],
            timestamp: 14,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };

        pred.update(&p0, t0);
        pred.update(&p1, t0 + Duration::from_millis(14));

        let predicted = pred.predict(0.014).expect("should have prediction");
        // At dt=14ms, horizon=14ms: t=1, pred = p1.pos + vel*14ms = p1 + (p1-p0) = [0.2, 1.6, 0.0]
        assert!(
            (predicted.position.x - 0.2).abs() < 0.01,
            "predicted x={}, expected ~0.2",
            predicted.position.x
        );
    }

    #[test]
    fn predictor_identity_orientation_stays_identity() {
        let mut pred = PosePredictor::new();
        let t0 = Instant::now();

        let identity_pose = |ts| TransportPose {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0], // identity — no rotation
            timestamp: ts,
            ipd: None,
            proj_left: None,
            proj_right: None,
            eye_width: None,
            eye_height: None,
        };

        pred.update(&identity_pose(0), t0);
        pred.update(&identity_pose(14), t0 + Duration::from_millis(14));

        let predicted = pred.predict(0.040).expect("prediction available");
        assert!(
            (predicted.orientation.w - 1.0).abs() < 1e-5,
            "identity orientation should remain identity"
        );
    }

    #[test]
    fn run_exits_cleanly_without_gpu() {
        let (video_tx, video_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1);
        let (pose_tag_tx, _pose_tag_rx) = tokio::sync::mpsc::channel::<[f32; 4]>(4);
        let (_pose_tx, pose_rx) = tokio::sync::watch::channel::<Option<TransportPose>>(None);
        drop(video_rx);

        let n = 4u32;
        let data = vec![0.5f32; (n * n * n) as usize];
        let volume = Arc::new(VolumeData::new(
            [n, n, n],
            [1.0, 1.0, 1.0],
            data,
            (0.0, 1.0),
        ));

        run(
            volume,
            72,
            8000,
            0.063,
            2.0,
            1.0,
            0.30,
            0.020,
            pose_rx,
            video_tx,
            pose_tag_tx,
        );
    }
}
