use std::net::SocketAddr;
use std::time::Instant;

use str0m::change::{SdpAnswer, SdpOffer, SdpPendingOffer};
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind, MediaTime, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, watch};

use crate::datachannel::HeadPose;

/// Binary magic byte for server→client rendered-pose tags.
const POSE_TAG_MAGIC: u8 = 0x52; // 'R' for "rendered"
const POSE_TAG_VERSION: u8 = 0x01;
/// Total size: magic(1) + version(1) + qx(4) + qy(4) + qz(4) + qw(4) = 18 bytes.
const POSE_TAG_SIZE: usize = 18;

/// Server-side WebRTC session.
///
/// - **Active** side: creates the SDP offer (the browser answers).
/// - Offers one H.264 `SendOnly` video track + one `pose` data channel.
/// - ICE-lite mode: suitable for a server with a public/local IP.
pub struct WebRtcSession {
    rtc: Rtc,
    video_mid: Option<Mid>,
}

impl WebRtcSession {
    /// Create a new session with H.264 video only (no audio, VP8/VP9).
    pub fn new() -> Self {
        let rtc = Rtc::builder()
            .set_ice_lite(true)
            .clear_codecs()
            .enable_h264(true)
            .build();

        WebRtcSession {
            rtc,
            video_mid: None,
        }
    }

    /// Register each of `local_addrs` as an ICE host candidate, then generate the SDP offer.
    ///
    /// Advertising more than one candidate (e.g. the LAN IP and loopback) lets the
    /// browser pick whichever path actually connects — needed because on WSL2 with
    /// mirrored networking, a browser on the same machine can fail to hairpin back
    /// to the host's own LAN IP even though external devices reach it fine.
    ///
    /// Each address must be the bound address of a UDP socket passed to [`run`], in
    /// the same order.
    pub fn create_offer(
        &mut self,
        local_addrs: &[SocketAddr],
    ) -> Result<(SdpOffer, SdpPendingOffer), crate::TransportError> {
        for local_addr in local_addrs {
            let candidate = Candidate::host(*local_addr, "udp")
                .map_err(|e| crate::TransportError::WebRtc(e.to_string()))?;
            self.rtc.add_local_candidate(candidate);
        }

        let mut change = self.rtc.sdp_api();
        let mid = change.add_media(MediaKind::Video, Direction::SendOnly, None, None);
        change.add_channel("pose".into());

        let (offer, pending) = change
            .apply()
            .ok_or_else(|| crate::TransportError::WebRtc("sdp_api apply returned None".into()))?;

        self.video_mid = Some(mid);
        Ok((offer, pending))
    }

    /// Apply the SDP answer received from the browser.
    pub fn accept_answer(
        &mut self,
        pending: SdpPendingOffer,
        answer: SdpAnswer,
    ) -> Result<(), crate::TransportError> {
        self.rtc
            .sdp_api()
            .accept_answer(pending, answer)
            .map_err(|e| crate::TransportError::WebRtc(e.to_string()))
    }

    /// Add a remote ICE candidate (trickle ICE).
    pub fn add_remote_candidate(&mut self, c: Candidate) {
        self.rtc.add_remote_candidate(c);
    }

    /// Drive the WebRTC session to completion.
    ///
    /// This is the hot path — runs in its own tokio task.
    ///
    /// - `lan_socket`/`lan_addr`   — UDP socket bound to the LAN-facing candidate
    ///   passed to [`create_offer`], and its bound address.
    /// - `loop_socket`/`loop_addr` — UDP socket bound to the loopback candidate
    ///   passed to [`create_offer`], and its bound address.
    /// - `video_rx`     — encoded H.264 NAL-unit bytes, one `Vec<u8>` per frame.
    /// - `pose_tx`      — publishes the latest head pose for the render loop.
    /// - `pose_tag_rx`  — rendered-pose orientations from the encode thread,
    ///   forwarded to the client as ATW tags over the data channel.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        mut self,
        lan_socket: UdpSocket,
        lan_addr: SocketAddr,
        loop_socket: UdpSocket,
        loop_addr: SocketAddr,
        mut video_rx: mpsc::Receiver<Vec<u8>>,
        pose_tx: watch::Sender<Option<HeadPose>>,
        mut pose_tag_rx: mpsc::Receiver<[f32; 4]>,
    ) {
        let mut lan_buf = vec![0u8; 2048];
        let mut loop_buf = vec![0u8; 2048];
        let session_start = Instant::now();
        let mut connected = false;
        let mut channel_id: Option<ChannelId> = None;
        let mut pose_tag_open = true; // becomes false when sender is dropped

        loop {
            if !self.rtc.is_alive() {
                tracing::info!("WebRTC session ended");
                break;
            }

            // Drain all pending output before blocking on I/O.
            let timeout = loop {
                match self.rtc.poll_output() {
                    Err(e) => {
                        tracing::error!("WebRTC poll_output: {e}");
                        return;
                    }
                    Ok(Output::Timeout(t)) => break t,
                    Ok(Output::Transmit(t)) => {
                        // Send from whichever socket owns the local candidate str0m picked.
                        let socket = if t.source == loop_addr {
                            &loop_socket
                        } else {
                            &lan_socket
                        };
                        if let Err(e) = socket.send_to(&t.contents, t.destination).await {
                            tracing::warn!("WebRTC UDP send: {e}");
                        }
                    }
                    Ok(Output::Event(event)) => {
                        handle_event(
                            &mut self.rtc,
                            event,
                            &pose_tx,
                            &mut connected,
                            &mut channel_id,
                        );
                    }
                }
            };

            // Wait for the next event: incoming UDP, timeout, video frame, or pose tag.
            let wait = timeout.saturating_duration_since(Instant::now());

            tokio::select! {
                result = lan_socket.recv_from(&mut lan_buf) => {
                    match result {
                        Ok((n, src)) => {
                            let data = &lan_buf[..n];
                            if let Ok(recv) = Receive::new(Protocol::Udp, src, lan_addr, data) {
                                let _ = self.rtc.handle_input(Input::Receive(Instant::now(), recv));
                            }
                        }
                        Err(e) => tracing::warn!("WebRTC UDP recv (LAN): {e}"),
                    }
                }
                result = loop_socket.recv_from(&mut loop_buf) => {
                    match result {
                        Ok((n, src)) => {
                            let data = &loop_buf[..n];
                            if let Ok(recv) = Receive::new(Protocol::Udp, src, loop_addr, data) {
                                let _ = self.rtc.handle_input(Input::Receive(Instant::now(), recv));
                            }
                        }
                        Err(e) => tracing::warn!("WebRTC UDP recv (loopback): {e}"),
                    }
                }
                _ = tokio::time::sleep(wait) => {
                    let _ = self.rtc.handle_input(Input::Timeout(Instant::now()));
                }
                frame = video_rx.recv() => {
                    match frame {
                        Some(data) if connected => {
                            write_video_frame(&mut self.rtc, self.video_mid, &data, session_start);
                        }
                        Some(_) => {} // not yet connected, drop frame
                        None => {
                            tracing::info!("WebRTC: video channel closed, ending session");
                            return;
                        }
                    }
                }
                tag = async {
                    if pose_tag_open { pose_tag_rx.recv().await }
                    else { std::future::pending().await }
                } => {
                    match tag {
                        Some(orient) => {
                            send_pose_tag(&mut self.rtc, channel_id, &orient);
                        }
                        None => {
                            pose_tag_open = false; // render loop exited; stop selecting
                        }
                    }
                }
            }
        }
    }
}

impl Default for WebRtcSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Write one encoded H.264 frame as an RTP sample.
fn write_video_frame(rtc: &mut Rtc, video_mid: Option<Mid>, data: &[u8], session_start: Instant) {
    let Some(mid) = video_mid else { return };

    // Borrow #1: read the payload type for H.264.
    let pt = {
        let Some(writer) = rtc.writer(mid) else {
            return;
        };
        let found = writer
            .payload_params()
            .find(|p| p.spec().codec == Codec::H264)
            .map(|p| p.pt());
        found
    };
    let Some(pt) = pt else {
        tracing::warn!("WebRTC: no H.264 payload type negotiated");
        return;
    };

    // RTP timestamp: 90 kHz clock derived from wall-clock elapsed time.
    let elapsed_secs = session_start.elapsed().as_secs_f64();
    let rtp_time = MediaTime::new(
        (elapsed_secs * 90_000.0) as u64,
        str0m::media::Frequency::NINETY_KHZ,
    );
    let wallclock = Instant::now();

    // Borrow #2: write the frame.
    let Some(writer) = rtc.writer(mid) else {
        return;
    };
    if let Err(e) = writer.write(pt, wallclock, rtp_time, data.to_vec()) {
        tracing::debug!("WebRTC writer.write: {e}");
    }
}

/// Process a single str0m event.
fn handle_event(
    rtc: &mut Rtc,
    event: Event,
    pose_tx: &watch::Sender<Option<HeadPose>>,
    connected: &mut bool,
    channel_id: &mut Option<ChannelId>,
) {
    match event {
        Event::Connected => {
            *connected = true;
            tracing::info!("WebRTC: DTLS connected, video streaming enabled");
        }
        Event::IceConnectionStateChange(state) => {
            tracing::info!("WebRTC ICE state: {state:?}");
            if state == IceConnectionState::Disconnected {
                rtc.disconnect();
            }
        }
        Event::ChannelOpen(id, label) => {
            tracing::info!("WebRTC data channel open: id={id:?} label={label}");
            *channel_id = Some(id);
        }
        Event::ChannelData(data) => {
            if let Some(pose) = crate::datachannel::try_parse_pose(&data.data) {
                let _ = pose_tx.send(Some(pose));
            }
        }
        Event::ChannelClose(id) => {
            tracing::info!("WebRTC data channel closed: id={id:?}");
            if *channel_id == Some(id) {
                *channel_id = None;
            }
        }
        _ => {}
    }
}

/// Encode and transmit a rendered-pose tag on the data channel.
///
/// Wire format (18 bytes, little-endian):
/// ```text
/// [0]     magic   = 0x52 ('R')
/// [1]     version = 0x01
/// [2..6]  qx  f32
/// [6..10] qy  f32
/// [10..14] qz f32
/// [14..18] qw f32
/// ```
fn send_pose_tag(rtc: &mut Rtc, channel_id: Option<ChannelId>, orient: &[f32; 4]) {
    let Some(id) = channel_id else { return };
    let Some(mut ch) = rtc.channel(id) else {
        return;
    };

    let mut buf = [0u8; POSE_TAG_SIZE];
    buf[0] = POSE_TAG_MAGIC;
    buf[1] = POSE_TAG_VERSION;
    buf[2..6].copy_from_slice(&orient[0].to_le_bytes());
    buf[6..10].copy_from_slice(&orient[1].to_le_bytes());
    buf[10..14].copy_from_slice(&orient[2].to_le_bytes());
    buf[14..18].copy_from_slice(&orient[3].to_le_bytes());

    if let Err(e) = ch.write(true, &buf) {
        tracing::debug!("pose tag send failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn localhost_addr() -> SocketAddr {
        "127.0.0.1:19876".parse().unwrap()
    }

    #[test]
    fn session_creates_without_panic() {
        let _ = WebRtcSession::new();
    }

    #[test]
    fn create_offer_returns_sdp() {
        let mut session = WebRtcSession::new();
        let (offer, _pending) = session
            .create_offer(&[localhost_addr()])
            .expect("create_offer should succeed");

        // SDP must contain H.264 and the data channel application line
        let sdp_str = serde_json::to_string(&offer).unwrap();
        assert!(
            sdp_str.to_lowercase().contains("h264"),
            "offer SDP should contain H264, got: {sdp_str}"
        );
    }

    #[test]
    fn create_offer_sets_video_mid() {
        let mut session = WebRtcSession::new();
        assert!(session.video_mid.is_none());
        session.create_offer(&[localhost_addr()]).unwrap();
        assert!(session.video_mid.is_some());
    }

    #[test]
    fn channel_creation_for_run() {
        // Verify that the channel types used by run() can be constructed correctly.
        let (video_tx, _video_rx) = mpsc::channel::<Vec<u8>>(16);
        let (pose_tx, _pose_rx) = watch::channel::<Option<HeadPose>>(None);
        drop(video_tx);
        drop(pose_tx);
    }
}
