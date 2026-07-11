use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, watch};
use tower_http::cors::CorsLayer;
use transport::{
    pairing::PairingState,
    signaling::{recv_json, send_json, ClientMsg, ServerMsg},
    HeadPose, WebRtcSession,
};

use crate::state::AppState;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/app.js", get(appjs_handler))
        .route("/health", get(health_handler))
        .route("/ws/signal", get(ws_signal_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../../../client/index.html"))
}

async fn appjs_handler() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        include_str!("../../../client/app.js"),
    )
}

async fn health_handler(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "status": "ok",
        "client_connected": state.pairing.is_connected(),
        "volume_loaded": state.volume.is_some(),
    }))
}

async fn ws_signal_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_signal(socket, state))
}

/// Handle a single WebSocket signaling connection.
///
/// Protocol:
/// 1. Receive `{"type":"pair","code":"XXXXXX"}` — validate against stored code.
/// 2. Send SDP offer.
/// 3. Receive SDP answer.
/// 4. Launch WebRTC session + render loop tasks.
/// 5. Drain the WebSocket until the client closes it.
async fn handle_signal(mut ws: WebSocket, state: Arc<AppState>) {
    // Bind the WebRTC media UDP socket to the configured LAN IP (not the
    // wildcard address) on a fixed port, so a single firewall rule for this
    // port stays valid across restarts and the loopback socket below can
    // bind the same port number on its own address.
    let lan_addr = SocketAddr::new(state.config.local_ip, state.config.media_port);
    let socket = match UdpSocket::bind(lan_addr).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("signal: failed to bind UDP socket: {e}");
            return;
        }
    };
    let udp_port = match socket.local_addr() {
        Ok(a) => a.port(),
        Err(e) => {
            tracing::error!("signal: failed to get UDP local addr: {e}");
            return;
        }
    };

    // The ICE candidate advertises the server's reachable IP and the bound UDP port.
    let ice_addr = SocketAddr::new(state.config.local_ip, udp_port);

    let mut session = WebRtcSession::new();
    let (offer, pending) = match session.create_offer(ice_addr) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("signal: create_offer failed: {e}");
                return;
            }
        };

    // ── Step 1: receive pair code ──────────────────────────────────────────────

    let code = match recv_json::<ClientMsg>(&mut ws).await {
        Some(ClientMsg::Pair { code }) => code,
        Some(_) => {
            tracing::warn!("signal: expected Pair message, got something else");
            return;
        }
        None => {
            tracing::warn!("signal: WebSocket closed before Pair message");
            return;
        }
    };

    if !state.pairing.try_pair(&code) {
        let _ = send_json(
            &mut ws,
            &ServerMsg::Error {
                msg: "Invalid pairing code or server already in use".into(),
            },
        )
        .await;
        return;
    }
    tracing::info!("Client paired — starting WebRTC handshake");

    // ── Step 2: send offer ─────────────────────────────────────────────────────

    if send_json(&mut ws, &ServerMsg::Offer { sdp: offer }).await.is_none() {
        tracing::warn!("signal: WebSocket closed while sending offer");
        state.pairing.disconnect();
        return;
    }

    // ── Step 3: receive answer ─────────────────────────────────────────────────

    let answer = match recv_json::<ClientMsg>(&mut ws).await {
        Some(ClientMsg::Answer { sdp }) => sdp,
        Some(_) => {
            tracing::warn!("signal: expected Answer message, got something else");
            state.pairing.disconnect();
            return;
        }
        None => {
            tracing::warn!("signal: WebSocket closed before Answer message");
            state.pairing.disconnect();
            return;
        }
    };

    if let Err(e) = session.accept_answer(pending, answer) {
        tracing::error!("signal: accept_answer failed: {e}");
        state.pairing.disconnect();
        return;
    }
    tracing::info!("WebRTC answer accepted — launching render pipeline");

    // ── Step 4: launch tasks ───────────────────────────────────────────────────

    let (video_tx, video_rx) = mpsc::channel::<Vec<u8>>(2);
    let (pose_tx, pose_rx) = watch::channel::<Option<HeadPose>>(None);
    // ATW pose-tag channel: render loop → WebRTC session → client data channel.
    let (pose_tag_tx, pose_tag_rx) = mpsc::channel::<[f32; 4]>(4);

    // `_no_volume_guard` keeps these senders/receivers alive in the no-volume
    // path so the WebRTC session starts cleanly.
    let _no_volume_guard: Option<(mpsc::Sender<Vec<u8>>, watch::Receiver<Option<HeadPose>>, mpsc::Sender<[f32; 4]>)>;

    if let Some(volume) = state.volume.clone() {
        let fps = state.config.fps;
        let bitrate = state.config.bitrate_kbps;
        let ipd = state.config.ipd;
        let viewing_distance = state.config.viewing_distance;
        let render_scale = state.config.render_scale;
        let sample_density = state.config.sample_density;
        let prediction_horizon_secs = state.config.prediction_horizon_secs;
        tokio::task::spawn_blocking(move || {
            crate::render_loop::run(
                volume, fps, bitrate, ipd, viewing_distance, render_scale, sample_density,
                prediction_horizon_secs, pose_rx, video_tx, pose_tag_tx,
            );
        });
        _no_volume_guard = None;
    } else {
        tracing::warn!("No volume loaded — WebRTC session will stream no video");
        _no_volume_guard = Some((video_tx, pose_rx, pose_tag_tx));
    }

    // Spawn the WebRTC drive loop; it resets pairing when it exits.
    let pairing: Arc<PairingState> = state.pairing.clone();
    tokio::spawn(async move {
        session.run(socket, lan_addr, video_rx, pose_tx, pose_tag_rx).await;
        pairing.disconnect();
        tracing::info!("WebRTC session ended — pairing reset");
    });

    // ── Step 5: drain WebSocket until client closes ────────────────────────────
    loop {
        match ws.recv().await {
            Some(Ok(_)) => {} // keep alive — absorb any trickle ICE or keepalive messages
            _ => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{AppState, Config};

    fn test_config() -> Config {
        Config {
            fps: 72,
            bitrate_kbps: 8000,
            ipd: 0.063,
            viewing_distance: 2.0,
            local_ip: "127.0.0.1".parse().unwrap(),
            media_port: 40100,
            render_scale: 1.0,
            sample_density: 0.30,
            prediction_horizon_secs: 0.0,
        }
    }

    #[test]
    fn router_creates_without_panic() {
        let state = AppState::new(None, test_config()).unwrap();
        let _ = create_router(state);
    }

    #[test]
    fn health_route_is_registered() {
        let state = AppState::new(None, test_config()).unwrap();
        let router = create_router(state);
        // Verify the router was built (non-empty)
        let _ = router;
    }
}
