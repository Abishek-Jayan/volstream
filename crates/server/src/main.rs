mod app;
mod net;
mod render_loop;
mod state;

use anyhow::Result;
use axum_server::tls_rustls::RustlsConfig;
use clap::Parser;
use std::net::IpAddr;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use state::{AppState, Config};

#[derive(Parser, Debug)]
#[command(name = "webxr-server", about = "Server-side WebXR volumetric renderer")]
struct Args {
    /// Path to volume file (.nrrd or .zarr directory)
    #[arg(short, long)]
    volume: Option<PathBuf>,

    /// HTTPS / WebSocket listen port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Local IP address used in WebRTC ICE candidates and the TLS SAN.
    /// Set to the IP your device can reach on the local network, e.g. 192.168.1.100.
    /// Auto-detected from the default outbound route when omitted.
    #[arg(long)]
    local_ip: Option<IpAddr>,

    /// Windows specific fix: Fixed UDP port for the WebRTC media socket. Pinned rather than OS-assigned so a single firewall rule stays valid across restarts.
    #[arg(long, default_value = "40100")]
    media_port: u16,

    /// Target render / stream frame rate (fps)
    #[arg(long, default_value = "72")]
    fps: u32,

    /// H.264 stream bitrate in kbps
    #[arg(long, default_value = "8000")]
    bitrate: u32,

    /// Inter-pupillary distance in metres
    #[arg(long, default_value = "0.063")]
    ipd: f32,

    /// Distance in metres from the user to the centre of the volume.
    /// Increase if the volume feels too close; decrease to move it nearer.
    #[arg(long, default_value = "2.0")]
    volume_distance: f32,

    /// Scale factor applied to the headset-reported per-eye resolution before
    /// rendering. 1.0 = native resolution. Use 0.75 to render at 75% of native
    /// and reduce GPU / encode latency by ~40%.
    #[arg(long, default_value = "1.0")]
    render_scale: f32,

    /// Fraction of ray-march samples to evaluate per ray (0.0–1.0).
    /// 0.60 = 60% of samples (default). Lower values are faster but less accurate.
    #[arg(long, default_value = "0.60")]
    sample_density: f32,

    /// Pose prediction horizon in milliseconds.
    /// How far ahead to extrapolate head movement to compensate for encode + network latency.
    /// 20 ms is a good default; increase if video lags behind movement, decrease if movement feels overshooted.
    #[arg(long, default_value = "20")]
    prediction_horizon_ms: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default to `info` for all workspace crates; honour RUST_LOG if set.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();

    let local_ip = args.local_ip.unwrap_or_else(|| {
        net::detect_local_ip().unwrap_or_else(|| {
            tracing::warn!(
                "Could not auto-detect a LAN IP; falling back to 127.0.0.1 — pass --local-ip \
                 explicitly if this server needs to be reached from another device"
            );
            IpAddr::from([127, 0, 0, 1])
        })
    });

    let config = Config {
        fps: args.fps,
        bitrate_kbps: args.bitrate,
        ipd: args.ipd,
        viewing_distance: args.volume_distance,
        local_ip,
        media_port: args.media_port,
        render_scale: args.render_scale.clamp(0.1, 1.0),
        sample_density: args.sample_density.clamp(0.0, 1.0),
        prediction_horizon_secs: args.prediction_horizon_ms as f32 / 1000.0,
    };

    let state = AppState::new(args.volume.as_deref(), config)?;
    let router = app::create_router(state);

    // Generate a self-signed TLS certificate valid for both localhost and the
    // specified local IP.  WebXR requires a secure context (HTTPS), so TLS is
    // mandatory.  Browsers will show a security warning for self-signed certs —
    // click "Advanced → Proceed" once to continue.
    let tls_config = make_tls_config(local_ip).await?;

    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", args.port).parse()?;
    tracing::info!("Listening on https://{}", addr);
    tracing::info!("Open https://{}:{}/  on your device", local_ip, args.port);
    tracing::warn!(
        "Self-signed certificate in use — click 'Advanced → Proceed' in the browser to continue"
    );

    axum_server::bind_rustls(addr, tls_config)
        .serve(router.into_make_service())
        .await?;

    Ok(())
}

async fn make_tls_config(local_ip: IpAddr) -> Result<RustlsConfig> {
    let sans = vec!["localhost".to_string(), local_ip.to_string()];
    let rcgen::CertifiedKey { cert, key_pair } = rcgen::generate_simple_self_signed(sans)?;
    let cert_pem = cert.pem().into_bytes();
    let key_pem = key_pair.serialize_pem().into_bytes();
    let config = RustlsConfig::from_pem(cert_pem, key_pem).await?;
    Ok(config)
}
