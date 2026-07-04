pub mod datachannel;
pub mod media;
pub mod pairing;
pub mod signaling;

pub use datachannel::HeadPose;
pub use media::WebRtcSession;
pub use pairing::PairingState;

#[derive(thiserror::Error, Debug)]
pub enum TransportError {
    #[error("Pairing failed: {0}")]
    Pairing(String),
    #[error("Signaling error: {0}")]
    Signaling(String),
    #[error("WebRTC error: {0}")]
    WebRtc(String),
    #[error("Serialization error: {0}")]
    Json(#[from] serde_json::Error),
}
