use anyhow::Result;
use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use transport::pairing::PairingState;
use volume_loader::VolumeData;

/// Runtime configuration derived from CLI args.
#[derive(Debug, Clone)]
pub struct Config {
    pub fps: u32,
    pub bitrate_kbps: u32,
    /// Inter-pupillary distance in metres.
    pub ipd: f32,
    /// Distance in metres from the user to the centre of the volume.
    pub viewing_distance: f32,
    /// Local IP address exposed in WebRTC ICE candidates.
    pub local_ip: IpAddr,
    /// Fixed UDP port for the WebRTC media socket. Pinned rather than
    /// OS-assigned so a single firewall rule stays valid across restarts.
    pub media_port: u16,
    /// Scale factor applied to headset-reported eye resolution before rendering.
    /// 1.0 = native resolution; 0.75 = 75% (reduces GPU/encode load).
    pub render_scale: f32,
    /// Fraction of ray-march samples to evaluate per ray (0.0–1.0).
    /// Lower values skip samples stochastically, trading quality for speed.
    pub sample_density: f32,
    /// How far ahead (in seconds) to extrapolate head pose for latency compensation.
    pub prediction_horizon_secs: f32,
}

/// Shared application state wrapped in `Arc` for axum handler access.
pub struct AppState {
    pub pairing: Arc<PairingState>,
    pub volume: Option<Arc<VolumeData>>,
    pub config: Config,
}

impl AppState {
    pub fn new(volume_path: Option<&Path>, config: Config) -> Result<Arc<Self>> {
        let volume = if let Some(path) = volume_path {
            tracing::info!("Loading volume from {}", path.display());
            let v = volume_loader::load_volume(path)?;
            tracing::info!(
                "Volume loaded: {:?} voxels, spacing {:?}",
                v.dims,
                v.spacing
            );
            Some(Arc::new(v))
        } else {
            tracing::warn!("No volume file specified. Use --volume <path> to load a volume.");
            None
        };

        let pairing = Arc::new(PairingState::generate());

        Ok(Arc::new(Self {
            pairing,
            volume,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            prediction_horizon_secs: 0.020,
        }
    }

    #[test]
    fn creates_without_volume() {
        let state = AppState::new(None, test_config()).unwrap();
        assert!(state.volume.is_none());
        assert!(!state.pairing.is_connected());
    }

    #[test]
    fn config_fields_are_stored() {
        let state = AppState::new(None, test_config()).unwrap();
        assert_eq!(state.config.fps, 72);
        assert_eq!(state.config.bitrate_kbps, 8000);
        assert!((state.config.ipd - 0.063).abs() < 1e-6);
    }

    #[test]
    fn pairing_code_is_generated() {
        let state = AppState::new(None, test_config()).unwrap();
        assert_eq!(state.pairing.code.len(), 6);
        assert!(state.pairing.code.chars().all(|c| c.is_ascii_digit()));
    }
}
