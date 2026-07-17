use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Holds the pairing code and tracks whether a client is currently paired.
#[derive(Debug)]
pub struct PairingState {
    pub code: String,
    pub client_connected: Arc<AtomicBool>,
}

impl PairingState {
    /// Generate a new random 6-digit pairing code and print it to the console.
    pub fn generate() -> Self {
        let code = format!("{:06}", rand::rng().random_range(0..=999999));
        println!("\n========================================");
        println!("  Pairing code: {}", code);
        println!("========================================\n");
        Self {
            code,
            client_connected: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns true if the given code matches and no client is currently connected.
    pub fn try_pair(&self, code: &str) -> bool {
        if self.client_connected.load(Ordering::SeqCst) {
            return false; // already have a client
        }
        if code == self.code {
            self.client_connected.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn disconnect(&self) {
        self.client_connected.store(false, Ordering::SeqCst);
    }

    pub fn is_connected(&self) -> bool {
        self.client_connected.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_is_six_digits() {
        let state = PairingState::generate();
        assert_eq!(state.code.len(), 6);
        assert!(state.code.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn correct_code_pairs_successfully() {
        let state = PairingState::generate();
        let code = state.code.clone();
        assert!(state.try_pair(&code));
        assert!(state.is_connected());
    }

    #[test]
    fn wrong_code_is_rejected() {
        let state = PairingState::generate();
        let wrong = if state.code == "000000" {
            "000001"
        } else {
            "000000"
        };
        assert!(!state.try_pair(wrong));
        assert!(!state.is_connected());
    }

    #[test]
    fn second_connection_is_rejected() {
        let state = PairingState::generate();
        let code = state.code.clone();
        assert!(state.try_pair(&code));
        assert!(!state.try_pair(&code)); // second attempt rejected
    }

    #[test]
    fn reconnect_allowed_after_disconnect() {
        let state = PairingState::generate();
        let code = state.code.clone();
        assert!(state.try_pair(&code));
        state.disconnect();
        assert!(!state.is_connected());
        assert!(state.try_pair(&code));
    }
}
