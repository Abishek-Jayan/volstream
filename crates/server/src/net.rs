//! Local address discovery shared by startup and ICE candidate selection.

use std::net::{IpAddr, Ipv4Addr};

/// Best-effort detection of this machine's LAN-facing IP by asking the OS
/// which local address it would use to route toward a public address.
/// `UdpSocket::connect` only resolves a route — no packets are sent.
pub fn detect_local_ip() -> Option<IpAddr> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|a| a.ip())
}

/// Resolve the concrete IPs to advertise as ICE host candidates for `local_ip`.
///
/// A wildcard address (`0.0.0.0` / `::`) can be bound but never dialled, so it
/// is expanded into the concrete addresses it stands for: the detected LAN IP
/// (for remote devices) plus loopback (for a browser on this machine).
///
/// Any other address is used as-is, with loopback added when it is genuinely
/// distinct — binding the same port twice on overlapping addresses fails with
/// `EADDRINUSE`.
///
/// The result is deduplicated and never empty.
pub fn candidate_ips_for(local_ip: IpAddr) -> Vec<IpAddr> {
    let loopback: IpAddr = Ipv4Addr::LOCALHOST.into();

    let mut ips = Vec::new();
    if local_ip.is_unspecified() {
        ips.extend(detect_local_ip());
        ips.push(loopback);
    } else {
        ips.push(local_ip);
        ips.push(loopback);
    }

    ips.dedup_by(|a, b| a == b);
    let mut seen = Vec::new();
    ips.retain(|ip| {
        let fresh = !seen.contains(ip);
        if fresh {
            seen.push(*ip);
        }
        fresh
    });

    if ips.is_empty() {
        ips.push(loopback);
    }
    ips
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

    fn lan() -> IpAddr {
        "192.168.1.50".parse().unwrap()
    }

    #[test]
    fn lan_ip_also_advertises_loopback() {
        let ips = candidate_ips_for(lan());
        assert_eq!(ips, vec![lan(), LOOPBACK]);
    }

    #[test]
    fn loopback_is_not_duplicated() {
        // Regression: binding 127.0.0.1 twice on one port fails with EADDRINUSE.
        let ips = candidate_ips_for(LOOPBACK);
        assert_eq!(ips, vec![LOOPBACK]);
    }

    #[test]
    fn wildcard_is_never_advertised_directly() {
        // Regression: 0.0.0.0 is bindable but not dialable, and a wildcard bind
        // already claims loopback, so it must expand rather than bind twice.
        let ips = candidate_ips_for("0.0.0.0".parse().unwrap());
        assert!(
            !ips.iter().any(|ip| ip.is_unspecified()),
            "wildcard must not be advertised: {ips:?}"
        );
        assert!(ips.contains(&LOOPBACK), "loopback expected: {ips:?}");
    }

    #[test]
    fn result_is_never_empty() {
        for ip in ["0.0.0.0", "::", "127.0.0.1", "192.168.1.50"] {
            let ips = candidate_ips_for(ip.parse().unwrap());
            assert!(!ips.is_empty(), "empty for {ip}");
        }
    }

    #[test]
    fn no_duplicate_addresses() {
        for ip in ["0.0.0.0", "::", "127.0.0.1", "10.0.0.7"] {
            let ips = candidate_ips_for(ip.parse().unwrap());
            let mut sorted = ips.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), ips.len(), "duplicates for {ip}: {ips:?}");
        }
    }
}
