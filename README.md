# volstream

[![Made with Claude](https://img.shields.io/badge/Made%20with-Claude-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/claude-code)

Real-time volumetric renderer streamed to WebXR clients over WebRTC. A Rust server GPU-renders a volume dataset (NRRD / OME-Zarr) frame-by-frame, encodes it to H.264, and streams it to a browser client for low-latency VR display.

## Architecture

Cargo workspace with five crates:

| Crate | Purpose |
|---|---|
| `volume-loader` | Loads volume data from NRRD and OME-Zarr sources |
| `renderer` | wgpu-based GPU raymarching/volume rendering pipeline |
| `encoder` | H.264 video encoding (software + NVENC) |
| `transport` | WebRTC signaling and data/media channels (str0m) |
| `server` | Axum HTTP/WebSocket server orchestrating the render loop |

`client/` holds the browser-side WebXR viewer (vanilla JS).

## Requirements

- Rust (stable, 2021 edition, workspace resolver `"2"`)
- A GPU with Vulkan support (via `wgpu`)
- `ffmpeg` libraries available at build time (via `ffmpeg-next`)

## Building

```sh
cargo build --workspace
```

## Running

```sh
cargo run -p server
```

Configuration lives in `config/default.toml`. Open `client/index.html` in a WebXR-capable browser to connect.

## Testing

```sh
cargo nextest run
```

Test timeouts are configured in `.config/nextest.toml`.

## License

MIT — see [LICENSE](LICENSE).
