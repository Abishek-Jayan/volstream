use crate::RendererError;
use wgpu::{Adapter, Device, Instance, Queue};

/// Holds the wgpu device, queue, and adapter.
/// Initialized headless (no surface).
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initialize wgpu headless. Prefers Vulkan on Linux, falls back to other backends.
    pub fn new() -> Result<Self, RendererError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, RendererError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                RendererError::GpuInit(
                    "No suitable GPU adapter found. Ensure a Vulkan-capable GPU is available \
                 and WGPU_BACKEND=vulkan if needed."
                        .into(),
                )
            })?;

        let info = adapter.get_info();
        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        tracing::info!("GPU  name    : {}", info.name);
        tracing::info!("GPU  backend : {:?}", info.backend);
        tracing::info!("GPU  type    : {:?}", info.device_type);
        tracing::info!("GPU  driver  : {}", info.driver);
        tracing::info!("GPU  driver v: {}", info.driver_info);
        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // FLOAT32_FILTERABLE: allows R32Float textures to use linear (trilinear)
        // sampling, which is essential for smooth volume ray marching.
        // This is supported on all discrete and most integrated GPUs.
        let required_features = wgpu::Features::FLOAT32_FILTERABLE;
        if !adapter.features().contains(required_features) {
            return Err(RendererError::GpuInit(
                "GPU does not support FLOAT32_FILTERABLE. \
                 This is required for volume texture trilinear sampling."
                    .into(),
            ));
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("webxr-renderer"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| RendererError::GpuInit(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_initializes() {
        // This test requires a GPU. Skip gracefully in CI without one.
        match GpuContext::new() {
            Ok(ctx) => {
                let info = ctx.adapter.get_info();
                println!("GPU: {} ({:?})", info.name, info.backend);
            }
            Err(e) => {
                println!("Skipping GPU test (no adapter): {e}");
            }
        }
    }
}
