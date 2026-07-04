// =============================================================================
// WebXR Server-Side Volume Renderer
// Ray marching shader — renders stereo side-by-side into a 2D output texture.
//
// Left eye:  pixel x ∈ [0, eye_width)
// Right eye: pixel x ∈ [eye_width, eye_width*2)
//
// Projection convention: z_ndc ∈ [0, 1]  (WebGPU / glam::Mat4::perspective_rh)
//   near plane → z_ndc = 0
//   far  plane → z_ndc = 1
// =============================================================================

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------

struct CameraUniforms {
    left_view_inv:   mat4x4<f32>,  // offset   0 (64 bytes)
    left_proj_inv:   mat4x4<f32>,  // offset  64
    right_view_inv:  mat4x4<f32>,  // offset 128
    right_proj_inv:  mat4x4<f32>,  // offset 192
    world_to_volume: mat4x4<f32>,  // offset 256 — maps world → [0,1]³ uvw
    // params.x = eye_width  (pixels per eye)
    // params.y = eye_height
    // params.z = step_size  (in volume [0,1]³ space)
    // params.w = max_steps
    params:  vec4<f32>,            // offset 320
    // params2.x = sample_density (0.0–1.0; 1.0 = all samples, 0.6 = 60%)
    // params2.yzw = unused
    params2: vec4<f32>,            // offset 336
}

@group(0) @binding(0) var<uniform> cam: CameraUniforms;

// Volume texture: R32Float single channel.
@group(1) @binding(0) var volume:      texture_3d<f32>;
@group(1) @binding(1) var vol_sampler: sampler;

// ----------------------------------------------------------------------------
// Vertex shader — fullscreen triangle, no vertex buffers needed
// ----------------------------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// ----------------------------------------------------------------------------
// Ray helpers
// ----------------------------------------------------------------------------

struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

// Unproject a screen-space NDC point to a world-space ray.
fn get_ray(ndc: vec2<f32>, view_inv: mat4x4<f32>, proj_inv: mat4x4<f32>) -> Ray {
    var view_near = proj_inv * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    var view_far  = proj_inv * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    view_near = view_near / view_near.w;
    view_far  = view_far  / view_far.w;
    let world_near = (view_inv * view_near).xyz;
    let world_far  = (view_inv * view_far).xyz;
    return Ray(world_near, normalize(world_far - world_near));
}

// Ray–AABB intersection. Returns (t_enter, t_exit).
fn ray_aabb(
    origin:  vec3<f32>,
    inv_dir: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>,
) -> vec2<f32> {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_enter = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let t_exit  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return vec2<f32>(t_enter, t_exit);
}

// ----------------------------------------------------------------------------
// Stochastic sparse sampling
// ----------------------------------------------------------------------------

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(443.8975, 441.423, 437.195));
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

// ----------------------------------------------------------------------------
// Single-channel volume ray march
// ----------------------------------------------------------------------------

fn march(ray: Ray, frag_coord: vec2<f32>) -> vec4<f32> {
    let step_size      = cam.params.z;
    let max_steps      = i32(cam.params.w);
    let sample_density = cam.params2.x;

    let vol_origin = (cam.world_to_volume * vec4<f32>(ray.origin, 1.0)).xyz;
    let vol_dir    = normalize((cam.world_to_volume * vec4<f32>(ray.dir, 0.0)).xyz);

    let inv_dir = 1.0 / vol_dir;
    let t       = ray_aabb(vol_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(1.0));

    if t.x > t.y || t.y < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let t_start = max(t.x, 0.0);
    let t_end   = t.y;

    var acc    = vec4<f32>(0.0);
    var t_curr = t_start;

    for (var i = 0; i < max_steps; i++) {
        if t_curr >= t_end || acc.a >= 0.99 {
            break;
        }

        let uvw = vol_origin + vol_dir * t_curr;

        if hash3(vec3<f32>(frag_coord.x, frag_coord.y, f32(i))) > sample_density {
            t_curr += step_size;
            continue;
        }

        let density = textureSample(volume, vol_sampler, uvw).r;

        // Front-to-back composite: render as white volume.
        let blend = 1.0 - acc.a;
        acc.r += density * blend;
        acc.g += density * blend;
        acc.b += density * blend;
        acc.a += density * blend;

        t_curr += step_size;
    }

    return acc;
}

// ----------------------------------------------------------------------------
// Fragment shader
// ----------------------------------------------------------------------------

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let eye_width  = cam.params.x;
    let eye_height = cam.params.y;

    let is_right = frag_coord.x >= eye_width;
    let x_in_eye = select(frag_coord.x, frag_coord.x - eye_width, is_right);

    let ndc_x = (x_in_eye  / eye_width)  * 2.0 - 1.0;
    let ndc_y = 1.0 - (frag_coord.y / eye_height) * 2.0;
    let ndc   = vec2<f32>(ndc_x, ndc_y);

    var view_inv: mat4x4<f32>;
    var proj_inv: mat4x4<f32>;
    if is_right {
        view_inv = cam.right_view_inv;
        proj_inv = cam.right_proj_inv;
    } else {
        view_inv = cam.left_view_inv;
        proj_inv = cam.left_proj_inv;
    }

    let ray = get_ray(ndc, view_inv, proj_inv);
    return march(ray, frag_coord.xy);
}
