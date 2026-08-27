'use strict';

// ─── Signaling ────────────────────────────────────────────────────────────────
// Wraps the WebSocket connection. Messages are queued so callers can await them.

class Signaling {
  constructor(wsUrl) {
    this._ws = new WebSocket(wsUrl);
    this._queue = [];
    this._waiting = null;

    this._ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (this._waiting) {
        const resolve = this._waiting;
        this._waiting = null;
        resolve(msg);
      } else {
        this._queue.push(msg);
      }
    };
  }

  /** Wait for the WebSocket connection to open. */
  open() {
    return new Promise((resolve, reject) => {
      if (this._ws.readyState === WebSocket.OPEN) { resolve(); return; }
      this._ws.onopen = resolve;
      this._ws.onerror = () => reject(new Error('WebSocket failed to connect'));
    });
  }

  /** Send a JSON message to the server. */
  send(msg) {
    this._ws.send(JSON.stringify(msg));
  }

  /** Wait for the next incoming message from the server. */
  recv() {
    return new Promise((resolve) => {
      if (this._queue.length > 0) {
        resolve(this._queue.shift());
      } else {
        this._waiting = resolve;
      }
    });
  }

  close() { this._ws.close(); }
}

// ─── PeerConn ─────────────────────────────────────────────────────────────────
// RTCPeerConnection wrapper.  Receives the video track and the pose data channel
// opened by the server.

class PeerConn {
  constructor() {
    this._pc = new RTCPeerConnection();
    this.videoElement = document.createElement('video');
    this.videoElement.autoplay = true;
    this.videoElement.playsInline = true;
    this.dataChannel = null;
    /** Most recent rendered-pose quaternion received from the server (for ATW). */
    this.lastRenderedOrientation = null;

    this._pc.ontrack = (ev) => {
      const stream = ev.streams[0] ?? new MediaStream([ev.track]);
      this.videoElement.srcObject = stream;
    };

    this._pc.ondatachannel = (ev) => {
      this.dataChannel = ev.channel;
      // Ensure binary messages arrive as ArrayBuffer (not Blob).
      this.dataChannel.binaryType = 'arraybuffer';
      this.dataChannel.onmessage = (e) => this._onDataChannelMessage(e);
    };
  }

  /**
   * Parse incoming server→client data channel messages.
   * Currently handles rendered-pose ATW tags (magic=0x52).
   * @param {MessageEvent} ev
   */
  _onDataChannelMessage(ev) {
    if (!(ev.data instanceof ArrayBuffer)) return;
    const dv = new DataView(ev.data);
    if (dv.byteLength < 18) return;
    // Rendered-pose tag: magic(0x52) + version(0x01) + qx qy qz qw (4×f32 LE)
    if (dv.getUint8(0) === 0x52 && dv.getUint8(1) === 0x01) {
      this.lastRenderedOrientation = {
        x: dv.getFloat32(2,  true),
        y: dv.getFloat32(6,  true),
        z: dv.getFloat32(10, true),
        w: dv.getFloat32(14, true),
      };
    }
  }

  /**
   * Set the remote offer, create an answer, and wait for ICE gathering.
   * Returns the local description (answer) ready to send to the server.
   *
   * @param {RTCSessionDescriptionInit} offerSdp – the {type,sdp} object from the server.
   */
  async start(offerSdp) {
    await this._pc.setRemoteDescription(new RTCSessionDescription(offerSdp));
    const answer = await this._pc.createAnswer();
    await this._pc.setLocalDescription(answer);

    // Wait for ICE gathering so all candidates are included in the answer SDP.
    await new Promise((resolve) => {
      if (this._pc.iceGatheringState === 'complete') { resolve(); return; }
      this._pc.onicegatheringstatechange = () => {
        if (this._pc.iceGatheringState === 'complete') resolve();
      };
      // Safety timeout (2 s) so we don't stall on unusual network setups.
      setTimeout(resolve, 2000);
    });

    return this._pc.localDescription;
  }

  /** Resolves once the peer connection reaches the 'connected' state. */
  waitForConnected() {
    return new Promise((resolve) => {
      if (this._pc.connectionState === 'connected') { resolve(); return; }
      this._pc.onconnectionstatechange = () => {
        if (this._pc.connectionState === 'connected') resolve();
      };
    });
  }

  get connectionState() { return this._pc.connectionState; }
}

// ─── PoseSender ───────────────────────────────────────────────────────────────
// Reads the XRFrame viewer pose and sends it as JSON over the WebRTC data channel.
// Wire format: {"position":[x,y,z],"orientation":[x,y,z,w],"timestamp":ms}

class PoseSender {
  constructor(dataChannel) {
    this._dc = dataChannel;
  }

  /**
   * @param {XRFrame} frame
   * @param {XRReferenceSpace} refSpace
   * @param {number} timeMs
   * @param {XRWebGLLayer} glLayer
   * @param {{x,y,z}|null} [virtualPos]  Override position to send (virtual camera after locomotion).
   * @param {{x,y,z,w}|null} [virtualOri] Override orientation to send (virtual camera after locomotion).
   */
  send(frame, refSpace, timeMs, glLayer, virtualPos, virtualOri) {
    if (this._dc?.readyState !== 'open') return;
    const pose = frame.getViewerPose(refSpace);
    if (!pose) return;
    // Use virtual camera pos/ori when provided; fall back to raw pose for non-locomotion setups.
    const { x: px, y: py, z: pz } = virtualPos ?? pose.transform.position;
    const { x: qx, y: qy, z: qz, w: qw } = virtualOri ?? pose.transform.orientation;

    let ipd = null;
    let proj_left = null, proj_right = null;
    let eye_width = null, eye_height = null;
    const lv = pose.views.find(v => v.eye === 'left');
    const rv = pose.views.find(v => v.eye === 'right');
    if (lv && rv) {
      const lp = lv.transform.position;
      const rp = rv.transform.position;
      const dx = rp.x - lp.x, dy = rp.y - lp.y, dz = rp.z - lp.z;
      ipd = Math.sqrt(dx * dx + dy * dy + dz * dz);
      proj_left  = lv.projectionMatrix;
      proj_right = rv.projectionMatrix;
      if (glLayer) {
        const vp = glLayer.getViewport(lv);
        if (vp) { eye_width = vp.width; eye_height = vp.height; }
      }
    }

    this._dc.send(PoseSender._encodeBinary(
      px, py, pz, qx, qy, qz, qw, Math.round(timeMs),
      ipd, proj_left, proj_right, eye_width, eye_height,
    ));
  }

  /**
   * Encode a pose to the compact binary wire format.
   * magic(1) + version(1) + flags(2) + pos(12) + orient(16) + ts(8) [+ optional]
   * Flags: bit0=ipd, bit1=proj, bit2=eye_dims
   * @returns {ArrayBuffer}
   */
  static _encodeBinary(px, py, pz, qx, qy, qz, qw, tsMs,
                        ipd, proj_left, proj_right, eye_w, eye_h) {
    const hasIpd      = ipd      != null;
    const hasProj     = proj_left != null && proj_right != null;
    const hasEyeDims  = eye_w    != null && eye_h != null;

    let size = 40;
    if (hasIpd)     size +=   4;
    if (hasProj)    size += 128;
    if (hasEyeDims) size +=   8;

    const buf = new ArrayBuffer(size);
    const dv  = new DataView(buf);
    const LE  = true;

    let flags = 0;
    if (hasIpd)     flags |= 1;
    if (hasProj)    flags |= 2;
    if (hasEyeDims) flags |= 4;

    dv.setUint8(0, 0x50);   // magic 'P'
    dv.setUint8(1, 0x01);   // version
    dv.setUint16(2, flags, LE);
    dv.setFloat32( 4, px, LE);
    dv.setFloat32( 8, py, LE);
    dv.setFloat32(12, pz, LE);
    dv.setFloat32(16, qx, LE);
    dv.setFloat32(20, qy, LE);
    dv.setFloat32(24, qz, LE);
    dv.setFloat32(28, qw, LE);
    // timestamp as two u32s (DataView lacks setUint64)
    dv.setUint32(32, tsMs >>> 0,           LE);
    dv.setUint32(36, Math.floor(tsMs / 2**32), LE);

    let off = 40;
    if (hasIpd) {
      dv.setFloat32(off, ipd, LE); off += 4;
    }
    if (hasProj) {
      for (let i = 0; i < 16; i++) { dv.setFloat32(off, proj_left[i],  LE); off += 4; }
      for (let i = 0; i < 16; i++) { dv.setFloat32(off, proj_right[i], LE); off += 4; }
    }
    if (hasEyeDims) {
      dv.setUint32(off, eye_w, LE); off += 4;
      dv.setUint32(off, eye_h, LE);
    }
    return buf;
  }
}

// ─── XRRenderer ───────────────────────────────────────────────────────────────
// Renders the server's side-by-side stereo video frame into the XR framebuffer.
//
// The server produces a 2048×1024 texture: left eye in the left half, right eye
// in the right half.  Each XRView has its own viewport from XRWebGLLayer, so we
// only need to control which horizontal half of the video we sample.

const VERT = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  // NDC [-1,1] → UV [0,1].  Flip Y so top of video maps to top of screen.
  vUV = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `
precision mediump float;
uniform sampler2D uVideo;
uniform float uXOff;      // 0.0 = left eye, 0.5 = right eye
uniform float uAtwOffset; // horizontal UV shift for async timewarp (per-eye space)
varying vec2 vUV;
void main() {
  // Apply ATW horizontal correction then map into the correct eye half.
  float u = clamp(vUV.x + uAtwOffset, 0.0, 1.0);
  gl_FragColor = texture2D(uVideo, vec2(u * 0.5 + uXOff, vUV.y));
}`;

class XRRenderer {
  /**
   * @param {WebGLRenderingContext} gl
   * @param {HTMLVideoElement} videoElement
   * @param {[number,number,number,number]} clearColor  RGBA clear color, default light gray-blue.
   */
  constructor(gl, videoElement, clearColor = [0.18, 0.18, 0.22, 1.0]) {
    this._gl = gl;
    this._video = videoElement;
    this._clearColor = clearColor;
    this._prog = null;
    this._buf = null;
    this._tex = null;
    this._aPos = -1;
    this._uVideo = null;
    this._uXOff = null;
    this._uAtwOffset = null;
  }

  init() {
    const gl = this._gl;

    const vs = this._shader(gl.VERTEX_SHADER, VERT);
    const fs = this._shader(gl.FRAGMENT_SHADER, FRAG);
    this._prog = gl.createProgram();
    gl.attachShader(this._prog, vs);
    gl.attachShader(this._prog, fs);
    gl.linkProgram(this._prog);

    this._aPos = gl.getAttribLocation(this._prog, 'aPos');
    this._uVideo = gl.getUniformLocation(this._prog, 'uVideo');
    this._uXOff = gl.getUniformLocation(this._prog, 'uXOff');
    this._uAtwOffset = gl.getUniformLocation(this._prog, 'uAtwOffset');

    // Full-screen quad as two triangles (TRIANGLE_STRIP).
    this._buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]), gl.STATIC_DRAW);

    // Video texture — seed with a 1×1 magenta pixel so the quad is visible
    // even before the first decoded video frame arrives.
    this._tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 0, 255, 255]));

    this._frameCount = 0;
  }

  _shader(type, src) {
    const gl = this._gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('Shader error: ' + gl.getShaderInfoLog(s));
    }
    return s;
  }

  /**
   * Draw one XR frame.
   * @param {XRWebGLLayer} glLayer
   * @param {readonly XRView[]} views
   * @param {number} [atwOffset=0] Per-eye horizontal UV offset for async timewarp.
   */
  drawFrame(glLayer, views, atwOffset = 0.0) {
    const gl = this._gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);

    // Clear color — light gray-blue, visible before the video stream arrives.
    // Overridden to (0,0,0,0) in AR/passthrough mode for transparent borders.
    gl.clearColor(...this._clearColor);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Upload the latest decoded video frame to the GPU texture.
    const videoReady = this._video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    if (videoReady) {
      gl.bindTexture(gl.TEXTURE_2D, this._tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this._video);
    }

    // Log video state once per second (first 10 s) to help diagnose blank frames.
    this._frameCount++;
    if (this._frameCount % 72 === 1 && this._frameCount < 720) {
      console.log(`[XRRenderer] frame=${this._frameCount} video.readyState=${this._video.readyState} videoReady=${videoReady} videoSize=${this._video.videoWidth}×${this._video.videoHeight}`);
    }

    gl.useProgram(this._prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buf);
    gl.enableVertexAttribArray(this._aPos);
    gl.vertexAttribPointer(this._aPos, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._tex);
    gl.uniform1i(this._uVideo, 0);
    gl.uniform1f(this._uAtwOffset, atwOffset);

    for (const view of views) {
      const vp = glLayer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      // Left eye → left half of the side-by-side frame (uXOff = 0.0).
      // Right eye → right half (uXOff = 0.5).
      gl.uniform1f(this._uXOff, view.eye === 'right' ? 0.5 : 0.0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }
}

// ─── ControllerRenderer ───────────────────────────────────────────────────────
// Draws a simple elongated box at each VR controller's grip-space position.
// Rendered after the full-screen video quad with depth test disabled so the
// controllers always appear in front regardless of the video quad's Z.

const CTRL_VERT = `
attribute vec3 aPos;
uniform mat4 uMVP;
uniform float uScale;
void main() {
  gl_Position = uMVP * vec4(aPos * uScale, 1.0);
}`;

const CTRL_FRAG = `
precision mediump float;
uniform vec3 uColor;
void main() {
  gl_FragColor = vec4(uColor, 1.0);
}`;

class ControllerRenderer {
  constructor(gl) {
    this._gl = gl;
    this._prog = null;
    this._vbuf = null;
    this._ibuf = null;
    this._indexCount = 0;
    this._jointVbuf = null;
    this._jointIbuf = null;
    this._jointIndexCount = 0;
    this._aPos = -1;
    this._uMVP = null;
    this._uColor = null;
    this._uScale = null;
  }

  init() {
    const gl = this._gl;

    const vs = this._compileShader(gl.VERTEX_SHADER,   CTRL_VERT);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, CTRL_FRAG);
    this._prog = gl.createProgram();
    gl.attachShader(this._prog, vs);
    gl.attachShader(this._prog, fs);
    gl.linkProgram(this._prog);

    this._aPos   = gl.getAttribLocation(this._prog, 'aPos');
    this._uMVP   = gl.getUniformLocation(this._prog, 'uMVP');
    this._uColor = gl.getUniformLocation(this._prog, 'uColor');
    this._uScale = gl.getUniformLocation(this._prog, 'uScale');

    // Box half-extents: 1.5 cm × 1.5 cm × 6 cm.
    // The long axis runs along −Z (grip-space "forward").
    const w = 0.015, h = 0.015, d = 0.06;
    const verts = new Float32Array([
      // Front face (+Z)
      -w, -h,  d,   w, -h,  d,   w,  h,  d,  -w,  h,  d,
      // Back  face (−Z)
      -w, -h, -d,   w, -h, -d,   w,  h, -d,  -w,  h, -d,
    ]);
    const idx = new Uint16Array([
      0,1,2, 0,2,3,   // front
      4,6,5, 4,7,6,   // back
      3,2,6, 3,6,7,   // top
      0,5,1, 0,4,5,   // bottom
      1,5,6, 1,6,2,   // right
      0,3,7, 0,7,4,   // left
    ]);
    this._indexCount = idx.length;

    this._vbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._vbuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    this._ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._ibuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

    // Unit cube (half-extent 1) reused for hand joints — scaled per-joint by
    // uScale (set to the joint radius reported by XRJointPose) at draw time.
    const jw = 1, jh = 1, jd = 1;
    const jointVerts = new Float32Array([
      -jw, -jh,  jd,   jw, -jh,  jd,   jw,  jh,  jd,  -jw,  jh,  jd,
      -jw, -jh, -jd,   jw, -jh, -jd,   jw,  jh, -jd,  -jw,  jh, -jd,
    ]);
    this._jointIndexCount = idx.length;

    this._jointVbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._jointVbuf);
    gl.bufferData(gl.ARRAY_BUFFER, jointVerts, gl.STATIC_DRAW);

    this._jointIbuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._jointIbuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  }

  _compileShader(type, src) {
    const gl = this._gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('Controller shader: ' + gl.getShaderInfoLog(s));
    }
    return s;
  }

  /**
   * Draw one box per active controller input source.
   * Skips hand-tracking sources — those have no grip geometry.
   *
   * @param {XRFrame}          frame
   * @param {XRSession}        session
   * @param {XRReferenceSpace} refSpace
   * @param {XRWebGLLayer}     glLayer
   * @param {readonly XRView[]} views
   */
  drawControllers(frame, session, refSpace, glLayer, views) {
    const sources = Array.from(session.inputSources).filter(
      src => src.gripSpace && !src.hand,
    );
    if (sources.length === 0) return;

    const gl = this._gl;
    // Disable depth test so controllers are always visible above the video quad.
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this._prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._vbuf);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._ibuf);
    gl.enableVertexAttribArray(this._aPos);
    gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
    gl.uniform1f(this._uScale, 1.0);

    for (const view of views) {
      const vp = glLayer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);

      // view.transform.matrix is eye-to-world; invert to get world-to-eye.
      const viewMat = mat4InverseRigid(view.transform.matrix);
      const projMat = view.projectionMatrix;

      for (const src of sources) {
        const gripPose = frame.getPose(src.gripSpace, refSpace);
        if (!gripPose) continue;

        // MVP = projection × view × model  (model = grip-space-to-world).
        const mvp = mat4Mul(projMat, mat4Mul(viewMat, gripPose.transform.matrix));
        gl.uniformMatrix4fv(this._uMVP, false, mvp);

        // Left hand: cool blue; right hand: warm red.
        if (src.handedness === 'left') {
          gl.uniform3f(this._uColor, 0.35, 0.55, 0.95);
        } else {
          gl.uniform3f(this._uColor, 0.95, 0.35, 0.35);
        }

        gl.drawElements(gl.TRIANGLES, this._indexCount, gl.UNSIGNED_SHORT, 0);
      }
    }

    gl.enable(gl.DEPTH_TEST);
  }

  /**
   * Draw a small cube at each tracked joint of every hand-tracking input
   * source, sized to the joint's reported radius. This is the hand-tracking
   * counterpart to drawControllers() — hand sources have no gripSpace, so
   * they're skipped there and rendered here instead.
   *
   * @param {XRFrame}          frame
   * @param {XRSession}        session
   * @param {XRReferenceSpace} refSpace
   * @param {XRWebGLLayer}     glLayer
   * @param {readonly XRView[]} views
   */
  drawHandJoints(frame, session, refSpace, glLayer, views) {
    const handSources = Array.from(session.inputSources).filter(src => src.hand);
    if (handSources.length === 0) return;

    const gl = this._gl;
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this._prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._jointVbuf);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._jointIbuf);
    gl.enableVertexAttribArray(this._aPos);
    gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);

    // Gather joint poses once per frame (not per eye) — same pose is reused
    // for both stereo views below.
    const joints = [];
    for (const src of handSources) {
      const color = src.handedness === 'left'
        ? [0.35, 0.55, 0.95]
        : [0.95, 0.35, 0.35];
      for (const jointSpace of src.hand.values()) {
        const jointPose = frame.getJointPose(jointSpace, refSpace);
        if (!jointPose) continue;
        joints.push({ matrix: jointPose.transform.matrix, radius: jointPose.radius || 0.008, color });
      }
    }
    if (joints.length === 0) {
      gl.enable(gl.DEPTH_TEST);
      return;
    }

    for (const view of views) {
      const vp = glLayer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);

      const viewMat = mat4InverseRigid(view.transform.matrix);
      const projMat = view.projectionMatrix;

      for (const joint of joints) {
        const mvp = mat4Mul(projMat, mat4Mul(viewMat, joint.matrix));
        gl.uniformMatrix4fv(this._uMVP, false, mvp);
        gl.uniform1f(this._uScale, joint.radius);
        gl.uniform3f(this._uColor, joint.color[0], joint.color[1], joint.color[2]);
        gl.drawElements(gl.TRIANGLES, this._jointIndexCount, gl.UNSIGNED_SHORT, 0);
      }
    }

    gl.enable(gl.DEPTH_TEST);
  }
}

// ─── Channel config ───────────────────────────────────────────────────────────

/** Default display colors for channels (matches server DEFAULT_CHANNEL_COLORS). */
const CH_DEFAULTS = [
  { hex: '#ffffff', opacity: 1.0 },
  { hex: '#00ff00', opacity: 1.0 },
  { hex: '#ff00ff', opacity: 1.0 },
  { hex: '#00ffff', opacity: 1.0 },
];

/**
 * Build the per-channel color/opacity controls and show the panel.
 * @param {number} count  Number of channels (1–4).
 */
function buildChannelControls(count) {
  if (count <= 1) return; // single-channel: server defaults are fine, no UI needed

  const container = document.getElementById('channel-controls');
  container.innerHTML = '';

  const labels = ['Ch 1', 'Ch 2', 'Ch 3', 'Ch 4'];
  for (let i = 0; i < count; i++) {
    const def = CH_DEFAULTS[i] || { hex: '#ffffff', opacity: 1.0 };
    const row = document.createElement('div');
    row.className = 'ch-row';

    const label = document.createElement('label');
    label.textContent = labels[i];

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = def.hex;
    colorInput.dataset.ch = i;
    colorInput.addEventListener('input', sendChannelConfig);

    const opacityInput = document.createElement('input');
    opacityInput.type = 'range';
    opacityInput.min = 0; opacityInput.max = 100;
    opacityInput.value = Math.round(def.opacity * 100);
    opacityInput.dataset.ch = i;
    opacityInput.addEventListener('input', () => {
      valSpan.textContent = opacityInput.value + '%';
      sendChannelConfig();
    });

    const valSpan = document.createElement('span');
    valSpan.className = 'ch-opacity-val';
    valSpan.textContent = Math.round(def.opacity * 100) + '%';

    row.appendChild(label);
    row.appendChild(colorInput);
    row.appendChild(opacityInput);
    row.appendChild(valSpan);
    container.appendChild(row);
  }

  document.getElementById('channel-section').style.display = 'block';
  // Send initial defaults so the server reflects the UI state on connect.
  sendChannelConfig();
}

/**
 * Encode current channel control values into binary and send on the data channel.
 *
 * Wire format (little-endian):
 *   [0]     magic   = 0x43 ('C')
 *   [1]     version = 0x01
 *   [2]     count   (u8)
 *   [3+i*16 .. 3+i*16+16]  channel i: f32×4 [r, g, b, opacity]
 */
function sendChannelConfig() {
  if (!pc || !pc.dataChannel || pc.dataChannel.readyState !== 'open') return;

  const colorInputs  = document.querySelectorAll('#channel-controls input[type="color"]');
  const opacityInputs = document.querySelectorAll('#channel-controls input[type="range"]');
  const count = colorInputs.length;
  if (count === 0) return;

  const buf = new ArrayBuffer(3 + count * 16);
  const u8  = new Uint8Array(buf);
  const f32 = new DataView(buf);

  u8[0] = 0x43; // 'C' magic
  u8[1] = 0x01; // version
  u8[2] = count;

  for (let i = 0; i < count; i++) {
    const hex = colorInputs[i].value; // '#rrggbb'
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const opacity = parseInt(opacityInputs[i].value) / 100;
    const base = 3 + i * 16;
    f32.setFloat32(base,      r,       true);
    f32.setFloat32(base + 4,  g,       true);
    f32.setFloat32(base + 8,  b,       true);
    f32.setFloat32(base + 12, opacity, true);
  }

  pc.dataChannel.send(buf);
}

// ─── Application ─────────────────────────────────────────────────────────────

/** @type {PeerConn|null} */
let pc = null;
/** @type {Signaling|null} */
let sig = null;
/** @type {PoseSender|null} */
let poseSender = null;
/** @type {XRSession|null} */
let xrSession = null;

// ─── Locomotion ───────────────────────────────────────────────────────────────
// All locomotion is resolved client-side by shifting the XRReferenceSpace
// origin each frame.  The resulting head pose that the server receives already
// reflects the player's world position — no server changes required.
//
// Physical (room-scale) walking is handled transparently by the 'local'
// reference space; the accumulated offset is simply additive on top of it.

const LOCO = {
  moveSpeed: 2.0,   // m/s forward / back / strafe
  turnSpeed: 1.2,   // rad/s yaw from right thumbstick
  deadzone:  0.12,  // thumbstick dead-zone (0–1 fraction of full range)
  offset: { x: 0, y: 0, z: 0 },  // accumulated world-space translation
  yaw: 0,                          // accumulated player yaw (radians)
};

/** Timestamp of the previous frame, used to compute dtSec. */
let _lastFrameTimeMs = 0;

/** FPS counter state. */
let _fpsFrameCount = 0;
let _fpsWindowStart = 0;

// ─── Locomotion math helpers ──────────────────────────────────────────────────

/** Build a unit quaternion for a pure yaw (Y-axis) rotation. */
function eulerYToQuat(yawRad) {
  const half = yawRad / 2;
  return { x: 0, y: Math.sin(half), z: 0, w: Math.cos(half) };
}

// ── Async Timewarp (ATW) helpers ──────────────────────────────────────────────

/**
 * Multiply two quaternions: returns `a * b`.
 * @param {{x,y,z,w}} a
 * @param {{x,y,z,w}} b
 * @returns {{x,y,z,w}}
 */
function quatMul(a, b) {
  return {
    x:  a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
    y:  a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
    z:  a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    w:  a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
  };
}

/**
 * Conjugate (= inverse for unit quaternion).
 * @param {{x,y,z,w}} q
 * @returns {{x,y,z,w}}
 */
function quatConj(q) {
  return { x: -q.x, y: -q.y, z: -q.z, w: q.w };
}

/**
 * Rotate a vector by a unit quaternion.
 * Uses the optimised form: v' = v + 2w(q×v) + 2(q×(q×v))
 * @param {{x,y,z,w}} q
 * @param {{x,y,z}} v
 * @returns {{x,y,z}}
 */
function quatRotateVec(q, v) {
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + q.y * tz - q.z * ty,
    y: v.y + q.w * ty + q.z * tx - q.x * tz,
    z: v.z + q.w * tz + q.x * ty - q.y * tx,
  };
}

/**
 * Compute the horizontal (yaw) UV offset that compensates for the rotation
 * between the rendered pose and the current head pose.
 *
 * The correction shifts the per-eye UV sample point so that objects appear
 * where the eye currently expects them, reducing perceived latency.
 *
 * @param {PeerConn|null} pc
 * @param {{x,y,z,w}|null} virtualOri  The virtual camera orientation sent to the server this frame.
 * @param {readonly XRView[]} views     Per-eye views (for projection matrix / FOV).
 * @returns {number} UV offset in per-eye space, clamped to ±0.25.
 */
function computeAtwOffset(pc, virtualOri, views) {
  if (!pc || !pc.lastRenderedOrientation || !virtualOri || !views) return 0.0;

  const rendered = pc.lastRenderedOrientation;
  const current  = virtualOri;

  // delta = q_current * q_rendered_inverse
  // Represents the rotation applied since the frame was rendered.
  const delta = quatMul(current, quatConj(rendered));

  // Extract the Y-axis (yaw) component of the delta rotation.
  // For a unit quaternion, 2*atan2(q.y, q.w) gives the rotation angle around Y.
  // Clamping delta.w avoids atan2(0,0) when delta is near identity.
  const deltaYaw = 2.0 * Math.atan2(delta.y, Math.max(Math.abs(delta.w), 1e-6) * Math.sign(delta.w || 1));

  // Find the horizontal FOV from the left eye's projection matrix.
  // projectionMatrix[0] = 1/tan(fovX/2) for a rectilinear projection.
  const leftView = views.find(v => v.eye === 'left') || views[0];
  if (!leftView) return 0.0;
  const m = leftView.projectionMatrix;
  if (!m || Math.abs(m[0]) < 1e-6) return 0.0;
  const fovX = 2.0 * Math.atan(1.0 / Math.abs(m[0]));

  // UV offset: negative deltaYaw because turning right (positive Y rotation)
  // means the rendered content is to the left — shift sampling left (negative u).
  const offset = -deltaYaw / fovX;
  return Math.max(-0.25, Math.min(0.25, offset));
}

/** Euclidean distance between two {x,y,z} / DOMPointReadOnly objects. */
function dist3(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ─── Matrix helpers ───────────────────────────────────────────────────────────

/**
 * Multiply two 4×4 column-major matrices.
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {Float32Array}
 */
function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = s;
    }
  }
  return out;
}

/**
 * Invert a 4×4 rigid-body matrix (pure rotation + translation).
 * R⁻¹ = Rᵀ, t⁻¹ = −Rᵀ·t  — exact and cheaper than a general inverse.
 * @param {ArrayLike<number>} m  Column-major input.
 * @returns {Float32Array}
 */
function mat4InverseRigid(m) {
  const out = new Float32Array(16);
  // Transpose the 3×3 rotation block.
  out[0] = m[0]; out[4] = m[1]; out[8]  = m[2];
  out[1] = m[4]; out[5] = m[5]; out[9]  = m[6];
  out[2] = m[8]; out[6] = m[9]; out[10] = m[10];
  // −Rᵀ · t
  out[12] = -(out[0]*m[12] + out[4]*m[13] + out[8] *m[14]);
  out[13] = -(out[1]*m[12] + out[5]*m[13] + out[9] *m[14]);
  out[14] = -(out[2]*m[12] + out[6]*m[13] + out[10]*m[14]);
  out[15] = 1;
  return out;
}

// ─── Joystick locomotion ──────────────────────────────────────────────────────

/**
 * Poll all XR input sources for thumbstick axes and accumulate movement.
 *
 * Axis layout (Oculus/Quest convention — most common):
 *   axes[0..1] = touchpad (ignored), axes[2..3] = thumbstick
 * Fallback for single-stick controllers: axes[0..1].
 *
 * Left hand  → translate (strafe + forward/back)
 * Right hand → yaw (turn)
 *
 * @param {number} dtSec  Seconds since the last frame.
 */
function pollGamepad(dtSec) {
  if (!xrSession) return;
  for (const src of xrSession.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;

    const ax = (i, fallback) => {
      const v = gp.axes[i] ?? gp.axes[fallback] ?? 0;
      return Math.abs(v) > LOCO.deadzone ? v : 0;
    };

    if (src.handedness === 'left') {
      // LOCO.offset is subtracted (not added) from the physical position to
      // produce the virtual camera pose, which inverts the usual stick→camera
      // relationship — negate both raw axes to compensate.
      const strafe  = -ax(2, 0);
      const forward = -ax(3, 1);
      if (strafe || forward) applyMove(strafe, forward, dtSec);
    } else if (src.handedness === 'right') {
      const turn = ax(2, 0);
      if (turn) LOCO.yaw -= turn * LOCO.turnSpeed * dtSec;
    }
  }
}

/**
 * Translate the accumulated offset in the direction the player is facing.
 *
 * WebXR coordinate system: +X right, +Y up, -Z forward.
 * stickForward is positive when the stick is pushed forward (away from player);
 * we negate it to produce motion in the -Z direction.
 *
 * @param {number} stickStrafe  Left/right axis value (-1…1).
 * @param {number} stickForward Forward/back axis value (-1…1, up = positive).
 * @param {number} dtSec
 */
function applyMove(stickStrafe, stickForward, dtSec) {
  const sin  = Math.sin(LOCO.yaw);
  const cos  = Math.cos(LOCO.yaw);
  const dist = LOCO.moveSpeed * dtSec;
  // Project stick axes into world XZ using the current yaw.
  LOCO.offset.x += (-sin * (-stickForward) + cos * stickStrafe) * dist;
  LOCO.offset.z += (-cos * (-stickForward) - sin * stickStrafe) * dist;
}

// ─── Hand-gesture locomotion ──────────────────────────────────────────────────

/** Thumb-tip to index-tip distance threshold for the pinch gesture (metres). */
const PINCH_THRESHOLD = 0.03;

/** Per-hand pinch-drag state, keyed by handedness ('left' | 'right'). */
const _pinchState = { left: null, right: null };

/**
 * Detect a pinch-and-drag "grab the world" gesture using the WebXR Hand
 * Tracking API.
 *
 * While pinching (thumb tip within PINCH_THRESHOLD of index tip), the point
 * midway between the fingertips is anchored to the world: moving the
 * physical hand drags the volume along with it, like grabbing a rope and
 * pulling. Releasing the pinch drops the anchor; pinching again starts a
 * fresh drag from wherever the fingertips are.
 *
 * Falls through silently when hand tracking is unavailable (src.hand === null),
 * so joystick locomotion continues to work on controller-only setups.
 *
 * @param {XRFrame} frame
 * @param {XRReferenceSpace} baseRefSpace  Unshifted tracking origin.
 */
function pollHandGestures(frame, baseRefSpace) {
  if (!xrSession) return;

  const seenHands = new Set();

  for (const src of xrSession.inputSources) {
    if (!src.hand || !src.handedness) continue;
    seenHands.add(src.handedness);

    const thumbTip = frame.getJointPose(src.hand.get('thumb-tip'),        baseRefSpace);
    const idxTip   = frame.getJointPose(src.hand.get('index-finger-tip'), baseRefSpace);
    if (!thumbTip || !idxTip) continue;

    const pinchPos = {
      x: (thumbTip.transform.position.x + idxTip.transform.position.x) / 2,
      y: (thumbTip.transform.position.y + idxTip.transform.position.y) / 2,
      z: (thumbTip.transform.position.z + idxTip.transform.position.z) / 2,
    };
    const pinching = dist3(thumbTip.transform.position, idxTip.transform.position)
                     < PINCH_THRESHOLD;
    const state = _pinchState[src.handedness];

    if (pinching && !state) {
      // Pinch just started — anchor the drag at the current fingertip midpoint.
      _pinchState[src.handedness] = { prevPos: pinchPos };
    } else if (pinching && state) {
      // Dragging — shift the world by the same physical delta the hand moved,
      // so the grabbed point stays under the fingertips (pan-style drag).
      LOCO.offset.x += pinchPos.x - state.prevPos.x;
      LOCO.offset.y += pinchPos.y - state.prevPos.y;
      LOCO.offset.z += pinchPos.z - state.prevPos.z;
      state.prevPos = pinchPos;
    } else if (!pinching && state) {
      _pinchState[src.handedness] = null;
    }
  }

  // Drop stale state for hands that dropped out of tracking this frame.
  for (const hand of ['left', 'right']) {
    if (!seenHands.has(hand)) _pinchState[hand] = null;
  }
}

function setStatus(msg, id = 'status') {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

/** Called when the user clicks Connect. */
async function connect() {
  const input = document.getElementById('code-input');
  const code = input.value.trim();
  if (!/^\d{6}$/.test(code)) { setStatus('Enter a 6-digit numeric code.'); return; }

  const btn = document.getElementById('connect-btn');
  btn.disabled = true;
  setStatus('Opening connection…');

  try {
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/signal`;
    sig = new Signaling(wsUrl);
    await sig.open();

    // Step 1 — pair
    sig.send({ type: 'pair', code });
    setStatus('Pairing…');

    // Step 2 — wait for offer or error
    const serverMsg = await sig.recv();
    if (serverMsg.type === 'error') {
      setStatus('Rejected: ' + serverMsg.msg);
      btn.disabled = false;
      return;
    }
    if (serverMsg.type !== 'offer') {
      setStatus('Unexpected message from server: ' + serverMsg.type);
      btn.disabled = false;
      return;
    }

    setStatus('Got offer, negotiating WebRTC…');
    const channelCount = serverMsg.channel_count || 1;

    // Step 3 — create peer connection and answer
    pc = new PeerConn();
    const localDesc = await pc.start(serverMsg.sdp);

    // Step 4 — send answer
    sig.send({ type: 'answer', sdp: localDesc });
    setStatus('Waiting for connection…');

    // Step 5 — wait for connected (or 15 s timeout)
    await Promise.race([
      pc.waitForConnected(),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Connection timed out')), 15000)),
    ]);

    setStatus('');
    document.getElementById('pair-section').style.display = 'none';
    document.getElementById('xr-section').style.display = 'block';
    updateARButtonVisibility();
    buildChannelControls(channelCount);

    // Wire up pose sender once data channel is ready.
    if (pc.dataChannel) {
      poseSender = new PoseSender(pc.dataChannel);
    } else {
      pc._pc.ondatachannel = (ev) => {
        pc.dataChannel = ev.channel;
        poseSender = new PoseSender(ev.channel);
      };
    }

  } catch (err) {
    setStatus('Error: ' + err.message);
    btn.disabled = false;
  }
}

/** Called when the user clicks Enter VR. */
async function enterXR() {
  await startXRSession('immersive-vr');
}

/** Called when the user clicks Enter AR (Passthrough). */
async function enterAR() {
  await startXRSession('immersive-ar');
}

/**
 * Start an XR session in the given mode ('immersive-vr' or 'immersive-ar').
 * Shared implementation for both VR and AR entry paths.
 * @param {'immersive-vr'|'immersive-ar'} mode
 */
async function startXRSession(mode) {
  const vrBtn = document.getElementById('xr-btn');
  const arBtn = document.getElementById('ar-btn');

  if (!navigator.xr) {
    setStatus('WebXR is not available (try Quest Browser or enable flags).', 'status2');
    return;
  }

  const supported = await navigator.xr.isSessionSupported(mode).catch(() => false);
  if (!supported) {
    setStatus(`${mode} not supported on this device.`, 'status2');
    return;
  }

  try {
    xrSession = await navigator.xr.requestSession(mode, {
      optionalFeatures: ['gamepad', 'hand-tracking'],
    });
  } catch (err) {
    setStatus('Could not start XR session: ' + err.message, 'status2');
    return;
  }

  if (vrBtn) vrBtn.disabled = true;
  if (arBtn) arBtn.disabled = true;

  // In AR mode use a transparent clear color so the real-world passthrough
  // shows around the edges of the video quad.  (The video content itself is
  // opaque — H.264 has no alpha — so passthrough through the volume requires
  // a future codec change.)
  const isAR = mode === 'immersive-ar';
  const clearColor = isAR ? [0, 0, 0, 0] : [0.18, 0.18, 0.22, 1.0];

  // Create a WebGL context that is compatible with the XR device.
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { xrCompatible: true, alpha: isAR });
  if (!gl) {
    setStatus('WebGL not available.', 'status2');
    xrSession.end();
    if (vrBtn) vrBtn.disabled = false;
    if (arBtn) arBtn.disabled = false;
    return;
  }

  const renderer = new XRRenderer(gl, pc.videoElement, clearColor);
  renderer.init();

  const controllerRenderer = new ControllerRenderer(gl);
  controllerRenderer.init();

  const glLayer = new XRWebGLLayer(xrSession, gl, { alpha: isAR });
  xrSession.updateRenderState({ baseLayer: glLayer });

  // baseRefSpace is the immutable tracking origin — never modified.
  // Locomotion is applied to the sent pose in JS rather than shifting this space,
  // so the volume stays fixed in the user's physical environment.
  let baseRefSpace;
  try {
    baseRefSpace = await xrSession.requestReferenceSpace('local');
  } catch {
    baseRefSpace = await xrSession.requestReferenceSpace('viewer');
  }

  // Reset locomotion and FPS state when a new XR session starts.
  LOCO.offset.x = 0; LOCO.offset.y = 0; LOCO.offset.z = 0;
  LOCO.yaw = 0;
  _pinchState.left = null; _pinchState.right = null;
  _lastFrameTimeMs = 0;
  _fpsFrameCount = 0;
  _fpsWindowStart = 0;

  xrSession.addEventListener('end', () => {
    xrSession = null;
    setStatus('XR session ended.', 'status2');
    if (vrBtn) vrBtn.disabled = false;
    if (arBtn) arBtn.disabled = false;
  });

  function onXRFrame(timeMs, frame) {
    xrSession.requestAnimationFrame(onXRFrame);

    // Delta time — capped at 50 ms to prevent large jumps on tab-focus-restore.
    const dtSec = Math.min((_lastFrameTimeMs > 0 ? timeMs - _lastFrameTimeMs : 0) / 1000, 0.05);
    _lastFrameTimeMs = timeMs;

    // FPS counter — update the overlay once per second.
    _fpsFrameCount++;
    if (_fpsWindowStart === 0) _fpsWindowStart = timeMs;
    const fpsElapsed = timeMs - _fpsWindowStart;
    if (fpsElapsed >= 1000) {
      const fps = (_fpsFrameCount * 1000 / fpsElapsed).toFixed(1);
      document.getElementById('fps-counter').textContent = fps + ' fps';
      _fpsFrameCount = 0;
      _fpsWindowStart = timeMs;
    }

    // Poll locomotion inputs and accumulate into LOCO.offset / LOCO.yaw.
    pollGamepad(dtSec);
    pollHandGestures(frame, baseRefSpace);

    // Get the physical head pose from the immutable tracking space.
    const physPose = frame.getViewerPose(baseRefSpace);
    if (!physPose) return;

    // Compose locomotion on top of the physical head pose to produce the
    // virtual camera pose sent to the server.  This keeps baseRefSpace fixed
    // (volume stays at its world-space origin) while still letting the user
    // fly through it via thumbstick / pinch gesture.
    //
    // Mirrors what getOffsetReferenceSpace(T) does mathematically:
    //   virtual_pos = rot_loco_inv * (phys_pos - loco_offset)
    //   virtual_ori = rot_loco_inv * phys_ori
    const locoQuat    = eulerYToQuat(LOCO.yaw);
    const locoQuatInv = quatConj(locoQuat);
    const dp = {
      x: physPose.transform.position.x - LOCO.offset.x,
      y: physPose.transform.position.y - LOCO.offset.y,
      z: physPose.transform.position.z - LOCO.offset.z,
    };
    const virtualPos = quatRotateVec(locoQuatInv, dp);
    const virtualOri = quatMul(locoQuatInv, physPose.transform.orientation);

    // Forward the virtual camera pose to the server.
    poseSender?.send(frame, baseRefSpace, timeMs, glLayer, virtualPos, virtualOri);

    // ATW uses virtualOri (what was sent) so the yaw offset cancels correctly.
    const atwOffset = computeAtwOffset(pc, virtualOri, physPose.views);

    // Render the stereo video frame with ATW correction applied.
    renderer.drawFrame(glLayer, physPose.views, atwOffset);

    // Render VR controller meshes and hand-tracking joints on top of the video quad.
    controllerRenderer.drawControllers(frame, xrSession, baseRefSpace, glLayer, physPose.views);
    controllerRenderer.drawHandJoints(frame, xrSession, baseRefSpace, glLayer, physPose.views);
  }

  xrSession.requestAnimationFrame(onXRFrame);
}

/**
 * Show/hide the AR button based on device support.
 * Called once after the XR section becomes visible.
 */
async function updateARButtonVisibility() {
  const arBtn = document.getElementById('ar-btn');
  if (!arBtn || !navigator.xr) return;
  const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  arBtn.style.display = supported ? 'block' : 'none';
}
