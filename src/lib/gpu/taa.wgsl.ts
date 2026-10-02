/**
 * Temporal anti-aliasing and upscaling (WGSL).
 * ---------------------------------------------------------------------------
 * The scene renders at the internal resolution with a sub-pixel jitter that
 * changes every frame (Halton 2, 3). `fsResolve` rebuilds each pixel centre's
 * camera-relative position from the depth buffer and the pixel's unjittered
 * ray, moves it into last frame's view (camera motion only: the world is
 * static apart from wind and the hero, which the colour clamp absorbs),
 * samples last frame's result there (Catmull-Rom), clips that history to the
 * spread of the current 3 × 3 neighbourhood in YCoCg, and blends 10 % of the
 * new sample in. `fsPresent` scales the resolved image to the output size
 * (Catmull-Rom) and sharpens it adaptively (after AMD's CAS).
 *
 * With `f16` (the device has shader-f16) the colour arithmetic runs in half
 * precision — twice the rate, half the registers on this GPU — while the
 * positions, the reprojection and the filters' coordinates stay in f32. Its
 * error (2⁻¹¹ of a value) stays under a fifth of an 8-bit step after the
 * sharpening's gain.
 */

export const TAA_BYTES = 48 * 2 + 16 * 5;

export function taaWGSL(f16: boolean): string {
  return /* wgsl */ `${f16 ? 'enable f16;\nalias H = f16;\nalias H3 = vec3<f16>;' : 'alias H = f32;\nalias H3 = vec3f;'}
struct TaaU {
  basis: mat3x3f,     // current view → world (unjittered)
  prevBasis: mat3x3f, // last frame's view → world
  camDelta: vec4f,    // camera position now − then (m), blend weight of the new sample
  proj: vec4f,        // tan(FOV/2), aspect, near, history valid (0 / 1)
  jitter: vec4f,      // this frame's jitter (NDC), sharpening, unused
  res: vec4f,         // internal width, height, 1/width, 1/height
  outRes: vec4f,      // output width, height, 1/width, 1/height
};
@group(0) @binding(0) var<uniform> t: TaaU;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_2d;
@group(0) @binding(3) var histTex: texture_2d<f32>;
@group(0) @binding(4) var linSamp: sampler;

fn toYCoCg(c: H3) -> H3 {
  return H3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn fromYCoCg(c: H3) -> H3 {
  return H3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
fn tap(tex: texture_2d<f32>, uv: vec2f) -> H3 { return H3(textureSampleLevel(tex, linSamp, uv, 0.0).rgb); }

// Catmull-Rom from 5 bilinear taps (the corner weights are dropped).
fn catmullRom(tex: texture_2d<f32>, uv: vec2f, size: vec2f) -> H3 {
  let p = uv * size;
  let c = floor(p - 0.5) + 0.5;
  let f = p - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let inv = 1.0 / size;
  let t0 = (c - 1.0) * inv;
  let t3 = (c + 2.0) * inv;
  let t12 = (c + o12) * inv;
  let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  var s = tap(tex, vec2f(t12.x, t0.y)) * H(w12.x * w0.y / wsum);
  s += tap(tex, vec2f(t0.x, t12.y)) * H(w0.x * w12.y / wsum);
  s += tap(tex, t12) * H(w12.x * w12.y / wsum);
  s += tap(tex, vec2f(t3.x, t12.y)) * H(w3.x * w12.y / wsum);
  s += tap(tex, vec2f(t12.x, t3.y)) * H(w12.x * w3.y / wsum);
  return max(s, H3(0.0));
}

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fsResolve(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let px = vec2i(fc.xy);
  let maxPx = vec2i(t.res.xy) - 1;
  let cur = H3(textureLoad(sceneTex, px, 0).rgb);
  let curY = toYCoCg(cur);
  var m1 = H3(0.0);
  var m2 = H3(0.0);
  var mn = curY;
  var mx = curY;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let c = toYCoCg(H3(textureLoad(sceneTex, clamp(px + vec2i(dx, dy), vec2i(0), maxPx), 0).rgb));
      m1 += c;
      m2 += c * c;
      mn = min(mn, c);
      mx = max(mx, c);
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, H3(0.0)));
  let lo = max(mn, mean - 1.25 * sigma);
  let hi = min(mx, mean + 1.25 * sigma);

  if (t.proj.w < 0.5) { return vec4f(vec3f(cur), 1.0); }

  // Where this pixel's centre was on screen last frame: the unjittered pixel
  // ray at the depth found here (reprojecting the jittered sample position
  // instead would re-read the history half a pixel off every frame).
  let ndc = vec2f(fc.x * t.res.z * 2.0 - 1.0, 1.0 - fc.y * t.res.w * 2.0);
  let dirView = vec3f(ndc.x * t.proj.x * t.proj.y, ndc.y * t.proj.x, 1.0);
  let d = textureLoad(depthTex, px, 0);
  var prev: vec3f;
  if (d > 0.0) {
    prev = (t.basis * (dirView * (t.proj.z / d)) + t.camDelta.xyz) * t.prevBasis;
  } else {
    prev = (t.basis * dirView) * t.prevBasis; // sky: a direction at infinity
  }
  if (prev.z <= 1e-4) { return vec4f(vec3f(cur), 1.0); }
  let pn = vec2f(prev.x / (prev.z * t.proj.x * t.proj.y), prev.y / (prev.z * t.proj.x));
  let uv = vec2f(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec4f(vec3f(cur), 1.0); }
  let hist = clamp(toYCoCg(catmullRom(histTex, uv, t.res.xy)), lo, hi);
  return vec4f(vec3f(fromYCoCg(mix(hist, curY, H(t.camDelta.w)))), 1.0);
}

@fragment
fn fsPresent(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let uv = fc.xy * t.outRes.zw;
  let c = catmullRom(histTex, uv, t.res.xy);
  // Contrast-adaptive sharpening on the source texel neighbourhood.
  let o = t.res.zw;
  let n = tap(histTex, uv + vec2f(0.0, -o.y));
  let s = tap(histTex, uv + vec2f(0.0, o.y));
  let e = tap(histTex, uv + vec2f(o.x, 0.0));
  let w = tap(histTex, uv + vec2f(-o.x, 0.0));
  let mn = min(min(min(n, s), min(e, w)), c);
  let mx = max(max(max(n, s), max(e, w)), c);
  let amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, H3(1e-4)), H3(0.0), H3(1.0)));
  let k = -amp * H(mix(0.125, 0.2, t.jitter.z) * step(0.001, t.jitter.z));
  let outc = (c + (n + s + e + w) * k) / (1.0 + 4.0 * k);
  return vec4f(vec3f(clamp(outc, H3(0.0), H3(1.0))), 1.0);
}
`;
}
