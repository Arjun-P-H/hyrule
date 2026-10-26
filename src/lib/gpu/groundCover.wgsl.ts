/**
 * Where the ground is trodden bare and how the grass grows on it, baked per clipmap lattice point.
 * ---------------------------------------------------------------------------
 * A lawn the same everywhere is a carpet. The ground cover that groundUse gives the biome's (the clipmap's
 * bioA cover, bioB blade height) breaks it up the way a walked land is broken:
 *   roads      a network of dirt roads: the borders of a Voronoi diagram of 640 m cells, half of them kept,
 *              bent by two scales of noise so they wind, 3.4 m wide, their width a little different from
 *              place to place;
 *   footpaths  the same on 150 m cells, a third of the borders, 1.3 m wide;
 *   bare       patches of bare earth in the grass, 3 to 9 m across;
 *   tall       grass that grows long in drifts (the blades two to four times the height of the short lawn
 *              beside it), and short, trampled grass round roads, footpaths and bare patches, thinning
 *              toward them: the edge of a road is sparse, short grass over soil, not a wall of blades.
 * Each biome says how much of this its land has (biomes.ts GROUND_USE). A road is a road only where the
 * lattice can show it: it fades out as the texel grows toward its width (past ~4 m texels, 200+ m from the
 * camera, the clipmap shows none; the grass is not drawn that far either).
 *
 * The fields are baked into the channels the ground and the grass already read (cover, blade height, the
 * grass's dryness, the soil's colour), so the surfaces and the scatter need no new bindings: a road is
 * cover 0, soil showing through; its fringe is cover and blades reduced.
 */

export const GROUND_COVER_WGSL = /* wgsl */ `
// The nearest border of a Voronoi diagram on cells of one unit (p in cells): x the distance to it (in cells),
// y a number 0 … 1 the same from either side of it (a hash of the two cells), z the nearest cell's own.
fn borderAt(p: vec2f, seed: i32) -> vec3f {
  let ip = floor(p);
  let fr = p - ip;
  let i0 = vec2i(ip);
  var m1 = 64.0;
  var m2 = 64.0;
  var d1 = vec2f(0.0);
  var d2 = vec2f(0.0);
  var h1 = 0u;
  var h2 = 0u;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let h = hash3(i0.x + i, i0.y + j, seed);
      let o = vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095;
      let d = vec2f(f32(i), f32(j)) + o - fr;
      let dd = dot(d, d);
      if (dd < m1) { m2 = m1; d2 = d1; h2 = h1; m1 = dd; d1 = d; h1 = h; }
      else if (dd < m2) { m2 = dd; d2 = d; h2 = h; }
    }
  }
  // (the gap to the bisector of the two feature points: the true distance to the border)
  let dist = (m2 - m1) / (2.0 * max(length(d2 - d1), 1e-4));
  let eh = f32(hashU((h1 ^ h2) * 0x2545f491u + (h1 + h2)) >> 8u) * (1.0 / 16777216.0);
  return vec3f(dist, eh, f32(h1 >> 8u) * (1.0 / 16777216.0));
}

// A road or footpath network at world g: x the mask (1 on the track), y how near it is (1 at its edge, down to 0
// two or three widths out), for lattice spacing s.
fn trackAt(g: vec2f, s: f32, seed: i32) -> vec2f {
  // bent in two scales, so the borders wind: broad bends of ~70 m, a wobble of ~13 m
  let b1 = vec2f(vnoise(g.x / 210.0, g.y / 210.0, seed + 8101), vnoise(g.x / 210.0 + 7.3, g.y / 210.0 - 2.9, seed + 8102));
  let b2 = vec2f(vnoise(g.x / 52.0, g.y / 52.0, seed + 8103), vnoise(g.x / 52.0 - 3.1, g.y / 52.0 + 9.7, seed + 8104));
  let q = g + b1 * 70.0 + b2 * 13.0;
  let wob = 0.5 + 0.5 * vnoise(g.x / 37.0, g.y / 37.0, seed + 8105);
  var mask = 0.0;
  var near = 0.0;
  // roads
  {
    let r = borderAt(q / 640.0, seed + 8110);
    let w = (1.7 * (0.75 + 0.5 * wob)) * (1.0 - smoothstep(0.5, 1.1, s / 3.4));
    let keep = select(0.0, 1.0, r.y < 0.5);
    let d = r.x * 640.0;
    mask = max(mask, keep * (1.0 - smoothstep(0.55 * w, 1.15 * w, d)) * step(0.01, w));
    near = max(near, keep * (1.0 - smoothstep(1.1 * w, 3.6 * w, d)) * step(0.01, w));
  }
  // footpaths
  {
    let r = borderAt(q / 150.0 + 11.0, seed + 8112);
    let w = (0.65 * (0.7 + 0.6 * wob)) * (1.0 - smoothstep(0.5, 1.1, s / 1.3));
    let keep = select(0.0, 1.0, r.y < 0.33);
    let d = r.x * 150.0;
    mask = max(mask, keep * (1.0 - smoothstep(0.55 * w, 1.15 * w, d)) * step(0.01, w));
    near = max(near, keep * (1.0 - smoothstep(1.1 * w, 3.6 * w, d)) * step(0.01, w));
  }
  return vec2f(mask, near);
}

// The cover's use at world g for lattice spacing s, gu = the biome's GROUND_USE blended by the region weights:
//   (gu: the biome's GROUND_USE blended by the region weights: trails, bare patches, tall grass, rocks)
//   x  the cover's multiplier (1 = the biome's own, 0 = bare earth),
//   y  the blades' height multiplier,
//   z  a shift of the grass toward its dry colour (long grass is the paler),
//   w  the trodden mask (roads, footpaths), which lightens the soil.
fn groundUse(g: vec2f, s: f32, seed: i32, gu: vec4f) -> vec4f {
  var trail = vec2f(0.0);
  if (gu.x > 0.01) { trail = trackAt(g, s, seed) * gu.x; }
  // bare patches (a biome with none, none: its noise is not asked)
  var bareN = 0.0;
  if (gu.y > 0.01) {
    let bn = 0.55 * vnoise(g.x / 9.0, g.y / 9.0, seed + 8121) + 0.3 * vnoise(g.x / 3.1, g.y / 3.1, seed + 8122) + 0.15 * vnoise(g.x / 37.0, g.y / 37.0, seed + 8123);
    let thr = mix(0.75, 0.3, gu.y);
    bareN = smoothstep(thr, thr + 0.1, bn) * (1.0 - smoothstep(1.5, 5.0, s));
  }
  // long and short grass
  var tall = 0.5;
  if (gu.z > 0.01) {
    let tn = 0.5 + 0.5 * (0.65 * vnoise(g.x / 44.0, g.y / 44.0, seed + 8131) + 0.35 * vnoise(g.x / 12.0, g.y / 12.0, seed + 8132));
    tall = smoothstep(0.4, 0.68, tn);
  }
  var hm = mix(1.0, mix(0.42, 1.85, tall), gu.z);
  // trodden ground and its edge: bare, then sparse short grass
  let bare = max(trail.x * 0.985, bareN);
  let edge = max(trail.y, 0.7 * bareN);
  hm *= 1.0 - 0.55 * edge;
  let cover = (1.0 - bare) * (1.0 - 0.5 * edge);
  return vec4f(cover, hm, (tall - 0.5) * 0.4 * gu.z, trail.x);
}
`;
