/**
 * Voxel masks -> smooth surfaces in scanner millimetres, in the browser (the server uses marching cubes;
 * here "surface nets", which needs no lookup tables and gives similarly smooth meshes), plus the brain mask
 * and a minimal .glb writer. Volumes are x-fastest: value(x, y, z) = data[x + X * (y + Y * z)].
 */
import type { Mat4 } from "./nifti";

export type Mesh = { positions: Float32Array; normals: Float32Array; indices: Uint32Array };

// ------------------------------------------------------------------------------------------ morphology
function dilate6(m: Uint8Array, [X, Y, Z]: number[]): Uint8Array {
  const o = new Uint8Array(m.length), XY = X * Y;
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    const i = x + X * y + XY * z;
    o[i] = m[i] || (x > 0 && m[i - 1]) || (x < X - 1 && m[i + 1]) || (y > 0 && m[i - X]) || (y < Y - 1 && m[i + X]) ||
      (z > 0 && m[i - XY]) || (z < Z - 1 && m[i + XY]) ? 1 : 0;
  }
  return o;
}
function erode6(m: Uint8Array, [X, Y, Z]: number[]): Uint8Array {
  const o = new Uint8Array(m.length), XY = X * Y;
  for (let z = 1; z < Z - 1; z++) for (let y = 1; y < Y - 1; y++) for (let x = 1; x < X - 1; x++) {
    const i = x + X * y + XY * z;
    o[i] = m[i] && m[i - 1] && m[i + 1] && m[i - X] && m[i + X] && m[i - XY] && m[i + XY] ? 1 : 0;
  }
  return o;
}

/** Connected pieces (6-connected), keep the largest. */
export function largestComponent(m: Uint8Array, [X, Y, Z]: number[]): Uint8Array {
  const lab = new Int32Array(m.length), XY = X * Y, queue = new Int32Array(m.length);
  let best = 0, bestSize = 0, next = 0;
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || lab[s]) continue;
    next++;
    let head = 0, tail = 0, size = 0;
    queue[tail++] = s; lab[s] = next;
    while (head < tail) {
      const i = queue[head++]; size++;
      const x = i % X, y = Math.floor(i / X) % Y, z = Math.floor(i / XY);
      const nb = [x > 0 ? i - 1 : -1, x < X - 1 ? i + 1 : -1, y > 0 ? i - X : -1, y < Y - 1 ? i + X : -1,
        z > 0 ? i - XY : -1, z < Z - 1 ? i + XY : -1];
      for (const j of nb) if (j >= 0 && m[j] && !lab[j]) { lab[j] = next; queue[tail++] = j; }
    }
    if (size > bestSize) [best, bestSize] = [next, size];
  }
  const o = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) o[i] = lab[i] === best ? 1 : 0;
  return o;
}

/** Background not reachable from the volume's border is a hole: fill it. */
function fillHoles(m: Uint8Array, [X, Y, Z]: number[]): Uint8Array {
  const outside = new Uint8Array(m.length), XY = X * Y, queue = new Int32Array(m.length);
  let tail = 0;
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    if (x && y && z && x < X - 1 && y < Y - 1 && z < Z - 1) continue;
    const i = x + X * y + XY * z;
    if (!m[i]) { outside[i] = 1; queue[tail++] = i; }
  }
  let head = 0;
  while (head < tail) {
    const i = queue[head++];
    const x = i % X, y = Math.floor(i / X) % Y, z = Math.floor(i / XY);
    const nb = [x > 0 ? i - 1 : -1, x < X - 1 ? i + 1 : -1, y > 0 ? i - X : -1, y < Y - 1 ? i + X : -1,
      z > 0 ? i - XY : -1, z < Z - 1 ? i + XY : -1];
    for (const j of nb) if (j >= 0 && !m[j] && !outside[j]) { outside[j] = 1; queue[tail++] = j; }
  }
  const o = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) o[i] = outside[i] ? 0 : 1;
  return o;
}

/** Same steps as brainviz.meshing.brain_mask: any scan > 0, closed (2x), holes filled, largest piece. */
export function brainMask(vols: Float32Array[], dims: number[]): Uint8Array {
  let m: Uint8Array = new Uint8Array(vols[0].length);
  for (const v of vols) for (let i = 0; i < v.length; i++) if (v[i] > 0) m[i] = 1;
  m = dilate6(dilate6(m, dims), dims);
  m = erode6(erode6(m, dims), dims);
  return largestComponent(fillHoles(m, dims), dims);
}

/** Drop connected pieces smaller than minVoxels (brainviz.build.remove_small). */
export function removeSmall(m: Uint8Array, dims: number[], minVoxels: number): Uint8Array {
  const [X, Y, Z] = dims, XY = X * Y;
  const out = new Uint8Array(m.length), seen = new Uint8Array(m.length), queue = new Int32Array(m.length);
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || seen[s]) continue;
    let head = 0, tail = 0;
    queue[tail++] = s; seen[s] = 1;
    while (head < tail) {
      const i = queue[head++];
      const x = i % X, y = Math.floor(i / X) % Y, z = Math.floor(i / XY);
      const nb = [x > 0 ? i - 1 : -1, x < X - 1 ? i + 1 : -1, y > 0 ? i - X : -1, y < Y - 1 ? i + X : -1,
        z > 0 ? i - XY : -1, z < Z - 1 ? i + XY : -1];
      for (const j of nb) if (j >= 0 && m[j] && !seen[j]) { seen[j] = 1; queue[tail++] = j; }
    }
    if (tail >= minVoxels) for (let k = 0; k < tail; k++) out[queue[k]] = 1;
  }
  return out;
}

// --------------------------------------------------------------------------------------------- surface
/** Gaussian blur (separable, truncated at 4 sigma like scipy) of a box, in place. */
function blur(f: Float32Array, [X, Y, Z]: number[], sigma: number) {
  const r = Math.ceil(4 * sigma);
  const k = Float32Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const ks = k.reduce((a, b) => a + b, 0);
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  const line = new Float32Array(Math.max(X, Y, Z));
  const run = (len: number, stride: number, starts: Generator<number>) => {
    for (const s of starts) {
      for (let q = 0; q < len; q++) line[q] = f[s + q * stride];
      for (let q = 0; q < len; q++) {
        let acc = 0;
        for (let t = -r; t <= r; t++) {
          let p = q + t;
          if (p < 0) p = -p - 1; else if (p >= len) p = 2 * len - p - 1; // scipy "reflect"
          p = Math.min(Math.max(p, 0), len - 1);
          acc += k[t + r] * line[p];
        }
        f[s + q * stride] = acc;
      }
    }
  };
  run(X, 1, (function* () { for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) yield X * (y + Y * z); })());
  run(Y, X, (function* () { for (let z = 0; z < Z; z++) for (let x = 0; x < X; x++) yield x + X * Y * z; })());
  run(Z, X * Y, (function* () { for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) yield x + X * y; })());
}

/**
 * Mask -> smooth surface in mm, like brainviz.meshing.mask_to_mesh: blur lightly, take the 0.5 level,
 * smooth (Taubin), place with the affine. `step` > 1 samples a coarser grid (for the large brain surface).
 * Returns null for masks too small to draw.
 */
export function maskToMesh(mask: Uint8Array, dims: number[], affine: Mat4, sigma: number,
  smoothIters = 10, minVoxels = 20, step = 1): Mesh | null {
  const [X, Y, Z] = dims;
  let count = 0;
  const lo = [X, Y, Z], hi = [-1, -1, -1];
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    if (!mask[x + X * (y + Y * z)]) continue;
    count++;
    if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
    if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
    if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
  }
  if (count < minVoxels) return null;
  // work in a padded box around the mask (the field is 0 outside it)
  const pad = 2 + Math.ceil(4 * sigma) + step;
  const o = lo.map((v) => v - pad);
  const bd = [0, 1, 2].map((a) => hi[a] - lo[a] + 1 + 2 * pad);
  const [BX, BY, BZ] = bd;
  let field = new Float32Array(BX * BY * BZ);
  for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) for (let x = lo[0]; x <= hi[0]; x++) {
    if (mask[x + X * (y + Y * z)]) field[x - o[0] + BX * (y - o[1] + BY * (z - o[2]))] = 1;
  }
  if (sigma > 0) {
    const blurred = field.slice();
    blur(blurred, bd, sigma);
    let max = 0;
    for (let i = 0; i < blurred.length; i++) if (blurred[i] > max) max = blurred[i];
    // scattered specks can blur away entirely: then use the unblurred voxels (as on the server)
    if (max > 0.5) field = blurred;
  }

  // sample grid (every `step` voxels)
  const g = [Math.floor((BX - 1) / step) + 1, Math.floor((BY - 1) / step) + 1, Math.floor((BZ - 1) / step) + 1];
  const [GX, GY, GZ] = g;
  const val = (x: number, y: number, z: number) => field[x * step + BX * (y * step + BY * z * step)];
  const L = 0.5;
  // one vertex per cell that the surface passes through: the mean of its edge crossings
  const cellVert = new Int32Array((GX - 1) * (GY - 1) * (GZ - 1)).fill(-1);
  const pos: number[] = [];
  const corner = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const cellEdges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const v = new Float32Array(8);
  for (let z = 0; z < GZ - 1; z++) for (let y = 0; y < GY - 1; y++) for (let x = 0; x < GX - 1; x++) {
    let mask8 = 0;
    for (let c = 0; c < 8; c++) {
      v[c] = val(x + corner[c][0], y + corner[c][1], z + corner[c][2]);
      if (v[c] > L) mask8 |= 1 << c;
    }
    if (mask8 === 0 || mask8 === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of cellEdges) {
      if ((v[a] > L) === (v[b] > L)) continue;
      const t = (L - v[a]) / (v[b] - v[a]);
      sx += corner[a][0] + t * (corner[b][0] - corner[a][0]);
      sy += corner[a][1] + t * (corner[b][1] - corner[a][1]);
      sz += corner[a][2] + t * (corner[b][2] - corner[a][2]);
      n++;
    }
    cellVert[x + (GX - 1) * (y + (GY - 1) * z)] = pos.length / 3;
    pos.push(x + sx / n, y + sy / n, z + sz / n);
  }
  // one quad per grid edge the surface crosses, joining the 4 cells around that edge
  const idx: number[] = [];
  const cell = (x: number, y: number, z: number) =>
    x < 0 || y < 0 || z < 0 || x >= GX - 1 || y >= GY - 1 || z >= GZ - 1 ? -1 : cellVert[x + (GX - 1) * (y + (GY - 1) * z)];
  for (let z = 0; z < GZ; z++) for (let y = 0; y < GY; y++) for (let x = 0; x < GX; x++) {
    const inside = val(x, y, z) > L;
    for (let a = 0; a < 3; a++) {
      const q = [x, y, z];
      q[a]++;
      if (q[a] >= g[a]) continue;
      if (inside === val(q[0], q[1], q[2]) > L) continue;
      const b = (a + 1) % 3, c = (a + 2) % 3;
      const at = (db: number, dc: number) => {
        const p = [x, y, z];
        p[b] -= db; p[c] -= dc;
        return cell(p[0], p[1], p[2]);
      };
      const c00 = at(1, 1), c10 = at(0, 1), c11 = at(0, 0), c01 = at(1, 0);
      if (c00 < 0 || c10 < 0 || c11 < 0 || c01 < 0) continue;
      // normal of (c00, c10, c11) points along +a: outward when the inside is at the lower end
      if (inside) idx.push(c00, c10, c11, c00, c11, c01);
      else idx.push(c00, c11, c10, c00, c01, c11);
    }
  }
  if (!idx.length) return null;

  const nv = pos.length / 3;
  const P = Float32Array.from(pos);
  const I = Uint32Array.from(idx);
  taubin(P, I, nv, smoothIters);

  // grid -> voxel index -> millimetres
  const A = affine;
  const out = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    const vx = P[3 * i] * step + o[0], vy = P[3 * i + 1] * step + o[1], vz = P[3 * i + 2] * step + o[2];
    for (let r = 0; r < 3; r++) out[3 * i + r] = A[r][0] * vx + A[r][1] * vy + A[r][2] * vz + A[r][3];
  }
  const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (det < 0) for (let f = 0; f < I.length; f += 3) [I[f + 1], I[f + 2]] = [I[f + 2], I[f + 1]];
  return { positions: out, normals: vertexNormals(out, I), indices: I };
}

/** Taubin smoothing (shrink-free): alternate +0.5 and -0.53 steps toward the neighbours' average. */
function taubin(P: Float32Array, I: Uint32Array, nv: number, iters: number) {
  if (!iters) return;
  const deg = new Uint32Array(nv), start = new Uint32Array(nv + 1);
  for (let f = 0; f < I.length; f += 3) for (let k = 0; k < 3; k++) deg[I[f + k]] += 2;
  for (let i = 0; i < nv; i++) start[i + 1] = start[i] + deg[i];
  const nb = new Uint32Array(start[nv]), fill = start.slice(0, nv);
  for (let f = 0; f < I.length; f += 3) for (let k = 0; k < 3; k++) {
    const a = I[f + k];
    nb[fill[a]++] = I[f + ((k + 1) % 3)];
    nb[fill[a]++] = I[f + ((k + 2) % 3)];
  }
  const tmp = new Float32Array(P.length);
  for (let it = 0; it < 2 * iters; it++) {
    const w = it % 2 === 0 ? 0.5 : -0.53;
    for (let i = 0; i < nv; i++) {
      const s = start[i], e = start[i + 1];
      if (e === s) { tmp.set(P.subarray(3 * i, 3 * i + 3), 3 * i); continue; }
      let ax = 0, ay = 0, az = 0;
      for (let k = s; k < e; k++) { const j = nb[k]; ax += P[3 * j]; ay += P[3 * j + 1]; az += P[3 * j + 2]; }
      const n = e - s;
      tmp[3 * i] = P[3 * i] + w * (ax / n - P[3 * i]);
      tmp[3 * i + 1] = P[3 * i + 1] + w * (ay / n - P[3 * i + 1]);
      tmp[3 * i + 2] = P[3 * i + 2] + w * (az / n - P[3 * i + 2]);
    }
    P.set(tmp);
  }
}

function vertexNormals(P: Float32Array, I: Uint32Array): Float32Array {
  const N = new Float32Array(P.length);
  for (let f = 0; f < I.length; f += 3) {
    const a = 3 * I[f], b = 3 * I[f + 1], c = 3 * I[f + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; // area-weighted
    for (const k of [a, b, c]) { N[k] += nx; N[k + 1] += ny; N[k + 2] += nz; }
  }
  for (let i = 0; i < N.length; i += 3) {
    const l = Math.hypot(N[i], N[i + 1], N[i + 2]) || 1;
    N[i] /= l; N[i + 1] /= l; N[i + 2] /= l;
  }
  return N;
}

export function meshBounds(m: Mesh): [number[], number[]] {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < m.positions.length; i += 3) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], m.positions[i + k]);
    hi[k] = Math.max(hi[k], m.positions[i + k]);
  }
  return [lo, hi];
}

// ------------------------------------------------------------------------------------------------- glb
/** Named meshes -> one binary glTF (positions, normals, indices), the format the viewer loads. */
export function writeGlb(meshes: Record<string, Mesh | null>): ArrayBuffer | null {
  const entries = Object.entries(meshes).filter((e): e is [string, Mesh] => e[1] !== null);
  if (!entries.length) return null;
  const views: object[] = [], accessors: object[] = [], gMeshes: object[] = [], nodes: object[] = [];
  const parts: Uint8Array[] = [];
  let offset = 0;
  const addView = (data: ArrayBufferView, target: number) => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    views.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length, target });
    parts.push(bytes);
    offset += bytes.length;
    const padLen = (4 - (offset % 4)) % 4;
    if (padLen) { parts.push(new Uint8Array(padLen)); offset += padLen; }
    return views.length - 1;
  };
  for (const [name, m] of entries) {
    const [lo, hi] = meshBounds(m);
    const n = m.positions.length / 3;
    accessors.push({ bufferView: addView(m.positions, 34962), componentType: 5126, count: n, type: "VEC3", min: lo, max: hi });
    accessors.push({ bufferView: addView(m.normals, 34962), componentType: 5126, count: n, type: "VEC3" });
    accessors.push({ bufferView: addView(m.indices, 34963), componentType: 5125, count: m.indices.length, type: "SCALAR" });
    const a = accessors.length - 3;
    gMeshes.push({ name, primitives: [{ attributes: { POSITION: a, NORMAL: a + 1 }, indices: a + 2 }] });
    nodes.push({ name, mesh: gMeshes.length - 1 });
  }
  const json = {
    asset: { version: "2.0", generator: "brain-npr browser" },
    scene: 0, scenes: [{ nodes: nodes.map((_, i) => i) }], nodes, meshes: gMeshes, accessors,
    bufferViews: views, buffers: [{ byteLength: offset }],
  };
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  if (jpad) {
    const padded = new Uint8Array(jsonBytes.length + jpad).fill(0x20);
    padded.set(jsonBytes);
    jsonBytes = padded;
  }
  const total = 12 + 8 + jsonBytes.length + 8 + offset;
  const out = new Uint8Array(total), dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true); dv.setUint32(16, 0x4e4f534a, true);
  out.set(jsonBytes, 20);
  let p = 20 + jsonBytes.length;
  dv.setUint32(p, offset, true); dv.setUint32(p + 4, 0x004e4942, true);
  p += 8;
  for (const part of parts) { out.set(part, p); p += part.length; }
  return out.buffer;
}
