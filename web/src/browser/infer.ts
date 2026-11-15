/**
 * Whole-brain prediction and scoring in the browser: the same steps as ml/inference.py and ml/metrics.py.
 *   1. crop to the box around the brain (voxels > 0 in any scan)
 *   2. normalise each scan: non-zero voxels -> mean 0, std 1 (MONAI NormalizeIntensity, nonzero, channel-wise)
 *   3. cover the crop with 160^3 windows, 50% overlap, averaging overlaps (MONAI sliding_window_inference,
 *      constant blending; a crop smaller than a window is padded with zeros, split evenly before/after)
 *   4. region = averaged logit > 0 (= sigmoid > 0.5), pasted back into the full-size volume
 */
import { CHANNELS, type Case } from "./case";

export type Regions = [Uint8Array, Uint8Array, Uint8Array]; // WT, TC, ET
/** Runs the model on one window: input (4, 160, 160, 160) -> logits (3, 160, 160, 160), both z-fastest. */
export type WindowModel = (input: Float32Array) => Promise<Float32Array>;

const ROI = 160;

export function brainBox(c: Case): { lo: number[]; hi: number[] } {
  const [X, Y, Z] = c.dims;
  const lo = [X, Y, Z], hi = [-1, -1, -1];
  const vols = CHANNELS.map((k) => c.images[k]);
  let i = 0;
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++, i++) {
    if (vols[0][i] > 0 || vols[1][i] > 0 || vols[2][i] > 0 || vols[3][i] > 0) {
      if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
      if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
      if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
    }
  }
  return { lo, hi: hi.map((h) => h + 1) };
}

/** MONAI dense_patch_slices along one axis. */
function starts(size: number): number[] {
  const interval = ROI === size ? ROI : Math.max(1, Math.floor(ROI * 0.5));
  const num = Math.ceil(size / interval);
  let scanNum = 1;
  for (let d = 0; d < num; d++) {
    if (d * interval + ROI >= size) {
      scanNum = d + 1;
      break;
    }
  }
  return Array.from({ length: scanNum }, (_, i) => {
    const s = i * interval;
    return s - Math.max(s + ROI - size, 0);
  });
}

export function windowCount(c: Case): number {
  const { lo, hi } = brainBox(c);
  return [0, 1, 2].map((a) => starts(Math.max(hi[a] - lo[a], ROI)).length).reduce((p, q) => p * q, 1);
}

export async function predict(c: Case, model: WindowModel, onWindow?: (done: number, total: number) => void): Promise<Regions> {
  const [X, Y] = c.dims;
  const { lo, hi } = brainBox(c);
  const cs = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  // padded size (at least one window) and where the crop sits inside it
  const ps = cs.map((s) => Math.max(s, ROI));
  const before = cs.map((s, a) => Math.floor((ps[a] - s) / 2));
  const [PX, PY, PZ] = ps;
  const pn = PX * PY * PZ;

  // cropped, normalised, padded scans in z-fastest order (what the ONNX model expects)
  const img = CHANNELS.map((k) => {
    const src = c.images[k];
    let n = 0, sum = 0, sum2 = 0;
    for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) {
      const row = X * (y + Y * z);
      for (let x = lo[0]; x < hi[0]; x++) {
        const v = src[row + x];
        if (v !== 0) { n++; sum += v; sum2 += v * v; }
      }
    }
    const mean = n ? sum / n : 0;
    let std = n ? Math.sqrt(Math.max(sum2 / n - mean * mean, 0)) : 1;
    if (std === 0) std = 1;
    const out = new Float32Array(pn);
    for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) {
      const row = X * (y + Y * z);
      const pz = z - lo[2] + before[2], py = y - lo[1] + before[1];
      for (let x = lo[0]; x < hi[0]; x++) {
        const v = src[row + x];
        const px = x - lo[0] + before[0];
        out[(px * PY + py) * PZ + pz] = v !== 0 ? (v - mean) / std : 0;
      }
    }
    return out;
  });

  const sx = starts(PX), sy = starts(PY), sz = starts(PZ);
  const total = sx.length * sy.length * sz.length;
  const acc = new Float32Array(3 * pn), count = new Uint8Array(pn);
  const win = new Float32Array(4 * ROI ** 3);
  let done = 0;
  for (const x0 of sx) for (const y0 of sy) for (const z0 of sz) {
    for (let ch = 0; ch < 4; ch++) {
      const src = img[ch], off = ch * ROI ** 3;
      for (let x = 0; x < ROI; x++) for (let y = 0; y < ROI; y++) {
        const s = ((x0 + x) * PY + (y0 + y)) * PZ + z0;
        win.set(src.subarray(s, s + ROI), off + (x * ROI + y) * ROI);
      }
    }
    const logits = await model(win);
    for (let r = 0; r < 3; r++) {
      const off = r * ROI ** 3, aoff = r * pn;
      for (let x = 0; x < ROI; x++) for (let y = 0; y < ROI; y++) {
        const d = aoff + ((x0 + x) * PY + (y0 + y)) * PZ + z0, s = off + (x * ROI + y) * ROI;
        for (let z = 0; z < ROI; z++) acc[d + z] += logits[s + z];
      }
    }
    for (let x = 0; x < ROI; x++) for (let y = 0; y < ROI; y++) {
      const d = ((x0 + x) * PY + (y0 + y)) * PZ + z0;
      for (let z = 0; z < ROI; z++) count[d + z]++;
    }
    onWindow?.(++done, total);
  }

  // averaged logit > 0, back to full size (x-fastest)
  const n = c.images.t2f.length;
  const regions: Regions = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
  for (let r = 0; r < 3; r++) {
    const a = acc.subarray(r * pn, (r + 1) * pn), out = regions[r];
    for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) {
      const row = X * (y + Y * z);
      const pz = z - lo[2] + before[2], py = y - lo[1] + before[1];
      for (let x = lo[0]; x < hi[0]; x++) {
        const p = ((x - lo[0] + before[0]) * PY + py) * PZ + pz;
        if (a[p] / count[p] > 0) out[row + x] = 1;
      }
    }
  }
  return regions;
}

// ------------------------------------------------------------------------------- scores (ml/metrics.py)
export const MISSED_HD95 = 373.13;
export type Scores = { dice: number; hd95: number | null; sensitivity: number | null; precision: number | null };

/** Squared Euclidean distance transform along one line (Felzenszwalb & Huttenlocher), sampling `s` mm. */
function edt1d(f: Float64Array, n: number, s: number, d: Float64Array, v: Int32Array, zb: Float64Array) {
  let k = 0;
  v[0] = 0; zb[0] = -Infinity; zb[1] = Infinity;
  const s2 = s * s;
  for (let q = 1; q < n; q++) {
    let sp: number;
    for (;;) {
      const p = v[k];
      sp = (f[q] + s2 * q * q - (f[p] + s2 * p * p)) / (2 * s2 * (q - p));
      if (sp <= zb[k] && k > 0) k--;
      else break;
    }
    if (sp <= zb[k]) { // k === 0 and still dominated: replace
      v[0] = q; zb[0] = -Infinity; zb[1] = Infinity; continue;
    }
    k++; v[k] = q; zb[k] = sp; zb[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (zb[k + 1] < q) k++;
    const dq = (q - v[k]) * s;
    d[q] = dq * dq + f[v[k]];
  }
}

/** Distance (mm) from every voxel to the nearest `target` voxel, in a box of size dims (x-fastest). */
function distanceTo(target: Uint8Array, dims: number[], spacing: number[]): Float64Array {
  const [X, Y, Z] = dims, n = X * Y * Z, m = Math.max(X, Y, Z);
  const g = new Float64Array(n);
  for (let i = 0; i < n; i++) g[i] = target[i] ? 0 : 1e20;
  const f = new Float64Array(m), d = new Float64Array(m), v = new Int32Array(m), zb = new Float64Array(m + 1);
  const pass = (len: number, stride: number, starts: number[], s: number) => {
    for (const st of starts) {
      for (let q = 0; q < len; q++) f[q] = g[st + q * stride];
      edt1d(f, len, s, d, v, zb);
      for (let q = 0; q < len; q++) g[st + q * stride] = d[q];
    }
  };
  const sx: number[] = [], sy: number[] = [], sz: number[] = [];
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) sx.push(X * (y + Y * z));
  for (let z = 0; z < Z; z++) for (let x = 0; x < X; x++) sy.push(x + X * Y * z);
  for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) sz.push(x + X * y);
  pass(X, 1, sx, spacing[0]);
  pass(Y, X, sy, spacing[1]);
  pass(Z, X * Y, sz, spacing[2]);
  for (let i = 0; i < n; i++) g[i] = Math.sqrt(g[i]);
  return g;
}

/** Surface voxels: inside the mask with at least one of the 6 neighbours outside (scipy binary_erosion edges). */
function edges(mask: Uint8Array, dims: number[]): Uint8Array {
  const [X, Y, Z] = dims, out = new Uint8Array(mask.length);
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    const i = x + X * (y + Y * z);
    if (!mask[i]) continue;
    if (x === 0 || y === 0 || z === 0 || x === X - 1 || y === Y - 1 || z === Z - 1 ||
      !mask[i - 1] || !mask[i + 1] || !mask[i - X] || !mask[i + X] || !mask[i - X * Y] || !mask[i + X * Y]) out[i] = 1;
  }
  return out;
}

function percentile(values: number[], q: number): number {
  if (!values.length) return 0;
  const s = Float64Array.from(values).sort();
  const pos = (s.length - 1) * (q / 100), lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Dice, sensitivity, precision and HD95 of one region, like ml/metrics.region_metrics. */
export function regionScores(pred: Uint8Array, truth: Uint8Array, dims: number[], spacing: number[], withHd95 = true): Scores {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < pred.length; i++) {
    if (pred[i]) { if (truth[i]) tp++; else fp++; } else if (truth[i]) fn++;
  }
  const nPred = tp + fp, nTrue = tp + fn;
  if (nTrue === 0 && nPred === 0) return { dice: 1, hd95: 0, sensitivity: 1, precision: 1 };
  const out: Scores = {
    dice: (2 * tp) / (nPred + nTrue),
    sensitivity: nTrue ? tp / nTrue : null,
    precision: nPred ? tp / nPred : null,
    hd95: null,
  };
  if (!withHd95) return out;
  if (nTrue === 0 || nPred === 0) {
    out.hd95 = MISSED_HD95;
    return out;
  }
  // only the box around both masks matters (+1 voxel margin)
  const [X, Y, Z] = dims;
  const lo = [X, Y, Z], hi = [0, 0, 0];
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    const i = x + X * (y + Y * z);
    if (pred[i] || truth[i]) {
      lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
      lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
      lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
    }
  }
  const a = lo.map((v) => Math.max(v - 1, 0)), b = hi.map((v, k) => Math.min(v + 2, dims[k]));
  const bd = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const crop = (m: Uint8Array) => {
    const o = new Uint8Array(bd[0] * bd[1] * bd[2]);
    let k = 0;
    for (let z = a[2]; z < b[2]; z++) for (let y = a[1]; y < b[1]; y++) {
      const row = X * (y + Y * z);
      for (let x = a[0]; x < b[0]; x++) o[k++] = m[row + x];
    }
    return o;
  };
  const ep = edges(crop(pred), bd), et = edges(crop(truth), bd);
  const toTruth = distanceTo(et, bd, spacing), toPred = distanceTo(ep, bd, spacing);
  const dPred: number[] = [], dTrue: number[] = [];
  for (let i = 0; i < ep.length; i++) {
    if (ep[i]) dPred.push(toTruth[i]);
    if (et[i]) dTrue.push(toPred[i]);
  }
  out.hd95 = Math.max(percentile(dPred, 95), percentile(dTrue, 95));
  return out;
}
