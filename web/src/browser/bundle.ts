/**
 * A checked case + segmentations -> the same bundle the server writes (pipeline/brainviz/build.py), held in
 * memory: brain.glb, seg_<source>.glb, diff_<source>.glb, volume_<scan>.bin (uint8, x-fastest) and meta.
 */
import { CHANNELS, scanHash, type Case } from "./case";
import { regionScores, type Regions, type Scores } from "./infer";
import { brainMask, maskToMesh, meshBounds, removeSmall, writeGlb } from "./mesh";

export type Files = Record<string, ArrayBuffer>;
export type Meta = Record<string, any>;

const REGION_NAMES = ["WT", "TC", "ET"] as const;
const ZONES: Record<string, { name: string; color: number[] }> = {
  edema: { name: "Swelling (edema)", color: [0.98, 0.78, 0.18] },
  non_enhancing: { name: "Inactive tumour (non-enhancing / necrotic)", color: [0.55, 0.12, 0.65] },
  enhancing: { name: "Active tumour (enhancing)", color: [0.9, 0.16, 0.2] },
};
const DIFF_MIN_VOXELS = 10;
const round = (v: number, d = 2) => Math.round(v * 10 ** d) / 10 ** d;

/** The 3 non-overlapping zones drawn in the viewer (ET inside TC inside WT enforced, as on the server). */
function zonesFromRegions([wt, tc, et]: Regions) {
  const n = wt.length;
  const z = { edema: new Uint8Array(n), non_enhancing: new Uint8Array(n), enhancing: new Uint8Array(n) };
  for (let i = 0; i < n; i++) {
    const w = wt[i], t = tc[i] & w, e = et[i] & t;
    if (e) z.enhancing[i] = 1;
    else if (t) z.non_enhancing[i] = 1;
    else if (w) z.edema[i] = 1;
  }
  return z;
}

const sum = (m: Uint8Array) => { let s = 0; for (let i = 0; i < m.length; i++) s += m[i]; return s; };

/** Linear-interpolated percentiles (numpy's default) of the brain voxels -> 0..255 for display. */
function toUint8(vol: Float32Array, brain: Uint8Array): Uint8Array {
  const vals: number[] = [];
  for (let i = 0; i < vol.length; i++) if (brain[i]) vals.push(vol[i]);
  const s = Float32Array.from(vals).sort();
  const pct = (q: number) => {
    if (!s.length) return q < 50 ? 0 : 1;
    const pos = (s.length - 1) * (q / 100), lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  };
  const lo = pct(0.5), hi = pct(99.5), span = Math.max(hi - lo, 1e-6);
  const out = new Uint8Array(vol.length);
  for (let i = 0; i < vol.length; i++) {
    if (!brain[i]) continue;
    out[i] = Math.floor(Math.min(Math.max((vol[i] - lo) / span, 0), 1) * 255);
  }
  return out;
}

export type Identity = { case_id: string; split: string; meaning: string; matched_scans: string[] } | null;

/** Is this brain from our dataset? Same fingerprints as ml/fingerprint.py. */
export async function identify(c: Case, fingerprints: any): Promise<Identity> {
  if (!fingerprints?.cases) return null;
  const table = new Map<string, [string, string, string]>();
  for (const [caseId, info] of Object.entries<any>(fingerprints.cases)) {
    for (const [scan, h] of Object.entries<string>(info.scans)) table.set(h, [caseId, info.split, scan]);
  }
  const hits: [string, string, string][] = [];
  for (const ch of CHANNELS) {
    const hit = table.get(await scanHash(c.images[ch], c.dims));
    if (hit) hits.push(hit);
  }
  if (!hits.length) return null;
  const [caseId, split] = hits[0];
  return { case_id: caseId, split, meaning: fingerprints.meaning?.[split] ?? "", matched_scans: hits.filter((h) => h[0] === caseId).map((h) => h[2]).sort() };
}

export function describeIdentity(ident: Identity): string {
  if (!ident) return "A brain from outside our dataset: the models have never seen it.";
  return `This is ${ident.case_id} from the MSD brain tumour dataset (${ident.split} set): ${ident.meaning}.`;
}

/** The brain surface, the 4 scans for the slice view, and the start of meta.json. */
export function buildBase(c: Case, extra: Meta): { meta: Meta; files: Files } {
  const dims = c.dims;
  const brain = brainMask(CHANNELS.map((k) => c.images[k]), dims);
  const brainMesh = maskToMesh(brain, dims, c.affine, 1.2, 15, 20, 2);
  if (!brainMesh) throw new Error("Could not find the brain in the scans.");
  const files: Files = { "brain.glb": writeGlb({ brain: brainMesh })! };
  // scans cropped to the brain's box, 0..255, x-fastest (what the slice view expects)
  const [X, Y, Z] = dims;
  const lo = [X, Y, Z], hi = [-1, -1, -1];
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    if (!brain[x + X * (y + Y * z)]) continue;
    if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
    if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
    if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
  }
  const shape = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
  const volFiles: Record<string, string> = {};
  for (const ch of CHANNELS) {
    const full = toUint8(c.images[ch], brain);
    const out = new Uint8Array(shape[0] * shape[1] * shape[2]);
    let k = 0;
    for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) {
      const row = X * (y + Y * z);
      for (let x = lo[0]; x <= hi[0]; x++) out[k++] = full[row + x];
    }
    volFiles[ch] = `volume_${ch}.bin`;
    files[volFiles[ch]] = out.buffer;
  }
  const bounds = meshBounds(brainMesh);
  const meta: Meta = {
    bundle_version: 3,
    case_id: "upload",
    units: "mm",
    spacing: c.spacing,
    bounds,
    center: bounds[0].map((v, i) => (v + bounds[1][i]) / 2),
    affine: c.affine,
    brain: "brain.glb",
    volumes: { files: volFiles, shape, offset: lo, default: "t2f", dtype: "uint8", encoding: "raw" },
    segmentations: {},
    default_segmentation: "",
    ...extra,
  };
  return { meta, files };
}

/** One segmentation's files and meta entry (brainviz.build.write_segmentation), plus scores if an outline exists. */
export function addSegmentation(c: Case, meta: Meta, files: Files, source: string, regions: Regions, extra: Meta,
  withHd95 = true) {
  const dims = c.dims;
  const voxelMl = (c.spacing[0] * c.spacing[1] * c.spacing[2]) / 1000;
  const zones = zonesFromRegions(regions);
  const meshes = Object.fromEntries(Object.entries(zones).map(([k, m]) => [k, maskToMesh(m, dims, c.affine, 0.8, 10)]));
  files[`seg_${source}.glb`] = writeGlb(meshes) ?? new ArrayBuffer(0);
  const info: Meta = {
    label: extra.label ?? source,
    file: `seg_${source}.glb`,
    zones: Object.fromEntries(Object.entries(ZONES).map(([k, z]) => [k, {
      name: z.name, color: z.color, volume_ml: round(sum(zones[k as keyof typeof zones]) * voxelMl), present: meshes[k] !== null,
    }])),
    regions_ml: Object.fromEntries(REGION_NAMES.map((r, i) => [r, round(sum(regions[i]) * voxelMl)])),
    ...Object.fromEntries(Object.entries(extra).filter(([k]) => k !== "label")),
  };
  if (c.regions && source !== "expert") {
    const scores: Record<string, Scores> = {};
    REGION_NAMES.forEach((r, i) => {
      const s = regionScores(regions[i], c.regions![i], dims, c.spacing, withHd95);
      const r4 = (v: number | null) => (v === null || !Number.isFinite(v) ? null : round(v, 4));
      scores[r] = { dice: round(s.dice, 4), hd95: r4(s.hd95), sensitivity: r4(s.sensitivity), precision: r4(s.precision) };
    });
    info.scores = scores;
    info.dice_mean = round(REGION_NAMES.reduce((a, r) => a + scores[r].dice, 0) / 3, 4);
    const diffMeshes: Record<string, ReturnType<typeof maskToMesh>> = {}, diffMl: Record<string, number> = {};
    REGION_NAMES.forEach((r, i) => {
      const e = c.regions![i], p = regions[i];
      for (const kind of ["missed", "extra"] as const) {
        let m: Uint8Array = new Uint8Array(p.length);
        for (let v = 0; v < m.length; v++) m[v] = kind === "missed" ? e[v] & (1 - p[v]) : p[v] & (1 - e[v]);
        m = removeSmall(m, dims, DIFF_MIN_VOXELS);
        diffMl[`${r}_${kind}`] = round(sum(m) * voxelMl);
        diffMeshes[`${r}_${kind}`] = maskToMesh(m, dims, c.affine, 0.6, 5);
      }
    });
    const glb = writeGlb(diffMeshes);
    if (glb) files[`diff_${source}.glb`] = glb;
    info.diff = { file: glb ? `diff_${source}.glb` : null, volume_ml: diffMl, min_voxels: DIFF_MIN_VOXELS };
  }
  meta.segmentations[source] = info;
}
