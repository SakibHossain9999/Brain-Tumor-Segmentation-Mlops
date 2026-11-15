/**
 * Uploaded files -> a checked case, in the browser. Same rules and messages as the server (api/uploads.py):
 * one MSD file with all 4 scans, or 4 BraTS files; an optional expert outline in MSD or BraTS numbering.
 */
import { NiftiError, readNifti, type Mat4, type Nifti } from "./nifti";

export const CHANNELS = ["t2f", "t1n", "t1c", "t2w"] as const; // FLAIR, T1, T1 + contrast, T2 (the models' order)
export type Channel = (typeof CHANNELS)[number];
export const SLOT_OF: Record<string, Channel> = { flair: "t2f", t1: "t1n", t1ce: "t1c", t2: "t2w" };
export const SCAN_NAMES: Record<Channel, string> = { t2f: "FLAIR", t1n: "T1", t1c: "T1 + contrast", t2w: "T2" };

export type Inputs = {
  scans: { one: ArrayBuffer } | { four: Record<string, ArrayBuffer> }; // four: keys flair / t1 / t1ce / t2
  label?: ArrayBuffer;
  labelFormat?: "auto" | "msd" | "brats";
};
export type Case = {
  dims: [number, number, number];
  affine: Mat4;
  spacing: [number, number, number];
  images: Record<Channel, Float32Array>;
  regions?: [Uint8Array, Uint8Array, Uint8Array]; // expert outline: WT, TC, ET (1 = inside)
  format: "msd" | "brats";
  labelFormat?: "msd" | "brats";
};

export class UploadError extends Error {}

async function read(buf: ArrayBuffer, what: string): Promise<Nifti> {
  try {
    return await readNifti(buf, what);
  } catch (e) {
    throw new UploadError(e instanceof NiftiError ? e.message : `${what}: not a readable NIfTI file (.nii or .nii.gz).`);
  }
}

const fmtShape = (d: number[]) => `(${d.join(", ")})`;

function sameGrid(a: Nifti, ref: Nifti, what: string, refWhat: string) {
  if (a.dims.some((v, i) => v !== ref.dims[i])) {
    throw new UploadError(`${what} has size ${fmtShape(a.dims)}, but ${refWhat} has size ${fmtShape(ref.dims)}. ` +
      "All scans must come from the same prepared case.");
  }
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    if (Math.abs(a.affine[i][j] - ref.affine[i][j]) > 1e-2 + 1e-5 * Math.abs(ref.affine[i][j])) {
      throw new UploadError(`${what} and ${refWhat} are positioned differently in space (their headers disagree). ` +
        "The scans must be aligned to each other, as BraTS scans are.");
    }
  }
}

/** Refuse scans the models were not made for, with a reason (same thresholds as the server). */
function checkScans(images: Record<Channel, Float32Array>, dims: number[], spacing: number[]) {
  if (dims.some((s) => s < 64 || s > 512)) {
    throw new UploadError(`The scans have size ${fmtShape(dims)}; that is not a brain MRI volume this tool can handle.`);
  }
  if (!spacing.every((z) => z >= 0.8 && z <= 1.25)) {
    throw new UploadError(`The voxels are (${spacing.map((z) => +z.toFixed(2)).join(", ")}) mm. The models need about 1 mm ` +
      "voxels (BraTS-style); resample the scans to 1 mm first.");
  }
  const masks: Partial<Record<Channel, Uint8Array>> = {};
  for (const name of CHANNELS) {
    const vol = images[name];
    const mask = new Uint8Array(vol.length);
    let count = 0;
    for (let i = 0; i < vol.length; i++) {
      const v = vol[i];
      if (!Number.isFinite(v)) throw new UploadError(`The ${SCAN_NAMES[name]} scan contains invalid values (NaN or infinity).`);
      if (v !== 0) { mask[i] = 1; count++; }
    }
    const frac = count / vol.length;
    if (frac < 0.01) throw new UploadError(`The ${SCAN_NAMES[name]} scan is (almost) empty.`);
    if (frac > 0.6) {
      throw new UploadError(`The ${SCAN_NAMES[name]} scan doesn't look skull-stripped: ${Math.round(frac * 100)}% of it is ` +
        "non-zero. The models need brain-only scans with the background set to 0, as in BraTS.");
    }
    masks[name] = mask;
  }
  const ref = masks.t2f!;
  for (const name of CHANNELS) {
    const m = masks[name]!;
    let both = 0, either = 0;
    for (let i = 0; i < m.length; i++) {
      if (m[i] & ref[i]) both++;
      if (m[i] | ref[i]) either++;
    }
    const overlap = both / Math.max(either, 1);
    if (overlap < 0.3) {
      throw new UploadError(`The ${SCAN_NAMES[name]} scan doesn't line up with the FLAIR scan (only ` +
        `${Math.round(overlap * 100)}% of their brain areas overlap). Were the files mixed up, or not registered to each other?`);
    }
  }
}

/** Label numbers -> WT / TC / ET masks (pipeline/brainviz/labels.py). */
export function regionsFromLabels(lab: Uint8Array, fmt: "msd" | "brats"): [Uint8Array, Uint8Array, Uint8Array] {
  const n = lab.length;
  const wt = new Uint8Array(n), tc = new Uint8Array(n), et = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = lab[i];
    if (!v) continue;
    wt[i] = 1;
    if (fmt === "msd") {
      if (v === 2 || v === 3) tc[i] = 1;
      if (v === 3) et[i] = 1;
    } else {
      if (v === 1 || v === 3 || v === 4) tc[i] = 1;
      if (v === 3 || v === 4) et[i] = 1;
    }
  }
  return [wt, tc, et];
}

export async function loadCase(inputs: Inputs): Promise<Case> {
  let ref: Nifti;
  const images = {} as Record<Channel, Float32Array>;
  let format: "msd" | "brats";
  if ("one" in inputs.scans) {
    ref = await read(inputs.scans.one, "The scan file");
    if (ref.frames.length !== 4) {
      throw new UploadError("A single scan file must hold all 4 scans (size X x Y x Z x 4, in the order FLAIR, T1, " +
        `T1 + contrast, T2), as in the MSD dataset. This file has ${ref.frames.length} scan(s). ` +
        "To upload 4 separate files, drop them together instead.");
    }
    CHANNELS.forEach((c, i) => (images[c] = ref.frames[i]));
    format = "msd";
  } else {
    const four = inputs.scans.four;
    const missing = Object.keys(SLOT_OF).filter((s) => !four[s]).map((s) => SCAN_NAMES[SLOT_OF[s]]);
    if (missing.length) {
      throw new UploadError(`Missing scan(s): ${missing.join(", ")}. All four are needed: FLAIR, T1, T1 + contrast, T2.`);
    }
    let first: Nifti | null = null;
    for (const [slot, name] of Object.entries(SLOT_OF)) {
      const img = await read(four[slot], `The ${SCAN_NAMES[name]} file`);
      if (img.frames.length !== 1) {
        throw new UploadError(`The ${SCAN_NAMES[name]} file should hold one 3D scan, but holds ${img.frames.length}.`);
      }
      if (!first) first = img;
      else sameGrid(img, first, `The ${SCAN_NAMES[name]} scan`, "the FLAIR scan");
      images[name] = img.frames[0];
    }
    ref = first!;
    format = "brats";
  }
  checkScans(images, ref.dims, ref.spacing);
  const c: Case = { dims: ref.dims, affine: ref.affine, spacing: ref.spacing, images, format };

  if (inputs.label) {
    const lab = await read(inputs.label, "The expert outline file");
    if (lab.frames.length !== 1) throw new UploadError("The expert outline should be one 3D label map.");
    sameGrid(lab, ref, "The expert outline", "the scans");
    const values = new Set<number>();
    const labels = new Uint8Array(lab.frames[0].length);
    for (let i = 0; i < labels.length; i++) {
      const v = Math.round(lab.frames[0][i]);
      values.add(v);
      labels[i] = v;
    }
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.some((v) => v < 0 || v > 4)) {
      throw new UploadError(`The expert outline has label values [${sorted.join(", ")}]; expected 0-3 (MSD) or 0, 1, 2, 4 / ` +
        "0-3 (BraTS).");
    }
    const choice = inputs.labelFormat;
    const fmt = values.has(4) ? "brats" : choice === "msd" || choice === "brats" ? choice : format;
    c.regions = regionsFromLabels(labels, fmt);
    if (!c.regions[0].some((v) => v)) throw new UploadError("The expert outline is empty: it marks no tumour at all.");
    c.labelFormat = fmt;
  }
  return c;
}

/** SHA-256 of a scan's float32 values in the server's (C, z-fastest) order: matches ml/fingerprint.py. */
export async function scanHash(vol: Float32Array, dims: [number, number, number]): Promise<string> {
  const [X, Y, Z] = dims;
  const c = new Float32Array(vol.length);
  let k = 0;
  for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) {
    const base = x + X * y, step = X * Y;
    for (let z = 0; z < Z; z++) c[k++] = vol[base + step * z];
  }
  const digest = await crypto.subtle.digest("SHA-256", c.buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
