/**
 * Reading NIfTI-1 files (.nii / .nii.gz) in the browser, the way the server reads them with nibabel:
 * voxel values with the header's scaling applied, the header's best affine (sform, else qform), and the
 * volume turned to the closest standard (RAS) orientation, like nibabel's as_closest_canonical.
 *
 * Volumes are stored with x varying fastest: value(x, y, z) = data[x + X * (y + Y * z)].
 */
export type Mat4 = number[][];
export type Nifti = {
  dims: [number, number, number];
  frames: Float32Array[]; // one 3D volume per 4th-dimension entry (1 for a 3D file, 4 for an MSD scan file)
  affine: Mat4;           // voxel index -> scanner millimetres
  spacing: [number, number, number];
};

export class NiftiError extends Error {}

async function gunzipIfNeeded(bytes: Uint8Array): Promise<Uint8Array> {
  if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) return bytes;
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function qformAffine(v: DataView, le: boolean, pix: number[]): Mat4 {
  const b = v.getFloat32(256, le), c = v.getFloat32(260, le), d = v.getFloat32(264, le);
  const a = Math.sqrt(Math.max(0, 1 - (b * b + c * c + d * d)));
  const qfac = pix[0] < 0 ? -1 : 1;
  const R = [
    [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
    [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
    [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
  ];
  const zooms = [pix[1], pix[2], pix[3] * qfac];
  const off = [v.getFloat32(268, le), v.getFloat32(272, le), v.getFloat32(276, le)];
  return [0, 1, 2].map((i) => [...[0, 1, 2].map((j) => R[i][j] * zooms[j]), off[i]]).concat([[0, 0, 0, 1]]);
}

/** Parse one NIfTI file. Throws NiftiError with a readable message if it isn't one. */
export async function readNifti(buffer: ArrayBuffer, what: string): Promise<Nifti> {
  let bytes: Uint8Array;
  try {
    bytes = await gunzipIfNeeded(new Uint8Array(buffer));
  } catch {
    throw new NiftiError(`${what}: not a readable NIfTI file (.nii or .nii.gz).`);
  }
  if (bytes.length < 352) throw new NiftiError(`${what}: not a readable NIfTI file (.nii or .nii.gz).`);
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let le = true;
  if (v.getInt32(0, true) !== 348) {
    if (v.getInt32(0, false) === 348) le = false;
    else if (v.getInt32(0, true) === 540 || v.getInt32(0, false) === 540) {
      throw new NiftiError(`${what}: NIfTI-2 files aren't supported here; save it as NIfTI-1 (.nii.gz).`);
    } else throw new NiftiError(`${what}: not a readable NIfTI file (.nii or .nii.gz).`);
  }
  const dim = Array.from({ length: 8 }, (_, i) => v.getInt16(40 + 2 * i, le));
  const datatype = v.getInt16(70, le);
  const pix = Array.from({ length: 8 }, (_, i) => v.getFloat32(76 + 4 * i, le));
  const voxOffset = Math.max(352, Math.round(v.getFloat32(108, le)));
  let slope = v.getFloat32(112, le);
  const inter = v.getFloat32(116, le);
  if (!Number.isFinite(slope) || slope === 0) slope = 1;
  const qcode = v.getInt16(252, le), scode = v.getInt16(254, le);

  const ndim = dim[0];
  if (ndim < 3 || ndim > 4) throw new NiftiError(`${what}: expected a 3D or 4D image, got ${ndim} dimensions.`);
  const dims: [number, number, number] = [dim[1], dim[2], dim[3]];
  const nFrames = ndim === 4 ? Math.max(1, dim[4]) : 1;
  const n = dims[0] * dims[1] * dims[2];

  const readers: Record<number, [number, (o: number) => number]> = {
    2: [1, (o) => v.getUint8(o)], 256: [1, (o) => v.getInt8(o)],
    4: [2, (o) => v.getInt16(o, le)], 512: [2, (o) => v.getUint16(o, le)],
    8: [4, (o) => v.getInt32(o, le)], 768: [4, (o) => v.getUint32(o, le)],
    16: [4, (o) => v.getFloat32(o, le)], 64: [8, (o) => v.getFloat64(o, le)],
  };
  const reader = readers[datatype];
  if (!reader) throw new NiftiError(`${what}: unsupported voxel type (NIfTI datatype ${datatype}).`);
  const [size, get] = reader;
  if (voxOffset + n * nFrames * size > bytes.length) throw new NiftiError(`${what}: the file is incomplete (cut off).`);
  const scaled = !(slope === 1 && inter === 0);
  const frames: Float32Array[] = [];
  for (let f = 0; f < nFrames; f++) {
    const out = new Float32Array(n);
    const start = voxOffset + f * n * size;
    // fast paths for the common types
    if (le && datatype === 16 && (bytes.byteOffset + start) % 4 === 0) {
      out.set(new Float32Array(bytes.buffer, bytes.byteOffset + start, n));
    } else {
      for (let i = 0; i < n; i++) out[i] = get(start + i * size);
    }
    if (scaled) for (let i = 0; i < n; i++) out[i] = out[i] * slope + inter;
    frames.push(out);
  }

  let affine: Mat4;
  if (scode > 0) {
    affine = [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => v.getFloat32(280 + 16 * r + 4 * c, le))).concat([[0, 0, 0, 1]]);
  } else if (qcode > 0) {
    affine = qformAffine(v, le, pix);
  } else {
    affine = [[pix[1], 0, 0, 0], [0, pix[2], 0, 0], [0, 0, pix[3], 0], [0, 0, 0, 1]];
  }
  const nii: Nifti = { dims, frames, affine, spacing: [Math.abs(pix[1]), Math.abs(pix[2]), Math.abs(pix[3])] };
  return toCanonical(nii);
}

/** Like nibabel.as_closest_canonical: reorder / flip the axes so they point Right, Anterior, Superior. */
export function toCanonical(nii: Nifti): Nifti {
  const A = nii.affine;
  // for each voxel axis, the world axis it is closest to (nibabel's io_orientation, for non-oblique scans)
  const cols = [0, 1, 2].map((j) => [0, 1, 2].map((i) => A[i][j]));
  const used = new Set<number>();
  const out: number[] = [], flip: boolean[] = [];
  for (let j = 0; j < 3; j++) {
    const norm = Math.hypot(...cols[j]) || 1;
    let best = -1, bestVal = -1;
    for (let i = 0; i < 3; i++) {
      const val = Math.abs(cols[j][i] / norm);
      if (!used.has(i) && val > bestVal) [best, bestVal] = [i, val];
    }
    used.add(best);
    out.push(best);
    flip.push(cols[j][best] < 0);
  }
  if (out.every((o, j) => o === j) && !flip.some(Boolean)) return nii;

  const d = nii.dims;
  const nd: [number, number, number] = [0, 0, 0];
  for (let j = 0; j < 3; j++) nd[out[j]] = d[j];
  const n = d[0] * d[1] * d[2];
  const stride = [1, nd[0], nd[0] * nd[1]];
  const frames = nii.frames.map((src) => {
    const dst = new Float32Array(n);
    let i = 0;
    for (let z = 0; z < d[2]; z++) {
      const nz = (flip[2] ? d[2] - 1 - z : z) * stride[out[2]];
      for (let y = 0; y < d[1]; y++) {
        const nyz = nz + (flip[1] ? d[1] - 1 - y : y) * stride[out[1]];
        for (let x = 0; x < d[0]; x++, i++) dst[nyz + (flip[0] ? d[0] - 1 - x : x) * stride[out[0]]] = src[i];
      }
    }
    return dst;
  });
  // old index = P * new index + c  ->  new affine = old affine * [[P, c], [0, 1]]
  const T: Mat4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]];
  for (let j = 0; j < 3; j++) {
    T[j][out[j]] = flip[j] ? -1 : 1;
    T[j][3] = flip[j] ? d[j] - 1 : 0;
  }
  const affine = A.map((row) => [0, 1, 2, 3].map((c) => row.reduce((s, a, k) => s + a * T[k][c], 0)));
  const spacing: [number, number, number] = [0, 0, 0];
  for (let j = 0; j < 3; j++) spacing[out[j]] = nii.spacing[j];
  return { dims: nd, frames, affine, spacing };
}
