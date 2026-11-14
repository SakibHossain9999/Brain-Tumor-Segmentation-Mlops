import * as THREE from "three";

export type SliceAxis = "axial" | "coronal" | "sagittal";

// which scanner axis each slice direction is perpendicular to (RAS: 0 = x, 1 = y, 2 = z)
export const AXIS_INDEX: Record<SliceAxis, number> = { sagittal: 0, coronal: 1, axial: 2 };

/** Download one scan file and turn it into raw bytes (handles gzip whether or not the server already unpacked it). */
export async function fetchScan(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const isGzip = bytes[0] === 0x1f && bytes[1] === 0x8b; // every gzip file starts with these two bytes
  if (!isGzip) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Raw scan bytes -> a 3D texture the GPU can sample at any point. */
export function makeVolumeTexture(data: Uint8Array, shape: [number, number, number]) {
  if (data.length !== shape[0] * shape[1] * shape[2]) {
    throw new Error(`scan has ${data.length} bytes, expected ${shape.join(" x ")}`);
  }
  // x varies fastest, then y, then z: the order the pipeline wrote the file in
  const texture = new THREE.Data3DTexture(data, shape[0], shape[1], shape[2]);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter; // smooth between voxels
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

/**
 * An MRI slice inside the 3D scene: a flat square whose pixels are looked up in a scan texture.
 * The square lives in the meshes' own coordinates (scanner millimetres), so it lines up with them.
 * Scans are cropped to the brain, starting at voxel `offset` of the full scan.
 */
export function createSlice(
  shape: [number, number, number],
  offset: [number, number, number],
  affine: number[][],
  bounds: [number[], number[]],
) {
  // affine: voxel index -> millimetres. The shader needs the reverse: millimetres -> voxel index.
  const toMm = new THREE.Matrix4().fromArray(affine.flat()).transpose(); // meta.json stores it row by row
  const toVoxel = toMm.clone().invert();

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uVolume: { value: null as THREE.Data3DTexture | null },
      uToVoxel: { value: toVoxel },
      uShape: { value: new THREE.Vector3(...shape) },
      uOffset: { value: new THREE.Vector3(...offset) },
      uOpacity: { value: 1.0 },
      uBrightness: { value: 0.0 },
      uContrast: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vPos;
      void main() {
        vPos = position;  // millimetres
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      precision highp sampler3D;
      uniform sampler3D uVolume;
      uniform mat4 uToVoxel;
      uniform vec3 uShape, uOffset;
      uniform float uOpacity, uBrightness, uContrast;
      varying vec3 vPos;
      void main() {
        vec3 voxel = (uToVoxel * vec4(vPos, 1.0)).xyz - uOffset;  // mm -> voxel index inside the crop
        vec3 uvw = (voxel + 0.5) / uShape;                         // voxel index -> texture coordinate (0..1)
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) discard;
        float v = texture(uVolume, uvw).r;                         // the MRI brightness at this point
        if (v < 0.02) discard;                                     // outside the brain: leave transparent
        // brightness / contrast ("window / level" in radiology): stretch around mid-grey, then shift
        float shown = clamp((v - 0.5) * uContrast + 0.5 + uBrightness, 0.0, 1.0);
        gl_FragColor = vec4(vec3(shown), uOpacity);
      }`,
    side: THREE.DoubleSide,
  });

  // a square spanning the brain's bounding box, perpendicular to the chosen axis
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.raycast = () => {}; // don't block tumour hover

  /** Show a different scan (FLAIR, T1, ...). */
  function setTexture(texture: THREE.Data3DTexture) {
    material.uniforms.uVolume.value = texture;
  }

  /** Move the slice: `fraction` 0..1 across the brain. Returns its position in millimetres. */
  function update(axis: SliceAxis, fraction: number, opacity: number) {
    const a = AXIS_INDEX[axis];
    const u = (a + 1) % 3;
    const v = (a + 2) % 3;
    const [lo, hi] = bounds;
    const pos = lo[a] + fraction * (hi[a] - lo[a]);
    const corners = [[lo[u], lo[v]], [hi[u], lo[v]], [hi[u], hi[v]], [lo[u], hi[v]]];
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    corners.forEach(([cu, cv], i) => {
      const p = [0, 0, 0];
      p[a] = pos;
      p[u] = cu;
      p[v] = cv;
      attr.setXYZ(i, p[0], p[1], p[2]);
    });
    attr.needsUpdate = true;
    geometry.computeBoundingSphere();
    material.uniforms.uOpacity.value = opacity;
    material.transparent = opacity < 1;
    material.depthWrite = opacity >= 1;
    return pos;
  }

  /** Brightness -0.5..0.5 (0 = as exported), contrast 0.5..3 (1 = as exported). */
  function setLevels(brightness: number, contrast: number) {
    material.uniforms.uBrightness.value = brightness;
    material.uniforms.uContrast.value = contrast;
  }

  return { mesh, update, setTexture, setLevels };
}
