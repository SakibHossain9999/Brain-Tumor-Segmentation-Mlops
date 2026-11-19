import * as THREE from "three";
import GUI from "lil-gui";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { ghostMaterial, hatchMaterial, inkMaterial, outlineMaterial, setHatchColor, toonMaterial } from "./npr";
import { AXIS_INDEX, createSlice, fetchScan, makeVolumeTexture, type SliceAxis } from "./slice";
import { browserModels, fileUrls, loadResult, runInBrowser, saveResult, type BrowserModel, type Saved } from "./browser/client";
import type { Inputs } from "./browser/case";

// ---------------------------------------------------------------- what the bundle files contain
type Zone = { name: string; color: [number, number, number]; volume_ml: number; present: boolean };
type RegionScores = { dice: number | null; hd95: number | null; sensitivity: number | null; precision: number | null };
type Segmentation = {
  label: string;                         // "Expert outline", "SegResNet", ...
  file: string;                          // seg_<source>.glb
  zones: Record<string, Zone>;
  regions_ml: Record<string, number>;    // WT, TC, ET
  model_version?: number;
  champion?: boolean;
  scores?: Record<string, RegionScores>; // this brain's scores against the expert, per region
  dice_mean?: number;
  diff?: { file: string | null; volume_ml: Record<string, number>; min_voxels: number }; // vs the expert
};
type Meta = {
  case_id: string;
  title?: string;
  description?: string;
  center: [number, number, number];
  bounds: [number[], number[]];
  affine: number[][];
  brain: string;
  volumes?: {
    files: Record<string, string>; // scan name ("t2f", "t1c", ...) -> file
    shape: [number, number, number];
    offset: [number, number, number];
    default: string;
  };
  segmentations: Record<string, Segmentation>;
  default_segmentation: string;
};
type GalleryIndex = {
  champion: string;
  models: Record<string, string>;
  cases: { case_id: string; title: string; description: string; tumour_ml: number }[];
};
type TryInfo = {
  samples_url: string;
  samples: { case_id: string; title: string; scans: string; scans_mb: number; label: string }[];
  counts: Record<string, number>; // train / val / test
  test_cases: string[];
  unlabelled: number;
};
/** One tumour zone's look, shared by every segmentation so settings survive switching between them. */
type ZoneStyle = {
  key: string;
  name: string;
  color: THREE.Color;
  settings: { color: string; opacity: number; visible: boolean }; // what the control panel edits
  realistic: THREE.MeshStandardMaterial;
  toon: THREE.MeshToonMaterial;
  hatch: THREE.ShaderMaterial;
};

const URL_PARAMS = new URLSearchParams(location.search);
// the upload service: the same server by default (VITE_API_BASE can point a separately hosted site at it)
const API_BASE = (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE ?? "";
const JOB = URL_PARAMS.get("job"); // ?job=<id>: show the result of an upload
const LOCAL = URL_PARAMS.get("local") === "1"; // ?local=1: show the result computed in this browser
const RUN_LOCALLY_URL = "https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops#run-it-locally";
let localSaved: Saved | undefined; // the in-browser result being shown (?local=1)
let localFiles: Map<string, string> | null = null; // its files, as blob URLs
/** Where a bundle file is: on a server, or (an in-browser result) in this browser's memory. */
const fileUrl = (f: string) => localFiles?.get(f) ?? `${base}/${f}`;
/** Rough time per brain in a browser: about 3x the server's 2-core time (measured: see the README). */
const browserSeconds = (m: BrowserModel) => Math.round((m.cpuSeconds ?? 10) * 3);
const modelLabels = new Map<string, string>(); // model name -> display name (for messages)
/** A link setting if it is one of the allowed values, else the default. */
function pick(value: string | null, allowed: string[], fallback: string) {
  return value && allowed.includes(value) ? value : fallback;
}
const MODALITY_NAMES: Record<string, string> = { t2f: "FLAIR", t1n: "T1", t1c: "T1 + contrast", t2w: "T2" };
const REGION_NAMES: Record<string, string> = { WT: "whole tumour", TC: "tumour core", ET: "active tumour" };
let caseId = URL_PARAMS.get("case") ?? ""; // decided in start(): the link's case, or the gallery's first
let base = "";

// everything the control panel can change (per-zone colour, opacity and visibility live in each ZoneStyle)
const params = {
  seg: "expert",
  compare: false,       // expert-vs-model view
  compareRegion: "WT",
  showMissed: true,     // blue: tumour the model missed
  showExtra: true,      // orange: tumour the model added
  expertOpacity: 0.25,  // the grey expert outline while comparing
  style: pick(URL_PARAMS.get("style"), ["toon", "hatching", "realistic"], "toon"), // ?style=hatching
  bands: 3,
  hatchSpacing: 2, // mm between hatching strokes
  outline: true,
  outlineThickness: 0.6,
  brain: pick(URL_PARAMS.get("brain"), ["ghost", "cutaway", "ink", "plain", "hidden"], "ghost"), // ?brain=hidden
  ghostEdge: 0.55,
  ghostBase: 0.04,
  glow: 0.6,
  glowRadius: 40,
  inkWidth: 0.22,
  cut: "front half",
  slice: URL_PARAMS.get("slice") !== "0", // ?slice=0 hides the MRI slice
  scan: "t2f",
  sliceAxis: "axial" as SliceAxis,
  slicePos: 0.5,
  sliceOpacity: 1,
  sliceBrightness: 0,
  sliceContrast: 1,
  sliceCut: false,
  sliceFlip: false,
  autoRotate: false,
};

// brain opacity, remembered separately for each mode (each mode has its own sensible default)
const brainOpacity: Record<string, number> = { ghost: 1, cutaway: 1, ink: 1, plain: 0.2, hidden: 0 };
const opacityState = { value: brainOpacity[params.brain] }; // what the slider shows: the current mode's value

// ---------------------------------------------------------------- renderer, scene, camera, mouse controls
// the 3D view fills #app, which the page layout sizes: the whole window on a desktop, the top part on a phone
const appEl = document.getElementById("app")!;
const viewSize = (): [number, number] => [appEl.clientWidth || innerWidth, appEl.clientHeight || innerHeight];
const NARROW = matchMedia("(max-width: 1100px), (max-height: 500px)"); // phones and tablets
const TOUCH = matchMedia("(pointer: coarse)").matches;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // sharper than 2x costs a lot on phones
renderer.setSize(...viewSize());
renderer.localClippingEnabled = true; // needed for the cutaway and the slice cut
appEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f1ea);
const camera = new THREE.PerspectiveCamera(35, viewSize()[0] / viewSize()[1], 1, 5000);
const controls = new OrbitControls(camera, renderer.domElement); // drag = rotate, scroll = zoom, right-drag = move
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
keyLight.position.set(1, 1, 2);
camera.add(keyLight); // the light moves with the camera, so the side you look at is lit
scene.add(camera);

// Scanner coordinates are RAS millimetres: x = patient's right, y = front, z = up.
// three.js uses y = up, so rotate -90 degrees around x. Afterwards: front = -z, up = +y, right = +x.
const holder = new THREE.Group();
holder.rotation.x = -Math.PI / 2;
scene.add(holder);
const root = new THREE.Group(); // everything in scanner millimetres, shifted so the brain is centred at 0
holder.add(root);

const zoneStyles = new Map<string, ZoneStyle>();
const segGroups = new Map<string, THREE.Group>(); // tumour meshes per segmentation, loaded when first picked
const diffGroups = new Map<string, THREE.Group>(); // a model's disagreement with the expert, loaded when compared
// comparison colours: blue / orange can be told apart with colour-blindness (red / green can't)
const DIFF_COLORS = { missed: new THREE.Color(0x2f6fdb), extra: new THREE.Color(0xf08a24) };
const diffMats = {
  missed: {
    toon: toonMaterial(DIFF_COLORS.missed, 3),
    realistic: new THREE.MeshStandardMaterial({ color: DIFF_COLORS.missed, roughness: 0.6 }),
    hatch: hatchMaterial(DIFF_COLORS.missed),
  },
  extra: {
    toon: toonMaterial(DIFF_COLORS.extra, 3),
    realistic: new THREE.MeshStandardMaterial({ color: DIFF_COLORS.extra, roughness: 0.6 }),
    hatch: hatchMaterial(DIFF_COLORS.extra),
  },
};
// while comparing, the expert outline is drawn as see-through grey: the reference shape
const expertGrey = new THREE.MeshStandardMaterial({
  color: 0x8a8a8a, transparent: true, opacity: 0.25, depthWrite: false, roughness: 0.8,
});
// which drawn zones make up each region
const REGION_ZONES: Record<string, string[]> = {
  WT: ["edema", "non_enhancing", "enhancing"],
  TC: ["non_enhancing", "enhancing"],
  ET: ["enhancing"],
};
let brainMesh: THREE.Mesh | undefined;
let slice: ReturnType<typeof createSlice> | undefined;
const scanTextures = new Map<string, THREE.Data3DTexture>(); // scans already downloaded
let scanCtrl: ReturnType<GUI["add"]> | undefined;
let segCtrl: ReturnType<GUI["add"]> | undefined;
let shownSeg = ""; // the segmentation currently on screen (the dropdown may briefly say something else)
let runningModel = ""; // a model being run on an upload right now
const NO_EXPERT = "no-expert"; // dropdown value: expert outline not uploaded
let compareCtrl: ReturnType<GUI["add"]> | undefined;
let regionCtrl: ReturnType<GUI["add"]> | undefined;
let greyCtrl: ReturnType<GUI["add"]> | undefined;
let missedCtrl: ReturnType<GUI["add"]> | undefined;
let extraCtrl: ReturnType<GUI["add"]> | undefined;
let meta: Meta | undefined;
const tumourCentre = new THREE.Vector3(); // world coordinates, used by the cutaway planes
const ghost = ghostMaterial();
const ink = inkMaterial();
const plainBrain = new THREE.MeshStandardMaterial({
  color: 0xd9d4cc, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide,
});
const cutawayBrain = toonMaterial(new THREE.Color(0xe8dccb), 3, { side: THREE.DoubleSide });
const outlineMat = outlineMaterial(params.outlineThickness);
let viewDistance = 400;
let fitFactor = 0; // camera distance / brain size that makes the brain fit this view (0 until a case loads)
const loader = new GLTFLoader();

// cutaway: which part of the brain to remove. Each plane keeps the side its normal points to.
const CUTS: Record<string, { normals: THREE.Vector3[]; wedge: boolean }> = {
  "front half": { normals: [new THREE.Vector3(0, 0, 1)], wedge: false },
  "top half": { normals: [new THREE.Vector3(0, -1, 0)], wedge: false },
  "right half": { normals: [new THREE.Vector3(-1, 0, 0)], wedge: false },
  "front-top quarter": { normals: [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, -1, 0)], wedge: true },
};

/** Every tumour mesh of every loaded segmentation. */
function tumourMeshes(onlyCurrent = false): THREE.Mesh[] {
  const out: THREE.Mesh[] = [];
  for (const [source, group] of segGroups) {
    if (onlyCurrent && source !== params.seg) continue;
    group.traverse((o) => {
      if (o instanceof THREE.Mesh && zoneStyles.has(o.name)) out.push(o);
    });
  }
  return out;
}

/** Is the expert-vs-model view on (and possible: a model is shown and the case has an expert outline)? */
function comparing(): boolean {
  const seg = meta?.segmentations[params.seg];
  return params.compare && params.seg !== "expert" && !!seg?.diff && !!meta?.segmentations.expert;
}

/** Keep the link in sync, so it always reopens this exact view. */
function updateUrl() {
  const p = new URLSearchParams(location.search);
  if (!JOB && !LOCAL) p.set("case", caseId);
  p.set("seg", params.seg);
  if (comparing()) p.set("cmp", params.compareRegion);
  else p.delete("cmp");
  history.replaceState(null, "", `?${p}${location.hash}`);
}

/** The material of the chosen tumour style. */
function styled<T extends { toon: THREE.Material; realistic: THREE.Material; hatch: THREE.Material }>(m: T) {
  return params.style === "toon" ? m.toon : params.style === "hatching" ? m.hatch : m.realistic;
}

function updateCut() {
  const cut = CUTS[params.cut];
  // planes pass through the tumour centre, so the cut opens right onto the tumour
  cutawayBrain.clippingPlanes = cut.normals.map((n) => new THREE.Plane(n.clone(), -n.dot(tumourCentre)));
  cutawayBrain.clipIntersection = cut.wedge; // wedge: remove only where BOTH planes cut
  cutawayBrain.needsUpdate = true;
}

/** Move the MRI slice, and (optionally) cut the brain and tumour at it. */
function updateSlice() {
  let planes: THREE.Plane[] = [];
  if (slice) {
    slice.mesh.visible = params.slice && scanTextures.has(params.scan);
    const pos = slice.update(params.sliceAxis, params.slicePos, params.sliceOpacity);
    slice.setLevels(params.sliceBrightness, params.sliceContrast);
    if (params.slice && params.sliceCut) {
      // a plane in scanner millimetres that keeps one side of the slice, then moved to world coordinates
      const normal = new THREE.Vector3();
      normal.setComponent(AXIS_INDEX[params.sliceAxis], params.sliceFlip ? 1 : -1);
      const plane = new THREE.Plane(normal, params.sliceFlip ? -pos : pos);
      root.updateMatrixWorld(true);
      planes = [plane.applyMatrix4(root.matrixWorld)];
    }
  }
  // the cutaway brain keeps its own cut; everything else is cut at the slice
  const materials: THREE.Material[] = [ghost, ink, plainBrain, outlineMat];
  for (const z of zoneStyles.values()) materials.push(z.realistic, z.toon, z.hatch);
  for (const d of Object.values(diffMats)) materials.push(d.toon, d.realistic, d.hatch);
  materials.push(expertGrey);
  for (const m of materials) {
    m.clippingPlanes = planes;
    m.needsUpdate = true;
  }
}

/** Apply the current control-panel settings to every object. */
function applySettings() {
  for (const z of zoneStyles.values()) {
    z.color.setStyle(z.settings.color); // the colour picker gives a CSS colour like "#e62933"
    const opaque = z.settings.opacity >= 1;
    for (const m of [z.realistic, z.toon]) {
      m.color.copy(z.color);
      m.opacity = z.settings.opacity;
      m.transparent = !opaque;
      m.depthWrite = opaque; // see-through zones must not hide the zones inside them
      m.needsUpdate = true;
    }
    setHatchColor(z.hatch, z.color);
    z.hatch.uniforms.uOpacity.value = z.settings.opacity;
    z.hatch.transparent = !opaque;
    z.hatch.depthWrite = opaque;
  }
  for (const m of [...[...zoneStyles.values()].map((z) => z.hatch), diffMats.missed.hatch, diffMats.extra.hatch]) {
    m.uniforms.uSpacing.value = params.hatchSpacing;
  }
  const cmp = comparing();
  const shownSource = cmp ? "expert" : params.seg; // comparing: the expert outline is the grey reference
  for (const [source, group] of segGroups) group.visible = source === shownSource;
  for (const mesh of tumourMeshes()) {
    const z = zoneStyles.get(mesh.name)!;
    if (cmp) {
      mesh.material = expertGrey;
      mesh.visible = REGION_ZONES[params.compareRegion].includes(mesh.name) && params.expertOpacity > 0;
      mesh.children[0].visible = false; // no ink outline around the grey reference
      continue;
    }
    mesh.material = styled(z);
    mesh.visible = z.settings.visible;
    // an ink shell behind a see-through zone would look like a dark blob
    mesh.children[0].visible = params.outline && z.settings.opacity >= 1;
  }
  expertGrey.opacity = params.expertOpacity;
  for (const [source, group] of diffGroups) {
    group.visible = cmp && source === params.seg;
    group.traverse((o) => {
      if (!(o instanceof THREE.Mesh) || !o.userData.kind) return;
      const kind = o.userData.kind as "missed" | "extra";
      o.material = styled(diffMats[kind]);
      const wanted = kind === "missed" ? params.showMissed : params.showExtra;
      o.visible = wanted && o.name.startsWith(`${params.compareRegion}_`); // chosen kind and region only
      o.children[0].visible = params.outline;
    });
  }
  zoneFolder.show(!cmp); // zone colours don't apply while comparing
  outlineMat.uniforms.uThickness.value = params.outlineThickness;
  if (brainMesh) {
    const materials: Record<string, THREE.Material> = { ghost, ink, plain: plainBrain, cutaway: cutawayBrain };
    brainMesh.visible = params.brain !== "hidden";
    brainMesh.material = materials[params.brain] ?? ghost;
    // see-through brains draw after the tumour, so the tumour shows through them
    const solidCutaway = params.brain === "cutaway" && brainOpacity.cutaway >= 1;
    brainMesh.renderOrder = solidCutaway ? 0 : 1;
  }
  ghost.uniforms.uEdge.value = params.ghostEdge;
  ghost.uniforms.uBase.value = params.ghostBase;
  ghost.uniforms.uFocusStrength.value = params.glow;
  ghost.uniforms.uFocusRadius.value = params.glowRadius;
  ink.uniforms.uWidth.value = params.inkWidth;
  ghost.uniforms.uOpacity.value = brainOpacity.ghost;
  ink.uniforms.uOpacity.value = brainOpacity.ink;
  plainBrain.opacity = brainOpacity.plain;
  cutawayBrain.opacity = brainOpacity.cutaway;
  cutawayBrain.transparent = brainOpacity.cutaway < 1;
  cutawayBrain.depthWrite = brainOpacity.cutaway >= 1;
  updateCut();
  updateSlice();
  controls.autoRotate = params.autoRotate;
  updatePanel();
  renderInfo();
}

function setView(direction: THREE.Vector3) {
  camera.position.copy(direction.normalize().multiplyScalar(viewDistance));
  controls.target.set(0, 0, 0);
  controls.update();
}

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? "–" : v.toFixed(digits));

/** The info box: case, which segmentation is shown, each zone's volume, and (for a model) its scores here. */
function renderInfo() {
  if (!meta) return;
  const seg = meta.segmentations[params.seg];
  if (!seg) return;
  if (comparing()) return renderCompareInfo(seg);
  const rows = [...zoneStyles.values()]
    .map((z) => {
      const zone = seg.zones[z.key];
      return `<div><span class="dot" style="background: ${z.settings.color}"></span>${zone.name}: ${zone.volume_ml} mL</div>`;
    })
    .join("");
  let scores = "";
  if (seg.scores) {
    const parts = Object.entries(seg.scores)
      .map(([r, s]) => `<span title="${REGION_NAMES[r] ?? r}: HD95 ${fmt(s.hd95, 1)} mm">${r} <b>${fmt(s.dice)}</b></span>`)
      .join("");
    scores = `<div class="scores">Dice vs expert on this brain<br>${parts}</div>`;
  }
  const badge = seg.champion ? `<span class="badge" title="the app's default model (MLflow alias @champion)">champion</span>` : "";
  const heading = caseHeading();
  const desc = meta.description ? `<div class="muted">${meta.description}</div>` : "";
  document.getElementById("info")!.innerHTML =
    `${heading}${desc}<div>Showing: <b>${seg.label}</b>${badge}</div>${rows}${scores}` +
    (TOUCH ? `<div class="hint">drag to rotate · pinch to zoom · tap the tumour</div>`
      : `<div class="hint">drag to rotate · scroll to zoom · hover the tumour</div>`);
}

/** Title and case id (an unrecognised upload has no id worth showing). */
function caseHeading() {
  const id = meta!.case_id === "upload" ? "" : meta!.case_id;
  return meta!.title ? `<b>${meta!.title}</b> <small>${id}</small>` : `<b>${id}</b>`;
}

/** The info box in the expert-vs-model view: what each colour means, and how much of each there is. */
function renderCompareInfo(seg: Segmentation) {
  const r = params.compareRegion;
  const ml = seg.diff!.volume_ml;
  const s = seg.scores?.[r];
  const expertMl = meta!.segmentations.expert.regions_ml[r];
  const heading = caseHeading() + (meta!.description ? `<div class="muted">${meta!.description}</div>` : "");
  const dot = (c: string) => `<span class="dot" style="background: ${c}"></span>`;
  document.getElementById("info")!.innerHTML =
    `${heading}<div>Comparing <b>${seg.label}</b> with the expert</div>` +
    `<div class="muted">Region: ${REGION_NAMES[r]}</div>` +
    `<div>${dot("#8a8a8a")}Expert outline: ${expertMl} mL</div>` +
    `<div style="opacity: ${params.showMissed ? 1 : 0.4}">${dot("#2f6fdb")}Missed by the model: ${ml[`${r}_missed`]} mL</div>` +
    `<div style="opacity: ${params.showExtra ? 1 : 0.4}">${dot("#f08a24")}Extra, not in the expert outline: ${ml[`${r}_extra`]} mL</div>` +
    `<div class="scores">Dice <b>${fmt(s?.dice)}</b> &nbsp; HD95 <b>${fmt(s?.hd95, 1)} mm</b></div>` +
    `<div class="hint">pieces smaller than ${seg.diff!.min_voxels} voxels (thin slivers along the edges) are not drawn</div>`;
}

/** The sample gallery: test brains to pick from (needs cases/index.json). */
function renderGallery(index: GalleryIndex) {
  const el = document.getElementById("cases")!;
  const items = index.cases
    .map((c) => {
      const cls = c.case_id === caseId ? "case current" : "case";
      return `<button class="${cls}" data-case="${c.case_id}" title="${c.description}">` +
        `${c.title} <small>· ${c.tumour_ml} mL · ${c.case_id}</small></button>`;
    })
    .join("");
  el.innerHTML = `<h3>Sample cases (test set)</h3>${items}`;
  el.hidden = false;
  el.querySelectorAll<HTMLButtonElement>("button.case").forEach((b) =>
    b.addEventListener("click", () => {
      const p = new URLSearchParams({ case: b.dataset.case!, seg: params.seg });
      location.search = p.toString(); // open the other case (keeps the chosen segmentation)
    }),
  );
}

/** "Try your own scans": which files make a fair test, sample downloads (upload arrives with the API). */
function renderTry(info: TryInfo) {
  const el = document.getElementById("try")!;
  const url = (f: string) => `${info.samples_url}/${f}`;
  const samples = info.samples
    .map((s) =>
      `<div class="sample">${s.title} <small>${s.case_id}</small><br>` +
      `<a href="${url(s.scans)}" download>scans (${s.scans_mb} MB)</a>` +
      `<a href="${url(s.label)}" download>expert outline</a></div>`)
    .join("");
  const ids = info.test_cases.map((c) => `<span>${c}</span>`).join("");
  el.innerHTML =
    `<details><summary>Try your own scans</summary>` +
    `<div class="muted">Upload 4 MRI scans of a BraTS-style brain (FLAIR, T1, T1 + contrast, T2): skull removed, ` +
    `scans aligned, 1 mm voxels. Add its expert outline to get Dice scores and the comparison view.</div>` +
    `<div id="upload"></div>` +
    `<h4>Sample brains (test set)</h4>${samples}` +
    `<h4>Testing with our dataset (MSD brain tumours)?</h4><ul>` +
    `<li><b>Test</b> brains (${info.counts.test}): the models never saw them: a fair check.</li>` +
    `<li><b>Training</b> (${info.counts.train}) and <b>validation</b> (${info.counts.val}) brains: ` +
    `used in training, so scores look better than they really are.</li>` +
    `<li><b>Unlabelled</b> brains (${info.unlabelled}, the dataset's imagesTs): never used, but no expert outline.</li>` +
    `</ul><div class="muted">The app recognises brains from this dataset and tells you which group ` +
    `they belong to. Full split: splits/msd_split.json in the repository.</div>` +
    `<details><summary>Show the ${info.counts.test} test brains</summary><div class="ids">${ids}</div></details>` +
    `</details>`;
  el.hidden = false;
  // a link ending in #try opens the panel straight away (e.g. from the README)
  if (location.hash === "#try") el.querySelector("details")!.open = true;
  void setupUpload(el.querySelector<HTMLElement>("#upload")!);
}

/** The upload form: two boxes (scans, optional outline) to drop files on or click, the models, progress. */
async function setupUpload(el: HTMLElement) {
  type ApiModel = { name: string; display_name: string; cpu_s_per_case_2_threads: number | null; champion: boolean };
  type Choice = { name: string; label: string; champion: boolean; seconds: number | null; localOnly?: boolean };
  // Where the models run: the upload service when there is one (the local app: all models), otherwise right
  // here in the browser (the website: the models small enough for a browser).
  let engine: "server" | "browser" = "server";
  let models: Choice[];
  let browser: Awaited<ReturnType<typeof browserModels>> = null;
  try {
    const r = await fetch(`${API_BASE}/api/models`);
    if (!r.ok) throw new Error(String(r.status));
    models = ((await r.json()) as { models: ApiModel[] }).models.map((m) =>
      ({ name: m.name, label: m.display_name, champion: m.champion, seconds: m.cpu_s_per_case_2_threads }));
  } catch {
    browser = await browserModels();
    if (!browser || typeof Worker === "undefined" || typeof indexedDB === "undefined") {
      el.innerHTML = `<div class="muted">The upload service isn't running here, so uploading is off. Everything else works.</div>`;
      return;
    }
    engine = "browser";
    models = [...browser.runnable.map((m) => ({ name: m.name, label: m.label, champion: !!m.champion, seconds: browserSeconds(m) })),
      ...browser.localOnly.map((m) => ({ ...m, seconds: null, localOnly: true }))];
  }
  const modelRow = (m: Choice) => m.localOnly
    ? `<div class="up-local"><span>${m.label}</span> <small>· not in the browser</small>` +
      `<div class="up-note">Too large for a web browser: it needs about 12 GB of memory while it runs. Its results are in ` +
      `the sample gallery. To use it on your own scans, <a href="${RUN_LOCALLY_URL}" target="_blank" rel="noopener">run the ` +
      `app locally</a>, where all 3 models work.</div></div>`
    : `<label><input type="checkbox" value="${m.name}" ${m.champion ? "checked" : ""}> ${m.champion ? "★ " : ""}${m.label}` +
      ` <small>about ${Math.max(1, Math.round(m.seconds ?? 10))} s</small></label>`;
  const SLOTS: [string, string][] = [["flair", "FLAIR"], ["t1", "T1"], ["t1ce", "T1 + contrast"], ["t2", "T2"]];
  el.innerHTML =
    `<h4>Upload</h4>` +
    (engine === "browser"
      ? `<div class="up-mode browser">Runs in your browser: your scans never leave your computer.</div>`
      : `<div class="up-mode server">Runs on the upload service: all models available.</div>`) +
    `<div class="up-box" data-box="scans" tabindex="0"><b>Scans</b>` +
    `<small>drop 1 file with all 4 scans (MSD) or 4 files: FLAIR, T1, T1 + contrast, T2 (BraTS), or click to choose</small>` +
    `<div class="up-files"></div></div>` +
    `<div class="up-box" data-box="label" tabindex="0"><b>Expert outline</b> <span class="opt">(optional)</span>` +
    `<small>drop the label file, or click to choose. Adds Dice scores and the comparison view</small>` +
    `<div class="up-files"></div></div>` +
    `<input type="file" data-input="scans" accept=".nii,.gz" multiple hidden>` +
    `<input type="file" data-input="label" accept=".nii,.gz" hidden>` +
    `<label class="up-fmt" hidden>Outline numbering <select>` +
    `<option value="auto">detect</option><option value="msd">MSD: 1 swelling, 2 core, 3 active</option>` +
    `<option value="brats">BraTS: 1 core, 2 swelling, 3 or 4 active</option></select></label>` +
    `<div class="up-models"><div>Models</div>${models.map(modelRow).join("")}</div>` +
    `<button class="up-go">Segment</button>` +
    `<div class="up-progress" hidden><div class="bar"><div></div></div><div class="step"></div></div>` +
    `<div class="up-error"></div>` +
    (engine === "browser"
      ? `<div class="muted">It runs on your computer's processor, so it takes a little while. The result is kept in this ` +
        `browser (the latest upload only), so you can run the other model on it afterwards.</div>`
      : `<div class="muted">Uploaded scans are kept with the result for about 6 hours, so you can run more models on them, then both are deleted.</div>`);

  const $q = <T extends Element>(s: string) => el.querySelector<T>(s)!;
  const error = (msg: string) => ($q<HTMLElement>(".up-error").textContent = msg);
  const mb = (f: File) => `${(f.size / 1e6).toFixed(f.size < 1e6 ? 2 : 1)} MB`;

  // what has been chosen so far
  let scans: { mode: "one"; file: File } | { mode: "four"; files: Record<string, File> } | null = null;
  let label: File | null = null;

  /** BraTS file names say which scan they are: *_flair, *_t1ce (2021) or *-t2f, *-t1c (2023), ... */
  const slotOf = (name: string): string | null => {
    const t = name.toLowerCase().replace(/\.nii(\.gz)?$/, "").split(/[^a-z0-9]+/);
    const has = (...w: string[]) => w.some((x) => t.includes(x));
    if (has("flair", "t2f")) return "flair";
    if (has("t1ce", "t1c", "t1gd")) return "t1ce";
    if (has("t2", "t2w")) return "t2";
    if (has("t1", "t1n", "t1w")) return "t1";
    return null;
  };
  // a label is mostly zeros: 0.01-0.06 MB compressed, while scan files are several MB
  const looksLikeLabel = (f: File) => f.size < 1e6 || /(^|[^a-z])(seg|label|mask)([^a-z]|$)/i.test(f.name);

  function show() {
    const scanList = !scans ? "" : scans.mode === "one"
      ? `${scans.file.name} <small>(${mb(scans.file)}, all 4 scans)</small>`
      : SLOTS.map(([s, l]) => `${l}: ${(scans as { files: Record<string, File> }).files[s].name}`).join("<br>");
    $q<HTMLElement>('[data-box="scans"] .up-files').innerHTML = scanList;
    $q<HTMLElement>('[data-box="label"] .up-files').innerHTML = label
      ? `${label.name} <small>(${mb(label)})</small> · <a href="#" class="up-clear">remove</a>` : "";
    $q<HTMLElement>('[data-box="scans"]').classList.toggle("filled", !!scans);
    $q<HTMLElement>('[data-box="label"]').classList.toggle("filled", !!label);
    $q<HTMLElement>(".up-fmt").hidden = !label;
    el.querySelector(".up-clear")?.addEventListener("click", (e) => {
      e.preventDefault();
      label = null;
      show();
    });
  }

  function takeScans(files: File[]) {
    error("");
    files = files.filter((f) => /\.nii(\.gz)?$/i.test(f.name));
    if (!files.length) return error("Please choose NIfTI files (.nii or .nii.gz).");
    if (files.length === 1) {
      if (looksLikeLabel(files[0])) {
        return error(`${files[0].name} (${mb(files[0])}) looks like a label file, not scans. Drop it in the Expert outline box.`);
      }
      scans = { mode: "one", file: files[0] };
    } else if (files.length === 4) {
      const sorted: Record<string, File> = {};
      for (const f of files) {
        const slot = slotOf(f.name);
        if (!slot || sorted[slot]) {
          return error("Couldn't tell which of the 4 files is which scan. Their names should contain flair / t1 / t1ce / t2 " +
            "(or t2f / t1n / t1c / t2w), as in BraTS.");
        }
        sorted[slot] = f;
      }
      scans = { mode: "four", files: sorted };
    } else {
      return error(`Got ${files.length} files. Drop either 1 file with all 4 scans, or the 4 separate scans together.`);
    }
    show();
  }

  function takeLabel(files: File[]) {
    error("");
    if (files.length !== 1) return error("Drop one label file in the Expert outline box.");
    const f = files[0];
    if (!/\.nii(\.gz)?$/i.test(f.name)) return error("Please choose a NIfTI file (.nii or .nii.gz).");
    if (!looksLikeLabel(f)) {
      return error(`${f.name} (${mb(f)}) looks like scans, not a label file (labels are under 1 MB). Drop it in the Scans box.`);
    }
    label = f;
    show();
  }

  // each box: drop files on it, or click (or press Enter) to choose
  for (const box of ["scans", "label"] as const) {
    const zone = $q<HTMLElement>(`[data-box="${box}"]`);
    const input = $q<HTMLInputElement>(`input[data-input="${box}"]`);
    const take = (files: File[]) => (box === "scans" ? takeScans(files) : takeLabel(files));
    zone.addEventListener("click", (e) => {
      if (!(e.target as HTMLElement).closest(".up-clear")) input.click();
    });
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") input.click();
    });
    input.addEventListener("change", () => {
      take([...(input.files ?? [])]);
      input.value = ""; // so choosing the same file again still counts
    });
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("over");
      take([...(e.dataTransfer?.files ?? [])]);
    });
  }

  const button = $q<HTMLButtonElement>(".up-go");
  button.addEventListener("click", async () => {
    error("");
    if (!scans) return error("Add the scans first (drop them on the Scans box, or click it).");
    if (engine === "browser") return runHere();
    const form = new FormData();
    if (scans.mode === "one") form.append("scans", scans.file);
    else for (const [s] of SLOTS) form.append(s, scans.files[s]);
    if (label) {
      form.append("label", label);
      form.append("label_format", $q<HTMLSelectElement>(".up-fmt select").value);
    }
    const chosen = [...el.querySelectorAll<HTMLInputElement>(".up-models input:checked")].map((c) => c.value);
    if (!chosen.length) return error("Pick at least one model.");
    form.append("models", chosen.join(","));

    button.disabled = true;
    const progress = $q<HTMLElement>(".up-progress");
    const bar = $q<HTMLElement>(".up-progress .bar div");
    const step = $q<HTMLElement>(".up-progress .step");
    progress.hidden = false;
    step.textContent = "Uploading…";
    bar.style.width = "2%";
    try {
      // XMLHttpRequest instead of fetch: it reports how much of the upload has been sent
      const { status, body } = await new Promise<{ status: number; body: any }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/api/segment`);
        xhr.upload.onprogress = (e) => {
          if (!e.lengthComputable) return;
          const m = (n: number) => (n / 1e6).toFixed(1);
          bar.style.width = `${Math.max(2, (e.loaded / e.total) * 100)}%`;
          step.textContent = `Uploading… ${Math.round((e.loaded / e.total) * 100)}% (${m(e.loaded)} of ${m(e.total)} MB)`;
        };
        xhr.onload = () => {
          let parsed: any = {};
          try { parsed = JSON.parse(xhr.responseText); } catch { /* not JSON: keep {} */ }
          resolve({ status: xhr.status, body: parsed });
        };
        xhr.onerror = () => reject(new Error("The upload failed: the server could not be reached."));
        xhr.send(form);
      });
      if (status !== 202) throw new Error(body.detail ?? `The upload failed (HTTP ${status}).`);
      // follow the job until it is done, then open the result like any other case
      for (;;) {
        await new Promise((res) => setTimeout(res, 700));
        const job = await (await fetch(`${API_BASE}/api/jobs/${body.job_id}`)).json();
        bar.style.width = `${Math.max(2, job.progress * 100)}%`;
        step.textContent = job.status === "queued" ? `Waiting: ${job.position} job(s) ahead` : job.step;
        if (job.status === "failed") throw new Error(job.error);
        if (job.status === "done") {
          location.href = `?job=${body.job_id}`;
          return;
        }
      }
    } catch (e) {
      progress.hidden = true;
      button.disabled = false;
      error(e instanceof Error ? e.message : String(e));
    }
  });

  /** The website: run the chosen models in this browser (a Web Worker), keep the result here, then show it. */
  async function runHere() {
    const chosen = [...el.querySelectorAll<HTMLInputElement>(".up-models input:checked")].map((c) => c.value);
    if (!chosen.length) return error("Pick at least one model.");
    const specs = browser!.runnable.filter((m) => chosen.includes(m.name));
    const progress = $q<HTMLElement>(".up-progress");
    const bar = $q<HTMLElement>(".up-progress .bar div");
    const step = $q<HTMLElement>(".up-progress .step");
    const show = (text: string, p: number) => {
      step.textContent = text;
      bar.style.width = `${Math.max(2, p * 100)}%`;
    };
    button.disabled = true;
    progress.hidden = false;
    show("Reading the files…", 0.01);
    try {
      const s = scans!;
      const inputs: Inputs = {
        scans: s.mode === "one" ? { one: await s.file.arrayBuffer() }
          : { four: Object.fromEntries(await Promise.all(SLOTS.map(async ([k]) => [k, await s.files[k].arrayBuffer()] as const))) },
        label: label ? await label.arrayBuffer() : undefined,
        labelFormat: $q<HTMLSelectElement>(".up-fmt select").value as Inputs["labelFormat"],
      };
      const result = await runInBrowser(inputs, specs, show);
      show("Saving the result in this browser…", 1);
      await saveResult({ inputs, ...result, savedAt: Date.now() });
      location.href = "?local=1";
    } catch (e) {
      progress.hidden = true;
      button.disabled = false;
      error(e instanceof Error ? e.message : String(e));
    }
  }
}

/** Load one segmentation's tumour meshes (once) and give them the shared zone looks. */
async function loadSegmentation(source: string) {
  if (segGroups.has(source) || !meta) return;
  const gltf = await loader.loadAsync(fileUrl(meta.segmentations[source].file));
  const group = gltf.scene;
  group.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh) || !zoneStyles.has(obj.name)) return;
    // shading, the ghost glow and the outlines all need normals; older bundles may lack them
    if (!obj.geometry.getAttribute("normal")) obj.geometry.computeVertexNormals();
    const outline = new THREE.Mesh(obj.geometry, outlineMat);
    outline.raycast = () => {}; // the mouse should hit the zone itself, not its outline shell
    obj.add(outline);
  });
  segGroups.set(source, group);
  root.add(group);
}

/** Load a model's disagreement meshes (once): <REGION>_missed and <REGION>_extra. */
async function loadDiff(source: string) {
  const diff = meta?.segmentations[source]?.diff;
  if (diffGroups.has(source) || !diff?.file) return;
  const group = (await loader.loadAsync(fileUrl(diff.file))).scene;
  // collect first, then add outlines: adding children while walking the tree would visit them too
  const meshes: THREE.Mesh[] = [];
  group.traverse((obj) => {
    if (obj instanceof THREE.Mesh) meshes.push(obj);
  });
  meshes.forEach((obj) => {
    if (!obj.geometry.getAttribute("normal")) obj.geometry.computeVertexNormals();
    obj.userData.kind = obj.name.endsWith("_missed") ? "missed" : "extra";
    const outline = new THREE.Mesh(obj.geometry, outlineMat);
    outline.raycast = () => {};
    obj.add(outline);
  });
  diffGroups.set(source, group);
  root.add(group);
}

/** A message above the info box (e.g. why the expert outline can't be shown, or a model's progress). */
let noticeTimer = 0;
function notice(html: string, kind: "info" | "error" = "info", sticky = false) {
  const el = document.getElementById("notice")!;
  el.innerHTML = html;
  el.className = `card ${kind === "error" ? "error" : ""}`;
  el.hidden = false;
  clearTimeout(noticeTimer);
  if (!sticky) noticeTimer = window.setTimeout(() => (el.hidden = true), 9000);
}

/** The dropdown changed: show it, explain why it can't be shown, or run the model first. */
async function onSegChange() {
  const choice = params.seg;
  if (choice === NO_EXPERT || choice.startsWith("run:") || choice.startsWith("localonly:")) {
    params.seg = shownSeg; // keep showing what was there
    segCtrl?.updateDisplay();
  }
  if (choice === NO_EXPERT) {
    notice("<b>No expert outline for this upload.</b> Only the scans were uploaded, so there is no expert outline " +
      "to show, and no scores. To compare the models with an expert, upload the scans again together with " +
      "the outline (label) file.", "error");
    return;
  }
  if (choice.startsWith("localonly:")) {
    const label = modelLabels.get(choice.slice(10)) ?? "This model";
    notice(`<b>${label} runs only in the local app.</b> It needs about 12 GB of memory while it works, more than a web ` +
      "browser allows, so this website can't run it on your scans. Its results are in the sample gallery. To use it on " +
      `your own scans, <a href="${RUN_LOCALLY_URL}" target="_blank" rel="noopener">run the app locally</a>: all 3 models work there.`,
    "error");
    return;
  }
  if (choice.startsWith("run:")) return runExtraModel(choice.slice(4));
  await showSegmentation();
}

/** Run one more model on this upload (on the server), then reopen the result with it. */
async function runExtraModel(name: string) {
  if (runningModel) {
    notice(`Please wait until ${runningModel} has finished.`, "error");
    return;
  }
  runningModel = name;
  const bar = (p: number, text: string) =>
    notice(`${text}<div class="bar"><div style="width: ${Math.max(2, p * 100)}%"></div></div>`, "info", true);
  if (LOCAL) return runExtraModelHere(name, bar);
  try {
    bar(0, "Starting…");
    const form = new FormData();
    form.append("model", name);
    const r = await fetch(`${API_BASE}/api/jobs/${JOB}/models`, { method: "POST", body: form });
    const body = await r.json();
    if (r.status !== 202) throw new Error(body.detail ?? `HTTP ${r.status}`);
    for (;;) {
      await new Promise((res) => setTimeout(res, 700));
      const job = await (await fetch(`${API_BASE}/api/jobs/${body.job_id}`)).json();
      if (job.status === "failed") throw new Error(job.error);
      bar(job.progress, job.status === "queued" ? `Waiting: ${job.position} job(s) ahead` : job.step);
      if (job.status === "done") {
        location.href = `?job=${JOB}&seg=${name}`; // reopen the upload, now with this model
        return;
      }
    }
  } catch (e) {
    notice(`<b>Could not run the model.</b> ${e instanceof Error ? e.message : e}`, "error");
    runningModel = "";
  }
}

/** In-browser result: run one more model here, on the scans kept in this browser, then reopen the result. */
async function runExtraModelHere(name: string, bar: (p: number, text: string) => void) {
  try {
    const spec = (await browserModels())?.runnable.find((m) => m.name === name);
    if (!spec || !localSaved) throw new Error("this model isn't available here.");
    bar(0, "Starting…");
    const result = await runInBrowser(localSaved.inputs, [spec], (step, p) => bar(p, step),
      { meta: localSaved.meta, files: localSaved.files });
    bar(1, "Saving the result in this browser…");
    await saveResult({ ...localSaved, ...result, savedAt: Date.now() });
    location.href = `?local=1&seg=${name}`;
  } catch (e) {
    notice(`<b>Could not run the model.</b> ${e instanceof Error ? e.message : e}`, "error");
    runningModel = "";
  }
}

/** Show the chosen segmentation, loading it the first time. */
async function showSegmentation() {
  const source = params.seg;
  await loadSegmentation(source);
  if (params.compare && meta?.segmentations.expert && source !== "expert") {
    await Promise.all([loadSegmentation("expert"), loadDiff(source)]); // grey reference + disagreement
  }
  if (source !== params.seg) return; // the user picked another one while this was loading
  shownSeg = source; // now on screen
  // tumour centre: for the proximity glow (brain's own coordinates) and the cutaway (world)
  const box = new THREE.Box3();
  for (const mesh of tumourMeshes(true)) {
    mesh.geometry.computeBoundingBox();
    box.union(mesh.geometry.boundingBox!.clone().applyMatrix4(mesh.matrix));
  }
  if (!box.isEmpty() && brainMesh) {
    const c = box.getCenter(new THREE.Vector3());
    ghost.uniforms.uFocus.value.copy(c.clone().applyMatrix4(brainMesh.matrix.clone().invert()));
    scene.updateMatrixWorld(true);
    tumourCentre.copy(c.applyMatrix4(root.matrixWorld));
  }
  updateUrl();
  applySettings();
}

async function load() {
  meta = LOCAL ? (structuredClone(localSaved!.meta) as Meta) : ((await (await fetch(`${base}/meta.json`)).json()) as Meta);
  if (!meta.segmentations) throw new Error("old bundle format: rebuild it with the pipeline");
  root.position.set(-meta.center[0], -meta.center[1], -meta.center[2]); // centre the brain at 0

  const brainScene = (await loader.loadAsync(fileUrl(meta.brain))).scene;
  brainScene.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      if (!obj.geometry.getAttribute("normal")) obj.geometry.computeVertexNormals();
      brainMesh = obj;
    }
  });
  root.add(brainScene);

  // zone looks: same zones (and default colours) in every segmentation
  const first = meta.segmentations[meta.default_segmentation];
  for (const [key, zone] of Object.entries(first.zones)) {
    const color = new THREE.Color().setRGB(...zone.color, THREE.SRGBColorSpace);
    const style: ZoneStyle = {
      key,
      name: zone.name,
      color,
      // the swelling surrounds everything else, so it starts half see-through
      settings: { color: `#${color.getHexString(THREE.SRGBColorSpace)}`, opacity: key === "edema" ? 0.45 : 1, visible: true },
      realistic: new THREE.MeshStandardMaterial({ color, roughness: 0.6 }),
      toon: toonMaterial(color, params.bands),
      hatch: hatchMaterial(color, params.hatchSpacing),
    };
    zoneStyles.set(key, style);
    const zoneControls = zoneFolder.addFolder(zone.name);
    zoneControls.add(style.settings, "visible").name("Show").onChange(applySettings);
    zoneControls.addColor(style.settings, "color").name("Colour").onChange(applySettings);
    zoneControls.add(style.settings, "opacity", 0.05, 1, 0.01).name("Opacity").onChange(applySettings);
    zoneControls.close(); // keep the panel short: open a zone to edit it
  }

  // segmentation dropdown: expert outline + each model (champion marked)
  const options: Record<string, string> = {};
  type UploadModel = { name: string; label: string; champion: boolean; seconds: number | null; localOnly?: boolean };
  let uploadModels: UploadModel[] | null = null;
  if (JOB) {
    const api = await fetch(`${API_BASE}/api/models`).then((r) => (r.ok ? r.json() : null)).then((d) => d?.models ?? null).catch(() => null);
    uploadModels = api?.map((m: { name: string; display_name: string; champion: boolean; cpu_s_per_case_2_threads: number | null }) =>
      ({ name: m.name, label: m.display_name, champion: m.champion, seconds: m.cpu_s_per_case_2_threads })) ?? null;
  } else if (LOCAL) {
    const bm = await browserModels();
    if (bm) {
      uploadModels = [...bm.runnable.map((m) => ({ name: m.name, label: m.label, champion: !!m.champion, seconds: browserSeconds(m) })),
        ...bm.localOnly.map((m) => ({ ...m, seconds: null, localOnly: true }))];
    }
  }
  if ((JOB || LOCAL) && !meta.segmentations.expert) options["Expert outline (not uploaded)"] = NO_EXPERT;
  for (const [source, seg] of Object.entries(meta.segmentations)) {
    if (uploadModels && source !== "expert") continue; // uploads: models listed below, in a fixed order
    options[seg.champion ? `★ ${seg.label}` : seg.label] = source; // ★ = champion (the dropdown is narrow)
  }
  for (const m of uploadModels ?? []) {
    const name = `${m.champion ? "★ " : ""}${m.label}`;
    modelLabels.set(m.name, m.label);
    if (meta.segmentations[m.name]) options[name] = m.name;
    // in-browser results: a model too large for a browser runs only in the local app
    else if (m.localOnly) options[`${name}: only when run locally`] = `localonly:${m.name}`;
    // not run on this upload yet: picking it runs it
    else options[`${name}: run it (about ${Math.max(1, Math.round(m.seconds ?? 10))} s)`] = `run:${m.name}`;
  }
  const wanted = URL_PARAMS.get("seg");
  params.seg = wanted && wanted in meta.segmentations ? wanted : meta.default_segmentation;
  segCtrl = segFolder.add(params, "seg", options).name("Show").onChange(onSegChange);
  const cmpParam = URL_PARAMS.get("cmp"); // e.g. ?cmp=ET opens the comparison of the active tumour
  if (cmpParam && cmpParam in REGION_ZONES) {
    params.compare = true;
    params.compareRegion = cmpParam;
  }
  compareCtrl = segFolder.add(params, "compare").name("Compare with expert").onChange(showSegmentation);
  regionCtrl = segFolder
    .add(params, "compareRegion", { "whole tumour": "WT", "tumour core": "TC", "active tumour": "ET" })
    .name("Region")
    .onChange(() => {
      applySettings();
      updateUrl();
    });
  missedCtrl = segFolder.add(params, "showMissed").name("Show missed (blue)").onChange(applySettings);
  extraCtrl = segFolder.add(params, "showExtra").name("Show extra (orange)").onChange(applySettings);
  greyCtrl = segFolder.add(params, "expertOpacity", 0, 0.8, 0.01).name("Expert outline (grey)").onChange(applySettings);
  if (Object.keys(options).length < 2 && !JOB && !LOCAL) segFolder.hide(); // nothing to choose between (uploads always show it)

  // the MRI scans for the slicing plane: only the default one now, the others when picked
  if (meta.volumes) {
    slice = createSlice(meta.volumes.shape, meta.volumes.offset, meta.affine, meta.bounds);
    root.add(slice.mesh);
    const wantedScan = URL_PARAMS.get("scan"); // e.g. ?scan=t1c
    params.scan = wantedScan && wantedScan in meta.volumes.files ? wantedScan : meta.volumes.default;
    const scans: Record<string, string> = {};
    for (const name of Object.keys(meta.volumes.files)) scans[MODALITY_NAMES[name] ?? name] = name;
    scanCtrl = sliceFolder.add(params, "scan", scans).name("Scan").onChange(showScan);
    // right under Show: you pick the scan before deciding where to cut
    sliceShowCtrl.domElement.after(scanCtrl.domElement);
    void showScan(); // runs in the background: the 3D view appears without waiting for the scan
  } else {
    sliceFolder.hide();
  }

  const size = Math.max(...meta.bounds[1].map((v, i) => v - meta!.bounds[0][i]));
  const zoom = Number(URL_PARAMS.get("zoom")) || 1; // ?zoom=2 starts twice as close
  // far enough that the brain fits the height (2.4) and, in a tall narrow view such as a phone, the width too
  fitFactor = Math.max(2.4, 1.35 / camera.aspect);
  viewDistance = (size * fitFactor) / Math.min(Math.max(zoom, 0.5), 6);
  await showSegmentation();
  setView(new THREE.Vector3(0, 0.3, -1)); // start facing the patient's face
}

/** Show the chosen scan on the slice, downloading it the first time (about 1.5 MB). */
async function showScan() {
  if (!meta?.volumes || !slice) return;
  const name = params.scan;
  if (!scanTextures.has(name)) {
    const data = await fetchScan(fileUrl(meta.volumes.files[name]));
    scanTextures.set(name, makeVolumeTexture(data, meta.volumes.shape));
  }
  if (name !== params.scan) return; // the user picked another scan while this one was downloading
  slice.setTexture(scanTextures.get(name)!);
  applySettings();
}

// ---------------------------------------------------------------- hover: show which zone is under the mouse
const tooltip = document.createElement("div");
tooltip.className = "tooltip";
document.body.appendChild(tooltip);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

/** Mouse: the zone under the pointer. Touch: the zone you tap (a drag rotates instead). */
function onPointer(e: PointerEvent) {
  const rect = renderer.domElement.getBoundingClientRect(); // the 3D view may not start at the window's corner
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const cmp = comparing();
  const diffMeshes: THREE.Mesh[] = [];
  diffGroups.get(params.seg)?.traverse((o) => {
    if (o instanceof THREE.Mesh && o.userData.kind && o.visible) diffMeshes.push(o);
  });
  const targets = cmp ? diffMeshes : tumourMeshes(true).filter((m) => m.visible);
  const hit = raycaster.intersectObjects(targets, false)[0];
  const seg = meta?.segmentations[params.seg];
  if (!hit || !seg) {
    tooltip.style.display = "none";
    return;
  }
  if (cmp) {
    const r = params.compareRegion;
    const kind = hit.object.userData.kind as string;
    const total = seg.diff!.volume_ml[`${r}_${kind}`];
    tooltip.innerHTML = kind === "missed"
      ? `<b>Missed by ${seg.label}</b><br>the expert marked this as ${REGION_NAMES[r]}<br>${total} mL missed in total`
      : `<b>Extra from ${seg.label}</b><br>not ${REGION_NAMES[r]} in the expert outline<br>${total} mL extra in total`;
  } else {
    const zone = seg.zones[hit.object.name];
    const share = seg.regions_ml.WT ? Math.round((100 * zone.volume_ml) / seg.regions_ml.WT) : 0;
    tooltip.innerHTML = `<b>${zone.name}</b><br>${zone.volume_ml} mL · ${share}% of the whole tumour<br><small>${seg.label}</small>`;
  }
  tooltip.style.display = "block";
  // keep the label on screen near the right and bottom edges (phones)
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  tooltip.style.left = `${Math.max(4, Math.min(e.clientX + 14, innerWidth - tw - 4))}px`;
  tooltip.style.top = `${Math.max(4, Math.min(e.clientY + 14, innerHeight - th - 4))}px`;
}
renderer.domElement.addEventListener("pointermove", (e) => {
  if (e.pointerType === "mouse") onPointer(e);
  else tooltip.style.display = "none"; // dragging on a touch screen: rotating, not pointing
});
let tapStart: { x: number; y: number; t: number } | null = null;
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (e.pointerType !== "mouse") tapStart = { x: e.clientX, y: e.clientY, t: performance.now() };
});
renderer.domElement.addEventListener("pointerup", (e) => {
  const s = tapStart;
  tapStart = null;
  if (!s || e.pointerType === "mouse") return;
  const still = Math.hypot(e.clientX - s.x, e.clientY - s.y) < 10 && performance.now() - s.t < 500;
  if (still) onPointer(e);
});

function screenshot() {
  renderer.render(scene, camera); // draw a fresh frame, then save it before the browser clears it
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${caseId}_${params.seg}_${params.brain}_${params.style}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

// ---------------------------------------------------------------- control panel: only relevant controls are shown
const gui = new GUI({ title: "Display" });

const segFolder = gui.addFolder("Segmentation"); // filled when the case loads

const tumourFolder = gui.addFolder("Tumour");
tumourFolder.add(params, "style", ["toon", "hatching", "realistic"]).name("Style").onChange(applySettings);
const bandsCtrl = tumourFolder.add(params, "bands", 2, 6, 1).name("Toon bands").onChange(() => {
  for (const z of zoneStyles.values()) z.toon = toonMaterial(z.color, params.bands);
  for (const k of ["missed", "extra"] as const) diffMats[k].toon = toonMaterial(DIFF_COLORS[k], params.bands);
  applySettings();
});
const hatchCtrl = tumourFolder.add(params, "hatchSpacing", 0.8, 4, 0.1).name("Stroke spacing (mm)").onChange(applySettings);
tumourFolder.add(params, "outline").name("Outlines").onChange(applySettings);
const outlineWidthCtrl = tumourFolder.add(params, "outlineThickness", 0.1, 2, 0.05).name("Outline width (mm)").onChange(applySettings);

const zoneFolder = gui.addFolder("Tumour zones"); // one sub-folder per zone, added when the case loads

const brainFolder = gui.addFolder("Brain");
brainFolder.add(params, "brain", ["ghost", "cutaway", "ink", "plain", "hidden"]).name("Mode").onChange(() => {
  opacityState.value = brainOpacity[params.brain]; // show the new mode's own opacity
  opacityCtrl.updateDisplay();
  applySettings();
});
const opacityCtrl = brainFolder.add(opacityState, "value", 0, 1, 0.01).name("Opacity").onChange(() => {
  brainOpacity[params.brain] = opacityState.value;
  applySettings();
});
const edgeCtrl = brainFolder.add(params, "ghostEdge", 0, 1, 0.01).name("Edge glow").onChange(applySettings);
const glowCtrl = brainFolder.add(params, "glow", 0, 1.5, 0.01).name("Glow near tumour").onChange(applySettings);
const ghostMore = brainFolder.addFolder("More ghost settings");
ghostMore.add(params, "ghostBase", 0, 0.3, 0.01).name("Fill").onChange(applySettings);
ghostMore.add(params, "glowRadius", 10, 100, 1).name("Glow radius (mm)").onChange(applySettings);
ghostMore.close();
const cutCtrl = brainFolder.add(params, "cut", Object.keys(CUTS)).name("Remove").onChange(applySettings);
const inkCtrl = brainFolder.add(params, "inkWidth", 0.05, 0.5, 0.01).name("Line width").onChange(applySettings);

const sliceFolder = gui.addFolder("MRI slice");
const sliceShowCtrl = sliceFolder.add(params, "slice").name("Show").onChange(applySettings);
const sliceAxisCtrl = sliceFolder.add(params, "sliceAxis", ["axial", "coronal", "sagittal"]).name("Direction").onChange(applySettings);
const slicePosCtrl = sliceFolder.add(params, "slicePos", 0, 1, 0.005).name("Position").onChange(applySettings);
const sliceOpacityCtrl = sliceFolder.add(params, "sliceOpacity", 0.1, 1, 0.01).name("Opacity").onChange(applySettings);
const sliceBrightCtrl = sliceFolder.add(params, "sliceBrightness", -0.5, 0.5, 0.01).name("Brightness").onChange(applySettings);
const sliceContrastCtrl = sliceFolder.add(params, "sliceContrast", 0.5, 3, 0.05).name("Contrast").onChange(applySettings);
const sliceCutCtrl = sliceFolder.add(params, "sliceCut").name("Cut brain + tumour here").onChange(applySettings);
const sliceFlipCtrl = sliceFolder.add(params, "sliceFlip").name("Keep the other side").onChange(applySettings);

const viewFolder = gui.addFolder("View");
const views = {
  front: () => setView(new THREE.Vector3(0, 0, -1)),
  back: () => setView(new THREE.Vector3(0, 0, 1)),
  left: () => setView(new THREE.Vector3(-1, 0, 0)),
  right: () => setView(new THREE.Vector3(1, 0, 0)),
  top: () => setView(new THREE.Vector3(0, 1, 0.001)),
  screenshot,
};
viewFolder.add(views, "front").name("Front");
viewFolder.add(views, "back").name("Back");
viewFolder.add(views, "left").name("Left side");
viewFolder.add(views, "right").name("Right side");
viewFolder.add(views, "top").name("Top");
viewFolder.add(params, "autoRotate").name("Auto-rotate").onChange(applySettings);
viewFolder.add(views, "screenshot").name("Save screenshot (PNG)");

/** Show only the controls that apply to the current choices. */
function updatePanel() {
  const mode = params.brain;
  opacityCtrl.show(mode !== "hidden");
  edgeCtrl.show(mode === "ghost");
  glowCtrl.show(mode === "ghost");
  ghostMore.show(mode === "ghost");
  cutCtrl.show(mode === "cutaway");
  inkCtrl.show(mode === "ink");
  bandsCtrl.show(params.style === "toon");
  hatchCtrl.show(params.style === "hatching");
  outlineWidthCtrl.show(params.outline);
  for (const c of [sliceAxisCtrl, slicePosCtrl, sliceOpacityCtrl, sliceBrightCtrl, sliceContrastCtrl, sliceCutCtrl]) {
    c.show(params.slice);
  }
  scanCtrl?.show(params.slice);
  sliceFlipCtrl.show(params.slice && params.sliceCut);
  const canCompare = !!meta?.segmentations.expert && !!meta?.segmentations[params.seg]?.diff;
  compareCtrl?.show(canCompare);
  regionCtrl?.show(canCompare && params.compare);
  greyCtrl?.show(canCompare && params.compare);
  missedCtrl?.show(canCompare && params.compare);
  extraCtrl?.show(canCompare && params.compare);
}
updatePanel();

/** Pick the case (link, or the gallery's first), show the gallery, load the case. */
async function start() {
  let index: GalleryIndex | undefined;
  try {
    const res = await fetch("cases/index.json");
    if (res.ok) index = (await res.json()) as GalleryIndex;
  } catch {
    // no gallery (e.g. a single uploaded case): just show the case
  }
  if (LOCAL) {
    localSaved = await loadResult();
    if (!localSaved) throw new Error("not stored");
    localFiles = fileUrls(localSaved.files);
    caseId = "upload";
  } else if (JOB) {
    caseId = "upload";
    base = `${API_BASE}/api/cases/${JOB}`; // an uploaded case, served by the API
  } else {
    if (!caseId) caseId = index?.cases[0]?.case_id ?? "BRATS_155";
    base = `cases/${caseId}`;
  }
  if (index) renderGallery(index);
  fetch("data/try.json")
    .then((r) => (r.ok ? r.json() : undefined))
    .then((info?: TryInfo) => info && renderTry(info))
    .catch(() => {}); // optional panel: the viewer works without it
  await load();
}

start().catch((err) => {
  document.getElementById("info")!.textContent = LOCAL
    ? "This result was computed in your browser, and it isn't stored here anymore (only the latest upload is kept). " +
      "Upload the scans again."
    : JOB
    ? `Could not load this upload: results are deleted after a few hours. (${err})`
    : `Could not load ${caseId}: ${err}`;
  console.error(err);
});

/** Keep the 3D view matched to its area (window resized, phone turned, panel hidden or shown). */
function fitView() {
  const [w, h] = viewSize();
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  // the view became narrower or wider (phone panel hidden, phone turned): move the camera so the brain still fits,
  // keeping whatever zoom the user chose
  if (fitFactor) {
    const f = Math.max(2.4, 1.35 / camera.aspect), k = f / fitFactor;
    if (Math.abs(k - 1) > 1e-3) {
      camera.position.sub(controls.target).multiplyScalar(k).add(controls.target);
      viewDistance *= k;
      fitFactor = f;
    }
  }
}
addEventListener("resize", fitView);
new ResizeObserver(fitView).observe(appEl);

// phones and tablets (small or touch screens): the settings start folded (tap "Display" to open)
if (NARROW.matches || TOUCH) gui.close();
// phones: the info panel below the 3D view can be folded away for a full-size view
const sheetToggle = document.getElementById("sheet-toggle");
sheetToggle?.addEventListener("click", () => {
  const folded = document.body.classList.toggle("sheet-min");
  sheetToggle.textContent = folded ? "Show panel" : "Hide panel";
  sheetToggle.setAttribute("aria-expanded", String(!folded));
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
