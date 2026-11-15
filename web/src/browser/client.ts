/**
 * The page's side of in-browser segmentation: which models can run here, running them in a Web Worker,
 * and keeping the result (and the uploaded files, so more models can be run) in this browser's IndexedDB.
 * Nothing is sent anywhere: the scans stay on the visitor's computer.
 */
import type { Inputs } from "./case";
import type { Files, Meta } from "./bundle";
import type { ModelSpec, RunRequest } from "./worker";

/** Models small enough to run in a browser (Swin UNETR needs ~12 GB while running: more than browsers allow). */
export const BROWSER_MODELS = ["segresnet", "segformer3d"];
export type BrowserModel = ModelSpec & { cpuSeconds: number | null };
export type LocalModelInfo = { name: string; label: string; champion: boolean };

/** Models listed in models/manifest.json (shipped with the website): the ones that run here, and the rest. */
export async function browserModels(): Promise<{ runnable: BrowserModel[]; localOnly: LocalModelInfo[] } | null> {
  try {
    const r = await fetch("models/manifest.json");
    if (!r.ok) return null;
    const m = await r.json();
    const runnable: BrowserModel[] = [], localOnly: LocalModelInfo[] = [];
    const order = Object.keys(m.models).sort((a, b) => (a === m.champion ? -1 : b === m.champion ? 1 : 0));
    for (const name of order) {
      const info = m.models[name];
      const base = { name, label: info.display_name, champion: name === m.champion };
      if (BROWSER_MODELS.includes(name)) {
        runnable.push({ ...base, url: new URL(`models/${info.onnx}`, location.href).href, version: info.version,
          cpuSeconds: info.cpu_s_per_case_2_threads ?? null });
      } else localOnly.push(base);
    }
    return runnable.length ? { runnable, localOnly } : null;
  } catch {
    return null;
  }
}

/** Run models in a Web Worker. `existing`: add them to an earlier result instead of starting a new one. */
export function runInBrowser(inputs: Inputs, models: ModelSpec[], onProgress: (step: string, p: number) => void,
  existing?: { meta: Meta; files: Files }): Promise<{ meta: Meta; files: Files }> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") onProgress(msg.step, msg.progress);
      else {
        worker.terminate();
        if (msg.type === "done") resolve({ meta: msg.meta, files: msg.files });
        else reject(new Error(msg.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`Your browser couldn't run the models (${e.message || "the worker stopped"}). ` +
        "Try a recent Chrome, Edge or Firefox, or run the app locally."));
    };
    const req: RunRequest = {
      type: "run", inputs, models, existing,
      fingerprintsUrl: new URL("data/fingerprints.json", location.href).href,
    };
    worker.postMessage(req);
  });
}

// --------------------------------------------------------------------------- this browser's own storage
const DB = "brain-npr", STORE = "upload";
function open(version?: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = version ? indexedDB.open(DB, version) : indexedDB.open(DB);
    r.onupgradeneeded = () => {
      if (!r.result.objectStoreNames.contains(STORE)) r.result.createObjectStore(STORE);
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new Error("storage is busy in another tab"));
  });
}
/** The database, with its store (created, or re-created one version up if something left it without one). */
async function db(): Promise<IDBDatabase> {
  const d = await open();
  if (d.objectStoreNames.contains(STORE)) return d;
  const next = d.version + 1;
  d.close();
  return open(next);
}
/** A promise that gives up after `ms` (storage should answer in well under a second). */
function within<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  return Promise.race([p, new Promise<T>((resolve) => setTimeout(() => resolve(fallback), ms))]);
}
async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const d = await db();
  return new Promise((resolve, reject) => {
    // a save counts only once it is committed to disk: the page navigates right after saving
    const t = d.transaction(STORE, mode, mode === "readwrite" ? { durability: "strict" } : undefined);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => {
      d.close();
      resolve(req.result);
    };
    t.onerror = t.onabort = () => {
      d.close();
      reject(t.error ?? req.error ?? new Error("could not save the result in this browser"));
    };
  });
}
export type Saved = { inputs: Inputs; meta: Meta; files: Files; savedAt: number };

/** Keep the latest in-browser result (one at a time) so the page can be reopened with it. */
export async function saveResult(s: Saved): Promise<void> {
  await tx("readwrite", (st) => st.put(s, "latest"));
}
export async function loadResult(): Promise<Saved | undefined> {
  try {
    // never leave the page on "Loading…": no answer within 15 s counts as "not stored"
    return await within(tx<Saved | undefined>("readonly", (st) => st.get("latest")), 15000, undefined);
  } catch {
    return undefined;
  }
}

/** Blob URLs for a result's files, so the viewer can load them like files from a server. */
export function fileUrls(files: Files): Map<string, string> {
  const urls = new Map<string, string>();
  for (const [name, buf] of Object.entries(files)) {
    const type = name.endsWith(".glb") ? "model/gltf-binary" : "application/octet-stream";
    urls.set(name, URL.createObjectURL(new Blob([buf], { type })));
  }
  return urls;
}
