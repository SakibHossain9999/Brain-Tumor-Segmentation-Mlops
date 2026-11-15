/// <reference lib="webworker" />
/**
 * Runs in a Web Worker: checks the uploaded files, runs the chosen models with ONNX Runtime Web
 * (WebAssembly), scores them against the expert outline, and builds the 3D bundle. The page stays usable.
 *
 * in:  { type: "run", inputs, models: [{name, label, url, version, champion}], fingerprintsUrl, existing? }
 * out: { type: "progress", step, progress }  ...  then { type: "done", meta, files } or { type: "error", message }
 */
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { loadCase, UploadError, type Inputs } from "./case";
import { addSegmentation, buildBase, describeIdentity, identify, type Files, type Meta } from "./bundle";
import { predict, windowCount, type WindowModel } from "./infer";

ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
// several threads need a cross-origin-isolated page; otherwise ONNX Runtime uses one
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1)) : 1;

export type ModelSpec = { name: string; label: string; url: string; version?: number; champion?: boolean };
export type RunRequest = {
  type: "run";
  inputs: Inputs;
  models: ModelSpec[];
  fingerprintsUrl: string;
  existing?: { meta: Meta; files: Files }; // add these models to an earlier result
};

const post = (msg: object, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);
const progress = (step: string, p: number) => post({ type: "progress", step, progress: p });
const tick = () => new Promise((r) => setTimeout(r, 0)); // let progress messages go out between heavy steps

async function session(url: string): Promise<WindowModel & { release: () => Promise<void> }> {
  const s = await ort.InferenceSession.create(url, { executionProviders: ["wasm"], enableCpuMemArena: false });
  const run = (async (input: Float32Array) => {
    const out = await s.run({ scans: new ort.Tensor("float32", input, [1, 4, 160, 160, 160]) });
    return out.logits.data as Float32Array;
  }) as WindowModel & { release: () => Promise<void> };
  run.release = () => s.release();
  return run;
}

self.onmessage = async (e: MessageEvent<RunRequest>) => {
  const req = e.data;
  const started = performance.now();
  try {
    progress("Checking the files", 0.02);
    await tick();
    const c = await loadCase(req.inputs);

    let meta: Meta, files: Files;
    if (req.existing) {
      ({ meta, files } = req.existing);
    } else {
      progress("Checking whether this brain is from our dataset", 0.06);
      const fp = await fetch(req.fingerprintsUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null);
      const ident = await identify(c, fp);
      let description = describeIdentity(ident);
      if (!c.regions) description += " No expert outline was uploaded, so there are no scores.";
      progress("Building the brain surface and scan views", 0.1);
      await tick();
      ({ meta, files } = buildBase(c, {
        title: "Your upload",
        description,
        split: ident ? ident.split : "unknown",
        upload: { format: c.format, label_format: c.labelFormat ?? null, identified: ident, models: [], engine: "browser" },
      }));
      if (ident) meta.case_id = ident.case_id;
      if (c.regions) addSegmentation(c, meta, files, "expert", c.regions, { label: "Expert outline" });
    }

    const windows = windowCount(c);
    const n = req.models.length;
    for (let m = 0; m < n; m++) {
      const spec = req.models[m];
      const base = 0.2 + (0.7 * m) / n, span = 0.7 / n;
      progress(`Loading ${spec.label}`, base);
      const model = await session(spec.url);
      const t0 = performance.now();
      progress(`Running ${spec.label}: part 1 of ${windows}`, base + span * 0.02);
      const regions = await predict(c, model, (done, total) => {
        if (done < total) progress(`Running ${spec.label}: part ${done + 1} of ${total}`, base + span * 0.85 * (done / total));
      });
      await model.release();
      const seconds = Math.round((performance.now() - t0) / 100) / 10;
      progress(`${spec.label}: scoring and building its 3D view`, base + span * 0.9);
      await tick();
      addSegmentation(c, meta, files, spec.name, regions, {
        label: spec.label, model_version: spec.version, champion: !!spec.champion, seconds,
      });
      meta.upload.models = [...new Set([...(meta.upload.models ?? []), spec.name])];
    }
    if (!meta.default_segmentation || !meta.segmentations[meta.default_segmentation]) {
      const champ = req.models.find((m) => m.champion)?.name;
      meta.default_segmentation = champ ?? req.models[0]?.name ?? "expert";
    }
    meta.upload.seconds = Math.round((performance.now() - started) / 100) / 10;
    meta.upload.threads = ort.env.wasm.numThreads;
    progress("Done", 1);
    post({ type: "done", meta, files });
  } catch (err) {
    const message = err instanceof UploadError ? err.message
      : `Something went wrong while processing the scans in your browser: ${err instanceof Error ? err.message : err}`;
    post({ type: "error", message });
  }
};
