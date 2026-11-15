/** The Models page: how the three models were trained, how they score on the test set, and how fast they are. */
import { boxChart, legend, lineChart, scatterChart, type Series } from "./charts";

type Regions = { WT: number; TC: number; ET: number };
type Model = {
  label: string;
  version: number;
  champion: boolean;
  checkpoint_step: number;
  val_dice_mean: number;
  test: { dice: Regions & { mean: number }; hd95_mean: Regions; hd95_median: Regions; sensitivity: Regions; precision: Regions };
  efficiency: {
    params_m: number; gflops_per_window: number | null; gpu_s_per_case: number; gpu_peak_gb: number;
    cpu_s_per_case_2: number | null; cpu_s_per_case_8: number | null; onnx_mb: number; onnx_vs_pytorch_dice: number | null;
    train_s_per_step: number; train_hours: number; train_note: string;
  };
  curves: { train: { step: number; loss: number }[]; val: ({ step: number; loss: number; mean: number } & Regions)[] };
  per_case_dice: Record<"WT" | "TC" | "ET", number[]>;
  dice_by_split: Partial<Record<"train" | "val" | "test", Regions & { mean: number; n: number }>>;
};
type Data = {
  generated: string; registered_model: string; champion: string; champion_reason: string;
  split: Record<string, number>; training: Record<string, string | number>;
  case_ids: string[]; gallery_cases: string[];
  models: Record<string, Model>;
  stats: { a: string; b: string; p: number; a_better: number; b_better: number }[];
};

// one fixed colour per model on every chart (categorical slots 1-3, checked for colour-blind safety)
const COLORS: Record<string, string> = { segresnet: "#2a78d6", swin_unetr: "#eb6834", segformer3d: "#1baf7a" };
const REGION_LABEL: Record<string, string> = { WT: "Whole tumour", TC: "Tumour core", ET: "Active tumour" };
const REGIONS = ["WT", "TC", "ET"] as const;

const $ = (id: string) => document.getElementById(id)!;
const f3 = (v: number) => v.toFixed(3);
const pText = (p: number) => (p < 0.0001 ? "p < 0.0001" : `p = ${p.toFixed(2)}`);
const key = (k: string) => `<i class="key" style="background:${COLORS[k]}"></i>`;

/** Bold the best value in a column (highest, or lowest when `low` is true). */
function best(values: number[], low = false) {
  const b = low ? Math.min(...values) : Math.max(...values);
  return (v: number, text: string) => (v === b ? `<b>${text}</b>` : text);
}

function render(d: Data) {
  const names = Object.keys(d.models);
  const m = (k: string) => d.models[k];

  // --- one card per model
  $("cards").innerHTML = names.map((k) => {
    const x = m(k), e = x.efficiency;
    const badge = x.champion ? `<span class="badge" title="MLflow alias @champion: the app's default model">champion</span>` : "";
    return `<div class="card model">
      <div class="name">${key(k)}${x.label}${badge}</div>
      <div class="hero">${f3(x.test.dice.mean)}</div>
      <div class="muted">mean Dice on the test set</div>
      <div class="facts">${e.params_m.toFixed(1)} M parameters · ${e.cpu_s_per_case_2} s per brain on 2 CPU cores · trained in ${e.train_hours} h</div>
    </div>`;
  }).join("");
  const champ = m(d.champion);
  $("champion").innerHTML = `${key(d.champion)}<b>${champ.label}</b> is the app's default model ` +
    `(registry <code>${d.registered_model}</code>, version ${champ.version}, alias <code>@champion</code>). ${d.champion_reason}.`;

  // --- test results table
  const cols = (fn: (x: Model) => number) => names.map((k) => fn(m(k)));
  const diceBest = Object.fromEntries([...REGIONS, "mean"].map((r) => [r, best(cols((x) => x.test.dice[r as keyof Regions | "mean"] as number))]));
  const hdBest = Object.fromEntries(REGIONS.map((r) => [r, best(cols((x) => x.test.hd95_median[r]), true)]));
  $("test-table").innerHTML = `<table>
    <thead><tr><th>Model</th><th>Dice WT</th><th>Dice TC</th><th>Dice ET</th><th>Dice mean</th>
      <th>HD95 WT</th><th>HD95 TC</th><th>HD95 ET</th><th>Sensitivity ET</th><th>Precision ET</th></tr></thead>
    <tbody>${names.map((k) => {
      const t = m(k).test;
      return `<tr><td>${key(k)}${m(k).label}</td>
        ${REGIONS.map((r) => `<td>${diceBest[r](t.dice[r], f3(t.dice[r]))}</td>`).join("")}
        <td>${diceBest.mean(t.dice.mean, f3(t.dice.mean))}</td>
        ${REGIONS.map((r) => `<td title="mean ${t.hd95_mean[r].toFixed(1)} mm">${hdBest[r](t.hd95_median[r], t.hd95_median[r].toFixed(1))}</td>`).join("")}
        <td>${f3(t.sensitivity.ET)}</td><td>${f3(t.precision.ET)}</td></tr>`;
    }).join("")}</tbody></table>`;

  // --- Dice per brain
  boxChart($("box"), {
    label: "Test-set Dice per brain, by region and model",
    groups: REGIONS.map((r) => ({ key: r, label: REGION_LABEL[r] })),
    series: names.map((k) => ({ key: k, label: m(k).label, color: COLORS[k], values: m(k).per_case_dice })),
    ids: d.case_ids,
    y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => v.toFixed(2), title: "Dice per brain" },
    // brains in the viewer's gallery open there, already comparing this model with the expert
    outlierLink: (model, region, id) => (d.gallery_cases.includes(id) ? `./?case=${id}&seg=${model}&cmp=${region}` : null),
  });

  // --- accuracy vs speed
  scatterChart($("speed"), {
    label: "Mean test Dice against prediction time per brain on 2 CPU cores",
    points: names.map((k) => {
      const e = m(k).efficiency;
      return {
        key: k, label: m(k).label, color: COLORS[k], x: e.cpu_s_per_case_2 ?? 1, y: m(k).test.dice.mean,
        note: `${e.cpu_s_per_case_2} s`,
        tip: `<div class="tip-head">${key(k)}${m(k).label}</div><div>mean Dice <b>${f3(m(k).test.dice.mean)}</b></div>` +
          `<div>${e.cpu_s_per_case_2} s per brain (2 CPU cores), ${e.cpu_s_per_case_8} s (8 cores)</div>` +
          `<div>${e.params_m.toFixed(1)} M parameters · ONNX file ${e.onnx_mb} MB</div>`,
      };
    }),
    x: { min: 1, max: 50, ticks: [1, 2, 5, 10, 20, 50], format: (v) => `${v} s`, title: "seconds per brain on 2 CPU cores (log scale) — further left is faster" },
    y: { min: 0.8, max: 0.85, ticks: [0.8, 0.81, 0.82, 0.83, 0.84, 0.85], format: (v) => v.toFixed(2), title: "mean test Dice" },
  });

  // --- training curves
  const steps = { min: 0, max: 30000, ticks: [0, 5000, 10000, 15000, 20000, 25000, 30000], format: (v: number) => (v ? `${v / 1000}k` : "0"), title: "training step" };
  const valSeries: Series[] = names.map((k) => ({
    key: k, label: m(k).label, color: COLORS[k],
    points: m(k).curves.val.map((v) => [v.step, v.mean] as [number, number]),
    mark: [m(k).checkpoint_step, m(k).val_dice_mean],
  }));
  lineChart($("val-curve"), {
    label: "Mean validation Dice during training", series: valSeries, value: f3,
    markNote: "ring: best checkpoint (the one that was kept)",
    x: steps, y: { min: 0.3, max: 0.85, ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], format: (v) => v.toFixed(1), title: "mean validation Dice" },
  });
  // loss: one small chart per model, training (solid) against validation (dashed). If the validation
  // line rose while the training line kept falling, the model would be memorising (overfitting).
  const lossBox = $("loss-curve");
  lossBox.innerHTML = `<div class="legend"><span><i class="solid"></i>training loss (average over each 1,000 steps)</span>` +
    `<span><i class="dashed"></i>validation loss (measured every 1,000 steps)</span></div>`;
  for (const k of names) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<div class="panel-title">${key(k)}${m(k).label}</div>`;
    lossBox.appendChild(panel);
    lineChart(panel, {
      label: `${m(k).label}: training and validation loss`, value: f3, height: 200, showLegend: false,
      series: [
        { key: `${k}-train`, label: "training", color: COLORS[k], points: m(k).curves.train.map((v) => [v.step, v.loss] as [number, number]) },
        { key: `${k}-val`, label: "validation", color: COLORS[k], dash: true, points: m(k).curves.val.map((v) => [v.step, v.loss] as [number, number]) },
      ],
      x: steps, y: { min: 0, max: 1.0, ticks: [0, 0.25, 0.5, 0.75, 1.0], format: (v) => v.toFixed(2), title: "loss (Dice + cross-entropy; lower is better)" },
    });
  }
  // the same numbers as a table, for anyone who can't or doesn't want to read the curves
  const rows = m(names[0]).curves.val.filter((v) => v.step % 3000 === 0).map((v) => v.step);
  $("val-table").innerHTML = `<table><thead><tr><th>Step</th>${names.map((k) => `<th>${key(k)}${m(k).label}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((s) => `<tr><td>${s.toLocaleString()}</td>${names.map((k) => {
      const v = m(k).curves.val.find((x) => x.step === s);
      return `<td>${v ? f3(v.mean) : "–"}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;

  // --- the same checkpoint on training, validation and test brains: a large gap would mean memorising
  const haveSplits = names.every((k) => m(k).dice_by_split.train && m(k).dice_by_split.val && m(k).dice_by_split.test);
  $("split-section").hidden = !haveSplits;
  if (haveSplits) {
    const cell = (x: Regions & { mean: number }) =>
      `<td title="WT ${f3(x.WT)} · TC ${f3(x.TC)} · ET ${f3(x.ET)}">${f3(x.mean)}</td>`;
    const n = m(names[0]).dice_by_split;
    $("split-table").innerHTML = `<table><thead><tr><th>Model</th>` +
      `<th>Training brains (${n.train!.n})</th><th>Validation (${n.val!.n})</th><th>Test (${n.test!.n})</th>` +
      `<th>Gap: training − test</th></tr></thead><tbody>${names.map((k) => {
        const s = m(k).dice_by_split;
        return `<tr><td>${key(k)}${m(k).label}</td>${cell(s.train!)}${cell(s.val!)}${cell(s.test!)}` +
          `<td>${(s.train!.mean - s.test!.mean >= 0 ? "+" : "")}${f3(s.train!.mean - s.test!.mean)}</td></tr>`;
      }).join("")}</tbody></table>`;

    // plain-language findings, worked out from the numbers (so they stay right if the models change)
    const gap = (k: string) => m(k).dice_by_split.train!.mean - m(k).dice_by_split.test!.mean;
    const gaps = names.map(gap);
    const memo = names.reduce((a, b) => (gap(b) > gap(a) ? b : a));
    const fitTop = names.reduce((a, b) => (m(b).dice_by_split.train!.mean > m(a).dice_by_split.train!.mean ? b : a));
    const champ = d.champion;
    const findings: string[] = [];
    const maxGap = Math.max(...gaps);
    findings.push(maxGap < 0.08
      ? `All gaps are small (${f3(Math.min(...gaps))} to ${f3(maxGap)}): every model does almost as well on new brains as on the ones it studied. None of them memorised its training brains.`
      : `${m(memo).label} has a large gap (${f3(gap(memo))}): it partly memorised its training brains.`);
    for (const k of names) {
      const s = m(k).dice_by_split;
      findings.push(`${m(k).label}: ${f3(s.train!.mean)} on the training brains, ${f3(s.test!.mean)} on the test brains it never saw, a gap of ${f3(gap(k))}.`);
    }
    if (fitTop !== champ && Math.abs(m(fitTop).dice_by_split.test!.mean - m(champ).dice_by_split.test!.mean) < 0.005) {
      findings.push(`${m(fitTop).label} fits its training brains best (${f3(m(fitTop).dice_by_split.train!.mean)}), ` +
        `but on the test brains it only ties ${m(champ).label} (${f3(m(fitTop).dice_by_split.test!.mean)} against ${f3(m(champ).dice_by_split.test!.mean)}). ` +
        `Its extra fit to the training brains did not carry over to new ones, while ${m(champ).label} reaches the same result ` +
        `with a smaller gap and far fewer parameters: one more reason it is the champion.`);
    }
    if (names.every((k) => m(k).dice_by_split.val!.mean < m(k).dice_by_split.test!.mean)) {
      findings.push(`Every model scores a little lower on the validation brains than on the test brains. Both groups were picked at random; ` +
        `the validation brains just happen to be slightly harder. With 72 brains per group, differences like this are normal.`);
    }
    $("split-findings").innerHTML = findings.map((t) => `<li>${t}</li>`).join("");
  }

  // --- efficiency table
  const eff = (fn: (e: Model["efficiency"]) => number | null) => names.map((k) => fn(m(k).efficiency) ?? Infinity);
  const effBest = {
    params: best(eff((e) => e.params_m), true), gflops: best(eff((e) => e.gflops_per_window), true),
    onnx: best(eff((e) => e.onnx_mb), true), gpu: best(eff((e) => e.gpu_s_per_case), true),
    mem: best(eff((e) => e.gpu_peak_gb), true), cpu2: best(eff((e) => e.cpu_s_per_case_2), true),
    cpu8: best(eff((e) => e.cpu_s_per_case_8), true), train: best(eff((e) => e.train_hours), true),
  };
  $("eff-table").innerHTML = `<table>
    <thead><tr><th>Model</th><th>Params</th><th title="compute for one 160³ window">Compute</th><th>ONNX file</th>
      <th>GPU / brain</th><th>GPU mem.</th><th title="2 CPU cores, like a free cloud machine">CPU / brain</th><th>CPU, 8 cores</th><th>Training</th></tr></thead>
    <tbody>${names.map((k) => {
      const e = m(k).efficiency;
      return `<tr><td>${key(k)}${m(k).label}</td>
        <td>${effBest.params(e.params_m, `${e.params_m.toFixed(1)} M`)}</td>
        <td>${e.gflops_per_window == null ? "–" : effBest.gflops(e.gflops_per_window, `${Math.round(e.gflops_per_window).toLocaleString()} GFLOPs`)}</td>
        <td>${effBest.onnx(e.onnx_mb, `${e.onnx_mb} MB`)}</td>
        <td>${effBest.gpu(e.gpu_s_per_case, `${e.gpu_s_per_case} s`)}</td>
        <td>${effBest.mem(e.gpu_peak_gb, `${e.gpu_peak_gb} GB`)}</td>
        <td>${e.cpu_s_per_case_2 == null ? "–" : effBest.cpu2(e.cpu_s_per_case_2, `${e.cpu_s_per_case_2} s`)}</td>
        <td>${e.cpu_s_per_case_8 == null ? "–" : effBest.cpu8(e.cpu_s_per_case_8, `${e.cpu_s_per_case_8} s`)}</td>
        <td>${effBest.train(e.train_hours, `${e.train_hours} h`)}${e.train_note ? "*" : ""}</td></tr>`;
    }).join("")}</tbody></table>`;
  const noted = names.filter((k) => m(k).efficiency.train_note);
  $("eff-note").textContent = noted.length
    ? `* ${noted.map((k) => m(k).label).join(", ")}: ${m(noted[0]).efficiency.train_note}.` : "";

  // --- how it was measured
  const s = d.split;
  $("method").innerHTML = `<ul>
    <li>All three models were trained the same way: ${s.train} training brains, ${d.training.steps.toLocaleString()} steps of
      ${d.training.crop} crops, batch ${d.training.batch}, ${d.training.optimiser}. The checkpoint with the best validation
      Dice (${s.val} brains) was kept.</li>
    <li>Scores are on the <b>${s.test} test brains</b>, which no model saw during training or checkpoint selection.</li>
    <li>Are the differences real? A Wilcoxon signed-rank test on each brain's mean Dice (the same brains for every model):
      ${d.stats.map((x) => `${m(x.a).label} vs ${m(x.b).label}: ${pText(x.p)} (${x.a_better} vs ${x.b_better} brains better)`).join("; ")}.
      So SegResNet and Swin UNETR are tied, and both beat SegFormer3D.</li>
    <li>HD95 (distance between predicted and true tumour edges, 95th percentile) is shown as the <b>median</b> over brains.
      The mean is dominated by a few brains where a region is missed or invented, which count as 373 mm; hover a cell for the mean.</li>
    <li>CPU times use the exported ONNX models with onnxruntime (what the upload service runs). Their predictions agree with the
      original PyTorch models to Dice ${names.map((k) => m(k).efficiency.onnx_vs_pytorch_dice).filter(Boolean).map((v) => (v as number).toFixed(3)).join(" / ")}.</li>
    <li>Data: Medical Segmentation Decathlon, Task01_BrainTumour (from BraTS 2016/2017), CC-BY-SA 4.0. Page generated ${d.generated}.</li>
  </ul>`;
  void legend;
}

fetch("data/models.json")
  .then((r) => r.json())
  .then(render)
  .catch((err) => {
    $("cards").textContent = `Could not load the model results: ${err}`;
    console.error(err);
  });
