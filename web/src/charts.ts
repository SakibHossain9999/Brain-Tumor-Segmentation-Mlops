/**
 * Small hand-made SVG charts for the Models page (no chart library).
 * Conventions: 2px lines, >=8px dots with a 2px surface-coloured ring, hairline grids,
 * text in ink colours (never the series colour), a hover tooltip on every chart,
 * and a legend whenever there is more than one series.
 */

const NS = "http://www.w3.org/2000/svg";
export const INK = {
  primary: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  surface: "#fcfcfb",
};

export type Series = {
  key: string;
  label: string;
  color: string;
  points: [number, number][];
  mark?: [number, number]; // one highlighted point (e.g. the best checkpoint)
  dash?: boolean;           // dashed line (e.g. validation next to solid training)
};
type Axis = { min: number; max: number; ticks: number[]; format: (v: number) => string; title?: string };

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
}

function svgText(parent: Element, x: number, y: number, s: string, attrs: Record<string, string | number> = {}) {
  const t = svgEl("text", { x, y, "font-size": 12, fill: INK.muted, ...attrs }, parent);
  t.textContent = s;
  return t;
}

/** Colour key + name for each series (the dependable identity channel; colour alone never is). */
export function legend(items: { label: string; color: string }[], note = ""): HTMLElement {
  const div = document.createElement("div");
  div.className = "legend";
  div.innerHTML =
    items.map((i) => `<span><i style="background:${i.color}"></i>${i.label}</span>`).join("") +
    (note ? `<span class="note">${note}</span>` : "");
  return div;
}

/** A chart container: the SVG, plus a tooltip that follows the mouse. */
function frame(container: HTMLElement, W: number, H: number, label: string) {
  const wrap = document.createElement("div");
  wrap.className = "chart";
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": label });
  wrap.appendChild(svg);
  const tip = document.createElement("div");
  tip.className = "tip";
  wrap.appendChild(tip);
  container.appendChild(wrap);

  /** Mouse position in chart units. */
  const toChart = (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    return [((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * H) / r.height];
  };
  const showTip = (e: MouseEvent, html: string) => {
    tip.innerHTML = html;
    tip.style.display = "block";
    const r = wrap.getBoundingClientRect();
    let x = e.clientX - r.left + 14;
    if (x + tip.offsetWidth > r.width) x = e.clientX - r.left - tip.offsetWidth - 14; // flip near the right edge
    tip.style.left = `${x}px`;
    tip.style.top = `${e.clientY - r.top + 12}px`;
  };
  const hideTip = () => (tip.style.display = "none");
  return { svg, toChart, showTip, hideTip };
}

function dot(parent: Element, x: number, y: number, color: string, r = 4) {
  return svgEl("circle", { cx: x, cy: y, r, fill: color, stroke: INK.surface, "stroke-width": 2 }, parent);
}

function gridAndAxes(svg: SVGSVGElement, x: Axis, y: Axis, sx: (v: number) => number, sy: (v: number) => number,
                     L: number, R: number, T: number, B: number, W: number, H: number) {
  for (const t of y.ticks) {
    svgEl("line", { x1: L, x2: W - R, y1: sy(t), y2: sy(t), stroke: t === y.min ? INK.axis : INK.grid, "stroke-width": 1 }, svg);
    svgText(svg, L - 8, sy(t) + 4, y.format(t), { "text-anchor": "end" });
  }
  for (const t of x.ticks) svgText(svg, sx(t), H - B + 18, x.format(t), { "text-anchor": "middle" });
  if (x.title) svgText(svg, (L + W - R) / 2, H - 4, x.title, { "text-anchor": "middle", fill: INK.secondary });
  if (y.title) svgText(svg, L - 40, T - 14, y.title, { fill: INK.secondary });
}

/** Lines over a shared x axis, with direct end labels and a crosshair tooltip. */
export function lineChart(container: HTMLElement, o: {
  series: Series[]; x: Axis; y: Axis; value: (v: number) => string; label: string; markNote?: string;
  height?: number; showLegend?: boolean;
}) {
  const W = 680, H = o.height ?? 280, L = 48, R = 112, T = 34, B = 40;
  if (o.showLegend !== false) container.appendChild(legend(o.series, o.markNote));
  const { svg, toChart, showTip, hideTip } = frame(container, W, H, o.label);
  const sx = (v: number) => L + ((v - o.x.min) / (o.x.max - o.x.min)) * (W - L - R);
  const sy = (v: number) => H - B - ((v - o.y.min) / (o.y.max - o.y.min)) * (H - T - B);
  gridAndAxes(svg, o.x, o.y, sx, sy, L, R, T, B, W, H);

  for (const s of o.series) {
    const d = s.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(Math.max(y, o.y.min)).toFixed(1)}`).join("");
    svgEl("path", {
      d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(s.dash ? { "stroke-dasharray": "6 5" } : {}),
    }, svg);
  }
  for (const s of o.series) {
    if (s.mark) {
      // best checkpoint: a hollow ring, so it reads differently from the end dot
      svgEl("circle", { cx: sx(s.mark[0]), cy: sy(s.mark[1]), r: 5, fill: INK.surface, stroke: s.color, "stroke-width": 2 }, svg);
    }
  }
  // direct labels at the line ends; when ends are close, labels are spread apart and joined by a leader line
  const ends = o.series.map((s) => {
    const [x, y] = s.points[s.points.length - 1];
    return { s, x: sx(x), y: sy(Math.max(y, o.y.min)), ly: sy(Math.max(y, o.y.min)) };
  }).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].ly = Math.max(ends[i].ly, ends[i - 1].ly + 15);
  for (const e of ends) {
    dot(svg, e.x, e.y, e.s.color);
    if (Math.abs(e.ly - e.y) > 2) {
      svgEl("line", { x1: e.x + 6, y1: e.y, x2: W - R + 8, y2: e.ly - 4, stroke: INK.axis, "stroke-width": 1 }, svg);
    }
    svgText(svg, W - R + 12, e.ly, e.s.label, { fill: INK.secondary });
  }

  // hover: a crosshair at the nearest step, with every series' value there
  const xs = [...new Set(o.series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const cross = svgEl("line", { y1: T, y2: H - B, stroke: INK.axis, "stroke-width": 1, visibility: "hidden" }, svg);
  const hoverDots = o.series.map((s) => {
    const c = dot(svg, 0, 0, s.color);
    c.setAttribute("visibility", "hidden");
    return c;
  });
  const hit = svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (e) => {
    const [mx] = toChart(e);
    const xv = o.x.min + ((mx - L) / (W - L - R)) * (o.x.max - o.x.min);
    const xn = xs.reduce((best, v) => (Math.abs(v - xv) < Math.abs(best - xv) ? v : best), xs[0]);
    cross.setAttribute("x1", String(sx(xn)));
    cross.setAttribute("x2", String(sx(xn)));
    cross.setAttribute("visibility", "visible");
    const rows = o.series.map((s, i) => {
      const p = s.points.reduce((b, q) => (Math.abs(q[0] - xn) < Math.abs(b[0] - xn) ? q : b), s.points[0]);
      hoverDots[i].setAttribute("cx", String(sx(p[0])));
      hoverDots[i].setAttribute("cy", String(sy(Math.max(p[1], o.y.min))));
      hoverDots[i].setAttribute("visibility", "visible");
      return `<div><i style="background:${s.color}"></i>${s.label} <b>${o.value(p[1])}</b></div>`;
    });
    showTip(e, `<div class="tip-head">step ${o.x.format(xn)}</div>${rows.join("")}`);
  });
  hit.addEventListener("mouseleave", () => {
    hideTip();
    cross.setAttribute("visibility", "hidden");
    hoverDots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
}

/** Five-number summary for a boxplot (whiskers reach the furthest point within 1.5 x the box height). */
function boxStats(values: number[]) {
  const v = [...values].sort((a, b) => a - b);
  const q = (p: number) => {
    const i = (v.length - 1) * p;
    return v[Math.floor(i)] + (v[Math.ceil(i)] - v[Math.floor(i)]) * (i - Math.floor(i));
  };
  const q1 = q(0.25), med = q(0.5), q3 = q(0.75), iqr = q3 - q1;
  const lo = v.find((x) => x >= q1 - 1.5 * iqr)!;
  const hi = [...v].reverse().find((x) => x <= q3 + 1.5 * iqr)!;
  return { q1, med, q3, lo, hi, n: v.length };
}

/** Boxplots: one group per region, one box per model. Outlier dots name their brain. */
export function boxChart(container: HTMLElement, o: {
  groups: { key: string; label: string }[];
  series: { key: string; label: string; color: string; values: Record<string, number[]> }[];
  ids: string[];
  y: Axis;
  label: string;
  outlierLink?: (series: string, group: string, id: string) => string | null;
}) {
  const W = 680, H = 300, L = 48, R = 16, T = 34, B = 40;
  container.appendChild(legend(o.series, "dots: individual brains outside the whiskers"));
  const { svg, showTip, hideTip } = frame(container, W, H, o.label);
  const band = (W - L - R) / o.groups.length;
  const sx = (gi: number, si: number) => L + band * (gi + 0.5) + (si - (o.series.length - 1) / 2) * 44;
  const sy = (v: number) => H - B - ((v - o.y.min) / (o.y.max - o.y.min)) * (H - T - B);
  gridAndAxes(svg, { min: 0, max: 1, ticks: [], format: String }, o.y, () => 0, sy, L, R, T, B, W, H);
  o.groups.forEach((g, gi) => svgText(svg, L + band * (gi + 0.5), H - B + 20, g.label, { "text-anchor": "middle", fill: INK.secondary }));

  o.groups.forEach((g, gi) => {
    o.series.forEach((s, si) => {
      const values = s.values[g.key];
      const st = boxStats(values);
      const x = sx(gi, si);
      svgEl("line", { x1: x, x2: x, y1: sy(st.lo), y2: sy(st.q1), stroke: s.color, "stroke-width": 2 }, svg);
      svgEl("line", { x1: x, x2: x, y1: sy(st.q3), y2: sy(st.hi), stroke: s.color, "stroke-width": 2 }, svg);
      for (const w of [st.lo, st.hi]) svgEl("line", { x1: x - 6, x2: x + 6, y1: sy(w), y2: sy(w), stroke: s.color, "stroke-width": 2 }, svg);
      svgEl("rect", { x: x - 11, y: sy(st.q3), width: 22, height: Math.max(sy(st.q1) - sy(st.q3), 2), rx: 3, fill: s.color }, svg);
      svgEl("line", { x1: x - 11, x2: x + 11, y1: sy(st.med), y2: sy(st.med), stroke: INK.surface, "stroke-width": 2 }, svg);
      // hover target: wider than the box, covering the whiskers
      const hit = svgEl("rect", { x: x - 20, y: sy(st.hi) - 6, width: 40, height: sy(st.lo) - sy(st.hi) + 12, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (e) => showTip(e,
        `<div class="tip-head"><i style="background:${s.color}"></i>${s.label} · ${g.label}</div>` +
        `<div>median <b>${st.med.toFixed(3)}</b></div><div>middle half ${st.q1.toFixed(3)} to ${st.q3.toFixed(3)}</div>` +
        `<div>whiskers ${st.lo.toFixed(3)} to ${st.hi.toFixed(3)}</div><div>${st.n} brains</div>`));
      hit.addEventListener("mouseleave", hideTip);
      // outliers: each one is a brain, named on hover; gallery brains open in the viewer
      values.forEach((v, i) => {
        if (v >= st.lo && v <= st.hi) return;
        const id = o.ids[i];
        const link = o.outlierLink?.(s.key, g.key, id) ?? null;
        const c = dot(svg, x, sy(v), s.color);
        const ring = svgEl("circle", { cx: x, cy: sy(v), r: 9, fill: "transparent" }, svg); // bigger hover target
        if (link) ring.style.cursor = "pointer";
        ring.addEventListener("mousemove", (e) => showTip(e,
          `<div class="tip-head"><i style="background:${s.color}"></i>${id}</div>` +
          `<div>${s.label} · ${g.label}: Dice <b>${v.toFixed(3)}</b></div>` +
          (link ? `<div class="tip-note">click to open it in the viewer</div>` : "")));
        ring.addEventListener("mouseleave", hideTip);
        if (link) ring.addEventListener("click", () => (location.href = link));
        void c;
      });
    });
  });
}

/** A few labelled points on a logarithmic x axis (e.g. accuracy against prediction time). */
export function scatterChart(container: HTMLElement, o: {
  points: { key: string; label: string; color: string; x: number; y: number; note: string; tip: string }[];
  x: Axis; y: Axis; label: string;
}) {
  const W = 680, H = 270, L = 56, R = 130, T = 34, B = 44;
  container.appendChild(legend(o.points));
  const { svg, showTip, hideTip } = frame(container, W, H, o.label);
  const lx = (v: number) => Math.log10(v);
  const sx = (v: number) => L + ((lx(v) - lx(o.x.min)) / (lx(o.x.max) - lx(o.x.min))) * (W - L - R);
  const sy = (v: number) => H - B - ((v - o.y.min) / (o.y.max - o.y.min)) * (H - T - B);
  gridAndAxes(svg, o.x, o.y, sx, sy, L, R, T, B, W, H);
  for (const p of o.points) {
    dot(svg, sx(p.x), sy(p.y), p.color, 6);
    svgText(svg, sx(p.x) + 12, sy(p.y) + 4, `${p.label} · ${p.note}`, { fill: INK.secondary });
    const hit = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 14, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, p.tip));
    hit.addEventListener("mouseleave", hideTip);
  }
}
