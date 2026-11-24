// Small SVG chart kit: time-series line chart with crosshair + playhead, bullet strip, diverging bars.
// All text goes in with textContent; colours come from CSS custom properties.
const NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function text(parent, x, y, str, attrs = {}) {
  const t = svg("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

export function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function fmtNum(v, decimals = 0) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const abs = Math.abs(v);
  if (abs >= 10000) return (v / 1000).toFixed(abs >= 100000 ? 0 : 1) + "K";
  return v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/**
 * Line chart over a shared x array.
 * opts: { x: number[], series: [{name, values, color}], events: [{label, x}], height, yUnit,
 *         xLabel, peakLabels: bool, onHover(index|null), title }
 * Returns { node, setPlayhead(x|null) }.
 */
export function lineChart(opts) {
  const wrap = document.createElement("div");
  wrap.className = "chart";
  const height = opts.height || 240;
  let playheadX = null;
  let width = 0;
  let layer = null;

  if (opts.series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of opts.series) {
      const item = document.createElement("span");
      item.className = "legend-item";
      const key = document.createElement("span");
      key.className = "line-key";
      key.style.background = s.color;
      item.append(key, document.createTextNode(s.name));
      legend.appendChild(item);
    }
    wrap.appendChild(legend);
  }
  const plot = document.createElement("div");
  plot.className = "plot";
  wrap.appendChild(plot);
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  plot.appendChild(tip);

  const xs = opts.x;
  const all = opts.series.flatMap(s => s.values.filter(v => v !== null && isFinite(v)));
  let ymin = Math.min(...all), ymax = Math.max(...all);
  if (!all.length) { ymin = 0; ymax = 1; }
  const pad = (ymax - ymin) * 0.08 || 1;
  if (ymin > 0 && ymin - pad < 0 && ymin < (ymax - ymin) * 0.3) ymin = 0; else ymin -= pad;
  ymax += pad;
  const yt = niceTicks(ymin, ymax, Math.max(3, Math.round(height / 55)));
  ymin = Math.min(ymin, yt[0]); ymax = Math.max(ymax, yt[yt.length - 1]);
  const xmin = xs[0], xmax = xs[xs.length - 1];

  function draw() {
    width = Math.max(260, Math.floor(plot.clientWidth || 600));
    plot.querySelector("svg")?.remove();
    const m = { l: 52, r: 14, t: 30, b: 34 };
    const iw = width - m.l - m.r, ih = height - m.t - m.b;
    const X = v => m.l + ((v - xmin) / (xmax - xmin || 1)) * iw;
    const Y = v => m.t + (1 - (v - ymin) / (ymax - ymin || 1)) * ih;
    const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": opts.title || "chart" });
    plot.prepend(root);

    for (const v of yt) {
      svg("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), class: v === 0 ? "axis" : "grid" }, root);
      text(root, m.l - 6, Y(v) + 4, fmtNum(v), { class: "tick", "text-anchor": "end" });
    }
    for (const v of niceTicks(xmin, xmax, Math.max(3, Math.round(iw / 90)))) {
      text(root, X(v), m.t + ih + 18, fmtNum(v), { class: "tick", "text-anchor": "middle" });
    }
    if (opts.xLabel) text(root, m.l + iw, height - 2, opts.xLabel, { class: "tick", "text-anchor": "end" });
    if (opts.yUnit) text(root, 4, 11, opts.yUnit, { class: "tick" });

    let lastLabelX = -Infinity, row = 0;
    for (const ev of [...(opts.events || [])].sort((a, b) => a.x - b.x)) {
      if (ev.x === null || ev.x < xmin || ev.x > xmax) continue;
      svg("line", { x1: X(ev.x), x2: X(ev.x), y1: m.t, y2: m.t + ih, class: "event-line" }, root);
      row = X(ev.x) - lastLabelX < 34 ? row + 1 : 0;       // stack labels of events that sit close together
      text(root, X(ev.x) + 3, m.t - 5 - row * 12, ev.label, { class: "event-label" });
      lastLabelX = X(ev.x);
    }

    const placed = [];   // [x0, x1, y] of peak labels already drawn, so close peaks do not print over each other
    for (const s of opts.series) {
      let d = "";
      let pen = false;
      s.values.forEach((v, i) => {
        if (v === null || !isFinite(v)) { pen = false; return; }
        d += `${pen ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
        pen = true;
      });
      svg("path", { d, fill: "none", style: `stroke:${s.color}`, "stroke-width": 2, "stroke-linejoin": "round",
                    "stroke-linecap": "round" }, root);
      if (opts.peakLabels) {
        let bi = -1;
        s.values.forEach((v, i) => { if (v !== null && (bi < 0 || v > s.values[bi])) bi = i; });
        if (bi >= 0) {
          svg("circle", { cx: X(xs[bi]), cy: Y(s.values[bi]), r: 4, style: `fill:${s.color}`, class: "ring" }, root);
          const anchor = X(xs[bi]) > m.l + iw - 70 ? "end" : "start";
          const label = fmtNum(s.values[bi]);
          const lx = X(xs[bi]) + (anchor === "end" ? -7 : 7);
          const w = label.length * 7;
          const x0 = anchor === "end" ? lx - w : lx, x1 = x0 + w;
          let ly = Y(s.values[bi]) - 6;
          while (placed.some(p => x0 < p[1] && x1 > p[0] && Math.abs(ly - p[2]) < 12)) ly -= 12;
          placed.push([x0, x1, ly]);
          text(root, lx, ly, label, { class: "direct-label", "text-anchor": anchor, style: `fill:${s.color}` });
        }
      }
    }

    layer = svg("g", {}, root);
    const cross = svg("line", { y1: m.t, y2: m.t + ih, class: "crosshair", visibility: "hidden" }, layer);
    const head = svg("line", { y1: m.t, y2: m.t + ih, class: "playhead", visibility: "hidden" }, layer);
    const hit = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, root);

    const nearest = px => {
      const v = xmin + ((px - m.l) / iw) * (xmax - xmin);
      let lo = 0, hi = xs.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] < v) lo = mid; else hi = mid; }
      return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
    };
    const show = i => {
      cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i]));
      cross.setAttribute("visibility", "visible");
      tip.replaceChildren();
      const head = document.createElement("div");
      head.className = "tip-head";
      head.textContent = `${fmtNum(xs[i])} ${opts.xUnit || ""}`.trim();
      tip.appendChild(head);
      for (const s of opts.series) {
        const row = document.createElement("div");
        row.className = "tip-row";
        const key = document.createElement("span");
        key.className = "line-key";
        key.style.background = s.color;
        const val = document.createElement("strong");
        val.textContent = `${fmtNum(s.values[i], opts.decimals || 0)}`;
        const name = document.createElement("span");
        name.className = "tip-name";
        name.textContent = s.name;
        row.append(key, val, name);
        tip.appendChild(row);
      }
      tip.hidden = false;
      const left = X(xs[i]) + 12;
      tip.style.left = `${Math.min(left, width - tip.offsetWidth - 4)}px`;
      tip.style.top = `${m.t}px`;
      if (opts.onHover) opts.onHover(i);
    };
    hit.addEventListener("pointermove", e => {
      const r = root.getBoundingClientRect();
      show(nearest(e.clientX - r.left));
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; if (opts.onHover) opts.onHover(null); });
    hit.addEventListener("click", e => {
      const r = root.getBoundingClientRect();
      if (opts.onPick) opts.onPick(nearest(e.clientX - r.left));
    });

    wrap._setHead = x => {
      if (x === null || x < xmin || x > xmax) { head.setAttribute("visibility", "hidden"); return; }
      head.setAttribute("x1", X(x)); head.setAttribute("x2", X(x));
      head.setAttribute("visibility", "visible");
    };
    wrap._setHead(playheadX);
  }

  const ro = new ResizeObserver(() => {
    const w = Math.floor(plot.clientWidth);
    if (w && Math.abs(w - width) > 2) draw();
  });
  ro.observe(plot);
  requestAnimationFrame(draw);
  return {
    node: wrap,
    setPlayhead(x) { playheadX = x; if (wrap._setHead) wrap._setHead(x); },
    destroy() { ro.disconnect(); },
  };
}

/** Reference distribution strip: p10-p90 track, p25-p75 band, median tick, the pitcher's dot. */
export function bulletStrip(t, value, label) {
  const w = 150, h = 20;
  const root = svg("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, class: "bullet", role: "img" });
  if (!t) return root;
  let lo = t["10"], hi = t["90"];
  if (value !== null && value !== undefined) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
  const span = (hi - lo) || 1;
  lo -= span * 0.06; hi += span * 0.06;
  const X = v => 4 + ((v - lo) / (hi - lo)) * (w - 8);
  svg("rect", { x: X(t["10"]), y: 8, width: Math.max(1, X(t["90"]) - X(t["10"])), height: 4, rx: 2, class: "b-track" }, root);
  svg("rect", { x: X(t["25"]), y: 6, width: Math.max(1, X(t["75"]) - X(t["25"])), height: 8, rx: 2, class: "b-iqr" }, root);
  svg("rect", { x: X(t["50"]) - 1, y: 4, width: 2, height: 12, class: "b-median" }, root);
  if (value !== null && value !== undefined) {
    svg("circle", { cx: X(value), cy: 10, r: 5, class: "b-dot ring" }, root);
  }
  const title = svg("title", {}, root);
  title.textContent = label || "";
  return root;
}

/**
 * Horizontal diverging bars around zero. items: [{label, delta}] -> HTML block.
 * positive = moving the metric to the reference median would raise the prediction.
 */
export function divergingBars(items, unit, decimals = 1) {
  const box = document.createElement("div");
  box.className = "diverging";
  const maxAbs = Math.max(...items.map(i => Math.abs(i.delta)), 1e-9);
  for (const it of items) {
    const row = document.createElement("div");
    row.className = "d-row";
    row.tabIndex = 0;
    const name = document.createElement("span");
    name.className = "d-label";
    name.textContent = it.label;
    const track = document.createElement("span");
    track.className = "d-track";
    const bar = document.createElement("span");
    bar.className = `d-bar ${it.delta >= 0 ? "pos" : "neg"}`;
    const pct = (Math.abs(it.delta) / maxAbs) * 50;
    bar.style.width = `${pct}%`;
    bar.style.left = it.delta >= 0 ? "50%" : `${50 - pct}%`;
    const zero = document.createElement("span");
    zero.className = "d-zero";
    track.append(zero, bar);
    const val = document.createElement("span");
    val.className = "d-value";
    val.textContent = `${it.delta >= 0 ? "+" : "−"}${fmtNum(Math.abs(it.delta), decimals)} ${unit}`;
    row.title = it.title || "";
    row.append(name, track, val);
    box.appendChild(row);
  }
  return box;
}
