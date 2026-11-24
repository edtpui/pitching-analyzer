// Pitch tracking: velocity, movement and spin of one pitch from a slow-motion clip filmed along the pitching
// line. Views: list, upload, field marking (calibration), progress, results.
// The helpers (el, api, section, ...) come from app.js so both sections look and behave the same.

const DISTANCES = [["60.5", "60 ft 6 in (high school and up)"], ["54", "54 ft (13-14U)"], ["50", "50 ft (12-13U)"],
                   ["46", "46 ft (Little League)"]];
const MOUND_IN = { "60.5": 10, "54": 8, "50": 8, "46": 6 };
const NS = "http://www.w3.org/2000/svg";

export function flightViews(h) {
  const { el, api, section, errorBox, badge, statusCell, stackTable, fmtDate, fmtNum, uploadXHR, radioGroup, onTeardown, $view } = h;
  const active = s => ["uploading", "queued", "running"].includes(s);
  const sd = (v, d = 1) => (v === null || v === undefined ? "" : ` ±${fmtNum(v, d)}`);
  const num = (v, d = 1) => (v === null || v === undefined ? "–" : fmtNum(v, d));

  // ---------------------------------------------------------------- list
  async function list() {
    $view.append(el("div", { class: "page-head" }, el("h1", { text: "Pitch tracking" }),
      el("span", { class: "muted small", text: "Velocity, movement and spin from a slow-motion clip filmed behind home plate" }),
      el("div", { class: "actions" }, el("a", { class: "button", href: "#/flights/new", text: "New pitch" }))));
    const body = el("div");
    $view.append(body);
    const render = items => {
      if (!items.length) {
        body.replaceChildren(section("No pitches yet", null, el("div", { class: "empty" },
          el("p", { text: "Upload a 240 fps clip filmed from behind home plate with the plate and the rubber in view." }),
          el("a", { class: "button", href: "#/flights/new", text: "Upload a pitch" }))));
        return;
      }
      const heads = [["Clip"], ["Uploaded"], ["Status"], ["Velocity", "num"], ["Spin", "num"], ["Tilt"], ["IVB", "num"], ["HB (arm side)", "num"], ["Result"]];
      const tb = el("tbody");
      for (const m of items) {
        const s = m.summary;
        const status = m.status === "calibrate" ? el("span", { class: "status queued", text: "Mark the field" }) : statusCell(m);
        tb.append(el("tr", {},
          el("td", {}, el("a", { href: `#/f/${m.id}`, text: m.options?.label || m.filename })),
          el("td", { class: "muted small", text: fmtDate(m.created_utc) }),
          el("td", {}, status),
          el("td", { class: "num", text: s ? `${num(s.release_speed_mph)} mph` : "–" }),
          el("td", { class: "num", text: s?.spin_rpm ? `${s.spin_rpm} rpm` : "–" }),
          el("td", { text: s?.tilt || "–" }),
          el("td", { class: "num", text: s?.ivb_in != null ? `${num(s.ivb_in)} in` : "–" }),
          el("td", { class: "num", text: s?.hb_arm_side_in != null ? `${num(s.hb_arm_side_in)} in` : "–" }),
          el("td", { text: s ? (s.strike ? "Strike" : "Ball") : "–" })));
      }
      body.replaceChildren(el("section", { class: "block" }, stackTable(el("table", {},
        el("thead", {}, el("tr", {}, heads.map(([t, c]) => el("th", { class: c || null, text: t })))), tb))));
    };
    const load = async () => {
      const items = await api("/api/flights");
      render(items);
      return items.some(m => active(m.status));
    };
    let busy = await load();
    const timer = setInterval(async () => { if (busy) busy = await load().catch(() => true); }, 2000);
    onTeardown(() => clearInterval(timer));
  }

  // ---------------------------------------------------------------- upload
  async function create() {
    $view.append(el("div", { class: "page-head" }, el("h1", { text: "New pitch" })));
    const file = el("input", { type: "file", accept: "video/*,.mov,.mp4,.m4v,.avi,.mkv,.webm", hidden: true, id: "ffile" });
    const fileName = el("span", { class: "muted", text: "MP4, MOV, M4V, AVI, MKV or WebM" });
    const drop = el("label", { class: "dropzone", for: "ffile" }, el("strong", { text: "Choose a video or drop it here" }), fileName);
    ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", e => { if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; file.dispatchEvent(new Event("change")); } });
    file.addEventListener("change", () => { const f = file.files[0]; fileName.textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : "MP4, MOV, M4V, AVI, MKV or WebM"; });
    const n = (name, attrs = {}) => el("input", { type: "number", name, step: "any", ...attrs });
    const dist = el("select", { name: "distance_ft" }, DISTANCES.map(([v, t]) => el("option", { value: v, text: t })));
    const mound = n("mound_in", { min: 0, max: 18, value: 10 });
    dist.addEventListener("change", () => { mound.value = MOUND_IN[dist.value] ?? 10; });
    const status = el("div", { class: "small muted", role: "status" });
    const bar = el("div", { class: "progress", hidden: true }, el("span", { style: "width:0%" }));
    const submit = el("button", { type: "submit", text: "Upload" });
    const form = el("form", { class: "fields", novalidate: true },
      el("label", { class: "f", for: "ffile", text: "Video" }), drop, file,
      el("label", { class: "f", text: "Label" }), el("input", { name: "label", placeholder: "e.g. Bullpen 11/24 - FB #2", maxlength: 120 }),
      el("label", { class: "f", text: "Throwing hand" }), radioGroup("throws", [["R", "Right"], ["L", "Left"]], "R"),
      el("label", { class: "f", text: "Pitching distance" }), dist,
      el("label", { class: "f", text: "Mound height" }), el("div", { class: "row" }, mound, el("span", { class: "muted small", text: "in above home plate" })),
      el("div", { class: "sep", text: "Capture" }),
      el("label", { class: "f", text: "Capture frame rate" }), el("div", { class: "row" }, n("capture_fps", { min: 15, max: 2000, placeholder: "from file" }), el("span", { class: "muted small", text: "fps" })),
      el("div", { class: "help", text: "Leave empty unless the clip is a slow-motion export that plays slowed down; then enter the rate it was filmed at (e.g. 240)." }),
      el("label", { class: "f", text: "Trim (seconds)" }), el("div", { class: "row" }, n("trim_start_s", { min: 0, placeholder: "start" }), "to", n("trim_end_s", { min: 0, placeholder: "end" })),
      el("div", { class: "help", text: "Optional. Keep about 1 s before release to 0.5 s after the catch; at most 10 s at 240 fps." }),
      el("div", { class: "sep", text: "Strike zone and conditions" }),
      el("label", { class: "f", text: "Zone bottom / top" }), el("div", { class: "row" }, n("zone_bottom_in", { min: 6, max: 48, value: 18 }), "to", n("zone_top_in", { min: 18, max: 72, value: 42 }), el("span", { class: "muted small", text: "in above the ground" })),
      el("label", { class: "f", text: "Pitcher height" }), el("div", { class: "row" }, n("pitcher_height_in", { min: 40, max: 90 }), el("span", { class: "muted small", text: "in (optional: sets the expected release extension)" })),
      el("label", { class: "f", text: "Altitude" }), el("div", { class: "row" }, n("altitude_ft", { min: -500, max: 12000, placeholder: "0" }), el("span", { class: "muted small", text: "ft (thinner air: less drag and less break)" })),
      el("div", {}), el("div", { class: "row" }, submit, bar, status));
    form.addEventListener("submit", async e => {
      e.preventDefault();
      if (!file.files.length) { status.textContent = "Choose a video first."; return; }
      const fd0 = new FormData(form);
      const fd = new FormData();
      fd.append("file", file.files[0]);
      for (const k of ["label", "throws", "distance_ft", "mound_in", "capture_fps", "trim_start_s", "trim_end_s", "zone_bottom_in", "zone_top_in", "pitcher_height_in", "altitude_ft"]) {
        const v = fd0.get(k);
        if (v !== null && v !== "") fd.append(k, v);
      }
      submit.disabled = true; bar.hidden = false; status.textContent = "Uploading…";
      try {
        const res = await uploadXHR(fd, f => { bar.firstChild.style.width = `${Math.round(f * 100)}%`; status.textContent = `Uploading… ${Math.round(f * 100)}%`; }, "/api/flights");
        location.hash = `#/f/${res.id}`;
      } catch (err) { status.textContent = err.message; submit.disabled = false; bar.hidden = true; }
    });
    const tips = el("aside", { class: "block tips" }, el("h2", { text: "Filming checklist", style: "margin-bottom:8px" }),
      el("ol", {},
        el("li", {}, el("strong", { text: "Behind home plate, on the line to the mound: " }), "tripod about 10-15 ft behind the plate, 4-6 ft high (above the catcher's head if there is one), landscape."),
        el("li", {}, el("strong", { text: "Plate and rubber in view, unmoving: " }), "both must be visible in at least one frame - start recording before the pitcher steps onto the rubber. No zoom or panning while filming."),
        el("li", {}, el("strong", { text: "240 fps slow motion. " }), "Spin from the seams needs 240 fps and the ball at least 14 px across near the plate; 120 fps gives velocity and movement only."),
        el("li", {}, el("strong", { text: "Bright light, fast shutter: " }), "outdoors in daylight works best; the ball must stand out from what is behind it."),
        el("li", {}, el("strong", { text: "One pitch per clip, " }), "from before the delivery to after the catch."),
        el("li", {}, el("strong", { text: "Behind the pitcher " }), "(raised and to the arm side) also works, with larger errors.")));
    $view.append(el("div", { class: "form-grid" }, el("section", { class: "block" }, form), tips));
  }

  // ---------------------------------------------------------------- detail
  async function detail(id) {
    const root = el("div");
    $view.append(root);
    let data = await api(`/api/flights/${encodeURIComponent(id)}`);
    const show = () => {
      const s = data.meta.status;
      if (active(s)) return renderProgress(root, data.meta);
      if (s === "calibrate") return renderCalibrate(root, data, refresh);
      if (s === "done") return renderDone(root, data, refresh);
      return renderFailed(root, data, refresh);
    };
    async function refresh(newData) {
      data = newData || await api(`/api/flights/${encodeURIComponent(id)}`);
      show();
    }
    show();
    const timer = setInterval(async () => {
      if (!active(data.meta.status)) return;
      try { await refresh(); } catch (e) { root.replaceChildren(errorBox(e.message)); }
    }, 1000);
    onTeardown(() => clearInterval(timer));
  }

  function headRow(meta, ...extra) {
    const del = el("button", { class: "danger", text: "Delete", onclick: async () => {
      if (!confirm("Delete this pitch and its files? This cannot be undone.")) return;
      try { await api(`/api/flights/${meta.id}`, { method: "DELETE" }); location.hash = "#/flights"; } catch (e) { alert(e.message); }
    } });
    return el("div", { class: "page-head" }, el("h1", { text: meta.options?.label || meta.filename }),
      el("span", { class: "muted small", text: `${meta.filename} · uploaded ${fmtDate(meta.created_utc)}` }),
      el("div", { class: "actions" }, ...extra, active(meta.status) ? null : del));
  }

  function renderProgress(root, meta) {
    const what = meta.phase === "analyze" ? "Analysing the pitch" : "Preparing the clip";
    root.replaceChildren(headRow(meta), section(what, null,
      el("div", { class: "progress", style: "height:8px;max-width:520px" }, el("span", { style: `width:${Math.round((meta.progress || 0) * 100)}%` })),
      el("p", { class: "muted", style: "margin-top:8px", text: `${meta.stage || meta.status} · ${Math.round((meta.progress || 0) * 100)}%` }),
      el("p", { class: "small muted", text: meta.phase === "analyze" ? "Finding and measuring the ball, fitting the flight and reading the seams usually takes 20-60 s." : "Building a picture of the field with the people removed, for marking home plate and the rubber." })));
  }

  function renderFailed(root, data, refresh) {
    const meta = data.meta;
    const again = data.calibration ? el("button", { text: "Run again", onclick: async () => { await api(`/api/flights/${meta.id}/analyze`, { method: "POST" }); refresh(); } }) : null;
    const remark = data.calibration ? el("button", { class: "secondary", text: "Re-mark the field", onclick: () => renderCalibrate(root, data, refresh) }) : null;
    root.replaceChildren(headRow(meta, remark, again), errorBox(meta.error || "The analysis failed.",
      el("p", { class: "small", style: "margin-top:8px" }, "Check the filming checklist on the ", el("a", { href: "#/flights/new", text: "upload page" }),
        ": home plate and the rubber in view, the camera still, the whole flight in the clip, ideally at 240 fps.")));
  }

  // ---------------------------------------------------------------- field marking
  async function renderCalibrate(root, data, refresh) {
    const meta = data.meta;
    const opts = meta.options || {};
    const prev = data.calibration;
    const marks = { plate: prev ? prev.plate_px.map(p => [...p]) : [], rubber: prev ? prev.rubber_px.map(p => [...p]) : [] };
    let overlay = null;
    let imageSource = prev && prev.image !== undefined ? prev.image : "background";
    const img = new Image();
    const canvas = el("canvas", { class: "calib-canvas", tabindex: 0, "aria-label": "Field image: click to mark home plate and the rubber" });
    const wrap = el("div", { class: "calib-wrap" }, canvas);
    const counter = el("span", { class: "small" });
    const summary = el("div", { class: "small", role: "status" });
    const view = { s: 1, ox: 0, oy: 0 };
    let dpr = window.devicePixelRatio || 1;

    const nFrames = meta.video?.frame_count || 0;
    const srcSel = el("select", { "aria-label": "Picture to mark" }, el("option", { value: "background", text: "Background (people removed)" }), el("option", { value: "frame", text: "A frame of the clip" }));
    const frameNo = el("input", { type: "number", min: 0, max: Math.max(0, nFrames - 1), value: typeof imageSource === "number" ? imageSource : (meta.video?.first_frame ?? 0), style: "width:90px", "aria-label": "Frame number" });
    const loadBtn = el("button", { class: "secondary", text: "Show", onclick: () => loadImage() });
    srcSel.value = typeof imageSource === "number" ? "frame" : "background";
    const syncSrc = () => { frameNo.hidden = loadBtn.hidden = srcSel.value !== "frame"; };
    srcSel.addEventListener("change", () => { syncSrc(); if (srcSel.value === "background") loadImage(); });
    syncSrc();

    const fld = (name, attrs) => el("input", { type: "number", step: "any", name, ...attrs });
    const distIn = el("select", {}, DISTANCES.map(([v, t]) => el("option", { value: v, text: t })));
    distIn.value = String(prev?.distance_ft ?? opts.distance_ft ?? 60.5);
    if (![...distIn.options].some(o => o.value === distIn.value)) distIn.append(el("option", { value: distIn.value, text: `${distIn.value} ft` })) && (distIn.value = String(prev?.distance_ft ?? opts.distance_ft));
    const moundIn = fld("mound_in", { min: 0, max: 18, value: prev?.mound_in ?? opts.mound_in ?? 10, style: "width:80px" });
    const camH = fld("camera_height_ft", { min: 0.5, max: 30, value: prev?.camera_height_ft ?? "", placeholder: "optional", style: "width:90px" });
    const check = el("button", { class: "secondary", text: "Check marks" });
    const analyze = el("button", { text: "Analyze pitch", disabled: !prev });
    const copySel = el("select", { "aria-label": "Copy the marks of another clip" }, el("option", { value: "", text: "Copy marks from another clip…" }));
    api("/api/flights").then(items => {
      for (const m of items) {
        if (m.id === meta.id || !m.video || m.video.width !== meta.video?.width || m.video.height !== meta.video?.height) continue;
        if (!["done", "calibrate", "failed"].includes(m.status)) continue;
        copySel.append(el("option", { value: m.id, text: `${m.options?.label || m.filename} · ${fmtDate(m.created_utc)}` }));
      }
    }).catch(() => {});

    function fit() {
      const w = wrap.clientWidth;
      const hMax = Math.max(240, window.innerHeight * 0.68);
      const scale = Math.min(w / img.naturalWidth, hMax / img.naturalHeight);
      const cw = Math.round(img.naturalWidth * scale), ch = Math.round(img.naturalHeight * scale);
      dpr = window.devicePixelRatio || 1;
      canvas.style.width = `${cw}px`; canvas.style.height = `${ch}px`;
      canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
      view.s = scale; view.ox = 0; view.oy = 0;
      draw();
    }
    const toImg = (cx, cy) => [(cx - view.ox) / view.s, (cy - view.oy) / view.s];
    const toCan = (x, y) => [x * view.s + view.ox, y * view.s + view.oy];

    function draw() {
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr);
      if (!img.naturalWidth) return;
      ctx.imageSmoothingEnabled = view.s < 3;
      ctx.drawImage(img, view.ox, view.oy, img.naturalWidth * view.s, img.naturalHeight * view.s);
      if (overlay) {
        ctx.lineWidth = 1.5;
        const line = (pts, color, closed) => {
          if (!pts) return;
          ctx.strokeStyle = color; ctx.beginPath();
          pts.forEach(([x, y], i) => { const [a, b] = toCan(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
          if (closed) ctx.closePath();
          ctx.stroke();
        };
        line(overlay.line, "rgba(255,255,255,0.6)", false);
        line(overlay.plate, "#ffd23f", true);
        line(overlay.rubber, "#ffd23f", true);
        line(overlay.zone, "#ffffff", true);
      }
      const mark = (p, label, color) => {
        const [a, b] = toCan(p[0], p[1]);
        ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(a, b, 7, 0, 2 * Math.PI); ctx.stroke();
        ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(a, b, 7, 0, 2 * Math.PI); ctx.moveTo(a - 11, b); ctx.lineTo(a + 11, b); ctx.moveTo(a, b - 11); ctx.lineTo(a, b + 11); ctx.stroke();
        ctx.font = "600 12px system-ui, sans-serif"; ctx.lineWidth = 3; ctx.strokeStyle = "#000"; ctx.strokeText(label, a + 9, b - 9);
        ctx.fillStyle = color; ctx.fillText(label, a + 9, b - 9);
      };
      marks.plate.forEach((p, i) => mark(p, `P${i + 1}`, "#4fc3ff"));
      marks.rubber.forEach((p, i) => mark(p, `R${i + 1}`, "#ff9a3c"));
      counter.textContent = `Home plate ${marks.plate.length}/5 · rubber ${marks.rubber.length}/2` +
        (marks.plate.length < 5 ? " · next: a corner of home plate" : marks.rubber.length < 2 ? " · next: a front corner of the rubber" : " · all marked");
      check.disabled = marks.plate.length < 5 || marks.rubber.length < 2;
    }

    function hit(cx, cy) {
      for (const kind of ["plate", "rubber"]) {
        for (let i = 0; i < marks[kind].length; i++) {
          const [a, b] = toCan(...marks[kind][i]);
          if (Math.hypot(a - cx, b - cy) <= 10) return [kind, i];
        }
      }
      return null;
    }
    let drag = null;
    const pos = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    canvas.addEventListener("pointerdown", e => {
      if (e.button === 2) return;
      const [cx, cy] = pos(e);
      canvas.setPointerCapture(e.pointerId);
      drag = { start: [cx, cy], last: [cx, cy], target: hit(cx, cy), moved: false };
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      const [cx, cy] = pos(e);
      if (Math.hypot(cx - drag.start[0], cy - drag.start[1]) > 4) drag.moved = true;
      if (drag.target) {
        const [kind, i] = drag.target;
        marks[kind][i] = toImg(cx, cy);
        overlay = null;
      } else if (drag.moved) {
        view.ox += cx - drag.last[0]; view.oy += cy - drag.last[1];
      }
      drag.last = [cx, cy];
      draw();
    });
    canvas.addEventListener("pointerup", e => {
      if (!drag) return;
      const [cx, cy] = pos(e);
      if (!drag.moved && !drag.target) {
        const p = toImg(cx, cy);
        if (p[0] >= 0 && p[1] >= 0 && p[0] <= img.naturalWidth && p[1] <= img.naturalHeight) {
          if (marks.plate.length < 5) marks.plate.push(p);
          else if (marks.rubber.length < 2) marks.rubber.push(p);
          overlay = null;
        }
      }
      drag = null;
      draw();
    });
    canvas.addEventListener("contextmenu", e => {
      e.preventDefault();
      const t = hit(...pos(e));
      if (t) { marks[t[0]].splice(t[1], 1); overlay = null; draw(); }
    });
    const zoomAt = (cx, cy, f) => {
      const ns = Math.min(Math.max(view.s * f, 0.05), 40);
      const [x, y] = toImg(cx, cy);
      view.s = ns; view.ox = cx - x * ns; view.oy = cy - y * ns;
      draw();
    };
    canvas.addEventListener("wheel", e => { e.preventDefault(); const [cx, cy] = pos(e); zoomAt(cx, cy, e.deltaY < 0 ? 1.25 : 0.8); }, { passive: false });
    const centre = () => [canvas.width / dpr / 2, canvas.height / dpr / 2];

    async function loadImage() {
      imageSource = srcSel.value === "frame" ? Math.max(0, parseInt(frameNo.value || "0", 10)) : "background";
      const src = imageSource === "background" ? `/api/flights/${meta.id}/background.jpg?v=${encodeURIComponent(meta.finished_utc || "")}` : `/api/flights/${meta.id}/frames/${imageSource}.jpg`;
      summary.textContent = "Loading the picture…";
      await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error("The picture could not be loaded.")); img.src = src; })
        .then(() => { summary.textContent = ""; fit(); })
        .catch(e => { summary.textContent = e.message; });
    }

    function showResult(rec) {
      overlay = rec.overlay;
      const p = rec.camera_position_ft;
      const behind = p[1] < 0 ? `${fmtNum(-p[1], 1)} ft behind the back of the plate` : `${fmtNum(p[1] - rec.distance_ft, 1)} ft behind the rubber`;
      const ref = rec.refinement || {};
      summary.replaceChildren(
        el("strong", { text: rec.view === "behind_plate" ? "Camera behind home plate. " : rec.view === "behind_pitcher" ? "Camera behind the pitcher. " : "Camera to the side. " }),
        `${behind}, ${fmtNum(p[2], 1)} ft high, ${fmtNum(Math.abs(p[0]), 1)} ft to the ${p[0] >= 0 ? "first" : "third"}-base side; field of view ${fmtNum(rec.hfov_deg, 0)}°. `,
        `Marks fit within ${fmtNum(rec.rms_px, 1)} px (worst ${fmtNum(Math.max(...rec.residuals_px), 1)} px). `,
        ref.used ? "The plate and rubber outlines were matched to the picture." : el("span", { class: "warn-text", text: `Outlines not matched (${ref.reason || "no reason given"}): the clicks are used as they are.` }),
        el("br"), "Check that the yellow outlines sit on the plate and the rubber and that the white box is where the strike zone should be.");
      analyze.disabled = false;
      draw();
    }

    check.addEventListener("click", async () => {
      check.disabled = true; summary.textContent = "Solving the camera…";
      try {
        const body = { plate_px: marks.plate, rubber_px: marks.rubber, image: imageSource, distance_ft: parseFloat(distIn.value), mound_in: parseFloat(moundIn.value) };
        const h = parseFloat(camH.value);
        if (isFinite(h)) body.camera_height_ft = h;
        showResult(await api(`/api/flights/${meta.id}/calibration`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
      } catch (e) { summary.textContent = e.message; analyze.disabled = true; }
      finally { check.disabled = marks.plate.length < 5 || marks.rubber.length < 2; }
    });
    copySel.addEventListener("change", async () => {
      if (!copySel.value) return;
      summary.textContent = "Copying the marks…";
      try {
        const rec = await api(`/api/flights/${meta.id}/calibration`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ copy_from: copySel.value }) });
        marks.plate = rec.plate_px.map(p => [...p]); marks.rubber = rec.rubber_px.map(p => [...p]);
        srcSel.value = "background"; syncSrc(); await loadImage();
        showResult(rec);
      } catch (e) { summary.textContent = e.message; }
      copySel.value = "";
    });
    analyze.addEventListener("click", async () => {
      analyze.disabled = true;
      try { await api(`/api/flights/${meta.id}/analyze`, { method: "POST" }); refresh(); }
      catch (e) { summary.textContent = e.message; analyze.disabled = false; }
    });

    const tools = el("div", { class: "calib-tools" },
      el("button", { class: "secondary", title: "Zoom in", text: "+", onclick: () => zoomAt(...centre(), 1.5) }),
      el("button", { class: "secondary", title: "Zoom out", text: "−", onclick: () => zoomAt(...centre(), 1 / 1.5) }),
      el("button", { class: "secondary", text: "Fit", onclick: () => fit() }),
      el("button", { class: "secondary", text: "Undo", onclick: () => { (marks.rubber.length ? marks.rubber : marks.plate).pop(); overlay = null; draw(); } }),
      el("button", { class: "secondary", text: "Clear", onclick: () => { marks.plate = []; marks.rubber = []; overlay = null; draw(); } }),
      counter);
    const back = data.result ? el("button", { class: "secondary", text: "Back to the results", onclick: () => refresh() }) : null;
    root.replaceChildren(headRow(meta, back),
      el("div", { class: "calib-grid" },
        el("section", { class: "block" }, el("div", { class: "calib-src" }, srcSel, frameNo, loadBtn), wrap, tools),
        el("aside", { class: "block" },
          el("h2", { text: "Mark the field", style: "margin-bottom:6px" }),
          el("ol", { class: "small calib-steps" },
            el("li", {}, "Click the ", el("strong", { text: "five corners of home plate" }), " (any order)."),
            el("li", {}, "Click the ", el("strong", { text: "two front corners of the pitching rubber" }), " - the edge facing home plate."),
            el("li", {}, "Zoom in (mouse wheel or +) and place each mark exactly on its corner; drag a mark to move it, right-click to remove it."),
            el("li", {}, "If someone covers the plate or the rubber, pick a frame where both are visible.")),
          el("div", { class: "fields compact" },
            el("label", { class: "f", text: "Pitching distance" }), distIn,
            el("label", { class: "f", text: "Mound height (in)" }), moundIn,
            el("label", { class: "f", text: "Camera height (ft)" }), camH,
            el("div", { class: "help", text: "Optional: tape-measured from the ground to the lens. Tightens the calibration." })),
          el("div", { class: "row calib-actions" }, check, analyze),
          copySel, summary)));
    new ResizeObserver(() => { if (img.naturalWidth) fit(); }).observe(wrap);
    await loadImage();
    if (prev) {
      try { showResult(await api(`/api/flights/${meta.id}/calibration`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plate_px: prev.plate_px, rubber_px: prev.rubber_px, image: prev.image, distance_ft: prev.distance_ft, mound_in: prev.mound_in, camera_height_ft: prev.camera_height_ft }) })); }
      catch { /* stale marks: the user re-marks */ }
    }
  }

  // ---------------------------------------------------------------- results
  function zonePlot(p, setup) {
    // catcher's view, inches, true proportions; x to the catcher's right (first base)
    const zb = setup.zone_bottom_in, zt = setup.zone_top_in;
    const zLo = Math.min(zb - 12, (p.plate_z_in ?? zb) - 6), zHi = Math.max(zt + 10, (p.plate_z_in ?? zt) + 6);
    const xHalf = Math.max(22, Math.abs(p.plate_x_in ?? 0) + 6);
    const sc = 240 / (2 * xHalf), W = 240, plateH = 22, H = Math.round((zHi - zLo) * sc) + plateH;
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("class", "zone-plot"); s.setAttribute("role", "img");
    s.setAttribute("aria-label", "Pitch location in the strike zone, catcher's view");
    const X = x => W / 2 + x * sc, Y = z => (zHi - z) * sc;
    const add = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); s.append(e); return e; };
    add("rect", { x: X(-8.5), y: Y(zt), width: 17 * sc, height: (zt - zb) * sc, class: "zone-box" });
    for (const f of [1 / 3, 2 / 3]) {
      add("line", { x1: X(-8.5 + 17 * f), x2: X(-8.5 + 17 * f), y1: Y(zt), y2: Y(zb), class: "zone-third" });
      add("line", { x1: X(-8.5), x2: X(8.5), y1: Y(zb + (zt - zb) * f), y2: Y(zb + (zt - zb) * f), class: "zone-third" });
    }
    const py = H - plateH + 6;
    add("polygon", { points: [[-8.5, 0], [8.5, 0], [8.5, 4], [0, 9], [-8.5, 4]].map(([x, y]) => `${X(x)},${py + y}`).join(" "), class: "zone-plate" });
    if (p.plate_x_in != null) {
      const c = add("circle", { cx: X(p.plate_x_in), cy: Y(p.plate_z_in), r: 1.45 * sc, class: p.strike ? "zone-ball strike" : "zone-ball" });
      const t = document.createElementNS(NS, "title");
      t.textContent = `${fmtNum(Math.abs(p.plate_x_in), 1)} in ${p.plate_x_in >= 0 ? "to the first-base side" : "to the third-base side"}, ${fmtNum(p.plate_z_in, 1)} in high`;
      c.append(t);
    }
    const label = add("text", { x: 4, y: 12, class: "zone-label" });
    label.textContent = "Catcher's view";
    return s;
  }

  function tile(label, value, sub, extra) {
    return el("div", { class: "tile" }, el("div", { class: "label", text: label }), el("div", { class: "value" }, value, extra || null),
      sub ? el("div", { class: "sub" }, sub) : null);
  }

  function renderDone(root, data, refresh) {
    const meta = data.meta, r = data.result;
    const p = r.pitch, u = r.uncertainty || {}, spin = r.spin || {}, mv = r.movement_spin || {};
    const id = meta.id;
    // video
    const video = el("video", { src: `/api/flights/${id}/tracer.mp4?v=${encodeURIComponent(r.computed_utc)}`, controls: true, playsInline: true, muted: true, preload: "auto" });
    const tr = r.tracer;
    const current = () => Math.max(0, Math.min(tr.frames - 1, Math.floor(video.currentTime * tr.fps + 1e-4)));
    const seek = k => { video.pause(); video.currentTime = (Math.max(0, Math.min(tr.frames - 1, k)) + 0.5) / tr.fps; };
    const readout = el("span", { class: "readout" });
    const update = () => { const k = current(); readout.textContent = `frame ${k + 1} / ${tr.frames}` + (tr.first_frame + k === r.release.frame ? " · release" : ""); };
    video.addEventListener("timeupdate", update); video.addEventListener("seeked", update); video.addEventListener("loadedmetadata", update);
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) { const loop = () => { update(); video.requestVideoFrameCallback(loop); }; video.requestVideoFrameCallback(loop); }
    const speed = el("select", { "aria-label": "Playback speed", onchange: e => { video.playbackRate = parseFloat(e.target.value); } },
      ["0.25", "0.1", "0.5", "1"].map(v => el("option", { value: v, text: `${v}×` })));
    video.addEventListener("loadedmetadata", () => { video.playbackRate = parseFloat(speed.value); });
    const keyHandler = e => {
      if (e.target instanceof Element && e.target.matches("input, select, textarea")) return;
      if (e.key === ",") { seek(current() - 1); e.preventDefault(); }
      if (e.key === ".") { seek(current() + 1); e.preventDefault(); }
    };
    document.addEventListener("keydown", keyHandler);
    onTeardown(() => document.removeEventListener("keydown", keyHandler));
    const rel = r.release;
    const relText = { auto: "first frame the ball was seen out of the hand", manual: "set by you", estimated: "not seen - estimated from the flight" }[rel.source] || rel.source;
    const relStatus = el("span", { class: "small muted", role: "status" });
    const setRel = async frame => {
      relStatus.textContent = "Re-fitting…";
      try { const out = await api(`/api/flights/${id}/release`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ frame }) }); refresh(out); }
      catch (e) { relStatus.textContent = e.message; }
    };
    const relRow = el("div", { class: "release-row" },
      el("span", {}, el("strong", { text: "Release: " }), rel.frame != null ? `frame ${rel.frame - tr.first_frame + 1}` : "–", el("span", { class: "muted small", text: ` (${relText})` })),
      el("button", { class: "ghost", text: "Go to", disabled: rel.frame == null, onclick: () => seek(rel.frame - tr.first_frame) }),
      el("button", { class: "secondary", text: "Set release here", title: "Step to the frame where the ball leaves the hand (keys , and .) and click", onclick: () => setRel(tr.first_frame + current()) }),
      rel.source === "manual" ? el("button", { class: "ghost", text: "Automatic", onclick: () => setRel(null) }) : null,
      relStatus);
    const controls = el("div", { class: "controls" },
      el("button", { class: "secondary", title: "Previous frame (,)", text: "◀ frame", onclick: () => seek(current() - 1) }),
      el("button", { class: "secondary", title: "Next frame (.)", text: "frame ▶", onclick: () => seek(current() + 1) }), speed, readout);

    // tiles
    const arm = p.hb_arm_side_in;
    const spinOk = spin.ok;
    const tiles = el("div", { class: "tiles" },
      tile("Velocity at release", `${num(p.release_speed_mph)} mph`, `${sd(u.release_speed_mph).trim() || ""} · ${num(p.plate_speed_mph)} mph at the plate · ${num(p.flight_time_s, 3)} s flight`),
      tile("Spin rate (seams)", spinOk ? `${Math.round(spin.rpm)} rpm` : "Not measured",
        spinOk ? `${Math.round(spin.efficiency * 100)}% efficiency · ${Math.round(spin.gyro_deg)}° gyro · active ${Math.round(spin.active_rpm)} rpm` : (spin.reason || "needs 240 fps and a close, sharp ball"),
        spinOk ? el("span", { class: "tile-badge" }, badge(spin.confidence)) : null),
      tile("Spin axis (tilt)", spinOk ? spin.tilt_clock : (mv.tilt_clock || "–"),
        spinOk ? `${Math.round(spin.axis_deg)}° · from the seams; movement says ${mv.tilt_clock || "–"}` : `${Math.round(mv.axis_deg ?? 0)}° · from the movement (seams not read)`),
      tile("Induced vertical break", `${num(p.ivb_in)} in`, `${sd(u.ivb_in).trim()} · with gravity ${num(p.vb_in)} in`),
      tile("Horizontal break", arm == null ? "–" : `${num(Math.abs(arm))} in`, arm == null ? "" : `${arm >= 0 ? "arm side" : "glove side"}${sd(u.hb_in)}`),
      tile("Active spin from movement", mv.active_rpm_est ? `≈${Math.round(mv.active_rpm_est)} rpm` : "–", "Magnus lift model, about ±20%"),
      tile("Release point", `${num(p.release_height_ft)} ft`, `height · side ${num(Math.abs(p.release_side_ft))} ft ${p.release_side_ft >= 0 ? "1B" : "3B"} · extension ${num(p.extension_ft)} ft`),
      tile("Location", p.strike ? "Strike" : "Ball", `${num(Math.abs(p.plate_x_in))} in ${p.plate_x_in >= 0 ? "1B side" : "3B side"} of centre, ${num(p.plate_z_in)} in high`));
    const warnings = r.quality.warnings || [];
    const warnBlock = warnings.length ? el("div", { style: "margin-top:4px" }, el("h3", { text: `Notes (${warnings.length})`, style: "margin-bottom:4px" }),
      el("ul", { class: "warnings" }, warnings.map(w => el("li", { text: w })))) : el("p", { class: "small muted", text: "No warnings." });

    // details
    const c = r.calibration, f = r.fit;
    const rows = [
      ["Camera", `${c.view?.replace("_", " ")} · ${fmtNum(-c.camera_position_ft[1], 1)} ft behind the plate's back tip, ${fmtNum(c.camera_position_ft[2], 1)} ft high · field of view ${fmtNum(c.hfov_deg, 0)}°`],
      ["Field marks", `fit within ${fmtNum(c.rms_px, 2)} px · outlines ${c.refinement?.used ? "matched to the picture" : "not matched"}${c.camera_height_ft ? ` · camera height ${c.camera_height_ft} ft` : ""}`],
      ["Ball track", `${f.points} frames (${f.inliers} used), frames ${f.first_frame}-${f.last_frame}, residual ${fmtNum(f.rms_px, 2)} px`],
      ["Scale", `${rel.observed ? "release distance (time of flight over a known distance)" : "ball size only"} · ball-size factor ${fmtNum(f.size_scale, 3)}`],
      ["Frame rate", `${fmtNum(r.video.fps_effective, 0)} fps${r.video.fps !== r.video.fps_effective ? ` (file ${fmtNum(r.video.fps, 1)})` : ""} · ${r.video.width}×${r.video.height}`],
      ["Spin fit", spin.mean_score != null ? `${spin.frames?.length || 0} frames, seam score ${fmtNum(spin.mean_score, 2)}${spin.margin != null ? `, margin ${fmtNum(spin.margin, 2)}` : ""}` : "–"],
      ["Movement", `TrackMan-style: release to the front of the plate, relative to a spinless pitch; ${mv.lift_model || ""}`],
      ["Processing", r.processing_s ? `${r.processing_s} s` : "–"],
    ];
    const details = el("section", { class: "block" }, el("details", {}, el("summary", { text: "Method and data quality" }),
      el("dl", { class: "kv", style: "margin-top:10px" }, rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])),
      el("p", { class: "small muted", style: "margin-top:10px", text: "The camera is solved from your marks and refined so the outlines of home plate and the rubber match the picture. The ball is found in every frame against a background with people removed, measured by its outer edge, and a constant-acceleration flight (the PITCHf/x model) is fitted through the camera. Its distance comes from the release point (pitching distance minus extension, when the ball is seen leaving the hand), the ball's apparent size and drag. Spin is the rotation that best lines a 3D seam model up with the stitches frame after frame. All accuracy figures come from synthetic clips; real footage has not been checked against a radar or a tracking unit." })));

    const remark = el("button", { class: "secondary", text: "Re-mark the field", onclick: () => renderCalibrate(root, data, refresh) });
    root.replaceChildren(headRow(meta, remark),
      el("div", { class: "top-grid" },
        el("section", { class: "block player" }, video, controls, relRow),
        el("section", { class: "block" }, tiles, warnBlock)),
      el("div", { class: "flight-grid" },
        section("Location", `zone ${fmtNum(r.setup.zone_bottom_in, 0)}-${fmtNum(r.setup.zone_top_in, 0)} in`, zonePlot(p, r.setup)),
        spinOk && spin.strip ? section("Seam fit", "top: the ball; bottom: its stitches, with the fitted seam in yellow",
          el("img", { class: "seam-strip", src: `/api/flights/${id}/seams.jpg?v=${encodeURIComponent(r.computed_utc)}`, alt: "Ball crops and the fitted seam" })) :
          section("Seam fit", null, el("p", { class: "muted small", text: spin.reason ? `Spin was not read from the seams: ${spin.reason}.` : "No seam fit." }))),
      details);
    update();
  }

  return { list, create, detail };
}
