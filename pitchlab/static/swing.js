// Swing mechanics: one swing from a slow-motion clip, compared with Driveline's hitters. Views: list, upload,
// progress, results, reference. The helpers (el, api, section, ...) come from app.js so every section of the
// app looks and behaves the same.
import { bulletStrip, divergingBars, lineChart } from "./charts.js";
import { renderMarkdown } from "./markdown.js";

export function swingViews(h) {
  const { el, api, section, errorBox, badge, statusCell, stackTable, fmtDate, fmtNum, fmtVal, fmtRange, ordinal,
          uploadXHR, radioGroup, onTeardown, getHealth, $view } = h;
  const EVENT_ORDER = ["lift", "load", "fp", "contact"];
  const EVENT_NAMES = { lift: "Leg lift", load: "Load (max hip coil)", fp: "Foot plant", contact: "Contact" };
  const EVENT_SHORT = { lift: "LIFT", load: "LOAD", fp: "FP", contact: "CONTACT" };
  const LEVELS = [["", "Not specified"], ["high_school", "High school"], ["college", "College"],
                  ["independent", "Independent"], ["milb", "Pro (MiLB)"], ["youth", "Youth"], ["other", "Other"]];
  const GROUPS = ["Lower half", "Rotation", "Posture & arms", "Speed & sequencing"];
  const TARGET_NAMES = { bat_speed_mph: "Bat speed (mph)", exit_velo_mph: "Exit velocity (mph)" };
  const sideName = s => (s === "R" ? "right" : "left");

  // ------------------------------------------------------------------ list
  async function viewList() {
    $view.append(el("div", { class: "page-head" }, el("h1", { text: "Swing analyses" }),
      el("div", { class: "actions" }, el("a", { class: "button", href: "#/swings/new", text: "New swing" }))));
    const body = el("div");
    $view.append(body);
    const render = items => {
      if (!items.length) {
        body.replaceChildren(section("No analyses yet", null, el("div", { class: "empty" },
          el("p", { text: "Upload a slow-motion clip of one swing to get started." }),
          el("a", { class: "button", href: "#/swings/new", text: "Upload a swing video" }))));
        return;
      }
      const heads = [["Clip"], ["Uploaded"], ["Status"], ["Frame rate", "num"], ["Camera"], ["Est. bat speed", "num"], ["Warnings", "num"]];
      const tb = el("tbody");
      for (const m of items) {
        const v = m.summary?.predictions?.bat_speed_mph;
        tb.append(el("tr", {},
          el("td", {}, el("a", { href: `#/s/${m.id}`, text: m.options?.label || m.filename })),
          el("td", { class: "muted small", text: fmtDate(m.created_utc) }),
          el("td", {}, statusCell(m)),
          el("td", { class: "num", text: m.summary ? `${m.summary.fps} fps` : "–" }),
          el("td", { text: m.summary?.camera_view || "–" }),
          el("td", { class: "num", text: v ? `${v} mph` : "–" }),
          el("td", { class: "num", text: m.summary ? String(m.summary.warnings) : "–" })));
      }
      body.replaceChildren(el("section", { class: "block" }, stackTable(el("table", {},
        el("thead", {}, el("tr", {}, heads.map(([h, c]) => el("th", { class: c || null, text: h })))), tb))));
    };
    const load = async () => {
      const items = await api("/api/swings");
      render(items);
      return items.some(m => ["queued", "running", "uploading"].includes(m.status));
    };
    let active = await load();
    const timer = setInterval(async () => { if (active) active = await load().catch(() => true); }, 2000);
    onTeardown(() => clearInterval(timer));
  }

  // ------------------------------------------------------------------ new swing

  async function viewNew() {
    $view.append(el("div", { class: "page-head" }, el("h1", { text: "New swing analysis" })));
    const file = el("input", { type: "file", accept: "video/*,.mov,.mp4,.m4v,.avi,.mkv,.webm", required: true, hidden: true, id: "file" });
    const fileName = el("span", { class: "muted", text: "MP4, MOV, M4V, AVI, MKV or WebM" });
    const drop = el("label", { class: "dropzone", for: "file" }, el("strong", { text: "Choose a video or drop it here" }), fileName);
    ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", e => { if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; file.dispatchEvent(new Event("change")); } });
    file.addEventListener("change", () => {
      const f = file.files[0];
      fileName.textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : "MP4, MOV, M4V, AVI, MKV or WebM";
    });

    const num = (name, attrs = {}) => el("input", { type: "number", name, step: "any", ...attrs });
    const heightUnit = el("select", { name: "height_unit" }, el("option", { value: "in", text: "in" }), el("option", { value: "cm", text: "cm" }));
    const weightUnit = el("select", { name: "weight_unit" }, el("option", { value: "lb", text: "lb" }), el("option", { value: "kg", text: "kg" }));
    const status = el("div", { class: "small muted", role: "status" });
    const bar = el("div", { class: "progress", hidden: true }, el("span", { style: "width:0%" }));
    const submit = el("button", { type: "submit", text: "Upload and analyse" });

    const form = el("form", { class: "fields", novalidate: true },
      el("label", { class: "f", for: "file", text: "Video" }), drop, file,
      el("label", { class: "f", text: "Label" }), el("input", { name: "label", placeholder: "e.g. Cage 11/22 - front toss #4", maxlength: 120 }),
      el("div", { class: "sep", text: "Capture" }),
      el("label", { class: "f", text: "Batting side" }), radioGroup("bats", [["auto", "Detect"], ["R", "Right"], ["L", "Left"]], "auto"),
      el("label", { class: "f", text: "Camera position" }),
      radioGroup("alignment", [["square", "Square to the pitcher-plate line (side-on or straight behind)"], ["angled", "Angled / not sure"]], "square"),
      el("label", { class: "f", text: "Capture frame rate" }), el("div", { class: "row" }, num("capture_fps", { min: 15, max: 2000, placeholder: "from file" }), el("span", { class: "muted small", text: "fps" })),
      el("div", { class: "help", text: "Leave empty unless the clip is a slow-motion export that plays slowed down; then enter the rate it was filmed at (e.g. 240)." }),
      el("label", { class: "f", text: "Trim (seconds)" }), el("div", { class: "row" }, num("trim_start_s", { min: 0, placeholder: "start" }), "to", num("trim_end_s", { min: 0, placeholder: "end" })),
      el("div", { class: "help", text: "Optional. Keep about 1 s before the stride starts to 0.5 s after contact." }),
      el("label", { class: "f", text: "Pose model" }),
      el("select", { name: "model" }, el("option", { value: "full", text: "Full (default)" }),
        el("option", { value: "heavy", text: "Heavy - most accurate, about 2x slower" }), el("option", { value: "lite", text: "Lite - fastest, least accurate" })),
      el("div", { class: "sep", text: "Hitter (optional - height enables hand speed in mph; height and weight enable the bat-speed estimate)" }),
      el("label", { class: "f", text: "Height" }), el("div", { class: "row" }, num("height", { min: 40, max: 240 }), heightUnit),
      el("label", { class: "f", text: "Weight" }), el("div", { class: "row" }, num("weight", { min: 30, max: 440 }), weightUnit),
      el("label", { class: "f", text: "Bat speed (sensor)" }), el("div", { class: "row" }, num("bat_speed_mph", { min: 30, max: 110 }), el("span", { class: "muted small", text: "mph from a bat sensor (used to pick the comparison group)" })),
      el("label", { class: "f", text: "Level" }), el("select", { name: "level" }, LEVELS.map(([v, t]) => el("option", { value: v, text: t }))),
      el("div", {}), el("div", { class: "row" }, submit, bar, status));

    form.addEventListener("submit", async e => {
      e.preventDefault();
      if (!file.files.length) { status.textContent = "Choose a video first."; return; }
      const fd0 = new FormData(form);
      const fd = new FormData();
      fd.append("file", file.files[0]);
      for (const k of ["label", "bats", "alignment", "capture_fps", "trim_start_s", "trim_end_s", "model", "bat_speed_mph", "level"]) {
        const v = fd0.get(k);
        if (v !== null && v !== "") fd.append(k, v);
      }
      const h = parseFloat(fd0.get("height"));
      if (isFinite(h)) fd.append("height_cm", String(fd0.get("height_unit") === "in" ? h * 2.54 : h));
      const w = parseFloat(fd0.get("weight"));
      if (isFinite(w)) fd.append("weight_kg", String(fd0.get("weight_unit") === "lb" ? w * 0.45359237 : w));
      submit.disabled = true;
      bar.hidden = false;
      status.textContent = "Uploading…";
      try {
        const res = await uploadXHR(fd, f => { bar.firstChild.style.width = `${Math.round(f * 100)}%`; status.textContent = `Uploading… ${Math.round(f * 100)}%`; }, "/api/swings");
        location.hash = `#/s/${res.id}`;
      } catch (err) {
        status.textContent = err.message;
        submit.disabled = false;
        bar.hidden = true;
      }
    });

    const tips = el("aside", { class: "block tips" }, el("h2", { text: "Filming checklist", style: "margin-bottom:8px" }),
      el("ol", {},
        el("li", {}, el("strong", { text: "Slow motion: " }), "240 fps if the phone allows, 120 fps minimum. At 30/60 fps the hands blur through contact."),
        el("li", {}, el("strong", { text: "Tripod, no zoom or panning. " }), "Foot plant, stride and the forward move are measured against a still background."),
        el("li", {}, el("strong", { text: "Side-on from the open side: " }), "facing the hitter's chest from across the plate (first-base side for a right-handed hitter, third-base side for a left-hander), camera square to the pitcher-plate line, about waist height. Straight behind the catcher also works but loses stride length and the forward move."),
        el("li", {}, el("strong", { text: "Whole body in frame " }), "from the stance through the follow-through, one swing per clip."),
        el("li", {}, el("strong", { text: "Clear view: " }), "bright, even light; no netting or people between the camera and the hitter; avoid clothes that match the background.")));
    $view.append(el("div", { class: "form-grid" }, el("section", { class: "block" }, form), tips));
  }

  // ------------------------------------------------------------------ analysis
  async function viewAnalysis(id) {
    const root = el("div");
    $view.append(root);
    let data = await api(`/api/swings/${encodeURIComponent(id)}`);
    const terminal = s => s === "done" || s === "failed";
    if (!terminal(data.meta.status)) {
      renderProgress(root, data.meta);
      await new Promise(resolve => {
        const timer = setInterval(async () => {
          try {
            data = await api(`/api/swings/${encodeURIComponent(id)}`);
            if (terminal(data.meta.status)) { clearInterval(timer); resolve(); }
            else renderProgress(root, data.meta);
          } catch (e) { clearInterval(timer); root.replaceChildren(errorBox(e.message)); }
        }, 1000);
        onTeardown(() => { clearInterval(timer); resolve(); });
      });
      if (!document.body.contains(root)) return;
    }
    if (data.meta.status === "failed") return renderFailed(root, data.meta);
    renderDone(root, data.meta, data.result);
  }

  function headRow(meta, extra) {
    const del = el("button", { class: "danger", text: "Delete", onclick: async () => {
      if (!confirm("Delete this analysis and its files? This cannot be undone.")) return;
      try { await api(`/api/swings/${meta.id}`, { method: "DELETE" }); location.hash = "#/swings"; }
      catch (e) { alert(e.message); }
    } });
    return el("div", { class: "page-head" },
      el("h1", { text: meta.options?.label || meta.filename }),
      el("span", { class: "muted small", text: `${meta.filename} · uploaded ${fmtDate(meta.created_utc)}` }),
      el("div", { class: "actions" }, extra || null, meta.status === "running" || meta.status === "queued" ? null : del));
  }

  function renderProgress(root, meta) {
    root.replaceChildren(headRow(meta), section("Analysing", null,
      el("div", { class: "progress", style: "height:8px;max-width:520px" }, el("span", { style: `width:${Math.round((meta.progress || 0) * 100)}%` })),
      el("p", { class: "muted", style: "margin-top:8px", text: `${meta.stage || meta.status} · ${Math.round((meta.progress || 0) * 100)}%` }),
      el("p", { class: "small muted", text: "Pose estimation runs on the CPU, at roughly 20-40 frames per second for the Full model. You can leave this page; the analysis keeps running." })));
  }

  function renderFailed(root, meta) {
    root.replaceChildren(headRow(meta), errorBox(meta.error || "The analysis failed.",
      el("p", { class: "small", style: "margin-top:8px" }, "Check the filming checklist on the ", el("a", { href: "#/swings/new", text: "upload page" }),
        ": the whole body must be visible, one swing per clip, ideally 120-240 fps.")));
  }

  function renderDone(root, meta, r) {
    const destroyers = [];
    onTeardown(() => destroyers.forEach(d => d()));
    const id = meta.id;
    const fpsEff = r.video.fps_effective;
    const ctSample = r.events.contact.sample;
    const msOf = k => ((k - ctSample) / fpsEff) * 1000;

    // ---- video + events
    const video = el("video", { src: `/api/swings/${id}/overlay.mp4`, controls: true, playsInline: true, muted: true, preload: "auto" });
    const nFrames = r.overlay.frames;
    const ofps = r.overlay.fps;
    const current = () => Math.max(0, Math.min(nFrames - 1, Math.floor(video.currentTime * ofps + 1e-4)));
    const seek = k => { video.pause(); video.currentTime = (Math.max(0, Math.min(nFrames - 1, k)) + 0.5) / ofps; };
    const readout = el("span", { class: "readout" });
    const charts = [];
    const update = () => {
      const k = current();
      readout.textContent = `frame ${k + 1} / ${nFrames} · ${fmtNum(msOf(k))} ms from contact`;
      charts.forEach(c => c.setPlayhead(msOf(k)));
    };
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const loop = () => { update(); video._rvfc = video.requestVideoFrameCallback(loop); };
      video.requestVideoFrameCallback(loop);
    }
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    video.addEventListener("loadedmetadata", update);
    const speed = el("select", { "aria-label": "Playback speed", onchange: e => { video.playbackRate = parseFloat(e.target.value); } },
      ["1", "0.5", "0.25", "0.1"].map(v => el("option", { value: v, text: `${v}×` })));
    const controls = el("div", { class: "controls" },
      el("button", { class: "secondary", title: "Previous frame (,)", text: "◀ frame", onclick: () => seek(current() - 1) }),
      el("button", { class: "secondary", title: "Next frame (.)", text: "frame ▶", onclick: () => seek(current() + 1) }),
      speed, readout);
    const keyHandler = e => {
      if (e.target instanceof Element && e.target.matches("input, select, textarea")) return;
      if (e.key === ",") { seek(current() - 1); e.preventDefault(); }
      if (e.key === ".") { seek(current() + 1); e.preventDefault(); }
    };
    document.addEventListener("keydown", keyHandler);
    destroyers.push(() => document.removeEventListener("keydown", keyHandler));

    const edits = Object.fromEntries(EVENT_ORDER.map(k => [k, r.events[k] ? r.events[k].sample : null]));
    const evBody = el("tbody");
    const recompute = el("button", { text: "Recompute with these events", disabled: true });
    const evStatus = el("span", { class: "small muted", role: "status" });
    const renderEvents = () => {
      evBody.replaceChildren();
      for (const k of EVENT_ORDER) {
        const orig = r.events[k];
        const v = edits[k];
        const changed = (orig ? orig.sample : null) !== v;
        evBody.append(el("tr", { class: changed ? "edited" : null },
          el("td", {}, EVENT_NAMES[k]),
          el("td", { class: "num", text: v === null ? "–" : String(v + 1) }),
          el("td", { class: "num", text: v === null ? "–" : `${fmtNum(msOf(v))} ms` }),
          el("td", { class: "small muted", text: changed ? "edited" : (orig ? orig.source : "missing") }),
          el("td", { class: "actions-cell" }, el("button", { class: "ghost", text: "Go to", disabled: v === null, onclick: () => seek(v) }),
            el("button", { class: "ghost", text: "Set to current frame", onclick: () => { edits[k] = current(); renderEvents(); } }))));
      }
      if (evBody.parentElement) stackTable(evBody.parentElement, 0, [4]);
      const dirty = EVENT_ORDER.some(k => (r.events[k] ? r.events[k].sample : null) !== edits[k]);
      recompute.disabled = !dirty;
    };
    recompute.addEventListener("click", async () => {
      recompute.disabled = true;
      evStatus.textContent = "Recomputing…";
      const body = {};
      for (const k of EVENT_ORDER) if ((r.events[k] ? r.events[k].sample : null) !== edits[k]) body[k] = edits[k];
      try {
        const out = await api(`/api/swings/${id}/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        destroyers.forEach(d => d());
        destroyers.length = 0;
        renderDone(root, out.meta, out.result);
      } catch (e) {
        evStatus.textContent = e.message;
        recompute.disabled = false;
      }
    });
    renderEvents();

    const eventsBlock = el("div", {},
      stackTable(el("table", { class: "events-table" }, el("thead", {}, el("tr", {}, ["Event", "Frame", "From contact", "Source", ""].map(h => el("th", { text: h })))), evBody), 0, [4]),
      el("div", { class: "row", style: "display:flex;gap:10px;align-items:center;margin-top:8px" }, recompute,
        el("button", { class: "secondary", text: "Reset", onclick: () => { for (const k of EVENT_ORDER) edits[k] = r.events[k] ? r.events[k].sample : null; renderEvents(); } }),
        evStatus),
      el("p", { class: "small muted", style: "margin-top:6px", text: "If an event is off, step to the right frame (keys , and .) and use “Set to current frame”. Metrics recompute in a few seconds without re-running pose estimation." }));

    // ---- tiles
    const tiles = el("div", { class: "tiles" });
    for (const p of r.predictions || []) {
      const shown = p.reportable;
      tiles.append(el("div", { class: "tile" },
        el("div", { class: "label", text: `Estimated ${p.label.toLowerCase()}` }),
        el("div", { class: "value", text: shown ? `${fmtNum(p.value, p.decimals)} ${p.unit}` : "Not shown" }),
        el("div", { class: "sub", text: shown
          ? `±${fmtNum(p.cv_mae, p.decimals)} ${p.unit} typical error on held-out hitters (R² ${p.cv_r2.toFixed(2)})`
          : `Model R² ${p.cv_r2.toFixed(2)} on held-out hitters is too low.${p.variant === "kinematics" ? " Add height and weight." : ""}` })));
    }
    const fpct = r.metrics.find(m => m.key === "fp_to_contact_time");
    tiles.append(el("div", { class: "tile" }, el("div", { class: "label", text: "Foot plant to contact" }),
      el("div", { class: "value", text: fpct?.value != null ? `${fmtNum(fpct.value)} ms` : "–" }),
      el("div", { class: "sub", text: fpct?.typical ? `Driveline hitters: ${fmtNum(fpct.typical["10"])}-${fmtNum(fpct.typical["90"])} ms (10th-90th pct)` : "" })));
    tiles.append(el("div", { class: "tile" }, el("div", { class: "label", text: "Video" }),
      el("div", { class: "value", text: `${fmtNum(fpsEff)} fps` }),
      el("div", { class: "sub", text: `${r.quality.camera_view} view · ${sideName(r.athlete.bats)}-handed hitter · ${r.video.width}×${r.video.height}` })));

    const warnings = r.quality.warnings || [];
    const warnBlock = warnings.length ? el("div", { style: "margin-top:4px" }, el("h3", { text: `Data-quality warnings (${warnings.length})`, style: "margin-bottom:4px" }),
      el("ul", { class: "warnings" }, warnings.map(w => el("li", { text: w })))) : el("p", { class: "small muted", text: "No data-quality warnings." });

    // ---- key frames
    const kf = el("div", { class: "keyframes" }, EVENT_ORDER.map(k => {
      const ev = r.events[k];
      const file = r.keyframes?.[k];
      if (!ev || !file) return el("figure", {}, el("div", { class: "empty", text: `${EVENT_NAMES[k]}: ${k === "lift" ? "no leg lift found" : "not in clip"}` }));
      return el("figure", {},
        el("img", { src: `/api/swings/${id}/keyframes/${file}?v=${encodeURIComponent(r.computed_utc)}`, alt: `${EVENT_NAMES[k]} frame`, loading: "lazy", onclick: () => seek(ev.sample) }),
        el("figcaption", {}, el("strong", { text: EVENT_NAMES[k] }), el("span", { class: "muted", text: k === "contact" ? `frame ${ev.sample + 1}` : `${fmtNum(ev.ms_from_contact)} ms` })));
    }));

    // ---- metrics table
    const band = r.comparison_band;
    const refN = r.reference_group?.n_swings ? `${r.reference_group.n_swings} swings from ${r.reference_group.n_hitters} Driveline hitters` : "Driveline hitters";
    const metricsBlock = section("Mechanics vs. Driveline hitters",
      band ? `Bat-speed band from ${fmtNum(band.bat_speed_mph, 1)} mph (${band.source}) · percentiles use all ${refN}` : `Percentiles against ${refN} (simulated phone-video conditions)`,
      metricsTable(r.metrics, band));

    // ---- charts
    const s = r.series;
    const events = EVENT_ORDER.filter(k => r.events[k]).map(k => ({ label: EVENT_SHORT[k], x: r.events[k].ms_from_contact }));
    const pick = i => seek(s.sample[i]);
    const seq = lineChart({ title: "Kinematic sequence", x: s.t_ms, xUnit: "ms", xLabel: "ms from contact", yUnit: "deg/s", height: 280, peakLabels: true, events, onPick: pick,
      series: [{ name: "Pelvis rotation", values: s.pelvis_rot_velo, color: "var(--s1)" },
               { name: "Shoulder rotation", values: s.torso_rot_velo, color: "var(--s2)" },
               { name: "Lead upper arm", values: s.lead_arm_speed, color: "var(--s3)" }] });
    charts.push(seq);
    const handMph = Array.isArray(s.hand_speed_mph);
    const hand = lineChart({ title: "Hand speed", x: s.t_ms, xUnit: "ms", xLabel: "ms from contact", yUnit: handMph ? "mph" : "leg lengths/s", height: 200, peakLabels: true, events, onPick: pick,
      series: [{ name: "Hands, relative to the pelvis", values: handMph ? s.hand_speed_mph : s.hand_speed, color: "var(--s1)" }], decimals: 1 });
    charts.push(hand);
    const multiples = el("div", { class: "multiples" });
    for (const [key, title] of [["hip_shoulder_sep", "Hip-shoulder separation"], ["lead_knee_flexion", "Lead knee flexion"],
                                ["trunk_lateral", "Trunk side bend (toward rear leg)"], ["rear_elbow_flexion", "Rear elbow flexion"]]) {
      const c = lineChart({ title, x: s.t_ms, xUnit: "ms", yUnit: "deg", height: 180, events, onPick: pick,
        series: [{ name: title, values: s[key], color: "var(--s1)" }], decimals: 0 });
      charts.push(c);
      multiples.append(el("div", {}, el("h3", { text: title }), c.node));
    }
    destroyers.push(() => charts.forEach(c => c.destroy()));
    const csv = el("button", { class: "secondary", text: "Download time series (CSV)", onclick: () => downloadCsv(s, r, meta) });

    root.replaceChildren(
      headRow(meta),
      el("div", { class: "top-grid" },
        el("section", { class: "block player" }, video, controls, el("div", { style: "margin-top:14px" }, eventsBlock)),
        el("section", { class: "block" }, tiles, warnBlock)),
      section("Key positions", "click a frame to jump the video there", kf),
      metricsBlock,
      section("Kinematic sequence", "click the chart to move the video", seq.node,
        el("p", { class: "small muted", style: "margin-top:6px", text: "Peaks should go pelvis → shoulders → lead arm. Shoulder rotation speed is measured on the shoulder joints, which ride on the shoulder blades, so it runs faster than the trunk itself; read it for timing, not magnitude." })),
      section("Hand speed", handMph ? "converted with the height you entered" : "enter height for mph", hand.node,
        el("p", { class: "small muted", style: "margin-top:6px", text: "The hands peak before contact and slow down as the bat whips through; with a side-on camera the analysis times contact from that peak as the camera sees it." })),
      section("Joint angles", null, multiples, el("div", { style: "margin-top:10px" }, csv)),
      predictionsBlock(r.predictions || []),
      coachingBlock(id, r),
      qualityDetails(r));
    update();
  }

  function metricsTable(metrics, band) {
    const tb = el("tbody");
    for (const g of GROUPS) {
      tb.append(el("tr", { class: "group-row" }, el("td", { colspan: band ? 7 : 6, text: g })));
      for (const m of metrics.filter(x => x.group === g)) {
        const na = m.value === null;
        const t = m.typical;
        const label = el("td", {}, el("span", { text: m.label }), el("span", { class: "desc", text: na ? m.unavailable || "Not measured." : m.description }));
        const valTitle = m.validity_r != null && !na ? `This app's definition correlates r = ${m.validity_r.toFixed(2)} with Driveline's lab value for the same instant (validation page).` : null;
        const cells = [
          label,
          el("td", { class: "num", title: valTitle }, el("strong", { text: na ? "–" : fmtVal(m.value, m.unit, m.decimals) }),
            !na && m.error_margin ? el("span", { class: "desc", text: `±${fmtNum(m.error_margin, m.decimals)}` }) : null),
          el("td", { class: "num", text: t ? fmtRange(t["25"], t["75"], m.unit, m.decimals) : "–" }),
          el("td", {}, t ? bulletStrip(t, na ? null : m.value, `10th-90th pct ${fmtVal(t["10"], m.unit, m.decimals)} to ${fmtVal(t["90"], m.unit, m.decimals)}; median ${fmtVal(t["50"], m.unit, m.decimals)}`) : null),
          el("td", { class: "num", text: m.percentile === null ? "–" : ordinal(m.percentile) }),
        ];
        if (band) cells.push(el("td", { class: "num", text: m.band ? fmtVal(m.band["50"], m.unit, m.decimals) : "–" }));
        cells.push(el("td", {}, na ? null : badge(m.rating)));
        tb.append(el("tr", { class: na ? "na" : null }, cells));
      }
    }
    const heads = ["Metric", "You", "Typical (25th-75th)", "Distribution", "Percentile"];
    if (band) heads.push(`Median, ${band.band ? bandLabel(band.band) : "band"}`);
    heads.push("Confidence");
    return el("div", { class: "table-wrap" }, stackTable(el("table", { class: "metrics-table" },
      el("thead", {}, el("tr", {}, heads.map((h, i) => el("th", { class: i === 1 || i === 2 || i === 4 || (band && i === 5) ? "num" : null, text: h })))), tb), 0, [3]),
      el("p", { class: "small muted", style: "margin-top:8px" },
        "Distribution: light bar = 10th-90th percentile, dark bar = 25th-75th, tick = median, dot = this swing. ",
        "± is the typical measurement error in simulation. Confidence combines how well the definition matches Driveline's lab metric (where Driveline has one at the same instant) and how well it survives video noise at this frame rate; real footage is usually noisier than the simulation."));
  }

  function bandLabel(b) {
    const [lo, hi] = b.split("-");
    return lo === "0" ? `<${hi} mph` : hi === "200" ? `${lo}+ mph` : `${lo}-${hi} mph`;
  }

  function predictionsBlock(preds) {
    if (!preds.length) return section("Model estimates", null, el("p", { class: "muted", text: "No model bundle was found." }));
    const body = el("div", { class: "findings two" });
    for (const p of preds) {
      const col = el("div", {}, el("h3", { text: p.label }));
      if (!p.reportable) {
        col.append(el("p", { class: "muted", text: `Not shown: on held-out Driveline hitters this model explains only ${Math.max(0, p.cv_r2 * 100).toFixed(0)}% of the variance (R² ${p.cv_r2.toFixed(2)}; typical error ${fmtNum(p.cv_mae, p.decimals)} ${p.unit} vs ${fmtNum(p.baseline_mae, p.decimals)} for guessing the average).${p.variant === "kinematics" ? " Adding height and weight may make it usable." : ""}` }));
      } else {
        col.append(el("p", {}, el("strong", { text: `${fmtNum(p.value, p.decimals)} ${p.unit}` }),
          ` ± ${fmtNum(p.cv_mae, p.decimals)} ${p.unit} typical error on held-out hitters. For comparison: guessing the average ±${fmtNum(p.baseline_mae, p.decimals)}` +
          (p.variant === "kinematics+body" ? `, height and weight alone ±${fmtNum(p.body_only_mae, p.decimals)}.` : ".")));
        if (p.variant === "kinematics+body" && p.body_only_mae - p.cv_mae < 0.1 * p.body_only_mae) {
          col.append(el("p", { class: "small muted", text: "Most of this estimate comes from body size; the mechanics add little on top of height and weight." }));
        }
        if (p.what_if.length) {
          col.append(el("p", { class: "small muted", text: "What moves the estimate: change in the prediction if each metric were at the Driveline median, one at a time. Right = this metric is currently costing you; left = it is helping. Model associations, not proven causes." }));
          col.append(divergingBars(p.what_if.map(w => ({ label: w.label, delta: w.delta, title: `This swing ${fmtNum(w.current, 1)}, reference median ${fmtNum(w.reference_median, 1)} (raw units)` })), p.unit, p.decimals));
        }
        if (p.missing_features.length) col.append(el("p", { class: "small muted", style: "margin-top:6px", text: `${p.missing_features.length} input metric(s) unavailable for this clip; the model fills them in.` }));
      }
      body.append(col);
    }
    return section("Model estimates", "trained on Driveline data, athlete-grouped cross-validation", body);
  }

  function coachingBlock(id, r) {
    const fs = r.findings || [];
    const col = (title, kind, empty) => {
      const items = fs.filter(f => f.kind === kind);
      return el("div", {}, el("h3", { text: title }), items.length ? items.map(f => el("div", { class: "finding" },
        el("div", { class: "t", text: f.title }), el("div", { class: "small", text: f.detail }), f.cue ? el("div", { class: "cue", text: `Cue: ${f.cue}` }) : null))
        : el("p", { class: "small muted", text: empty }));
    };
    const reportBox = el("div");
    const tabs = el("div", { class: "report-tabs" });
    const health = getHealth();
    const llmOk = health?.llm?.configured;
    const gen = el("button", { text: r.llm_report ? "Rewrite with Claude" : "Write report with Claude", disabled: !llmOk,
                               title: llmOk ? `Sends the metrics (no video) to ${health.llm.model}` : "Set ANTHROPIC_API_KEY in .env and restart the server" });
    const genStatus = el("span", { class: "small muted", role: "status", text: llmOk ? "Sends the numbers on this page (not the video) to the Claude API; billed to your API key." : "Claude reports are off: set ANTHROPIC_API_KEY in .env and restart the server." });
    let mode = r.llm_report ? "claude" : "rules";
    const show = () => {
      tabs.replaceChildren(
        el("button", { class: `ghost ${mode === "rules" ? "active" : ""}`, text: "Rule-based summary", onclick: () => { mode = "rules"; show(); } }),
        el("button", { class: `ghost ${mode === "claude" ? "active" : ""}`, text: "Claude report", disabled: !r.llm_report, onclick: () => { mode = "claude"; show(); } }));
      if (mode === "claude" && r.llm_report) {
        reportBox.replaceChildren(renderMarkdown(r.llm_report.text), el("p", { class: "small muted",
          text: `Written by ${r.llm_report.model} on ${fmtDate(r.llm_report.generated_utc)}${r.llm_report.stale ? " - before the events were edited; rewrite it to use the new numbers" : ""}${r.llm_report.truncated ? " (cut off at the length limit)" : ""}.` }));
      } else {
        reportBox.replaceChildren(renderMarkdown(r.report?.text || ""));
      }
    };
    gen.addEventListener("click", async () => {
      gen.disabled = true;
      genStatus.textContent = "Writing the report… usually 20-60 s.";
      try {
        r.llm_report = await api(`/api/swings/${id}/report`, { method: "POST" });
        mode = "claude";
        gen.textContent = "Rewrite with Claude";
        genStatus.textContent = `Done - ${r.llm_report.usage.input_tokens} input / ${r.llm_report.usage.output_tokens} output tokens.`;
        show();
      } catch (e) {
        genStatus.textContent = e.message;
      } finally {
        gen.disabled = !llmOk;
      }
    });
    show();
    return section("Coaching", null,
      el("div", { class: "findings" }, col("Priorities", "priority", "Nothing reliably measured falls in the bottom quarter."),
        col("Strengths", "strength", "Nothing stands out above the 75th percentile."), col("Notes", "note", "No other notes.")),
      el("div", { style: "margin-top:16px;border-top:1px solid var(--grid);padding-top:12px" }, tabs, reportBox,
        el("div", { style: "display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap" }, gen, genStatus)));
  }

  function qualityDetails(r) {
    const q = r.quality;
    const pct = v => `${Math.round(v * 100)}%`;
    const rows = [
      ["Frames analysed", `${q.frames} at ${fmtNum(q.fps_effective)} fps (container ${fmtNum(r.video.fps, 2)} fps${r.video.capture_fps_override ? `, capture override ${r.video.capture_fps_override}` : ""})`],
      ["Pose found", `${pct(q.pose_detected_fraction)} of frames, ${pct(q.pose_detected_fraction_window)} around the swing`],
      ["Key-joint visibility around the swing", pct(q.key_joint_visibility_window)],
      ["Left/right swaps repaired", `legs ${q.swaps_repaired.legs}, arms ${q.swaps_repaired.arms}`],
      ["Hand frames bridged around the swing", q.hands_untracked_frames_window === undefined ? "–" : `${q.hands_untracked_frames_window} (blurred or untracked; hand metrics there are not reported)`],
      ["Camera view", `${q.camera_view}${q.heading_known ? ", squared to the pitcher" : ", angled"}`],
      ["Batting side", `${sideName(r.athlete.bats)}-handed${r.athlete.bats_detected_in_video !== r.athlete.bats ? ` (body rotation in the video suggests ${sideName(r.athlete.bats_detected_in_video)})` : ""}; hips + shoulders turned ${fmtNum(q.batting_side_turn_deg)}° through the swing`],
      ["Processing time", r.processing_s ? `${r.processing_s} s` : "–"],
      ["Analysis version", `${r.analysis_version}, computed ${fmtDate(r.computed_utc)}`],
    ];
    return el("section", { class: "block" }, el("details", {}, el("summary", { text: "Data quality and method" }),
      el("dl", { class: "kv", style: "margin-top:10px" }, rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])),
      el("p", { class: "small muted", style: "margin-top:10px" }, "Joint positions come from MediaPipe Pose (single camera, 3D estimate). Events, angles and the reference comparison are computed with the same code that was run on Driveline's lab data after simulating phone-video noise and frame rates. ",
        el("a", { href: "#/swings/reference", text: "Validation details" }), ".")));
  }

  function downloadCsv(s, r, meta) {
    const keys = ["sample", "t_ms", ...Object.keys(s).filter(k => k !== "sample" && k !== "t_ms")];
    const lines = [keys.join(",")];
    for (let i = 0; i < s.t_ms.length; i++) lines.push(keys.map(k => s[k][i] ?? "").join(","));
    lines.push("");
    lines.push("event,sample,ms_from_contact,source");
    for (const k of EVENT_ORDER) { const v = r.events[k]; if (v) lines.push([k, v.sample, v.ms_from_contact.toFixed(1), v.source].join(",")); }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `${(meta.options?.label || meta.filename).replace(/[^\w.-]+/g, "_")}_series.csv` });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  // ------------------------------------------------------------------ reference
  async function viewReference() {
    const ref = await api("/api/swings/reference");
    const src = ref.source;
    $view.append(el("div", { class: "page-head" }, el("h1", { text: "Hitting reference data and validation" }),
      el("span", { class: "muted small", text: `built ${fmtDate(ref.built_utc)} · analysis version ${ref.version}` }),
      el("div", { class: "actions" }, el("a", { class: "button secondary", href: "#/reference", text: "Pitching reference" }))));

    $view.append(section("Source", null,
      el("p", {}, `${src.name}: ${src.n_swings} swings from ${src.n_athletes} hitters (`,
        Object.entries(src.levels).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", "),
        `; ${Object.entries(src.bats || {}).map(([k, v]) => `${v} ${sideName(k)}-handed`).join(", ")}). Bat speed at contact ${fmtNum(src.bat_speed_mph.min, 1)}-${fmtNum(src.bat_speed_mph.max, 1)} mph (mean ${fmtNum(src.bat_speed_mph.mean, 1)}), exit velocity mean ${fmtNum(src.exit_velo_mph.mean, 1)} mph, pitching machine at ${src.pitch}. Marker-based motion capture at 360 Hz with force plates.`),
      el("p", { class: "small muted" }, `Licence: ${src.license}. `, el("a", { href: src.url, target: "_blank", rel: "noopener", text: "Dataset repository" })),
      el("p", { class: "small" }, "How the comparison is built: this app's metric definitions are run on Driveline's joint centres. The data is resampled to 240, 120 and 60 fps, given simulated pose-estimation noise, seen through a randomly placed phone camera and smoothed with the app's own filter. Then the app's own event detectors are applied, so the reference distributions carry the same processing as your clip.")));

    const metrics = ref.metrics;
    const tb = el("tbody");
    for (const m of metrics) {
      const v = ref.validity[m.key];
      const rel = k => ref.reliability[k][m.key]?.r;
      const f = x => (x === null || x === undefined ? "–" : x.toFixed(2));
      tb.append(el("tr", {},
        el("td", {}, m.label, el("span", { class: "desc", text: v ? `Driveline: ${v.driveline_column}` : "no Driveline equivalent at the same instant" })),
        el("td", { class: "num", text: f(v?.r) }),
        el("td", { class: "num", text: f(rel("240")) }), el("td", { class: "num", text: f(rel("120")) }), el("td", { class: "num", text: f(rel("60")) }),
        el("td", {}, badge(ref.ratings["240"][m.key])), el("td", {}, badge(ref.ratings["120"][m.key])),
        el("td", { class: "num small muted", text: v ? `${fmtNum(v.mean_this, m.decimals)} vs ${fmtNum(v.mean_driveline, m.decimals)}` : "–" })));
    }
    $view.append(section("Metric validation", "r = Pearson correlation",
      el("div", { class: "table-wrap" }, stackTable(el("table", {}, el("thead", {}, el("tr", {},
        ["Metric", "Validity r", "Reliability r 240", "Reliability r 120", "Reliability r 60", "Confidence 240 fps", "Confidence 120 fps", "Mean: this app vs Driveline"].map((h, i) => el("th", { class: (i >= 1 && i <= 4) || i === 7 ? "num" : null, text: h })))), tb))),
      el("p", { class: "small muted", style: "margin-top:8px", text: `Validity: this app's definition, computed on clean lab data at the true events, against Driveline's published value where Driveline measures the same thing at the same instant. Driveline's foot-plant (\"launch position\") values are taken at an instant that matches neither the force-plate foot plant in the same dataset nor this app's foot-down-flat definition, so foot-plant metrics have no validity check and are rated on reliability alone. Reliability: the same metric after simulated video at that frame rate, with the app's own event detection, against the clean value. Ratings are for a side-on camera. Values are always shown as measured, never rescaled onto Driveline's scale, so the tables, charts and coaching text agree; the last column shows where the two scales differ (for example, the shoulder line turns faster and further than Driveline's thorax segment).` })));

    const ee = ref.event_error;
    const etb = el("tbody");
    for (const k of EVENT_ORDER) {
      etb.append(el("tr", {}, el("td", { text: EVENT_NAMES[k] }),
        ...["240", "120", "60"].flatMap(f => [el("td", { class: "num", text: `${fmtNum(ee[f][k].mae_ms, 1)} ms` }), el("td", { class: "num", text: `${fmtNum(ee[f][k].p90_abs_ms, 1)} ms` })])));
    }
    $view.append(section("Event detection error (simulated video vs the true events)", `${src.n_with_valid_events} swings with usable events`,
      stackTable(el("table", {}, el("thead", {}, el("tr", {}, ["Event", "240 fps median", "240 fps p90", "120 fps median", "120 fps p90", "60 fps median", "60 fps p90"].map((h, i) => el("th", { class: i ? "num" : null, text: h })))), etb)),
      el("p", { class: "small muted", style: "margin-top:8px", text: "Contact is compared with Driveline's contact event; foot plant (front foot down flat), leg lift and load with the same definitions applied to the clean lab data. Side-on camera; from behind, contact is timed from the 3D hands and shoulders instead of the image." })));

    const mtb = el("tbody");
    for (const [k, v] of Object.entries(ref.models)) {
      const [target, variant] = k.split("|");
      mtb.append(el("tr", {}, el("td", { text: TARGET_NAMES[target] || target }),
        el("td", { text: variant === "kinematics" ? "Video metrics only" : "Video metrics + height/weight" }),
        el("td", { text: v.algorithm.replace("_", " ") }), el("td", { class: "num", text: fmtNum(v.mae, 2) }), el("td", { class: "num", text: v.r2.toFixed(2) }),
        el("td", { class: "num", text: fmtNum(v.baseline_mean_mae, 2) }), el("td", { class: "num", text: fmtNum(v.body_only_mae, 2) }),
        el("td", { text: v.reportable ? "shown" : "hidden (R² < 0.25)" })));
    }
    $view.append(section("Model accuracy", "5-fold cross-validation grouped by athlete, scored on simulated 240 fps clips",
      stackTable(el("table", {}, el("thead", {}, el("tr", {}, ["Target", "Inputs", "Algorithm", "MAE", "R²", "MAE guessing mean", "MAE height/weight only", "In app"].map((h, i) => el("th", { class: i >= 3 && i <= 6 ? "num" : null, text: h })))), mtb))));
  }
  return { list: viewList, create: viewNew, detail: viewAnalysis, reference: viewReference };
}
