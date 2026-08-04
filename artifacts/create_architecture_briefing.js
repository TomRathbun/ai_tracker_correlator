const pptxgen = require("pptxgenjs");
const path = require("path");

const ART = path.join(__dirname);
const OUT = path.join(ART, "Software_Architecture.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Tom Rathbun";
pres.title = "Software Architecture — AI Tracker Correlator";
pres.subject = "System architecture, components, and design trade-offs";

const C = {
  bg: "0B1220",
  card: "172033",
  soft: "1E293B",
  ice: "A8C5E2",
  white: "FFFFFF",
  muted: "94A3B8",
  accent: "38BDF8",
  success: "34D399",
  warn: "F59E0B",
  danger: "F87171",
  line: "334155",
  purple: "A78BFA",
};

function base(title, eyebrow = "SOFTWARE ARCHITECTURE  ·  AI TRACKER CORRELATOR") {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText(eyebrow, {
    x: 0.45, y: 0.2, w: 9.1, h: 0.24,
    fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, margin: 0, charSpacing: 1.2,
  });
  s.addText(title, {
    x: 0.45, y: 0.44, w: 9.1, h: 0.45,
    fontSize: 24, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  return s;
}

function card(s, x, y, w, h, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.card },
    line: { color: opts.line || C.line, width: 1 },
    rectRadius: 0.08,
  });
  if (opts.top) {
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.07,
      fill: { color: opts.top }, line: { color: opts.top },
    });
  }
}

function pill(s, x, y, w, h, text, fill, textColor) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: fill },
    rectRadius: 0.06,
  });
  s.addText(text, {
    x, y, w, h,
    fontSize: 11, fontFace: "Calibri", color: textColor || C.bg,
    bold: true, align: "center", valign: "middle", margin: 0,
  });
}

// ── 1. Title ──────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("SYSTEM DESIGN OVERVIEW", {
    x: 0.55, y: 1.35, w: 9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0, charSpacing: 2,
  });
  s.addText("Software Architecture\n& Components", {
    x: 0.55, y: 1.8, w: 9, h: 1.15,
    fontSize: 36, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  s.addText(
    "How multi-sensor radar plots become correlated aircraft tracks —\npipeline, hybrid ML+physics core, GNN variants, and platform tooling.",
    {
      x: 0.55, y: 3.15, w: 8.8, h: 0.75,
      fontSize: 16, fontFace: "Calibri", color: C.ice, margin: 0,
    }
  );
  s.addText("AI Tracker Correlator  ·  Capstone architecture briefing", {
    x: 0.55, y: 4.65, w: 9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0,
  });
}

// ── 2. Agenda ─────────────────────────────────────────────
{
  const s = base("Agenda");
  const items = [
    { n: "01", t: "Problem & design goal", d: "Why unify tracker + correlator into one system" },
    { n: "02", t: "System context", d: "Inputs, outputs, and the modular research platform" },
    { n: "03", t: "End-to-end pipeline", d: "Windowing → clutter → updater → track manager" },
    { n: "04", t: "Hybrid core", d: "Pairwise ML association + continuous-time Kalman" },
    { n: "05", t: "GNN lineage", d: "V3–V6 design choices and trade-offs" },
    { n: "06", t: "Components & tooling", d: "Code map, data layer, CLI, dashboard, MLflow" },
  ];
  items.forEach((it, i) => {
    const col = i < 3 ? 0 : 1;
    const row = i % 3;
    const x = 0.45 + col * 4.7;
    const y = 1.15 + row * 1.3;
    card(s, x, y, 4.45, 1.15, { top: C.accent });
    s.addText(it.n, {
      x: x + 0.2, y: y + 0.25, w: 0.7, h: 0.35,
      fontSize: 18, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
    });
    s.addText(it.t, {
      x: x + 1.0, y: y + 0.22, w: 3.2, h: 0.35,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(it.d, {
      x: x + 1.0, y: y + 0.6, w: 3.2, h: 0.4,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 3. Problem ────────────────────────────────────────────
{
  const s = base("The architectural problem");
  card(s, 0.45, 1.1, 4.4, 4.0, { top: C.danger });
  s.addText("TRADITIONAL STACK", {
    x: 0.65, y: 1.3, w: 4.0, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.danger, bold: true, margin: 0, charSpacing: 1,
  });
  const trad = [
    "One physics tracker per radar",
    "Separate track-to-track correlator",
    "Duplicate tracks across sensors",
    "Latency & maintenance overhead",
    "Hard to share features across stages",
  ];
  trad.forEach((t, i) => {
    s.addText("▸  " + t, {
      x: 0.7, y: 1.8 + i * 0.5, w: 3.9, h: 0.4,
      fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });

  card(s, 5.1, 1.1, 4.4, 4.0, { top: C.success });
  s.addText("UNIFIED AI TRACKER", {
    x: 5.3, y: 1.3, w: 4.0, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.success, bold: true, margin: 0, charSpacing: 1,
  });
  const uni = [
    "Single multi-sensor tracker",
    "Clutter, association & fusion in-line",
    "One correlated track picture out",
    "Hybrid: ML match + physics motion",
    "Configurable GNN / Kalman modes",
  ];
  uni.forEach((t, i) => {
    s.addText("▸  " + t, {
      x: 5.35, y: 1.8 + i * 0.5, w: 3.9, h: 0.4,
      fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });
}

// ── 4. System context ─────────────────────────────────────
{
  const s = base("System context");
  const boxes = [
    { x: 0.45, title: "INPUTS", items: ["Decoded CAT 048 / 062 plots", "Multi-radar JSONL streams", "PSR + SSR heterogeneity", "Canonical Sweden / UAE sets"], top: C.accent },
    { x: 3.5, title: "CORE ENGINE", items: ["Pipeline orchestrator", "Clutter filter (MLP)", "Hybrid or GNN updater", "Track lifecycle manager"], top: C.warn },
    { x: 6.55, title: "OUTPUTS", items: ["Correlated active tracks", "MOTA / MOTP metrics", "MLflow experiment runs", "Dashboard & video viz"], top: C.success },
  ];
  boxes.forEach((b) => {
    card(s, b.x, 1.15, 2.9, 3.9, { top: b.top });
    s.addText(b.title, {
      x: b.x + 0.15, y: 1.4, w: 2.6, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: b.top, bold: true, margin: 0, charSpacing: 1,
    });
    b.items.forEach((it, i) => {
      s.addText(it, {
        x: b.x + 0.2, y: 2.0 + i * 0.65, w: 2.5, h: 0.55,
        fontSize: 13, fontFace: "Calibri", color: C.white, margin: 0,
      });
    });
  });
}

// ── 5. High-level architecture ────────────────────────────
{
  const s = base("High-level architecture");
  const stages = [
    { t: "Stream\nIngest", d: "JSONL windows", c: C.accent },
    { t: "Clutter\nFilter", d: "MLP gate", c: C.purple },
    { t: "State\nUpdater", d: "Hybrid / GNN", c: C.warn },
    { t: "Track\nManager", d: "M/N lifecycle", c: C.success },
    { t: "Tracks\nOut", d: "Correlated", c: C.ice },
  ];
  stages.forEach((st, i) => {
    const x = 0.4 + i * 1.9;
    card(s, x, 1.4, 1.7, 1.7, { top: st.c });
    s.addText(st.t, {
      x: x + 0.08, y: 1.7, w: 1.54, h: 0.75,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, align: "center", margin: 0,
    });
    s.addText(st.d, {
      x: x + 0.08, y: 2.55, w: 1.54, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center", margin: 0,
    });
    if (i < stages.length - 1) {
      s.addText("→", {
        x: x + 1.55, y: 1.95, w: 0.4, h: 0.4,
        fontSize: 20, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
      });
    }
  });

  card(s, 0.45, 3.4, 9.1, 1.8);
  s.addText("Design principle", {
    x: 0.7, y: 3.55, w: 8.6, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText(
    "One modular pipeline (src/pipeline.py) owns the control flow. The state updater is a swappable strategy: Hybrid (NewHybridUpdater), GNN (GNNUpdater with V3–V6 models), or Kalman-only fallback. Clutter filtering and track management stay shared so ablations compare fair.",
    {
      x: 0.7, y: 3.95, w: 8.6, h: 1.0,
      fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
    }
  );
}

// ── 6. Pipeline detail ────────────────────────────────────
{
  const s = base("End-to-end pipeline (windowed stream)");
  const rows = [
    { n: "1", t: "Windowing", d: "Slice multi-radar stream into time chunks (e.g. 1–2 s). Asynchronous scans (≈5.5–9 s) arrive together in each window." },
    { n: "2", t: "Clutter filter", d: "ClutterClassifier MLP scores each plot; high-probability false alarms are dropped before association." },
    { n: "3", t: "Sensor routing", d: "Optional PSR/SSR branching via SensorRouter for sensor-aware features and gates." },
    { n: "4", t: "State update", d: "Hybrid or GNN associates measurements to tracks and updates kinematic state." },
    { n: "5", t: "Track management", d: "Tentative → confirmed (min_hits); coast through shadows (max_age); delete stale tracks." },
  ];
  rows.forEach((r, i) => {
    const y = 1.1 + i * 0.82;
    card(s, 0.45, y, 9.1, 0.74);
    s.addShape(pres.shapes.OVAL, {
      x: 0.65, y: y + 0.17, w: 0.4, h: 0.4,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText(r.n, {
      x: 0.65, y: y + 0.2, w: 0.4, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.bg, bold: true, align: "center", margin: 0,
    });
    s.addText(r.t, {
      x: 1.25, y: y + 0.12, w: 2.2, h: 0.5,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0,
    });
    s.addText(r.d, {
      x: 3.5, y: y + 0.12, w: 5.8, h: 0.52,
      fontSize: 12, fontFace: "Calibri", color: C.muted, valign: "middle", margin: 0,
    });
  });
}

// ── 7. Component map ──────────────────────────────────────
{
  const s = base("Software component map");
  const comps = [
    { f: "run_cli.py", r: "Evaluation entry point (hybrid / gnn / kalman)" },
    { f: "src/pipeline.py", r: "Orchestrator: clutter → route → update → manage" },
    { f: "src/updater.py", r: "NewHybridUpdater, GNNUpdater, FallbackUpdater" },
    { f: "src/clutter_classifier.py", r: "MLP clutter vs real plot features" },
    { f: "src/pairwise_*.py", r: "PSR–PSR & SSR–ANY association classifiers" },
    { f: "src/kalman_filter.py", r: "Continuous-time predict / update" },
    { f: "src/model_v3…v6.py", r: "Recurrent GAT / cross-attention trackers" },
    { f: "src/factory.py", r: "Versioned model suite resolution" },
    { f: "src/metrics.py", r: "MOTA, MOTP, precision, recall, ID switches" },
    { f: "dashboard/app.py", r: "Streamlit experiments & frame replay" },
  ];
  comps.forEach((c, i) => {
    const col = i < 5 ? 0 : 1;
    const row = i % 5;
    const x = 0.45 + col * 4.7;
    const y = 1.1 + row * 0.8;
    card(s, x, y, 4.5, 0.7);
    s.addText(c.f, {
      x: x + 0.15, y: y + 0.1, w: 4.2, h: 0.25,
      fontSize: 12, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
    });
    s.addText(c.r, {
      x: x + 0.15, y: y + 0.35, w: 4.2, h: 0.25,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 8. Hybrid overview ────────────────────────────────────
{
  const s = base("Hybrid tracker — production core");
  s.addText("ML decides who matches; physics decides how motion evolves.", {
    x: 0.45, y: 1.05, w: 9.1, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
  });

  const steps = [
    { n: "01", t: "Async projection", d: "Project each track to exact meas_t (not window end) to avoid temporal dragging." },
    { n: "02", t: "Spatial clustering", d: "Fuse co-located PSR/SSR reports (~2 km) with pairwise ML into meta-measurements." },
    { n: "03", t: "Temporal association", d: "Score track↔meas pairs (gate ~8 km); Hungarian assignment for global match." },
    { n: "04", t: "Kalman update", d: "Exact dt = meas_t − track_t continuous-time KF update; initiate unmatched plots." },
  ];
  steps.forEach((st, i) => {
    const y = 1.45 + i * 0.95;
    card(s, 0.45, y, 9.1, 0.88, { top: i % 2 === 0 ? C.accent : C.purple });
    s.addText(st.n, {
      x: 0.65, y: y + 0.25, w: 0.7, h: 0.4,
      fontSize: 18, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
    });
    s.addText(st.t, {
      x: 1.5, y: y + 0.15, w: 7.8, h: 0.3,
      fontSize: 15, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(st.d, {
      x: 1.5, y: y + 0.48, w: 7.8, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 9. ML components ──────────────────────────────────────
{
  const s = base("Learned components");
  card(s, 0.45, 1.15, 4.45, 3.95, { top: C.purple });
  s.addText("CLUTTER CLASSIFIER", {
    x: 0.65, y: 1.4, w: 4.0, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.purple, bold: true, margin: 0,
  });
  s.addText("3-layer MLP", {
    x: 0.65, y: 1.8, w: 4.0, h: 0.3,
    fontSize: 18, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  const clutter = [
    "Features: amp, velocity, normalized XYZ, type",
    "Output: P(clutter) via sigmoid",
    "Loss: weighted BCE (class imbalance)",
    "Role: hard-drop false alarms early",
    "Module: clutter_classifier.py",
  ];
  clutter.forEach((t, i) => {
    s.addText("•  " + t, {
      x: 0.7, y: 2.3 + i * 0.45, w: 4.0, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });

  card(s, 5.1, 1.15, 4.4, 3.95, { top: C.accent });
  s.addText("PAIRWISE CLASSIFIERS", {
    x: 5.3, y: 1.4, w: 4.0, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText("Dual specialized MLPs", {
    x: 5.3, y: 1.8, w: 4.0, h: 0.3,
    fontSize: 18, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  const pair = [
    "PSR–PSR: kinematics & geometry",
    "SSR–ANY: + Mode 3A / Mode S identity",
    "Features: dist, Δv cosine, altitude, …",
    "Used in Hybrid gates and GNN edges",
    "Modules: pairwise_classifier / features",
  ];
  pair.forEach((t, i) => {
    s.addText("•  " + t, {
      x: 5.35, y: 2.3 + i * 0.45, w: 4.0, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });
}

// ── 10. Async Kalman ──────────────────────────────────────
{
  const s = base("Continuous-time Kalman (no temporal dragging)");
  card(s, 0.45, 1.15, 9.1, 1.5);
  s.addText("Naive windowing projects all tracks to the window end → wrong geometry when radars scan asynchronously.", {
    x: 0.7, y: 1.4, w: 8.6, h: 0.45,
    fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
  });
  s.addText("Hybrid fix: store track at last measurement time; for scoring and update use exact  dt = meas_t − track_t.", {
    x: 0.7, y: 1.95, w: 8.6, h: 0.45,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });

  const kcards = [
    { t: "Predict", d: "Propagate mean & covariance with continuous-time process noise Q(dt)." },
    { t: "Associate", d: "ML pairwise scores inside spatial gate; optimal bipartite matching." },
    { t: "Update", d: "Kalman gain balances prediction vs measurement noise R." },
    { t: "Coast", d: "Unobserved tracks advance in time; deleted after max_age (~20 s)." },
  ];
  kcards.forEach((k, i) => {
    const x = 0.45 + i * 2.35;
    card(s, x, 2.95, 2.2, 2.15, { top: C.accent });
    s.addText(k.t, {
      x: x + 0.12, y: 3.2, w: 1.96, h: 0.35,
      fontSize: 15, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(k.d, {
      x: x + 0.12, y: 3.65, w: 1.96, h: 1.2,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 11. Track lifecycle ───────────────────────────────────
{
  const s = base("Track lifecycle management");
  const phases = [
    { t: "Unmatched plot", d: "Spawn tentative track with large initial P", c: C.muted },
    { t: "Tentative", d: "Needs min_hits (e.g. 3) before export", c: C.warn },
    { t: "Confirmed", d: "Active output track; KF updates on match", c: C.success },
    { t: "Coasting", d: "No hit this window; predict forward", c: C.accent },
    { t: "Deleted", d: "Age > max_age → remove", c: C.danger },
  ];
  phases.forEach((p, i) => {
    const x = 0.4 + i * 1.9;
    card(s, x, 1.5, 1.75, 2.4, { top: p.c });
    s.addText(String(i + 1), {
      x: x + 0.1, y: 1.75, w: 1.55, h: 0.35,
      fontSize: 20, fontFace: "Consolas", color: p.c, bold: true, align: "center", margin: 0,
    });
    s.addText(p.t, {
      x: x + 0.1, y: 2.25, w: 1.55, h: 0.55,
      fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, align: "center", margin: 0,
    });
    s.addText(p.d, {
      x: x + 0.1, y: 2.9, w: 1.55, h: 0.8,
      fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center", margin: 0,
    });
  });
  card(s, 0.45, 4.2, 9.1, 1.0);
  s.addText(
    "Raising min_hits and extending coast time lifted recall from ~0.47 to ~0.94 on dense async simulation by surviving radar blind spots without flooding false tracks.",
    {
      x: 0.7, y: 4.4, w: 8.6, h: 0.6,
      fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
    }
  );
}

// ── 12. GNN lineage ───────────────────────────────────────
{
  const s = base("GNN model lineage (V3 → V6)");
  const versions = [
    { v: "V3", t: "Recurrent GAT", d: "Nodes = tracks + meas; edges gated by pairwise ML; GRU memory; Δstate + existence." },
    { v: "V4", t: "Fusion / learnable", d: "Richer fusion path; streaming train loop; still window-batched graph." },
    { v: "V5", t: "Clutter head", d: "Integrated clutter logits with GAT; early filtering inside model." },
    { v: "V6", t: "Cross-attention", d: "Bipartite QKV: tracks query measurements only — no M↔M ghost edges." },
  ];
  versions.forEach((v, i) => {
    const y = 1.15 + i * 1.0;
    card(s, 0.45, y, 9.1, 0.9);
    pill(s, 0.65, y + 0.25, 0.85, 0.4, v.v, C.accent, C.bg);
    s.addText(v.t, {
      x: 1.7, y: y + 0.15, w: 7.5, h: 0.3,
      fontSize: 15, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(v.d, {
      x: 1.7, y: y + 0.48, w: 7.5, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 13. V6 detail ─────────────────────────────────────────
{
  const s = base("V6 bipartite cross-attention design");
  card(s, 0.45, 1.15, 5.9, 4.0, { top: C.warn });
  s.addText("Information flow", {
    x: 0.7, y: 1.4, w: 5.4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.warn, bold: true, margin: 0,
  });
  const flow = [
    { t: "Tracks → Queries (Q)", d: "Existing track hidden states" },
    { t: "Measurements → Keys / Values", d: "After early clutter hard-drop" },
    { t: "Association A = softmax(QKᵀ/√d)", d: "Strictly N_tracks × N_meas" },
    { t: "H′_T = H_T + MultiHead(Q,K,V)", d: "Measurement-informed track update" },
    { t: "Decoder", d: "State deltas + new track seeds" },
  ];
  flow.forEach((f, i) => {
    s.addText(f.t, {
      x: 0.75, y: 1.9 + i * 0.55, w: 5.3, h: 0.25,
      fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(f.d, {
      x: 0.75, y: 2.12 + i * 0.55, w: 5.3, h: 0.22,
      fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });

  card(s, 6.55, 1.15, 2.95, 4.0, { top: C.accent });
  s.addText("Why bipartite?", {
    x: 6.75, y: 1.45, w: 2.55, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText(
    "Full graphs let measurements reinforce each other, creating ghost tracks. V6 forces association through tracks only.",
    {
      x: 6.75, y: 2.0, w: 2.55, h: 1.8,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
    }
  );
  s.addText(
    "factory.py detects cross_attn weights and loads model_v6 automatically.",
    {
      x: 6.75, y: 4.0, w: 2.55, h: 0.9,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    }
  );
}

// ── 14. Hybrid vs GNN ─────────────────────────────────────
{
  const s = base("Hybrid vs GNN — design trade-offs");
  // header
  card(s, 0.45, 1.15, 3.0, 0.55, { fill: C.soft });
  s.addText("Dimension", {
    x: 0.55, y: 1.25, w: 2.8, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.muted, bold: true, margin: 0,
  });
  card(s, 3.5, 1.15, 3.0, 0.55, { fill: C.soft });
  s.addText("Hybrid", {
    x: 3.6, y: 1.25, w: 2.8, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.success, bold: true, margin: 0,
  });
  card(s, 6.55, 1.15, 3.0, 0.55, { fill: C.soft });
  s.addText("GNN (V3–V6)", {
    x: 6.65, y: 1.25, w: 2.8, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });

  const rows = [
    ["Philosophy", "Modular rules + ML scores", "End-to-end graph learning"],
    ["Motion model", "Continuous-time Kalman", "GRU / decoder deltas"],
    ["Association", "Pairwise + Hungarian", "Attention / edge weights"],
    ["Interpretability", "High (gates, KF, IDs)", "Lower (black-box)"],
    ["Async streaming", "Strong (exact dt)", "Harder (window batches)"],
    ["Best use", "Operational hybrid path", "Research & ablation"],
  ];
  rows.forEach((r, i) => {
    const y = 1.8 + i * 0.55;
    const bg = i % 2 === 0 ? C.card : C.soft;
    card(s, 0.45, y, 3.0, 0.5, { fill: bg });
    card(s, 3.5, y, 3.0, 0.5, { fill: bg });
    card(s, 6.55, y, 3.0, 0.5, { fill: bg });
    s.addText(r[0], { x: 0.55, y: y + 0.1, w: 2.8, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.white, bold: true, margin: 0 });
    s.addText(r[1], { x: 3.6, y: y + 0.1, w: 2.8, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.ice, margin: 0 });
    s.addText(r[2], { x: 6.65, y: y + 0.1, w: 2.8, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.ice, margin: 0 });
  });
}

// ── 15. Data layer ────────────────────────────────────────
{
  const s = base("Data layer & training assets");
  card(s, 0.45, 1.15, 4.45, 3.95, { top: C.accent });
  s.addText("CANONICAL DATA", {
    x: 0.65, y: 1.4, w: 4.0, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  const can = [
    "data/canonical/ streams & episodes",
    "Sweden 15 / 30 / 60 min packages",
    "UAE 2 min multi-radar stream",
    "sim_batch_hetero for classifiers",
    "Manifests + schema.md inventory",
  ];
  can.forEach((t, i) => {
    s.addText("•  " + t, {
      x: 0.7, y: 1.95 + i * 0.5, w: 4.0, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
    });
  });

  card(s, 5.1, 1.15, 4.4, 3.95, { top: C.warn });
  s.addText("TRAINING PATHS", {
    x: 5.3, y: 1.4, w: 4.0, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.warn, bold: true, margin: 0,
  });
  const tr = [
    "train_clutter_filter.py",
    "train_hetero_pairwise.py",
    "train_gnn_tracker.py",
    "train_streaming_v3…v6.py",
    "scripts/data/* inventory & stitch",
  ];
  tr.forEach((t, i) => {
    s.addText("•  " + t, {
      x: 5.35, y: 1.95 + i * 0.5, w: 4.0, h: 0.4,
      fontSize: 13, fontFace: "Consolas", color: C.ice, margin: 0,
    });
  });
}

// ── 16. Platform tooling ──────────────────────────────────
{
  const s = base("Platform & research tooling");
  const tools = [
    { t: "CLI", d: "run_cli.py — mode hybrid|gnn|kalman, data path, MLflow flags", c: C.accent },
    { t: "Dashboard", d: "Streamlit app — ablations, frame replay, video export", c: C.purple },
    { t: "MLflow", d: "Experiment tracking — MOTA, ID switches, FP/frame", c: C.success },
    { t: "Config", d: "Pydantic schemas (config_schemas.py) — type-safe pipelines", c: C.warn },
    { t: "Optimize", d: "Optuna studies via src/optimize.py", c: C.ice },
    { t: "Checkpoints", d: "Versioned .pt weights for classifiers + GNN suite", c: C.danger },
  ];
  tools.forEach((t, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.45 + col * 3.15;
    const y = 1.2 + row * 2.0;
    card(s, x, y, 3.0, 1.8, { top: t.c });
    s.addText(t.t, {
      x: x + 0.15, y: y + 0.3, w: 2.7, h: 0.35,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(t.d, {
      x: x + 0.15, y: y + 0.75, w: 2.7, h: 0.85,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 17. Performance snapshot ──────────────────────────────
{
  const s = base("Performance snapshot (Hybrid, dense 120s sim)");
  const metrics = [
    { m: "MOTA", v: "0.82", n: "Overall tracking quality" },
    { m: "MOTP", v: "806 m", n: "~3σ radar noise floor" },
    { m: "Precision", v: "0.89", n: "True tracks vs false" },
    { m: "Recall", v: "0.94", n: "Targets maintained" },
    { m: "F1", v: "0.91", n: "Balanced score" },
    { m: "ID SW", v: "0", n: "Identity swaps" },
  ];
  metrics.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.45 + col * 3.15;
    const y = 1.2 + row * 1.9;
    card(s, x, y, 3.0, 1.7, { top: C.success });
    s.addText(m.m, {
      x: x + 0.15, y: y + 0.25, w: 2.7, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.success, bold: true, margin: 0,
    });
    s.addText(m.v, {
      x: x + 0.15, y: y + 0.55, w: 2.7, h: 0.5,
      fontSize: 28, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(m.n, {
      x: x + 0.15, y: y + 1.15, w: 2.7, h: 0.3,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 18. Design principles ─────────────────────────────────
{
  const s = base("Architectural principles");
  const principles = [
    { n: "01", t: "Strategy pattern for tracking", d: "Pipeline fixed; Hybrid / GNN / Kalman are pluggable updaters." },
    { n: "02", t: "ML where discrete decisions win", d: "Clutter and association are classification problems; motion stays physics-constrained." },
    { n: "03", t: "Respect asynchronous time", d: "Exact measurement timestamps beat batch window centers." },
    { n: "04", t: "Versioned models behind a factory", d: "Checkpoints auto-detect V4–V6 layout; training suites stay paired." },
    { n: "05", t: "Canonical data is first-class", d: "Schema, manifests, and Git-tracked mid-size sets keep experiments reproducible." },
    { n: "06", t: "Measure what operators care about", d: "MOTA, ID switches, and recall under coasting — not only loss curves." },
  ];
  principles.forEach((p, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.45 + col * 4.7;
    const y = 1.1 + row * 1.35;
    card(s, x, y, 4.5, 1.2);
    s.addText(p.n, {
      x: x + 0.2, y: y + 0.25, w: 0.6, h: 0.35,
      fontSize: 16, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
    });
    s.addText(p.t, {
      x: x + 0.9, y: y + 0.2, w: 3.4, h: 0.3,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(p.d, {
      x: x + 0.9, y: y + 0.55, w: 3.4, h: 0.5,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 19. Summary ───────────────────────────────────────────
{
  const s = base("Summary");
  const bullets = [
    { t: "Unified pipeline", d: "One path from multi-radar plots to correlated tracks." },
    { t: "Hybrid is the strong core", d: "Pairwise ML + continuous-time Kalman delivers MOTA ≈ 0.82." },
    { t: "GNN family for research", d: "V3–V6 explore attention; V6 bipartite design targets ghost tracks." },
    { t: "Platform ready", d: "CLI, Streamlit, MLflow, Pydantic configs, canonical data." },
  ];
  bullets.forEach((b, i) => {
    const y = 1.2 + i * 0.9;
    card(s, 0.45, y, 9.1, 0.8);
    s.addText(b.t, {
      x: 0.7, y: y + 0.12, w: 8.6, h: 0.28,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(b.d, {
      x: 0.7, y: y + 0.42, w: 8.6, h: 0.28,
      fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
}

// ── 20. Close ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("THANK YOU", {
    x: 0.55, y: 1.8, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.accent, bold: true, margin: 0, charSpacing: 2,
  });
  s.addText("Questions & discussion", {
    x: 0.55, y: 2.3, w: 9, h: 0.7,
    fontSize: 34, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  s.addText(
    "Repo: github.com/TomRathbun/ai_tracker_correlator\nDocs: ai_tracker_architecture.md · architecture_diagrams.md · artifacts/design_v6.md",
    {
      x: 0.55, y: 3.3, w: 9, h: 0.8,
      fontSize: 14, fontFace: "Calibri", color: C.muted, margin: 0,
    }
  );
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Wrote", OUT);
}).catch((err) => {
  console.error(err);
  process.exit(1);
});
