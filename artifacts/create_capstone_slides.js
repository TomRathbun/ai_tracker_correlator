const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Thomas Rathbun, PhD";
pres.title = "AI Tracker Correlator — Capstone Brief";
pres.subject = "Company Capstone Project";

const C = {
  bg: "0B1220",
  card: "172033",
  ice: "A8C5E2",
  white: "FFFFFF",
  muted: "94A3B8",
  accent: "38BDF8",
  warn: "F59E0B",
  danger: "F87171",
  success: "34D399",
  line: "334155",
  soft: "1E293B",
};

function baseSlide(title, eyebrow = "CAPSTONE  ·  AIR TRAFFIC SURVEILLANCE") {
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  slide.addText(eyebrow, {
    x: 0.45, y: 0.22, w: 9.1, h: 0.26,
    fontSize: 11, fontFace: "Calibri", color: C.accent,
    bold: true, margin: 0, charSpacing: 1.2,
  });
  slide.addText(title, {
    x: 0.45, y: 0.48, w: 9.1, h: 0.5,
    fontSize: 28, fontFace: "Calibri", color: C.white,
    bold: true, margin: 0,
  });
  slide.addText("LOCKHEED MARTIN PROPRIETARY INFORMATION", {
    x: 0.45, y: 0.02, w: 6.4, h: 0.18,
    fontSize: 8, fontFace: "Calibri", color: "FECACA", bold: true, margin: 0,
  });
  slide.addText("Thomas Rathbun, PhD  ·  LOCKHEED MARTIN PROPRIETARY INFORMATION", {
    x: 0.45, y: 5.38, w: 8.2, h: 0.2,
    fontSize: 9, fontFace: "Calibri", color: C.muted, margin: 0,
  });
  return slide;
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.card },
    line: { color: opts.line || C.line, width: 1 },
    rectRadius: 0.08,
  });
  if (opts.topAccent) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.07,
      fill: { color: opts.topAccent },
      line: { color: opts.topAccent },
    });
  }
}

// ─────────────────────────────────────────────
// 1. Business Problem Statement
// ─────────────────────────────────────────────
{
  const slide = baseSlide("Business Problem Statement");
  card(slide, 0.45, 1.15, 9.1, 1.05);
  slide.addText(
    "Air traffic surveillance still depends on a fragmented pipeline: each radar runs its own tracker, then a separate correlator tries to resolve duplicates across sensors. That design is costly to maintain, slow to adapt to clutter and sensor bias, and hard to scale as airspace density and multi-radar coverage grow.",
    {
      x: 0.65, y: 1.28, w: 8.7, h: 0.8,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0, valign: "middle",
    }
  );

  const pillars = [
    {
      n: "01",
      t: "Operational Complexity",
      b: "Per-sensor physics trackers plus a downstream correlator create many failure points, hand-tuned gates, and high sustainment cost.",
      a: C.accent,
    },
    {
      n: "02",
      t: "Fusion Ambiguity",
      b: "PSR clutter, SSR intermittency, registration bias, and asynchronous scans produce duplicate tracks and ID switches.",
      a: C.warn,
    },
    {
      n: "03",
      t: "Limited Adaptivity",
      b: "Rule-based association does not learn from data and struggles with non-linear maneuvers and novel clutter without expert re-tuning.",
      a: C.accent,
    },
  ];
  pillars.forEach((p, i) => {
    const x = 0.45 + i * 3.1;
    card(slide, x, 2.4, 2.95, 2.0, { topAccent: p.a });
    slide.addText(p.n, {
      x: x + 0.18, y: 2.58, w: 0.6, h: 0.28,
      fontSize: 12, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
    });
    slide.addText(p.t, {
      x: x + 0.18, y: 2.9, w: 2.6, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    slide.addText(p.b, {
      x: x + 0.18, y: 3.3, w: 2.6, h: 0.95,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
  slide.addText(
    "Opportunity: Evaluate a unified AI/ML tracker-correlator that simplifies the stack while meeting operational tracking quality.",
    {
      x: 0.45, y: 4.6, w: 9.1, h: 0.35,
      fontSize: 12, fontFace: "Calibri", color: C.ice, italic: true, margin: 0,
    }
  );
  slide.addText("1 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 2. Proposed Solution
// ─────────────────────────────────────────────
{
  const slide = baseSlide("Proposed Solution");
  card(slide, 0.45, 1.15, 9.1, 0.72);
  slide.addText(
    "A hybrid AI/ML tracker-correlator that ingests multi-sensor radar plots (PSR/SSR), learns association and clutter rejection, and produces a single correlated track picture—keeping physics-based state estimation where it is strongest.",
    {
      x: 0.65, y: 1.25, w: 8.7, h: 0.55,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0, valign: "middle",
    }
  );

  const steps = [
    { n: "1", t: "Clutter Filter", d: "Unary MLP scores false alarms early to cut noise before association." },
    { n: "2", t: "Learned Association", d: "Dual pairwise classifiers (PSR kinematics + SSR identity) fuse multi-sensor reports." },
    { n: "3", t: "Async Kalman Update", d: "Exact-time prediction/update removes temporal dragging across staggered scans." },
    { n: "4", t: "Track Management", d: "M/N initiation, coasting, and de-duplicated correlated outputs." },
  ];
  steps.forEach((s, i) => {
    const x = 0.45 + i * 2.35;
    card(slide, x, 2.1, 2.2, 2.15, { topAccent: C.accent });
    slide.addShape(pres.shapes.OVAL, {
      x: x + 0.18, y: 2.3, w: 0.38, h: 0.38,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    slide.addText(s.n, {
      x: x + 0.18, y: 2.33, w: 0.38, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.bg, bold: true, align: "center", margin: 0,
    });
    slide.addText(s.t, {
      x: x + 0.18, y: 2.85, w: 1.85, h: 0.45,
      fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    slide.addText(s.d, {
      x: x + 0.18, y: 3.35, w: 1.85, h: 0.75,
      fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });

  slide.addText(
    "Research path: pairwise + hybrid baseline → assess pure GNN/transformer correlator feasibility → document recommendations for production path.",
    {
      x: 0.45, y: 4.5, w: 9.1, h: 0.4,
      fontSize: 12, fontFace: "Calibri", color: C.ice, italic: true, margin: 0,
    }
  );
  slide.addText("2 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 3. ROI / Value Statement
// ─────────────────────────────────────────────
{
  const slide = baseSlide("ROI Calculation / Value Statement");
  card(slide, 0.45, 1.15, 9.1, 0.65);
  slide.addText(
    "This is a feasibility capstone: value is measured in reduced system complexity, tracking quality, and a clear go / no-go path—not immediate product revenue.",
    {
      x: 0.65, y: 1.25, w: 8.7, h: 0.48,
      fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0, valign: "middle",
    }
  );

  const vals = [
    {
      t: "Engineering Efficiency",
      items: [
        "Fewer hand-tuned association gates per sensor suite",
        "One fused pipeline vs. N trackers + correlator",
        "Reusable evaluation harness (MOTA, ID switches)",
      ],
    },
    {
      t: "Operational Quality",
      items: [
        "Target: high MOTA / precision with near-zero ID switches on sim streams",
        "Better multi-radar de-duplication → cleaner common operating picture",
        "Earlier clutter rejection → lower downstream load",
      ],
    },
    {
      t: "Strategic Upside",
      items: [
        "Evidence package for ML investment vs. pure classical fusion",
        "Identifies where hybrid beats end-to-end deep models",
        "De-risks future transformer / GNN correlator work",
      ],
    },
  ];
  vals.forEach((v, i) => {
    const x = 0.45 + i * 3.1;
    card(slide, x, 2.0, 2.95, 2.55, { topAccent: i === 1 ? C.success : C.accent });
    slide.addText(v.t, {
      x: x + 0.18, y: 2.2, w: 2.6, h: 0.4,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    slide.addText(
      v.items.map((it, idx) => ({
        text: it,
        options: { bullet: true, breakLine: idx < v.items.length - 1 },
      })),
      {
        x: x + 0.15, y: 2.7, w: 2.65, h: 1.65,
        fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0, valign: "top",
      }
    );
  });
  slide.addText("3 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 4. Stakeholders
// ─────────────────────────────────────────────
{
  const slide = baseSlide("End-user Stakeholder(s) and Roles");

  const rows = [
    { role: "Surveillance / ATC Engineering", need: "Primary consumers of architecture, metrics, and integration guidance for multi-radar fusion.", how: "Review design; assess fit vs. current correlators." },
    { role: "Tracker / Sensor Fusion SMEs", need: "Validate association, clutter, and async-timing assumptions against operational practice.", how: "Technical review of hybrid vs. pure-ML results." },
    { role: "Data / ML Engineering", need: "Own training data pipelines, labeling, and model ops if the approach is adopted.", how: "Advise on data quality, schema, and retrain path." },
    { role: "Product / Program Leadership", need: "Decide investment: extend hybrid, pursue transformer correlator, or stay classical.", how: "Use feasibility + risk findings for roadmap." },
    { role: "Operations (future users)", need: "Need a stable common track picture with low false tracks and ID stability.", how: "Indirect; success criteria framed around their COP quality." },
  ];

  // Header bar
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.45, y: 1.15, w: 9.1, h: 0.38,
    fill: { color: C.soft }, line: { color: C.line, width: 1 }, rectRadius: 0.05,
  });
  slide.addText("Stakeholder", {
    x: 0.55, y: 1.2, w: 2.6, h: 0.28,
    fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  slide.addText("Interest / Need", {
    x: 3.2, y: 1.2, w: 3.5, h: 0.28,
    fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  slide.addText("Role in Capstone", {
    x: 6.8, y: 1.2, w: 2.6, h: 0.28,
    fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });

  rows.forEach((r, i) => {
    const y = 1.6 + i * 0.68;
    const bg = i % 2 === 0 ? C.card : "141C2E";
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.45, y, w: 9.1, h: 0.62,
      fill: { color: bg }, line: { color: C.line, width: 0.5 }, rectRadius: 0.04,
    });
    slide.addText(r.role, {
      x: 0.55, y: y + 0.1, w: 2.55, h: 0.42,
      fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, margin: 0, valign: "middle",
    });
    slide.addText(r.need, {
      x: 3.2, y: y + 0.08, w: 3.45, h: 0.48,
      fontSize: 10, fontFace: "Calibri", color: C.muted, margin: 0, valign: "middle",
    });
    slide.addText(r.how, {
      x: 6.8, y: y + 0.08, w: 2.6, h: 0.48,
      fontSize: 10, fontFace: "Calibri", color: C.ice, margin: 0, valign: "middle",
    });
  });
  slide.addText("4 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 5. Data Requirements
// ─────────────────────────────────────────────
{
  const slide = baseSlide("Data Requirements");

  const blocks = [
    {
      t: "Inputs",
      a: C.accent,
      items: [
        "Multi-sensor plots: PSR position/velocity/amplitude; SSR Mode 3A/S + position",
        "Decoded Asterix-style streams (CAT 048 / 062 derived) in Cartesian form",
        "Timestamps, sensor IDs, and modality tags for async fusion",
      ],
    },
    {
      t: "Labels / Truth",
      a: C.warn,
      items: [
        "Ground-truth track IDs for association training & MOTA",
        "True kinematics (x,y,z,vx,vy,vz) independent of noisy plots",
        "Clutter / false-alarm flags where available",
      ],
    },
    {
      t: "Quality Bars",
      a: C.success,
      items: [
        "Canonical schema (one field language across sim & real)",
        "Scenario splits by track / time / region (no leakage)",
        "Hard negatives: crossing tracks, multi-radar bias, dropouts",
      ],
    },
  ];
  blocks.forEach((b, i) => {
    const x = 0.45 + i * 3.1;
    card(slide, x, 1.15, 2.95, 2.55, { topAccent: b.a });
    slide.addText(b.t, {
      x: x + 0.18, y: 1.35, w: 2.6, h: 0.35,
      fontSize: 15, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    slide.addText(
      b.items.map((it, idx) => ({
        text: it,
        options: { bullet: true, breakLine: idx < b.items.length - 1 },
      })),
      {
        x: x + 0.15, y: 1.85, w: 2.65, h: 1.7,
        fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
      }
    );
  });

  card(slide, 0.45, 3.9, 9.1, 1.1);
  slide.addText("Current project assets", {
    x: 0.65, y: 4.05, w: 8.7, h: 0.28,
    fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  slide.addText(
    "Synthetic multi-radar JSONL streams · CAT-62–derived scenarios · Pairwise/clutter training sets · Eval metrics (MOTA, MOTP, ID switches). Gap: clean regional real-data packages and consistent GT kinematics remain a risk for pure end-to-end models.",
    {
      x: 0.65, y: 4.38, w: 8.7, h: 0.5,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
    }
  );
  slide.addText("5 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 6. Deliverable and Due Date
// ─────────────────────────────────────────────
{
  const slide = baseSlide("Deliverable and Due Date");

  // Critical schedule callout
  card(slide, 0.45, 1.15, 9.1, 0.85, { topAccent: C.warn, fill: "1A2333" });
  slide.addText("Hard constraint", {
    x: 0.65, y: 1.3, w: 8.7, h: 0.25,
    fontSize: 12, fontFace: "Calibri", color: C.warn, bold: true, margin: 0,
  });
  slide.addText(
    "Write-up due 2 Sep 2026 · Leave starts 20 Aug 2026 → all substantive work, demos, and draft write-up complete by 19 Aug 2026.",
    {
      x: 0.65, y: 1.55, w: 8.7, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    }
  );

  // Deliverables
  const dels = [
    { t: "Technical write-up", d: "Problem, method, experiments, results, recommendations", due: "2 Sep (draft 19 Aug)" },
    { t: "Working prototype", d: "Hybrid + pairwise pipeline; selected model experiments", due: "≤ 15 Aug" },
    { t: "Evaluation package", d: "Metrics tables, plots/GIFs, baseline comparison", due: "≤ 18 Aug" },
    { t: "Stakeholder brief", d: "This deck + short demo narrative", due: "≤ 19 Aug" },
  ];
  dels.forEach((d, i) => {
    const y = 2.2 + i * 0.58;
    card(slide, 0.45, y, 9.1, 0.52);
    slide.addShape(pres.shapes.OVAL, {
      x: 0.62, y: y + 0.12, w: 0.28, h: 0.28,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    slide.addText(String(i + 1), {
      x: 0.62, y: y + 0.13, w: 0.28, h: 0.26,
      fontSize: 11, fontFace: "Calibri", color: C.bg, bold: true, align: "center", margin: 0,
    });
    slide.addText(d.t, {
      x: 1.1, y: y + 0.1, w: 2.6, h: 0.32,
      fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, margin: 0, valign: "middle",
    });
    slide.addText(d.d, {
      x: 3.8, y: y + 0.1, w: 3.5, h: 0.32,
      fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0, valign: "middle",
    });
    slide.addText(d.due, {
      x: 7.4, y: y + 0.1, w: 1.95, h: 0.32,
      fontSize: 12, fontFace: "Calibri", color: C.ice, bold: true, margin: 0, valign: "middle", align: "right",
    });
  });
  slide.addText("6 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

// ─────────────────────────────────────────────
// 7. Risks / Issues
// ─────────────────────────────────────────────
{
  const slide = baseSlide("Risks / Issues");

  const risks = [
    {
      level: "HIGH",
      color: C.danger,
      t: "Schedule compression (leave 20 Aug)",
      b: "Official due date is 2 Sep, but availability ends 19 Aug. Mitigation: freeze scope now; finalize draft + metrics before leave; buffer only for light edits if reachable remotely.",
    },
    {
      level: "HIGH",
      color: C.danger,
      t: "Data quality & label noise",
      b: "Schema inconsistency, reconstructed GT, and region mislabels hurt pure end-to-end models. Mitigation: prioritize hybrid baseline; document data gaps honestly; fix only highest-impact QA items.",
    },
    {
      level: "MED",
      color: C.warn,
      t: "End-to-end GNN underperformance",
      b: "Pure GAT models lag hybrid on async streams. Mitigation: treat as research finding, not failure; recommend transformer association + KF path as future work.",
    },
    {
      level: "MED",
      color: C.warn,
      t: "Scope creep (new architectures)",
      b: "Full transformer rebuild may not fit pre-leave window. Mitigation: architecture proposal + limited prototype only if hybrid write-up is locked first.",
    },
    {
      level: "LOW",
      color: C.success,
      t: "Stakeholder access / review lag",
      b: "Late SME feedback can slip narrative polish. Mitigation: share draft outline mid-July; book one review before 15 Aug.",
    },
  ];

  risks.forEach((r, i) => {
    const y = 1.1 + i * 0.78;
    card(slide, 0.45, y, 9.1, 0.72);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.58, y: y + 0.2, w: 0.7, h: 0.32,
      fill: { color: r.color }, line: { color: r.color }, rectRadius: 0.04,
    });
    slide.addText(r.level, {
      x: 0.58, y: y + 0.22, w: 0.7, h: 0.28,
      fontSize: 10, fontFace: "Calibri", color: C.bg, bold: true, align: "center", margin: 0,
    });
    slide.addText(r.t, {
      x: 1.45, y: y + 0.08, w: 7.9, h: 0.26,
      fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    slide.addText(r.b, {
      x: 1.45, y: y + 0.34, w: 7.9, h: 0.32,
      fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });
  slide.addText("7 / 7", {
    x: 8.7, y: 5.25, w: 0.9, h: 0.25,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "right", margin: 0,
  });
}

pres
  .writeFile({ fileName: "artifacts/Capstone_Project_Brief.pptx" })
  .then(() => console.log("Wrote artifacts/Capstone_Project_Brief.pptx"))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
