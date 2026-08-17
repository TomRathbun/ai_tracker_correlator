const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Thomas Rathbun, PhD";
pres.title = "AI Tracker Correlator — Business Problem Statement";
pres.subject = "Capstone Project";

// Palette: Midnight air-traffic / radar ops
const C = {
  bg: "0B1220",
  panel: "121A2B",
  card: "172033",
  navy: "1E3A5F",
  ice: "A8C5E2",
  white: "FFFFFF",
  muted: "94A3B8",
  accent: "38BDF8",
  warn: "F59E0B",
  soft: "1E293B",
  line: "334155",
};

const slide = pres.addSlide();
slide.background = { color: C.bg };

// Left accent bar
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 0.12, h: 5.625,
  fill: { color: C.accent }, line: { color: C.accent },
});

// Eyebrow
slide.addText("CAPSTONE  ·  AIR TRAFFIC SURVEILLANCE", {
  x: 0.45, y: 0.28, w: 9.1, h: 0.28,
  fontSize: 11, fontFace: "Calibri", color: C.accent,
  bold: true, margin: 0, charSpacing: 1.5,
});
slide.addText("LOCKHEED MARTIN PROPRIETARY INFORMATION", {
  x: 0.45, y: 0.06, w: 9.1, h: 0.2,
  fontSize: 9, fontFace: "Calibri", color: "FECACA", bold: true, margin: 0,
});
slide.addText("Thomas Rathbun, PhD  ·  LOCKHEED MARTIN PROPRIETARY INFORMATION", {
  x: 0.45, y: 5.38, w: 9.1, h: 0.2,
  fontSize: 9, fontFace: "Calibri", color: C.muted, margin: 0,
});

// Title
slide.addText("Business Problem Statement", {
  x: 0.45, y: 0.55, w: 9.1, h: 0.55,
  fontSize: 32, fontFace: "Calibri", color: C.white,
  bold: true, margin: 0,
});

// Core statement card
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.45, y: 1.2, w: 9.1, h: 1.15,
  fill: { color: C.card },
  line: { color: C.line, width: 1 },
  rectRadius: 0.08,
});

slide.addText("Air traffic surveillance still depends on a fragmented pipeline: each radar runs its own tracker, then a separate correlator tries to resolve duplicates across sensors. That design is costly to maintain, slow to adapt to clutter and sensor bias, and hard to scale as airspace density and multi-radar coverage grow.", {
  x: 0.65, y: 1.35, w: 8.7, h: 0.9,
  fontSize: 14, fontFace: "Calibri", color: C.ice,
  margin: 0, valign: "middle",
});

// Three problem pillars
const pillars = [
  {
    title: "Operational Complexity",
    body: "Per-sensor physics trackers plus a downstream correlator create many failure points, hand-tuned gates, and high sustainment cost for ATC and defense surveillance systems.",
  },
  {
    title: "Fusion Ambiguity",
    body: "PSR clutter, SSR intermittency, registration bias, and asynchronous scans produce duplicate tracks, ID switches, and degraded situation awareness in dense airspace.",
  },
  {
    title: "Limited Adaptivity",
    body: "Rule-based association does not learn from data. It struggles with non-linear maneuvers, modality mix (PSR/SSR), and novel clutter patterns without expert re-tuning.",
  },
];

pillars.forEach((p, i) => {
  const x = 0.45 + i * 3.1;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: 2.55, w: 2.95, h: 2.15,
    fill: { color: C.card },
    line: { color: C.line, width: 1 },
    rectRadius: 0.08,
  });
  // top accent strip
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 2.55, w: 2.95, h: 0.08,
    fill: { color: i === 1 ? C.warn : C.accent },
    line: { color: i === 1 ? C.warn : C.accent },
  });
  slide.addText(String(i + 1).padStart(2, "0"), {
    x: x + 0.18, y: 2.75, w: 0.6, h: 0.3,
    fontSize: 12, fontFace: "Consolas", color: C.accent, bold: true, margin: 0,
  });
  slide.addText(p.title, {
    x: x + 0.18, y: 3.1, w: 2.6, h: 0.4,
    fontSize: 15, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
  });
  slide.addText(p.body, {
    x: x + 0.18, y: 3.5, w: 2.6, h: 1.05,
    fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
  });
});

// Footer opportunity line
slide.addText("Opportunity:  Evaluate whether a unified AI/ML tracker-correlator can simplify the stack while meeting operational tracking quality (MOTA, ID stability, real-time fusion).", {
  x: 0.45, y: 4.95, w: 9.1, h: 0.4,
  fontSize: 12, fontFace: "Calibri", color: C.ice, italic: true, margin: 0,
});

pres.writeFile({ fileName: "artifacts/Business_Problem_Statement.pptx" })
  .then(() => console.log("Wrote artifacts/Business_Problem_Statement.pptx"))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
