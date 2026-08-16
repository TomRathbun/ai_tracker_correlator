/**
 * Capstone progress report: CAT → Hybrid → V3–V8 → ensemble.
 * Generates a US-Letter Word document for submission.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, LevelFormat, PageNumber, VerticalAlign, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "Capstone_Progress_Report.docx");

const NAVY = "0B3D5C";
const STEEL = "1F4E79";
const HEADER_BG = "0B3D5C";
const ALT_BG = "E8EEF4";
const LINE = "B0BEC5";
const MUTED = "5A6A75";

const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1080; // 0.75"
const CONTENT = PAGE_W - 2 * MARGIN; // 10080

const thin = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const borders = { top: thin, bottom: thin, left: thin, right: thin };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function r(text, opts = {}) {
  return new TextRun({
    text,
    font: "Arial",
    size: opts.size || 22,
    bold: !!opts.bold,
    italics: !!opts.italics,
    color: opts.color || "222222",
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 160, before: opts.before ?? 0, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [r(text, opts)],
  });
}

function rich(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 160, before: opts.before ?? 0, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: runs,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 4 } },
    spacing: { before: 360, after: 200 },
    children: [r(text, { size: 32, bold: true, color: NAVY })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [r(text, { size: 26, bold: true, color: STEEL })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [r(text, { size: 24, bold: true, color: "334155" })],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 240 },
    alignment: AlignmentType.CENTER,
    children: [r(text, { size: 18, italics: true, color: MUTED })],
  });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 260 },
    children: [r(text, { size: 22 })],
  });
}

function cell(text, width, opts = {}) {
  const fill = opts.header ? HEADER_BG : opts.alt ? ALT_BG : "FFFFFF";
  const color = opts.header ? "FFFFFF" : "222222";
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 50, bottom: 50, left: 70, right: 70 },
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        alignment: opts.align || (opts.header ? AlignmentType.CENTER : AlignmentType.LEFT),
        children: [r(String(text), { size: opts.size || 18, bold: !!opts.header || !!opts.bold, color })],
      }),
    ],
  });
}

function table(headers, rows, widths) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const hdr = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, widths[i], { header: true })),
  });
  const body = rows.map((row, ri) =>
    new TableRow({
      children: row.map((c, i) =>
        cell(c, widths[i], { alt: ri % 2 === 1, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })
      ),
    })
  );
  return new Table({
    width: { size: sum, type: WidthType.DXA },
    columnWidths: widths,
    rows: [hdr, ...body],
  });
}

function img(relPath, maxW, maxH, alt) {
  const abs = path.join(ROOT, relPath);
  const data = fs.readFileSync(abs);
  // Natural sizes known from inspection; keep aspect.
  const natives = {
    "artifacts/tracker_simulation_holdout_2min_trails.png": [4500, 3000],
    "artifacts/tracks_sweden_30min_holdout.png": [1960, 980],
    "artifacts/tracks_canonical_comparison.png": [2084, 1796],
  };
  const [nw, nh] = natives[relPath.replace(/\\/g, "/")] || [1600, 900];
  let w = maxW;
  let h = Math.round((maxW * nh) / nw);
  if (h > maxH) {
    h = maxH;
    w = Math.round((maxH * nw) / nh);
  }
  const ext = path.extname(abs).slice(1).toLowerCase();
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [
      new ImageRun({
        type: ext === "jpg" ? "jpg" : "png",
        data,
        transformation: { width: w, height: h },
        altText: { name: alt, description: alt, title: alt },
      }),
    ],
  });
}

async function main() {
  const children = [];

  // ----- Title block -----
  children.push(
    new Paragraph({
      spacing: { after: 80 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 20, color: NAVY, space: 8 } },
      children: [r("CAPSTONE PROGRESS REPORT", { size: 20, bold: true, color: NAVY })],
    }),
    new Paragraph({
      spacing: { before: 200, after: 80 },
      children: [r("From Per-Sensor CAT Trackers to a Unified Hybrid Correlator", { size: 40, bold: true, color: NAVY })],
    }),
    p("An AI / ML tracker-correlator for multi-radar air-traffic surveillance: problem, architecture lineage (V3–V8), and what the numbers actually say.", { size: 22, italics: true, color: MUTED, align: AlignmentType.LEFT, after: 240 }),
    rich([
      r("Author: ", { bold: true }), r("Tom Rathbun"),
      r("    ·    "),
      r("Project: ", { bold: true }), r("AI Tracker Correlator"),
      r("    ·    "),
      r("Date: ", { bold: true }), r("16 August 2026"),
    ], { align: AlignmentType.LEFT, after: 80 }),
    rich([
      r("Status: ", { bold: true }),
      r("Research pause for write-up. Hybrid-MLP remains the operational default. V8 transformer is a drop-in associator, not a replacement tracker."),
    ], { align: AlignmentType.LEFT, after: 320 }),
  );

  // ----- 1. Executive summary -----
  children.push(
    h1("1.  Executive summary"),
    p("Today’s surveillance stack still looks like the 1990s: each radar runs its own physics tracker on one ASTERIX category (typically CAT-048 plots or CAT-062 system tracks), then a separate correlator tries to decide which of those local tracks are the same aircraft. That split creates latency, duplicate tracks, and a maintenance surface that grows with every sensor."),
    p("This project asked whether one AI/ML system can ingest multi-sensor plots, reject clutter, associate PSR with SSR, and output a single correlated track picture. The short answer is yes — but not by handing the entire job to one network."),
    p("The working system is Hybrid: a clutter MLP, two pairwise association MLPs, Hungarian assignment, and an asynchronous Kalman filter. On the dense 120-second multi-radar stream it holds MOTA 0.865 with zero ID switches. On a 30-minute Sweden holdout it reaches MOTA 0.971. End-to-end learned trackers (GNN V3–V6 and the V7 transformer) did not. They flooded the picture with false tracks."),
    p("V8 is the course correction. It is not a new tracker. It is a transformer that scores gated pairs inside Hybrid — looking at the surrounding traffic, not just one pair in isolation. Pure V8 is not ready to ship. Averaging it with the MLP (ensemble) slightly beats Hybrid on the dense sim stream (MOTA 0.872 vs 0.865) and ties Hybrid on Sweden. That is the research result, not the operational default."),
    p("Hybrid has the best standalone numbers on the data we have. The reason to keep the transformer is the next environment: crossings, SSR dropouts, and overlapping trails where a pair looked at in isolation is ambiguous, and the set around it is not."),
  );

  // ----- 2. The problem -----
  children.push(
    h1("2.  The operational problem"),
    h2("2.1  One CAT, one tracker, one correlator"),
    p("A conventional multi-radar site is not one tracker. It is a fleet of them. Each sensor decodes its own ASTERIX category — CAT-034 status, CAT-048 plots, or CAT-062 tracks already formed by a local tracker — and runs a Kalman-style filter on that stream alone. Only after those local tracks exist does a correlator try to merge duplicates."),
    p("That architecture is interpretable. It is also the source of the failure modes this project exists to fix:"),
  );
  children.push(
    bullet("Duplicate tracks when two radars see the same aircraft and the correlator is late or conservative."),
    bullet("Broken identity when an SSR squawk drops and the PSR-only tracker cannot hold the ID."),
    bullet("Temporal dragging when asynchronous scans (5.5–9 s) are forced into a common window."),
    bullet("Clutter leakage from PSR, which the local tracker was never trained to share with the correlator."),
    bullet("A growing integration tax: every new radar is another tracker to tune and another input to the correlator."),
  );
  children.push(
    h2("2.2  The research question"),
    p("Can a single process ingest decoded Cartesian plots from every radar, perform clutter rejection, data association, and state estimation, and emit one correlated track per aircraft — without a per-sensor tracker and without a downstream correlator?"),
    p("Feasibility, not a fielded product, is the bar. Metrics are MOTA (accuracy, penalizing false tracks and misses), MOTP (position error), precision, recall, and ID switches. A system that swaps identities is unusable for air traffic, no matter how good MOTA looks."),
  );

  // ----- 3. Hybrid -----
  children.push(
    h1("3.  The Hybrid pipeline: what actually works"),
    p("Hybrid is the system we would run tomorrow. It is hybrid in the strict sense: learned modules do association; physics owns time and kinematics."),
  );
  children.push(
    bullet("Clutter MLP. A small unary network rejects PSR false alarms before association."),
    bullet("Spatial cluster (2 km). Pairwise MLPs decide whether two plots in the same sweep are the same aircraft (typical PSR + SSR pair). Connected components fuse them into one meta-measurement."),
    bullet("Temporal assign (8 km). Each live track is projected to the measurement time (dt = meas_t − track_t). Pairwise MLPs score the gated pairs. Hungarian assignment enforces one-to-one."),
    bullet("Async CV Kalman. The filter updates at the measurement’s own timestamp. That is what killed temporal dragging."),
    bullet("M/N manager. Three hits to confirm, about ten coasts through a radar shadow, then delete."),
  );
  children.push(
    p("Two specialist MLPs do the learned work. PSR–PSR uses six kinematic features (distance, velocity cosine, speed difference, Δaz, Δel, amplitude). SSR–ANY uses four (distance, Δaz, Mode-3A match, Mode-S match). Each pair is scored alone. Hungarian and the Kalman filter never see a neural state vector."),
    img("artifacts/tracker_simulation_holdout_2min_trails.png", 620, 420, "Hybrid tracker simulation with confirmed tracks and trails"),
    caption("Figure 1. Hybrid correlator on a streaming holdout: clutter, measurements, ground truth, and confirmed tracks with trails."),
  );

  // ----- 4. MLP vs Transformer (the definition they asked for) -----
  children.push(
    h1("4.  MLP vs transformer: what each one actually sees"),
    p("This distinction is the rest of the story. Both models answer the same question — “are these two things the same aircraft?” — and they live in the same Hybrid pipeline. They do not see the same world."),
    h2("4.1  The MLP looks at one pair"),
    p("The pairwise MLP is a specialist that never looks up from the two objects in front of it. Give it plot A and plot B (or track A and plot B). It computes a handful of hand-built features — how far apart they are, whether their velocities agree, whether the squawks match — and outputs a probability. If a third aircraft is crossing 3 km away, the MLP does not know. If six other SSR plots share a similar Mode-3A, it does not know. Isolation is why it is precise: it only fires when the pair itself looks right."),
    h2("4.2  The transformer looks at the pair and the traffic around it"),
    p("The V8 transformer is a set matcher, not a tracker. It turns every gated track and every gated plot into a token (position, velocity, identity embeddings, sensor, role). It runs self-attention inside the track set and inside the measurement set, so token A can change because C is nearby. Then it scores the pair with both contextual embeddings and the same kind of geometry/identity features the MLP uses."),
    p("In a crossing, the MLP sees two close plots and may say “same.” The transformer can see that two established tracks are already claiming those plots, and that a third measurement is a better fit. That is the capability Hybrid does not have today, and the reason the transformer is still worth the research even when Hybrid wins the current scoreboard."),
  );

  const cmpW = [2200, 3940, 3940];
  children.push(
    table(
      ["", "Pairwise MLP", "V8 transformer"],
      [
        ["Question", "Same aircraft?", "Same aircraft?"],
        ["Sees", "Only this pair", "This pair and the other gated tracks / plots"],
        ["Features", "4–6 hand numbers", "Tokens + 12-d pair geometry / identity"],
        ["Identity", "Match flags only (+1 / 0 / −1)", "Embeds Mode-3A and hashed Mode-S"],
        ["Specialists", "Two nets (PSR vs SSR)", "One net, type / role embeddings"],
        ["Uniqueness", "Hungarian after the fact", "Hungarian; optional unmatched (dustbin) score"],
        ["Strength", "High precision, 0 ID switches", "Set context in crossings and dropouts"],
        ["Weakness", "Blind to the rest of the scene", "Can over-associate; extra false tracks"],
      ],
      cmpW
    ),
    caption("Table 1. Scorer vs scorer. Both sit inside Hybrid. Neither replaces the Kalman filter."),
    p("Ensemble is not a third architecture. It is the average of the two probabilities (0.5 MLP + 0.5 V8). The MLP vetoes reckless transformer edges; the transformer still nudges the cases where the pair alone is ambiguous. That mix is the best V8-related result we have."),
  );

  // ----- 5. Lineage -----
  children.push(
    h1("5.  How we got here: V3 through V8"),
    p("The project did not start at Hybrid. It started with the bet in the research proposal: one learned recurrent network would replace the CAT tracker and the correlator. Each version taught us something. Most of those lessons were negative, and they are the reason Hybrid exists."),
    h2("5.1  V3 — recurrent graph attention"),
    p("V3 (RecurrentGATTracker) put tracks and measurements in one graph, ran GATv2 attention over spatially gated edges, and carried a GRU hidden state per track. It predicted a residual state update and an existence probability. On aligned batch frames it looked plausible. On asynchronous streaming data the GRU decayed whenever a 2-second window contained no hit for that track. Kalman coasts; a starved GRU forgets."),
    h2("5.2  V4 — fusion and learnable edges"),
    p("V4 kept the GAT + GRU skeleton and pushed more of the association into learned edge features (kinematics plus dt). It was an incremental attempt to stop hand-building the graph. Streaming MOTA stayed negative. The network still owned initiation, coast, and state — three jobs it had not earned."),
    h2("5.3  V5 — one factory, synchronized gating"),
    p("V5 unified the model factory and the forward pass so every experiment spoke the same language. Attention gating was synchronized. This was an engineering win (reproducible pipeline, zero-track MotA bugs fixed) and not an accuracy win. The architecture was still “the net is the tracker.”"),
    h2("5.4  V6 — bipartite cross-attention and early clutter"),
    p("V6 stopped measurements attending to measurements (a source of ghost tracks) and treated tracks as queries and measurements as keys/values. A dedicated clutter head hard-dropped junk before attention. That was the right inductive bias — and still not enough. On the same 120-second stream as Hybrid, V6 GNN recorded MOTA −0.70, precision 0.005, recall 0.004. Soft attention is not a substitute for one-to-one assignment or a process model."),
    h2("5.5  V7 — a transformer that tried to be the tracker"),
    p("V7 replaced GAT with a pure transformer: measurement self-attention, track–meas cross-attention, residual Δs, existence heads, GRU memory. Holdout MOTA stayed negative (best −1.09, default −3.03) with tens to hundreds of ID switches. The failure mode was a false-track flood. The net initiated anything with a high existence logit and had no Hungarian uniqueness. V7 is closed. We do not retry it."),
    h2("5.6  V8 — transformer as a pair scorer only"),
    p("V8 is the reaction to V7. The transformer scores gated pairs. The Kalman filter owns state. Hungarian owns uniqueness. Hybrid’s time model stays. Two call sites change: spatial cluster (score_pairs) and track–plot assign (score_assignment). There is no V8 tracker class and no factory key “v8.” Default CLI remains Hybrid-MLP (--assoc mlp). Transformer is opt-in (--assoc transformer). Ensemble averages the two (--assoc ensemble)."),
  );

  const linW = [1400, 3200, 2200, 3280];
  children.push(
    table(
      ["Version", "Idea", "Owns state?", "Outcome"],
      [
        ["V3", "GAT + GRU, residual Δs, existence", "Yes", "Fails on async streams"],
        ["V4", "Learned edge / fusion features", "Yes", "Still a learned tracker"],
        ["V5", "Unified factory and gating", "Yes", "Engineering, not MOTA"],
        ["V6", "Bipartite track→meas attention", "Yes", "MOTA −0.70 on stream_radar"],
        ["V7", "Full transformer tracker", "Yes", "MOTA −1.1 to −3.0; ID floods"],
        ["Hybrid", "MLP pairs + async KF + Hungarian", "Kalman", "MOTA 0.87 / 0.97; 0 IDs"],
        ["V8", "Transformer scores pairs inside Hybrid", "Kalman", "Recall up, precision down"],
        ["Ensemble", "0.5 MLP + 0.5 V8, same Hybrid", "Kalman", "Slight MOTA win on dense sim"],
      ],
      linW
    ),
    caption("Table 2. Lineage. The turning point is V8: the network stops owning time and identity uniqueness."),
  );

  // ----- 6. Results -----
  children.push(
    h1("6.  What we measured"),
    h2("6.1  Protocol"),
    p("Unless noted, numbers below use the same contract: 1-second evaluation windows, min_hits = 3, max_age = 10, match threshold 7 km. The dense stream is data/stream_radar_001.jsonl (~126 s, five asynchronous radars, 246 truth IDs). The real-traffic holdout is data/canonical/stream_sweden_30min_holdout.jsonl (~30 min, 163 IDs, built from CAT-062 traffic through a multi-radar observation model). V6/V7 rows on the dense stream are from the earlier streaming campaign on that same file."),
    h2("6.2  Dense multi-radar stream"),
  );

  const resW = [2400, 1400, 1600, 1560, 1560, 1560];
  children.push(
    table(
      ["System", "MOTA", "MOTP", "Prec.", "Recall", "IDs"],
      [
        ["Hybrid-MLP (default)", "0.865", "877 m", "0.929", "0.937", "0"],
        ["V6 GNN tracker", "−0.70", "3240 m", "0.005", "0.004", "high"],
        ["V7 transformer (best / default)", "−1.09 / −3.03", "~3.3 km", "0.05 / 0.11", "0.06 / 0.37", "29 / 546"],
        ["V8 transformer only", "0.526", "1056 m", "0.692", "0.946", "0"],
        ["MLP cluster + V8 assign", "0.845", "995 m", "0.916", "0.929", "0"],
        ["Ensemble (Sweden-tuned V8)", "0.872", "901 m", "0.931", "0.941", "0"],
      ],
      resW
    ),
    caption("Table 3. stream_radar_001. Hybrid is the first system that is actually a tracker. Ensemble is a small, honest gain on top of it."),
    h2("6.3  Sweden 30-minute holdout"),
  );
  children.push(
    table(
      ["System", "MOTA", "MOTP", "Prec.", "Recall", "IDs"],
      [
        ["Hybrid-MLP", "0.971", "105 m", "0.979", "0.992", "0"],
        ["V8 transformer only", "0.745", "114 m", "0.801", "0.991", "0"],
        ["Ensemble", "0.971", "105 m", "0.979", "0.992", "0"],
      ],
      resW
    ),
    caption("Table 4. Sweden holdout. Hybrid is already near the ceiling. Ensemble ties it. Pure V8 again loses precision."),
    img("artifacts/tracks_sweden_30min_holdout.png", 620, 340, "Sweden 30-minute holdout track picture"),
    caption("Figure 2. Sweden 30-minute holdout traffic used for the Hybrid / V8 / ensemble comparison."),
    h2("6.4  How to read this"),
    p("Pure V8 does not ship. It keeps zero ID switches — Hungarian is doing its job — and it slightly raises recall on the dense stream. Precision collapses because it starts extra tracks. Ablations showed the leak is mostly in clustering: the transformer draws edges the MLP would refuse. Giving clustering back to the MLP (split) recovered most of the precision. Raising the transformer’s cluster threshold made things worse (under-merge, then two tracks for one aircraft)."),
    p("Dustbin — an extra Hungarian column so a track can refuse every plot — did nothing with the current weights. The unmatched head is not calibrated."),
    p("Training longer on the 16-ID sim set was the wrong next step. Loss was still falling because the net was memorizing easy pairs. Early-stopping on pair precision made it too timid; early-stopping on pair F1 kept a better epoch and still did not beat Hybrid as a standalone scorer. The useful training signal is denser, more overlapping traffic, not more epochs on sim_hetero_001."),
  );

  // ----- 7. Why transformer still -----
  children.push(
    h1("7.  Why the transformer is still the next architecture"),
    p("Hybrid wins the tables we can fill in today. That is not the same as Hybrid being the last word on association."),
    p("The Sweden holdout is not an easy dataset in duration — 30 minutes, 163 tracks — but geometrically it is medium. Median nearest-neighbor distance is about 32 km. Concurrent traffic tops out around 32 aircraft. Crossings inside the 2 km / 8 km gates are uncommon. A pair-only MLP is in its comfort zone: the pair features already separate “same” from “other.” There is little for set attention to do, and a lot of ways for it to invent extra edges."),
    p("The dense sim stream is harder (many more IDs in two minutes, five unsynchronized radars). Ensemble’s small MOTA and recall gain there is the first quantitative hint that set context helps when pairs get crowded. It is not yet a crossing study."),
    p("The environment that will need the transformer is the one Hybrid’s own literature flags: terminal-area overlaps, formation or parallel tracks inside the spatial gate, SSR dropouts at the moment two aircraft cross, and multi-radar bias that makes raw distance lie. In those frames the MLP is looking at a pair that looks legal, and the transformer is the only module we have that can see the other claimants."),
    p("That is why V8 was designed as a drop-in scorer rather than V7-again. We can keep Hybrid’s Kalman, gates, and Hungarian — the parts that produced zero ID switches — and still grow a set model behind them. Ensemble is the staging mode: Hybrid stays in charge, the transformer is allowed to vote."),
  );

  // ----- 8. Recommendations -----
  children.push(
    h1("8.  What we ship, and what we do next"),
    h2("8.1  For this submission"),
  );
  children.push(
    bullet("Operational default: Hybrid-MLP (--mode hybrid --assoc mlp). It is trained, measured, and stable on sim and Sweden."),
    bullet("Research path: Hybrid + ensemble with the Sweden-tuned V8 checkpoint. Slight dense-sim gain, Sweden tie, zero ID switches."),
    bullet("Do not run --assoc transformer alone in an operational eval. It under-clusters or over-clusters depending on threshold and floods false tracks."),
    bullet("Do not reopen V7. A transformer that owns initiation and state has already failed this problem."),
  );
  children.push(
    h2("8.2  After the pause"),
  );
  children.push(
    bullet("Build a crossing / overlap holdout (UAE density or a Sweden slice with nearest-neighbor < 5 km) and re-run Hybrid vs ensemble vs split. That is the experiment that can justify the transformer, not another epoch on sim."),
    bullet("Keep MLP on clustering; let V8 score assignment only. That split already recovered most precision without a new net."),
    bullet("Calibrate V8 (temperature / assign threshold) before turning dustbin back on."),
    bullet("Early-stop training on pair F1, not precision, and train on Hybrid’s actual Kalman states rather than the previous plot."),
  );

  // ----- 9. Close -----
  children.push(
    h1("9.  Closing"),
    p("We set out to replace a stack of single-CAT trackers and a correlator with one system. We tried to make that system a single network. The networks that owned everything (V3–V7) failed in the same way: they could not be trusted with birth, death, and time. Hybrid succeeded because it refused those jobs to the net."),
    p("V8 is how the net comes back — as a pair scorer that can see the traffic around the pair. Ensemble is the first time that idea helps the scoreboard without breaking identity. Hybrid still has the highest standalone results. The transformer is the architecture we will need when the pairs start to lie."),
    p("Research is paused here so this story can be submitted. The code path is on main: Hybrid by default, V8 behind --assoc, ensemble as the blend."),
  );

  const doc = new Document({
    creator: "Tom Rathbun",
    title: "Capstone Progress Report — AI Tracker Correlator",
    description: "From per-sensor CAT trackers to Hybrid, V3–V8, and the MLP / transformer ensemble.",
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: "Arial", size: 32, bold: true, color: NAVY },
          paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: "Arial", size: 26, bold: true, color: STEEL },
          paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: "Arial", size: 24, bold: true, color: "334155" },
          paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              tabStops: [{ type: TabStopType.RIGHT, position: CONTENT }],
              border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY, space: 6 } },
              spacing: { after: 120 },
              children: [
                r("AI Tracker Correlator", { size: 16, bold: true, color: NAVY }),
                r("\tCapstone progress report", { size: 16, color: MUTED }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              tabStops: [{ type: TabStopType.RIGHT, position: CONTENT }],
              border: { top: { style: BorderStyle.SINGLE, size: 6, color: LINE, space: 6 } },
              spacing: { before: 80 },
              children: [
                r("Tom Rathbun  ·  16 August 2026  ·  Not for operational use", { size: 16, color: MUTED }),
                r("\t"),
                r("Page ", { size: 16, color: MUTED }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: MUTED }),
              ],
            }),
          ],
        }),
      },
      children,
    }],
  });

  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
