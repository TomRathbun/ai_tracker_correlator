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
    "artifacts/tracking_visualization.png": [1600, 900],
    "artifacts/architecture_hybrid_v8.png": [2412, 1728],
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
    p("This project asked whether one AI/ML system can ingest multi-sensor plots, reject clutter, associate PSR with SSR, and output a single correlated track picture. The short answer is yes — but not by handing the entire job to one network. The learned part is association. Section 4 is the deep dive on those two associators."),
    p("The working system is Hybrid: a clutter MLP, two pairwise association MLPs, Hungarian assignment, and an asynchronous Kalman filter. On the dense 120-second multi-radar stream, under the max_age = 10 eval contract in §6.1, it holds MOTA 0.865 with zero ID switches. On a 30-minute Sweden holdout it reaches MOTA 0.971. End-to-end learned trackers (GNN V3–V6 and the V7 transformer) did not. They flooded the picture with false tracks."),
    p("V8 is the course correction. It is not a new tracker. It is a transformer that scores gated pairs inside Hybrid — looking at the surrounding traffic, not just one pair in isolation. Pure V8 is not ready to ship. Averaging it with the MLP (ensemble) is a single-run +0.007 MOTA on the dense sim stream (0.872 vs 0.865) with zero ID switches, and it ties Hybrid on Sweden. That is a hint, not a claim that the transformer beats Hybrid, and not the operational default."),
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
    p("Two specialist MLPs do the learned work today. Each pair is scored alone. Hungarian and the Kalman filter never see a neural state vector. Section 4 specifies those MLPs — features, training, and the transformer that can replace or average them."),
    img("artifacts/architecture_hybrid_v8.png", 640, 460, "Hybrid pipeline and V8 associator architecture"),
    caption("Figure 1. How the model works. Top: Hybrid pipeline. Purple block is the only neural scorer — MLP, V8, ensemble, or split. Kalman, gates, Hungarian, and M/N stay classical. Bottom: V8 token path. Self-attention is within tracks and within plots; geometry stays in rel_ij. The net never predicts state or existence."),
    img("artifacts/tracking_visualization.png", 620, 360, "Hybrid tracker simulation with confirmed tracks and trails"),
    caption("Figure 2. Hybrid correlator on a streaming holdout: clutter, measurements, ground truth, and confirmed tracks with trails."),
  );

  // ----- 4. AI/ML deep dive -----
  children.push(
    h1("4.  The AI/ML core: pairwise MLP vs association transformer"),
    p("Everything that is actually machine learning in the shipping system lives here. The Kalman filter, the 2 km / 8 km gates, Hungarian assignment, and the M/N track manager are classical. The learned question is only: given two gated objects, how likely are they the same aircraft? Two different networks answer that question. They share a pipeline and a loss family. They do not share a worldview."),
    p("Say it once, then unpack it. The MLP looks at one pair. The transformer looks at that pair and the traffic around it. That is the entire AI/ML distinction that matters going forward."),

    h2("4.1  What both models are allowed to do — and forbidden to do"),
    p("Both models emit a logit. Callers apply a sigmoid to get p ∈ [0, 1]. Hybrid turns p into a graph edge (cluster if p > 0.5) or a Hungarian cost (cost = 1 − p). Neither model predicts position, velocity, existence, or a next-step residual. Neither carries hidden state across time. Time is the Kalman filter’s job. Uniqueness is Hungarian’s job. Those constraints are why V8 is an associator and V7 was a failed tracker."),
    p("Supervision is the same label for both: track_id equality. A pair is positive if both objects share a real track_id (not clutter, not −1). The networks never see track_id as an input feature. Identity, if used, comes only from Mode-3A / Mode-S fields that an operational plot would actually carry."),

    h2("4.2  Pairwise MLPs — a specialist that never looks up"),
    h3("Job and inductive bias"),
    p("The pairwise MLP is a feed-forward classifier on a hand-built feature vector of one pair. It is deliberately local. If aircraft C is crossing three kilometers away, or if six other SSR plots share a similar squawk, the MLP does not know. Isolation is the bias: only fire when this pair’s own kinematics and identity look right. That is why Hybrid is precise, and why it can miss a crossing that is only obvious from the rest of the set."),
    h3("Two specialists, not one net"),
    p("Heterogeneous radar forced a split. Primary plots have Doppler velocity and amplitude, and no reliable identity. Secondary plots have Mode-3A (12-bit squawk) and Mode-S (ICAO address), and often no velocity. A single feature vector that pretends both sides always have both kinds of information trains poorly. Hybrid therefore loads two small networks from checkpoints/pairwise_psr_psr.pt and checkpoints/pairwise_ssr_any.pt."),
  );

  children.push(
    table(
      ["Network", "When used", "Input dim", "Architecture"],
      [
        ["PSR–PSR", "Both objects are primary", "6", "6 → 64 → 32 → 1, ReLU, dropout 0.2"],
        ["SSR–ANY", "Either object is secondary", "4", "4 → 64 → 32 → 1, ReLU, dropout 0.2"],
      ],
      [1800, 2800, 1400, 4080]
    ),
    caption("Table 1. Dual pairwise MLPs (src/pairwise_classifier.py). Each is a few thousand parameters."),
    h3("PSR–PSR features (kinematics + amplitude)"),
    p("Both sides are assumed to have velocity. Features are normalized so raw meters never enter the first linear layer."),
  );
  children.push(
    table(
      ["#", "Feature", "Scale / encoding", "Why it is there"],
      [
        ["1", "Position distance", "‖p1 − p2‖ / 1e5", "Same-aircraft plots sit inside ~2 km"],
        ["2", "Velocity cosine", "v1·v2 / (|v1||v2|)", "Co-located PSR hits share heading"],
        ["3", "Speed difference", "| |v1| − |v2| | / 1e3", "Rejects crossing traffic at similar range"],
        ["4", "Azimuth separation", "wrapped |az1 − az2|", "Angular gate in radar coordinates"],
        ["5", "Elevation separation", "|el1 − el2|", "Separates stacked traffic"],
        ["6", "Amplitude difference", "|amp1 − amp2| / 100", "PSR-only; similar RCS / range"],
      ],
      [700, 2400, 2800, 4180]
    ),
    caption("Table 2. compute_psr_psr_features. Six numbers, one pair, no identity."),
    h3("SSR–ANY features (geometry + identity flags)"),
    p("Used for PSR–SSR fusion and SSR–SSR pairs. Identity is not the raw squawk. It is a three-way flag so a missing code is not confused with squawk 0000."),
  );
  children.push(
    table(
      ["#", "Feature", "Encoding", "Why it is there"],
      [
        ["1", "Position distance", "‖p1 − p2‖ / 1e5", "Still the spatial prior"],
        ["2", "Azimuth separation", "wrapped |az1 − az2|", "Works when velocity is missing"],
        ["3", "Mode-3A match", "+1 match / −1 mismatch / 0 missing", "Squawk is first-class SSR cue"],
        ["4", "Mode-S match", "+1 / −1 / 0", "ICAO address; stronger than squawk"],
      ],
      [700, 2400, 3400, 3580]
    ),
    caption("Table 3. compute_ssr_any_features. An ablation (--no-identity-features) zeros channels 3–4 so association is kinematics-only; dimension stays 4 so pretrained weights still load."),
    h3("How the MLP is trained"),
    p("scripts/train_hetero_pairwise.py extracts every specialized pair in each training frame. Label is 1 if track_ids match and are not clutter. Batches are 512 independent pairs — there is no sequence and no set. Optimizer is Adam at 1e-3. Loss is binary cross-entropy with a pos_weight = n_neg / n_pos so the rare true pairs are not drowned by easy far negatives. The network never sees a third object in the same forward pass. After sigmoid, Hybrid treats p > 0.5 as a cluster edge and cost = 1 − p as a Hungarian entry."),
    h3("What that means in a crossing"),
    p("Two aircraft cross inside the 8 km gate. The MLP scores track A vs plot B using only (A, B). If distance is small and headings are briefly similar, p can be high even though track C is the rightful owner. Hungarian may still pick the globally cheapest assignment — but only from those independent scores. The MLP cannot say “B looks good for A until I notice C is closer and already carries this Mode-S.” That sentence requires looking around."),

    h2("4.3  V8 transformer — a set matcher, not a tracker"),
    h3("Job and inductive bias"),
    p("AssociationTransformerV8 (src/model_v8_associator.py, ~150–250k parameters) is SuperGlue-style matching, not DETR-style tracking. It builds a token per gated track and per gated plot, contextualizes each side with self-attention, then scores pairs with an MLP head on [h_i ; h_j ; rel_ij]. The transformer is the context encoder. The last layer is still an MLP. The difference is what h_i contains: after attention, h_i knows about the other tokens on its side of the gate."),
    h3("Token: 15 numbers plus embeddings"),
    p("Raw meters are never fed in. Numeric features are normalized, then projected Linear(15 → 64). Five embeddings, each 64-d, are added: role (track vs measurement), type (PSR vs SSR), sensor id (0–8), Mode-3A (0–4095), and hashed Mode-S (1024 buckets). Missing identity uses pad index 0. Present squawk 0000 collides with that pad; the numeric has_mode_3a flag is the disambiguator, not a separate unused pad token."),
  );
  children.push(
    table(
      ["Block", "Dim", "Scale / encoding", "Notes"],
      [
        ["x, y, z", "3", "/ 1e5", "Same scale as pairwise MLPs"],
        ["vx, vy, vz", "3", "/ 1e3; 0 if missing", "SSR often has no Doppler"],
        ["has_vx, has_vy, has_vz", "3", "{0, 1}", "Stops a missing vel looking like 0 m/s"],
        ["amplitude, has_amp", "2", "amp / 100", "PSR cue; SSR usually missing"],
        ["age, hits", "2", "min(·,20)/20; 0 on plots", "Track maturity; 0 on measurements"],
        ["dt", "1", "seconds", "0 inside a cluster; track already projected for assign"],
        ["has_mode_3a", "1", "{0, 1}", "Numeric companion to the embed"],
      ],
      [2400, 900, 3200, 3580]
    ),
    caption("Table 4. Numeric token (15-d) before Linear(15, 64). Embeddings for role, type, sensor, Mode-3A, and Mode-S are added after the projection."),
    h3("Relative pair features rel_ij (12-d)"),
    p("Attention is not asked to rediscover geometry. The score head always concatenates an explicit 12-d pair vector. This is the inductive bias V7 promised and never built: the transformer sees context; the head still sees physics."),
  );
  children.push(
    table(
      ["Group", "Channels", "Detail"],
      [
        ["Geometry", "dx, dy, dz, dist / 1e5", "Same spatial language as the MLP"],
        ["Velocity", "Δ|v| / 1e3, cos_vel", "cos_vel = 0 if either velocity is missing — no fake 0-vector match"],
        ["Angles", "Δaz, Δel", "Radar-native separation"],
        ["Time", "dt", "Track already projected to meas_t"],
        ["Identity", "Mode-3A match, Mode-S match", "+1 / 0 / −1, same convention as SSR–ANY"],
        ["Sensor", "same_sensor ∈ {0,1}", "Two plots from one radar vs two radars"],
      ],
      [1800, 3600, 4680]
    ),
    caption("Table 5. rel_ij. The transformer is not allowed to ignore the pair geometry the MLP already uses."),
    h3("Self-attention"),
    p("Two pre-norm TransformerEncoder layers, d_model = 64, 4 heads (16-d each), FFN 256, GELU, dropout 0.1. Attention runs within the track set and within the measurement set separately. There is no cross-attention in v1 — rel_ij already carries the geometry between sides. Optional later: one cross-attention block after self-attention, only if an ablation shows set context without identity does nothing."),
    p("Hard gates stay in front of the net. V8 never scores a 50 km pair. That was V7’s max_assoc_m = 50,000 mistake. Clustering is usually a tiny 2 km clique. Assignment is the set problem: tens of tracks times tens of metas, typically N < 80, no padding required at inference."),
    h3("Two heads, two call signatures, same weights"),
    p("score_pairs(left, right, pair_index) returns a (P,) logit vector for the gated meas–meas pairs Hybrid already enumerated. Used by _spatial_cluster."),
    p("score_assignment(tracks, metas) returns S ∈ R^{T×M} and a dustbin vector ∈ R^T. Used by _associate. Dustbin is the unmatched score: “this track owns none of these plots.” Softmax over competitors is not required at inference. Hungarian can take an extra column cost[:, dust] = 1 − σ(dustbin). A coasting track in a radar shadow can choose that column instead of stealing a neighbor. With current weights the dustbin column is unused — the head is not calibrated — but the API is there."),
    p("Logits leave the module unsigned. Hybrid applies sigmoid. That keeps training and eval on the same numeric scale as the MLPs."),
    h3("How the transformer is trained"),
    p("src/train_associator_v8.py is supervised matching, not train_streaming_v7. There is no residual loss, no existence logit, no GRU, no Hungarian-on-state. Windows are 2 s. Two tasks share weights:"),
  );
  children.push(
    bullet("Cluster task. All 2 km gated plot–plot pairs in the window. Label 1 if same track_id and not clutter."),
    bullet("Assign task. Teacher-forced “tracks” are the last plot of each live id, time-projected to each candidate. Label 1 if ids match and distance < 8 km. A track with no true in-gate plot is a dustbin positive."),
  );
  children.push(
    p("Split is by track id (seed 42, 80/20), not by time, so holdout aircraft are unseen. Loss is focal BCE on positives and gated negatives, plus dustbin BCE, plus a light 0.1 entropy term so assignment rows peak. Negatives are capped at about 8× positives. Class mass is balanced inside each batch (real pos_weight), unlike the first V8 run, which used a constant α and overfit 16 sim IDs. Early-stop is on holdout pair F1, not train loss and not precision-only (precision-only produced a timid net that refused true pairs). AdamW 1e-3, weight decay 1e-4, grad clip 1.0."),
    p("A later fine-tune on the Sweden 30-minute train stream, from the sim checkpoint, is the weight file used in ensemble eval (checkpoints/model_v8_assoc_sweden_best.pt)."),
    h3("What that means in a crossing"),
    p("The same two aircraft cross. After self-attention, track A’s token has mixed with track C’s token: both are live, close, and claiming plots. Plot B’s token has mixed with the other plots in the gate, including the one that actually carries C’s Mode-S. The score head then sees [h_A ; h_B ; rel_AB]. rel_AB still says “close.” The intended mechanism is that h_A and h_B can now carry “C is a better owner.” We have not shown an attention map or a crossing slice where V8 flips a pair the MLP got wrong. Until that overlap holdout exists, it is a hypothesis with a single-run +0.007 MOTA hint, not a demonstrated capability."),

    h2("4.4  Side by side"),
  );

  const cmpW = [2200, 3940, 3940];
  children.push(
    table(
      ["", "Pairwise MLP", "V8 transformer"],
      [
        ["Question", "Same aircraft?", "Same aircraft?"],
        ["Sees", "Only this pair", "This pair and the other gated tracks / plots"],
        ["Forward pass", "One 4–6-d vector", "Set of tokens, then one pair head"],
        ["Parameters", "A few thousand × 2 nets", "~180k (d=64, 2 layers, 4 heads)"],
        ["Identity", "Match flags only", "Flags in rel_ij plus Mode-3A / Mode-S embeds"],
        ["Specialists", "PSR–PSR and SSR–ANY", "One net, type / role embeds"],
        ["Time", "Uses Hybrid’s projected dict", "Same — must consume tmp_t, never raw KF state"],
        ["Training", "i.i.d. pairs, weighted BCE", "Windowed sets, focal BCE + dustbin + entropy"],
        ["Uniqueness", "Hungarian after the fact", "Hungarian; optional dustbin column"],
        ["Strength", "High precision, 0 ID switches", "Set context in crossings and dropouts"],
        ["Weakness", "Blind to the rest of the scene", "Over-associates; extra false tracks if used alone"],
      ],
      cmpW
    ),
    caption("Table 6. Both sit inside Hybrid. Neither replaces the Kalman filter. The AI/ML difference is pairwise isolation versus set context."),

    h2("4.5  Ensemble and split — composing the two learners"),
    p("Ensemble is not a third network. At each gated pair Hybrid computes p = 0.5 p_MLP + 0.5 p_V8, then uses that p exactly as it would use a single scorer. The MLP vetoes reckless transformer edges; the transformer still nudges pairs the MLP cannot see in context. On the dense stream that mix is a single-run +0.007 MOTA (0.872 vs 0.865) with zero ID switches. On Sweden the three reported decimals are identical to Hybrid — consistent with V8 not changing those decisions. Dustbin is off on this path."),
    p("Split scoring is the other composition: MLP owns clustering (the local 2 km clique the specialists already solve), V8 owns assignment (the set problem). That recovered most of the precision pure V8 lost, without training a new net. Raising V8’s own cluster threshold made things worse — the net under-merged PSR+SSR and started two tracks for one aircraft. The lesson for the AI/ML design is sharp: give the transformer the job that needs set context, not the job the pairwise MLP already does well."),

    h2("4.6  Why this section is the project’s ML claim"),
    p("The feasibility question was never “can a Kalman filter track an airplane.” It was whether a learned associator can replace the per-sensor CAT tracker plus correlator without breaking identity. The MLP is the learned associator that already does that, by looking at one pair at a time with physics features. The transformer is the learned associator that can look around — and, on today’s medium-geometry Sweden holdout, does not yet need to. The next hard environment (overlaps, dropouts, crossings inside the gate) is exactly the environment where looking around stops being optional."),
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
        ["V8", "Transformer scores pairs inside Hybrid", "Kalman", "Dense stream: recall up, precision down"],
        ["Ensemble", "0.5 MLP + 0.5 V8, same Hybrid", "Kalman", "Single-run +0.007 MOTA on dense sim"],
      ],
      linW
    ),
    caption("Table 7. Lineage. The turning point is V8: the network stops owning time and identity uniqueness."),
  );

  // ----- 6. Results -----
  children.push(
    h1("6.  What we measured"),
    h2("6.1  Protocol"),
    p("Unless noted, numbers below use the same contract: 1-second evaluation windows, min_hits = 3, max_age = 10, match threshold 7 km. The CLI default is max_age = 2; that shorter coast produces a different Hybrid MOTA (about 0.58 on this file) and is not the table below. The dense stream is data/stream_radar_001.jsonl (~126 s, five asynchronous radars, 246 truth IDs). The real-traffic holdout is data/canonical/stream_sweden_30min_holdout.jsonl (~30 min, 163 IDs). It is CAT-062 traffic passed through a multi-radar observation model, then packed as a 6-tile / 3×-mini stream — duration is real, geometry is repeated, not 30 independent minutes of new crossings. V6/V7 rows on the dense stream are from the earlier streaming campaign on that same file."),
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
    caption("Table 8. stream_radar_001, max_age = 10. Hybrid is the first system that is actually a tracker. Ensemble is a single-run +0.007 MOTA on top of it, not a statistically tested win."),
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
    caption("Table 9. Sweden holdout (tiled CAT-062 observation-model stream). Hybrid is already near the ceiling. Ensemble matches it to three decimals — V8 is not changing those decisions. Pure V8 again loses precision."),
    p("The holdout is the second half of a 6-tile pack of Sweden CAT-062 traffic, regenerated through the multi-radar observation model (see data/canonical/DATA_MANIFEST.md). Median nearest-neighbor distance is about 32 km and concurrent traffic tops out around 32 aircraft, so pairs inside the 2 km / 8 km gates are uncommon. Tiling stretches duration without adding new geometry. Treat Table 9 as a long easy-geometry run, not as a crossing study."),
    img("artifacts/tracks_sweden_30min_holdout.png", 620, 340, "Sweden 30-minute holdout track picture"),
    caption("Figure 3. Sweden 30-minute holdout traffic used for the Hybrid / V8 / ensemble comparison."),
    h2("6.4  Ablations on the dense stream"),
  );
  children.push(
    table(
      ["Variant", "MOTA", "Prec.", "Recall", "IDs"],
      [
        ["V8 only (cluster + assign)", "0.526", "0.692", "0.946", "0"],
        ["MLP cluster + V8 assign (split)", "0.845", "0.916", "0.929", "0"],
        ["Ensemble 0.5 / 0.5", "0.872", "0.931", "0.941", "0"],
        ["V8 + dustbin column", "no gain", "—", "—", "0"],
      ],
      [3600, 1600, 1600, 1640, 1640]
    ),
    caption("Table 10. stream_radar_001 ablations, same max_age = 10 contract as Table 8. Dustbin is implemented but uncalibrated — it did not move the scoreboard. Raising V8’s cluster threshold (not tabulated) under-merged PSR+SSR and birthed two tracks per aircraft."),
    h2("6.5  How to read this"),
    p("Pure V8 does not ship. It keeps zero ID switches — Hungarian is doing its job — and it slightly raises recall on the dense stream. Precision collapses because it starts extra tracks. Table 10 shows the leak is mostly in clustering: the transformer draws edges the MLP would refuse. Giving clustering back to the MLP (split) recovered most of the precision. Raising the transformer’s cluster threshold made things worse (under-merge, then two tracks for one aircraft)."),
    p("Dustbin — an extra Hungarian column so a track can refuse every plot — did nothing with the current weights. The unmatched head is not calibrated."),
    p("Training longer on the 16-ID sim set was the wrong next step. Loss was still falling because the net was memorizing easy pairs. Early-stopping on pair precision made it too timid; early-stopping on pair F1 kept a better epoch and still did not beat Hybrid as a standalone scorer. The useful training signal is denser, more overlapping traffic, not more epochs on sim_hetero_001."),
  );

  // ----- 7. Why transformer still -----
  children.push(
    h1("7.  Why the transformer is still the next architecture"),
    p("Hybrid wins the tables we can fill in today. That is not the same as Hybrid being the last word on association."),
    p("The Sweden holdout is not an easy dataset in duration — 30 minutes, 163 tracks — but geometrically it is medium. Median nearest-neighbor distance is about 32 km. Concurrent traffic tops out around 32 aircraft. Crossings inside the 2 km / 8 km gates are uncommon. A pair-only MLP is in its comfort zone: the pair features already separate “same” from “other.” There is little for set attention to do, and a lot of ways for it to invent extra edges."),
    p("The dense sim stream is harder (many more IDs in two minutes, five unsynchronized radars). Ensemble’s single-run +0.007 MOTA and small recall bump there is the first quantitative hint that set context helps when pairs get crowded. It is not yet a crossing study, and it is not a repeated-seed result."),
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
    bullet("Research path: Hybrid + ensemble with the Sweden-tuned V8 checkpoint. Single-run +0.007 MOTA on the dense stream, Sweden tie to three decimals, zero ID switches."),
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
    p("V8 is how the net comes back — as a pair scorer that can see the traffic around the pair. Ensemble is the first time that idea moves the dense-stream scoreboard without breaking identity, and only by a single-run +0.007 MOTA. Hybrid still has the highest standalone results. The transformer is the architecture we will need when the pairs start to lie."),
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
