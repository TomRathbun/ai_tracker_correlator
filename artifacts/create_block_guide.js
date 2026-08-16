/**
 * Briefing companion to architecture_hybrid_v8.png.
 * One sheet per diagram block: what to say, in/out, every term defined.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, LevelFormat, PageNumber, VerticalAlign, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "Architecture_Block_Guide.docx");

const NAVY = "0B3D5C";
const STEEL = "1F4E79";
const HEADER_BG = "0B3D5C";
const ALT_BG = "E8EEF4";
const LINE = "B0BEC5";
const MUTED = "5A6A75";
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 900;
const CONTENT = PAGE_W - 2 * MARGIN;

const thin = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const borders = { top: thin, bottom: thin, left: thin, right: thin };

function r(text, opts = {}) {
  return new TextRun({
    text,
    font: "Arial",
    size: opts.size || 21,
    bold: !!opts.bold,
    italics: !!opts.italics,
    color: opts.color || "222222",
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 260 },
    alignment: opts.align || AlignmentType.LEFT,
    children: [r(text, opts)],
  });
}
function rich(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 260 },
    alignment: opts.align || AlignmentType.LEFT,
    children: runs,
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 4 } },
    spacing: { before: 280, after: 140 },
    children: [r(text, { size: 30, bold: true, color: NAVY })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [r(text, { size: 24, bold: true, color: STEEL })],
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { before: 40, after: 160 },
    alignment: AlignmentType.CENTER,
    children: [r(text, { size: 17, italics: true, color: MUTED })],
  });
}
function cell(text, width, opts = {}) {
  const fill = opts.header ? HEADER_BG : opts.alt ? ALT_BG : "FFFFFF";
  const color = opts.header ? "FFFFFF" : "222222";
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        children: [r(String(text), { size: opts.size || 17, bold: !!opts.header || !!opts.bold, color })],
      }),
    ],
  });
}
function table(headers, rows, widths) {
  const hdr = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, widths[i], { header: true })),
  });
  const body = rows.map((row, ri) =>
    new TableRow({
      children: row.map((c, i) => cell(c, widths[i], { alt: ri % 2 === 1 })),
    })
  );
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [hdr, ...body],
  });
}

const TW = [2400, 8040]; // term | definition
const FIG = path.join(__dirname, "architecture_hybrid_v8.png");

function block(id, title, say, does, inout, terms) {
  const out = [
    h2(id + "   " + title),
    rich([r("Say this.  ", { bold: true, color: NAVY }), r(say)], { after: 80 }),
    rich([r("What it does.  ", { bold: true, color: NAVY }), r(does)], { after: 80 }),
    rich([r("In / out.  ", { bold: true, color: NAVY }), r(inout)], { after: 100 }),
    table(["Term / acronym", "Meaning in this block"], terms, TW),
    new Paragraph({ spacing: { after: 160 } }),
  ];
  return out;
}

async function main() {
  const children = [];

  children.push(
    new Paragraph({
      spacing: { after: 80 },
      children: [r("ARCHITECTURE BLOCK GUIDE", { size: 36, bold: true, color: NAVY })],
    }),
    p("Companion to Figure 1 of the capstone progress report. Brief from the diagram; this sheet is the script. One heading per box, in reading order: Panel A left-to-right, then the purple scorer, then Panel B.", { size: 21, color: MUTED, after: 80 }),
    p("Color on the figure: teal / green = learned. Amber = physics / classical. Purple = the only neural scoring slot. Grey = inputs.", { size: 21, after: 160 }),
  );

  if (fs.existsSync(FIG)) {
    const data = fs.readFileSync(FIG);
    children.push(
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new ImageRun({ type: "png", data, transformation: { width: 620, height: 444 }, altText: { title: "Architecture", description: "Hybrid + V8 figure", name: "arch" } })],
      }),
      caption("The figure you are briefing. Point at a box, then read that heading below."),
    );
  }

  children.push(
    h1("How to open"),
    p("Three sentences cover the whole figure. Use them before you walk the boxes."),
    p("1.  Hybrid is one process that turns multi-radar plots into one track per aircraft. Learned modules only answer “same aircraft?” Physics owns time, uniqueness, birth, and death."),
    p("2.  The purple block is the only neural net. Today it is two small MLPs. V8 is a transformer we can drop into the same two sockets. Ensemble averages them; split gives clustering to the MLP and assignment to V8."),
    p("3.  V8 is SuperGlue-style matching, not a tracker. It scores gated pairs. It does not predict position, existence, or the next Kalman step. That last sentence is how we killed V7."),
  );

  children.push(
    h1("Pocket glossary  —  say these if someone interrupts"),
  );
  children.push(
    table(
      ["Acronym / term", "Expand, then the one-line meaning"],
      [
        ["PSR", "Primary Surveillance Radar — skin-paint return. Position, often Doppler velocity and amplitude. No identity code."],
        ["SSR", "Secondary Surveillance Radar — transponder reply. Mode-3A squawk and/or Mode-S ICAO address. Often no velocity."],
        ["Plot / measurement", "One decoded radar detection at one time. Cartesian x, y, z plus whatever the sensor gave."],
        ["Track", "A living estimate of one aircraft: Kalman state, last time, hit count, optional identity."],
        ["Meta-measurement", "Fused cluster of plots believed to be the same aircraft in one sweep (typical PSR+SSR pair)."],
        ["MLP", "Multi-Layer Perceptron — a small feed-forward neural net. No memory, no set context."],
        ["V8", "AssociationTransformerV8 — transformer pair scorer inside Hybrid. Not a tracker version number in the factory."],
        ["KF", "Kalman filter — recursive estimator of position/velocity with a process model and a measurement model."],
        ["CV", "Constant velocity — the KF motion model. Aircraft coasts in a straight line between updates."],
        ["Async", "Asynchronous — each update happens at that plot’s own timestamp, not at a common window end."],
        ["dt", "Time delta. Here dt = meas_t − track_t. How far the filter must predict before it can score or update."],
        ["Gate", "Hard distance cutoff. 2 km for clustering, 8 km for assignment. Pairs outside are never scored."],
        ["Hungarian", "linear_sum_assignment — optimal one-to-one matching on a cost matrix. Enforces uniqueness."],
        ["p / logit", "Logit is the raw network output. p = σ(logit) is the probability in [0, 1]. Hybrid uses p, not the logit."],
        ["σ / sigmoid", "Logistic function that maps any real number to (0, 1)."],
        ["M/N", "Hits-to-confirm / coasts-to-delete. We use 3 hits to confirm, about 10 missed scans to delete."],
        ["Coast", "Keep a track alive with KF predict only — no measurement — through a radar shadow."],
        ["Mode-3A", "12-bit squawk, 0–4095. Air-traffic identity. Missing is pad 0; present 0000 collides, so we also keep has_mode_3a."],
        ["Mode-S", "24-bit ICAO aircraft address. Stronger than squawk. We hash it into 1024 embedding buckets."],
        ["Token", "One object’s vector inside the transformer: numbers + embeddings, then contextualized by attention."],
        ["Self-attn", "Self-attention — each token looks at other tokens on the same side (tracks with tracks, plots with plots)."],
        ["rel_ij", "12 hand-built numbers describing the pair (geometry, identity match, same radar). Concatenated into the score head."],
        ["Dustbin", "Extra Hungarian column: “this track matches none of these plots.” Lets a track refuse a steal."],
        ["Ensemble", "p = 0.5 p_MLP + 0.5 p_V8 on the same gated pair. Not a third network."],
        ["Split", "MLP scores 2 km clustering; V8 scores 8 km assignment. Same weights, different job."],
        ["SuperGlue", "Learned matcher that contextualizes two sets then scores pairs. V8’s pattern. Not a citation of their weights."],
        ["V7", "Failed transformer tracker. Owned initiation, coast, and state. Holdout MOTA went negative. Closed."],
        ["MOTA", "Multi-Object Tracking Accuracy. Penalizes false tracks, misses, and ID switches. Our headline metric."],
        ["ID switch", "A track that was aircraft A becomes aircraft B. Unusable in ATC even if MOTA looks fine."],
      ],
      TW
    ),
    new Paragraph({ spacing: { after: 200 } }),
  );

  children.push(h1("Panel A  —  Hybrid correlator"));

  children.push(...block(
    "A1",
    "Multi-radar plots  ·  PSR + SSR",
    "These are the raw detections coming in this window from every radar, already decoded to Cartesian coordinates.",
    "A plot is one detection: time, x y z, sensor id, type (primary or secondary), and whatever extras that sensor produced. Five radars scan on different periods (about 5.5–9 s), so plots in one 1-second eval window are not synchronized.",
    "In: ASTERIX-decoded stream (JSONL). Out: list of measurement dicts into the clutter filter.",
    [
      ["ASTERIX", "Eurocontrol binary radar message standard. We do not ingest raw ASTERIX in this process — decode is offline."],
      ["CAT-048 / CAT-062", "ASTERIX categories. 048 = plots from a radar. 062 = system tracks already formed by a local tracker. Hybrid eats plots, not CAT-062 tracks, except when we replay CAT-062 through an observation model."],
      ["Cartesian", "x, y, z in metres in a local tangent frame, not range/azimuth. Association is done in this frame."],
      ["Window", "Time slice we process together (1 s at eval, 2 s in V8 training). Not a radar scan."],
    ],
  ));

  children.push(...block(
    "A2",
    "Live tracks  ·  KF state",
    "These are aircraft we already believe exist. Each carries a Kalman state at the time of its last update — not at “now.”",
    "State is typically position and velocity. The track also stores last timestamp, hit count, age, and any Mode-3A / Mode-S it has inherited. Tracks sit still in time until we project them for scoring.",
    "In: previous frame’s confirmed and tentative tracks. Out: the same objects, later projected to each candidate measurement time.",
    [
      ["State", "The KF mean: where we think the aircraft is and how it is moving."],
      ["Covariance P", "How uncertain that state is. Grows when we coast, shrinks when we update. Not drawn on the figure."],
      ["Tentative vs confirmed", "A new track is tentative until it collects min_hits (3). Only confirmed tracks are usually scored as output."],
    ],
  ));

  children.push(...block(
    "A3",
    "Clutter MLP  ·  unary reject",
    "A tiny network looks at one PSR plot by itself and drops likely false alarms before anyone associates them.",
    "Unary means one object, not a pair. Features are things like amplitude and kinematics of that plot alone. If the clutter checkpoint is missing or fails to load, Hybrid passes plots through unfiltered — it must not leave a half-loaded net in train mode.",
    "In: raw plots. Out: plots that survived the threshold, still unpaired.",
    [
      ["Clutter", "A detection that is not an aircraft: weather, ground, birds, multipath. PSR is the usual source."],
      ["Unary", "One-input classifier. Contrast with pairwise (two objects) and set (many objects)."],
      ["False alarm", "A plot that would start or update a track if we let it through. Precision dies if this block is weak."],
    ],
  ));

  children.push(...block(
    "A4",
    "2 km cluster  ·  connected components",
    "If two plots in this sweep sit inside two kilometres and the scorer says they are the same aircraft, fuse them into one meta-measurement. Typical case: PSR and SSR of the same plane.",
    "Build a graph: edge if distance < 2 km and p > 0.5. Connected components become one fused plot (position averaged, identity kept). This is the local clique problem the pairwise MLP already solves well. V8 used alone over-draws edges here — that is why split scoring exists.",
    "In: surviving plots. Out: meta-measurements. The purple scorer supplies p.",
    [
      ["Cluster / spatial cluster", "Same-time fusion, not track-to-plot association. “Are these two blips one aircraft right now?”"],
      ["Connected components", "Graph algorithm: if A–B and B–C are edges, {A,B,C} is one cluster."],
      ["2 km gate", "Hard radius. A pair 2.1 km apart is never scored for clustering, period."],
      ["p > 0.5", "Default edge threshold after sigmoid. Not Hungarian — clustering allows a small clique, assignment does not."],
    ],
  ));

  children.push(...block(
    "A5",
    "Project track to meas_t  ·  dt = tₘ − tₜ",
    "Before we compare a track to a plot, slide the track forward (or back) to that plot’s exact time using the Kalman predict step.",
    "This is the fix for temporal dragging. We do not snap every radar to the end of a 2-second window. Distance is then “where the track would be at the plot’s time,” not “where it was three seconds ago.” V8 must consume this projected dict (tmp_t), never the raw unprojected KF state.",
    "In: live tracks + a candidate plot’s timestamp. Out: a temporary projected copy used only for scoring.",
    [
      ["meas_t / tₘ", "Timestamp of the measurement (plot)."],
      ["track_t / tₜ", "Timestamp of the track’s last KF update."],
      ["dt", "meas_t − track_t, in seconds. Positive means the plot is newer; we predict forward."],
      ["Temporal dragging", "Bug when async scans are forced into one snapshot: the filter updates at the wrong time and the track lags or leads."],
      ["tmp_t", "Field name in code for the projected time. If V8 ignores it, association is geometrically wrong."],
    ],
  ));

  children.push(...block(
    "A6",
    "8 km assign  ·  Hungarian  ·  cost = 1 − p",
    "Now the set problem: which live track owns which meta-measurement? Score every pair inside 8 km, turn probability into cost, and let Hungarian pick a one-to-one assignment.",
    "cost = 1 − p so a confident match is cheap. Hungarian cannot give two tracks the same plot or two plots the same track. Unmatched metas can start new tracks. Unmatched tracks coast. Optional dustbin is an extra column so a track can refuse every plot.",
    "In: projected tracks + metas + scores from the purple block. Out: matched pairs, leftovers.",
    [
      ["Assign / association", "Track-to-plot matching across time. Different job from clustering."],
      ["8 km gate", "Hard radius for assignment. Wider than clustering because the track may have coasted."],
      ["Hungarian / LSA", "Kuhn–Munkres / scipy linear_sum_assignment. Global minimum cost, one-to-one."],
      ["cost = 1 − p", "Converts “likely same” into “cheap to assign.” A pair with p = 0 is cost 1 and is rejected."],
      ["Uniqueness", "At most one plot per track per step. This is why V8 has zero ID switches even when it over-initiates."],
    ],
  ));

  children.push(...block(
    "A7",
    "Async CV KF  ·  update at tₘ",
    "For each matched pair, run a constant-velocity Kalman update at the measurement’s own time. That is the kinematics. The net never does this.",
    "Predict from track_t to meas_t with a continuous-time process model, then update with the plot. Process noise Q and measurement noise R stay classical. After the update the track’s time becomes meas_t.",
    "In: matched (track, meta). Out: updated KF mean and covariance at tₘ.",
    [
      ["Kalman filter", "Optimal linear estimator under Gaussian noise. Predict then update."],
      ["CV", "Constant velocity model: position integrates velocity; velocity is a random walk."],
      ["Q / R", "Process noise / measurement noise. Tuning knobs, not learned in V8."],
      ["No time-drag", "Because dt is exact, a late SSR and an early PSR do not yank the track to a fake common time."],
    ],
  ));

  children.push(...block(
    "A8",
    "M/N manager  ·  3 hits / ~10 coasts",
    "Birth and death. Three associated hits promote a tentative track to confirmed. About ten consecutive misses delete it. In between, it coasts.",
    "This is why radar shadows do not instantly kill a track, and why one-off junk does not become a confirmed track. Eval tables in the report use max_age = 10; the CLI default is 2 — those are different MOTA numbers.",
    "In: updated tracks + unmatched metas (new tentative). Out: confirmed picture for this frame.",
    [
      ["M/N", "Classic track logic: M hits out of N tries. We implement it as min_hits = 3, max_age ≈ 10."],
      ["Hit", "A successful association this step."],
      ["Coast / age", "Steps (or missed associations) since the last hit. Not calendar age of the aircraft."],
      ["Promote / delete", "Tentative → confirmed at min_hits. Confirmed → gone at max_age."],
    ],
  ));

  children.push(h1("Purple block  —  the only neural net"));

  children.push(...block(
    "A9",
    "Learned scorer  (slot, not a third tracker)",
    "Every purple arrow is the same question: given two gated objects, how likely are they the same aircraft? Hybrid does not care which net answers, as long as it returns p in [0, 1].",
    "Two call sites share the slot: clustering (score_pairs) and assignment (score_assignment). CLI: --assoc mlp | transformer | ensemble. Missing V8 weights fall back to MLP.",
    "In: gated pairs, already projected. Out: a logit per pair; Hybrid applies sigmoid.",
    [
      ["Scorer / associator", "A function pair → probability. Not a tracker. V8 is this, not a new updater class."],
      ["--assoc", "CLI switch. Default mlp. transformer is opt-in. ensemble averages."],
    ],
  ));

  children.push(...block(
    "A9a",
    "MLP pair  ·  4–6 features  ·  this pair only",
    "Two specialist feed-forward nets. Each looks at one pair and nothing else. That isolation is why Hybrid is precise, and why a crossing that is only obvious from the rest of the set can fool it.",
    "PSR–PSR uses 6 kinematic/amplitude features. SSR–ANY uses 4: distance, azimuth, Mode-3A match, Mode-S match. About a few thousand parameters each. Trained with weighted binary cross-entropy on track_id equality.",
    "In: one pair’s hand features. Out: one logit. No third object in the forward pass.",
    [
      ["Pairwise", "Two objects, no set. The opposite of V8’s self-attention."],
      ["PSR–PSR / SSR–ANY", "The two checkpoints. ANY means the other side can be PSR or SSR."],
      ["Match flag +1 / 0 / −1", "Both present and equal / either missing / both present and different. Not the raw squawk number."],
    ],
  ));

  children.push(...block(
    "A9b",
    "V8 transformer  ·  set self-attn  ·  then pair head",
    "Same question as the MLP, but each object’s vector has already mixed with the other gated objects on its side. Then a small MLP head scores the pair with explicit geometry glued on.",
    "About 180k parameters. d_model 64, 2 pre-norm layers, 4 heads. See Panel B for the insides. Alone it over-associates in clustering. It does not own state.",
    "In: the gated set. Out: logits for the enumerated pairs, plus a dustbin per track.",
    [
      ["Set context", "Token i is allowed to see token k on the same side before we score (i, j)."],
      ["Pair head", "Last layer is still an MLP on [h_i ; h_j ; rel_ij]. Attention is the encoder, not the decision."],
    ],
  ));

  children.push(...block(
    "A9c",
    "Compose  ·  ensemble ½+½  ·  or split",
    "Two ways to use both learners without training a third net. Ensemble averages probabilities. Split gives each the job that matches its bias.",
    "Ensemble: p = 0.5 p_MLP + 0.5 p_V8. Single-run +0.007 MOTA on the dense stream; Sweden tie. Split: MLP on 2 km cluster, V8 on 8 km assign — recovered most of the precision pure V8 lost. Raising V8’s own cluster threshold under-merged PSR+SSR and birthed two tracks per aircraft.",
    "In: both scorers’ p. Out: one p Hybrid already knows how to use.",
    [
      ["Ensemble", "Average, not a mixture-of-experts gate. Dustbin is off on this path."],
      ["Split", "Different call site, different net. Best current use of V8 if you do not want extras."],
    ],
  ));

  children.push(
    h2("A10   “Not learned”  (Kalman · gates · Hungarian · M/N)"),
    p("Say this.  If they ask “so the AI tracks the plane?” — no. The AI only votes on pairs. These four blocks are classical and they are why identity does not collapse."),
    p("What it does.  Gates throw away 50 km nonsense. Hungarian makes the vote one-to-one. Kalman moves the state. M/N starts and kills tracks. V7 tried to learn all four and flooded the picture."),
  );

  children.push(h1("Panel B  —  AssociationTransformerV8"));

  children.push(
    p("Say this first.  SuperGlue-style means: embed two sets, let each set talk to itself, then score pairs with geometry in the open. It is a matcher. DETR / TrackFormer / V7 are detectors-plus-trackers. We are not doing that."),
  );

  children.push(...block(
    "B1",
    "Track tokens  ·  projected state  ·  role = track",
    "Each live track becomes one token. The kinematics on that token are the projected KF state, not the stale one.",
    "role embedding tells the net “I am a track.” age and hits (maturity) are numeric; they are 0 on plots so the net can tell a brand-new plot from a 20-hit track.",
    "In: projected track dicts. Out: raw fields into token build.",
    [
      ["Token", "The unit of attention. One row in the (N × 64) matrix after encoding."],
      ["role", "A learned 64-d vector: index 0 = track, 1 = measurement."],
    ],
  ));

  children.push(...block(
    "B2",
    "Plot / meta tokens  ·  role = meas",
    "Each plot or fused meta-measurement is a token with role = meas and type = PSR or SSR.",
    "After clustering, assignment usually sees metas, not raw dual plots. Clustering sees the raw plots. Same encoder weights, different role/type embeds.",
    "In: measurements or metas. Out: raw fields into token build.",
    [
      ["meas", "Short for measurement. Same as plot / meta in this figure."],
      ["type embed", "PSR vs SSR. One net, not two specialists — the embed replaces the MLP split."],
    ],
  ));

  children.push(...block(
    "B3",
    "Token build  ·  15-d numeric → Linear 64  +  5 embeds",
    "We never pour raw metres into attention. Fifteen scaled numbers go through a linear layer to 64-d. Five embeddings (also 64-d) are added: role, type, sensor, Mode-3A, Mode-S.",
    "Numeric block: x y z / 1e5; vx vy vz / 1e3; has_vx has_vy has_vz; amplitude / 100 and has_amp; age, hits; dt; has_mode_3a. Missing velocity is 0 with has_v = 0 so “no Doppler” does not look like “parked.”",
    "In: dict fields. Out: h⁰ ∈ R^{64} before attention.",
    [
      ["15-d / 64-d", "Input width vs model width (d_model). Linear(15, 64) is the first learned map."],
      ["Embedding", "A looked-up vector for a discrete id. Trainable. Not a one-hot."],
      ["Five embeds, not six", "role, type, sensor (0–8), Mode-3A (0–4095), Mode-S (1024 buckets)."],
      ["Pad 0 vs squawk 0000", "Missing Mode-3A is index 0. A real squawk 0000 is also 0. has_mode_3a is the disambiguator."],
      ["Mode-S hash", "24-bit address → md5 bucket 1–1023. Bucket 0 is missing. Not a cryptographic claim — just a stable id."],
      ["sensor embed", "Which of the (up to 8) radars. Tracks use a dummy id 0."],
    ],
  ));

  children.push(...block(
    "B4",
    "Self-attn tracks  ·  2 layers · 4 heads · d=64 · FFN 256 · GELU",
    "Each track token may look at the other track tokens in this gated set. After this, track A’s vector can carry “C is also live and close.”",
    "Two pre-norm Transformer encoder layers. Four heads → 16-d per head. Feed-forward width 256. GELU activation. Dropout 0.1. This is standard encoder, not a custom radar layer.",
    "In: track token matrix. Out: contextualized h_track.",
    [
      ["Self-attention", "Query/key/value all come from the same set. Soft weights, not a hard neighbor pick."],
      ["Head", "One attention subspace. Four heads see different mixed patterns."],
      ["d_model = 64", "Width of every token throughout. Small on purpose (~180k params total)."],
      ["FFN", "Feed-forward network inside each layer: 64 → 256 → 64."],
      ["GELU", "Gaussian Error Linear Unit — smooth ReLU-like activation."],
      ["Pre-norm", "LayerNorm before attention/FFN, not after. More stable small-net training."],
    ],
  ));

  children.push(...block(
    "B5",
    "Self-attn plots  ·  same weights  ·  no cross-attention",
    "Plots attend only to plots. The two self-attn boxes share the encoder weights. There is no track-attends-to-plot layer in v1.",
    "Cross-attention is omitted on purpose: rel_ij already carries geometry between sides. Adding cross-attn is a later ablation, not a missing piece. “Context stays on its own side” is the line to say while you point here.",
    "In: plot token matrix. Out: contextualized h_meas.",
    [
      ["Cross-attention", "Queries from one set, keys/values from the other. V7 used this as a soft assigner. We do not."],
      ["Same weights", "One TransformerEncoder applied twice, once per role — not two independently trained stacks."],
    ],
  ));

  children.push(...block(
    "B6",
    "rel_ij  (12-d, explicit)",
    "Attention is not asked to rediscover that two objects are 400 metres apart. We concatenate twelve physics features into the score head every time.",
    "Groups: dx dy dz dist / 1e5; Δ|v| / 1e3 and cos_vel (0 if either velocity missing); Δaz Δel; dt; Mode-3A match; Mode-S match; same_sensor in {0,1}. This is the inductive bias V7 promised and never built.",
    "In: the same pair the MLP would have seen, richer. Out: a 12-vector glued to [h_i ; h_j].",
    [
      ["rel_ij", "Relative features of object i vs object j. Always used, never optional."],
      ["cos_vel", "Cosine of heading difference. Forced to 0 if a side has no velocity — no fake “both parked” match."],
      ["Δaz / Δel", "Wrapped azimuth and elevation difference. Radar-native, useful when velocity is missing."],
      ["same_sensor", "1 if both detections came from the same radar. Two-radar vs one-radar is a real cue."],
    ],
  ));

  children.push(...block(
    "B7",
    "score_pairs  ·  [hᵢ ; hⱼ ; rel] → MLP → logit",
    "The clustering call. Hybrid already listed the 2 km pairs. V8 returns one logit per pair. Hybrid sigmoids and draws an edge if p > 0.5.",
    "The head is Linear(140 → 64 → 1) because 64 + 64 + 12 = 140. No sigmoid inside the module — training and Hybrid share the same logit scale as the pairwise MLPs.",
    "In: left plots, right plots, pair index. Out: vector of P logits.",
    [
      ["score_pairs", "Python method name. Clustering only."],
      ["[hᵢ ; hⱼ ; rel]", "Concatenation, not a product. The head sees context and physics together."],
      ["Logit", "Unbounded real. Do not call it a probability until after σ."],
    ],
  ));

  children.push(...block(
    "B8",
    "score_assignment  ·  S ∈ ℝᵀˣᴹ  +  dustbin ∈ ℝᵀ",
    "The assignment call. Returns a T by M score matrix and a dustbin value per track. Hybrid builds costs and may append the dustbin as an extra Hungarian column.",
    "S[i, j] is “track i owns plot j.” dustbin[i] is “track i owns none of these.” With current weights the dustbin is not calibrated — we leave that column off in the reported ensemble. The API is there so a coasting track can refuse a neighbor instead of stealing.",
    "In: tracks, metas. Out: (S, dustbin).",
    [
      ["T, M", "Number of tracks, number of metas in this gated call. Typically N < 80; no padding at inference."],
      ["S", "Score matrix, logits. Not a softmax distribution unless a trainer applies one."],
      ["Dustbin", "Named after SuperGlue’s unmatched bin. Extra assignable column, not a trash plot."],
      ["ℝᵀˣᴹ", "Real matrix with T rows and M columns."],
    ],
  ));

  children.push(...block(
    "B9",
    "Allowed  (green strip)",
    "p = σ(logit). Cluster if p > 0.5. Assignment cost = 1 − p. Label is track_id equality, never fed in as a feature. Identity on the wire is Mode-3A / Mode-S only.",
    "This strip is the contract with Hybrid. If a change violates it — predicting Δs, carrying GRU state, scoring a 50 km pair — it is no longer V8.",
    "In: logits. Out: the p Hybrid already consumes.",
    [
      ["track_id", "Simulator / eval identity. Supervision only. Using it as an input would be cheating and would not exist in the field."],
      ["has_mode_3a", "Numeric 0/1 companion to the Mode-3A embed. Separates missing from squawk 0000."],
    ],
  ));

  children.push(...block(
    "B10",
    "Forbidden  (red strip)  ·  that is V7",
    "No residual Δs. No existence or init heads. No GRU. No scoring outside the 2 km / 8 km gates. V8 does not own birth, death, or time.",
    "V7 was a DETR-style tracker: residual state, existence logits, GRU memory, 50 km association. Holdout MOTA −1.1 to −3.0 and tens to hundreds of ID switches. Point at this box if someone asks “why not just let the transformer track?”",
    "In: temptation. Out: a closed research path.",
    [
      ["Residual Δs", "Network-predicted state increment. Kalman already does this with physics."],
      ["Existence / init head", "A learned “this should be a track” bit. That is how V7 flooded false tracks."],
      ["GRU", "Gated Recurrent Unit — a small RNN memory. Starves on async gaps; Kalman coasts instead."],
      ["50 km gate", "V7’s max_assoc_m. Soft attention then invented ghosts. Hard gates stay in front of V8."],
    ],
  ));

  children.push(
    h1("If they only give you one minute"),
    p("Hybrid is the system. Two tiny MLPs score pairs; a Kalman filter tracks; Hungarian keeps identities unique. That is MOTA 0.865 / 0.971 and zero ID switches."),
    p("V8 is a transformer we plug into those two scoring sockets so a pair can see the traffic around it. Alone it starts extra tracks. Averaged with the MLP it is a single-run +0.007 MOTA. It is not the operational default."),
    p("We will know whether the transformer is required when we have a crossing / overlap holdout. Sweden is 30 minutes long and geometrically easy. Do not oversell the 0.007."),
  );

  const doc = new Document({
    creator: "Tom Rathbun",
    title: "Architecture Block Guide — AI Tracker Correlator",
    description: "Briefing script: every Figure 1 block, every acronym.",
    styles: {
      default: { document: { run: { font: "Arial", size: 21 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: "Arial", size: 30, bold: true, color: NAVY },
          paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: "Arial", size: 24, bold: true, color: STEEL },
          paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
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
              spacing: { after: 80 },
              children: [
                r("AI Tracker Correlator", { size: 16, bold: true, color: NAVY }),
                r("\tArchitecture block guide  ·  briefing script", { size: 16, color: MUTED }),
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
              spacing: { before: 60 },
              children: [
                r("Tom Rathbun  ·  16 August 2026  ·  Point at the box, then read its heading", { size: 15, color: MUTED }),
                r("\t"),
                r("Page ", { size: 15, color: MUTED }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 15, color: MUTED }),
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
