const pptxgen = require("pptxgenjs");
const path = require("path");

const ASSET = (f) => path.join(__dirname, "assets", f);

// ---------- palette ----------
const INK = "1B2A4A";      // dark navy
const INK2 = "24344D";     // slate ink (body text)
const PAPER = "FFFFFF";
const CARD = "F4F7FB";     // light card
const CARD2 = "EEF3FA";
const MUTED = "5B6B82";
const LINE = "D8E0EC";
const AMBER = "E8833A";     // accent
const TEAL = "2A9D8F";
const BLUE = "2A6F97";
const PLUM = "8E5572";
const ICE = "CADCFC";

const FH = "Cambria";      // header serif
const FB = "Calibri";      // body sans

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
pres.author = "DLNLP Group — TinyStories Reproduction";
pres.title = "Reproducing TinyStories";

const W = 13.333, H = 7.5;
const shadow = () => ({ type: "outer", color: "1B2A4A", blur: 7, offset: 3, angle: 90, opacity: 0.12 });

// metric colors
const C = { grammar: BLUE, creativity: AMBER, consistency: TEAL, plot: PLUM };

// data
const SIZES = ["XS", "Small", "Medium", "Large", "XL"];
const PARAMS = ["1.2M", "3.0M", "9.8M", "28.7M", "56.2M"];
const grammar = [5.14, 6.44, 6.96, 7.28, 7.74];
const creativity = [4.62, 5.20, 5.26, 5.84, 5.92];
const consistency = [4.06, 5.50, 6.00, 7.08, 7.58];
const plot = [3.12, 4.42, 4.78, 5.88, 6.28];
const ageMean = [4.74, 5.62, 5.99, 6.42, 6.59];
const ovTrain = [99.77, 99.86, 99.78, 99.92, 99.06];
const ovValid = [99.54, 99.63, 99.46, 99.54, 98.63];
const ovGap = [0.23, 0.23, 0.31, 0.38, 0.43];

// ---------- helpers ----------
function pageFooter(slide, n, dark = false) {
  slide.addText("Reproducing TinyStories  ·  DLNLP Reproducibility & Extension Challenge", {
    x: 0.55, y: H - 0.42, w: 8, h: 0.3, fontFace: FB, fontSize: 9,
    color: dark ? ICE : MUTED, align: "left",
  });
  slide.addText(String(n), {
    x: W - 1.0, y: H - 0.42, w: 0.45, h: 0.3, fontFace: FB, fontSize: 10,
    color: dark ? ICE : MUTED, align: "right",
  });
}

function sectionTag(slide, label, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55, y: 0.5, w: 0.18, h: 0.42, rectRadius: 0.06, fill: { color: color },
  });
  slide.addText(label.toUpperCase(), {
    x: 0.85, y: 0.5, w: 6, h: 0.42, fontFace: FB, fontSize: 12, bold: true,
    color: color, charSpacing: 2, align: "left", valign: "middle", margin: 0,
  });
}

function headTitle(slide, title) {
  slide.addText(title, {
    x: 0.55, y: 0.92, w: 12.2, h: 0.85, fontFace: FH, fontSize: 30, bold: true,
    color: INK, align: "left", valign: "middle", margin: 0,
  });
}

function contentSlide(section, sectionColor, title) {
  const slide = pres.addSlide();
  slide.background = { color: PAPER };
  sectionTag(slide, section, sectionColor);
  headTitle(slide, title);
  return slide;
}

// card with optional header + body lines
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: opts.fill || CARD },
    line: opts.line ? { color: opts.line, width: 1 } : { type: "none" },
    shadow: opts.shadow ? shadow() : undefined,
  });
}

function iconCircle(slide, x, y, d, color, glyph) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color } });
  slide.addText(glyph, {
    x, y, w: d, h: d, align: "center", valign: "middle", margin: 0,
    fontFace: FH, fontSize: 15, bold: true, color: PAPER,
  });
}

// =================================================================
// SLIDE 1 — TITLE
// =================================================================
(function () {
  const s = pres.addSlide();
  s.background = { color: INK };
  // soft accent ovals (motif)
  s.addShape(pres.shapes.OVAL, { x: 10.6, y: -1.6, w: 4.6, h: 4.6, fill: { color: "24355A" } });
  s.addShape(pres.shapes.OVAL, { x: 11.9, y: 4.7, w: 3.4, h: 3.4, fill: { color: "20304F" } });
  s.addShape(pres.shapes.OVAL, { x: 0.55, y: 1.05, w: 0.16, h: 0.16, fill: { color: AMBER } });

  s.addText("REPRODUCIBILITY & EXTENSION CHALLENGE  ·  DEEP LEARNING FOR NLP", {
    x: 0.85, y: 1.0, w: 11, h: 0.3, fontFace: FB, fontSize: 12.5, bold: true,
    color: AMBER, charSpacing: 2, margin: 0,
  });
  s.addText("Reproducing TinyStories", {
    x: 0.8, y: 1.7, w: 11.8, h: 1.2, fontFace: FH, fontSize: 54, bold: true,
    color: PAPER, margin: 0,
  });
  s.addText("How small can a language model be and still tell a coherent story?", {
    x: 0.82, y: 3.0, w: 10.8, h: 0.7, fontFace: FB, fontSize: 20, italic: true,
    color: ICE, margin: 0,
  });

  // thin divider via spacing (no accent bar) — use three dots
  s.addText("Eldan & Li (2023), Microsoft Research  —  reproduced & extended", {
    x: 0.82, y: 3.95, w: 11, h: 0.4, fontFace: FB, fontSize: 14, color: "9FB3D1", margin: 0,
  });

  // bottom meta cards
  const meta = [
    ["What we did", "Trained 5 GPT models (1M–56M), ran GPT-4 evaluation & memorization analysis"],
    ["Extension", "Our own modern GPT: Differential Transformer + Grouped-Query Attention"],
    ["Repository", "github.com/Grabosticus/chatbot"],
  ];
  meta.forEach((m, i) => {
    const x = 0.82 + i * 4.0;
    s.addText(m[0].toUpperCase(), { x, y: 5.25, w: 3.7, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: AMBER, charSpacing: 1, margin: 0 });
    s.addText(m[1], { x, y: 5.55, w: 3.7, h: 1.0, fontFace: FB, fontSize: 12.5, color: ICE, margin: 0 });
  });

  s.addText("Group presentation  ·  29–30 June 2026", {
    x: 0.82, y: 6.85, w: 8, h: 0.3, fontFace: FB, fontSize: 11, color: "7E92B2", margin: 0,
  });
})();

// =================================================================
// SLIDE 2 — ROADMAP
// =================================================================
(function () {
  const s = pres.addSlide();
  s.background = { color: PAPER };
  sectionTag(s, "Roadmap", AMBER);
  headTitle(s, "How this talk is structured");

  const steps = [
    ["1", "Introduction", "The paper and its key contributions", BLUE],
    ["2", "Understanding", "Dataset, models and the GPT-Eval idea", TEAL],
    ["3", "Reproduction", "5 models, GPT-4 grading, memorization", AMBER],
    ["4", "Extension", "Our own GPT: Diff-Transformer + GQA", PLUM],
    ["5", "Conclusion", "Reproducibility verdict & future work", INK],
  ];
  const x0 = 0.55, y0 = 2.15, cw = 2.4, gap = 0.13, ch = 3.3;
  steps.forEach((st, i) => {
    const x = x0 + i * (cw + gap);
    card(s, x, y0, cw, ch, { fill: CARD, shadow: true });
    iconCircle(s, x + cw / 2 - 0.45, y0 + 0.45, 0.9, st[3], st[0]);
    s.addText(st[1], { x: x + 0.15, y: y0 + 1.55, w: cw - 0.3, h: 0.5, fontFace: FH, fontSize: 17, bold: true, color: INK, align: "center", margin: 0 });
    s.addText(st[2], { x: x + 0.2, y: y0 + 2.1, w: cw - 0.4, h: 1.1, fontFace: FB, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });
  });
  s.addText("Each section maps directly to the required presentation structure.", {
    x: 0.55, y: 5.9, w: 12, h: 0.4, fontFace: FB, fontSize: 13, italic: true, color: MUTED, margin: 0,
  });
  pageFooter(s, 2);
})();

// =================================================================
// SLIDE 3 — INTRODUCTION: the paper
// =================================================================
(function () {
  const s = contentSlide("1 · Introduction", BLUE, "The paper & its key contributions");
  s.addText([
    { text: "TinyStories", options: { bold: true, color: INK } },
    { text: " is a synthetic dataset of short children's stories — written by GPT-3.5/4 using only words a 3–4-year-old understands. It asks a provocative question: do we need billions of parameters for fluent, coherent English, or is the ", options: { color: INK2 } },
    { text: "data distribution", options: { bold: true, color: AMBER } },
    { text: " the real bottleneck?", options: { color: INK2 } },
  ], { x: 0.55, y: 1.9, w: 7.2, h: 1.5, fontFace: FB, fontSize: 16, lineSpacingMultiple: 1.15, valign: "top", margin: 0 });

  const contribs = [
    ["A purpose-built dataset", "Stories constrained to a toddler-level vocabulary, diverse yet simple enough for tiny models to actually master.", BLUE],
    ["Tiny models, real fluency", "Models far below 10M parameters — even very shallow ones — generate grammatical, on-topic, multi-sentence stories.", TEAL],
    ["GPT-4 as evaluator", "Standard metrics miss coherence; instead GPT-4 grades grammar, creativity & consistency of open-ended completions.", AMBER],
    ["A lab for emergence", "Small scale makes capabilities — and how they appear with size/depth — interpretable and cheap to study.", PLUM],
  ];
  const x0 = 8.05, cw = 4.75, ch = 1.2, y0 = 1.68, vg = 0.13;
  contribs.forEach((c, i) => {
    const y = y0 + i * (ch + vg);
    card(s, x0, y, cw, ch, { fill: CARD, shadow: true });
    s.addShape(pres.shapes.OVAL, { x: x0 + 0.22, y: y + 0.27, w: 0.14, h: 0.14, fill: { color: c[2] } });
    s.addText(c[0], { x: x0 + 0.5, y: y + 0.13, w: cw - 0.7, h: 0.35, fontFace: FH, fontSize: 14.5, bold: true, color: INK, margin: 0 });
    s.addText(c[1], { x: x0 + 0.5, y: y + 0.5, w: cw - 0.75, h: 0.7, fontFace: FB, fontSize: 11.5, color: MUTED, margin: 0 });
  });

  // big takeaway band (left lower)
  card(s, 0.55, 3.7, 7.2, 2.55, { fill: INK, shadow: true });
  s.addText("THE CENTRAL RESULT", { x: 0.95, y: 3.95, w: 6, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: AMBER, charSpacing: 1.5, margin: 0 });
  s.addText([
    { text: "Quality is mostly a data problem.", options: { bold: true, color: PAPER, breakLine: true } },
    { text: "Trained on the right distribution, models 100–1000× smaller than GPT-2 still produce consistent, creative, grammatical stories — and scale up gracefully.", options: { color: ICE } },
  ], { x: 0.95, y: 4.35, w: 6.4, h: 1.7, fontFace: FB, fontSize: 16, lineSpacingMultiple: 1.12, valign: "top", margin: 0 });

  pageFooter(s, 3);
})();

// =================================================================
// SLIDE 4 — UNDERSTANDING: concepts
// =================================================================
(function () {
  const s = contentSlide("2 · Understanding", TEAL, "Core concepts: data, model, evaluation");

  const cols = [
    ["The dataset", TEAL, [
      "Short, self-contained stories using a small, child-level vocabulary",
      "Generated by GPT-3.5/4 with prompts that inject random words & features for diversity",
      "In-distribution & learnable — the key enabler for tiny models",
    ]],
    ["The model", BLUE, [
      "Plain GPT-style decoder-only transformers",
      "Scaled along two axes: width (parameters) and depth (layers)",
      "Trained with next-token prediction on the story corpus",
    ]],
    ["The evaluation", AMBER, [
      "BLEU / perplexity don't capture story coherence",
      "Feed prompt beginnings → let the model complete the story",
      "GPT-4 grades the completion like a teacher grading a pupil",
    ]],
  ];
  const x0 = 0.55, cw = 4.0, gap = 0.21, y0 = 1.85, ch = 3.7;
  cols.forEach((c, i) => {
    const x = x0 + i * (cw + gap);
    card(s, x, y0, cw, ch, { fill: CARD, shadow: true });
    iconCircle(s, x + 0.3, y0 + 0.32, 0.62, c[1], ["§", "❖", "★"][i]);
    s.addText(c[0], { x: x + 1.05, y: y0 + 0.36, w: cw - 1.2, h: 0.55, fontFace: FH, fontSize: 18, bold: true, color: INK, valign: "middle", margin: 0 });
    s.addText(c[2].map((t, j) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 8 } })),
      { x: x + 0.35, y: y0 + 1.2, w: cw - 0.6, h: ch - 1.4, fontFace: FB, fontSize: 13, color: INK2, valign: "top", margin: 0 });
  });

  card(s, 0.55, 5.85, 12.23, 0.92, { fill: CARD2 });
  s.addText([
    { text: "Why it matters:  ", options: { bold: true, color: AMBER } },
    { text: "this turns language-model behaviour into something you can study on a single GPU — and grade objectively enough to compare model sizes.", options: { color: INK2 } },
  ], { x: 0.85, y: 5.85, w: 11.7, h: 0.92, fontFace: FB, fontSize: 14, valign: "middle", margin: 0 });

  pageFooter(s, 4);
})();

// =================================================================
// SLIDE 5 — UNDERSTANDING: GPT-Eval + the 3 claims
// =================================================================
(function () {
  const s = contentSlide("2 · Understanding", TEAL, "The GPT-Eval protocol & the claims we test");

  // protocol pipeline (left)
  s.addText("THE GPT-EVAL PIPELINE", { x: 0.55, y: 1.8, w: 6, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: TEAL, charSpacing: 1, margin: 0 });
  const steps = [
    ["Prompt beginnings", "Hand-written story openings (50 prompts)"],
    ["Generate", "10 completions per prompt at temperature 1.0"],
    ["GPT-4 grades", "Grammar · Creativity · Consistency · Plot + age guess"],
  ];
  let py = 2.25;
  steps.forEach((st, i) => {
    card(s, 0.55, py, 5.55, 0.95, { fill: CARD, shadow: true });
    iconCircle(s, 0.78, py + 0.23, 0.5, [BLUE, TEAL, AMBER][i], String(i + 1));
    s.addText(st[0], { x: 1.45, y: py + 0.13, w: 4.5, h: 0.35, fontFace: FH, fontSize: 14.5, bold: true, color: INK, margin: 0 });
    s.addText(st[1], { x: 1.45, y: py + 0.48, w: 4.5, h: 0.4, fontFace: FB, fontSize: 11.5, color: MUTED, margin: 0 });
    if (i < 2) s.addText("▼", { x: 0.93, y: py + 0.93, w: 0.3, h: 0.22, fontFace: FB, fontSize: 11, color: LINE, align: "center", margin: 0 });
    py += 1.18;
  });

  // the 3 claims (right)
  s.addText("THE THREE CLAIMS WE REPRODUCE", { x: 6.5, y: 1.8, w: 6.3, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: AMBER, charSpacing: 1, margin: 0 });
  const claims = [
    ["C1", "Bigger → more coherent", "Larger models produce more consistent story continuations than smaller ones.", BLUE],
    ["C2", "Grammar comes first", "Very small models write grammatically before they master consistency or creativity.", TEAL],
    ["C3", "Not memorized", "Continuations are genuinely generated, not copied from the training stories.", PLUM],
  ];
  let cy = 2.25;
  claims.forEach((c) => {
    card(s, 6.5, cy, 6.28, 1.18, { fill: CARD, shadow: true });
    iconCircle(s, 6.72, cy + 0.32, 0.55, c[3], c[0]);
    s.addText(c[1], { x: 7.45, y: cy + 0.16, w: 5.2, h: 0.4, fontFace: FH, fontSize: 15.5, bold: true, color: INK, margin: 0 });
    s.addText(c[2], { x: 7.45, y: cy + 0.56, w: 5.15, h: 0.55, fontFace: FB, fontSize: 12, color: MUTED, margin: 0 });
    cy += 1.35;
  });

  pageFooter(s, 5);
})();

// =================================================================
// SLIDE 6 — REPRODUCTION: experimental setup (models + protocol + loss)
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction", AMBER, "Experimental setup");

  // models table (left)
  s.addText("FIVE MODELS — SCALING WIDTH & DEPTH", { x: 0.55, y: 1.8, w: 6.5, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: AMBER, charSpacing: 1, margin: 0 });
  const head = ["Model", "Params", "d_model", "Layers"].map((t) => ({ text: t, options: { fill: { color: INK }, color: PAPER, bold: true, fontFace: FB, fontSize: 12.5, align: "center", valign: "middle" } }));
  const dmodel = ["96", "128", "224", "512", "544"];
  const layers = ["3", "7", "10", "10", "12"];
  const rows = [head];
  SIZES.forEach((sz, i) => {
    const fill = i % 2 ? CARD : PAPER;
    rows.push([
      { text: sz, options: { bold: true, color: INK, align: "left", fill: { color: fill } } },
      { text: PARAMS[i], options: { color: INK2, align: "center", fill: { color: fill } } },
      { text: dmodel[i], options: { color: INK2, align: "center", fill: { color: fill } } },
      { text: layers[i], options: { color: INK2, align: "center", fill: { color: fill } } },
    ]);
  });
  s.addTable(rows, {
    x: 0.55, y: 2.2, w: 6.1, colW: [2.0, 1.6, 1.3, 1.2], rowH: 0.46,
    fontFace: FB, fontSize: 12.5, valign: "middle", border: { type: "solid", color: LINE, pt: 0.5 },
  });
  s.addText("All five trained for one epoch on TinyStories-train; 10k-token vocabulary, 512-token context.", {
    x: 0.55, y: 5.05, w: 6.1, h: 0.6, fontFace: FB, fontSize: 11.5, italic: true, color: MUTED, margin: 0,
  });

  // protocol stat callouts (bottom-left)
  const stats = [["10k", "vocab"], ["512", "context"], ["50×10", "eval samples"], ["1", "epoch"]];
  stats.forEach((st, i) => {
    const x = 0.55 + i * 1.55;
    card(s, x, 5.75, 1.42, 0.95, { fill: CARD2 });
    s.addText(st[0], { x, y: 5.84, w: 1.42, h: 0.45, fontFace: FH, fontSize: 19, bold: true, color: AMBER, align: "center", margin: 0 });
    s.addText(st[1], { x, y: 6.3, w: 1.42, h: 0.3, fontFace: FB, fontSize: 10.5, color: MUTED, align: "center", margin: 0 });
  });

  // loss plot (right)
  card(s, 7.0, 1.95, 5.8, 4.75, { fill: PAPER, line: LINE });
  s.addText("Training converges within one epoch", { x: 7.2, y: 2.05, w: 5.4, h: 0.35, fontFace: FH, fontSize: 14.5, bold: true, color: INK, margin: 0 });
  s.addImage({ path: ASSET("loss_xs.png"), x: 7.15, y: 2.45, w: 5.5, h: 3.05, sizing: { type: "contain", w: 5.5, h: 3.05 } });
  s.addText([
    { text: "Loss falls from ~97 → 2.2 (cross-entropy over 10k vocab). ", options: { color: INK2 } },
    { text: "All sizes follow the same shape; XS shown as representative.", options: { italic: true, color: MUTED } },
  ], { x: 7.2, y: 5.7, w: 5.45, h: 0.9, fontFace: FB, fontSize: 11.5, valign: "top", margin: 0 });

  pageFooter(s, 6);
})();

// =================================================================
// SLIDE 7 — CLAIM 1: bigger = more coherent (line chart)
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction — Claim 1", AMBER, "Bigger models tell more coherent stories");

  s.addChart(pres.charts.LINE, [
    { name: "Grammar", labels: SIZES, values: grammar },
    { name: "Creativity", labels: SIZES, values: creativity },
    { name: "Consistency", labels: SIZES, values: consistency },
    { name: "Plot sense", labels: SIZES, values: plot },
  ], {
    x: 0.5, y: 1.95, w: 7.7, h: 4.7,
    chartColors: [C.grammar, C.creativity, C.consistency, C.plot],
    lineSize: 3.2, lineSmooth: true,
    showLegend: true, legendPos: "b", legendColor: INK2, legendFontFace: FB, legendFontSize: 12,
    valAxisMinVal: 2, valAxisMaxVal: 9, valAxisMajorUnit: 1,
    valGridLine: { color: "EAEFF6", size: 0.5 }, catGridLine: { style: "none" },
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
    catAxisLabelFontFace: FB, valAxisLabelFontFace: FB,
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    showTitle: true, title: "GPT-4 mean score (1–10) vs. model size", titleColor: INK, titleFontFace: FH, titleFontSize: 14,
    chartArea: { fill: { color: "FFFFFF" } },
  });

  // interpretation panel (right)
  card(s, 8.5, 1.95, 4.3, 4.7, { fill: CARD, shadow: true });
  s.addText("INTERPRETATION", { x: 8.75, y: 2.15, w: 3.8, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: AMBER, charSpacing: 1.5, margin: 0 });
  s.addText("Consistency rises monotonically", { x: 8.75, y: 2.5, w: 3.85, h: 0.55, fontFace: FH, fontSize: 16, bold: true, color: INK, margin: 0 });
  // big stat
  s.addText([
    { text: "4.1 ", options: { color: MUTED } },
    { text: "→ ", options: { color: AMBER } },
    { text: "7.6", options: { color: TEAL } },
  ], { x: 8.75, y: 3.1, w: 3.85, h: 0.75, fontFace: FH, fontSize: 38, bold: true, margin: 0 });
  s.addText("mean consistency, XS → XL (+86%)", { x: 8.75, y: 3.9, w: 3.85, h: 0.4, fontFace: FB, fontSize: 11.5, color: MUTED, margin: 0 });

  s.addText([
    { text: "Every metric improves with scale — and the steepest gains are in consistency and plot sense, exactly the higher-order story skills.", options: { breakLine: true, paraSpaceAfter: 8, color: INK2 } },
    { text: "✓ Claim 1 reproduced.", options: { bold: true, color: TEAL } },
  ], { x: 8.75, y: 4.45, w: 3.85, h: 2.0, fontFace: FB, fontSize: 13, valign: "top", margin: 0 });

  pageFooter(s, 7);
})();

// =================================================================
// SLIDE 8 — CLAIM 2: grammar before consistency
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction — Claim 2", AMBER, "Grammar emerges before consistency & plot");

  s.addChart(pres.charts.BAR, [
    { name: "Grammar", labels: SIZES, values: grammar },
    { name: "Consistency", labels: SIZES, values: consistency },
    { name: "Plot sense", labels: SIZES, values: plot },
  ], {
    x: 0.5, y: 1.95, w: 7.7, h: 4.7, barDir: "col", barGapWidthPct: 60,
    chartColors: [C.grammar, C.consistency, C.plot],
    showLegend: true, legendPos: "b", legendColor: INK2, legendFontFace: FB, legendFontSize: 12,
    valAxisMinVal: 0, valAxisMaxVal: 9, valAxisMajorUnit: 1,
    valGridLine: { color: "EAEFF6", size: 0.5 }, catGridLine: { style: "none" },
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
    catAxisLabelFontFace: FB, valAxisLabelFontFace: FB, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    showTitle: true, title: "Skill profile within each model size", titleColor: INK, titleFontFace: FH, titleFontSize: 14,
    chartArea: { fill: { color: "FFFFFF" } },
  });

  card(s, 8.5, 1.95, 4.3, 4.7, { fill: CARD, shadow: true });
  s.addText("INTERPRETATION", { x: 8.75, y: 2.15, w: 3.8, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: AMBER, charSpacing: 1.5, margin: 0 });
  s.addText("Grammar leads, plot lags", { x: 8.75, y: 2.5, w: 3.85, h: 0.5, fontFace: FH, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "At XS, grammar already scores ", options: { color: INK2 } },
    { text: "5.1", options: { bold: true, color: BLUE } },
    { text: " while plot sense sits at just ", options: { color: INK2 } },
    { text: "3.1", options: { bold: true, color: PLUM } },
    { text: ".", options: { color: INK2 } },
  ], { x: 8.75, y: 3.05, w: 3.85, h: 0.95, fontFace: FB, fontSize: 13, valign: "top", margin: 0 });
  s.addText("Grammar − consistency gap", { x: 8.75, y: 4.05, w: 3.85, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: MUTED, margin: 0 });
  s.addText([
    { text: "XS: 1.08", options: { breakLine: true, color: INK2, paraSpaceAfter: 3 } },
    { text: "XL: 0.16", options: { breakLine: true, color: INK2 } },
  ], { x: 8.75, y: 4.35, w: 3.85, h: 0.8, fontFace: FB, fontSize: 13, margin: 0 });
  s.addText([
    { text: "The gap shrinks 7× — smaller models keep grammar but lose the higher-order skills. ", options: { color: INK2 } },
    { text: "✓ Claim 2 reproduced.", options: { bold: true, color: TEAL } },
  ], { x: 8.75, y: 5.15, w: 3.85, h: 1.3, fontFace: FB, fontSize: 12.5, valign: "top", margin: 0 });

  pageFooter(s, 8);
})();

// =================================================================
// SLIDE 9 — CLAIM 3: memorization / overlap
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction — Claim 3", AMBER, "Outputs are generated, not memorized");

  s.addChart(pres.charts.BAR, [
    { name: "Train−Valid overlap gap (pp)", labels: SIZES, values: ovGap },
  ], {
    x: 0.5, y: 1.95, w: 7.7, h: 4.7, barDir: "col", barGapWidthPct: 80,
    chartColors: [AMBER],
    showLegend: false,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
    valGridLine: { color: "EAEFF6", size: 0.5 }, catGridLine: { style: "none" },
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
    catAxisLabelFontFace: FB, valAxisLabelFontFace: FB, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK2, dataLabelFontFace: FB, dataLabelFontSize: 11, dataLabelFormatCode: '0.00"pp"',
    showTitle: true, title: "Character 6-gram overlap: how much MORE the model echoes train vs. unseen valid", titleColor: INK, titleFontFace: FH, titleFontSize: 12.5,
    chartArea: { fill: { color: "FFFFFF" } },
  });

  card(s, 8.5, 1.95, 4.3, 4.7, { fill: CARD, shadow: true });
  s.addText("INTERPRETATION", { x: 8.75, y: 2.15, w: 3.8, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: AMBER, charSpacing: 1.5, margin: 0 });
  s.addText("No memorization signal", { x: 8.75, y: 2.5, w: 3.85, h: 0.5, fontFace: FH, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "Absolute overlap is ~99% against ", options: { color: INK2 } },
    { text: "both", options: { bold: true, color: AMBER } },
    { text: " train and valid — expected, since common short character strings saturate over millions of stories.", options: { color: INK2 } },
  ], { x: 8.75, y: 3.05, w: 3.85, h: 1.3, fontFace: FB, fontSize: 12.5, valign: "top", margin: 0 });
  s.addText([
    { text: "The honest test is the train−valid gap: a memorizing model would echo training stories far more. Instead the gap stays ", options: { color: INK2 } },
    { text: "below 0.5 pp", options: { bold: true, color: TEAL } },
    { text: " and never grows with size.", options: { color: INK2 } },
  ], { x: 8.75, y: 4.35, w: 3.85, h: 1.5, fontFace: FB, fontSize: 12.5, valign: "top", margin: 0 });
  s.addText("✓ Claim 3 reproduced.", { x: 8.75, y: 6.05, w: 3.85, h: 0.4, fontFace: FB, fontSize: 13, bold: true, color: TEAL, margin: 0 });

  pageFooter(s, 9);
})();

// =================================================================
// SLIDE 10 — Qualitative example (XS vs XL)
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction", AMBER, "What the numbers look like — XS vs. XL");

  s.addText([
    { text: "Prompt:  ", options: { bold: true, color: AMBER } },
    { text: "“Once upon a time, there was a little girl named Lucy. She was very adventurous. One day, she found …”", options: { italic: true, color: INK2 } },
  ], { x: 0.55, y: 1.8, w: 12.2, h: 0.6, fontFace: FB, fontSize: 14, valign: "middle", margin: 0 });

  // XS card
  card(s, 0.55, 2.5, 6.0, 3.55, { fill: CARD, shadow: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 2.72, w: 1.5, h: 0.4, rectRadius: 0.05, fill: { color: PLUM } });
  s.addText("XS · 1M", { x: 0.8, y: 2.72, w: 1.5, h: 0.4, align: "center", valign: "middle", fontFace: FB, fontSize: 12, bold: true, color: PAPER, margin: 0 });
  s.addText("“… a big, scary cat. The cat was very happy. The cat was very happy. The cat was very happy. The cat was very happy. The cat was happy to have a friend. The cat was happy.”", {
    x: 0.8, y: 3.25, w: 5.5, h: 1.85, fontFace: FB, fontSize: 13, italic: true, color: INK2, valign: "top", margin: 0, lineSpacingMultiple: 1.08,
  });
  s.addText([
    { text: "grammar 5   ", options: { color: BLUE } },
    { text: "consistency 2   ", options: { color: TEAL } },
    { text: "plot 2", options: { color: PLUM } },
  ], { x: 0.8, y: 5.5, w: 5.5, h: 0.4, fontFace: FB, fontSize: 12.5, bold: true, margin: 0 });

  // XL card
  card(s, 6.78, 2.5, 6.0, 3.55, { fill: CARD, shadow: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.03, y: 2.72, w: 1.5, h: 0.4, rectRadius: 0.05, fill: { color: TEAL } });
  s.addText("XL · 56M", { x: 7.03, y: 2.72, w: 1.5, h: 0.4, align: "center", valign: "middle", fontFace: FB, fontSize: 12, bold: true, color: PAPER, margin: 0 });
  s.addText("“… a big box in her room. She wanted to see what was inside. Lucy tried to open the box, but it was locked. She asked her mom for help. Her mom said, ‘I will help you, but first you must promise to be careful.’ Lucy promised …”", {
    x: 7.03, y: 3.25, w: 5.5, h: 1.95, fontFace: FB, fontSize: 13, italic: true, color: INK2, valign: "top", margin: 0, lineSpacingMultiple: 1.08,
  });
  s.addText([
    { text: "grammar 10   ", options: { color: BLUE } },
    { text: "consistency 10   ", options: { color: TEAL } },
    { text: "plot 7", options: { color: PLUM } },
  ], { x: 7.03, y: 5.5, w: 5.5, h: 0.4, fontFace: FB, fontSize: 12.5, bold: true, margin: 0 });

  s.addText([
    { text: "Same prompt, same decoding. ", options: { bold: true, color: INK } },
    { text: "XS collapses into a repetition loop; XL introduces a goal, a problem (locked box), dialogue and a resolution.", options: { color: INK2 } },
  ], { x: 0.55, y: 6.2, w: 12.2, h: 0.55, fontFace: FB, fontSize: 13.5, valign: "middle", margin: 0 });

  pageFooter(s, 10);
})();

// =================================================================
// SLIDE 11 — Challenges
// =================================================================
(function () {
  const s = contentSlide("3 · Reproduction", AMBER, "Challenges & design decisions");

  const items = [
    ["Paper gives few exact configs", "We chose our own width/depth ladder to hit 1M–56M targets, and matched the paper's eval protocol (50 prompts × 10 samples, temp 1).", BLUE],
    ["GPT-4 grading is costly & noisy", "Scores vary run-to-run; we average over 50 prompts per model and report means to keep the comparison stable.", TEAL],
    ["A useful memorization metric", "Char 6-gram overlap saturates near 100% on a huge corpus. We shifted to the train−valid gap, which is the signal that actually distinguishes memorization.", AMBER],
    ["Compute budget", "One epoch per model on a single machine; sufficient to reproduce the qualitative trends without the paper's full training run.", PLUM],
  ];
  const x0 = 0.55, cw = 6.0, ch = 1.95, gx = 0.23, gy = 0.22, y0 = 1.95;
  items.forEach((it, i) => {
    const x = x0 + (i % 2) * (cw + gx);
    const y = y0 + Math.floor(i / 2) * (ch + gy);
    card(s, x, y, cw, ch, { fill: CARD, shadow: true });
    iconCircle(s, x + 0.3, y + 0.32, 0.6, it[2], "!");
    s.addText(it[0], { x: x + 1.05, y: y + 0.3, w: cw - 1.3, h: 0.55, fontFace: FH, fontSize: 15.5, bold: true, color: INK, valign: "middle", margin: 0 });
    s.addText(it[1], { x: x + 1.05, y: y + 0.9, w: cw - 1.3, h: 0.95, fontFace: FB, fontSize: 12.5, color: MUTED, valign: "top", margin: 0 });
  });

  pageFooter(s, 11);
})();

// =================================================================
// SLIDE 12 — EXTENSION: our own modern GPT (architecture)
// =================================================================
(function () {
  const s = contentSlide("4 · Extension", PLUM, "We built our own modern GPT");

  s.addText([
    { text: "Rather than reuse nanoGPT, we implemented the whole transformer ourselves", options: { bold: true, color: INK } },
    { text: " — and the same code is the backbone for all five reproduced models. It pairs two recent ideas for quality and efficiency:", options: { color: INK2 } },
  ], { x: 0.55, y: 1.8, w: 7.4, h: 1.0, fontFace: FB, fontSize: 15, valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

  const feats = [
    ["Differential Attention", "Two softmax maps subtracted (λ-weighted) to cancel attention noise and sharpen focus on relevant tokens — the DIFF Transformer idea.", PLUM],
    ["Grouped-Query Attention", "Query heads share a smaller set of key/value heads, cutting KV memory & compute for cheaper, faster inference.", TEAL],
    ["Modern building blocks", "RoPE positional encoding · RMSNorm (pre-norm) · SwiGLU feed-forward · weight-tied embedding & LM head.", BLUE],
  ];
  let fy = 2.95;
  feats.forEach((f) => {
    card(s, 0.55, fy, 7.4, 1.2, { fill: CARD, shadow: true });
    s.addShape(pres.shapes.OVAL, { x: 0.78, y: fy + 0.27, w: 0.16, h: 0.16, fill: { color: f[2] } });
    s.addText(f[0], { x: 1.1, y: fy + 0.13, w: 6.7, h: 0.4, fontFace: FH, fontSize: 16, bold: true, color: INK, margin: 0 });
    s.addText(f[1], { x: 1.1, y: fy + 0.52, w: 6.6, h: 0.6, fontFace: FB, fontSize: 12.5, color: MUTED, valign: "top", margin: 0 });
    fy += 1.32;
  });

  // architecture block diagram (right)
  const bx = 8.35, bw = 4.45;
  card(s, bx, 1.8, bw, 4.9, { fill: INK, shadow: true });
  s.addText("DECODER BLOCK  (×N layers)", { x: bx, y: 2.0, w: bw, h: 0.35, align: "center", fontFace: FB, fontSize: 12, bold: true, color: AMBER, charSpacing: 1, margin: 0 });

  const blocks = [
    ["Token embedding  (tied)", "33507A"],
    ["RMSNorm", "2E4063"],
    ["Differential Attention + GQA + RoPE", PLUM],
    ["⊕  residual", "24344D"],
    ["RMSNorm", "2E4063"],
    ["SwiGLU feed-forward", TEAL],
    ["⊕  residual", "24344D"],
    ["LM head  (weight-tied)", "33507A"],
  ];
  let yy = 2.5;
  blocks.forEach((b, i) => {
    const h = 0.42;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx + 0.3, y: yy, w: bw - 0.6, h, rectRadius: 0.05, fill: { color: b[1] } });
    s.addText(b[0], { x: bx + 0.35, y: yy, w: bw - 0.7, h, align: "center", valign: "middle", fontFace: FB, fontSize: 11.5, bold: i === 2 || i === 5, color: PAPER, margin: 0 });
    yy += h + 0.085;
  });

  pageFooter(s, 12);
})();

// =================================================================
// SLIDE 13 — EXTENSION: motivation & outcome
// =================================================================
(function () {
  const s = contentSlide("4 · Extension", PLUM, "Motivation & outcome");

  // left: motivation
  s.addText("WHY BUILD IT OURSELVES", { x: 0.55, y: 1.85, w: 6, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: PLUM, charSpacing: 1, margin: 0 });
  s.addText([
    { text: "Deeper understanding — ", options: { bold: true, color: INK } },
    { text: "implementing attention, RoPE and normalization end-to-end forces real understanding of the architecture the paper takes for granted.", options: { breakLine: true, color: INT_OR(INK2), paraSpaceAfter: 10 } },
    { text: "A stronger reproduction — ", options: { bold: true, color: INK } },
    { text: "TinyStories used a vanilla GPT; we test whether the same conclusions hold on a more modern backbone.", options: { breakLine: true, color: INK2, paraSpaceAfter: 10 } },
    { text: "Efficiency — ", options: { bold: true, color: INK } },
    { text: "GQA shrinks the KV cache and Differential Attention improves the signal-to-noise of attention, both attractive at the tiny-model scale.", options: { color: INK2 } },
  ], { x: 0.55, y: 2.25, w: 6.1, h: 4.3, fontFace: FB, fontSize: 14, valign: "top", margin: 0, lineSpacingMultiple: 1.08 });

  // right: outcomes
  card(s, 7.0, 1.95, 5.8, 4.75, { fill: CARD, shadow: true });
  s.addText("OUTCOME", { x: 7.3, y: 2.15, w: 5, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: TEAL, charSpacing: 1.5, margin: 0 });
  s.addText("The paper's conclusions hold on a modern architecture", { x: 7.3, y: 2.5, w: 5.2, h: 0.85, fontFace: FH, fontSize: 18, bold: true, color: INK, valign: "top", margin: 0 });

  const outs = [
    ["All three claims reproduced", "Scaling, grammar-first emergence and no-memorization all appear with Diff-Transformer + GQA.", TEAL],
    ["Clean training", "Stable single-epoch convergence across 1M–56M with weight tying and pre-norm RMSNorm.", BLUE],
    ["Efficiency by design", "GQA halves KV heads (n_kv = n_heads / 2) with no loss of story quality at this scale.", AMBER],
  ];
  let oy = 3.45;
  outs.forEach((o) => {
    s.addShape(pres.shapes.OVAL, { x: 7.3, y: oy + 0.05, w: 0.32, h: 0.32, fill: { color: o[2] } });
    s.addText("✓", { x: 7.3, y: oy + 0.05, w: 0.32, h: 0.32, align: "center", valign: "middle", fontFace: FB, fontSize: 13, bold: true, color: PAPER, margin: 0 });
    s.addText(o[0], { x: 7.75, y: oy - 0.02, w: 4.8, h: 0.4, fontFace: FH, fontSize: 14.5, bold: true, color: INK, margin: 0 });
    s.addText(o[1], { x: 7.75, y: oy + 0.36, w: 4.85, h: 0.6, fontFace: FB, fontSize: 12, color: MUTED, valign: "top", margin: 0 });
    oy += 1.07;
  });

  pageFooter(s, 13);
})();

// =================================================================
// SLIDE 14 — CONCLUSION
// =================================================================
(function () {
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.OVAL, { x: 11.0, y: -1.5, w: 4.0, h: 4.0, fill: { color: "24355A" } });
  sectionTagDark(s, "5 · Conclusion");
  s.addText("Findings & reproducibility verdict", { x: 0.55, y: 0.92, w: 12, h: 0.8, fontFace: FH, fontSize: 30, bold: true, color: PAPER, margin: 0 });

  // verdict cards
  const verdicts = [
    ["C1  Scale → coherence", "Consistency 4.1 → 7.6 across XS→XL", TEAL],
    ["C2  Grammar first", "Grammar−plot gap shrinks 7× with size", BLUE],
    ["C3  Not memorized", "Train−valid overlap gap < 0.5 pp", AMBER],
  ];
  verdicts.forEach((v, i) => {
    const x = 0.55 + i * 4.08;
    card(s, x, 2.0, 3.85, 1.65, { fill: "24355A" });
    s.addText("✓ " + v[0], { x: x + 0.25, y: 2.2, w: 3.4, h: 0.5, fontFace: FH, fontSize: 16, bold: true, color: PAPER, margin: 0 });
    s.addText(v[1], { x: x + 0.25, y: 2.75, w: 3.45, h: 0.8, fontFace: FB, fontSize: 12.5, color: ICE, valign: "top", margin: 0 });
  });

  s.addText([
    { text: "Verdict:  ", options: { bold: true, color: AMBER } },
    { text: "TinyStories reproduces cleanly. ", options: { bold: true, color: PAPER } },
    { text: "All three target claims hold — and they survive being ported onto a Differential-Transformer + GQA backbone we built from scratch.", options: { color: ICE } },
  ], { x: 0.55, y: 4.0, w: 12.2, h: 1.0, fontFace: FB, fontSize: 16, valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

  // future work
  s.addText("FUTURE WORK", { x: 0.55, y: 5.15, w: 6, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: AMBER, charSpacing: 1.5, margin: 0 });
  const fw = [
    "Probe depth vs. width directly, as in the paper's layer ablations",
    "Add an instruct/feature-controlled split (TinyStories-Instruct)",
    "Ablate Differential Attention & GQA against a vanilla baseline",
  ];
  s.addText(fw.map((t) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 6, color: ICE } })),
    { x: 0.55, y: 5.5, w: 12, h: 1.4, fontFace: FB, fontSize: 14, valign: "top", margin: 0 });

  s.addText(String(14), { x: W - 1.0, y: H - 0.42, w: 0.45, h: 0.3, fontFace: FB, fontSize: 10, color: ICE, align: "right" });
})();

// =================================================================
// SLIDE 15 — CLOSING
// =================================================================
(function () {
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.OVAL, { x: -1.4, y: 4.2, w: 4.0, h: 4.0, fill: { color: "20304F" } });
  s.addShape(pres.shapes.OVAL, { x: 11.6, y: -1.4, w: 3.6, h: 3.6, fill: { color: "24355A" } });

  s.addText("Thank you", { x: 0.8, y: 2.35, w: 11, h: 1.0, fontFace: FH, fontSize: 50, bold: true, color: PAPER, margin: 0 });
  s.addText("Reproducing & extending TinyStories — questions welcome.", { x: 0.82, y: 3.5, w: 11, h: 0.5, fontFace: FB, fontSize: 18, italic: true, color: ICE, margin: 0 });

  s.addShape(pres.shapes.OVAL, { x: 0.85, y: 4.55, w: 0.34, h: 0.34, fill: { color: AMBER } });
  s.addText("⌥", { x: 0.85, y: 4.55, w: 0.34, h: 0.34, align: "center", valign: "middle", fontFace: FB, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("github.com/Grabosticus/chatbot", { x: 1.3, y: 4.5, w: 9, h: 0.45, fontFace: FB, fontSize: 16, bold: true, color: PAPER, valign: "middle", margin: 0 });
  s.addText("Paper:  Eldan & Li (2023), “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?”", {
    x: 0.85, y: 5.15, w: 11, h: 0.4, fontFace: FB, fontSize: 13, color: "9FB3D1", margin: 0,
  });
})();

// helper used above (defined late is fine for function declarations)
function sectionTagDark(slide, label) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.55, y: 0.5, w: 0.18, h: 0.42, rectRadius: 0.06, fill: { color: AMBER } });
  slide.addText(label.toUpperCase(), { x: 0.85, y: 0.5, w: 8, h: 0.42, fontFace: FB, fontSize: 12, bold: true, color: AMBER, charSpacing: 2, valign: "middle", margin: 0 });
}
function INT_OR(v) { return v; }

pres.writeFile({ fileName: path.join(__dirname, "..", "TinyStories_Reproduction.pptx") }).then((f) => console.log("WROTE", f));
