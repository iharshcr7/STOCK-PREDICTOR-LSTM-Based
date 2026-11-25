/* ============================================================
   STOCK MARKET INTELLIGENCE ENGINE
   10-slide widescreen academic/technical presentation
   Built with pptxgenjs (LAYOUT_WIDE = 13.333 x 7.5 in)
   ============================================================ */

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Computer Science and Engineering";
pres.subject = "AI-Based Stock Market Prediction and Distributed Analytics System";
pres.title = "Stock Market Intelligence Engine";
pres.company = "University Technical Project";
pres.lang = "en-US";

/* ---------------- design tokens ---------------- */
const C = {
  bg:      "0B1220", // dark navy background
  bg2:     "111C2E", // secondary background
  card:    "162338", // card surface
  card2:   "132033", // subtle card surface
  text:    "F1F5F9", // primary text
  text2:   "94A3B8", // secondary text
  text3:   "64748B", // tertiary text
  green:   "22C55E",
  red:     "EF4444",
  accent:  "6366F1",
  accent2: "818CF8",
  sky:     "38BDF8",
  amber:   "F59E0B",
  slate:   "64748B",
  border:  "2A3A54",
  grid:    "1A2940",
};
const F = { head: "Arial", body: "Calibri", mono: "Consolas" };
const W = 13.333;
const H = 7.5;

/* ---------------- generic helpers ---------------- */
function T(slide, str, x, y, w, h, o = {}) {
  slide.addText(str, {
    x, y, w, h,
    fontFace: o.face || F.body,
    fontSize: o.size || 14,
    bold: o.bold || false,
    italic: o.italic || false,
    color: o.color || C.text,
    align: o.align || "left",
    valign: o.valign || "top",
    margin: o.margin === undefined ? 0 : o.margin,
    lineSpacingMultiple: o.lsm === undefined ? 1.0 : o.lsm,
    charSpacing: o.spacing,
    breakLine: false,
    isTextBox: true,
  });
}

function card(slide, x, y, w, h, o = {}) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: o.radius === undefined ? 0.12 : o.radius,
    fill: { color: o.fill || C.card, transparency: o.trans || 0 },
    line: { color: o.line || C.border, width: o.lineW === undefined ? 0.8 : o.lineW },
  });
}

function rect(slide, x, y, w, h, fill, trans = 0, lineColor, lineW) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill, transparency: trans },
    line: lineColor ? { color: lineColor, width: lineW || 0.6 } : { width: 0 },
  });
}

function line(slide, x, y, w, h, color, width = 0.8) {
  // OOXML requires non-negative extents; use flip flags to preserve direction.
  const o = {
    x: w < 0 ? x + w : x,
    y: h < 0 ? y + h : y,
    w: Math.abs(w),
    h: Math.abs(h),
    line: { color, width },
  };
  if (w < 0) o.flipH = true;
  if (h < 0) o.flipV = true;
  slide.addShape(pres.ShapeType.line, o);
}

function dot(slide, x, y, d, color, trans = 0) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d,
    fill: { color, transparency: trans },
    line: { width: 0 },
  });
}

function arrowRight(slide, x, y, w, h, color) {
  slide.addShape(pres.ShapeType.rightArrow, {
    x, y, w, h,
    fill: { color }, line: { width: 0 },
  });
}

function arrowLeft(slide, x, y, w, h, color) {
  slide.addShape(pres.ShapeType.leftArrow, {
    x, y, w, h,
    fill: { color }, line: { width: 0 },
  });
}

function arrowDown(slide, x, y, w, h, color) {
  slide.addShape(pres.ShapeType.downArrow, {
    x, y, w, h,
    fill: { color }, line: { width: 0 },
  });
}

function chip(slide, label, x, y, w, h, o = {}) {
  card(slide, x, y, w, h, {
    fill: o.fill || C.card2,
    line: o.line || C.border,
    radius: 0.08,
    lineW: 0.6,
  });
  T(slide, label, x + 0.04, y, w - 0.08, h, {
    size: o.size || 10,
    color: o.color || C.text2,
    bold: o.bold || false,
    align: "center",
    valign: "middle",
    spacing: o.spacing,
  });
}

function phaseChip(slide, label, x, y, w, color) {
  card(slide, x, y, w, 0.27, { fill: color, line: color, radius: 0.06, trans: 78, lineW: 0.6 });
  T(slide, label, x, y, w, 0.27, {
    size: 8.5, bold: true, color, align: "center", valign: "middle", spacing: 1,
  });
}

function badge(slide, n, x, y, d, color, txtColor) {
  dot(slide, x, y, d, color);
  T(slide, String(n).padStart(2, "0"), x, y, d, d, {
    size: 10.5, bold: true, color: txtColor || C.bg, align: "center", valign: "middle",
    face: F.head,
  });
}

function bulletList(slide, items, x, y, w, lineH, o = {}) {
  items.forEach((it, i) => {
    const yy = y + i * lineH;
    dot(slide, x, yy + 0.085, 0.075, o.dot || C.accent2);
    T(slide, it, x + 0.22, yy, w - 0.22, lineH, {
      size: o.size || 12.2,
      color: o.color || C.text2,
      valign: "top",
      lsm: 1.02,
    });
  });
}

function checkMark(slide, x, y, color) {
  line(slide, x, y + 0.055, 0.045, 0.055, color, 1.8);
  line(slide, x + 0.045, y + 0.11, 0.075, -0.085, color, 1.8);
}

/* ---------------- small shape-based icon set ---------------- */
function icon(slide, kind, x, y, s, color) {
  switch (kind) {
    case "database": {
      slide.addShape(pres.ShapeType.can, {
        x, y: y + s * 0.12, w: s, h: s * 0.76,
        fill: { color, transparency: 88 }, line: { color, width: 1.3 },
      });
      line(slide, x + s * 0.03, y + s * 0.36, s * 0.94, 0, color, 1.1);
      break;
    }
    case "spark": {
      slide.addShape(pres.ShapeType.star5, {
        x: x + s * 0.06, y: y + s * 0.06, w: s * 0.88, h: s * 0.88,
        fill: { color, transparency: 82 }, line: { color, width: 1.3 },
      });
      break;
    }
    case "brain": {
      slide.addShape(pres.ShapeType.roundRect, {
        x: x + s * 0.08, y: y + s * 0.16, w: s * 0.84, h: s * 0.68,
        rectRadius: 0.08, fill: { color, transparency: 88 }, line: { color, width: 1.3 },
      });
      dot(slide, x + s * 0.26, y + s * 0.34, s * 0.18, color, 20);
      dot(slide, x + s * 0.56, y + s * 0.34, s * 0.18, color, 20);
      line(slide, x + s * 0.5, y + s * 0.7, 0, s * 0.12, color, 1.2);
      break;
    }
    case "layers": {
      for (let i = 0; i < 3; i++) {
        slide.addShape(pres.ShapeType.diamond, {
          x, y: y + i * s * 0.22, w: s, h: s * 0.42,
          fill: { color, transparency: 82 + i * 5 }, line: { color, width: 1.1 },
        });
      }
      break;
    }
    case "chart": {
      line(slide, x, y + s * 0.85, s * 0.9, 0, color, 1.2);
      line(slide, x, y + s * 0.1, 0, s * 0.75, color, 1.2);
      rect(slide, x + s * 0.18, y + s * 0.45, s * 0.16, s * 0.4, color, 25);
      rect(slide, x + s * 0.44, y + s * 0.25, s * 0.16, s * 0.6, color, 25);
      rect(slide, x + s * 0.7, y + s * 0.5, s * 0.16, s * 0.35, color, 25);
      break;
    }
    case "cloud": {
      dot(slide, x + s * 0.05, y + s * 0.35, s * 0.46, color, 86);
      dot(slide, x + s * 0.36, y + s * 0.2, s * 0.5, color, 86);
      dot(slide, x + s * 0.5, y + s * 0.4, s * 0.45, color, 86);
      break;
    }
    case "search": {
      slide.addShape(pres.ShapeType.ellipse, {
        x: x + s * 0.08, y: y + s * 0.08, w: s * 0.58, h: s * 0.58,
        fill: { color, transparency: 90 }, line: { color, width: 1.4 },
      });
      line(slide, x + s * 0.58, y + s * 0.58, s * 0.3, s * 0.3, color, 1.6);
      break;
    }
    case "flow": {
      dot(slide, x, y + s * 0.34, s * 0.3, color, 35);
      dot(slide, x + s * 0.35, y + s * 0.34, s * 0.3, color, 65);
      dot(slide, x + s * 0.7, y + s * 0.34, s * 0.3, color, 35);
      line(slide, x + s * 0.28, y + s * 0.49, s * 0.1, 0, color, 1.2);
      line(slide, x + s * 0.63, y + s * 0.49, s * 0.1, 0, color, 1.2);
      break;
    }
    case "server": {
      slide.addShape(pres.ShapeType.roundRect, {
        x: x + s * 0.05, y: y + s * 0.1, w: s * 0.9, h: s * 0.32,
        rectRadius: 0.05, fill: { color, transparency: 88 }, line: { color, width: 1.2 },
      });
      slide.addShape(pres.ShapeType.roundRect, {
        x: x + s * 0.05, y: y + s * 0.55, w: s * 0.9, h: s * 0.32,
        rectRadius: 0.05, fill: { color, transparency: 88 }, line: { color, width: 1.2 },
      });
      dot(slide, x + s * 0.15, y + s * 0.21, s * 0.09, color);
      dot(slide, x + s * 0.15, y + s * 0.66, s * 0.09, color);
      break;
    }
    case "shield": {
      slide.addShape(pres.ShapeType.pentagon, {
        x: x + s * 0.08, y: y + s * 0.05, w: s * 0.84, h: s * 0.9,
        fill: { color, transparency: 88 }, line: { color, width: 1.3 },
      });
      break;
    }
    default: {
      dot(slide, x + s * 0.1, y + s * 0.1, s * 0.8, color, 80);
    }
  }
}

/* ---------------- frame helpers ---------------- */
function header(slide, kicker, title, page) {
  T(slide, kicker.toUpperCase(), 0.62, 0.32, 10.0, 0.26, {
    size: 10, bold: true, color: C.accent2, spacing: 1.6,
  });
  T(slide, title, 0.62, 0.61, 11.2, 0.63, {
    size: 27, bold: true, color: C.text, face: F.head, valign: "middle",
  });
  T(slide, String(page).padStart(2, "0") + " / 10", 11.55, 0.36, 1.15, 0.28, {
    size: 10, color: C.text3, align: "right", spacing: 0.6,
  });
}

function footer(slide, page) {
  line(slide, 0.62, 6.96, 12.1, 0, C.grid, 0.8);
  T(slide, "STOCK MARKET INTELLIGENCE ENGINE", 0.62, 7.03, 8.0, 0.22, {
    size: 8, color: C.text3, spacing: 1.1,
  });
  T(slide, String(page).padStart(2, "0"), 12.1, 7.03, 0.6, 0.22, {
    size: 8, color: C.text3, align: "right",
  });
}

function baseSlide(kicker, title, page) {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, kicker, title, page);
  footer(s, page);
  return s;
}

/* ============================================================
   SLIDE 1 — TITLE
   ============================================================ */
(function () {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // faint market grid + candlestick decoration (right side only)
  for (let i = 0; i < 7; i++) line(s, 8.55, 1.25 + i * 0.62, 4.25, 0, C.grid, 0.7);
  for (let i = 0; i < 8; i++) line(s, 8.55 + i * 0.6, 1.25, 0, 3.72, C.grid, 0.7);

  const candles = [0.62, 0.78, 0.55, 0.92, 0.7, 1.05, 0.84, 1.18, 0.96, 1.32, 1.1, 1.45, 1.24, 1.6, 1.38, 1.72];
  candles.forEach((v, i) => {
    const cx = 8.72 + i * 0.245;
    const yTop = 4.35 - v * 0.42;
    const yBot = 4.35 - (v - 0.42) * 0.42;
    const up = i % 3 !== 1;
    const col = up ? C.green : C.red;
    line(s, cx + 0.055, yTop - 0.16, 0, (yBot - yTop) + 0.32, col, 0.9);
    const bodyTop = Math.min(yTop, yBot);
    const bodyH = Math.max(0.1, Math.abs(yBot - yTop));
    rect(s, cx, bodyTop, 0.11, bodyH, col, 28);
  });
  // faint trend line
  for (let i = 0; i < candles.length - 1; i++) {
    const x1 = 8.77 + i * 0.245;
    const x2 = 8.77 + (i + 1) * 0.245;
    const y1 = 3.85 - candles[i] * 0.22;
    const y2 = 3.85 - candles[i + 1] * 0.22;
    line(s, x1, y1, x2 - x1, y2 - y1, C.accent2, 1);
  }

  T(s, "UNIVERSITY TECHNICAL PROJECT   /   2026–2027", 0.72, 0.62, 7.6, 0.3, {
    size: 10.5, bold: true, color: C.accent2, spacing: 1.7,
  });
  T(s, "STOCK MARKET\nINTELLIGENCE ENGINE", 0.72, 1.25, 7.75, 1.85, {
    size: 35, bold: true, color: C.text, face: F.head, lsm: 0.98, valign: "middle",
  });
  T(s, "An AI-Based Stock Market Prediction and Distributed Analytics System",
    0.72, 3.18, 7.9, 0.72, { size: 15.5, color: C.text2, lsm: 1.05 });

  const tech = ["Apache Hadoop", "Apache Spark", "LSTM Deep Learning", "MongoDB", "Streamlit"];
  let cx = 0.72;
  tech.forEach((t) => {
    const w = 0.28 + t.length * 0.083;
    chip(s, t, cx, 4.12, w, 0.34, { size: 9.5, color: C.text2, fill: C.card2, line: C.border });
    cx += w + 0.12;
  });

  card(s, 0.72, 4.85, 11.9, 1.62, { fill: C.card, line: C.border });
  T(s, "STUDENT DETAILS", 0.95, 5.03, 3.4, 0.22, { size: 9, bold: true, color: C.accent2, spacing: 1.3 });

  const fieldsRow1 = [
    ["STUDENT NAME", "[Student Name]"],
    ["ENROLLMENT NUMBER", "[Enrollment Number]"],
    ["PROJECT GUIDE", "[Project Guide]"],
  ];
  fieldsRow1.forEach((f, i) => {
    const fx = 0.95 + i * 3.85;
    T(s, f[0], fx, 5.32, 3.4, 0.2, { size: 8.5, color: C.text3, spacing: 1 });
    T(s, f[1], fx, 5.55, 3.5, 0.3, { size: 12, color: C.text, bold: true });
  });
  T(s, "DEPARTMENT", 0.95, 5.97, 2.0, 0.2, { size: 8.5, color: C.text3, spacing: 1 });
  T(s, "Computer Science and Engineering", 0.95, 6.17, 6.4, 0.24, { size: 11.5, color: C.text });
  T(s, "ACADEMIC YEAR", 7.45, 5.97, 2.0, 0.2, { size: 8.5, color: C.text3, spacing: 1 });
  T(s, "2026–2027", 7.45, 6.17, 3.0, 0.24, { size: 11.5, color: C.text });

  s.addNotes(
    "Title slide. Introduce the project as a university technical project submitted in the 2026–2027 academic year. " +
    "The system combines Hadoop, Spark, an existing LSTM checkpoint, MongoDB and Streamlit. " +
    "Replace the bracketed student name, enrollment number and project guide before submission. " +
    "Emphasise that the deep-learning component is an already-trained model used for inference, not a newly trained model."
  );
})();

/* ============================================================
   SLIDE 2 — INTRODUCTION & PROBLEM STATEMENT
   ============================================================ */
(function () {
  const s = baseSlide("01 / Context", "Introduction & Problem Statement", 2);

  card(s, 0.62, 1.42, 12.1, 0.92, { fill: C.card2, line: C.border });
  icon(s, "flow", 0.88, 1.62, 0.42, C.accent2);
  T(s, "OVERVIEW", 1.42, 1.58, 1.9, 0.24, { size: 9.5, bold: true, color: C.accent2, spacing: 1.2 });
  T(s,
    "An integrated platform that processes large-scale historical and updated stock data, generates AI-based directional predictions, and delivers interactive financial analytics.",
    1.42, 1.86, 10.9, 0.36, { size: 12.2, color: C.text2 });

  // Problem card
  card(s, 0.62, 2.58, 5.5, 3.9, { fill: C.card, line: C.border });
  phaseChip(s, "PROBLEM STATEMENT", 0.88, 2.84, 2.05, C.red);
  bulletList(s, [
    "Large volumes of historical and continuously changing market data.",
    "Processing thousands of stock records efficiently is challenging.",
    "Manual analysis requires significant time and effort.",
    "Market information is distributed across multiple sources.",
    "A unified platform is needed for distributed processing and prediction.",
  ], 0.9, 3.32, 5.0, 0.59, { dot: C.red, size: 12 });

  arrowRight(s, 6.32, 4.35, 0.52, 0.34, C.accent);

  // Solution card
  card(s, 7.05, 2.58, 5.67, 3.9, { fill: C.card, line: C.border });
  phaseChip(s, "PROPOSED SOLUTION", 7.31, 2.84, 2.15, C.green);
  const sol = [
    ["Distributed storage", "Hadoop HDFS", "database"],
    ["Parallel processing", "Apache Spark", "spark"],
    ["Deep-learning inference", "Existing trained LSTM", "brain"],
    ["Interactive analytics", "Streamlit dashboard", "chart"],
  ];
  sol.forEach((o, i) => {
    const yy = 3.3 + i * 0.76;
    icon(s, o[2], 7.34, yy + 0.02, 0.34, C.green);
    T(s, o[0], 7.82, yy, 4.6, 0.26, { size: 12.4, bold: true, color: C.text });
    T(s, o[1], 7.82, yy + 0.27, 4.6, 0.24, { size: 10.8, color: C.text2 });
  });

  s.addNotes(
    "Motivate the project: stock datasets are large and constantly updated. A single machine and manual spreadsheet work do not scale, " +
    "and the data lives in several places. The problem statement has five points; walk through them briefly. " +
    "Then use the arrow to frame the solution: one pipeline that stores data in HDFS, processes it with Spark, " +
    "runs inference with an existing LSTM checkpoint, and exposes results through a Streamlit dashboard. " +
    "Do not claim the pipeline is fully complete; this slide is the intended design."
  );
})();

/* ============================================================
   SLIDE 3 — PROJECT OBJECTIVES
   ============================================================ */
(function () {
  const s = baseSlide("02 / Goals", "Project Objectives", 3);

  const objectives = [
    ["01", "Data Collection", "Assemble large-scale historical stock market datasets.", "cloud", C.sky],
    ["02", "HDFS Storage", "Store raw datasets reliably in Hadoop HDFS.", "database", C.accent2],
    ["03", "Spark Processing", "Apply distributed, parallel data processing.", "spark", C.sky],
    ["04", "Feature Engineering", "Derive technical indicators and predictive features.", "layers", C.accent2],
    ["05", "LSTM Inference", "Reuse an existing trained model for direction prediction.", "brain", C.green],
    ["06", "MongoDB Storage", "Persist prediction records with full metadata.", "server", C.amber],
    ["07", "Streamlit Dashboard", "Deliver interactive financial analytics.", "chart", C.green],
    ["08", "Large Stock Universe", "Support analysis across thousands of stocks and ETFs.", "flow", C.accent2],
  ];

  const cw = 2.87, gap = 0.25;
  const left = (W - (4 * cw + 3 * gap)) / 2;
  objectives.forEach((o, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = left + col * (cw + gap);
    const y = 1.45 + row * 2.38;
    card(s, x, y, cw, 2.18, { fill: C.card, line: C.border });
    badge(s, o[0], x + 0.24, y + 0.22, 0.43, o[4], C.bg);
    icon(s, o[3], x + cw - 0.66, y + 0.24, 0.4, o[4]);
    T(s, o[1], x + 0.24, y + 0.82, cw - 0.48, 0.44, {
      size: 13.2, bold: true, color: C.text, face: F.head,
    });
    T(s, o[2], x + 0.24, y + 1.28, cw - 0.48, 0.72, {
      size: 10.6, color: C.text2, lsm: 1.05,
    });
  });

  card(s, 0.62, 6.36, 12.1, 0.42, { fill: C.card2, line: C.border, radius: 0.08 });
  T(s, "EIGHT CONNECTED OBJECTIVES  •  FROM RAW MARKET DATA TO AN INTERACTIVE ANALYTICS DASHBOARD",
    0.86, 6.36, 11.6, 0.42, { size: 10, color: C.accent2, bold: true, align: "center", valign: "middle", spacing: 0.9 });

  s.addNotes(
    "These eight objectives map directly onto the pipeline. Point out the ordering: collection, storage, distributed processing, " +
    "feature engineering, reuse of the trained LSTM, persistence in MongoDB, visualisation in Streamlit, and coverage of a large stock universe. " +
    "Objective five is deliberately worded as reuse of an existing model. The large-universe objective is a target, not a completed result."
  );
})();

/* ============================================================
   SLIDE 4 — SYSTEM ARCHITECTURE
   ============================================================ */
(function () {
  const s = baseSlide("03 / System Design", "System Architecture", 4);
  T(s, "End-to-end workflow — from market data sources to the analytics dashboard",
    0.62, 1.19, 11.0, 0.26, { size: 11, color: C.text2 });

  const cw = 3.65, ch = 1.42;
  const xs = [0.84, 4.84, 8.84];
  const ys = [1.55, 3.27, 4.99];

  const stages = [
    { n: 1, row: 0, col: 0, name: "MARKET DATA\nSOURCES", phase: "DATA SOURCE", color: C.slate, desc: "Historical stock and ETF market data." },
    { n: 2, row: 0, col: 1, name: "DATA\nACQUISITION", phase: "DATA SOURCE", color: C.slate, desc: "Collect and organise incoming datasets." },
    { n: 3, row: 0, col: 2, name: "HADOOP HDFS", phase: "RAW STORAGE", color: C.accent2, desc: "Distributed storage of raw data." },
    { n: 4, row: 1, col: 2, name: "APACHE SPARK", phase: "PROCESSING", color: C.sky, desc: "Distributed preprocessing at scale." },
    { n: 5, row: 1, col: 1, name: "FEATURE ENGINEERING\n& VALIDATION", phase: "PROCESSING", color: C.sky, desc: "Indicators, features and validation." },
    { n: 6, row: 1, col: 0, name: "HDFS — PROCESSED\nPARQUET", phase: "PROCESSED STORAGE", color: C.accent2, desc: "Columnar processed dataset storage." },
    { n: 7, row: 2, col: 0, name: "EXISTING LSTM\nMODEL", phase: "AI INFERENCE", color: C.green, desc: "Binary UP / DOWN inference." },
    { n: 8, row: 2, col: 1, name: "MONGODB", phase: "PREDICTION STORAGE", color: C.amber, desc: "Persist predictions with metadata." },
    { n: 9, row: 2, col: 2, name: "STREAMLIT", phase: "VISUALISATION", color: C.green, desc: "Interactive analytics dashboard." },
  ];

  stages.forEach((st) => {
    const x = xs[st.col];
    const y = ys[st.row];
    card(s, x, y, cw, ch, { fill: C.card, line: C.border });
    rect(s, x + 0.18, y + 0.2, 0.055, 0.24, st.color);
    phaseChip(s, st.phase, x + 0.32, y + 0.19, 1.86, st.color);
    badge(s, st.n, x + cw - 0.52, y + 0.18, 0.34, st.color, C.bg);
    T(s, st.name, x + 0.22, y + 0.55, cw - 0.44, 0.48, {
      size: 12.2, bold: true, color: C.text, face: F.head, lsm: 0.95,
    });
    T(s, st.desc, x + 0.22, y + 1.04, cw - 0.44, 0.28, {
      size: 9.8, color: C.text2,
    });
  });

  // snake-flow arrows
  arrowRight(s, 4.52, 2.14, 0.26, 0.22, C.text3);
  arrowRight(s, 8.52, 2.14, 0.26, 0.22, C.text3);
  arrowLeft(s, 4.52, 3.86, 0.26, 0.22, C.text3);
  arrowLeft(s, 8.52, 3.86, 0.26, 0.22, C.text3);
  arrowDown(s, 10.55, 2.98, 0.22, 0.27, C.text3);
  arrowDown(s, 2.55, 4.70, 0.22, 0.27, C.text3);

  // legend
  const legend = [
    ["SOURCE", C.slate], ["STORAGE", C.accent2], ["PROCESSING", C.sky], ["AI / OUTPUT", C.green],
  ];
  let lx = 0.84;
  legend.forEach((l) => {
    dot(s, lx, 6.62, 0.13, l[1]);
    T(s, l[0], lx + 0.2, 6.57, 1.55, 0.24, { size: 9, color: C.text2, bold: true, spacing: 0.8 });
    lx += 2.35;
  });

  s.addNotes(
    "This is the most important slide. Follow the numbers in order: sources, acquisition, HDFS raw storage, Spark processing, " +
    "feature engineering and validation, processed Parquet back in HDFS, the existing LSTM checkpoint for inference, MongoDB for prediction records, " +
    "and finally Streamlit. The arrows snake across three rows, so trace stage three down to four, then right to left on the middle row, " +
    "then down to seven and left to right again. The colours distinguish data sources, storage layers, processing, and AI/output stages. " +
    "Note that the AI stage is inference from a saved checkpoint, not training."
  );
})();

/* ============================================================
   SLIDE 5 — DATASET & FEATURE ENGINEERING
   ============================================================ */
(function () {
  const s = baseSlide("04 / Data Layer", "Dataset & Feature Engineering", 5);

  // dataset statistics
  const stats = [
    ["5,884", "Historical stock CSV files", C.accent2],
    ["2,165", "ETF CSV files", C.sky],
    ["≈ 2.33 GB", "Stock data volume", C.green],
    ["≈ 417 MB", "ETF data volume", C.amber],
  ];
  const sw = 2.82, sgap = 0.34;
  const sleft = (W - (4 * sw + 3 * sgap)) / 2;
  stats.forEach((st, i) => {
    const x = sleft + i * (sw + sgap);
    card(s, x, 1.42, sw, 0.98, { fill: C.card, line: C.border });
    T(s, st[0], x + 0.2, 1.55, sw - 0.4, 0.42, {
      size: 21, bold: true, color: st[2], face: F.head,
    });
    T(s, st[1], x + 0.2, 2.02, sw - 0.4, 0.24, { size: 10.2, color: C.text2 });
  });

  // LEFT: composition chart
  card(s, 0.62, 2.62, 5.85, 1.98, { fill: C.card, line: C.border });
  T(s, "DATASET COMPOSITION", 0.86, 2.78, 4.0, 0.24, {
    size: 9.5, bold: true, color: C.accent2, spacing: 1.1,
  });
  // shape-based horizontal bar chart (fully editable, no chart XML)
  const bars = [
    ["Historical stocks", 5884, C.accent, "5,884"],
    ["ETF files", 2165, C.sky, "2,165"],
  ];
  bars.forEach((b, i) => {
    const y = 3.16 + i * 0.58;
    T(s, b[0], 0.86, y + 0.02, 1.42, 0.24, { size: 9.6, color: C.text2, valign: "middle" });
    rect(s, 2.38, y, 3.35, 0.3, C.card2);
    rect(s, 2.38, y, 3.35 * (b[1] / 7000), 0.3, b[2]);
    T(s, b[3], 2.44 + 3.35 * (b[1] / 7000), y, 0.72, 0.3, {
      size: 9.5, bold: true, color: b[2], valign: "middle", face: F.head,
    });
  });
  line(s, 2.38, 4.4, 3.35, 0, C.border, 0.7);
  T(s, "FILE COUNT  •  SCALE 0–7,000", 0.86, 4.36, 1.45, 0.22, {
    size: 7.6, color: C.text3,
  });

  // LEFT: OHLCV
  card(s, 0.62, 4.73, 5.85, 0.96, { fill: C.card, line: C.border });
  T(s, "OHLCV FIELDS", 0.86, 4.86, 2.0, 0.22, { size: 9.5, bold: true, color: C.accent2, spacing: 1.1 });
  const ohlcv = ["Open", "High", "Low", "Close", "Adj Close", "Volume"];
  ohlcv.forEach((o, i) => {
    chip(s, o, 0.86 + i * 0.91, 5.15, 0.82, 0.34, { size: 9.2, color: C.text, fill: C.card2 });
  });

  // LEFT: indicators
  card(s, 0.62, 5.81, 5.85, 0.96, { fill: C.card, line: C.border });
  T(s, "TECHNICAL INDICATORS", 0.86, 5.94, 3.0, 0.22, { size: 9.5, bold: true, color: C.accent2, spacing: 1.1 });
  const inds = ["SMA", "EMA", "MACD", "RSI", "Bollinger", "Returns"];
  inds.forEach((o, i) => {
    chip(s, o, 0.86 + i * 0.91, 6.23, 0.82, 0.34, { size: 9.2, color: C.text, fill: C.card2 });
  });

  // RIGHT: preprocessing workflow
  card(s, 6.72, 2.62, 6.0, 3.38, { fill: C.card, line: C.border });
  T(s, "PREPROCESSING WORKFLOW", 6.96, 2.78, 4.2, 0.24, {
    size: 9.5, bold: true, color: C.accent2, spacing: 1.1,
  });
  const steps = [
    "Data cleaning",
    "Duplicate removal",
    "Missing-value validation",
    "Date normalisation",
    "Technical indicator generation",
    "Feature engineering",
    "Time-series sequence preparation",
  ];
  steps.forEach((st, i) => {
    const col = i < 4 ? 0 : 1;
    const row = i < 4 ? i : i - 4;
    const x = 6.96 + col * 2.85;
    const y = 3.18 + row * 0.62;
    badge(s, i + 1, x, y, 0.34, C.accent, C.bg);
    T(s, st, x + 0.46, y - 0.01, 2.35, 0.42, { size: 10.8, color: C.text2, valign: "middle" });
  });

  // highlight
  card(s, 6.72, 6.13, 6.0, 0.64, { fill: C.card2, line: C.accent, lineW: 1 });
  T(s, "21", 6.94, 6.2, 0.95, 0.48, { size: 26, bold: true, color: C.green, face: F.head });
  T(s, "engineered input features are used by the existing LSTM model",
    7.95, 6.24, 4.6, 0.42, { size: 11.2, color: C.text, bold: true, valign: "middle" });

  s.addNotes(
    "Give the inventory first: 5,884 historical stock CSV files and 2,165 ETF CSV files, about 2.33 GB and 417 MB respectively. " +
    "Each record carries open, high, low, close, adjusted close and volume. The preprocessing workflow cleans data, removes duplicates, " +
    "validates missing values, normalises dates, derives indicators, engineers features and finally prepares fixed-length time-series sequences. " +
    "Indicators include SMA, EMA, MACD, RSI, Bollinger Bands and daily returns. The critical number is 21: the existing LSTM expects exactly 21 input features per time step. " +
    "Do not say the model was trained on all 5,884 stocks; it is only applied across this universe."
  );
})();

/* ============================================================
   SLIDE 6 — HADOOP & SPARK
   ============================================================ */
(function () {
  const s = baseSlide("05 / Big Data Layer", "Hadoop & Spark Distributed Processing", 6);

  const cols = [
    {
      title: "APACHE HADOOP", tag: "DISTRIBUTED STORAGE", color: C.accent2, icon: "database",
      items: ["HDFS distributed storage", "Historical data management", "Raw data organisation", "Processed dataset storage"],
    },
    {
      title: "APACHE SPARK", tag: "PARALLEL PROCESSING", color: C.sky, icon: "spark",
      items: ["Distributed data reading", "Cleaning and transformation", "Feature engineering at scale", "Batch processing → Parquet"],
    },
    {
      title: "PARQUET", tag: "ANALYTICAL FORMAT", color: C.green, icon: "layers",
      items: ["Columnar storage format", "Efficient analytical reads", "Compressed processed datasets", "Structured data organisation"],
    },
  ];

  const cw = 3.85, gap = 0.25;
  const left = (W - (3 * cw + 2 * gap)) / 2;
  cols.forEach((c, i) => {
    const x = left + i * (cw + gap);
    card(s, x, 1.45, cw, 2.55, { fill: C.card, line: C.border });
    icon(s, c.icon, x + 0.26, 1.7, 0.46, c.color);
    T(s, c.title, x + 0.86, 1.68, cw - 1.1, 0.32, {
      size: 14.4, bold: true, color: C.text, face: F.head,
    });
    T(s, c.tag, x + 0.86, 2.03, cw - 1.1, 0.22, {
      size: 9.5, color: c.color, bold: true, spacing: 0.9,
    });
    bulletList(s, c.items, x + 0.32, 2.48, cw - 0.62, 0.36, { dot: c.color, size: 11.5 });
  });

  T(s, "PROCESSING FLOW", 0.64, 4.25, 4.0, 0.25, {
    size: 9.5, bold: true, color: C.accent2, spacing: 1.2,
  });

  const flow = [
    ["HDFS RAW", "Ingested datasets", C.accent2],
    ["SPARK PROCESSING", "Distributed transforms", C.sky],
    ["FEATURE ENGINEERING", "Indicators + sequences", C.accent],
    ["HDFS PARQUET", "Processed dataset", C.green],
  ];
  const fw = 2.68, fgap = 0.55;
  const fleft = (W - (4 * fw + 3 * fgap)) / 2;
  flow.forEach((f, i) => {
    const x = fleft + i * (fw + fgap);
    card(s, x, 4.62, fw, 1.05, { fill: C.card2, line: f[2], lineW: 1 });
    dot(s, x + 0.18, 4.84, 0.14, f[2]);
    T(s, f[0], x + 0.42, 4.79, fw - 0.6, 0.28, {
      size: 11.6, bold: true, color: C.text, face: F.head,
    });
    T(s, f[1], x + 0.42, 5.13, fw - 0.6, 0.24, { size: 10, color: C.text2 });
    if (i < 3) arrowRight(s, x + fw + 0.1, 4.99, 0.36, 0.25, C.text3);
  });

  card(s, 0.64, 5.98, 12.06, 0.66, { fill: C.card2, line: C.border, radius: 0.08 });
  icon(s, "shield", 0.88, 6.12, 0.36, C.amber);
  T(s, "Architecture description only — no measured speed improvement is claimed without benchmark evidence.",
    1.38, 6.12, 11.0, 0.36, { size: 11.2, color: C.text2, valign: "middle" });

  s.addNotes(
    "Hadoop provides the storage layer: HDFS holds raw datasets and later the processed Parquet output. " +
    "Spark does the compute: reading data in parallel, cleaning it, engineering features and writing batches. " +
    "Parquet is chosen because its columnar layout suits analytical reads and compresses well. " +
    "The flow diagram shows the hand-off from HDFS raw to Spark, to feature engineering, and back to HDFS as Parquet. " +
    "Be careful in the viva: this slide describes the intended architecture, not measured performance. We do not claim any speed-up without benchmark evidence."
  );
})();

/* ============================================================
   SLIDE 7 — EXISTING LSTM MODEL
   ============================================================ */
(function () {
  const s = baseSlide("06 / AI Inference", "Existing LSTM Stock Prediction Model", 7);

  // configuration table
  card(s, 0.62, 1.45, 5.45, 3.18, { fill: C.card, line: C.border });
  T(s, "MODEL CONFIGURATION", 0.88, 1.64, 4.2, 0.24, {
    size: 9.5, bold: true, color: C.accent2, spacing: 1.1,
  });
  const rows = [
    ["Model", "LSTM"],
    ["Sequence length", "60 sessions"],
    ["Input features", "21"],
    ["Input shape", "(None, 60, 21)"],
    ["Output shape", "(None, 1)"],
    ["Activation", "Sigmoid"],
    ["Classification", "Binary"],
    ["Threshold", "0.5"],
  ];
  rows.forEach((r, i) => {
    const y = 2.0 + i * 0.31;
    if (i % 2 === 0) rect(s, 0.86, y - 0.03, 4.94, 0.3, C.card2, 20);
    T(s, r[0], 0.99, y, 2.5, 0.24, { size: 10.8, color: C.text2, valign: "middle" });
    T(s, r[1], 3.5, y, 2.2, 0.24, {
      size: 10.8, bold: true, color: i === 0 || i === 2 || i === 6 ? C.accent2 : C.text,
      align: "right", valign: "middle", face: F.mono,
    });
  });

  // explanation
  card(s, 6.3, 1.45, 6.42, 3.18, { fill: C.card, line: C.border });
  T(s, "60 × 21", 6.58, 1.72, 3.2, 0.72, {
    size: 38, bold: true, color: C.green, face: F.head,
  });
  T(s, "INPUT WINDOW", 9.55, 1.94, 2.6, 0.24, {
    size: 9.5, bold: true, color: C.text3, spacing: 1.2,
  });
  T(s,
    "Each prediction consumes 60 historical trading sessions with 21 engineered features per session. The saved checkpoint outputs a single sigmoid probability for upward direction.",
    6.6, 2.62, 5.82, 0.86, { size: 12, color: C.text2, lsm: 1.08 });
  card(s, 6.58, 3.66, 2.55, 0.62, { fill: C.card2, line: C.red, lineW: 0.9 });
  T(s, "0  =  DOWN", 6.58, 3.66, 2.55, 0.62, {
    size: 13.5, bold: true, color: C.red, align: "center", valign: "middle", face: F.head,
  });
  card(s, 9.42, 3.66, 2.55, 0.62, { fill: C.card2, line: C.green, lineW: 0.9 });
  T(s, "1  =  UP", 9.42, 3.66, 2.55, 0.62, {
    size: 13.5, bold: true, color: C.green, align: "center", valign: "middle", face: F.head,
  });

  // inference flow
  T(s, "INFERENCE FLOW", 0.64, 4.8, 4.0, 0.24, {
    size: 9.5, bold: true, color: C.accent2, spacing: 1.2,
  });
  const nodes = [
    ["Historical\nfeatures", "60 × 21", C.sky],
    ["LSTM\nlayers", "checkpoint", C.accent2],
    ["Dense\noutput", "1 unit", C.accent],
    ["Sigmoid", "probability", C.amber],
    ["UP / DOWN", "threshold 0.5", C.green],
  ];
  const nw = 2.2, ngap = 0.3;
  const nleft = (W - (5 * nw + 4 * ngap)) / 2;
  nodes.forEach((n, i) => {
    const x = nleft + i * (nw + ngap);
    card(s, x, 5.12, nw, 0.98, { fill: C.card2, line: n[2], lineW: 1 });
    T(s, n[0], x + 0.12, 5.22, nw - 0.24, 0.46, {
      size: 11.4, bold: true, color: C.text, align: "center", valign: "middle", lsm: 0.95,
    });
    T(s, n[1], x + 0.12, 5.72, nw - 0.24, 0.22, {
      size: 9.3, color: n[2], align: "center", valign: "middle",
    });
    if (i < 4) arrowRight(s, x + nw + 0.045, 5.48, 0.21, 0.2, C.text3);
  });

  card(s, 0.62, 6.32, 12.1, 0.42, { fill: C.card2, line: C.border, radius: 0.08 });
  T(s, "BINARY CLASSIFICATION ONLY  •  NO THREE-CLASS OR RETURN-VALUE OUTPUT  •  EXISTING-MODEL INFERENCE, NOT NEW TRAINING",
    0.82, 6.32, 11.7, 0.42, {
      size: 9.6, bold: true, color: C.accent2, align: "center", valign: "middle", spacing: 0.6,
    });

  s.addNotes(
    "Be precise here. The checkpoint is an LSTM that takes a 60-session sequence of 21 features and produces one sigmoid probability. " +
    "A threshold of 0.5 maps that probability to a binary label: 0 means DOWN, 1 means UP. " +
    "This is inference using an already-trained model and its original fitted scaler; no new training is performed in this phase. " +
    "Do not claim three classes, do not claim predicted returns, and do not invent accuracy numbers — no performance metric is being presented."
  );
})();

/* ============================================================
   SLIDE 8 — MONGODB & STREAMLIT DASHBOARD
   ============================================================ */
(function () {
  const s = baseSlide("07 / Persistence & UI", "MongoDB & Streamlit Dashboard", 8);

  // MongoDB column
  card(s, 0.62, 1.45, 3.02, 5.25, { fill: C.card, line: C.border });
  icon(s, "server", 0.88, 1.7, 0.44, C.green);
  T(s, "MONGODB", 1.44, 1.7, 1.9, 0.32, { size: 15, bold: true, color: C.text, face: F.head });
  T(s, "PREDICTION STORE", 1.44, 2.06, 1.9, 0.22, {
    size: 9.2, bold: true, color: C.green, spacing: 0.9,
  });
  bulletList(s, [
    "Prediction storage",
    "Stock symbols",
    "UP / DOWN probabilities",
    "Prediction timestamps",
    "Model identification",
    "Data provenance",
  ], 0.9, 2.55, 2.5, 0.58, { dot: C.green, size: 11.8 });
  card(s, 0.88, 6.06, 2.5, 0.42, { fill: C.card2, line: C.border, radius: 0.08 });
  T(s, "Persistent inference records", 0.98, 6.06, 2.3, 0.42, {
    size: 9.5, color: C.text2, align: "center", valign: "middle",
  });

  // Dashboard mockup frame
  card(s, 3.88, 1.45, 8.84, 5.25, { fill: "0E192B", line: C.border });
  T(s, "STREAMLIT ANALYTICS DASHBOARD", 4.12, 1.62, 4.4, 0.26, {
    size: 10.5, bold: true, color: C.accent2, spacing: 1.1,
  });
  chip(s, "ILLUSTRATIVE MOCKUP — NO LIVE VALUES", 9.55, 1.6, 2.9, 0.3, {
    size: 8.2, color: C.amber, fill: C.card2, line: C.border,
  });
  chip(s, "SELECT SYMBOL  ▾", 4.12, 1.98, 2.25, 0.32, {
    size: 9.5, color: C.text, fill: C.card, line: C.accent, bold: true,
  });
  chip(s, "Single-stock inference", 6.48, 1.98, 2.35, 0.32, { size: 9, color: C.text2 });
  chip(s, "Multi-stock inference", 8.94, 1.98, 2.35, 0.32, { size: 9, color: C.text2 });

  // metric cards
  const metrics = ["OPEN", "HIGH", "LOW", "CLOSE"];
  metrics.forEach((m, i) => {
    const x = 4.12 + i * 2.14;
    card(s, x, 2.46, 1.98, 0.78, { fill: C.card, line: C.border, radius: 0.08 });
    T(s, m, x + 0.14, 2.56, 1.7, 0.2, { size: 8.8, color: C.text3, bold: true, spacing: 0.8 });
    T(s, "—", x + 0.14, 2.78, 1.7, 0.34, { size: 17, bold: true, color: C.text, face: F.head });
  });

  // candlestick chart
  card(s, 4.12, 3.4, 6.05, 2.12, { fill: C.card2, line: C.border, radius: 0.08 });
  T(s, "CANDLESTICK — ILLUSTRATIVE", 4.32, 3.56, 3.2, 0.2, {
    size: 8.6, color: C.text3, bold: true, spacing: 0.7,
  });
  for (let i = 0; i < 5; i++) line(s, 4.34, 3.85 + i * 0.31, 5.6, 0, C.grid, 0.6);
  const cs = [0.42, 0.55, 0.48, 0.7, 0.62, 0.85, 0.74, 0.96, 0.88, 1.1, 0.98, 1.2, 1.12, 1.32];
  cs.forEach((v, i) => {
    const x = 4.48 + i * 0.38;
    const yTop = 5.35 - v * 0.62;
    const yBot = 5.35 - (v - 0.3) * 0.62;
    const up = i % 4 !== 2;
    const col = up ? C.green : C.red;
    line(s, x + 0.085, yTop - 0.1, 0, (yBot - yTop) + 0.2, col, 0.9);
    rect(s, x, Math.min(yTop, yBot), 0.17, Math.max(0.09, Math.abs(yBot - yTop)), col, 18);
  });

  // prediction panel
  card(s, 10.34, 3.4, 2.14, 2.12, { fill: C.card, line: C.border, radius: 0.08 });
  T(s, "PREDICTION", 10.54, 3.55, 1.8, 0.22, {
    size: 8.6, bold: true, color: C.text3, spacing: 0.7,
  });
  card(s, 10.54, 3.86, 1.74, 0.48, { fill: "13321F", line: C.green, radius: 0.07, lineW: 0.8 });
  T(s, "UP   —", 10.54, 3.86, 1.74, 0.48, {
    size: 12, bold: true, color: C.green, align: "center", valign: "middle", face: F.head,
  });
  card(s, 10.54, 4.42, 1.74, 0.48, { fill: "341A1D", line: C.red, radius: 0.07, lineW: 0.8 });
  T(s, "DOWN   —", 10.54, 4.42, 1.74, 0.48, {
    size: 12, bold: true, color: C.red, align: "center", valign: "middle", face: F.head,
  });
  T(s, "Model: existing LSTM\nSource: HDFS Parquet", 10.54, 5.02, 1.78, 0.42, {
    size: 8.6, color: C.text3, lsm: 1.05,
  });

  // volume chart
  card(s, 4.12, 5.68, 8.36, 0.82, { fill: C.card2, line: C.border, radius: 0.08 });
  T(s, "VOLUME", 4.32, 5.8, 1.0, 0.18, { size: 8.4, bold: true, color: C.text3, spacing: 0.6 });
  const vols = [0.26, 0.42, 0.34, 0.56, 0.48, 0.68, 0.52, 0.74, 0.6, 0.82, 0.66, 0.9, 0.72, 0.96, 0.8, 1.02];
  vols.forEach((v, i) => {
    const x = 4.5 + i * 0.47;
    rect(s, x, 6.38 - v * 0.48, 0.3, v * 0.48, i % 3 === 0 ? C.accent2 : C.slate, 25);
  });

  s.addNotes(
    "MongoDB stores each inference record: symbol, UP and DOWN probabilities, timestamp, model identification and data provenance. " +
    "On the right is an illustrative Streamlit dashboard layout, not a live screenshot. It shows a stock selector, single and multi-stock inference modes, " +
    "OHLCV metrics, a candlestick chart, a volume chart, technical indicators and prediction history. " +
    "All values are shown as dashes deliberately — no real or invented prices are presented. " +
    "In the viva, describe the capabilities of the dashboard rather than treating this mockup as evidence of live data."
  );
})();

/* ============================================================
   SLIDE 9 — STATUS, TESTING & LIMITATIONS
   ============================================================ */
(function () {
  const s = baseSlide("08 / Validation", "Implementation Status, Testing & Limitations", 9);

  // completed
  card(s, 0.62, 1.45, 4.0, 4.72, { fill: C.card, line: C.border });
  phaseChip(s, "COMPLETED — PHASE 4", 0.88, 1.7, 2.2, C.green);
  const done = [
    "Existing trained LSTM loaded successfully",
    "Original fitted scaler reused",
    "CPU inference validated",
    "Multi-stock inference implemented",
    "MongoDB persistence integration implemented",
    "Streamlit integration implemented",
    "46 Phase 4 tests passed",
  ];
  done.forEach((t, i) => {
    const y = 2.22 + i * 0.54;
    dot(s, 0.92, y + 0.06, 0.22, C.green, 82);
    checkMark(s, 0.975, y + 0.075, C.green);
    T(s, t, 1.28, y, 3.08, 0.44, { size: 11.3, color: C.text, valign: "top", lsm: 1.02 });
  });

  // ongoing
  card(s, 4.78, 1.45, 4.0, 4.72, { fill: C.card, line: C.border });
  phaseChip(s, "ONGOING WORK", 5.04, 1.7, 1.9, C.amber);
  const ongoing = [
    "Complete distributed HDFS–Spark–Parquet integration",
    "Full-universe processing validation",
    "End-to-end MongoDB verification",
    "Live market-data reliability",
    "Complete market analytics integration",
  ];
  ongoing.forEach((t, i) => {
    const y = 2.32 + i * 0.72;
    dot(s, 5.08, y + 0.08, 0.17, C.amber, 78);
    T(s, t, 5.4, y, 3.12, 0.6, { size: 11.4, color: C.text2, valign: "top", lsm: 1.03 });
  });

  // limitations
  card(s, 8.94, 1.45, 3.78, 4.72, { fill: C.card, line: C.border });
  phaseChip(s, "LIMITATIONS", 9.2, 1.7, 1.6, C.red);
  const limits = [
    "Binary prediction only",
    "Generalisation to non-NASDAQ stocks requires validation",
    "Market data depends on provider availability",
    "Prediction probabilities do not guarantee actual movement",
  ];
  limits.forEach((t, i) => {
    const y = 2.32 + i * 0.88;
    dot(s, 9.24, y + 0.08, 0.17, C.red, 78);
    T(s, t, 9.56, y, 2.86, 0.74, { size: 11.3, color: C.text2, valign: "top", lsm: 1.03 });
  });

  // accuracy banner
  card(s, 0.62, 6.32, 12.1, 0.44, { fill: C.card2, line: C.accent, lineW: 0.9, radius: 0.08 });
  T(s, "IMPORTANT: 46 tests refer to Phase 4 validation only — not the complete distributed project.",
    0.86, 6.32, 11.6, 0.44, {
      size: 10.6, bold: true, color: C.accent2, align: "center", valign: "middle",
    });

  s.addNotes(
    "Separate completed work from ongoing work clearly. The verified Phase 4 result is that the existing checkpoint and its original fitted scaler load, " +
    "CPU inference runs, multi-stock inference works, MongoDB and Streamlit integration exist, and 46 tests pass. " +
    "Those 46 tests validate Phase 4 only — they are not evidence that the entire distributed pipeline is complete. " +
    "Still in progress: full HDFS–Spark–Parquet integration, full-universe validation, end-to-end MongoDB checks, data-provider reliability and analytics integration. " +
    "Concede the limitations openly: binary output only, non-NASDAQ generalisation unvalidated, provider dependence, and probabilities are not guarantees."
  );
})();

/* ============================================================
   SLIDE 10 — CONCLUSION & FUTURE SCOPE
   ============================================================ */
(function () {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  T(s, "09 / CONCLUSION", 0.62, 0.34, 8.0, 0.26, {
    size: 10, bold: true, color: C.accent2, spacing: 1.6,
  });
  T(s, "Conclusion & Future Scope", 0.62, 0.63, 11.2, 0.63, {
    size: 27, bold: true, color: C.text, face: F.head, valign: "middle",
  });
  T(s, "10 / 10", 11.55, 0.36, 1.15, 0.28, {
    size: 10, color: C.text3, align: "right", spacing: 0.6,
  });

  card(s, 0.62, 1.45, 5.45, 3.92, { fill: C.card, line: C.border });
  icon(s, "shield", 0.9, 1.72, 0.44, C.accent2);
  T(s, "CONCLUSION", 1.46, 1.74, 4.0, 0.32, {
    size: 15, bold: true, color: C.text, face: F.head,
  });
  T(s,
    "The Stock Market Intelligence Engine combines distributed data infrastructure, feature engineering, existing deep-learning inference and interactive financial analytics.",
    0.92, 2.36, 4.85, 1.42, { size: 13.2, color: C.text2, lsm: 1.12 });
  T(s,
    "It demonstrates how big data technologies can support large-scale market information processing and predictive insight, while keeping trained-model inference separate from model development.",
    0.92, 3.92, 4.85, 1.22, { size: 13.2, color: C.text2, lsm: 1.12 });

  card(s, 6.3, 1.45, 6.42, 3.92, { fill: C.card, line: C.border });
  T(s, "FUTURE SCOPE", 6.58, 1.72, 4.0, 0.32, {
    size: 15, bold: true, color: C.text, face: F.head,
  });
  const future = [
    "Complete full-universe distributed processing",
    "Improve automated market data ingestion",
    "Complete end-to-end HDFS–Spark–MongoDB integration",
    "Expand multi-stock analytics",
    "Improve prediction monitoring and evaluation",
    "Introduce additional financial indicators",
    "Develop portfolio and risk analytics",
  ];
  future.forEach((f, i) => {
    const col = i < 4 ? 0 : 1;
    const row = i < 4 ? i : i - 4;
    const x = 6.58 + col * 3.05;
    const y = 2.3 + row * 0.72;
    badge(s, i + 1, x, y, 0.31, C.accent, C.bg);
    T(s, f, x + 0.42, y - 0.03, 2.55, 0.62, {
      size: 10.6, color: C.text2, valign: "top", lsm: 1.04,
    });
  });

  card(s, 0.62, 5.6, 12.1, 1.36, { fill: "0E1A2F", line: C.accent, lineW: 1 });
  T(s, "THANK YOU", 0.62, 5.78, 12.1, 0.58, {
    size: 36, bold: true, color: C.text, face: F.head, align: "center", valign: "middle", spacing: 2,
  });
  T(s, "Questions & Discussion", 0.62, 6.4, 12.1, 0.32, {
    size: 14, color: C.accent2, align: "center", valign: "middle", spacing: 1,
  });

  line(s, 0.62, 6.96, 12.1, 0, C.grid, 0.8);
  T(s, "STOCK MARKET INTELLIGENCE ENGINE", 0.62, 7.03, 8.0, 0.22, {
    size: 8, color: C.text3, spacing: 1.1,
  });

  s.addNotes(
    "Summarise the contribution: one system that spans distributed storage, parallel processing, feature engineering, " +
    "reuse of a trained LSTM for binary direction inference, persistence and interactive analytics. " +
    "The future scope is explicitly future work: finish full-universe distributed processing, automate ingestion, close the end-to-end integration, " +
    "expand analytics, improve monitoring, add indicators and explore portfolio and risk analytics. " +
    "Close with the limitations already stated — no accuracy claim is made. Invite questions."
  );
})();

/* ---------------- write file ---------------- */
const out = require("path").join(__dirname, "Stock_Market_Intelligence_Engine.pptx");
pres.writeFile({ fileName: out }).then(() => {
  console.log("WROTE " + out);
}).catch((e) => {
  console.error("FAILED: " + e.message);
  process.exit(1);
});
