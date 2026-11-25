const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Research Project";
pres.title = "Intelligent Indian Legal Document Analysis Using NLP";

// ----- Design tokens -----
const NAVY = "0E1F3C";
const NAVY2 = "16305A";
const NAVY3 = "22406E";
const GOLD = "C6A75E";
const GOLDD = "9C7C34";
const LIGHT = "F4F6FA";
const WHITE = "FFFFFF";
const GRAY = "46506A";
const GRAYL = "6E7A93";
const ICE = "DCE6F7";
const CARD = "FFFFFF";

const HFONT = "Cambria"; // serif headers
const BFONT = "Calibri"; // body

const W = 13.33;
const H = 7.5;

// ----- Helpers -----
function footer(slide, page) {
  slide.addText("Intelligent Indian Legal Document Analysis Using NLP", {
    x: 0.5, y: 6.97, w: 9.5, h: 0.28,
    fontFace: BFONT, fontSize: 9, color: GRAYL, align: "left", valign: "middle", margin: 0,
  });
  slide.addText(String(page), {
    x: 12.3, y: 6.97, w: 0.55, h: 0.28,
    fontFace: BFONT, fontSize: 9, color: GRAYL, align: "right", valign: "middle", margin: 0,
  });
}

function sectionHeader(slide, num, title) {
  slide.addShape(pres.ShapeType.ellipse, {
    x: 0.5, y: 0.42, w: 0.62, h: 0.62,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  slide.addText(num, {
    x: 0.5, y: 0.42, w: 0.62, h: 0.62,
    fontFace: HFONT, fontSize: 22, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });
  slide.addText(title, {
    x: 1.35, y: 0.4, w: 11.4, h: 0.7,
    fontFace: HFONT, fontSize: 29, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0,
  });
}

function baseSlide(num, title) {
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionHeader(s, num, title);
  return s;
}

function numBadge(slide, n, x, y, d, fill, txtColor) {
  slide.addShape(pres.ShapeType.ellipse, {
    x: x, y: y, w: d, h: d,
    fill: { color: fill }, line: { color: fill, width: 0 },
  });
  slide.addText(String(n), {
    x: x, y: y, w: d, h: d,
    fontFace: HFONT, fontSize: 15, bold: true, color: txtColor, align: "center", valign: "middle", margin: 0,
  });
}

/* =========================================================
   SLIDE 1 - TITLE
   ========================================================= */
(function titleSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // subtle decorative circles (motif: gold circles)
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.6, y: -1.5, w: 4.6, h: 4.6,
    fill: { color: NAVY2, transparency: 40 }, line: { color: NAVY2, width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.6, y: 5.2, w: 4.4, h: 4.4,
    fill: { color: NAVY2, transparency: 55 }, line: { color: NAVY2, width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 0.9, y: 1.15, w: 0.5, h: 0.5,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });

  s.addText("Intelligent Indian Legal Document\nAnalysis Using NLP", {
    x: 0.9, y: 1.9, w: 11.4, h: 2.1,
    fontFace: HFONT, fontSize: 40, bold: true, color: WHITE, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 1.0,
  });

  s.addText("An NLP-based framework for summarizing, extracting, and retrieving information from Indian court judgments.",
    { x: 0.9, y: 4.05, w: 11.0, h: 0.9,
      fontFace: BFONT, fontSize: 17, color: ICE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addText("Research Project Presentation", { x: 0.9, y: 5.25, w: 6.0, h: 0.35,
    fontFace: BFONT, fontSize: 13, bold: true, color: GOLD, align: "left", margin: 0 });

  s.addText("Presented by:  [Student Name]        Guide:  [Mentor Name]        Department of Computer Science",
    { x: 0.9, y: 5.75, w: 11.6, h: 0.35,
      fontFace: BFONT, fontSize: 13, color: ICE, align: "left", margin: 0 });

  s.addText("2026", { x: 0.9, y: 6.15, w: 4.0, h: 0.3,
    fontFace: BFONT, fontSize: 12, color: GRAYL, align: "left", margin: 0 });

  s.addNotes("Title slide. Replace [Student Name], [Mentor Name], department and date with your details. The deck explains the research topic, why it matters, the problem, objectives, methodology, and how 2021-2025 papers inform the proposed contribution. All intended outcomes are labelled as proposed/expected, not as achieved results.");
})();

/* =========================================================
   SLIDE 2 - INTRODUCTION
   ========================================================= */
(function introSlide() {
  const s = baseSlide("1", "Introduction to the Selected Topic");

  s.addText("An AI-based research project to help users understand, organize, and retrieve information from lengthy Indian legal documents, including court judgments and case records.",
    { x: 0.5, y: 1.35, w: 12.3, h: 0.8,
      fontFace: BFONT, fontSize: 15, color: GRAY, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  // Left card: context
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.35, w: 5.4, h: 4.2, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
    shadow: { type: "outer", color: "9AA6B8", blur: 8, offset: 2, angle: 90, opacity: 0.25 },
  });
  s.addText("Why legal text is hard", {
    x: 0.8, y: 2.55, w: 4.8, h: 0.4,
    fontFace: HFONT, fontSize: 18, bold: true, color: NAVY2, align: "left", margin: 0 });
  s.addText([
    { text: "Complex terminology and lengthy sentences", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "References to statutes and previous judgments", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "Interleaved facts, arguments, and decisions", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "Manual analysis needs time and legal expertise", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "NLP can identify meaning and structure in legal text", options: { bullet: { code: "2022" } } },
  ], {
    x: 0.8, y: 3.05, w: 4.85, h: 3.3,
    fontFace: BFONT, fontSize: 13.5, color: GRAY, align: "left", valign: "top",
    margin: 0, paraSpaceAfter: 8, lineSpacingMultiple: 1.0,
  });

  // Right card: capabilities
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.25, y: 2.35, w: 6.55, h: 4.2, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Proposed system capabilities", {
    x: 6.55, y: 2.55, w: 6.0, h: 0.4,
    fontFace: HFONT, fontSize: 18, bold: true, color: GOLD, align: "left", margin: 0 });

  const caps = [
    ["Legal document summarization", "Concise summaries that preserve important legal detail"],
    ["Information extraction", "Parties, courts, dates, case numbers, statutes, entities"],
    ["Document structure analysis", "Separate facts, arguments, reasoning, and decisions"],
    ["Legal statute identification", "Suggest potentially relevant legal provisions"],
    ["Semantic search", "Retrieve similar cases and relevant legal passages"],
    ["Explainability", "Show source passages behind extracted information"],
  ];
  let cy = 3.05;
  caps.forEach((c, i) => {
    numBadge(s, i + 1, 6.55, cy + 0.02, 0.32, GOLD, NAVY);
    s.addText([
      { text: c[0] + " — ", options: { bold: true, color: WHITE } },
      { text: c[1], options: { color: ICE } },
    ], { x: 6.98, y: cy - 0.04, w: 5.65, h: 0.5,
      fontFace: BFONT, fontSize: 12, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.95 });
    cy += 0.57;
  });

  s.addNotes("Introduction. Indian legal documents are long and unstructured; NLP, and Indian legal-domain models such as InLegalBERT, provide a foundation for summarization, statute identification, segmentation, and classification. The six capabilities on the right are proposed, not yet fully implemented.");
  footer(s, 2);
})();

/* =========================================================
   SLIDE 3 - MOTIVATION
   ========================================================= */
(function motivationSlide() {
  const s = baseSlide("2", "Motivation & Topic Selection");

  const items = [
    ["Increasing complexity of legal documents", "Judgments are long, detailed, and difficult to understand without legal knowledge."],
    ["Time-consuming manual analysis", "Finding relevant facts, provisions, and precedents requires reading and cross-referencing."],
    ["Need for better access to legal information", "Structured summaries and search tools help students, researchers, and professionals work faster."],
    ["Limitations of general-purpose NLP", "Indian legal terminology and structure benefit from domain-specific models."],
    ["Research opportunity", "Recent Indian legal NLP work on summarization, multilingual and long-document processing can be integrated reliably."],
  ];

  let y = 1.5;
  items.forEach((it, i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.5, y: y, w: 12.3, h: 0.82, rectRadius: 0.06,
      fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
    });
    numBadge(s, i + 1, 0.72, y + 0.16, 0.5, i % 2 === 0 ? NAVY2 : NAVY3, WHITE);
    s.addText([
      { text: it[0] + "  ", options: { bold: true, color: NAVY, fontSize: 15 } },
      { text: "— " + it[1], options: { color: GRAY, fontSize: 13 } },
    ], { x: 1.42, y: y, w: 11.2, h: 0.82,
      fontFace: BFONT, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.95 });
    y += 0.95;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 6.25, w: 12.3, h: 0.62, rectRadius: 0.06,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText([
    { text: "Goal:  ", options: { bold: true, color: GOLD } },
    { text: "Bridge the gap between complex, unstructured Indian legal documents and accessible, structured legal information through domain-aware NLP.", options: { color: WHITE } },
  ], { x: 0.8, y: 6.25, w: 11.8, h: 0.62,
    fontFace: BFONT, fontSize: 13, align: "left", valign: "middle", margin: 0 });

  s.addNotes("Motivation. Five drivers: document complexity, time-consuming manual review, need for better access, limits of general-purpose NLP, and an active research opportunity. The project goal is to bridge unstructured legal documents and structured, accessible legal information.");
  footer(s, 3);
})();

/* =========================================================
   SLIDE 4 - PROBLEM STATEMENT
   ========================================================= */
(function problemSlide() {
  const s = baseSlide("3", "Problem Statement");

  s.addText("Indian legal documents contain large amounts of unstructured information spread across lengthy passages, complex sentences, and references to laws and prior cases.",
    { x: 0.5, y: 1.4, w: 12.3, h: 0.75,
      fontFace: BFONT, fontSize: 15, color: GRAY, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  // Three problem panels
  const panels = [
    ["Manual analysis is slow", "Reading and cross-referencing facts, provisions, and precedents consumes significant expert time."],
    ["Documents are hard to parse", "Information sits in long passages and complex sentences with cross-references to statutes and prior cases."],
    ["Automated systems struggle", "Long documents, legal context, accurate summarization, and explainable outputs remain difficult."],
  ];
  const px = 0.5;
  const pw = 4.0;
  const gap = 0.15;
  panels.forEach((p, i) => {
    const x = px + i * (pw + gap);
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 2.35, w: pw, h: 2.25, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
      shadow: { type: "outer", color: "9AA6B8", blur: 8, offset: 2, angle: 90, opacity: 0.22 },
    });
    numBadge(s, i + 1, x + 0.3, 2.62, 0.5, GOLD, NAVY);
    s.addText(p[0], { x: x + 0.3, y: 3.2, w: pw - 0.6, h: 0.5,
      fontFace: HFONT, fontSize: 15, bold: true, color: NAVY2, align: "left", valign: "top", margin: 0 });
    s.addText(p[1], { x: x + 0.3, y: 3.68, w: pw - 0.55, h: 0.85,
      fontFace: BFONT, fontSize: 12.5, color: GRAY, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.0 });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.95, w: 12.3, h: 1.9, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Research aim", { x: 0.85, y: 5.15, w: 11.6, h: 0.35,
    fontFace: HFONT, fontSize: 15, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText("Develop and evaluate an NLP-based framework that extracts important legal information, generates concise summaries, identifies potentially relevant statutes, and supports retrieval of related legal passages — while maintaining traceability to the original document.",
    { x: 0.85, y: 5.55, w: 11.6, h: 1.15,
      fontFace: BFONT, fontSize: 14, color: WHITE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addNotes("Problem statement. Manual analysis is time-consuming; automated systems struggle with long documents, legal context, accurate summarization, and explainable outputs. The research aim is to build and evaluate a framework for extraction, summarization, statute identification, and retrieval that stays traceable to the source.");
  footer(s, 4);
})();

/* =========================================================
   SLIDE 5 - OBJECTIVES
   ========================================================= */
(function objectivesSlide() {
  const s = baseSlide("4", "Research Aim & Objectives");

  const objs = [
    ["Document preprocessing", "Extract text from legal PDFs, clean formatting artifacts, preserve structure, and handle long documents via suitable chunking."],
    ["Legal information extraction", "Identify important entities, case details, legal provisions, and structural roles in legal documents."],
    ["Summarization & retrieval", "Generate concise summaries and retrieve relevant legal passages or similar cases using semantic similarity."],
    ["Reliability & explainability", "Evaluate factual consistency, extraction accuracy, retrieval relevance, and the ability to trace results to source passages."],
    ["Experimental evaluation", "Compare suitable baseline and domain-specific NLP methods using measurable evaluation metrics."],
  ];

  const cw = 3.95;
  const ch = 2.05;
  const gapx = 0.15;
  const gapy = 0.2;
  const x0 = 0.5;
  const y0 = 1.45;

  objs.forEach((o, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = x0 + col * (cw + gapx);
    const y = y0 + row * (ch + gapy);
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: y, w: cw, h: ch, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
      shadow: { type: "outer", color: "9AA6B8", blur: 8, offset: 2, angle: 90, opacity: 0.22 },
    });
    numBadge(s, i + 1, x + 0.28, y + 0.25, 0.52, NAVY2, WHITE);
    s.addText(o[0], { x: x + 0.95, y: y + 0.22, w: cw - 1.2, h: 0.6,
      fontFace: HFONT, fontSize: 15, bold: true, color: NAVY2, align: "left", valign: "middle", margin: 0 });
    s.addText(o[1], { x: x + 0.3, y: y + 0.95, w: cw - 0.6, h: 0.95,
      fontFace: BFONT, fontSize: 12, color: GRAY, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.0 });
  });

  // 6th cell - focus
  const x6 = x0 + 2 * (cw + gapx);
  const y6 = y0 + 1 * (ch + gapy);
  s.addShape(pres.ShapeType.roundRect, {
    x: x6, y: y6, w: cw, h: ch, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Primary focus", { x: x6 + 0.3, y: y6 + 0.22, w: cw - 0.6, h: 0.4,
    fontFace: HFONT, fontSize: 15, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText("Structure-aware summarization with source-grounded legal information extraction; other modules provide supporting functionality.",
    { x: x6 + 0.3, y: y6 + 0.72, w: cw - 0.6, h: 1.2,
      fontFace: BFONT, fontSize: 12.5, color: WHITE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addNotes("Objectives (from the source document): O1 preprocessing, O2 information extraction, O3 summarization and retrieval, O4 reliability and explainability, O5 experimental evaluation. The recommended primary contribution is structure-aware summarization with source-grounded legal information extraction.");
  footer(s, 5);
})();

/* =========================================================
   SLIDE 6 - METHODOLOGY
   ========================================================= */
(function methodologySlide() {
  const s = baseSlide("5", "Proposed Methodology");

  const steps = [
    ["Legal document collection", "Indian court judgments and legally available datasets."],
    ["PDF extraction & preprocessing", "Text extraction, cleaning, section detection, and chunking."],
    ["Legal NLP processing", "InLegalBERT, entity extraction, rhetorical-role classification, plus task modules."],
    ["Evaluation & validation", "Compare outputs with reference data; assess accuracy and reliability."],
    ["Interactive analysis interface", "Summaries, extracted details, search results, and evidence passages."],
  ];

  let y = 1.35;
  steps.forEach((st, i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.5, y: y, w: 7.35, h: 0.98, rectRadius: 0.06,
      fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
    });
    numBadge(s, i + 1, 0.72, y + 0.2, 0.58, i === 2 ? GOLD : NAVY2, i === 2 ? NAVY : WHITE);
    s.addText(st[0], { x: 1.5, y: y + 0.1, w: 6.1, h: 0.36,
      fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0 });
    s.addText(st[1], { x: 1.5, y: y + 0.45, w: 6.15, h: 0.45,
      fontFace: BFONT, fontSize: 11.5, color: GRAY, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 0.95 });
    y += 1.08;
  });

  // Right column - core modules
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.05, y: 1.35, w: 4.78, h: 2.55, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Task modules (from stage 3)", { x: 8.3, y: 1.5, w: 4.3, h: 0.35,
    fontFace: HFONT, fontSize: 14.5, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText([
    { text: "Summarization", options: { bold: true, color: WHITE, breakLine: true } },
    { text: "Key facts and legal reasoning", options: { color: ICE, breakLine: true } },
    { text: "Legal retrieval", options: { bold: true, color: WHITE, breakLine: true } },
    { text: "Relevant passages and cases", options: { color: ICE, breakLine: true } },
    { text: "Statute identification", options: { bold: true, color: WHITE, breakLine: true } },
    { text: "Potentially applicable provisions", options: { color: ICE, breakLine: true } },
    { text: "Source attribution", options: { bold: true, color: WHITE, breakLine: true } },
    { text: "Evidence for generated results", options: { color: ICE } },
  ], { x: 8.3, y: 1.95, w: 4.35, h: 1.85,
    fontFace: BFONT, fontSize: 11.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 2, lineSpacingMultiple: 0.95 });

  // Right column - datasets & models
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.05, y: 4.08, w: 4.78, h: 2.55, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
    shadow: { type: "outer", color: "9AA6B8", blur: 8, offset: 2, angle: 90, opacity: 0.2 },
  });
  s.addText("Datasets & models", { x: 8.3, y: 4.22, w: 4.3, h: 0.35,
    fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY2, align: "left", margin: 0 });
  s.addText([
    { text: "ILDC ", options: { bold: true, color: NAVY2 } },
    { text: "— ~35,000 Indian Supreme Court cases.", options: { color: GRAY, breakLine: true } },
    { text: "InLegalBERT ", options: { bold: true, color: NAVY2 } },
    { text: "— domain-adapted model for Indian legal text.", options: { color: GRAY, breakLine: true } },
    { text: "L-SUMM / InLSum ", options: { bold: true, color: NAVY2 } },
    { text: "— 1,800 judgments with expert English summaries.", options: { color: GRAY } },
  ], { x: 8.3, y: 4.65, w: 4.35, h: 1.85,
    fontFace: BFONT, fontSize: 11.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 6, lineSpacingMultiple: 0.98 });

  s.addNotes("Methodology: (1) collect judgments/datasets, (2) PDF extraction and preprocessing with section detection and chunking, (3) legal NLP processing with InLegalBERT plus summarization, retrieval, statute identification and source attribution, (4) evaluation against reference data, (5) an interactive analysis interface. This is a proposed architecture; final modules depend on dataset, resources, and timeline. Start with summarization + information extraction + semantic retrieval; statute identification if labels exist; outcome prediction is optional.");
  footer(s, 6);
})();

/* =========================================================
   SLIDE 7 - LITERATURE REVIEW
   ========================================================= */
(function litReviewSlide() {
  const s = baseSlide("6", "Literature Review (2021–2025)");

  const head = ["Year", "Paper & Authors", "Contribution & Limitation", "Relevance to This Project"];
  const rows = [
    ["2021", "ILDC for CJPE\nMalik et al., ACL-IJCNLP 2021",
      "~35,000 Indian Supreme Court cases; judgment prediction with explanations. Machine explanations differ from expert ones.",
      "Explainability gap: how facts, summaries and source-grounded explanations aid transparency."],
    ["2021", "LeSICiN\narXiv:2112.14731",
      "Heterogeneous graph approach to identify relevant legal statutes from Indian documents.",
      "Combine statute identification with summaries and evidence passages."],
    ["2023", "Pre-trained LMs for Indian Law\nPaul, Mandal, Goyal & Ghosh, ICAIL 2023",
      "Legal-domain pretraining for Indian law; statutes, segmentation, prediction. Domain models beat general-purpose ones.",
      "Evaluate a domain-specific model for extraction and classification plus retrieval."],
    ["2024", "Automated Attribute Extraction via LLMs\nArtificial Intelligence and Law, 2024",
      "Extracts structured attributes from long, intricate legal text using large language models.",
      "Validate extracted fields against reference annotations; measure misses and errors."],
    ["2025", "JUST-NLP 2025 Shared Task (L-SUMM)\nACL Anthology",
      "1,800 Indian judgments with expert English summaries; reports challenges with extremely long documents.",
      "Use structure-aware chunking and test factual consistency, not fluency alone."],
    ["2025", "Contextors at L-SUMM\nNeelamegam & Nirmala, JUST-NLP 2025",
      "Retriever-driven multi-generator; iterative retrieval and cross-model validation for coverage and consistency.",
      "Test a simpler integrated system with directly traceable evidence and lower cost."],
  ];

  const colW = [0.85, 3.25, 4.35, 3.85];
  const startX = 0.5;
  const startY = 1.35;
  const headH = 0.42;
  const rowH = 0.86;

  // Header
  let hx = startX;
  head.forEach((h, i) => {
    s.addShape(pres.ShapeType.rect, {
      x: hx, y: startY, w: colW[i], h: headH,
      fill: { color: NAVY }, line: { color: WHITE, width: 1 },
    });
    s.addText(h, { x: hx + 0.08, y: startY, w: colW[i] - 0.16, h: headH,
      fontFace: HFONT, fontSize: 11.5, bold: true, color: WHITE, align: "left", valign: "middle", margin: 0 });
    hx += colW[i];
  });

  // Rows
  rows.forEach((r, ri) => {
    const y = startY + headH + ri * rowH;
    const bg = ri % 2 === 0 ? WHITE : "EAEEF6";
    let cx = startX;
    r.forEach((cell, ci) => {
      s.addShape(pres.ShapeType.rect, {
        x: cx, y: y, w: colW[ci], h: rowH,
        fill: { color: bg }, line: { color: "D3DAE6", width: 1 },
      });
      const isYear = ci === 0;
      const isPaper = ci === 1;
      s.addText(cell, { x: cx + 0.08, y: y + 0.03, w: colW[ci] - 0.16, h: rowH - 0.06,
        fontFace: BFONT, fontSize: isYear ? 12 : 9.5,
        bold: isYear || isPaper, color: isPaper ? NAVY2 : GRAY,
        align: isYear ? "center" : "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.92 });
      cx += colW[ci];
    });
  });

  s.addText("Starting set for the review, not an exhaustive systematic survey. Verify authors, venues, and reported limitations in the full papers.",
    { x: 0.5, y: 6.85, w: 12.3, h: 0.3,
      fontFace: BFONT, fontSize: 9.5, italic: true, color: GRAYL, align: "left", margin: 0 });

  s.addNotes("Literature review, six papers spanning 2021-2025. Each row states the contribution and limitation and its relevance to this project. This is a starting set, not an exhaustive systematic review; verify bibliographic details and reported limitations in the full papers before submission.");
  footer(s, 7);
})();

/* =========================================================
   SLIDE 8 - RESEARCH GAPS
   ========================================================= */
(function gapsSlide() {
  const s = baseSlide("7", "Research Gaps → Proposed Contribution");

  const head = ["Identified gap", "How this project will address it"];
  const rows = [
    ["Long-document handling", "Section-aware chunking that preserves facts, arguments, and decisions instead of splitting text solely by token count."],
    ["Factual reliability of summaries", "Ground generated summaries in retrieved source passages and check whether each key claim is supported by the document."],
    ["Fragmented analysis tasks", "Integrate summarization, entity extraction, and legal retrieval into one workflow with a consistent output format."],
    ["Limited source traceability", "Display page numbers, section references, or original snippets alongside extracted information and generated answers."],
    ["Insufficient evaluation across tasks", "Evaluate extraction, retrieval, and summarization separately, then assess the usefulness of the integrated system."],
  ];

  const colW = [4.35, 7.95];
  const startX = 0.5;
  const startY = 1.35;
  const headH = 0.45;
  const rowH = 0.92;

  let hx = startX;
  head.forEach((h, i) => {
    s.addShape(pres.ShapeType.rect, {
      x: hx, y: startY, w: colW[i], h: headH,
      fill: { color: NAVY }, line: { color: WHITE, width: 1 },
    });
    s.addText(h, { x: hx + 0.12, y: startY, w: colW[i] - 0.24, h: headH,
      fontFace: HFONT, fontSize: 13, bold: true, color: WHITE, align: "left", valign: "middle", margin: 0 });
    hx += colW[i];
  });

  rows.forEach((r, ri) => {
    const y = startY + headH + ri * rowH;
    const bg = ri % 2 === 0 ? WHITE : "EAEEF6";
    // gap col
    s.addShape(pres.ShapeType.rect, {
      x: startX, y: y, w: colW[0], h: rowH,
      fill: { color: bg }, line: { color: "D3DAE6", width: 1 },
    });
    s.addText(r[0], { x: startX + 0.3, y: y, w: colW[0] - 0.5, h: rowH,
      fontFace: BFONT, fontSize: 13, bold: true, color: NAVY2, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.95 });
    // response col
    s.addShape(pres.ShapeType.rect, {
      x: startX + colW[0], y: y, w: colW[1], h: rowH,
      fill: { color: bg }, line: { color: "D3DAE6", width: 1 },
    });
    s.addText(r[1], { x: startX + colW[0] + 0.12, y: y, w: colW[1] - 0.24, h: rowH,
      fontFace: BFONT, fontSize: 11.5, color: GRAY, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.95 });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 6.15, w: 12.3, h: 0.72, rectRadius: 0.06,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("These are candidate gaps synthesized from the reviewed papers — not a claim that no existing system addresses them. Confirm novelty through a broader review.",
    { x: 0.8, y: 6.15, w: 11.8, h: 0.72,
      fontFace: BFONT, fontSize: 11.5, italic: true, color: NAVY, align: "left", valign: "middle", margin: 0, lineSpacingMultiple: 0.95 });

  s.addNotes("Five candidate research gaps and how the project addresses each: long-document handling (section-aware chunking), factual reliability (source-grounding), fragmented tasks (integration), traceability (page/section refs), and cross-task evaluation. These are synthesized from the literature, not proof of novelty.");
  footer(s, 8);
})();

/* =========================================================
   SLIDE 9 - EXPECTED OUTCOMES & EVALUATION
   ========================================================= */
(function outcomesSlide() {
  const s = baseSlide("8", "Expected Outcomes & Evaluation");

  // Left: outcomes
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 1.35, w: 6.35, h: 5.05, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
    shadow: { type: "outer", color: "9AA6B8", blur: 8, offset: 2, angle: 90, opacity: 0.2 },
  });
  s.addText("Expected outcomes", { x: 0.8, y: 1.52, w: 5.8, h: 0.4,
    fontFace: HFONT, fontSize: 18, bold: true, color: NAVY2, align: "left", margin: 0 });
  s.addText([
    { text: "Structured legal summaries", options: { bold: true, color: NAVY, breakLine: false } },
    { text: " — shorter, organized views of lengthy judgments.", options: { color: GRAY, breakLine: true } },
    { text: "Automated information extraction", options: { bold: true, color: NAVY, breakLine: false } },
    { text: " — parties, case details, provisions, sections.", options: { color: GRAY, breakLine: true } },
    { text: "Context-aware search", options: { bold: true, color: NAVY, breakLine: false } },
    { text: " — relevant passages and similar cases.", options: { color: GRAY, breakLine: true } },
    { text: "Evidence-linked results", options: { bold: true, color: NAVY, breakLine: false } },
    { text: " — citations to the source passages used.", options: { color: GRAY, breakLine: true } },
    { text: "Evaluation report", options: { bold: true, color: NAVY, breakLine: false } },
    { text: " — quantitative and qualitative model comparison.", options: { color: GRAY } },
  ], { x: 0.8, y: 2.05, w: 5.8, h: 4.2,
    fontFace: BFONT, fontSize: 13.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 10, lineSpacingMultiple: 0.98 });

  // Right top: evaluation metrics
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.2, y: 1.35, w: 5.63, h: 2.55, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Evaluation plan", { x: 7.45, y: 1.5, w: 5.1, h: 0.4,
    fontFace: HFONT, fontSize: 16, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText([
    { text: "Extraction: ", options: { bold: true, color: WHITE } },
    { text: "precision, recall, F1", options: { color: ICE, breakLine: true } },
    { text: "Retrieval: ", options: { bold: true, color: WHITE } },
    { text: "Recall@k, MRR", options: { color: ICE, breakLine: true } },
    { text: "Summarization: ", options: { bold: true, color: WHITE } },
    { text: "ROUGE, BERTScore", options: { color: ICE, breakLine: true } },
    { text: "Human review: ", options: { bold: true, color: WHITE } },
    { text: "factuality check of key claims", options: { color: ICE } },
  ], { x: 7.45, y: 1.95, w: 5.1, h: 1.85,
    fontFace: BFONT, fontSize: 12.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 6, lineSpacingMultiple: 0.98 });

  // Right bottom: applications
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.2, y: 4.05, w: 5.63, h: 2.35, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: "E1E7F0", width: 1 },
  });
  s.addText("Potential applications", { x: 7.45, y: 4.18, w: 5.1, h: 0.4,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY2, align: "left", margin: 0 });
  s.addText([
    { text: "Legal research assistance", options: { bullet: { code: "2022" }, color: GRAY, breakLine: true } },
    { text: "Law-student learning and document review", options: { bullet: { code: "2022" }, color: GRAY, breakLine: true } },
    { text: "Case preparation support", options: { bullet: { code: "2022" }, color: GRAY, breakLine: true } },
    { text: "Legal information portals", options: { bullet: { code: "2022" }, color: GRAY } },
  ], { x: 7.45, y: 4.62, w: 5.15, h: 1.6,
    fontFace: BFONT, fontSize: 12.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 6, lineSpacingMultiple: 0.98 });
  s.addText("A research and document-understanding assistant — not a replacement for a lawyer or judge.",
    { x: 7.45, y: 6.02, w: 5.15, h: 0.32,
      fontFace: BFONT, fontSize: 10.5, italic: true, color: GOLDD, align: "left", margin: 0 });

  s.addNotes("Expected outcomes are proposed: structured summaries, information extraction, context-aware search, evidence-linked results, and an evaluation report. Evaluation uses F1 for extraction, Recall@k/MRR for retrieval, ROUGE/BERTScore for summarization, plus human factuality review. Applications include research, education, review, case prep, and legal portals. This is an assistant, not a replacement for legal professionals.");
  footer(s, 9);
})();

/* =========================================================
   SLIDE 10 - CONCLUSION & REFERENCES
   ========================================================= */
(function conclusionSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  sectionHeaderDark(s, "9", "Conclusion & References");

  // Left: conclusion
  s.addText("The problem", { x: 0.5, y: 1.4, w: 5.6, h: 0.35,
    fontFace: HFONT, fontSize: 16, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText("Indian legal documents are long and unstructured; manual analysis is slow and automated systems struggle with long context, accuracy, and explainability.",
    { x: 0.5, y: 1.78, w: 5.6, h: 1.0,
      fontFace: BFONT, fontSize: 13, color: ICE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addText("Proposed contribution", { x: 0.5, y: 2.85, w: 5.6, h: 0.35,
    fontFace: HFONT, fontSize: 16, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText("An integrated, source-grounded framework for structure-aware summarization, legal information extraction, statute identification, and semantic retrieval with traceable evidence.",
    { x: 0.5, y: 3.23, w: 5.6, h: 1.1,
      fontFace: BFONT, fontSize: 13, color: ICE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addText("Expected impact", { x: 0.5, y: 4.4, w: 5.6, h: 0.35,
    fontFace: HFONT, fontSize: 16, bold: true, color: GOLD, align: "left", margin: 0 });
  s.addText("More accessible and reliable legal information for research, education, and case preparation — with professional verification.",
    { x: 0.5, y: 4.78, w: 5.6, h: 1.0,
      fontFace: BFONT, fontSize: 13, color: ICE, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });

  s.addText("Ongoing work", { x: 0.5, y: 5.75, w: 5.6, h: 0.35,
    fontFace: HFONT, fontSize: 14, bold: true, color: WHITE, align: "left", margin: 0 });
  s.addText("Select a baseline, target one main gap, build the improvement, then compare on held-out documents and report limitations.",
    { x: 0.5, y: 6.1, w: 5.6, h: 0.8,
      fontFace: BFONT, fontSize: 12, color: GRAYL, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.02 });

  // Right: references
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.45, y: 1.4, w: 6.38, h: 5.5, rectRadius: 0.08,
    fill: { color: WHITE }, line: { color: WHITE, width: 0 },
  });
  s.addText("Key references (2021–2025)", { x: 6.7, y: 1.55, w: 5.9, h: 0.4,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY2, align: "left", margin: 0 });
  s.addText([
    { text: "1. Malik, V. et al. ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation. ACL-IJCNLP, 2021.", options: { breakLine: true } },
    { text: "2. LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification. arXiv:2112.14731, 2021.", options: { breakLine: true } },
    { text: "3. Paul, S., Mandal, A., Goyal, P., Ghosh, S. Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law. ICAIL, 2023.", options: { breakLine: true } },
    { text: "4. A Case Study for Automated Attribute Extraction from Legal Documents Using Large Language Models. Artificial Intelligence and Law, 2024 (Springer).", options: { breakLine: true } },
    { text: "5. Findings of the JUST-NLP 2025 Shared Task on Summarization of Indian Court Judgments. ACL Anthology, 2025.", options: { breakLine: true } },
    { text: "6. Neelamegam, P., Nirmala, S. J. Contextors at L-SUMM: Retriever-Driven Multi-Generator Summarization. JUST-NLP, 2025.", options: {} },
  ], { x: 6.7, y: 2.0, w: 5.9, h: 4.7,
    fontFace: BFONT, fontSize: 10, color: GRAY, align: "left", valign: "top", margin: 0, paraSpaceAfter: 7, lineSpacingMultiple: 0.95 });

  s.addNotes("Conclusion: the project targets a real problem in Indian legal document analysis and proposes an integrated, source-grounded NLP framework with structure-aware summarization as the main contribution. References list the six papers reviewed; add exact DOIs or official ACL/Springer links from the source document. Remarks remain proposed/expected, not completed results.");
})();

function sectionHeaderDark(slide, num, title) {
  slide.addShape(pres.ShapeType.ellipse, {
    x: 0.5, y: 0.42, w: 0.62, h: 0.62,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  slide.addText(num, {
    x: 0.5, y: 0.42, w: 0.62, h: 0.62,
    fontFace: HFONT, fontSize: 22, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });
  slide.addText(title, {
    x: 1.35, y: 0.4, w: 11.4, h: 0.7,
    fontFace: HFONT, fontSize: 29, bold: true, color: WHITE, align: "left", valign: "middle", margin: 0,
  });
}

pres.writeFile({ fileName: "Intelligent_Indian_Legal_Document_Analysis_Using_NLP.pptx" })
  .then((fn) => console.log("WROTE " + fn))
  .catch((e) => { console.error(e); process.exit(1); });
