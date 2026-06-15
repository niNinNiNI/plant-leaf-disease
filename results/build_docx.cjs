const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  TableOfContents
} = require("docx");

const md = fs.readFileSync("/home/nini/文档/深度学习大作业/课堂作业/results/课堂作业报告正文.md", "utf-8");
const lines = md.split("\n");

// ====== CONSTANTS ======
const BW = 9360; // US Letter 1" margins: 12240 - 2*1440 = 9360 DXA
const bd = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const bds = { top: bd, bottom: bd, left: bd, right: bd };
const cm = { top: 60, bottom: 60, left: 110, right: 110 };

// ====== LATEX TO UNICODE CONVERTER ======
function latexToUnicode(text) {
  return text
    // Greek letters
    .replace(/\\alpha/g, "α").replace(/\\beta/g, "β").replace(/\\gamma/g, "γ")
    .replace(/\\delta/g, "δ").replace(/\\epsilon/g, "ϵ").replace(/\\varepsilon/g, "ε")
    .replace(/\\zeta/g, "ζ").replace(/\\eta/g, "η").replace(/\\theta/g, "θ")
    .replace(/\\iota/g, "ι").replace(/\\kappa/g, "κ").replace(/\\lambda/g, "λ")
    .replace(/\\mu/g, "μ").replace(/\\nu/g, "ν").replace(/\\xi/g, "ξ")
    .replace(/\\pi/g, "π").replace(/\\rho/g, "ρ").replace(/\\sigma/g, "σ")
    .replace(/\\tau/g, "τ").replace(/\\upsilon/g, "υ").replace(/\\phi/g, "φ")
    .replace(/\\chi/g, "χ").replace(/\\psi/g, "ψ").replace(/\\omega/g, "ω")
    // Capital Greek
    .replace(/\\Gamma/g, "Γ").replace(/\\Delta/g, "Δ").replace(/\\Theta/g, "Θ")
    .replace(/\\Lambda/g, "Λ").replace(/\\Xi/g, "Ξ").replace(/\\Pi/g, "Π")
    .replace(/\\Sigma/g, "Σ").replace(/\\Phi/g, "Φ").replace(/\\Psi/g, "Ψ")
    .replace(/\\Omega/g, "Ω")
    // Math operators & symbols
    .replace(/\\sum/g, "∑").replace(/\\prod/g, "∏").replace(/\\int/g, "∫")
    .replace(/\\partial/g, "∂").replace(/\\infty/g, "∞").replace(/\\approx/g, "≈")
    .replace(/\\cdot/g, "·").replace(/\\times/g, "×").replace(/\\pm/g, "±")
    .replace(/\\rightarrow/g, "→").replace(/\\leftarrow/g, "←")
    .replace(/\\Rightarrow/g, "⇒").replace(/\\Leftrightarrow/g, "⇔")
    .replace(/\\leq/g, "≤").replace(/\\geq/g, "≥").replace(/\\neq/g, "≠")
    .replace(/\\propto/g, "∝").replace(/\\sim/g, "∼").replace(/\\equiv/g, "≡")
    // Special symbols
    .replace(/\\nabla/g, "∇").replace(/\\forall/g, "∀").replace(/\\exists/g, "∃")
    .replace(/\\in/g, "∈").replace(/\\notin/g, "∉").replace(/\\subset/g, "⊂")
    .replace(/\\mathbb\{R\}/g, "ℝ").replace(/\\mathbb\{N\}/g, "ℕ")
    .replace(/\\mathbb\{Z\}/g, "ℤ").replace(/\\mathbb\{C\}/g, "ℂ")
    // Calligraphic (use script-like Unicode)
    .replace(/\\mathcal\{L\}/g, "ℒ").replace(/\\mathcal\{F\}/g, "ℱ")
    .replace(/\\mathcal\{C\}/g, "𝒞").replace(/\\mathcal\{N\}/g, "𝒩")
    .replace(/\\mathcal\{L\}_\{/g, "ℒ_").replace(/\\mathcal\{L\}/g, "ℒ")
    // Functions
    .replace(/\\log/g, "log").replace(/\\exp/g, "exp").replace(/\\sin/g, "sin")
    .replace(/\\cos/g, "cos").replace(/\\tan/g, "tan").replace(/\\max/g, "max")
    .replace(/\\min/g, "min").replace(/\\lim/g, "lim")
    .replace(/\\text\{ReLU\}/g, "ReLU").replace(/\\text\{Softmax\}/g, "Softmax")
    .replace(/\\text\{/g, "").replace(/\\operatorname\{/g, "")
    // Fractions: \frac{a}{b} → a/b
    .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1)/($2)")
    // Overbrace/underbrace - just keep the text
    .replace(/\\overbrace\{([^}]+)\}\^\{([^}]+)\}/g, "$1")
    // Left/right delimiters
    .replace(/\\left\(/g, "(").replace(/\\right\)/g, ")")
    .replace(/\\left\[/g, "[").replace(/\\right\]/g, "]")
    .replace(/\\left\\{/g, "{").replace(/\\right\\}/g, "}")
    .replace(/\\left\|/g, "|").replace(/\\right\|/g, "|")
    // Remaining formatting
    .replace(/\\tilde\{([^}]+)\}/g, "$1̃")  // tilde accent
    .replace(/\\hat\{([^}]+)\}/g, "$1̂")   // hat accent
    .replace(/\\bar\{([^}]+)\}/g, "$1̄")   // bar accent
    .replace(/\\dot\{([^}]+)\}/g, "$1̇")   // dot accent
    // Remove remaining backslash commands we can't render
    .replace(/\\[a-zA-Z]+/g, "")
    // Clean up: convert _ to subscript, ^ to superscript (already fine)
    // Clean up double spaces
    .replace(/  +/g, " ")
    // Remove unmatched closing braces
    .replace(/\}/g, "")
    .replace(/\{/g, "")
    .trim();
}

// ====== TEXT PARSING ======
function parseTextRuns(text, baseOpts = {}) {
  // Handle **bold**, inline `code`, $math$
  const runs = [];
  let remaining = text;

  while (remaining.length > 0) {
    // Bold **...**
    let m = remaining.match(/^(.*?)\*\*(.+?)\*\*/s);
    if (m) {
      if (m[1]) runs.push(new TextRun({ text: m[1], font: "Arial", size: 24, ...baseOpts }));
      runs.push(new TextRun({ text: m[2], font: "Arial", size: 24, bold: true, ...baseOpts }));
      remaining = m[3] || "";
      continue;
    }
    // Inline code `...`
    m = remaining.match(/^(.*?)`([^`]+)`/s);
    if (m) {
      if (m[1]) runs.push(new TextRun({ text: m[1], font: "Arial", size: 24, ...baseOpts }));
      runs.push(new TextRun({ text: m[2], font: "Courier New", size: 21, ...baseOpts }));
      remaining = m[3] || "";
      continue;
    }
    // Inline math $...$
    m = remaining.match(/^(.*?)\$([^$]+)\$/s);
    if (m) {
      if (m[1]) runs.push(new TextRun({ text: m[1], font: "Arial", size: 24, ...baseOpts }));
      runs.push(new TextRun({ text: latexToUnicode(m[2]), font: "Cambria Math", size: 24, ...baseOpts }));
      remaining = m[3] || "";
      continue;
    }
    // Plain text
    runs.push(new TextRun({ text: remaining, font: "Arial", size: 24, ...baseOpts }));
    break;
  }
  return runs;
}

// ====== ELEMENT BUILDERS ======
function makePara(text, opts = {}) {
  const runs = typeof text === "string" ? parseTextRuns(text, opts.run || {}) : text;
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    ...opts.para,
    children: runs
  });
}

function makeH1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: parseTextRuns(text, { size: 36, bold: true })
  });
}

function makeH2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: parseTextRuns(text, { size: 30, bold: true })
  });
}

function makeH3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: parseTextRuns(text, { size: 26, bold: true })
  });
}

function makeCode(text) {
  return new Paragraph({
    spacing: { after: 30, line: 280 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Courier New", size: 19 })]
  });
}

function makeFormula(text) {
  const rendered = latexToUnicode(text);
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: rendered, font: "Cambria Math", size: 22 })]
  });
}

function makePageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function makeSpacer(h) {
  return new Paragraph({ spacing: { after: h || 120 }, children: [] });
}

function parseTable(mdTable) {
  const rows = mdTable.trim().split("\n");
  if (rows.length < 2) return null;
  const parseRow = (r) => r.replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
  const headers = parseRow(rows[0]);
  const data = rows.slice(2).map(parseRow);
  const n = headers.length;
  const colW = Math.floor(BW / n);

  return new Table({
    width: { size: BW, type: WidthType.DXA },
    columnWidths: Array(n).fill(colW),
    rows: [
      new TableRow({ children: headers.map(h => new TableCell({
        borders: bds, width: { size: colW, type: WidthType.DXA },
        shading: { fill: "1F4E79", type: ShadingType.CLEAR }, margins: cm,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: h, font: "Arial", size: 19, bold: true, color: "FFFFFF" })] })]
      })) }),
      ...data.map((row, ri) => new TableRow({ children: row.map(cell => new TableCell({
        borders: bds, width: { size: colW, type: WidthType.DXA },
        shading: ri % 2 === 0 ? { fill: "F2F7FB", type: ShadingType.CLEAR } : undefined,
        margins: cm,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: parseTextRuns(cell, { size: 19 }) })]
      })) }))
    ]
  });
}

// ====== MAIN PARSER ======
const elements = [];
let i = 0;
let inFrontMatter = true;
let frontMatterLines = [];

// Collect all lines until first H1 for front matter detection
// The markdown starts with: # title, ## subtitle, ---, metadata, ---, ## 摘要
// We want to skip the title + subtitle for the main content (they go on cover)

while (i < lines.length) {
  const line = lines[i];

  // Skip empty lines
  if (!line.trim()) { i++; continue; }

  // Code block
  if (line.startsWith("```")) {
    i++;
    let codeText = "";
    while (i < lines.length && !lines[i].startsWith("```")) {
      codeText += (codeText ? "\n" : "") + lines[i];
      i++;
    }
    i++; // skip closing ```
    codeText.split("\n").forEach(ct => elements.push(makeCode(ct)));
    continue;
  }

  // Table
  if (line.startsWith("|") && line.endsWith("|")) {
    let tableText = "";
    while (i < lines.length && lines[i].startsWith("|") && lines[i].endsWith("|")) {
      tableText += (tableText ? "\n" : "") + lines[i];
      i++;
    }
    const tbl = parseTable(tableText);
    if (tbl) {
      elements.push(makeSpacer(80));
      elements.push(tbl);
      elements.push(makeSpacer(80));
    }
    continue;
  }

  // H1
  if (line.match(/^# (.+)/) && !line.match(/^## /)) {
    const title = line.replace(/^# /, "");
    // Skip the very first H1 (it's the document title, goes on cover)
    if (elements.length === 0) {
      i++; continue;
    }
    elements.push(makeH1(title));
    i++; continue;
  }

  // H2
  if (line.match(/^## (.+)/) && !line.match(/^### /)) {
    const title = line.replace(/^## /, "");
    // Skip the subtitle H2 right after the title
    if (elements.length === 0) { i++; continue; }
    elements.push(makeH2(title));
    i++; continue;
  }

  // H3
  if (line.match(/^### (.+)/)) {
    elements.push(makeH3(line.replace(/^### /, "")));
    i++; continue;
  }

  // Horizontal rule
  if (line.match(/^---$/)) {
    i++; continue;
  }

  // Blockquote
  if (line.startsWith("> ")) {
    let qt = "";
    while (i < lines.length && lines[i].startsWith("> ")) {
      qt += (qt ? " " : "") + lines[i].replace(/^> /, "");
      i++;
    }
    elements.push(makePara(qt, { para: { indent: { left: 360 } }, run: { italics: true, color: "555555" } }));
    continue;
  }

  // Formula line (contains LaTeX-style math)
  if (line.match(/\\mathcal|\\sum|\\frac|\\text|\\eta|\\gamma|\\varepsilon|\\alpha|\\tilde|\\lambda/) ||
      (line.match(/\$.*\$/) && line.length < 120)) {
    elements.push(makeFormula(line.replace(/\$/g, "").replace(/\\\\/g, "").trim()));
    i++; continue;
  }

  // Regular paragraph
  // Remove markdown links
  let text = line.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  elements.push(makePara(text));
  i++;
}

console.log(`Parsed ${elements.length} elements from markdown`);

// ====== BUILD DOCUMENT ======
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "404040" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } }
    ]
  },
  sections: [
    // ====== COVER PAGE ======
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        makeSpacer(2000),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
          children: [new TextRun({ text: "深度学习与计算机视觉", font: "Arial", size: 36, color: "2E75B6" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
          children: [new TextRun({ text: "课程大作业", font: "Arial", size: 36, color: "2E75B6" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } },
          children: [] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600, line: 600 },
          children: [new TextRun({ text: "基于迁移学习的\n农业植物叶片病害识别", font: "Arial", size: 52, bold: true, color: "1F4E79" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
          children: [new TextRun({ text: "PlantVillage 15 类别细粒度图像分类\n系统消融实验与模型集成", font: "Arial", size: 26, color: "666666" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "学    号：XXXXXXXX", font: "Arial", size: 24, color: "333333" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "姓    名：XXX", font: "Arial", size: 24, color: "333333" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "日    期：2026 年 6 月 15 日", font: "Arial", size: 24, color: "333333" })] }),
      ]
    },
    // ====== MAIN CONTENT ======
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
            children: [new TextRun({ text: "基于迁移学习的农业植物叶片病害识别", font: "Arial", size: 18, color: "888888", italics: true })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 4 } },
            children: [
              new TextRun({ text: "— ", font: "Arial", size: 18, color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }),
              new TextRun({ text: " —", font: "Arial", size: 18, color: "888888" }),
            ]
          })]
        })
      },
      children: [
        // TOC
        makeH1("目  录"),
        new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }),
        makePageBreak(),
        // All parsed content
        ...elements
      ]
    }
  ]
});

// ====== WRITE ======
Packer.toBuffer(doc).then(buffer => {
  const outPath = "/home/nini/文档/深度学习大作业/课堂作业/results/课堂作业报告.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Done! Saved to ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
