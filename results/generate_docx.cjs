const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  TableOfContents
} = require("docx");

// Read the markdown file
const md = fs.readFileSync("/home/nini/文档/深度学习大作业/课堂作业/results/课堂作业报告正文.md", "utf-8");
const lines = md.split("\n");

// ====== GLOBALS ======
const BW = 9360;
const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const cm = { top: 60, bottom: 60, left: 100, right: 120 };

// ====== HELPERS ======
function mkTextRun(text, opts = {}) {
  // Handle bold markers **text**
  if (text.includes("**")) {
    const parts = [];
    let remaining = text;
    while (remaining.length > 0) {
      const m = remaining.match(/^(.*?)\*\*(.+?)\*\*(.*)/s);
      if (m) {
        if (m[1]) parts.push(new TextRun({ text: m[1], font: "Arial", size: 24, ...opts }));
        parts.push(new TextRun({ text: m[2], font: "Arial", size: 24, bold: true, ...opts }));
        remaining = m[3];
      } else {
        parts.push(new TextRun({ text: remaining, font: "Arial", size: 24, ...opts }));
        break;
      }
    }
    return parts;
  }
  return [new TextRun({ text, font: "Arial", size: 24, ...opts })];
}

function mkPara(text, opts = {}) {
  const runs = mkTextRun(text, opts.run || {});
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    ...opts.para,
    children: runs
  });
}

function mkCode(text) {
  return new Paragraph({
    spacing: { after: 40, line: 280 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Courier New", size: 19 })]
  });
}

function mkFormula(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: "Arial", size: 22, italics: true })]
  });
}

function mkH1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: mkTextRun(text, { size: 36, bold: true })
  });
}

function mkH2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: mkTextRun(text, { size: 30, bold: true })
  });
}

function mkH3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: mkTextRun(text, { size: 26, bold: true })
  });
}

function mkPageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function mkSpacer(h = 120) {
  return new Paragraph({ spacing: { after: h }, children: [] });
}

function parseTable(mdTable) {
  // Parse markdown table: first line = headers, second line = ---, rest = data
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
        borders,
        width: { size: colW, type: WidthType.DXA },
        shading: { fill: "1F4E79", type: ShadingType.CLEAR },
        margins: cm,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: mkTextRun(h, { size: 19, bold: true, color: "FFFFFF" })
        })]
      })) }),
      ...data.map((row, ri) => new TableRow({ children: row.map(cell => new TableCell({
        borders,
        width: { size: colW, type: WidthType.DXA },
        shading: ri % 2 === 0 ? { fill: "F2F7FB", type: ShadingType.CLEAR } : undefined,
        margins: cm,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: mkTextRun(cell, { size: 19 })
        })]
      })) }))
    ]
  });
}

// ====== PARSE MARKDOWN ======
const elements = [];
let i = 0;

while (i < lines.length) {
  const line = lines[i];

  // Skip empty lines
  if (!line.trim()) { i++; continue; }

  // H1: # xxx
  if (line.match(/^# (.+)/) && !line.match(/^## /)) {
    elements.push(mkH1(line.replace(/^# /, "")));
    i++; continue;
  }

  // H2: ## xxx
  if (line.match(/^## (.+)/) && !line.match(/^### /)) {
    elements.push(mkH2(line.replace(/^## /, "")));
    i++; continue;
  }

  // H3: ### xxx
  if (line.match(/^### (.+)/)) {
    elements.push(mkH3(line.replace(/^### /, "")));
    i++; continue;
  }

  // Horizontal rule ---
  if (line.match(/^---$/)) {
    elements.push(mkSpacer(60));
    i++; continue;
  }

  // Code block
  if (line.startsWith("```")) {
    i++;
    let codeText = "";
    while (i < lines.length && !lines[i].startsWith("```")) {
      codeText += (codeText ? "\n" : "") + lines[i];
      i++;
    }
    // Split multi-line code into separate paragraphs
    codeText.split("\n").forEach(ct => {
      elements.push(mkCode(ct));
    });
    i++; // skip closing ```
    continue;
  }

  // Table detection
  if (line.startsWith("|") && line.endsWith("|")) {
    let tableText = "";
    while (i < lines.length && lines[i].startsWith("|") && lines[i].endsWith("|")) {
      tableText += (tableText ? "\n" : "") + lines[i];
      i++;
    }
    const tbl = parseTable(tableText);
    if (tbl) {
      elements.push(mkSpacer(60));
      elements.push(tbl);
      elements.push(mkSpacer(60));
    }
    continue;
  }

  // Blockquote
  if (line.startsWith("> ")) {
    let quoteText = "";
    while (i < lines.length && lines[i].startsWith("> ")) {
      quoteText += (quoteText ? " " : "") + lines[i].replace(/^> /, "");
      i++;
    }
    elements.push(mkPara(quoteText, { para: { indent: { left: 360 } }, run: { italics: true, color: "555555" } }));
    continue;
  }

  // Formula (inline $...$)
  if (line.includes("$") && (line.match(/\$.+\$/) || line.match(/\\mathcal/))) {
    elements.push(mkFormula(line.replace(/\$/g, "").trim()));
    i++; continue;
  }

  // Regular paragraph
  // Check for bold markers and links
  let text = line;
  // Remove markdown links [text](url) -> just text
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  elements.push(mkPara(text));
  i++;
}

// ====== BUILD DOCUMENT ======
// Find the cover page content (it's the first H1 and its surrounding text)
// Actually, let me handle the cover separately

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
    // Cover page
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        mkSpacer(2000),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "深度学习与计算机视觉", font: "Arial", size: 36, color: "2E75B6" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: "课程大作业", font: "Arial", size: 36, color: "2E75B6" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } }, children: [] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600, line: 600 }, children: [new TextRun({ text: "基于迁移学习的\n农业植物叶片病害识别", font: "Arial", size: 52, bold: true, color: "1F4E79" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 }, children: [new TextRun({ text: "PlantVillage 15 类别细粒度图像分类\n系统消融实验与模型集成", font: "Arial", size: 26, color: "666666" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "学    号：XXXXXXXX", font: "Arial", size: 24, color: "333333" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "姓    名：XXX", font: "Arial", size: 24, color: "333333" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "日    期：2026 年 6 月 15 日", font: "Arial", size: 24, color: "333333" })] }),
      ]
    },
    // Main content
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
        ...(() => {
          // Find H1s in elements to build TOC entries
          const tocChildren = [mkH1("目录")];
          // Add TOC field
          tocChildren.push(new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }));
          tocChildren.push(mkPageBreak());
          return tocChildren;
        })(),
        // Remove duplicate H1 (the first one is the title "基于迁移学习...")
        // Skip the first H1 and H2 that form the cover info, keep the rest
        ...elements.filter((el, idx) => {
          // Skip the first few elements that are cover-page material
          // The title H1 is the very first element
          if (idx === 0 && el.type === "paragraph" && el._heading === "HEADING_1") return false;
          return true;
        })
      ]
    }
  ]
});

// Generate
console.log(`Generated ${elements.length} elements from markdown`);
Packer.toBuffer(doc).then(buffer => {
  const outPath = "/home/nini/文档/深度学习大作业/课堂作业/results/课堂作业报告.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Document saved to ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
}).catch(err => {
  console.error("Error generating document:", err);
  process.exit(1);
});
