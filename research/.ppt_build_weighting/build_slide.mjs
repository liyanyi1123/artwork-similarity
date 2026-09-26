import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = fileURLToPath(new URL("../.ppt_build_weighting", import.meta.url));
const FINAL_PPTX = fileURLToPath(new URL("../high_performance_region_weighting.pptx", import.meta.url));

const C = {
  ink: "#162235",
  muted: "#596579",
  pale: "#F1F4F7",
  rule: "#D2D9E2",
  accent: "#2D82C4",
  accentPale: "#E6F2FA",
  white: "#FFFFFF",
};

function addText(slide, name, value, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    typeface: "Helvetica Neue",
    fontSize: 18,
    color: C.ink,
    ...style,
  };
  return box;
}

function addRect(slide, name, position, fill, line = "none") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: line === "none"
      ? { style: "solid", fill: "none", width: 0 }
      : { style: "solid", fill: line, width: 1 },
  });
}

function styleTable(table, rows, columns, options = {}) {
  table.borders.assign({ style: "solid", fill: C.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: rows, columnCount: columns }).assign({
    textStyle: {
      typeface: "Helvetica Neue",
      fontSize: options.fontSize ?? 16,
      color: C.ink,
    },
    margins: { left: 12, right: 12, top: 7, bottom: 7 },
    anchor: "middle",
  });
  for (let c = 0; c < columns; c += 1) {
    const cell = table.getCell(0, c);
    cell.fill = C.ink;
    cell.text.style = {
      typeface: "Helvetica Neue",
      fontSize: options.headerSize ?? 15,
      bold: true,
      color: C.white,
    };
  }
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBytes(path) {
  const bytes = await fs.readFile(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });
  const slide = presentation.slides.add();
  slide.background.fill = C.white;

  addText(
    slide,
    "title",
    "Joint Thresholds from High-Performance Region Weighting",
    { left: 56, top: 32, width: 1168, height: 52 },
    { fontSize: 38, bold: true, color: C.ink },
  );
  addText(
    slide,
    "subtitle",
    "Each dataset is weighted by the breadth of its threshold region achieving F1 > 0.99.",
    { left: 56, top: 88, width: 1168, height: 30 },
    { fontSize: 18, color: C.muted },
  );
  addRect(slide, "top-rule", { left: 56, top: 126, width: 1168, height: 2 }, C.ink);

  addText(
    slide,
    "step-1-label",
    "1   DEFINE DATASET WEIGHTS",
    { left: 56, top: 151, width: 690, height: 25 },
    { fontSize: 16, bold: true, color: C.accent },
  );
  addRect(slide, "formula-panel", { left: 56, top: 184, width: 690, height: 148 }, C.pale);
  slide.images.add({
    blob: await readImageBytes(`${OUT_DIR}/formula_count.png`),
    contentType: "image/png",
    alt: "n sub i equals the number of CLIP and ViT threshold pairs for which dataset i has F1 greater than 0.99",
    fit: "contain",
    position: { left: 86, top: 203, width: 610, height: 46 },
  });
  slide.images.add({
    blob: await readImageBytes(`${OUT_DIR}/formula_weight.png`),
    contentType: "image/png",
    alt: "dataset weight equals n sub i divided by the sum of n sub j",
    fit: "contain",
    position: { left: 86, top: 263, width: 300, height: 52 },
  });

  addText(
    slide,
    "step-2-label",
    "2   COMPUTE THE WEIGHTED JOINT SCORE",
    { left: 56, top: 358, width: 690, height: 25 },
    { fontSize: 16, bold: true, color: C.accent },
  );
  slide.images.add({
    blob: await readImageBytes(`${OUT_DIR}/formula_objective.png`),
    contentType: "image/png",
    alt: "J of t equals the sum over datasets of the dataset weight multiplied by its F1 at threshold t",
    fit: "contain",
    position: { left: 56, top: 391, width: 360, height: 56 },
  });

  const weights = slide.tables.add({
    rows: 4,
    columns: 3,
    left: 56,
    top: 468,
    width: 690,
    height: 156,
    columnWidths: [330, 190, 170],
    values: [
      ["Dataset", "Valid groups", "Weight"],
      ["WikiArt1k", "243", "0.3938"],
      ["ArtBench1k", "374", "0.6062"],
      ["Total", "617", "1.0000"],
    ],
  });
  styleTable(weights, 4, 3, { fontSize: 16, headerSize: 15 });
  for (let c = 0; c < 3; c += 1) {
    weights.getCell(3, c).fill = C.pale;
    weights.getCell(3, c).text.style = {
      typeface: "Helvetica Neue",
      fontSize: 16,
      bold: true,
      color: C.ink,
    };
  }

  addRect(slide, "threshold-panel", { left: 790, top: 151, width: 434, height: 215 }, C.accentPale);
  addRect(slide, "threshold-accent", { left: 790, top: 151, width: 8, height: 215 }, C.accent);
  addText(
    slide,
    "threshold-label",
    "OPTIMAL JOINT THRESHOLDS",
    { left: 824, top: 178, width: 360, height: 25 },
    { fontSize: 16, bold: true, color: C.accent },
  );
  addText(
    slide,
    "threshold-value",
    "CLIP > 0.87\nViT > 0.50",
    { left: 824, top: 218, width: 360, height: 112 },
    { typeface: "Cambria Math", fontSize: 32, bold: true, color: C.ink },
  );

  addText(
    slide,
    "results-label",
    "RESULTING PERFORMANCE",
    { left: 790, top: 398, width: 434, height: 25 },
    { fontSize: 16, bold: true, color: C.accent },
  );
  const results = slide.tables.add({
    rows: 4,
    columns: 2,
    left: 790,
    top: 435,
    width: 434,
    height: 189,
    columnWidths: [270, 164],
    values: [
      ["Dataset", "F1"],
      ["WikiArt1k", "0.992661"],
      ["ArtBench1k", "0.994475"],
      ["Weighted F1", "0.993761"],
    ],
  });
  styleTable(results, 4, 2, { fontSize: 16, headerSize: 15 });
  for (let c = 0; c < 2; c += 1) {
    results.getCell(3, c).fill = C.accentPale;
    results.getCell(3, c).text.style = {
      typeface: "Helvetica Neue",
      fontSize: 17,
      bold: true,
      color: C.accent,
    };
  }

  addText(
    slide,
    "takeaway",
    "The weighted optimum maintains near-perfect F1 across both datasets.",
    { left: 56, top: 652, width: 1080, height: 28 },
    { fontSize: 17, bold: true, color: C.ink },
  );
  addText(
    slide,
    "page-number",
    "1",
    { left: 1170, top: 650, width: 54, height: 26 },
    { fontSize: 14, color: C.muted, alignment: "right" },
  );

  slide.speakerNotes.textFrame.setText(
    "[Sources]\n- User-provided screenshot: codex-clipboard-440483b2-d434-417a-9d28-50303cd56a15.png (content extraction and translation).",
  );

  const png = await presentation.export({ slide, format: "png", scale: 2 });
  await writeBlob(`${OUT_DIR}/slide-1.png`, png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${OUT_DIR}/slide-1.layout.json`, await layout.text());
  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,table,notes",
    maxChars: 20000,
  });
  await fs.writeFile(`${OUT_DIR}/inspect.ndjson`, inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
