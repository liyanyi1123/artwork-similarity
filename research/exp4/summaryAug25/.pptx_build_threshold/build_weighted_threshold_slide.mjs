import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import { Presentation, PresentationFile, layers, shape, text } from "@oai/artifact-tool";

const FINAL_PPTX = fileURLToPath(new URL("../../../exp4/summaryAug25/weighted_threshold_selection.pptx", import.meta.url));
const PREVIEW_PNG = fileURLToPath(new URL("../../../exp4/summaryAug25/.pptx_build_threshold/rendered/slide-1.png", import.meta.url));
const LAYOUT_JSON = fileURLToPath(new URL("../../../exp4/summaryAug25/.pptx_build_threshold/rendered/slide-1.layout.json", import.meta.url));

const FONT = "PingFang SC";
const MATH_FONT = "Helvetica Neue";
const BLACK = "#171717";
const MUTED = "#5F6368";
const LIGHT = "#F2F3F4";
const LIGHTER = "#F7F8F8";
const TEAL = "#0B6E69";
const TEAL_LIGHT = "#E5F2F0";

function paragraph(run, { size = 20, color = BLACK, bold = false, typeface = FONT } = {}) {
  return {
    runs: [{ run, textStyle: { fontSize: `${size}px`, typeface, color, bold } }],
    paragraphStyle: { lineSpacingPercent: 112000 },
  };
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";

  slide.compose(
    layers({ name: "weighted-threshold-selection", width: "fill", height: "fill" }, [
      shape({
        name: "accent-line",
        geometry: "rect",
        fill: TEAL,
        position: { left: 42, top: 31 },
        width: 82,
        height: 5,
      }),

      text([paragraph("CROSS-DATASET THRESHOLD SELECTION", { size: 13, color: MUTED, bold: true, typeface: MATH_FONT })], {
        name: "eyebrow",
        position: { left: 42, top: 43, width: 500, height: 24 },
        width: 500,
        height: 24,
        style: { fontSize: "13px", typeface: MATH_FONT, color: MUTED, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("用高性能区域宽度确定统一阈值", { size: 48, bold: true })], {
        name: "title",
        position: { left: 42, top: 73, width: 1196, height: 65 },
        width: 1196,
        height: 65,
        style: { fontSize: "48px", typeface: FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("两个数据集的独立最优点不同；以 F1 > 0.99 的阈值区域宽度构建数据集权重。", { size: 19, color: MUTED })], {
        name: "subtitle",
        position: { left: 42, top: 139, width: 1196, height: 30 },
        width: 1196,
        height: 30,
        style: { fontSize: "19px", typeface: FONT, color: MUTED, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      shape({
        name: "method-band",
        geometry: "rect",
        fill: LIGHTER,
        position: { left: 42, top: 184 },
        width: 1196,
        height: 123,
      }),

      text([paragraph("加权定义", { size: 18, color: TEAL, bold: true })], {
        name: "method-label",
        position: { left: 70, top: 205, width: 135, height: 26 },
        width: 135,
        height: 26,
        style: { fontSize: "18px", typeface: FONT, color: TEAL, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("n_i = # { (t_CLIP, t_ViT) : F1_i > 0.99 }", { size: 24, bold: true, typeface: MATH_FONT })], {
        name: "formula-count",
        position: { left: 225, top: 199, width: 455, height: 36 },
        width: 455,
        height: 36,
        style: { fontSize: "24px", typeface: MATH_FONT, color: BLACK, autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("w_i = n_i / sum_j n_j", { size: 24, bold: true, typeface: MATH_FONT })], {
        name: "formula-weight",
        position: { left: 705, top: 199, width: 285, height: 36 },
        width: 285,
        height: 36,
        style: { fontSize: "24px", typeface: MATH_FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("J(t) = sum_i w_i F1_i(t)", { size: 24, bold: true, typeface: MATH_FONT })], {
        name: "formula-joint",
        position: { left: 70, top: 252, width: 400, height: 36 },
        width: 400,
        height: 36,
        style: { fontSize: "24px", typeface: MATH_FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("满足条件的阈值对合计 617 组", { size: 18, color: MUTED })], {
        name: "formula-note",
        position: { left: 830, top: 256, width: 365, height: 28 },
        width: 365,
        height: 28,
        style: { fontSize: "18px", typeface: FONT, color: MUTED, alignment: "right", autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      shape({ name: "wiki-panel", geometry: "rect", fill: LIGHT, position: { left: 42, top: 331 }, width: 374, height: 300 }),
      shape({ name: "artbench-panel", geometry: "rect", fill: LIGHT, position: { left: 453, top: 331 }, width: 374, height: 300 }),
      shape({ name: "final-panel", geometry: "rect", fill: TEAL_LIGHT, position: { left: 864, top: 331 }, width: 374, height: 300 }),

      text([paragraph("WikiArt1k", { size: 22, bold: true })], {
        name: "wiki-label",
        position: { left: 72, top: 359, width: 310, height: 32 },
        width: 310,
        height: 32,
        style: { fontSize: "22px", typeface: FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([paragraph("243", { size: 58, bold: true, typeface: MATH_FONT })], {
        name: "wiki-count",
        position: { left: 72, top: 399, width: 310, height: 76 },
        width: 310,
        height: 76,
        style: { fontSize: "58px", typeface: MATH_FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([
        paragraph("F1 > 0.99 的阈值对", { size: 18, color: MUTED }),
        paragraph("权重  0.3938", { size: 21, bold: true }),
        paragraph("最终阈值下 F1  0.992661", { size: 18, color: MUTED }),
      ], {
        name: "wiki-details",
        position: { left: 72, top: 493, width: 310, height: 107 },
        width: 310,
        height: 107,
        style: { fontSize: "18px", typeface: FONT, color: MUTED, autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("ArtBench1k", { size: 22, bold: true })], {
        name: "artbench-label",
        position: { left: 483, top: 359, width: 310, height: 32 },
        width: 310,
        height: 32,
        style: { fontSize: "22px", typeface: FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([paragraph("374", { size: 58, bold: true, typeface: MATH_FONT })], {
        name: "artbench-count",
        position: { left: 483, top: 399, width: 310, height: 76 },
        width: 310,
        height: 76,
        style: { fontSize: "58px", typeface: MATH_FONT, color: BLACK, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([
        paragraph("F1 > 0.99 的阈值对", { size: 18, color: MUTED }),
        paragraph("权重  0.6062", { size: 21, bold: true }),
        paragraph("最终阈值下 F1  0.994475", { size: 18, color: MUTED }),
      ], {
        name: "artbench-details",
        position: { left: 483, top: 493, width: 310, height: 107 },
        width: 310,
        height: 107,
        style: { fontSize: "18px", typeface: FONT, color: MUTED, autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("统一最优阈值", { size: 22, color: TEAL, bold: true })], {
        name: "final-label",
        position: { left: 894, top: 359, width: 310, height: 32 },
        width: 310,
        height: 32,
        style: { fontSize: "22px", typeface: FONT, color: TEAL, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([
        paragraph("CLIP > 0.87", { size: 36, color: TEAL, bold: true, typeface: MATH_FONT }),
        paragraph("ViT  > 0.50", { size: 36, color: TEAL, bold: true, typeface: MATH_FONT }),
      ], {
        name: "final-threshold",
        position: { left: 894, top: 404, width: 310, height: 94 },
        width: 310,
        height: 94,
        style: { fontSize: "36px", typeface: MATH_FONT, color: TEAL, autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      shape({ name: "final-rule", geometry: "rect", fill: TEAL, position: { left: 894, top: 515 }, width: 62, height: 4 }),
      text([
        paragraph("加权 F1", { size: 18, color: MUTED }),
        paragraph("0.993761", { size: 30, color: BLACK, bold: true, typeface: MATH_FONT }),
      ], {
        name: "final-f1",
        position: { left: 894, top: 536, width: 310, height: 72 },
        width: 310,
        height: 72,
        style: { fontSize: "24px", typeface: FONT, color: BLACK, autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),

      text([paragraph("结论：加权联合得分在 CLIP > 0.87、ViT > 0.50 时达到最大。", { size: 17, color: MUTED })], {
        name: "conclusion",
        position: { left: 42, top: 656, width: 1080, height: 28 },
        width: 1080,
        height: 28,
        style: { fontSize: "17px", typeface: FONT, color: MUTED, autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
      text([paragraph("EXP. 4  |  26 AUG 2026", { size: 12, color: MUTED, typeface: MATH_FONT })], {
        name: "footer",
        position: { left: 1060, top: 659, width: 178, height: 22 },
        width: 178,
        height: 22,
        style: { fontSize: "12px", typeface: MATH_FONT, color: MUTED, alignment: "right", autoFit: "none", insets: { top: 0, right: 0, bottom: 0, left: 0 } },
      }),
    ]),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );

  slide.speakerNotes.textFrame.setText(
    "[Sources]\n" +
      "- Local sweep: exp4/datasets6_sweep/threshold_sweep_summary.csv\n" +
      "- Local sweep: exp4/threshold_eval/full_sweep/threshold_sweep_summary.csv\n" +
      "- Derived calculation: count threshold pairs with F1 > 0.99, normalize weights by 617, and maximize J(t)."
  );

  const png = await presentation.export({ slide, format: "png", scale: 2 });
  await writeBlob(PREVIEW_PNG, png);

  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(LAYOUT_JSON, await layout.text(), "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
