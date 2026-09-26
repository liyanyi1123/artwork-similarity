#!/usr/bin/env python3
"""
Use ViT as the baseline and compare six fallback strategies across four thresholds.

Strategies:
- ViT only
- ViT + CLIP
- ViT + pHash
- ViT + aHash
- ViT + dHash
- ViT + wHash
- ViT + PDQ

For each threshold in {0.90, 0.89, 0.88, 0.87}:
1. If ViT similarity >= threshold, count as correct.
2. If ViT similarity < threshold, try one fallback metric.
3. If fallback similarity > threshold, count as a rescued correct case.

Outputs:
- One Excel file per threshold
- One combined four-threshold line chart
"""

from __future__ import annotations

import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xlsxwriter

# 添加项目根目录到路径
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.utils.file_manager import ProjectPaths

# 使用路径管理器
paths = ProjectPaths(project_root)
THRESHOLD_EVAL_DIR = paths.threshold_evaluation

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# 更新默认路径
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = project_root
TABLE_DIR = paths.experiments / "ex2" / "tables"

THRESHOLDS = [0.90, 0.89, 0.88, 0.87]
THRESHOLD_TAGS = {
    0.90: "090",
    0.89: "089",
    0.88: "088",
    0.87: "087",
}

# 生成带日期的文件名
date_str = datetime.now().strftime("%Y%m%d")
EXCEL_TEMPLATE = f"vit_all_fallback_comparison_{{tag}}_{date_str}.xlsx"
IMAGE_OUTPUT = f"vit_all_fallback_4thresholds_line_only_{date_str}.png"

FALLBACK_COLUMNS = {
    "CLIP": "CLIP Cosine",
    "pHash": "pHash Sim",
    "aHash": "aHash Sim",
    "dHash": "dHash Sim",
    "wHash": "wHash Sim",
    "PDQ": "PDQ Sim",
}


@dataclass
class ReportRow:
    output_filename: str
    vit_similarity: float
    fallback_values: dict[str, float]


def load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", NS):
        strings.append("".join(node.text or "" for node in item.iterfind(".//a:t", NS)))
    return strings


def get_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", NS)
    inline_node = cell.find("a:is", NS)
    if cell_type == "s" and value_node is not None:
        return shared_strings[int(value_node.text)]
    if cell_type == "inlineStr" and inline_node is not None:
        return "".join(node.text or "" for node in inline_node.iterfind(".//a:t", NS))
    if value_node is not None and value_node.text is not None:
        return value_node.text
    return ""


def read_sheet_rows(xlsx_path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = load_shared_strings(archive)
        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet_root.findall(".//a:sheetData/a:row", NS):
        values: dict[str, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)(\d+)", ref)
            if not match:
                continue
            values[match.group(1)] = get_cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def iter_report_rows(
    table_dir: Path,
    include_f2: bool = False,
    include_test_report: bool = False,
) -> list[ReportRow]:
    report_rows: list[ReportRow] = []

    for report_path in sorted(table_dir.glob("*_report.xlsx")):
        if not include_test_report and report_path.name == "test_report.xlsx":
            continue

        rows = read_sheet_rows(report_path)
        if not rows:
            continue

        header_map = {value: key for key, value in rows[0].items()}
        filename_col = header_map.get("Output Filename")
        vit_col = header_map.get("ViT Cosine")
        if not filename_col or not vit_col:
            continue

        fallback_cols = {}
        for method_name, column_name in FALLBACK_COLUMNS.items():
            col = header_map.get(column_name)
            if col:
                fallback_cols[method_name] = col
        if len(fallback_cols) != len(FALLBACK_COLUMNS):
            continue

        for row in rows[1:]:
            output_filename = row.get(filename_col, "").strip()
            vit_value = row.get(vit_col, "").strip()
            if not output_filename or not vit_value:
                continue
            if not include_f2 and (
                output_filename.startswith("F2") or "F2_" in output_filename
            ):
                continue

            fallback_values = {}
            missing = False
            for method_name, col in fallback_cols.items():
                raw_value = row.get(col, "").strip()
                if not raw_value:
                    missing = True
                    break
                fallback_values[method_name] = float(raw_value)
            if missing:
                continue

            report_rows.append(
                ReportRow(
                    output_filename=output_filename,
                    vit_similarity=float(vit_value),
                    fallback_values=fallback_values,
                )
            )

    return report_rows


def analyze_threshold(report_rows: list[ReportRow], threshold: float) -> list[dict[str, object]]:
    total_cases = len(report_rows)
    baseline_correct = sum(row.vit_similarity >= threshold for row in report_rows)
    baseline_accuracy = baseline_correct / total_cases if total_cases else 0.0

    results = [
        {
            "method": "ViT only",
            "total_cases": total_cases,
            "baseline_correct": baseline_correct,
            "rescued_cases": 0,
            "combined_correct": baseline_correct,
            "accuracy": baseline_accuracy,
            "accuracy_gain_pct_points": 0.0,
        }
    ]

    for method_name in FALLBACK_COLUMNS:
        rescued_cases = 0
        combined_correct = 0
        for row in report_rows:
            vit_pass = row.vit_similarity >= threshold
            rescued = (row.vit_similarity < threshold) and (
                row.fallback_values[method_name] > threshold
            )
            rescued_cases += int(rescued)
            combined_correct += int(vit_pass or rescued)

        accuracy = combined_correct / total_cases if total_cases else 0.0
        results.append(
            {
                "method": f"ViT + {method_name}",
                "total_cases": total_cases,
                "baseline_correct": baseline_correct,
                "rescued_cases": rescued_cases,
                "combined_correct": combined_correct,
                "accuracy": accuracy,
                "accuracy_gain_pct_points": (accuracy - baseline_accuracy) * 100,
            }
        )
    return results


def save_excel(rows: list[dict[str, object]], output_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(output_path))
    worksheet = workbook.add_worksheet("overall_comparison")
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#DCE6F1", "border": 1})
    pct_fmt = workbook.add_format({"num_format": "0.00%"})

    headers = [
        "method",
        "total_cases",
        "baseline_correct",
        "rescued_cases",
        "combined_correct",
        "accuracy",
        "accuracy_gain_pct_points",
    ]
    worksheet.set_column(0, 0, 18)
    worksheet.set_column(1, 4, 16)
    worksheet.set_column(5, 6, 18)
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_fmt)

    for row_idx, row in enumerate(rows, start=1):
        worksheet.write(row_idx, 0, row["method"])
        worksheet.write_number(row_idx, 1, int(row["total_cases"]))
        worksheet.write_number(row_idx, 2, int(row["baseline_correct"]))
        worksheet.write_number(row_idx, 3, int(row["rescued_cases"]))
        worksheet.write_number(row_idx, 4, int(row["combined_correct"]))
        worksheet.write_number(row_idx, 5, float(row["accuracy"]), pct_fmt)
        worksheet.write_number(
            row_idx,
            6,
            float(row["accuracy_gain_pct_points"]) / 100.0,
            pct_fmt,
        )
    workbook.close()


def plot_combined_line(all_results: dict[float, list[dict[str, object]]], output_path: Path) -> None:
    method_order = [
        row["method"].replace("ViT only", "ViT").replace("ViT + ", "+")
        for row in all_results[THRESHOLDS[0]]
    ]

    colors = {0.90: "#4C78A8", 0.89: "#F58518", 0.88: "#54A24B", 0.87: "#E45756"}
    markers = {0.90: "o", 0.89: "s", 0.88: "^", 0.87: "D"}
    offsets = {0.90: 0.018, 0.89: 0.038, 0.88: -0.038, 0.87: -0.018}

    plt.figure(figsize=(12.5, 7.2))
    for threshold in THRESHOLDS:
        values = [float(row["accuracy"]) * 100 for row in all_results[threshold]]
        plt.plot(
            method_order,
            values,
            marker=markers[threshold],
            linewidth=2.4,
            markersize=7,
            color=colors[threshold],
            label=f"Threshold = {threshold:.2f}",
        )
        for idx, value in enumerate(values):
            dy = offsets[threshold]
            va = "bottom" if dy > 0 else "top"
            plt.text(
                idx,
                value + dy,
                f"{value:.2f}%",
                ha="center",
                va=va,
                fontsize=8.5,
                color=colors[threshold],
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.22,
                },
            )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Method")
    plt.ylim(99.0, 100.0)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend(frameon=False, ncol=2, loc="lower right")
    plt.title("Accuracy Comparison Across Four ViT Thresholds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    report_rows = iter_report_rows(TABLE_DIR)
    if not report_rows:
        raise SystemExit(f"No valid report rows found in {TABLE_DIR}")

    # 确保输出目录存在
    results_dir = THRESHOLD_EVAL_DIR / "results"
    plots_dir = THRESHOLD_EVAL_DIR / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[float, list[dict[str, object]]] = {}
    for threshold in THRESHOLDS:
        rows = analyze_threshold(report_rows, threshold)
        all_results[threshold] = rows
        excel_path = results_dir / EXCEL_TEMPLATE.format(tag=THRESHOLD_TAGS[threshold])
        save_excel(rows, excel_path)
        print(f"Saved Excel: {excel_path}")
        for row in rows:
            print(
                f"{threshold:.2f} | {row['method']}: rescued={row['rescued_cases']}, "
                f"accuracy={row['accuracy'] * 100:.2f}%, "
                f"gain={row['accuracy_gain_pct_points']:.2f} pts"
            )

    image_path = plots_dir / IMAGE_OUTPUT
    plot_combined_line(all_results, image_path)
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()
