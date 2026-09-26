#!/usr/bin/env python3
"""
Compare CLIP baseline and CLIP+6 fallback strategies in one table and one chart.

Strategies:
- CLIP only
- CLIP + ViT
- CLIP + pHash
- CLIP + aHash
- CLIP + dHash
- CLIP + wHash
- CLIP + PDQ

Rule:
1. If CLIP similarity >= 0.90, count as correct.
2. If CLIP similarity < 0.90, try one fallback metric.
3. If fallback similarity > 0.90, count as a rescued correct case.

This script reads the existing Excel reports in ``ex1_2/table`` directly and
does not rerun any image models. By default it excludes:
- ``F2_Vertical_flip`` rows
- ``test_report.xlsx``
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xlsxwriter

# 添加项目根目录到路径
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.utils.file_manager import ProjectPaths, get_filename

# 使用路径管理器
paths = ProjectPaths(project_root)
THRESHOLD_EVAL_DIR = paths.threshold_evaluation

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

DEFAULT_CLIP_THRESHOLD = 0.90
DEFAULT_FALLBACK_THRESHOLD = 0.90

# 更新默认路径
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = project_root
DEFAULT_TABLE_DIR = paths.experiments / "ex2" / "tables"
DEFAULT_EXCEL_OUTPUT = THRESHOLD_EVAL_DIR / "results" / get_filename("clip_all_fallback_comparison", "xlsx")
DEFAULT_IMAGE_OUTPUT = THRESHOLD_EVAL_DIR / "plots" / get_filename("clip_all_fallback_line_comparison", "png")

FALLBACK_COLUMNS = {
    "ViT": "ViT Cosine",
    "pHash": "pHash Sim",
    "aHash": "aHash Sim",
    "dHash": "dHash Sim",
    "wHash": "wHash Sim",
    "PDQ": "PDQ Sim",
}


@dataclass
class ReportRow:
    report_name: str
    output_filename: str
    clip_similarity: float
    fallback_values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CLIP baseline and all 6 fallback strategies."
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"Directory containing *_report.xlsx files (default: {DEFAULT_TABLE_DIR})",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=DEFAULT_EXCEL_OUTPUT,
        help=f"Excel output path (default: {DEFAULT_EXCEL_OUTPUT})",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help=f"Image output path (default: {DEFAULT_IMAGE_OUTPUT})",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=DEFAULT_CLIP_THRESHOLD,
        help=f"CLIP acceptance threshold (default: {DEFAULT_CLIP_THRESHOLD})",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=DEFAULT_FALLBACK_THRESHOLD,
        help=f"Fallback threshold; strictly greater than this value passes (default: {DEFAULT_FALLBACK_THRESHOLD})",
    )
    parser.add_argument(
        "--include-f2",
        action="store_true",
        help="Include F2_Vertical_flip rows in the analysis.",
    )
    parser.add_argument(
        "--include-test-report",
        action="store_true",
        help="Include test_report.xlsx in the analysis.",
    )
    return parser.parse_args()


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
        clip_col = header_map.get("CLIP Cosine")
        if not filename_col or not clip_col:
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
            clip_value = row.get(clip_col, "").strip()
            if not output_filename or not clip_value:
                continue

            if not include_f2 and (
                output_filename.startswith("F2") or "F2_" in output_filename
            ):
                continue

            fallback_values = {}
            missing_value = False
            for method_name, col in fallback_cols.items():
                raw_value = row.get(col, "").strip()
                if not raw_value:
                    missing_value = True
                    break
                fallback_values[method_name] = float(raw_value)
            if missing_value:
                continue

            report_rows.append(
                ReportRow(
                    report_name=report_path.name,
                    output_filename=output_filename,
                    clip_similarity=float(clip_value),
                    fallback_values=fallback_values,
                )
            )

    return report_rows


def analyze_methods(
    report_rows: list[ReportRow],
    clip_threshold: float,
    fallback_threshold: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    total_cases = len(report_rows)
    clip_correct = sum(row.clip_similarity >= clip_threshold for row in report_rows)
    clip_accuracy = clip_correct / total_cases if total_cases else 0.0

    results = [
        {
            "method": "CLIP only",
            "total_cases": total_cases,
            "baseline_correct": clip_correct,
            "rescued_cases": 0,
            "combined_correct": clip_correct,
            "accuracy": clip_accuracy,
            "accuracy_gain_pct_points": 0.0,
        }
    ]

    for method_name in FALLBACK_COLUMNS:
        rescued_cases = 0
        combined_correct = 0

        for row in report_rows:
            clip_pass = row.clip_similarity >= clip_threshold
            rescued = (row.clip_similarity < clip_threshold) and (
                row.fallback_values[method_name] > fallback_threshold
            )
            combined_correct += int(clip_pass or rescued)
            rescued_cases += int(rescued)

        accuracy = combined_correct / total_cases if total_cases else 0.0
        results.append(
            {
                "method": f"CLIP + {method_name}",
                "total_cases": total_cases,
                "baseline_correct": clip_correct,
                "rescued_cases": rescued_cases,
                "combined_correct": combined_correct,
                "accuracy": accuracy,
                "accuracy_gain_pct_points": (accuracy - clip_accuracy) * 100,
            }
        )

    overall = {
        "total_cases": total_cases,
        "clip_correct": clip_correct,
        "clip_accuracy": clip_accuracy,
    }
    return overall, results


def save_excel(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
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


def plot_line_comparison(
    rows: list[dict[str, object]],
    output_path: Path,
    clip_threshold: float,
    fallback_threshold: float,
) -> None:
    labels = [str(row["method"]) for row in rows]
    accuracies = [float(row["accuracy"]) * 100 for row in rows]
    rescued_cases = [int(row["rescued_cases"]) for row in rows]
    x = list(range(len(rows)))

    fig, ax1 = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(
        x,
        accuracies,
        color="#F58518",
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="Accuracy (%)",
    )
    bars = ax2.bar(
        x,
        rescued_cases,
        color="#4C78A8",
        alpha=0.8,
        width=0.55,
        label="Rescued cases",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18)
    ax1.set_ylim(99.0, 100.0)
    ax2.set_ylim(0, max(rescued_cases) + 2)
    ax1.set_ylabel("Accuracy (%)", color="#F58518")
    ax2.set_ylabel("Rescued cases", color="#4C78A8")
    ax1.set_title(
        f"Accuracy and Rescued Cases Comparison\nCLIP >= {clip_threshold:.2f}, fallback > {fallback_threshold:.2f}"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.tick_params(axis="y", colors="#F58518")
    ax2.tick_params(axis="y", colors="#4C78A8")

    for idx, value in enumerate(accuracies):
        if 1 <= idx <= 3:
            ax1.text(
                idx,
                value + 0.015,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#F58518",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 0.35,
                },
            )
            continue
        else:
            y = value + 0.045
            va = "bottom"
            bbox = {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.4,
            }
        ax1.text(
            idx,
            y,
            f"{value:.2f}%",
            ha="center",
            va=va,
            fontsize=9,
            color="#F58518",
            bbox=bbox,
        )
    for bar, value in zip(bars, rescued_cases):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4C78A8",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.excel_output.parent.mkdir(parents=True, exist_ok=True)
    args.image_output.parent.mkdir(parents=True, exist_ok=True)

    report_rows = iter_report_rows(
        args.table_dir,
        include_f2=args.include_f2,
        include_test_report=args.include_test_report,
    )
    if not report_rows:
        raise SystemExit(f"No valid report rows found in {args.table_dir}")

    overall, rows = analyze_methods(
        report_rows,
        clip_threshold=args.clip_threshold,
        fallback_threshold=args.fallback_threshold,
    )

    save_excel(rows, args.excel_output)
    plot_line_comparison(
        rows,
        args.image_output,
        clip_threshold=args.clip_threshold,
        fallback_threshold=args.fallback_threshold,
    )

    print(f"Analyzed cases: {overall['total_cases']}")
    for row in rows:
        print(
            f"{row['method']}: rescued={row['rescued_cases']}, "
            f"accuracy={row['accuracy'] * 100:.2f}%, "
            f"gain={row['accuracy_gain_pct_points']:.2f} pts"
        )
    print(f"Saved Excel: {args.excel_output}")
    print(f"Saved image: {args.image_output}")


if __name__ == "__main__":
    main()
