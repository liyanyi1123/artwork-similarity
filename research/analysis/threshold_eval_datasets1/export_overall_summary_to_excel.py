#!/usr/bin/env python3
"""
导出阈值实验结果到 Excel，包括：
1. Overall 摘要
2. 固定 CLIP=0.8 的 ViT 单变量扫描
3. CLIP x ViT 的 126 组组合结果
4. 折线图与热力图
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

RESULTS_DIR = Path(__file__).parent
SUMMARY_PATH = RESULTS_DIR / "experiment_summary_20260707.json"
POSITIVE_PAIRS_PATH = RESULTS_DIR / "positive_pairs_20260707.csv"
NEGATIVE_PAIRS_PATH = RESULTS_DIR / "negative_pairs_20260707.csv"
OUTPUT_PATH = RESULTS_DIR / "overall_summary_excel_chart_20260707.xlsx"
CLIP_THRESHOLD = 0.8
CLIP_THRESHOLDS = np.round(np.arange(0.80, 0.85 + 0.001, 0.01), 2)
VIT_THRESHOLDS = np.round(np.arange(0.30, 0.50 + 0.001, 0.01), 2)
SHEET_OVERVIEW = "00_overview"
SHEET_VIT_SWEEP = "01_vit_sweep"
SHEET_GRID = "02_clip_vit_grid"
SHEET_CURVES = "03_threshold_curves"
SHEET_F1_HEATMAP = "04_f1_heatmap"
SHEET_FP_HEATMAP = "05_fp_heatmap"


def load_overall_summary(summary_path: Path) -> dict:
    summary = pd.read_json(summary_path)
    matched = summary[summary["model"] == "Overall (CLIP + ViT)"]
    if matched.empty:
        raise ValueError(f"未在 {summary_path} 中找到 Overall (CLIP + ViT) 结果")
    return matched.iloc[0].to_dict()


def build_overall_metrics_dataframe(overall_summary: dict) -> pd.DataFrame:
    rows = [
        {"metric": "TP", "value": overall_summary["TP"]},
        {"metric": "TN", "value": overall_summary["TN"]},
        {"metric": "FP", "value": overall_summary["FP"]},
        {"metric": "FN", "value": overall_summary["FN"]},
        {"metric": "accuracy", "value": overall_summary["accuracy"]},
        {"metric": "recall", "value": overall_summary["recall"]},
        {"metric": "precision", "value": overall_summary["precision"]},
        {"metric": "f1", "value": overall_summary["f1"]},
    ]
    return pd.DataFrame(rows)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def build_threshold_sweep_dataframe() -> pd.DataFrame:
    pos_df = pd.read_csv(POSITIVE_PAIRS_PATH)
    neg_df = pd.read_csv(NEGATIVE_PAIRS_PATH)

    clip_sims = pos_df["clip_similarity"].to_numpy(dtype=float)
    vit_sims = neg_df["vit_similarity"].to_numpy(dtype=float)

    clip_true = np.ones(len(clip_sims), dtype=int)
    vit_true = np.zeros(len(vit_sims), dtype=int)
    clip_pred = (clip_sims >= CLIP_THRESHOLD).astype(int)
    clip_metrics = evaluate(clip_true, clip_pred)

    rows = []
    for vit_threshold in VIT_THRESHOLDS:
        vit_pred = (vit_sims >= vit_threshold).astype(int)
        vit_metrics = evaluate(vit_true, vit_pred)

        overall_true = np.concatenate([clip_true, vit_true])
        overall_pred = np.concatenate([clip_pred, vit_pred])
        overall_metrics = evaluate(overall_true, overall_pred)

        rows.append(
            {
                "clip_threshold": CLIP_THRESHOLD,
                "vit_threshold": round(float(vit_threshold), 2),
                "clip_TP": clip_metrics["TP"],
                "clip_FN": clip_metrics["FN"],
                "vit_TN": vit_metrics["TN"],
                "vit_FP": vit_metrics["FP"],
                "overall_TP": overall_metrics["TP"],
                "overall_TN": overall_metrics["TN"],
                "overall_FP": overall_metrics["FP"],
                "overall_FN": overall_metrics["FN"],
                "accuracy": overall_metrics["accuracy"],
                "recall": overall_metrics["recall"],
                "precision": overall_metrics["precision"],
                "f1": overall_metrics["f1"],
            }
        )

    return pd.DataFrame(rows)


def build_clip_vit_grid_dataframe() -> pd.DataFrame:
    pos_df = pd.read_csv(POSITIVE_PAIRS_PATH)
    neg_df = pd.read_csv(NEGATIVE_PAIRS_PATH)

    clip_sims = pos_df["clip_similarity"].to_numpy(dtype=float)
    vit_sims = neg_df["vit_similarity"].to_numpy(dtype=float)

    clip_true = np.ones(len(clip_sims), dtype=int)
    vit_true = np.zeros(len(vit_sims), dtype=int)

    rows = []
    for clip_threshold in CLIP_THRESHOLDS:
        clip_pred = (clip_sims >= clip_threshold).astype(int)
        clip_metrics = evaluate(clip_true, clip_pred)

        for vit_threshold in VIT_THRESHOLDS:
            vit_pred = (vit_sims >= vit_threshold).astype(int)
            vit_metrics = evaluate(vit_true, vit_pred)

            overall_true = np.concatenate([clip_true, vit_true])
            overall_pred = np.concatenate([clip_pred, vit_pred])
            overall_metrics = evaluate(overall_true, overall_pred)

            rows.append(
                {
                    "clip_threshold": round(float(clip_threshold), 2),
                    "vit_threshold": round(float(vit_threshold), 2),
                    "clip_TP": clip_metrics["TP"],
                    "clip_FN": clip_metrics["FN"],
                    "vit_TN": vit_metrics["TN"],
                    "vit_FP": vit_metrics["FP"],
                    "overall_TP": overall_metrics["TP"],
                    "overall_TN": overall_metrics["TN"],
                    "overall_FP": overall_metrics["FP"],
                    "overall_FN": overall_metrics["FN"],
                    "accuracy": overall_metrics["accuracy"],
                    "recall": overall_metrics["recall"],
                    "precision": overall_metrics["precision"],
                    "f1": overall_metrics["f1"],
                }
            )

    return pd.DataFrame(rows)


def build_best_configs_dataframe(clip_vit_grid_df: pd.DataFrame) -> pd.DataFrame:
    best_f1 = clip_vit_grid_df.sort_values(
        by=["f1", "precision", "accuracy", "overall_FP"],
        ascending=[False, False, False, True],
    ).head(5)
    best_f1 = best_f1.copy()
    best_f1.insert(0, "ranking", [f"top_f1_{i}" for i in range(1, len(best_f1) + 1)])

    lowest_fp = clip_vit_grid_df.sort_values(
        by=["overall_FP", "f1", "accuracy"],
        ascending=[True, False, False],
    ).head(5)
    lowest_fp = lowest_fp.copy()
    lowest_fp.insert(0, "ranking", [f"low_fp_{i}" for i in range(1, len(lowest_fp) + 1)])

    return pd.concat([best_f1, lowest_fp], ignore_index=True)


def build_heatmap_matrix(clip_vit_grid_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    matrix = clip_vit_grid_df.pivot(index="clip_threshold", columns="vit_threshold", values=value_column)
    matrix = matrix.reindex(index=CLIP_THRESHOLDS, columns=VIT_THRESHOLDS)
    matrix.index.name = "clip_threshold"
    return matrix.reset_index()


def add_multi_clip_chart(
    workbook,
    chart_sheet,
    data_sheet_name: str,
    metric_name: str,
    value_col: int,
    insert_cell: str,
    y_axis_name: str,
    y_num_format: str,
    y_min: float | None = None,
    y_max: float | None = None,
):
    chart = workbook.add_chart({"type": "line"})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    block_size = len(VIT_THRESHOLDS)

    for idx, clip_threshold in enumerate(CLIP_THRESHOLDS):
        start_row = 1 + idx * block_size
        end_row = start_row + block_size - 1
        color = colors[idx % len(colors)]
        chart.add_series(
            {
                "name": f"CLIP={clip_threshold:.2f}",
                "categories": [data_sheet_name, start_row, 1, end_row, 1],
                "values": [data_sheet_name, start_row, value_col, end_row, value_col],
                "line": {"color": color, "width": 2.0},
                "marker": {
                    "type": "circle",
                    "size": 4,
                    "border": {"color": color},
                    "fill": {"color": color},
                },
            }
        )

    chart.set_title({"name": f"ViT threshold vs {metric_name}"})
    chart.set_x_axis({"name": "ViT threshold"})
    y_axis = {"name": y_axis_name, "num_format": y_num_format}
    if y_min is not None:
        y_axis["min"] = y_min
    if y_max is not None:
        y_axis["max"] = y_max
    chart.set_y_axis(y_axis)
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 720, "height": 360})
    chart_sheet.insert_chart(insert_cell, chart)


def format_dataframe_sheet(
    worksheet,
    df: pd.DataFrame,
    header_fmt,
    int_fmt,
    float_fmt,
    percent_fmt,
    text_fmt,
    freeze_cell: str = "A2",
):
    worksheet.freeze_panes(freeze_cell)
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
    for col_idx, column_name in enumerate(df.columns):
        worksheet.write(0, col_idx, column_name, header_fmt)
        if column_name in {
            "clip_TP",
            "clip_FN",
            "vit_TN",
            "vit_FP",
            "overall_TP",
            "overall_TN",
            "overall_FP",
            "overall_FN",
        }:
            column_fmt = int_fmt
            width = 12
        elif column_name in {"accuracy", "recall", "precision", "f1"}:
            column_fmt = percent_fmt
            width = 11
        elif column_name in {"clip_threshold", "vit_threshold", "value"}:
            column_fmt = float_fmt
            width = 12
        else:
            column_fmt = text_fmt
            width = max(12, len(str(column_name)) + 2)
        worksheet.set_column(col_idx, col_idx, width, column_fmt)


def apply_heatmap_conditional_formatting(worksheet, first_row: int, last_row: int, first_col: int, last_col: int):
    worksheet.conditional_format(
        first_row,
        first_col,
        last_row,
        last_col,
        {
            "type": "3_color_scale",
            "min_color": "#D73027",
            "mid_color": "#FFFFBF",
            "max_color": "#1A9850",
        },
    )


def export_to_excel(
    overall_summary: dict,
    overall_metrics_df: pd.DataFrame,
    threshold_sweep_df: pd.DataFrame,
    clip_vit_grid_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_configs_df = build_best_configs_dataframe(clip_vit_grid_df)
    f1_heatmap_df = build_heatmap_matrix(clip_vit_grid_df, "f1")
    fp_heatmap_df = build_heatmap_matrix(clip_vit_grid_df, "overall_FP")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        curve_ws = workbook.add_worksheet(SHEET_CURVES)
        f1_ws = workbook.add_worksheet(SHEET_F1_HEATMAP)
        fp_ws = workbook.add_worksheet(SHEET_FP_HEATMAP)
        writer.sheets[SHEET_CURVES] = curve_ws
        writer.sheets[SHEET_F1_HEATMAP] = f1_ws
        writer.sheets[SHEET_FP_HEATMAP] = fp_ws

        summary_df = pd.DataFrame(
            [
                {
                    "model": overall_summary["model"],
                    "threshold": overall_summary["threshold"],
                }
            ]
        )
        summary_df.to_excel(writer, sheet_name=SHEET_OVERVIEW, index=False, startrow=1)
        overall_metrics_df.to_excel(writer, sheet_name=SHEET_OVERVIEW, index=False, startrow=5)
        best_configs_df.to_excel(writer, sheet_name=SHEET_OVERVIEW, index=False, startrow=17)
        threshold_sweep_df.to_excel(writer, sheet_name=SHEET_VIT_SWEEP, index=False)
        clip_vit_grid_df.to_excel(writer, sheet_name=SHEET_GRID, index=False)

        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        note_fmt = workbook.add_format({"italic": True, "font_color": "#666666"})
        text_fmt = workbook.add_format({"border": 1})
        int_fmt = workbook.add_format({"border": 1, "num_format": "0"})
        float_fmt = workbook.add_format({"border": 1, "num_format": "0.0000"})
        percent_fmt = workbook.add_format({"border": 1, "num_format": "0.00%"})

        overview_ws = writer.sheets[SHEET_OVERVIEW]
        overview_ws.set_column("A:A", 18)
        overview_ws.set_column("B:B", 14)
        overview_ws.set_column("C:N", 12)
        overview_ws.write("A1", "Threshold Experiment Overview", title_fmt)
        overview_ws.write("A4", "Baseline overall summary", title_fmt)
        overview_ws.write("A17", "Best threshold combinations", title_fmt)
        overview_ws.write("A30", "Sheet guide", title_fmt)
        overview_ws.write("A31", f"{SHEET_VIT_SWEEP}: 固定 CLIP=0.80 的 ViT 单变量扫描（0.30-0.50）", note_fmt)
        overview_ws.write("A32", f"{SHEET_GRID}: 126 组 CLIP x ViT 原始结果（ViT 0.30-0.50）", note_fmt)
        overview_ws.write("A33", f"{SHEET_CURVES}: Accuracy / Precision / F1 / FP 曲线图", note_fmt)
        overview_ws.write("A34", f"{SHEET_F1_HEATMAP}: F1 热力图", note_fmt)
        overview_ws.write("A35", f"{SHEET_FP_HEATMAP}: FP 热力图", note_fmt)

        for col, value in enumerate(summary_df.columns):
            overview_ws.write(1, col, value, header_fmt)
        for col, value in enumerate(summary_df.iloc[0]):
            overview_ws.write(2, col, value, text_fmt)

        metrics_header_row = 5
        for col, value in enumerate(overall_metrics_df.columns):
            overview_ws.write(metrics_header_row, col, value, header_fmt)

        for row_offset, (_, row) in enumerate(overall_metrics_df.iterrows(), start=1):
            overview_ws.write(metrics_header_row + row_offset, 0, row["metric"], text_fmt)
            number_fmt = int_fmt if row["metric"] in {"TP", "TN", "FP", "FN"} else float_fmt
            overview_ws.write(metrics_header_row + row_offset, 1, row["value"], number_fmt)

        best_start_row = 17
        for col, value in enumerate(best_configs_df.columns):
            overview_ws.write(best_start_row, col, value, header_fmt)
        for row_offset, (_, row) in enumerate(best_configs_df.iterrows(), start=1):
            for col_idx, column_name in enumerate(best_configs_df.columns):
                if column_name in {
                    "clip_TP",
                    "clip_FN",
                    "vit_TN",
                    "vit_FP",
                    "overall_TP",
                    "overall_TN",
                    "overall_FP",
                    "overall_FN",
                }:
                    fmt = int_fmt
                elif column_name in {"accuracy", "recall", "precision", "f1"}:
                    fmt = percent_fmt
                elif column_name in {"clip_threshold", "vit_threshold"}:
                    fmt = float_fmt
                else:
                    fmt = text_fmt
                overview_ws.write(best_start_row + row_offset, col_idx, row[column_name], fmt)

        integer_columns = {
            "clip_TP",
            "clip_FN",
            "vit_TN",
            "vit_FP",
            "overall_TP",
            "overall_TN",
            "overall_FP",
            "overall_FN",
        }
        metric_percent_columns = {"accuracy", "recall", "precision", "f1"}

        sweep_ws = writer.sheets[SHEET_VIT_SWEEP]
        format_dataframe_sheet(sweep_ws, threshold_sweep_df, header_fmt, int_fmt, float_fmt, percent_fmt, text_fmt)
        for row_idx, (_, row) in enumerate(threshold_sweep_df.iterrows(), start=1):
            for col_idx, column_name in enumerate(threshold_sweep_df.columns):
                if column_name in integer_columns:
                    fmt = int_fmt
                elif column_name in metric_percent_columns:
                    fmt = percent_fmt
                else:
                    fmt = float_fmt
                sweep_ws.write(row_idx, col_idx, row[column_name], fmt)

        last_row = len(threshold_sweep_df)
        chart = workbook.add_chart({"type": "line"})
        series_specs = [
            ("accuracy", 10, "#1f77b4"),
            ("precision", 12, "#ff7f0e"),
            ("f1", 13, "#2ca02c"),
        ]
        for _, value_col, color in series_specs:
            chart.add_series(
                {
                    "name": [SHEET_VIT_SWEEP, 0, value_col],
                    "categories": [SHEET_VIT_SWEEP, 1, 1, last_row, 1],
                    "values": [SHEET_VIT_SWEEP, 1, value_col, last_row, value_col],
                    "line": {"color": color, "width": 2.0},
                    "marker": {
                        "type": "circle",
                        "size": 5,
                        "border": {"color": color},
                        "fill": {"color": color},
                    },
                }
            )

        chart.set_title({"name": "ViT threshold vs Accuracy / Precision / F1"})
        chart.set_x_axis({"name": "ViT threshold"})
        chart.set_y_axis({"name": "Percentage", "num_format": "0%", "min": 0, "max": 1})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 880, "height": 420})
        sweep_ws.insert_chart("P2", chart)

        grid_ws = writer.sheets[SHEET_GRID]
        format_dataframe_sheet(grid_ws, clip_vit_grid_df, header_fmt, int_fmt, float_fmt, percent_fmt, text_fmt)
        for row_idx, (_, row) in enumerate(clip_vit_grid_df.iterrows(), start=1):
            for col_idx, column_name in enumerate(clip_vit_grid_df.columns):
                if column_name in integer_columns:
                    fmt = int_fmt
                elif column_name in metric_percent_columns:
                    fmt = percent_fmt
                else:
                    fmt = float_fmt
                grid_ws.write(row_idx, col_idx, row[column_name], fmt)

        chart_ws = writer.sheets[SHEET_CURVES]
        chart_ws.set_column("A:A", 60)
        chart_ws.write("A1", "Threshold Curves", title_fmt)
        chart_ws.write("A2", "每条线对应一个 CLIP threshold；横轴是 ViT threshold（0.30-0.50）。", note_fmt)

        add_multi_clip_chart(workbook, chart_ws, SHEET_GRID, "Accuracy", 10, "A4", "Percentage", "0%", 0, 1)
        add_multi_clip_chart(workbook, chart_ws, SHEET_GRID, "Precision", 12, "J4", "Percentage", "0%", 0, 1)
        add_multi_clip_chart(workbook, chart_ws, SHEET_GRID, "F1", 13, "A24", "Percentage", "0%", 0, 1)
        add_multi_clip_chart(workbook, chart_ws, SHEET_GRID, "Overall FP", 8, "J24", "Count", "0", 0, None)

        f1_ws.write("A1", "F1 Heatmap", title_fmt)
        f1_ws.write("A2", "行是 CLIP threshold，列是 ViT threshold，颜色越绿表示 F1 越高。", note_fmt)
        f1_heatmap_df.to_excel(writer, sheet_name=SHEET_F1_HEATMAP, index=False, startrow=3)
        for col_idx, column_name in enumerate(f1_heatmap_df.columns):
            f1_ws.write(3, col_idx, column_name, header_fmt)
            f1_ws.set_column(col_idx, col_idx, 12, float_fmt if col_idx == 0 else percent_fmt)
        for row_idx in range(1, len(f1_heatmap_df) + 1):
            f1_ws.write(row_idx + 3, 0, f1_heatmap_df.iloc[row_idx - 1, 0], float_fmt)
            for col_idx in range(1, len(f1_heatmap_df.columns)):
                f1_ws.write(row_idx + 3, col_idx, f1_heatmap_df.iloc[row_idx - 1, col_idx], percent_fmt)
        f1_ws.freeze_panes("B5")
        f1_ws.autofilter(3, 0, len(f1_heatmap_df) + 3, len(f1_heatmap_df.columns) - 1)
        apply_heatmap_conditional_formatting(f1_ws, 4, len(f1_heatmap_df) + 3, 1, len(f1_heatmap_df.columns) - 1)

        fp_ws.write("A1", "Overall FP Heatmap", title_fmt)
        fp_ws.write("A2", "行是 CLIP threshold，列是 ViT threshold，颜色越绿表示 FP 越少。", note_fmt)
        fp_heatmap_df.to_excel(writer, sheet_name=SHEET_FP_HEATMAP, index=False, startrow=3)
        for col_idx, column_name in enumerate(fp_heatmap_df.columns):
            fp_ws.write(3, col_idx, column_name, header_fmt)
            fp_ws.set_column(col_idx, col_idx, 12, int_fmt if col_idx > 0 else float_fmt)
        for row_idx in range(1, len(fp_heatmap_df) + 1):
            fp_ws.write(row_idx + 3, 0, fp_heatmap_df.iloc[row_idx - 1, 0], float_fmt)
            for col_idx in range(1, len(fp_heatmap_df.columns)):
                fp_ws.write(row_idx + 3, col_idx, fp_heatmap_df.iloc[row_idx - 1, col_idx], int_fmt)
        fp_ws.freeze_panes("B5")
        fp_ws.autofilter(3, 0, len(fp_heatmap_df) + 3, len(fp_heatmap_df.columns) - 1)
        fp_ws.conditional_format(
            4,
            1,
            len(fp_heatmap_df) + 3,
            len(fp_heatmap_df.columns) - 1,
            {
                "type": "3_color_scale",
                "min_color": "#1A9850",
                "mid_color": "#FFFFBF",
                "max_color": "#D73027",
            },
        )

        top_f1_row = clip_vit_grid_df.sort_values(by=["f1", "precision"], ascending=[False, False]).iloc[0]
        best_fp_row = clip_vit_grid_df.sort_values(by=["overall_FP", "f1"], ascending=[True, False]).iloc[0]
        overview_ws.write("F1", "Best F1", title_fmt)
        overview_ws.write("F2", f"CLIP={top_f1_row['clip_threshold']:.2f}, ViT={top_f1_row['vit_threshold']:.2f}", text_fmt)
        overview_ws.write("F3", top_f1_row["f1"], percent_fmt)
        overview_ws.write("H1", "Lowest FP", title_fmt)
        overview_ws.write("H2", f"CLIP={best_fp_row['clip_threshold']:.2f}, ViT={best_fp_row['vit_threshold']:.2f}", text_fmt)
        overview_ws.write("H3", int(best_fp_row["overall_FP"]), int_fmt)

    return output_path


def main():
    overall_summary = load_overall_summary(SUMMARY_PATH)
    overall_metrics_df = build_overall_metrics_dataframe(overall_summary)
    threshold_sweep_df = build_threshold_sweep_dataframe()
    clip_vit_grid_df = build_clip_vit_grid_dataframe()
    saved_path = export_to_excel(
        overall_summary,
        overall_metrics_df,
        threshold_sweep_df,
        clip_vit_grid_df,
        OUTPUT_PATH,
    )
    print(f"Excel 已保存到: {saved_path}")


if __name__ == "__main__":
    main()
