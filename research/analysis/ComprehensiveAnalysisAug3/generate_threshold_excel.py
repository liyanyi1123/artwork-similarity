#!/usr/bin/env python3
"""
Generate a 4-sheet Excel file: threshold evaluation summary for datasets1-4.
CLIP thresholds: 0.80–0.85 (step 0.01)
ViT thresholds:  0.40–0.50 (step 0.01)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "threshold_evaluation_4datasets.xlsx"

# datasets1 raw CSVs
DS1_POS = BASE / "threshold_eval_datasets1" / "positive_pairs_20260707.csv"
DS1_NEG = BASE / "threshold_eval_datasets1" / "negative_pairs_20260707.csv"

# datasets2/3/4 threshold grids
DS2_GRID = BASE / "threshold_eval_datasets2" / "threshold_grid.csv"
DS3_GRID = BASE / "threshold_eval_datasets3" / "threshold_grid.csv"
DS4_GRID = BASE / "threshold_eval_datasets4" / "threshold_grid.csv"

# ── Threshold ranges ───────────────────────────────────────────────────
CLIP_THRESHOLDS = [round(x, 2) for x in np.arange(0.80, 0.86, 0.01)]  # 0.80–0.85
VIT_THRESHOLDS  = [round(x, 2) for x in np.arange(0.40, 0.51, 0.01)]  # 0.40–0.50


def compute_from_raw(pos_path: Path, neg_path: Path):
    """Compute TP/FN/TN/FP for all threshold combos from raw pair CSVs."""
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)
    clip_sims = pos["clip_similarity"].values
    vit_sims  = neg["vit_similarity"].values

    rows = []
    for ct in CLIP_THRESHOLDS:
        tp = int(np.sum(clip_sims >= ct))
        fn = int(np.sum(clip_sims < ct))
        for vt in VIT_THRESHOLDS:
            tn = int(np.sum(vit_sims < vt))
            fp = int(np.sum(vit_sims >= vt))
            total = tp + fn + tn + fp
            accuracy = (tp + tn) / total if total else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append({
                "CLIP Threshold": ct,
                "ViT Threshold": vt,
                "TP (CLIP)": tp,
                "FN (CLIP)": fn,
                "TN (ViT)": tn,
                "FP (ViT)": fp,
                "Overall TP": tp,
                "Overall TN": tn,
                "Overall FP": fp,
                "Overall FN": fn,
                "Accuracy": round(accuracy, 6),
                "Recall": round(recall, 6),
                "Precision": round(precision, 6),
                "F1": round(f1, 6),
            })
    return rows


def extract_from_grid(grid_path: Path):
    """Extract relevant threshold rows from an existing threshold_grid.csv."""
    df = pd.read_csv(grid_path)

    # Normalise column names: some files use TP/TN/FP/FN, others use overall_TP etc.
    rows = []
    for _, row in df.iterrows():
        ct = float(row["clip_threshold"])
        vt = float(row["vit_threshold"])
        if ct < 0.80 or ct > 0.85 or vt < 0.40 or vt > 0.50:
            continue

        # Detect column naming convention
        if "overall_TP" in row.index:
            tp, tn, fp, fn = (
                int(row["overall_TP"]),
                int(row["overall_TN"]),
                int(row["overall_FP"]),
                int(row["overall_FN"]),
            )
            clip_tp = int(row["clip_TP"])
            clip_fn = int(row["clip_FN"])
            vit_tn  = int(row["vit_TN"])
            vit_fp  = int(row["vit_FP"])
        else:
            tp = int(row["TP"])
            tn = int(row["TN"])
            fp = int(row["FP"])
            fn = int(row["FN"])
            clip_tp, clip_fn = tp, fn
            vit_tn, vit_fp = tn, fp

        rows.append({
            "CLIP Threshold": ct,
            "ViT Threshold": vt,
            "TP (CLIP)": clip_tp,
            "FN (CLIP)": clip_fn,
            "TN (ViT)": vit_tn,
            "FP (ViT)": vit_fp,
            "Overall TP": tp,
            "Overall TN": tn,
            "Overall FP": fp,
            "Overall FN": fn,
            "Accuracy": round(float(row["accuracy"]), 6),
            "Recall": round(float(row["recall"]), 6),
            "Precision": round(float(row["precision"]), 6),
            "F1": round(float(row["f1"]), 6),
        })
    return rows


def save_excel(datasets: dict[str, list[dict]], output_path: Path):
    """Write a multi-sheet Excel file with formatting."""
    wb = xlsxwriter.Workbook(str(output_path))

    header_fmt = wb.add_format({
        "bold": True, "bg_color": "#4472C4", "font_color": "white",
        "border": 1, "text_wrap": True, "align": "center", "valign": "vcenter",
    })
    num_fmt_4d = wb.add_format({"num_format": "0.0000", "border": 1, "align": "center"})
    num_fmt_int = wb.add_format({"num_format": "#,##0", "border": 1, "align": "center"})
    pct_fmt = wb.add_format({"num_format": "0.00%", "border": 1, "align": "center"})

    # Highlight best F1 per sheet
    best_fmt = wb.add_format({
        "num_format": "0.0000", "border": 2, "align": "center",
        "bg_color": "#FFC000", "bold": True,
    })
    best_int_fmt = wb.add_format({
        "num_format": "#,##0", "border": 2, "align": "center",
        "bg_color": "#FFC000", "bold": True,
    })
    best_pct_fmt = wb.add_format({
        "num_format": "0.00%", "border": 2, "align": "center",
        "bg_color": "#FFC000", "bold": True,
    })

    headers = [
        "CLIP Threshold", "ViT Threshold",
        "TP (CLIP)", "FN (CLIP)", "TN (ViT)", "FP (ViT)",
        "Overall TP", "Overall TN", "Overall FP", "Overall FN",
        "Accuracy", "Recall", "Precision", "F1",
    ]

    sheet_names = {
        "datasets1": "Datasets1 (Archive 100)",
        "datasets2": "Datasets2 (OpenImages 100)",
        "datasets3": "Datasets3 (Combined 200)",
        "datasets4": "Datasets4 (OpenImages 1000)",
    }

    for key, name in sheet_names.items():
        ws = wb.add_worksheet(name)
        rows = datasets[key]

        # Column widths
        ws.set_column(0, 1, 13)
        ws.set_column(2, 9, 13)
        ws.set_column(10, 13, 12)

        # Write headers
        for ci, h in enumerate(headers):
            ws.write(0, ci, h, header_fmt)
        ws.set_row(0, 30)

        # Find best F1 row index
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["F1"])

        # Write data
        for ri, row in enumerate(rows):
            is_best = (ri == best_idx)
            int_f = best_int_fmt if is_best else num_fmt_int
            flt_f = best_fmt if is_best else num_fmt_4d
            pct_f = best_pct_fmt if is_best else pct_fmt

            ws.write_number(ri + 1, 0, row["CLIP Threshold"], flt_f)
            ws.write_number(ri + 1, 1, row["ViT Threshold"], flt_f)
            ws.write_number(ri + 1, 2, row["TP (CLIP)"], int_f)
            ws.write_number(ri + 1, 3, row["FN (CLIP)"], int_f)
            ws.write_number(ri + 1, 4, row["TN (ViT)"], int_f)
            ws.write_number(ri + 1, 5, row["FP (ViT)"], int_f)
            ws.write_number(ri + 1, 6, row["Overall TP"], int_f)
            ws.write_number(ri + 1, 7, row["Overall TN"], int_f)
            ws.write_number(ri + 1, 8, row["Overall FP"], int_f)
            ws.write_number(ri + 1, 9, row["Overall FN"], int_f)
            ws.write_number(ri + 1, 10, row["Accuracy"], pct_f)
            ws.write_number(ri + 1, 11, row["Recall"], pct_f)
            ws.write_number(ri + 1, 12, row["Precision"], pct_f)
            ws.write_number(ri + 1, 13, row["F1"], flt_f)

        # Freeze top row
        ws.freeze_panes(1, 0)
        # Auto-filter
        ws.autofilter(0, 0, len(rows), len(headers) - 1)

        # Add a summary section below the table
        gap = len(rows) + 3
        summary_hdr_fmt = wb.add_format({
            "bold": True, "bg_color": "#DCE6F1", "border": 1,
            "align": "center", "valign": "vcenter",
        })
        ws.merge_range(gap, 0, gap, 3, "Dataset Summary", summary_hdr_fmt)
        ws.write(gap + 1, 0, "Total Positive Pairs", summary_hdr_fmt)
        total_pos = rows[0]["TP (CLIP)"] + rows[0]["FN (CLIP)"]
        ws.write_number(gap + 1, 1, total_pos, num_fmt_int)
        ws.write(gap + 2, 0, "Total Negative Pairs", summary_hdr_fmt)
        total_neg = rows[0]["TN (ViT)"] + rows[0]["FP (ViT)"]
        ws.write_number(gap + 2, 1, total_neg, num_fmt_int)
        ws.write(gap + 3, 0, "Total Pairs", summary_hdr_fmt)
        ws.write_number(gap + 3, 1, total_pos + total_neg, num_fmt_int)
        ws.write(gap + 4, 0, "Best F1", summary_hdr_fmt)
        ws.write_number(gap + 4, 1, rows[best_idx]["F1"], best_fmt)
        ws.write(gap + 5, 0, "Best CLIP / ViT", summary_hdr_fmt)
        ws.write(gap + 5, 1, f"{rows[best_idx]['CLIP Threshold']:.2f} / {rows[best_idx]['ViT Threshold']:.2f}")

    wb.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("Computing datasets1 from raw pair CSVs ...")
    ds1 = compute_from_raw(DS1_POS, DS1_NEG)
    print(f"  → {len(ds1)} rows (CLIP={len(CLIP_THRESHOLDS)} × ViT={len(VIT_THRESHOLDS)})")

    print("Extracting datasets2 from threshold_grid.csv ...")
    ds2 = extract_from_grid(DS2_GRID)
    print(f"  → {len(ds2)} rows")

    print("Extracting datasets3 from threshold_grid.csv ...")
    ds3 = extract_from_grid(DS3_GRID)
    print(f"  → {len(ds3)} rows")

    print("Extracting datasets4 from threshold_grid.csv ...")
    ds4 = extract_from_grid(DS4_GRID)
    print(f"  → {len(ds4)} rows")

    datasets = {
        "datasets1": ds1,
        "datasets2": ds2,
        "datasets3": ds3,
        "datasets4": ds4,
    }

    save_excel(datasets, OUTPUT)

    # Print quick summary
    for key, rows in datasets.items():
        best = max(rows, key=lambda r: r["F1"])
        print(f"  {key}: best F1={best['F1']:.4f} @ CLIP={best['CLIP Threshold']:.2f}, ViT={best['ViT Threshold']:.2f}")


if __name__ == "__main__":
    main()
