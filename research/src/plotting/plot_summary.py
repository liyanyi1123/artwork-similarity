#!/usr/bin/env python3
"""Generate summary charts from ex1_2 report Excel files.

This script reads all *_report.xlsx files under the report directory and
produces the same summary charts that live in ex1_2/plot/summary_ai_art.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = BASE_DIR / "table"
DEFAULT_OUT_DIR = BASE_DIR / "plot" / "summary_ai_art"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

HASH_SPEC = [
    ("phash", "pHash"),
    ("ahash", "aHash"),
    ("dhash", "dHash"),
    ("whash", "wHash"),
    ("pdq", "PDQ"),
]

OP_COLORS = {
    "B":  (0.298, 0.447, 0.690),  "C":  (0.333, 0.659, 0.408),
    "F1": (0.769, 0.306, 0.322),  "F2": (0.506, 0.447, 0.698),
    "K":  (0.800, 0.725, 0.455),  "N":  (0.392, 0.710, 0.804),
    "R":  (0.549, 0.549, 0.549),  "S1": (0.910, 0.655, 0.208),
    "S2": (0.435, 0.749, 0.353),  "Sh": (0.847, 0.533, 0.424),
    "T":  (0.482, 0.690, 0.835),  "V":  (0.710, 0.482, 0.612),
}

LEVEL_ORDER = {"low": 0, "mid": 1, "high": 2}


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_report_data(xlsx_path: Path) -> list[dict[str, object]]:
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    data = []
    for row in rows:
        op_name = row[0]
        if op_name is None:
            continue
        vit = to_float(row[3])
        clip = to_float(row[4])
        phash = to_float(row[5])
        ahash = to_float(row[6])
        dhash = to_float(row[7])
        whash = to_float(row[8])
        pdq = to_float(row[9])

        parts = str(op_name).split("_")
        if parts[0] in ("F1", "F2"):
            label = parts[0]
            level = "mid"
        else:
            label = f"{parts[0]}_{parts[1]}"
            level = parts[1]

        data.append({
            "label": label,
            "op_sym": parts[0],
            "level": level,
            "vit": vit,
            "clip": clip,
            "phash": phash,
            "ahash": ahash,
            "dhash": dhash,
            "whash": whash,
            "pdq": pdq,
        })

    data.sort(key=lambda item: (item["op_sym"], LEVEL_ORDER.get(item["level"], 1)))
    return data


def load_all_reports(report_dir: Path) -> list[tuple[str, list[dict[str, object]]]]:
    files = sorted(report_dir.glob("*_report.xlsx"))
    reports = []
    for xlsx in files:
        data = parse_report_data(xlsx)
        if data:
            reports.append((xlsx.stem.replace("_report", ""), data))
    return reports


def save_vit_summary(reports, out_dir: Path):
    labels = [item["label"] for item in reports[0][1]]
    x = np.arange(len(labels))
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, (name, data) in enumerate(reports):
        vals = [d["vit"] if isinstance(d["vit"], (int, float)) else np.nan for d in data]
        ax.plot(x, vals, marker="o", linewidth=1.8, color=cmap(idx % 10), alpha=0.9, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Transformation method")
    ax.set_ylabel("ViT similarity (0–1)")
    ax.set_title("ViT similarity across transformation methods")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "vit_summary.png")
    plt.close(fig)


def save_clip_summary(reports, out_dir: Path):
    labels = [item["label"] for item in reports[0][1]]
    x = np.arange(len(labels))
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, (name, data) in enumerate(reports):
        vals = [d["clip"] if isinstance(d["clip"], (int, float)) else np.nan for d in data]
        ax.plot(x, vals, marker="s", linewidth=1.8, color=cmap(idx % 10), alpha=0.9, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Transformation method")
    ax.set_ylabel("CLIP similarity (0–1)")
    ax.set_title("CLIP similarity across transformation methods")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "clip_summary.png")
    plt.close(fig)


def save_hash_summary(reports, out_dir: Path):
    labels = [item["label"] for item in reports[0][1]]
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, (metric_key, metric_label) in enumerate(HASH_SPEC):
        all_values = []
        for _, data in reports:
            all_values.append([d[metric_key] if isinstance(d[metric_key], (int, float)) else np.nan for d in data])
        avg_values = [
            float(np.mean([vals[j] for vals in all_values if not np.isnan(vals[j])]))
            if any(not np.isnan(vals[j]) for vals in all_values) else np.nan
            for j in range(len(labels))
        ]
        ax.plot(x, avg_values, marker="o", linewidth=1.8, color=colors[idx], alpha=0.9, label=metric_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Transformation method")
    ax.set_ylabel("Hash similarity (0–1)")
    ax.set_title("Hash similarity across transformation methods")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hash_summary.png")
    plt.close(fig)


def save_each_hash_summary(reports, out_dir: Path):
    labels = [item["label"] for item in reports[0][1]]
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for idx, (metric_key, metric_label) in enumerate(HASH_SPEC):
        fig, ax = plt.subplots(figsize=(18, 7))
        for report_idx, (_, data) in enumerate(reports):
            vals = [d[metric_key] if isinstance(d[metric_key], (int, float)) else np.nan for d in data]
            ax.plot(x, vals, marker="o", linewidth=1.4, color=plt.cm.tab10(report_idx % 10), alpha=0.75, label=reports[report_idx][0])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Transformation method")
        ax.set_ylabel(f"{metric_label} similarity (0–1)")
        ax.set_title(f"{metric_label} similarity across transformation methods")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_key}_summary.png")
        plt.close(fig)


def build_level_buckets(reports):
    level_order = ["low", "mid", "high"]
    buckets = {level: {"vit": [], "clip": [], "phash": [], "ahash": [], "dhash": [], "whash": [], "pdq": []} for level in level_order}
    for _, data in reports:
        for d in data:
            if d["op_sym"] in ("F1", "F2"):
                continue
            level = d["level"]
            if level not in buckets:
                continue
            for key in buckets[level]:
                value = d[key]
                if isinstance(value, (int, float)):
                    buckets[level][key].append(value)
    return buckets


def save_transformer_level_summary(reports, out_dir: Path):
    level_buckets = build_level_buckets(reports)
    level_order = ["low", "mid", "high"]
    x = np.arange(len(level_order))

    transformer_levels = {
        "ViT": [float(np.mean(level_buckets[level]["vit"])) if level_buckets[level]["vit"] else np.nan for level in level_order],
        "CLIP": [float(np.mean(level_buckets[level]["clip"])) if level_buckets[level]["clip"] else np.nan for level in level_order],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in transformer_levels.items():
        ax.plot(x, values, marker="o", linewidth=1.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(level_order)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Transformer similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "transformer_level_summary.png")
    plt.close(fig)


def save_hash_level_summary(reports, out_dir: Path):
    level_buckets = build_level_buckets(reports)
    level_order = ["low", "mid", "high"]
    x = np.arange(len(level_order))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (metric_key, metric_label) in enumerate(HASH_SPEC):
        values = [float(np.mean(level_buckets[level][metric_key])) if level_buckets[level][metric_key] else np.nan for level in level_order]
        ax.plot(x, values, marker="o", linewidth=1.8, label=metric_label, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][idx])
    ax.set_xticks(x)
    ax.set_xticklabels(level_order)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Hash similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hash_level_summary.png")
    plt.close(fig)


def save_combined_transformer_hash_summary(reports, out_dir: Path):
    level_buckets = build_level_buckets(reports)
    level_order = ["low", "mid", "high"]
    x = np.arange(len(level_order))

    metric_colors = {
        "ViT": "#2A6F97",
        "CLIP": "#5DA5E0",
        "pHash": "#E45756",
        "aHash": "#F5853F",
        "dHash": "#F9A03F",
        "wHash": "#D85040",
        "PDQ": "#A23E48",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    transformer_levels = {
        "ViT": [float(np.mean(level_buckets[level]["vit"])) if level_buckets[level]["vit"] else np.nan for level in level_order],
        "CLIP": [float(np.mean(level_buckets[level]["clip"])) if level_buckets[level]["clip"] else np.nan for level in level_order],
    }
    for name, values in transformer_levels.items():
        ax.plot(x, values, marker="o", linewidth=1.8, color=metric_colors[name], label=name)

    for metric_key, metric_label in HASH_SPEC:
        values = [float(np.mean(level_buckets[level][metric_key])) if level_buckets[level][metric_key] else np.nan for level in level_order]
        ax.plot(x, values, marker="s", linewidth=1.8, color=metric_colors[metric_label], linestyle="--", label=metric_label)

    ax.set_xticks(x)
    ax.set_xticklabels(level_order)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Transformer vs Hash similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_transformer_hash_level_summary.png")
    plt.close(fig)


def save_low_level_transform_summary(reports, out_dir: Path):
    low_values = OrderedDict()
    for _, data in reports:
        for d in data:
            if d["level"] != "low" or d["op_sym"] in ("F1", "F2"):
                continue
            low_values.setdefault(d["label"], {"vit": [], "clip": []})
            if isinstance(d["vit"], (int, float)):
                low_values[d["label"]]["vit"].append(d["vit"])
            if isinstance(d["clip"], (int, float)):
                low_values[d["label"]]["clip"].append(d["clip"])

    labels = list(low_values.keys())
    x = np.arange(len(labels))
    vit_vals = [float(np.mean(low_values[label]["vit"])) if low_values[label]["vit"] else np.nan for label in labels]
    clip_vals = [float(np.mean(low_values[label]["clip"])) if low_values[label]["clip"] else np.nan for label in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, vit_vals, marker="o", linewidth=1.8, color="#4C72B0", alpha=0.9, label="ViT")
    ax.plot(x, clip_vals, marker="s", linewidth=1.8, color="#C44E52", alpha=0.9, label="CLIP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Low-level transformation")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Low-level transformation similarity comparison")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "low_level_transform_summary.png")
    plt.close(fig)


def build_summary_charts(report_dir: Path, out_dir: Path) -> None:
    report_dir = report_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = load_all_reports(report_dir)
    if not reports:
        raise FileNotFoundError(f"No report files found in {report_dir}")

    save_vit_summary(reports, out_dir)
    save_clip_summary(reports, out_dir)
    save_hash_summary(reports, out_dir)
    save_each_hash_summary(reports, out_dir)
    save_transformer_level_summary(reports, out_dir)
    save_hash_level_summary(reports, out_dir)
    save_combined_transformer_hash_summary(reports, out_dir)
    save_low_level_transform_summary(reports, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summary charts from ex1_2 reports.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Folder containing *_report.xlsx files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Folder to save summary charts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_summary_charts(args.report_dir, args.out_dir)
    print(f"✅ Summary charts generated in {args.out_dir}")
