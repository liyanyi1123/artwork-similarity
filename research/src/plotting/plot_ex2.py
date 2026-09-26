"""Plot charts from all *_report.xlsx files in the table directory."""
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(__file__).parent
REPORT_DIR = BASE_DIR / "table"
OUT_ROOT = BASE_DIR / "plot"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

OP_COLORS = {
    "B":  (0.298, 0.447, 0.690),  "C":  (0.333, 0.659, 0.408),
    "F1": (0.769, 0.306, 0.322),  "F2": (0.506, 0.447, 0.698),
    "K":  (0.800, 0.725, 0.455),  "N":  (0.392, 0.710, 0.804),
    "R":  (0.549, 0.549, 0.549),  "S1": (0.910, 0.655, 0.208),
    "S2": (0.435, 0.749, 0.353),  "Sh": (0.847, 0.533, 0.424),
    "T":  (0.482, 0.690, 0.835),  "V":  (0.710, 0.482, 0.612),
}


def to_f(v):
    try:
        return float(v)
    except Exception:
        return None


def parse_report_data(xlsx_path):
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    data = []
    for row in rows:
        op_name = row[0]
        vit = to_f(row[3])
        clip = to_f(row[4])
        phash = to_f(row[5])
        ahash = to_f(row[6])
        dhash = to_f(row[7])
        whash = to_f(row[8])
        pdq = to_f(row[9])
        if op_name is None:
            continue

        parts = op_name.split("_")
        if parts[0] in ("F1", "F2"):
            label = parts[0]
        else:
            label = f"{parts[0]}_{parts[1]}"

        data.append({
            "label": label,
            "op_sym": parts[0],
            "level": parts[1] if parts[0] not in ("F1", "F2") else "mid",
            "vit": vit,
            "clip": clip,
            "phash": phash,
            "ahash": ahash,
            "dhash": dhash,
            "whash": whash,
            "pdq": pdq,
        })

    LEVEL_ORDER = {"low": 0, "mid": 1, "high": 2}
    data.sort(key=lambda d: (d["op_sym"], LEVEL_ORDER.get(d["level"], 1)))
    return data


def plot_bars(ax, data, labels, title, ylabel):
    x = np.arange(len(labels))
    width = 0.35
    colors_vit = [OP_COLORS.get(d["op_sym"], "#AAAAAA") for d in data]
    colors_clip = [(c[0] * 0.6, c[1] * 0.6, c[2] * 0.6) for c in colors_vit]

    y1 = [d["vit"] if isinstance(d["vit"], (int, float)) else 0 for d in data]
    y2 = [d["clip"] if isinstance(d["clip"], (int, float)) else 0 for d in data]

    ax.bar(x - width / 2, y1, width, color=colors_vit, edgecolor="white", linewidth=0.3, label="ViT")
    ax.bar(x + width / 2, y2, width, color=colors_clip, edgecolor="white", linewidth=0.3, label="CLIP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_lines(ax, data, labels, title, ylabel):
    x = np.arange(len(labels))
    y1 = [d["vit"] if isinstance(d["vit"], (int, float)) else 0 for d in data]
    y2 = [d["clip"] if isinstance(d["clip"], (int, float)) else 0 for d in data]

    ax.plot(x, y1, "o-", color="#4C72B0", markersize=4, linewidth=1.2, label="ViT")
    ax.plot(x, y2, "s-", color="#C44E52", markersize=4, linewidth=1.2, label="CLIP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    prev_sym = None
    start = -0.5
    for i, d in enumerate(data):
        if d["op_sym"] != prev_sym:
            if prev_sym is not None:
                ax.axvspan(start, i - 0.5, alpha=0.05, color=OP_COLORS.get(prev_sym, "gray"))
            start = i - 0.5
            prev_sym = d["op_sym"]
    ax.axvspan(start, len(labels) - 0.5, alpha=0.05, color=OP_COLORS.get(prev_sym, "gray"))


def hash_bar_plot(ax, data, labels, title):
    x = np.arange(len(labels))
    hash_names = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
    hash_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    n_hashes = len(hash_names)
    width = 0.8 / n_hashes

    hash_data = {
        "pHash": [d["phash"] if isinstance(d["phash"], (int, float)) else 0 for d in data],
        "aHash": [d["ahash"] if isinstance(d["ahash"], (int, float)) else 0 for d in data],
        "dHash": [d["dhash"] if isinstance(d["dhash"], (int, float)) else 0 for d in data],
        "wHash": [d["whash"] if isinstance(d["whash"], (int, float)) else 0 for d in data],
        "PDQ": [d["pdq"] if isinstance(d["pdq"], (int, float)) else 0 for d in data],
    }

    for j, (hname, hcolor) in enumerate(zip(hash_names, hash_colors)):
        offset = (j - n_hashes / 2 + 0.5) * width
        vals = [hash_data[hname][i] for i in range(len(labels))]
        ax.bar(x + offset, vals, width, color=hcolor, edgecolor="white", linewidth=0.2, label=hname, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("Similarity (0–1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def hash_line_plot(ax, data, labels, title):
    x = np.arange(len(labels))
    hash_names = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
    hash_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    markers = ["o", "s", "D", "^", "v"]

    hash_data = {
        "pHash": [d["phash"] if isinstance(d["phash"], (int, float)) else 0 for d in data],
        "aHash": [d["ahash"] if isinstance(d["ahash"], (int, float)) else 0 for d in data],
        "dHash": [d["dhash"] if isinstance(d["dhash"], (int, float)) else 0 for d in data],
        "wHash": [d["whash"] if isinstance(d["whash"], (int, float)) else 0 for d in data],
        "PDQ": [d["pdq"] if isinstance(d["pdq"], (int, float)) else 0 for d in data],
    }

    for hname, hcolor, mk in zip(hash_names, hash_colors, markers):
        vals = [hash_data[hname][i] for i in range(len(labels))]
        ax.plot(x, vals, marker=mk, color=hcolor, markersize=3, linewidth=1.0, label=hname)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("Similarity (0–1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    ax.grid(alpha=0.3)

    prev_sym = None
    start = -0.5
    for i, d in enumerate(data):
        if d["op_sym"] != prev_sym:
            if prev_sym is not None:
                ax.axvspan(start, i - 0.5, alpha=0.05, color=OP_COLORS.get(prev_sym, "gray"))
            start = i - 0.5
            prev_sym = d["op_sym"]
    ax.axvspan(start, len(labels) - 0.5, alpha=0.05, color=OP_COLORS.get(prev_sym, "gray"))


def save_plots_for_report(xlsx_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = parse_report_data(xlsx_path)
    if not data:
        print(f"⚠️ No data found in {xlsx_path.name}")
        return

    labels = [d["label"] for d in data]

    fig, ax = plt.subplots(figsize=(16, 6))
    plot_bars(ax, data, labels, "ViT & CLIP Cosine Similarity — grouped bars", "Cosine Similarity (0–1)")
    fig.tight_layout()
    fig.savefig(out_dir / "vit_clip_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    plot_lines(ax, data, labels, "ViT & CLIP Cosine Similarity — line chart", "Cosine Similarity (0–1)")
    fig.tight_layout()
    fig.savefig(out_dir / "vit_clip_line.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    hash_bar_plot(ax, data, labels, "Perceptual Hash Similarity — grouped bars")
    fig.tight_layout()
    fig.savefig(out_dir / "hash_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    hash_line_plot(ax, data, labels, "Perceptual Hash Similarity — line chart")
    fig.tight_layout()
    fig.savefig(out_dir / "hash_line.png")
    plt.close(fig)

    op_summary = OrderedDict()
    for d in data:
        sym = d["op_sym"]
        if sym not in op_summary:
            op_summary[sym] = {"vit": [], "clip": [], "phash": [], "ahash": [], "dhash": [], "whash": [], "pdq": []}
        for k in ["vit", "clip", "phash", "ahash", "dhash", "whash", "pdq"]:
            v = d[k]
            if isinstance(v, (int, float)):
                op_summary[sym][k].append(v)

    means = {k: {m: np.mean(op_summary[k][m]) for m in op_summary[k]} for k in op_summary}
    op_symbols = list(op_summary.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(op_symbols))
    w = 0.3
    ax1.bar(x - w / 2, [means[s]["vit"] for s in op_symbols], w, color="#4C72B0", label="ViT", edgecolor="white")
    ax1.bar(x + w / 2, [means[s]["clip"] for s in op_symbols], w, color="#C44E52", label="CLIP", edgecolor="white")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Mean ViT & CLIP Cosine Similarity per Operation")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    hash_names = ["phash", "ahash", "dhash", "whash", "pdq"]
    hash_labels = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
    hash_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    n_h = len(hash_names)
    w_h = 0.8 / n_h
    for j, (hname, hcolor, hlabel) in enumerate(zip(hash_names, hash_colors, hash_labels)):
        offset = (j - n_h / 2 + 0.5) * w_h
        ax2.bar(x + offset, [means[s][hname] for s in op_symbols], w_h, color=hcolor, label=hlabel, edgecolor="white", linewidth=0.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(op_symbols)
    ax2.set_ylabel("Similarity (0–1)")
    ax2.set_title("Mean Hash Similarity per Operation")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, ncol=5, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_per_operation.png")
    plt.close(fig)

    sorted_by_vit = sorted(data, key=lambda d: d["vit"] if isinstance(d["vit"], (int, float)) else 0)
    fig, ax = plt.subplots(figsize=(18, 5))
    x_all = np.arange(len(sorted_by_vit))
    vit_sorted = [d["vit"] if isinstance(d["vit"], (int, float)) else 0 for d in sorted_by_vit]
    clip_sorted = [d["clip"] if isinstance(d["clip"], (int, float)) else 0 for d in sorted_by_vit]
    ax.plot(x_all, vit_sorted, linewidth=0.8, color=(0.298, 0.447, 0.690), label="ViT", alpha=0.8)
    ax.plot(x_all, clip_sorted, linewidth=0.8, color=(0.769, 0.306, 0.322), label="CLIP", alpha=0.8)
    prev_sym = None
    start = 0
    for i, d in enumerate(sorted_by_vit):
        if d["op_sym"] != prev_sym:
            if prev_sym is not None:
                ax.axvspan(start - 0.5, i - 0.5, alpha=0.08, color=OP_COLORS.get(prev_sym, (0.5, 0.5, 0.5)))
            start = i
            prev_sym = d["op_sym"]
    ax.axvspan(start - 0.5, len(sorted_by_vit) - 0.5, alpha=0.08, color=OP_COLORS.get(prev_sym, (0.5, 0.5, 0.5)))
    ax.set_title("ViT & CLIP Cosine Similarity — all 32 images (sorted by ViT)")
    ax.set_ylabel("Cosine Similarity (0–1)")
    ax.set_xlabel("Images (sorted by ViT similarity, shaded by operation)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "vit_clip_sorted_line.png")
    plt.close(fig)

    sorted_by_phash = sorted(data, key=lambda d: d["phash"] if isinstance(d["phash"], (int, float)) else 0)
    fig, ax = plt.subplots(figsize=(18, 5))
    x_all = np.arange(len(sorted_by_phash))
    hash_names = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
    hash_labels = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
    hash_colors = [(0.298, 0.447, 0.690), (0.333, 0.659, 0.408), (0.769, 0.306, 0.322), (0.506, 0.447, 0.698), (0.800, 0.725, 0.455)]
    data_hash_keys = ["phash", "ahash", "dhash", "whash", "pdq"]
    for hkey, hcolor, hlabel in zip(data_hash_keys, hash_colors, hash_labels):
        vals = [d[hkey] if isinstance(d[hkey], (int, float)) else 0 for d in sorted_by_phash]
        ax.plot(x_all, vals, linewidth=0.6, color=hcolor, label=hlabel, alpha=0.7)
    prev_sym = None
    start = 0
    for i, d in enumerate(sorted_by_phash):
        if d["op_sym"] != prev_sym:
            if prev_sym is not None:
                ax.axvspan(start - 0.5, i - 0.5, alpha=0.08, color=OP_COLORS.get(prev_sym, (0.5, 0.5, 0.5)))
            start = i
            prev_sym = d["op_sym"]
    ax.axvspan(start - 0.5, len(sorted_by_phash) - 0.5, alpha=0.08, color=OP_COLORS.get(prev_sym, (0.5, 0.5, 0.5)))
    ax.set_title("Perceptual Hash Similarity — all 32 images (sorted by pHash)")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_xlabel("Images (sorted by pHash similarity, shaded by operation)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=5, loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "hash_sorted_line.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for sym, color in OP_COLORS.items():
        kd = [d for d in data if d["op_sym"] == sym]
        xv = [d["vit"] if isinstance(d["vit"], (int, float)) else 0 for d in kd]
        yv = [d["clip"] if isinstance(d["clip"], (int, float)) else 0 for d in kd]
        ax.scatter(xv, yv, c=[color], alpha=0.7, s=20, label=sym, edgecolors="white", linewidths=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("ViT Cosine Similarity")
    ax.set_ylabel("CLIP Cosine Similarity")
    ax.set_title("ViT vs CLIP Similarity by Operation Type")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vit_vs_clip_scatter.png")
    plt.close(fig)

    level_data = {"low": [], "mid": [], "high": []}
    for d in data:
        lv = d["level"]
        if lv in level_data and d["op_sym"] not in ("F1", "F2"):
            level_data[lv].append(d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for i, lv in enumerate(["low", "mid", "high"]):
        items = level_data[lv]
        vit_m = np.mean([x["vit"] for x in items if isinstance(x["vit"], (int, float))])
        clip_m = np.mean([x["clip"] for x in items if isinstance(x["clip"], (int, float))])
        ax1.bar([i - 0.15, i + 0.15], [vit_m, clip_m], 0.25, color=[(0.298, 0.447, 0.690), (0.769, 0.306, 0.322)], edgecolor="white")
        if i == 0:
            ax1.bar([-0.15], [0], 0.25, color=(0.298, 0.447, 0.690), label="ViT")
            ax1.bar([0.15], [0], 0.25, color=(0.769, 0.306, 0.322), label="CLIP")
        for hkey, hcolor, hlabel in zip(["phash", "ahash", "dhash", "whash", "pdq"], ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"], ["pHash", "aHash", "dHash", "wHash", "PDQ"]):
            vals = [x[hkey] for x in items if isinstance(x[hkey], (int, float))]
            hm = np.mean(vals) if vals else 0
            offset = (["phash", "ahash", "dhash", "whash", "pdq"].index(hkey) - 2) * 0.1
            ax2.bar([i + offset], [hm], 0.17, color=hcolor, edgecolor="white", linewidth=0.2, label=hlabel if i == 0 else "")

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["low", "mid", "high"])
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("ViT & CLIP by Parameter Level")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["low", "mid", "high"])
    ax2.set_ylabel("Similarity")
    ax2.set_title("Hash Similarity by Parameter Level")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, ncol=5, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "level_comparison.png")
    plt.close(fig)

    print(f"✅ 9 charts saved to {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"   {f.name}")


def save_summary_chart(report_files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_reports = []

    for xlsx_path in report_files:
        data = parse_report_data(xlsx_path)
        if not data:
            continue
        parsed_reports.append((xlsx_path.stem.replace("_report", ""), data))

    if not parsed_reports:
        print(f"⚠️ No valid summary data found in {REPORT_DIR}")
        return

    parsed_reports.sort(key=lambda item: item[0])
    labels = [d["label"] for d in parsed_reports[0][1]]
    x = np.arange(len(labels))

    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, (name, data) in enumerate(parsed_reports):
        vit_vals = [d["vit"] if isinstance(d["vit"], (int, float)) else np.nan for d in data]
        color = cmap(idx % 10)
        ax.plot(x, vit_vals, marker="o", linewidth=1.4, color=color, alpha=0.9, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Transformation method")
    ax.set_ylabel("ViT similarity (0–1)")
    ax.set_title("ViT similarity across transformation methods")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "vit_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, (name, data) in enumerate(parsed_reports):
        clip_vals = [d["clip"] if isinstance(d["clip"], (int, float)) else np.nan for d in data]
        color = cmap(idx % 10)
        ax.plot(x, clip_vals, marker="s", linewidth=1.4, color=color, alpha=0.9, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Transformation method")
    ax.set_ylabel("CLIP similarity (0–1)")
    ax.set_title("CLIP similarity across transformation methods")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "clip_summary.png")
    plt.close(fig)

    hash_specs = [
        ("phash", "pHash", "phash_summary.png"),
        ("ahash", "aHash", "ahash_summary.png"),
        ("dhash", "dHash", "dhash_summary.png"),
        ("whash", "wHash", "whash_summary.png"),
        ("pdq", "PDQ", "pdq_summary.png"),
    ]
    for metric_key, metric_label, filename in hash_specs:
        fig, ax = plt.subplots(figsize=(18, 7))
        for idx, (name, data) in enumerate(parsed_reports):
            vals = [d[metric_key] if isinstance(d[metric_key], (int, float)) else np.nan for d in data]
            color = cmap(idx % 10)
            ax.plot(x, vals, marker="o", linewidth=1.4, color=color, alpha=0.9, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Transformation method")
        ax.set_ylabel(f"{metric_label} similarity (0–1)")
        ax.set_title(f"{metric_label} similarity across transformation methods")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    hash_ordered = [k for k, _, _ in hash_specs]
    hash_labels = [label for _, label, _ in hash_specs]
    hash_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    fig, ax = plt.subplots(figsize=(18, 7))
    for metric_key, metric_label, color in zip(hash_ordered, hash_labels, hash_colors):
        all_values = []
        for _, data in parsed_reports:
            all_values.append([d[metric_key] if isinstance(d[metric_key], (int, float)) else np.nan for d in data])
        avg_values = []
        for j in range(len(labels)):
            col_vals = [values[j] for values in all_values if not np.isnan(values[j])]
            avg_values.append(float(np.mean(col_vals)) if col_vals else np.nan)
        ax.plot(x, avg_values, marker="o", linewidth=1.8, color=color, label=metric_label)
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

    level_order = ["low", "mid", "high"]
    level_buckets = {level: {"vit": [], "clip": [], "phash": [], "ahash": [], "dhash": [], "whash": [], "pdq": []} for level in level_order}
    for _, data in parsed_reports:
        for d in data:
            if d["op_sym"] in ("F1", "F2"):
                continue
            level = d["level"]
            if level not in level_buckets:
                continue
            for key in level_buckets[level]:
                value = d[key]
                if isinstance(value, (int, float)):
                    level_buckets[level][key].append(value)

    transformer_levels = {
        "ViT": [float(np.mean(level_buckets[level]["vit"])) if level_buckets[level]["vit"] else np.nan for level in level_order],
        "CLIP": [float(np.mean(level_buckets[level]["clip"])) if level_buckets[level]["clip"] else np.nan for level in level_order],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in transformer_levels.items():
        ax.plot(level_order, values, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Transformer similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "transformer_level_summary.png")
    plt.close(fig)

    hash_levels = {}
    for metric_key, metric_label in [("phash", "pHash"), ("ahash", "aHash"), ("dhash", "dHash"), ("whash", "wHash"), ("pdq", "PDQ")]:
        hash_levels[metric_label] = [float(np.mean(level_buckets[level][metric_key])) if level_buckets[level][metric_key] else np.nan for level in level_order]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in hash_levels.items():
        ax.plot(level_order, values, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Hash similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hash_level_summary.png")
    plt.close(fig)

    combined_colors = {
        "ViT": "#2A6F97",
        "CLIP": "#5DA5E0",
        "pHash": "#E45756",
        "aHash": "#F5853F",
        "dHash": "#F9A03F",
        "wHash": "#D85040",
        "PDQ": "#A23E48",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in transformer_levels.items():
        ax.plot(level_order, values, marker="o", linewidth=1.8, color=combined_colors[name], label=name)
    for name, values in hash_levels.items():
        ax.plot(level_order, values, marker="s", linewidth=1.8, color=combined_colors[name], linestyle="--", label=name)
    ax.set_xlabel("Level")
    ax.set_ylabel("Similarity (0–1)")
    ax.set_title("Transformer vs Hash similarity by level (excluding F1/F2)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_transformer_hash_level_summary.png")
    plt.close(fig)

    print(f"✅ Summary charts saved to {out_dir.resolve()}/vit_summary.png, {out_dir.resolve()}/clip_summary.png, {out_dir.resolve()}/transformer_level_summary.png, {out_dir.resolve()}/hash_level_summary.png and {out_dir.resolve()}/combined_transformer_hash_level_summary.png")

def save_low_level_transform_summary(report_files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    low_data = {}

    for xlsx_path in report_files:
        data = parse_report_data(xlsx_path)
        for d in data:
            if d["level"] != "low" or d["op_sym"] in ("F1", "F2"):
                continue
            op_name = xlsx_path.name
            label = d["label"]
            low_data.setdefault(label, {"vit": [], "clip": []})
            if isinstance(d["vit"], (int, float)):
                low_data[label]["vit"].append(d["vit"])
            if isinstance(d["clip"], (int, float)):
                low_data[label]["clip"].append(d["clip"])

    if not low_data:
        print(f"⚠️ No low-level transformation data found in {REPORT_DIR}")
        return

    labels = sorted(low_data.keys(), key=lambda x: x)
    x = np.arange(len(labels))
    vit_vals = [float(np.mean(low_data[label]["vit"])) if low_data[label]["vit"] else np.nan for label in labels]
    clip_vals = [float(np.mean(low_data[label]["clip"])) if low_data[label]["clip"] else np.nan for label in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, vit_vals, marker="o", linewidth=1.8, color="#4C72B0", label="ViT")
    ax.plot(x, clip_vals, marker="s", linewidth=1.8, color="#C44E52", label="CLIP")
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

    print(f"✅ Low-level transform summary saved to {out_dir.resolve()}/low_level_transform_summary.png")

def main():
    report_files = sorted(REPORT_DIR.glob("*_report.xlsx"))
    if not report_files:
        print(f"No report files found in {REPORT_DIR}")
        return

    summary_out_dir = OUT_ROOT / "summary_ai_art"
    save_summary_chart(report_files, summary_out_dir)
    save_low_level_transform_summary(report_files, summary_out_dir)

    for xlsx_path in report_files:
        folder_name = xlsx_path.name[:-len("_report.xlsx")] if xlsx_path.name.endswith("_report.xlsx") else xlsx_path.stem
        out_dir = OUT_ROOT / folder_name
        print(f"\nProcessing: {xlsx_path.name} -> {out_dir}")
        save_plots_for_report(xlsx_path, out_dir)


if __name__ == "__main__":
    main()

