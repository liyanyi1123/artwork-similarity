from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "system_implementation_flowchart.png"
SVG_PATH = OUT_DIR / "system_implementation_flowchart.svg"


COLORS = {
    "ink": "#172033",
    "line": "#53657A",
    "stage1_bg": "#F5F9FD",
    "stage1_edge": "#7196B5",
    "stage2_bg": "#F5FAF7",
    "stage2_edge": "#6E9B83",
    "stage3_bg": "#FFFAF2",
    "stage3_edge": "#C9904D",
    "input_bg": "#F8FAFC",
    "input_edge": "#64748B",
    "model_bg": "#E8F1FB",
    "model_edge": "#28689D",
    "vector_bg": "#EAF7F1",
    "vector_edge": "#27805E",
    "process_bg": "#FFF4DE",
    "process_edge": "#C97916",
    "decision_bg": "#FFF0C9",
    "decision_edge": "#A85F00",
    "success_bg": "#E7F6EC",
    "success_edge": "#23824B",
    "failure_bg": "#FDECEC",
    "failure_edge": "#B43A3A",
    "database_bg": "#EEF0F8",
    "database_edge": "#515B8A",
    "note_bg": "#FFFFFF",
    "note_edge": "#94A3B8",
}


def rounded_box(ax, center, size, text, face, edge, fontsize=10.5, linewidth=1.5):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
        zorder=4,
    )
    return patch


def panel(ax, xy, size, title, face, edge):
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.25,
        y + height - 0.28,
        title,
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )


def arrow(ax, start, end, label=None, rad=0.0, dashed=False, zorder=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=COLORS["line"],
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + 0.12
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            zorder=5,
        )


fig, ax = plt.subplots(figsize=(16, 12), dpi=160)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

# Stage panels
panel(ax, (0.35, 7.35), (7.25, 4.25), "Stage 1: Data Preparation", COLORS["stage1_bg"], COLORS["stage1_edge"])
panel(ax, (8.0, 7.35), (7.65, 4.25), "Stage 2: User Input", COLORS["stage2_bg"], COLORS["stage2_edge"])
panel(ax, (0.35, 0.35), (15.3, 6.55), "Stage 3: Candidate Retrieval and Comparison", COLORS["stage3_bg"], COLORS["stage3_edge"])

# Stage 1 nodes
rounded_box(ax, (3.95, 10.8), (2.25, 0.62), "Dataset Images", COLORS["input_bg"], COLORS["input_edge"])
rounded_box(ax, (2.15, 9.72), (1.95, 0.62), "CLIP Encoder", COLORS["model_bg"], COLORS["model_edge"])
rounded_box(ax, (5.75, 9.72), (1.95, 0.62), "ViT Encoder", COLORS["model_bg"], COLORS["model_edge"])
rounded_box(ax, (2.15, 8.65), (2.55, 0.7), "CLIP Embedding\nVectors", COLORS["vector_bg"], COLORS["vector_edge"])
rounded_box(ax, (5.75, 8.65), (2.55, 0.7), "ViT Embedding\nVectors", COLORS["vector_bg"], COLORS["vector_edge"])
rounded_box(
    ax,
    (3.95, 7.78),
    (3.2, 0.72),
    "Registry Vector Database\nImage IDs + CLIP + ViT Embeddings",
    COLORS["database_bg"],
    COLORS["database_edge"],
    fontsize=9.8,
    linewidth=1.8,
)

arrow(ax, (3.7, 10.48), (2.35, 10.05))
arrow(ax, (4.2, 10.48), (5.55, 10.05))
arrow(ax, (2.15, 9.4), (2.15, 9.0))
arrow(ax, (5.75, 9.4), (5.75, 9.0))
arrow(ax, (2.15, 8.3), (3.15, 8.05))
arrow(ax, (5.75, 8.3), (4.75, 8.05))

# Stage 2 nodes
rounded_box(ax, (11.82, 10.8), (2.35, 0.62), "Submitted Image", COLORS["input_bg"], COLORS["input_edge"])
rounded_box(ax, (10.05, 9.72), (1.95, 0.62), "CLIP Encoder", COLORS["model_bg"], COLORS["model_edge"])
rounded_box(ax, (13.6, 9.72), (1.95, 0.62), "ViT Encoder", COLORS["model_bg"], COLORS["model_edge"])
rounded_box(ax, (10.05, 8.65), (2.55, 0.7), "Query CLIP\nEmbedding", COLORS["vector_bg"], COLORS["vector_edge"])
rounded_box(ax, (13.6, 8.65), (2.55, 0.7), "Query ViT\nEmbedding", COLORS["vector_bg"], COLORS["vector_edge"])

arrow(ax, (11.57, 10.48), (10.25, 10.05))
arrow(ax, (12.07, 10.48), (13.4, 10.05))
arrow(ax, (10.05, 9.4), (10.05, 9.0))
arrow(ax, (13.6, 9.4), (13.6, 9.0))

# Stage 3 nodes
rounded_box(ax, (7.95, 6.15), (3.2, 0.78), "Retrieve the Top-3 Most Similar\nRegistry Images", COLORS["process_bg"], COLORS["process_edge"])
rounded_box(ax, (7.95, 5.0), (2.95, 0.68), "Compare the Query with\nEach Candidate", COLORS["process_bg"], COLORS["process_edge"])
rounded_box(ax, (5.05, 3.9), (3.2, 0.72), "Compute Absolute CLIP\nCosine Similarity", COLORS["process_bg"], COLORS["process_edge"])
rounded_box(ax, (10.85, 3.9), (3.2, 0.72), "Compute Absolute ViT\nCosine Similarity", COLORS["process_bg"], COLORS["process_edge"])

diamond_center = (7.95, 2.45)
diamond_width = 5.0
diamond_height = 1.75
diamond = Polygon(
    [
        (diamond_center[0], diamond_center[1] + diamond_height / 2),
        (diamond_center[0] + diamond_width / 2, diamond_center[1]),
        (diamond_center[0], diamond_center[1] - diamond_height / 2),
        (diamond_center[0] - diamond_width / 2, diamond_center[1]),
    ],
    closed=True,
    facecolor=COLORS["decision_bg"],
    edgecolor=COLORS["decision_edge"],
    linewidth=1.8,
    zorder=3,
)
ax.add_patch(diamond)
ax.text(
    *diamond_center,
    "Does any Top-3 candidate satisfy both?\n\n"
    "|CLIP similarity| >= CLIP threshold\n"
    "AND\n"
    "|ViT similarity| >= ViT threshold",
    ha="center",
    va="center",
    fontsize=10.3,
    color=COLORS["ink"],
    linespacing=1.2,
    zorder=4,
)

rounded_box(ax, (3.7, 0.85), (2.45, 0.65), "Registration Failed", COLORS["failure_bg"], COLORS["failure_edge"], linewidth=1.8)
rounded_box(ax, (8.9, 0.85), (2.65, 0.65), "Registration Successful", COLORS["success_bg"], COLORS["success_edge"], linewidth=1.8)
rounded_box(
    ax,
    (13.2, 0.85),
    (3.5, 0.75),
    "Store the New Image ID,\nCLIP Vector, and ViT Vector",
    COLORS["process_bg"],
    COLORS["process_edge"],
    fontsize=9.8,
)

rounded_box(
    ax,
    (12.95, 2.45),
    (4.35, 1.1),
    "Equivalent success rule for every candidate:\n"
    "|CLIP similarity| < CLIP threshold OR\n"
    "|ViT similarity| < ViT threshold",
    COLORS["note_bg"],
    COLORS["note_edge"],
    fontsize=9.2,
    linewidth=1.1,
)

# Cross-stage inputs
arrow(ax, (3.95, 7.4), (7.05, 6.55), rad=-0.05)
arrow(ax, (10.05, 8.3), (7.45, 6.55), rad=0.08)
arrow(ax, (13.6, 8.3), (8.45, 6.55), rad=0.12)

# Comparison logic
arrow(ax, (7.95, 5.75), (7.95, 5.36))
arrow(ax, (7.45, 4.66), (5.45, 4.28), rad=0.04)
arrow(ax, (8.45, 4.66), (10.45, 4.28), rad=-0.04)
arrow(ax, (5.25, 3.52), (6.45, 2.9), rad=-0.05)
arrow(ax, (10.65, 3.52), (9.45, 2.9), rad=0.05)
arrow(ax, (6.55, 1.85), (4.1, 1.18), label="Yes: Potential Match", rad=0.04)
arrow(ax, (9.35, 1.85), (8.9, 1.18), label="No: No Joint Match", rad=-0.04)
arrow(ax, (10.25, 0.85), (11.43, 0.85))

# Feedback into the Stage 1 registry
ax.plot([14.95, 15.35, 15.35], [0.85, 0.85, 7.78], color=COLORS["line"], linewidth=1.25, zorder=1)
arrow(ax, (15.35, 7.78), (5.58, 7.78), label="Append to Registry Vector Database", zorder=2)

fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(PNG_PATH)
print(SVG_PATH)
