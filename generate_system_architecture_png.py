from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "assets" / "architecture"
OUT_PATH = OUT_DIR / "system_architecture.png"


BG = "#f5f7fb"
INK = "#16324f"
SLATE = "#456173"
ACCENT = "#1f7a8c"
ACCENT_2 = "#2a9d8f"
ACCENT_3 = "#e9c46a"
ACCENT_4 = "#f4a261"
ACCENT_5 = "#e76f51"
WHITE = "#ffffff"


def add_box(ax, x, y, w, h, title, body, fc=WHITE, ec=INK, title_color=INK):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.055,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + 0.02,
        y + h - 0.10,
        body,
        ha="left",
        va="top",
        fontsize=10.2,
        color=SLATE,
        linespacing=1.4,
    )


def add_arrow(ax, x1, y1, x2, y2, color=INK, lw=1.8, style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=16,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.95,
        "EV Charging Grid Optimization System Architecture",
        fontsize=26,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    ax.text(
        0.05,
        0.915,
        "High-level view of the data, analytics pipeline, local dashboard, and user interaction flow.",
        fontsize=12.5,
        color=SLATE,
        ha="left",
        va="top",
    )

    add_box(
        ax,
        0.14,
        0.78,
        0.72,
        0.09,
        "User",
        "Student or analyst explores the dashboard, changes filters, compares scenarios,\n"
        "and interprets recommendations for EV charging decisions.",
        fc="#eef7fb",
        ec=ACCENT,
        title_color=ACCENT,
    )

    add_box(
        ax,
        0.14,
        0.57,
        0.72,
        0.14,
        "Local Dashboard  |  Streamlit App",
        "Interactive browser interface that presents KPIs, charging recommendations,\n"
        "strategy comparisons, model insights, and sensitivity-analysis results.",
        fc="#f0fbf8",
        ec=ACCENT_2,
        title_color=ACCENT_2,
    )

    add_box(
        ax,
        0.14,
        0.34,
        0.72,
        0.16,
        "Analytics and Optimization Engine  |  Python Pipeline",
        "Loads data, prepares features, trains regression models, predicts outcomes,\n"
        "runs the explicit charging optimizer, compares strategies, and performs weight sensitivity analysis.",
        fc="#fff8eb",
        ec=ACCENT_3,
        title_color="#8d6a00",
    )

    add_box(
        ax,
        0.08,
        0.09,
        0.38,
        0.16,
        "Data Layer",
        "EV charging dataset with session, station, demand, electrical,\n"
        "and grid-performance variables used as the system's source of truth.",
        fc=WHITE,
        ec=INK,
    )

    add_box(
        ax,
        0.54,
        0.09,
        0.38,
        0.16,
        "Outputs",
        "Recommended charging allocations, strategy comparison metrics,\n"
        "EDA figures, sensitivity-analysis summaries, and report-ready documentation.",
        fc=WHITE,
        ec=INK,
    )

    add_arrow(ax, 0.50, 0.78, 0.50, 0.71, color=ACCENT)
    add_arrow(ax, 0.50, 0.57, 0.50, 0.50, color=ACCENT_2)
    add_arrow(ax, 0.50, 0.34, 0.28, 0.25, color=ACCENT_3)
    add_arrow(ax, 0.50, 0.34, 0.72, 0.25, color=ACCENT_3)
    add_arrow(ax, 0.28, 0.25, 0.44, 0.34, color=INK, lw=1.3, style="<|-|>")
    add_arrow(ax, 0.72, 0.25, 0.56, 0.34, color=INK, lw=1.3, style="<|-|>")

    ax.text(
        0.05,
        0.03,
        "Flow: data -> analytics engine -> dashboard -> user decisions",
        fontsize=11.5,
        color=SLATE,
        ha="left",
        va="bottom",
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved architecture diagram to {OUT_PATH}")


if __name__ == "__main__":
    main()
