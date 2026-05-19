#!/usr/bin/env python3
"""Create final-layer attention summary figures from per-sample CSVs.

The script expects four CSV files in ``final_layer_per_sample_attention`` next
to this file:

    self_attention_NoGate_Final Layer.csv
    self_attention_PreGate_Final Layer.csv
    cross_attention_NoGate_Final Layer.csv
    cross_attention_PreGate_Final Layer.csv

It writes two figure files and compact summary tables to the selected output
folder. By default, the output folder is this script's directory.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Colors are fixed here so that regenerated figures remain consistent across runs.
C_NG = "#6B9FD4"  # Cross-attention without pre-gating, light blue.
C_PG = "#114283"  # Cross-attention with pre-gating, dark blue.
S_NG = "#74C476"  # Self-attention without pre-gating, light green.
S_PG = "#2E8B57"  # Self-attention with pre-gating, dark green.
INTRA_PURPLE = "#C7B9FF"
INTER_PURPLE = "#5B2A86"

SELF_ENTROPY_COL = "self_inter_norm_entropy"
LEGACY_SELF_ENTROPY_COL = "overall" + "_inter_norm_entropy_true"
CROSS_ENTROPY_COL = "aggregate_pair_equal_norm_entropy"


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create final-layer attention summary figures from four CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_root / "final_layer_per_sample_attention",
        help="Directory containing the four per-sample attention CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root,
        help="Directory for the generated figures and summary tables.",
    )
    return parser.parse_args()


def configure_matplotlib(output_dir: Path):
    # Keep Matplotlib cache files inside the output directory, which is useful on
    # shared compute systems where the default cache location may be unavailable.
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 14,
        }
    )
    return plt


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def load_data(input_dir: Path):
    import pandas as pd

    paths = {
        "self_no": input_dir / "self_attention_NoGate_Final Layer.csv",
        "self_pre": input_dir / "self_attention_PreGate_Final Layer.csv",
        "cross_no": input_dir / "cross_attention_NoGate_Final Layer.csv",
        "cross_pre": input_dir / "cross_attention_PreGate_Final Layer.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input CSV(s):\n" + "\n".join(missing))

    frames = (
        pd.read_csv(paths["self_no"]),
        pd.read_csv(paths["self_pre"]),
        pd.read_csv(paths["cross_no"]),
        pd.read_csv(paths["cross_pre"]),
    )
    # Some exported self-attention tables use a longer entropy column name;
    # accepting it here keeps the plotting step robust across table schemas.
    for frame in frames[:2]:
        if SELF_ENTROPY_COL not in frame.columns and LEGACY_SELF_ENTROPY_COL in frame.columns:
            frame.rename(columns={LEGACY_SELF_ENTROPY_COL: SELF_ENTROPY_COL}, inplace=True)
    return frames


def add_bar_labels(ax, bars, dy: float) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )


def plot_entropy_overview(plt, output_dir: Path, self_no, self_pre, cross_no, cross_pre):
    import matplotlib.patches as mpatches
    import numpy as np

    labels = ["S-NoGate", "S-PreGate", "C-NoGate", "C-PreGate"]
    values = [
        self_no[SELF_ENTROPY_COL].mean(),
        self_pre[SELF_ENTROPY_COL].mean(),
        cross_no[CROSS_ENTROPY_COL].mean(),
        cross_pre[CROSS_ENTROPY_COL].mean(),
    ]
    colors = [S_NG, S_PG, C_NG, C_PG]
    x = np.array([0.0, 1.0, 2.35, 3.35])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(x, values, width=0.62, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8, zorder=3)
    add_bar_labels(ax, bars, dy=0.020)

    ax.set_ylabel("Normalized entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, 3.9)
    ax.set_title("Final-layer inter-modality entropy")
    ax.legend(
        handles=[
            mpatches.Patch(color=S_NG, label="S-NoGate"),
            mpatches.Patch(color=S_PG, label="S-PreGate"),
            mpatches.Patch(color=C_NG, label="C-NoGate"),
            mpatches.Patch(color=C_PG, label="C-PreGate"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    path = output_dir / "01_entropy_overview.png"
    fig.savefig(path)
    plt.close(fig)
    return path, {
        "self_entropy_nogate": values[0],
        "self_entropy_pregate": values[1],
        "self_entropy_gap": values[0] - values[1],
        "cross_entropy_nogate": values[2],
        "cross_entropy_pregate": values[3],
        "cross_entropy_gap": values[2] - values[3],
    }


def plot_self_macro_mass(plt, output_dir: Path, self_no, self_pre):
    import matplotlib.patches as mpatches
    import numpy as np

    labels = ["S-NoGate", "S-PreGate"]
    intra = [self_no["row_weighted_intra"].mean(), self_pre["row_weighted_intra"].mean()]
    inter = [self_no["row_weighted_inter"].mean(), self_pre["row_weighted_inter"].mean()]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(
        x - width / 2,
        intra,
        width,
        color=INTRA_PURPLE,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        label="Intra-modal mass",
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        inter,
        width,
        color=INTER_PURPLE,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        label="Inter-modal mass",
        zorder=3,
    )
    add_bar_labels(ax, b1, dy=0.018)
    add_bar_labels(ax, b2, dy=0.018)
    ax.set_ylabel("Row-weighted attention mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.78)
    ax.set_title("Self-attention macro allocation")
    ax.legend(
        handles=[
            mpatches.Patch(color=INTRA_PURPLE, label="Intra-modal mass"),
            mpatches.Patch(color=INTER_PURPLE, label="Inter-modal mass"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    path = output_dir / "02_self_macro_mass.png"
    fig.savefig(path)
    plt.close(fig)
    return path, {
        "nogate_intra": intra[0],
        "nogate_inter": inter[0],
        "pregate_intra": intra[1],
        "pregate_inter": inter[1],
        "inter_delta_pregate_minus_nogate": inter[1] - inter[0],
        "intra_delta_pregate_minus_nogate": intra[1] - intra[0],
    }


def write_mean_sd_table(output_dir: Path, self_no, self_pre, cross_no, cross_pre) -> None:
    rows = [
        ("entropy_overview", "Self", "S-NoGate", SELF_ENTROPY_COL, self_no),
        ("entropy_overview", "Self", "S-PreGate", SELF_ENTROPY_COL, self_pre),
        ("entropy_overview", "Cross", "C-NoGate", CROSS_ENTROPY_COL, cross_no),
        ("entropy_overview", "Cross", "C-PreGate", CROSS_ENTROPY_COL, cross_pre),
        ("self_macro_mass", "Self", "S-NoGate", "row_weighted_intra", self_no),
        ("self_macro_mass", "Self", "S-NoGate", "row_weighted_inter", self_no),
        ("self_macro_mass", "Self", "S-PreGate", "row_weighted_intra", self_pre),
        ("self_macro_mass", "Self", "S-PreGate", "row_weighted_inter", self_pre),
    ]
    table = []
    for figure, family, model, metric, frame in rows:
        table.append(
            {
                "figure": figure,
                "attention_family": family,
                "model": model,
                "metric": metric,
                "n": len(frame),
                "mean": f"{frame[metric].mean():.6f}",
                "sd": f"{frame[metric].std(ddof=0):.6f}",
            }
        )
    write_csv(
        output_dir / "attention_metric_mean_sd.csv",
        ["figure", "attention_family", "model", "metric", "n", "mean", "sd"],
        table,
    )


def make_readme(output_dir: Path, files: Sequence[Path]) -> None:
    text = [
        "# Generated attention figures",
        "",
        "Generated by `plot_final_layer_attention_figures.py` from the four per-sample attention CSVs.",
        "",
        "Generated figure files:",
    ]
    text.extend(f"- {path.name}" for path in files)
    (output_dir / "README_visualizations.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(args.output_dir)
    self_no, self_pre, cross_no, cross_pre = load_data(args.input_dir)

    files = []
    summary = {}
    path, stats = plot_entropy_overview(plt, args.output_dir, self_no, self_pre, cross_no, cross_pre)
    files.append(path)
    summary.update(stats)
    path, stats = plot_self_macro_mass(plt, args.output_dir, self_no, self_pre)
    files.append(path)
    summary.update(stats)

    write_csv(
        args.output_dir / "visualization_summary.csv",
        ["metric", "value"],
        [{"metric": key, "value": f"{value:.6f}"} for key, value in sorted(summary.items())],
    )
    write_csv(
        args.output_dir / "visualization_manifest.csv",
        ["plot_file", "source_dir"],
        [{"plot_file": path.name, "source_dir": display_path(args.input_dir, args.output_dir)} for path in files],
    )
    write_mean_sd_table(args.output_dir, self_no, self_pre, cross_no, cross_pre)
    make_readme(args.output_dir, files)
    print(f"Wrote {len(files)} visualization PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
