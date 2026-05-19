"""
statistical_analysis.py
───────────────────────
Statistical analysis pipeline for multimodal sentiment-regression experiments.

Reads per-run results from a CSV, computes descriptive statistics, runs a
2×2 repeated-measures ANOVA (Architecture × Gating) on the four multimodal
conditions, and runs two families of planned paired t-tests with
Holm–Bonferroni correction. All result tables and three summary figures
are saved to the specified output directory.

Input CSV — one row per training run × model:
    Seed    – integer run index (used to align paired observations)
    Model   – Text-only | Audio-only | Visual-only |
              C-NoGate | S-NoGate | C-PreGate | S-PreGate
    MAE     – mean absolute error (lower is better)
    Corr    – Pearson correlation (higher is better)
    Acc-2   – binary accuracy (higher is better)
    Acc-7   – 7-class accuracy (higher is better)
    F1      – macro F1 (higher is better)

Usage:
    python statistical_analysis.py
    python statistical_analysis.py --input path/to/raw_results.csv
    python statistical_analysis.py --input data.csv --output results/

Dependencies: numpy, pandas, scipy, matplotlib, pingouin
"""

import argparse
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Statistical analysis of multimodal model results.")
parser.add_argument(
    "--input", default="raw_all_seeds.csv",
    help="Path to the raw per-run CSV (default: raw_all_seeds.csv)")
parser.add_argument(
    "--output", default="output_stats",
    help="Directory for result tables and figures (default: output_stats/)")
args = parser.parse_args()

RAW_CSV = args.input
OUT     = args.output
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(RAW_CSV):
    raise FileNotFoundError(
        f"Input file not found: {RAW_CSV}\n"
        "Pass the correct path via --input <path>.")

df_raw = pd.read_csv(RAW_CSV)
N = df_raw["Seed"].nunique()

# Short keys used internally; LABELS maps back to display names for tables.
LABEL2KEY = {
    "C-NoGate":   "CNG", "C-PreGate":  "CPG",
    "S-NoGate":   "SNG", "S-PreGate":  "SPG",
    "Text-only":  "Text", "Audio-only": "Audio", "Visual-only": "Visual",
}
LABELS   = {v: k for k, v in LABEL2KEY.items()}
ALL_KEYS = ["Text", "Audio", "Visual", "CNG", "SNG", "CPG", "SPG"]

# Sort by Seed so that paired t-tests and ANOVA align observations correctly.
acc2, mae, corr, acc7, f1 = {}, {}, {}, {}, {}
for label, key in LABEL2KEY.items():
    sub = df_raw[df_raw["Model"] == label].sort_values("Seed")
    acc2[key] = sub["Acc-2"].values
    mae[key]  = sub["MAE"].values
    corr[key] = sub["Corr"].values
    acc7[key] = sub["Acc-7"].values
    f1[key]   = sub["F1"].values

print(f"Loaded {RAW_CSV}  (N={N} runs, {len(LABEL2KEY)} models)")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def holm(pvals):
    """Holm–Bonferroni step-down correction for a family of p-values.

    Sorts p-values ascending, applies sequential Bonferroni multipliers, then
    enforces monotonicity with a cumulative maximum so that adjusted p-values
    never decrease as we move down the sorted list.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    idx = np.argsort(pvals)
    raw = np.array([min(pvals[idx[k]] * (n - k), 1.0) for k in range(n)])
    for k in range(1, n):
        raw[k] = max(raw[k], raw[k - 1])
    adj = np.empty(n)
    for k, i in enumerate(idx):
        adj[i] = raw[k]
    return adj


def fmt_p(p, star=True):
    """Format a p-value as a string with optional significance markers.

    Thresholds: *** p < .001, ** p < .01, * p < .05, † p < .06 (marginal).
    """
    s = "< .001" if p < 0.001 else f"{p:.3f}"
    if star:
        if p < 0.001:  s += "***"
        elif p < 0.01: s += "**"
        elif p < 0.05: s += "*"
        elif p < 0.06: s += "†"
    return s


def fmt_small(v):
    """Format an F-statistic or η² — show '< .001' to avoid misleading 0.000."""
    return "< .001" if v < 0.0005 else f"{v:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def ms(arr):
    return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f}"

summary = [
    {"Model": LABELS[k], "MAE ↓": ms(mae[k]), "Corr ↑": ms(corr[k]),
     "Acc-2 ↑": ms(acc2[k]), "Acc-7 ↑": ms(acc7[k]), "F1 ↑": ms(f1[k])}
    for k in ALL_KEYS
]
df_sum = pd.DataFrame(summary)
df_sum.to_csv(f"{OUT}/summary_mean_sd.csv", index=False)
print("\n=== SUMMARY TABLE (mean ± SD) ===")
print(df_sum.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4.  2×2 REPEATED-MEASURES ANOVA  (Architecture × Gating)
# ══════════════════════════════════════════════════════════════════════════════
# Each training run is a subject; Architecture (Cross / Self) and Gating
# (NoGate / PreGate) are the two within-subject factors.
def run_anova():
    rows = []
    for s in range(N):
        for arch, gate, key in [
            ("Cross", "NoGate", "CNG"), ("Cross", "PreGate", "CPG"),
            ("Self",  "NoGate", "SNG"), ("Self",  "PreGate", "SPG"),
        ]:
            rows.append(dict(seed=s, Architecture=arch, Gating=gate,
                             acc2=acc2[key][s], mae=mae[key][s]))
    df = pd.DataFrame(rows)
    an2 = pg.rm_anova(df, dv="acc2", within=["Architecture", "Gating"], subject="seed")
    anm = pg.rm_anova(df, dv="mae",  within=["Architecture", "Gating"], subject="seed")
    return an2, anm


an2, anm = run_anova()

print("\n=== 2×2 REPEATED-MEASURES ANOVA ===")
print("\n--- Acc-2 ---")
print(an2[["Source", "F", "p_unc", "ng2"]].to_string(index=False))
print("\n--- MAE ---")
print(anm[["Source", "F", "p_unc", "ng2"]].to_string(index=False))

# Build a compact summary table with formatted values.
anova_rows = []
for pat, label in [(r"^Architecture", "Architecture"), (r"^Gating", "Gating")]:
    r2 = an2[an2["Source"].str.contains(pat, regex=True)].iloc[0]
    rm = anm[anm["Source"].str.contains(pat, regex=True)].iloc[0]
    anova_rows.append({
        "Effect": label,
        "Acc-2 F": fmt_small(r2["F"]), "Acc-2 p": fmt_p(r2["p_unc"]), "Acc-2 η²": fmt_small(r2["ng2"]),
        "MAE F":   fmt_small(rm["F"]), "MAE p":   fmt_p(rm["p_unc"]), "MAE η²":   fmt_small(rm["ng2"]),
    })

# Interaction term: pingouin labels it with '*' between factor names.
inter2 = an2[an2["Source"].str.contains(r"\*", regex=True)]
interm = anm[anm["Source"].str.contains(r"\*", regex=True)]
if len(inter2):
    r2i = inter2.iloc[0]
    rmi = interm.iloc[0] if len(interm) else {}
    anova_rows.append({
        "Effect": "Architecture × Gating",
        "Acc-2 F": fmt_small(r2i["F"]), "Acc-2 p": fmt_p(r2i["p_unc"]), "Acc-2 η²": fmt_small(r2i["ng2"]),
        "MAE F":   fmt_small(rmi.get("F", float("nan"))),
        "MAE p":   fmt_p(rmi.get("p_unc", 1.0)),
        "MAE η²":  fmt_small(rmi.get("ng2", float("nan"))),
    })

df_anova = pd.DataFrame(anova_rows)
df_anova.to_csv(f"{OUT}/anova_results.csv", index=False)
print("\n=== ANOVA SUMMARY ===")
print(df_anova.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLANNED PAIRED T-TESTS  (Holm–Bonferroni corrected)
# ══════════════════════════════════════════════════════════════════════════════
# Family 1: pairwise contrasts within the 2×2 multimodal design.
# Tests whether gating or architecture systematically improves performance.
FAMILY1_PAIRS = [
    ("CPG", "CNG", "C-PreGate vs C-NoGate", "gating within Cross-attn"),
    ("SPG", "SNG", "S-PreGate vs S-NoGate", "gating within Self-attn"),
    ("SNG", "CNG", "S-NoGate vs C-NoGate",  "architecture without gating"),
    ("SPG", "CPG", "S-PreGate vs C-PreGate", "architecture with gating"),
]

# Family 2: each multimodal model vs the text-only baseline.
FAMILY2_MODELS = [
    ("CNG", "C-NoGate vs Text-only",  "ungated cross-attn vs text"),
    ("SNG", "S-NoGate vs Text-only",  "ungated self-attn vs text"),
    ("CPG", "C-PreGate vs Text-only", "gated cross-attn vs text"),
    ("SPG", "S-PreGate vs Text-only", "gated self-attn vs text"),
]


def family1_ttests():
    res = {}
    for dv_name, data in [("Acc-2", acc2), ("MAE", mae)]:
        raw_p, tvals, diffs = [], [], []
        for a, b, *_ in FAMILY1_PAIRS:
            if dv_name == "Acc-2":
                t, p = st.ttest_rel(data[a], data[b])
                diff = float(np.mean(data[a] - data[b]))
            else:
                # For MAE (lower is better), flip direction so a positive diff
                # means model a outperforms model b.
                t, p = st.ttest_rel(data[b], data[a])
                diff = float(np.mean(data[b] - data[a]))
            tvals.append(t); raw_p.append(p); diffs.append(diff)
        res[dv_name] = dict(pairs=FAMILY1_PAIRS, t=tvals, raw_p=raw_p,
                            adj_p=holm(raw_p), diff=diffs)
    return res


def family2_ttests():
    res = {}
    for dv_name, data in [("Acc-2", acc2), ("MAE", mae)]:
        raw_p, tvals, diffs = [], [], []
        for key, *_ in FAMILY2_MODELS:
            if dv_name == "Acc-2":
                t, p = st.ttest_rel(data[key], data["Text"])
                diff = float(np.mean(data[key] - data["Text"]))
            else:
                t, p = st.ttest_rel(data["Text"], data[key])
                diff = float(np.mean(data["Text"] - data[key]))
            tvals.append(t); raw_p.append(p); diffs.append(diff)
        res[dv_name] = dict(models=FAMILY2_MODELS, t=tvals, raw_p=raw_p,
                            adj_p=holm(raw_p), diff=diffs)
    return res


def make_ttest_df(d, key="pairs"):
    rows = []
    for i, item in enumerate(d[key]):
        label   = item[2] if key == "pairs" else item[1]
        purpose = item[3] if key == "pairs" else item[2]
        rows.append({
            "Comparison":   label,
            "Purpose":      purpose,
            "t":            f"{d['t'][i]:.3f}",
            "raw p":        fmt_p(d["raw_p"][i]),
            "adj p (Holm)": fmt_p(d["adj_p"][i]),
            "Mean diff":    f"{d['diff'][i]:+.3f}",
            "Significant?": "Yes" if d["adj_p"][i] < 0.05 else "No",
        })
    return pd.DataFrame(rows)


fam1 = family1_ttests()
fam2 = family2_ttests()

for dv in ("Acc-2", "MAE"):
    df_f1 = make_ttest_df(fam1[dv], key="pairs")
    df_f2 = make_ttest_df(fam2[dv], key="models")
    tag = dv.replace("-", "")
    df_f1.to_csv(f"{OUT}/ttest_Family1_{tag}.csv", index=False)
    df_f2.to_csv(f"{OUT}/ttest_Family2_{tag}.csv", index=False)
    print(f"\n=== FAMILY 1 T-TESTS ({dv}) ===")
    print(df_f1.to_string(index=False))
    print(f"\n=== FAMILY 2 T-TESTS ({dv}) ===")
    print(df_f2.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})

# Colour palette — Cross-attn models: blue family; Self-attn models: green family.
C_NG   = "#6B9FD4"   # C-NoGate
C_PG   = "#114283"   # C-PreGate
S_NG   = "#74C476"   # S-NoGate
S_PG   = "#2E8B57"   # S-PreGate
RED    = "#d62728"   # Text-only
YELLOW = "#FFD700"   # Audio-only
ORANGE = "#BD4800"   # Visual-only


# ── Fig 1: 2×2 grouped bar chart (Architecture × Gating) ─────────────────────
# Two groups on x: Cross-attention (left) and Self-attention (right).
# Within each group, NoGate bar is placed left and PreGate bar right.
# Dashed red lines connect NoGate means and PreGate means across architectures,
# making the architecture effect visually salient.
# A dotted reference line marks the Text-only baseline for comparison.

BAR_W   = 0.35
x_grp   = np.array([0.0, 1.0])
x_ng    = x_grp - BAR_W / 2
x_pg    = x_grp + BAR_W / 2

NG_KEYS = ["CNG", "SNG"]
PG_KEYS = ["CPG", "SPG"]
NG_COLS = [C_NG, S_NG]
PG_COLS = [C_PG, S_PG]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, dv, data_d, title in [
    (axes[0], "Acc-2", acc2, "(a) Binary Accuracy (Acc-2)"),
    (axes[1], "MAE",   mae,  "(b) Mean Absolute Error (MAE)"),
]:
    is_acc2 = (dv == "Acc-2")
    means_ng = [np.mean(data_d[k]) for k in NG_KEYS]
    sds_ng   = [np.std(data_d[k], ddof=1) for k in NG_KEYS]
    means_pg = [np.mean(data_d[k]) for k in PG_KEYS]
    sds_pg   = [np.std(data_d[k], ddof=1) for k in PG_KEYS]

    for x, m, sd, col in zip(x_ng, means_ng, sds_ng, NG_COLS):
        ax.bar(x, m, width=BAR_W, color=col, alpha=0.85,
               edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(x, m, yerr=sd, fmt="none",
                    color="black", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    for x, m, sd, col in zip(x_pg, means_pg, sds_pg, PG_COLS):
        ax.bar(x, m, width=BAR_W, color=col, alpha=0.85,
               edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(x, m, yerr=sd, fmt="none",
                    color="black", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    ax.plot(x_ng, means_ng, color="red", linewidth=1.8, linestyle="--", zorder=5)
    ax.plot(x_pg, means_pg, color="red", linewidth=1.8, linestyle="--", zorder=5)

    ann_off = 0.006
    for x, m, sd in zip(x_ng, means_ng, sds_ng):
        txt = f"{m*100:.2f}%" if is_acc2 else f"{m:.3f}"
        ax.text(x, m + sd + ann_off, txt, ha="center", va="bottom",
                fontsize=16, fontweight="bold", zorder=6)
    for x, m, sd in zip(x_pg, means_pg, sds_pg):
        txt = f"{m*100:.2f}%" if is_acc2 else f"{m:.3f}"
        ax.text(x, m + sd + ann_off, txt, ha="center", va="bottom",
                fontsize=16, fontweight="bold", zorder=6)

    text_val = np.mean(acc2["Text"]) if is_acc2 else np.mean(mae["Text"])
    text_line = ax.axhline(text_val, color=RED, linewidth=2.0, linestyle=":",
                           label="Text-only", zorder=2)

    if is_acc2:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.set_ylabel("Acc-2 (%)", fontweight="bold", fontsize=20)
    else:
        ax.set_ylabel("MAE (lower is better)", fontweight="bold", fontsize=20)
    ax.set_ylim(0.30, 1.00)

    ax.set_xticks(x_grp)
    ax.set_xticklabels(["Cross-attention", "Self-attention"], fontweight="bold", fontsize=20)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(title, fontweight="bold", fontsize=20)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=C_NG, label="C-NoGate"),
        mpatches.Patch(color=C_PG, label="C-PreGate"),
        mpatches.Patch(color=S_NG, label="S-NoGate"),
        mpatches.Patch(color=S_PG, label="S-PreGate"),
        text_line,
    ]
    legend_loc = "lower right" if is_acc2 else "upper right"
    ax.legend(handles=legend_handles, fontsize=15, loc=legend_loc)

plt.tight_layout()
fig.savefig(f"{OUT}/fig1_anova_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFig 1 → {OUT}/fig1_anova_bar.png")


# ── Fig 2: unimodal baseline comparison ───────────────────────────────────────
uni_keys   = ["Audio", "Visual", "Text"]
uni_labels = ["Audio-only", "Visual-only", "Text-only"]
uni_colors = [YELLOW, ORANGE, RED]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data_d, ylabel, title in [
    (axes[0], acc2, "Acc-2 (%)", "(a) Binary Accuracy (Acc-2)"),
    (axes[1], mae,  "MAE",       "(b) Mean Absolute Error (MAE)"),
]:
    means = [np.mean(data_d[k]) for k in uni_keys]
    sds   = [np.std(data_d[k], ddof=1) for k in uni_keys]
    bars  = ax.bar(uni_labels, means, color=uni_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax.errorbar(uni_labels, means, yerr=sds, fmt="none",
                color="black", capsize=5, capthick=1.5, linewidth=1.5)
    ax.set_xticklabels(uni_labels, rotation=15, ha="right", fontweight="bold", fontsize=20)
    if data_d is acc2:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.set_ylim(0.30, 1.00)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
    ax.set_title(title, fontweight="bold", fontsize=20)
    ax.grid(axis="y", alpha=0.3)
    for bar, m, sd in zip(bars, means, sds):
        txt = f"{m*100:.2f}%" if data_d is acc2 else f"{m:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.006,
                txt, ha="center", va="bottom", fontsize=17, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/fig2_unimodal_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig 2 → {OUT}/fig2_unimodal_bars.png")


# ── Fig 3: text baseline alongside all four multimodal models ─────────────────
multi_keys   = ["Text", "CNG", "SNG", "CPG", "SPG"]
multi_labels = ["Text-only", "C-NoGate", "S-NoGate", "C-PreGate", "S-PreGate"]
multi_colors = [RED, C_NG, S_NG, C_PG, S_PG]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data_d, ylabel, title in [
    (axes[0], acc2, "Acc-2 (%)", "(a) Binary Accuracy (Acc-2)"),
    (axes[1], mae,  "MAE",       "(b) Mean Absolute Error (MAE)"),
]:
    means = [np.mean(data_d[k]) for k in multi_keys]
    sds   = [np.std(data_d[k], ddof=1) for k in multi_keys]
    xs    = np.arange(len(multi_keys))
    bars  = ax.bar(xs, means, color=multi_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax.errorbar(xs, means, yerr=sds, fmt="none",
                color="black", capsize=5, capthick=1.5, linewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(multi_labels, rotation=20, ha="right", fontweight="bold", fontsize=20)
    if data_d is acc2:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.set_ylim(0.30, 1.00)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
    ax.set_title(title, fontweight="bold", fontsize=20)
    ax.grid(axis="y", alpha=0.3)
    for bar, m, sd in zip(bars, means, sds):
        txt = f"{m*100:.2f}%" if data_d is acc2 else f"{m:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.006,
                txt, ha="center", va="bottom", fontsize=17, fontweight="bold")
    legend_handles = [
        mpatches.Patch(color=RED,  label="Text-only"),
        mpatches.Patch(color=C_NG, label="C-NoGate"),
        mpatches.Patch(color=C_PG, label="C-PreGate"),
        mpatches.Patch(color=S_NG, label="S-NoGate"),
        mpatches.Patch(color=S_PG, label="S-PreGate"),
    ]
    legend_loc = "lower right" if data_d is acc2 else "upper right"
    ax.legend(handles=legend_handles, fontsize=15, loc=legend_loc)

plt.tight_layout()
fig.savefig(f"{OUT}/fig3_multimodal_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig 3 → {OUT}/fig3_multimodal_bars.png")

print("\n=== DONE ===")
print(f"All outputs written to: {OUT}/")