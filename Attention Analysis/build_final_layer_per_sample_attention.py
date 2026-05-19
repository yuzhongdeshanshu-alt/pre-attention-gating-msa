#!/usr/bin/env python3
"""Assemble final-layer per-sample attention tables for plotting.

Inputs are the per-sample outputs from the three metric extraction scripts:

1. analyze_self_attention_mass.py
   -> attention_allocation_per_sample.csv
2. analyze_self_inter_modality_entropy.py
   -> self_attention_contribution_per_sample.csv
3. analyze_cross_inter_modality_entropy.py
   -> cross_attention_contribution_aggregate_per_sample.csv

The script filters the selected layer and mask mode, joins the self-attention
macro and micro metrics by sample, and writes four CSV files beside this script:

    final_layer_per_sample_attention/
      self_attention_NoGate_Final Layer.csv
      self_attention_PreGate_Final Layer.csv
      cross_attention_NoGate_Final Layer.csv
      cross_attention_PreGate_Final Layer.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SELF_ENTROPY_COL = "self_inter_norm_entropy"
LEGACY_SELF_ENTROPY_COL = "overall" + "_inter_norm_entropy_true"
CROSS_ENTROPY_COL = "aggregate_pair_equal_norm_entropy"
LEGACY_CROSS_ENTROPY_COL = "pair_equal_norm_entropy"

# Keep the plot-input CSVs deliberately small: only identifiers and metrics
# needed for the two published attention figures are retained.
SELF_OUTPUT_COLUMNS = [
    "attention_family",
    "model_role",
    "condition_label",
    "layer",
    "sample_idx",
    "row_weighted_intra",
    "row_weighted_inter",
    SELF_ENTROPY_COL,
]

CROSS_OUTPUT_COLUMNS = [
    "attention_family",
    "model_role",
    "condition_label",
    "layer",
    "sample_idx",
    CROSS_ENTROPY_COL,
]


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    # Defaults match the original local analysis layout, while command-line
    # arguments make the script usable on HPC outputs or copied result folders.
    downloads_root = script_root.parents[1] / "attention_distribution_analysis_downloads"
    parser = argparse.ArgumentParser(description="Assemble final-layer per-sample attention CSVs for plotting.")
    parser.add_argument(
        "--self-allocation-csv",
        type=Path,
        default=downloads_root / "self_attention_N2_vs_C3" / "attention_allocation_per_sample.csv",
        help="Output from analyze_self_attention_mass.py.",
    )
    parser.add_argument(
        "--self-entropy-csv",
        type=Path,
        default=downloads_root
        / "contribution_analysis_20260515"
        / "self_attention_N2_vs_C3"
        / "self_attention_contribution_per_sample.csv",
        help="Output from analyze_self_inter_modality_entropy.py.",
    )
    parser.add_argument(
        "--cross-entropy-csv",
        type=Path,
        default=downloads_root
        / "contribution_analysis_20260515"
        / "cross_attention_b3c2_none_vs_pre"
        / "cross_attention_contribution_aggregate_per_sample.csv",
        help="Output from analyze_cross_inter_modality_entropy.py.",
    )
    parser.add_argument(
        "--final-csv-dir",
        type=Path,
        default=script_root / "final_layer_per_sample_attention",
        help="Directory for the four plot-input CSVs.",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Layer to keep; -1 means max layer in each source CSV.")
    parser.add_argument("--mask-mode", default="include_special", choices=["include_special", "content_only"])
    return parser.parse_args()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")
    return pd.read_csv(path)


def normalize_role(condition: object) -> str:
    # Source scripts use short condition labels such as "none" and "pre". The
    # final CSVs use figure-facing roles that are consistent across architectures.
    value = str(condition).strip().lower()
    if value in {"none", "nogate", "no_gate", "no-gate"}:
        return "NoGate"
    if value in {"pre", "pregate", "pre_gate", "pre-gate"}:
        return "PreGate"
    raise ValueError(f"Cannot map condition to model role: {condition!r}")


def filter_layer_and_mask(df, layer: int, mask_mode: str):
    # Keep a single layer and one masking convention so all four CSVs describe
    # the same attention view.
    out = df.copy()
    if "mask_mode" in out.columns:
        out = out[out["mask_mode"].astype(str) == mask_mode]
    if "layer" not in out.columns:
        raise ValueError("Source CSV must contain a layer column.")
    numeric_layer = out["layer"].astype(int)
    selected_layer = numeric_layer.max() if layer < 0 else layer
    out = out[numeric_layer == selected_layer].copy()
    if out.empty:
        raise ValueError(f"No rows left after filtering mask_mode={mask_mode!r}, layer={selected_layer}.")
    out["layer"] = selected_layer
    return out


def first_existing_column(df, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"None of the expected columns were found: {', '.join(candidates)}")


def build_final_csvs(args: argparse.Namespace):
    # The self-attention plot needs both macro mass and micro entropy, so those
    # tables are joined before splitting NoGate and PreGate into separate files.
    allocation = filter_layer_and_mask(load_csv(args.self_allocation_csv), args.layer, args.mask_mode)
    self_entropy = filter_layer_and_mask(load_csv(args.self_entropy_csv), args.layer, args.mask_mode)
    cross_entropy = filter_layer_and_mask(load_csv(args.cross_entropy_csv), args.layer, args.mask_mode)

    self_entropy_col = first_existing_column(self_entropy, [SELF_ENTROPY_COL, LEGACY_SELF_ENTROPY_COL])
    cross_entropy_col = first_existing_column(
        cross_entropy,
        [CROSS_ENTROPY_COL, "direction_mean_norm_entropy", LEGACY_CROSS_ENTROPY_COL],
    )

    allocation["model_role"] = allocation["condition"].map(normalize_role)
    self_entropy["model_role"] = self_entropy["condition"].map(normalize_role)
    cross_entropy["model_role"] = cross_entropy["condition"].map(normalize_role)

    merge_keys = ["model_role", "layer", "sample_idx"]
    # Join self-attention macro mass and micro entropy by sample so each
    # self-attention condition has one compact per-sample table. Cross-attention
    # contributes only the micro entropy metric used in the overview figure.
    self_final = allocation[
        merge_keys + ["condition_label", "row_weighted_intra", "row_weighted_inter"]
    ].merge(
        self_entropy[merge_keys + [self_entropy_col]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    self_final.rename(columns={self_entropy_col: SELF_ENTROPY_COL}, inplace=True)
    self_final["attention_family"] = "self_attention"

    cross_final = cross_entropy[merge_keys + ["condition_label", cross_entropy_col]].copy()
    cross_final.rename(columns={cross_entropy_col: CROSS_ENTROPY_COL}, inplace=True)
    cross_final["attention_family"] = "cross_attention"

    # Fail early if one condition silently drops during filtering or joining; a
    # missing role would otherwise produce a misleading figure without error.
    expected = {"NoGate", "PreGate"}
    if set(self_final["model_role"]) != expected:
        raise ValueError(f"Self final CSVs should contain roles {expected}, found {set(self_final['model_role'])}.")
    if set(cross_final["model_role"]) != expected:
        raise ValueError(f"Cross final CSVs should contain roles {expected}, found {set(cross_final['model_role'])}.")

    args.final_csv_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "self_no": args.final_csv_dir / "self_attention_NoGate_Final Layer.csv",
        "self_pre": args.final_csv_dir / "self_attention_PreGate_Final Layer.csv",
        "cross_no": args.final_csv_dir / "cross_attention_NoGate_Final Layer.csv",
        "cross_pre": args.final_csv_dir / "cross_attention_PreGate_Final Layer.csv",
    }

    # Split by model role to keep the plotting script simple and transparent: it
    # loads one CSV per attention family and gating condition.
    for role, path in [("NoGate", paths["self_no"]), ("PreGate", paths["self_pre"])]:
        rows = (
            self_final[self_final["model_role"] == role]
            .sort_values("sample_idx")
            [SELF_OUTPUT_COLUMNS]
            .to_dict("records")
        )
        write_csv(path, SELF_OUTPUT_COLUMNS, rows)

    for role, path in [("NoGate", paths["cross_no"]), ("PreGate", paths["cross_pre"])]:
        rows = (
            cross_final[cross_final["model_role"] == role]
            .sort_values("sample_idx")
            [CROSS_OUTPUT_COLUMNS]
            .to_dict("records")
        )
        write_csv(path, CROSS_OUTPUT_COLUMNS, rows)

    return paths


def main() -> None:
    args = parse_args()
    paths = build_final_csvs(args)
    print(f"Wrote per-sample attention CSVs to {args.final_csv_dir}")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
