from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.reporting import markdown_table, write_csv

TE_HEAT_TASKS = ["Re{Ez}", "Im{Ez}", "T"]

ALL_CONDITIONS = ["symmetric_only", "skew_structured", "skew_generic_matched", "ccld_independent", "ddpm", "sdm"]

CONDITION_METHOD = {
    "symmetric_only": "ccld_pairwise",
    "skew_structured": "ccld_pairwise",
    "skew_generic_matched": "ccld_pairwise",
    "ccld_independent": "ccld_pairwise",
    "ddpm": "ddpm",
    "sdm": "sdm",
}

CONDITION_COLORS = {
    "symmetric_only": "#2b7a78",
    "ccld_independent": "#8c8c8c",
    "skew_structured": "#b8860b",
    "skew_generic_matched": "#3a6ea5",
    "ddpm": "#c1440e",
    "sdm": "#5b5f97",
}
CONDITION_LABELS = {
    "symmetric_only": "CCLD (coupled)",
    "ccld_independent": "CCLD (uncoupled)",
    "skew_structured": "CCLD (skew, structured)",
    "skew_generic_matched": "CCLD (skew, generic)",
    "ddpm": "DDPM",
    "sdm": "SDM",
}


class Condition(NamedTuple):
    label: str         
    file_prefix: str     
    results_dir: str   


def _find_condition_dir(label: str, sweep_dirs: List[str]) -> str:
    candidates = [os.path.join(d, label) for d in sweep_dirs]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"no '{label}' subdirectory found in any of {sweep_dirs} (looked for: {candidates})")


def resolve_conditions(condition_names: List[str], sweep_dirs: List[str]) -> List[Condition]:
    conditions = []
    for name in condition_names:
        if name not in CONDITION_METHOD:
            raise ValueError(f"unknown condition '{name}' -- expected one of {ALL_CONDITIONS}")
        conditions.append(Condition(name, CONDITION_METHOD[name], _find_condition_dir(name, sweep_dirs)))
    return conditions


def load_results(conditions: List[Condition], seeds: List[int]):
    """Per-seed metrics for `seeds[0]` (used for the field-map row), and per-sample
    rel_l2 values pooled across ALL of `seeds` (used for the violin row)."""
    per_seed: Dict[str, Dict[str, float]] = {}
    per_sample_pooled: Dict[str, Dict[str, List[float]]] = {}
    sample_maps: Dict[str, Dict[str, np.ndarray]] = {}

    for cond in conditions:
        results_path = os.path.join(cond.results_dir, f"{cond.file_prefix}_results.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"missing {results_path} -- has {cond.label} finished training?")
        with open(results_path) as f:
            r = json.load(f)
        seed0 = seeds[0]
        seed_metrics = r["per_seed"].get(str(seed0), r["per_seed"].get(seed0))
        if seed_metrics is None:
            raise KeyError(f"seed {seed0} not in {results_path}'s per_seed -- available: {list(r['per_seed'])}")
        per_seed[cond.label] = seed_metrics

        pooled: Dict[str, List[float]] = {}
        for seed in seeds:
            per_sample_path = os.path.join(cond.results_dir, f"{cond.file_prefix}_per_sample_rel_l2_seed{seed}.json")
            with open(per_sample_path) as f:
                this_seed = json.load(f)
            for name, values in this_seed.items():
                pooled.setdefault(name, []).extend(values)
        per_sample_pooled[cond.label] = pooled

        maps_path = os.path.join(cond.results_dir, f"{cond.file_prefix}_sample_maps_seed{seed0}.npz")
        sample_maps[cond.label] = dict(np.load(maps_path))

    return per_seed, per_sample_pooled, sample_maps


def build_summary_table(per_seed: Dict[str, Dict[str, float]], conditions: List[Condition], task_names: List[str], out_dir: str) -> List[Dict]:
    rows = []
    for name in task_names:
        row = {"field": name}
        for cond in conditions:
            row[CONDITION_LABELS.get(cond.label, cond.label)] = round(per_seed[cond.label].get(f"{name}_rel_l2", float("nan")), 4)
        rows.append(row)

    write_csv(rows, os.path.join(out_dir, "rel_l2_summary.csv"))
    table_md = markdown_table(rows, columns=["field"] + [CONDITION_LABELS.get(c.label, c.label) for c in conditions])
    with open(os.path.join(out_dir, "rel_l2_summary.md"), "w") as f:
        f.write(table_md)
    print(table_md)
    return rows


def plot_field_comparison(
    task_names: List[str],
    conditions: List[Condition],
    per_sample_pooled: Dict[str, Dict[str, List[float]]],
    sample_maps: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    n_seeds_pooled: int,
    sample_idx: int = 0,
) -> str:
    n_fields = len(task_names)
    fig, axes = plt.subplots(2, n_fields, figsize=(5.0 * n_fields, 9.0),
                              gridspec_kw={"height_ratios": [1.0, 1.3]})
    if n_fields == 1:
        axes = axes.reshape(2, 1)

    map_ref_label = conditions[0].label
    labels_for_title = " | ".join(["true"] + [CONDITION_LABELS.get(c.label, c.label) for c in conditions])

    for col, name in enumerate(task_names):
        target = sample_maps[map_ref_label][f"{name}_target"][sample_idx, 0]
        panels = [target] + [sample_maps[c.label][f"{name}_pred"][sample_idx, 0] for c in conditions]
        combined = np.concatenate(panels, axis=1)

        ax_map = axes[0, col]
        im = ax_map.imshow(combined, cmap="viridis")
        ax_map.set_title(name, fontsize=10)
        ax_map.axis("off")
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        ax_violin = axes[1, col]
        data = [np.log10(np.clip(per_sample_pooled[c.label][name], 1e-8, None)) for c in conditions]
        parts = ax_violin.violinplot(data, showmeans=True, showextrema=True)
        for pc, cond in zip(parts["bodies"], conditions):
            pc.set_facecolor(CONDITION_COLORS.get(cond.label, "#777777"))
            pc.set_alpha(0.7)
        ax_violin.set_xticks(range(1, len(conditions) + 1))
        ax_violin.set_xticklabels([CONDITION_LABELS.get(c.label, c.label) for c in conditions], rotation=20, ha="right")
        ax_violin.set_ylabel("log10(rel_l2)")
        ax_violin.set_title(f"{name} rel_l2 distribution (pooled over {n_seeds_pooled} seeds)")
        ax_violin.grid(alpha=0.3)

    fig.suptitle(f"left to right within each field-map panel: {labels_for_title}", fontsize=8, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = os.path.join(out_dir, "te_heat_field_comparison_multicond.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Multi-condition (default: symmetric_only/skew_structured/skew_generic_matched/ccld_independent/ddpm/sdm) violin comparison figure + rel_l2 table for TE_heat")
    p.add_argument("--sweep-dir", action="append", required=True,
                    help="results/experiment_2_physics/skew_coupling_sweep-style directory holding one subdir per condition (e.g. symmetric_only/, ddpm/). "
                         "Repeatable -- pass once per separate sweep invocation (e.g. one dir for the CCLD-family conditions, another for a later ddpm/sdm-only run).")
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                    help=f"comma-separated subset of {ALL_CONDITIONS}")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="seeds to pool per-sample values over for the violin row")
    p.add_argument("--sample-idx", type=int, default=0, help="which saved sample-map row to plot (uses seeds[0]'s maps)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    condition_names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    conditions = resolve_conditions(condition_names, args.sweep_dir)
    per_seed, per_sample_pooled, sample_maps = load_results(conditions, seeds)
    build_summary_table(per_seed, conditions, TE_HEAT_TASKS, args.out_dir)
    fig_path = plot_field_comparison(TE_HEAT_TASKS, conditions, per_sample_pooled, sample_maps, args.out_dir, len(seeds), args.sample_idx)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
