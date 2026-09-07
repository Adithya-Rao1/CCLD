from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.reporting import markdown_table, write_csv

METHODS = ["csho", "ddpm", "sdm"]
METHOD_LABELS = {"csho": "CSHO", "ddpm": "DDPM", "sdm": "SDM"}
METHOD_COLORS = {"csho": "#2b7a78", "ddpm": "#c1440e", "sdm": "#5b5f97"}
TE_HEAT_TASKS = ["Re{Ez}", "Im{Ez}", "T"]


def load_results(results_dir: str, seed: int):
    per_seed: Dict[str, Dict[str, float]] = {}
    per_sample: Dict[str, Dict[str, List[float]]] = {}
    sample_maps: Dict[str, Dict[str, np.ndarray]] = {}

    for method in METHODS:
        results_path = os.path.join(results_dir, f"{method}_results.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"missing {results_path} -- has {method} finished training?")
        with open(results_path) as f:
            r = json.load(f)
        seed_metrics = r["per_seed"].get(str(seed), r["per_seed"].get(seed))
        if seed_metrics is None:
            raise KeyError(f"seed {seed} not in {results_path}'s per_seed -- available: {list(r['per_seed'])}")
        per_seed[method] = seed_metrics

        per_sample_path = os.path.join(results_dir, f"{method}_per_sample_rel_l2_seed{seed}.json")
        with open(per_sample_path) as f:
            per_sample[method] = json.load(f)

        maps_path = os.path.join(results_dir, f"{method}_sample_maps_seed{seed}.npz")
        sample_maps[method] = dict(np.load(maps_path))

    return per_seed, per_sample, sample_maps


def build_summary_table(per_seed: Dict[str, Dict[str, float]], task_names: List[str], out_dir: str) -> List[Dict]:
    rows = []
    for name in task_names:
        row = {"field": name}
        for method in METHODS:
            row[METHOD_LABELS[method]] = round(per_seed[method].get(f"{name}_rel_l2", float("nan")), 4)
        rows.append(row)

    write_csv(rows, os.path.join(out_dir, "rel_l2_summary.csv"))
    table_md = markdown_table(rows, columns=["field"] + [METHOD_LABELS[m] for m in METHODS])
    with open(os.path.join(out_dir, "rel_l2_summary.md"), "w") as f:
        f.write(table_md)
    print(table_md)
    return rows


def plot_field_comparison(
    task_names: List[str],
    per_sample: Dict[str, Dict[str, List[float]]],
    sample_maps: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    sample_idx: int = 0,
) -> str:
    n_fields = len(task_names)
    fig, axes = plt.subplots(2, n_fields, figsize=(4.5 * n_fields, 8.5),
                              gridspec_kw={"height_ratios": [1.0, 1.3]})
    if n_fields == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(task_names):
        target = sample_maps["csho"][f"{name}_target"][sample_idx, 0]
        panels = [target] + [sample_maps[m][f"{name}_pred"][sample_idx, 0] for m in METHODS]
        combined = np.concatenate(panels, axis=1)

        ax_map = axes[0, col]
        im = ax_map.imshow(combined, cmap="viridis")
        ax_map.set_title(f"{name}: true | csho | ddpm | sdm", fontsize=10)
        ax_map.axis("off")
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        ax_violin = axes[1, col]
        data = [np.log10(np.clip(per_sample[m][name], 1e-8, None)) for m in METHODS]
        parts = ax_violin.violinplot(data, showmeans=True, showextrema=True)
        for pc, m in zip(parts["bodies"], METHODS):
            pc.set_facecolor(METHOD_COLORS[m])
            pc.set_alpha(0.7)
        ax_violin.set_xticks([1, 2, 3])
        ax_violin.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
        ax_violin.set_ylabel("log10(rel_l2)")
        ax_violin.set_title(f"{name} rel_l2 distribution")
        ax_violin.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "te_heat_field_comparison.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="CSHO/DDPM/SDM comparison figure + rel_l2 table for TE_heat")
    p.add_argument("--results-dir", required=True,)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-idx", type=int, default=0, help="which of the saved sample-map rows to plot")
    p.add_argument("--out-dir", default=None, help="defaults to --results-dir")
    args = p.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    per_seed, per_sample, sample_maps = load_results(args.results_dir, args.seed)
    build_summary_table(per_seed, TE_HEAT_TASKS, out_dir)
    fig_path = plot_field_comparison(TE_HEAT_TASKS, per_sample, sample_maps, out_dir, args.sample_idx)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
