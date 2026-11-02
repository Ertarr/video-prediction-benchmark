"""
Summary of the memory runs: mean +/- std over seeds, tables and figures.

Reads results/memory/memory-*.json (train_memory.py), eval-rmt-*.json
(the RMT memory control, eval_memory.py on CPU) and eval-compact-*.json
if present. Zone SSIM is reported too, but on these blurry predictions
it rewards drawing nothing, so the conclusions use the zone MSE.

Writes results/memory_summary.json and figures/memory_*.png.

Usage:
    python scripts/memory_summary.py
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from scripts.eval_memory import KS


ORDER = ["window", "rmt", "knn", "full"]
LABELS = {"window": "Window (no memory)", "rmt": "RMT memory tokens",
          "knn": "kNN memory", "full": "Full context (control)",
          "compact": "Compact n_layers=3, last 10 frames"}
COLORS = {"window": "#2a78d6", "rmt": "#eb6834", "knn": "#1baf7a",
          "full": "#eda100", "compact": "gray"}
MARKERS = {"window": "o", "rmt": "s", "knn": "^", "full": "D", "compact": None}
# test-time controls: (key in the JSON, what is limited)
CONTROLS = {"knn": ("curve_by_cache", "cache"), "rmt": ("curve_by_memory", "memory reads")}


def mean_std(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def curve_stats(curves, metric):
    """curves: list of {k: {...}} -> {k: (mean, std)}"""
    return {str(k): mean_std([c[str(k)][metric] for c in curves]) for k in KS}


def controlled_pairs(by_limit):
    """For each limit C: last k with B still in memory (k <= 9 + C) and first k without it."""
    pairs = {}
    for c, curve in by_limit.items():
        k_in = max(k for k in KS if k <= 9 + int(c))
        k_out = min(k for k in KS if k > 9 + int(c))
        a, b = curve[str(k_in)]["zone_mse_avg"], curve[str(k_out)]["zone_mse_avg"]
        pairs[c] = {"k_in": k_in, "k_out": k_out, "with_b": a, "without_b": b,
                    "change": b / a - 1}
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/memory")
    parser.add_argument("--summary-out", default="results/memory_summary.json")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()
    fig_dir = Path(args.fig_dir)
    results_dir = Path(args.results_dir)

    runs = {}
    for path in sorted(results_dir.glob("memory-*.json")):
        r = json.loads(path.read_text())
        runs.setdefault(r["model"], []).append(r)
    # the RMT control was computed afterwards, from the checkpoints
    for path in sorted(results_dir.glob("eval-rmt-*.json")):
        ev = json.loads(path.read_text())
        seed = int(path.stem.split("seed")[-1])
        for r in runs.get("rmt", []):
            if r["seed"] == seed:
                r["curve_by_memory"] = ev["curve_by_memory"]
    compact = [json.loads(p.read_text()) for p in results_dir.glob("eval-compact-*.json")]

    summary = {}
    for model in [m for m in ORDER if m in runs]:
        rs = runs[model]
        entry = {
            "n_runs": len(rs),
            "seeds": [r["seed"] for r in rs],
            "n_params": rs[0]["n_params"],
            "best_epochs": [r["best_epoch"] for r in rs],
            "zone_mse": curve_stats([r["curve"] for r in rs], "zone_mse_avg"),
            "zone_ssim": curve_stats([r["curve"] for r in rs], "zone_ssim_avg"),
            "full_mse": curve_stats([r["curve"] for r in rs], "mse_avg"),
            "full_ssim": curve_stats([r["curve"] for r in rs], "ssim_avg"),
            "ssim_per_step_k0": [mean_std([r["curve"]["0"]["ssim_per_step"][t] for r in rs])
                                 for t in range(10)],
            "per_run": {str(r["seed"]): {"zone_mse_k0": r["curve"]["0"]["zone_mse_avg"],
                                         "zone_mse_k20": r["curve"]["20"]["zone_mse_avg"],
                                         "best_val_loss": r["best_val_loss"]} for r in rs},
        }
        if model in CONTROLS:
            key = CONTROLS[model][0]
            with_control = [r for r in rs if key in r]
            if with_control:
                limits = list(with_control[0][key])
                entry["control"] = {
                    "seeds": [r["seed"] for r in with_control],
                    # the "all frames" row uses the same runs as the limited rows
                    "zone_mse": {"all": curve_stats([r["curve"] for r in with_control],
                                                    "zone_mse_avg"),
                                 **{c: curve_stats([r[key][c] for r in with_control],
                                                   "zone_mse_avg") for c in limits}},
                    "pairs": {str(r["seed"]): controlled_pairs(r[key]) for r in with_control},
                }
        summary[model] = entry
    if compact:
        summary["compact"] = {
            "n_runs": 1,
            "zone_mse": curve_stats([compact[0]["curve"]], "zone_mse_avg"),
            "zone_ssim": curve_stats([compact[0]["curve"]], "zone_ssim_avg"),
        }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    # ---- tables (markdown)
    for metric, title, d in [("zone_mse", "MSE", 4), ("zone_ssim", "SSIM", 3)]:
        print(f"\nZone {title} of digit B after reappearance (mean +/- std over runs)\n")
        print("| Model | Runs | " + " | ".join(f"k={k}" for k in KS) + " |")
        print("|---|---|" + "---|" * len(KS))
        for model, e in summary.items():
            cells = [f"{e[metric][str(k)][0]:.{d}f} +/- {e[metric][str(k)][1]:.{d}f}"
                     if e["n_runs"] > 1 else f"{e[metric][str(k)][0]:.{d}f}" for k in KS]
            if model == "compact":
                # trained without the square: it only means something at k=0
                cells = cells[:1] + ["-"] * (len(KS) - 1)
            print(f"| {LABELS[model]} | {e['n_runs']} | " + " | ".join(cells) + " |")

    print("\nPer run: zone MSE when B was never hidden (k=0) and hidden for 20 frames\n")
    print("| Model | Seed | k=0 | k=20 | k=0 minus k=20 | Best val loss |")
    print("|---|---|---|---|---|---|")
    for model in [m for m in ORDER if m in summary]:
        for seed, p in summary[model]["per_run"].items():
            print(f"| {model} | {seed} | {p['zone_mse_k0']:.4f} | {p['zone_mse_k20']:.4f} | "
                  f"{p['zone_mse_k0'] - p['zone_mse_k20']:+.4f} | {p['best_val_loss']:.5f} |")

    for model in [m for m in CONTROLS if m in summary and "control" in summary[m]]:
        ctrl, what = summary[model]["control"], CONTROLS[model][1]
        print(f"\n{LABELS[model]}, {what} limited at test time: zone MSE, mean over seeds "
              f"{ctrl['seeds']}\n")
        print("| Limit | " + " | ".join(f"k={k}" for k in KS) + " |")
        print("|---|" + "---|" * len(KS))
        for limit, stats in ctrl["zone_mse"].items():
            name = "all frames" if limit == "all" else f"last {limit} frames"
            print(f"| {name} | " + " | ".join(f"{stats[str(k)][0]:.4f}" for k in KS) + " |")
        print("\nSame window, same number of frames kept, B in them or not:\n")
        limits = list(next(iter(ctrl["pairs"].values())))
        print("| Seed | " + " | ".join(
            f"last {c} frames: k={p['k_in']} -> k={p['k_out']}"
            for c, p in next(iter(ctrl["pairs"].values())).items()) + " |")
        print("|---|" + "---|" * len(limits))
        for seed, pairs in ctrl["pairs"].items():
            print(f"| {seed} | " + " | ".join(
                f"{p['with_b']:.4f} -> {p['without_b']:.4f} ({p['change'] * 100:+.1f}%)"
                for p in pairs.values()) + " |")

    # ---- forgetting curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, e in summary.items():
        for ax, metric in zip(axes, ["zone_mse", "zone_ssim"]):
            if model == "compact":
                ax.scatter([0], [e[metric]["0"][0]], color=COLORS[model], marker="*", s=120,
                           zorder=3, label=f"{LABELS[model]} (k=0 only)")
                continue
            mean = np.array([e[metric][str(k)][0] for k in KS])
            std = np.nan_to_num(np.array([e[metric][str(k)][1] for k in KS]))
            ax.plot(KS, mean, color=COLORS[model], marker=MARKERS[model], linewidth=2,
                    markersize=6, linestyle="--" if model == "compact" else "-",
                    label=f"{LABELS[model]} ({e['n_runs']} run{'s' if e['n_runs'] > 1 else ''})")
            ax.fill_between(KS, mean - std, mean + std, color=COLORS[model], alpha=0.15)
    for ax, ylabel in zip(axes, ["MSE on digit B's square (lower is better)",
                                 "SSIM on digit B's square"]):
        ax.axvline(10, color="gray", linewidth=1, linestyle=":")
        ax.set_xlabel("Frames during which digit B was hidden (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(KS)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Dotted line: the window holds 10 frames", fontsize=10)
    fig.suptitle("Error on digit B after it reappears (band = +/- 1 std over seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "memory_forgetting_curve.png", dpi=200, bbox_inches="tight")

    # ---- kNN cache and RMT memory limited at test time
    controlled = [m for m in CONTROLS if m in summary and "control" in summary[m]]
    if controlled:
        fig, axes = plt.subplots(1, len(controlled), figsize=(6.5 * len(controlled), 4.5),
                                 squeeze=False, sharey=True)
        for ax, model in zip(axes[0], controlled):
            ctrl = summary[model]["control"]
            for (limit, stats), color, marker in zip(ctrl["zone_mse"].items(),
                                                     [COLORS[model], "#eda100", "#e87ba4"],
                                                     [MARKERS[model], "s", "o"]):
                name = "all frames" if limit == "all" else f"last {limit} frames"
                ax.plot(KS, [stats[str(k)][0] for k in KS], color=color, marker=marker,
                        linewidth=2, markersize=6, label=f"{CONTROLS[model][1]}: {name}")
            if "window" in summary:
                ax.plot(KS, [summary["window"]["zone_mse"][str(k)][0] for k in KS],
                        color=COLORS["window"], linestyle="--", linewidth=1.5,
                        label="Window (no memory)")
            ax.set_xlabel("Frames during which digit B was hidden (k)")
            ax.set_xticks(KS)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            n = len(ctrl["seeds"])
            ax.set_title(f"{LABELS[model]}, limited at test time ({n} run{'s' if n > 1 else ''})",
                         fontsize=11)
        axes[0][0].set_ylabel("MSE on digit B's square (lower is better)")
        plt.tight_layout()
        plt.savefig(fig_dir / "memory_controls.png", dpi=200, bbox_inches="tight")

    # ---- per run: k=0 against k=20
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in [m for m in ORDER if m in summary]:
        pr = summary[model]["per_run"].values()
        ax.scatter([p["zone_mse_k0"] for p in pr],
                   [p["zone_mse_k0"] - p["zone_mse_k20"] for p in pr],
                   color=COLORS[model], marker=MARKERS[model], s=60, label=LABELS[model])
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("MSE on digit B's square at k=0 (how well the run draws a visible B)")
    ax.set_ylabel("k=0 minus k=20 (below 0: seeing B helps)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("One point per run", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "memory_k0_inversion.png", dpi=200, bbox_inches="tight")

    # ---- full-frame quality per step, no occlusion
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, e in summary.items():
        if "ssim_per_step_k0" not in e:
            continue
        mean = np.array([m for m, _ in e["ssim_per_step_k0"]])
        std = np.nan_to_num(np.array([s for _, s in e["ssim_per_step_k0"]]))
        ax.plot(range(1, 11), mean, color=COLORS[model], marker=MARKERS[model],
                linewidth=2, markersize=6, label=LABELS[model])
        ax.fill_between(range(1, 11), mean - std, mean + std, color=COLORS[model], alpha=0.15)
    ax.set_xlabel("Predicted step")
    ax.set_ylabel("Full-frame SSIM (k = 0)")
    ax.set_xticks([1, 5, 10])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Quality per step without occlusion")
    plt.tight_layout()
    plt.savefig(fig_dir / "memory_per_step.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.summary_out} and {fig_dir}/memory_*.png")


if __name__ == "__main__":
    main()
