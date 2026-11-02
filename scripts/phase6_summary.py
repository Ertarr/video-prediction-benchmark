"""
Phase 6 summary: mean +/- std over runs for each variant, plus a figure.

Reads every JSON in results/phase6/ (written by scripts/phase6_train.py
on Colab). The Phase 4 ablations already contain runs with the exact
Phase 6 setup (100 epochs, same data and loop): three runs of the default
n_layers=2 config and one n_layers=3 run. With --import-phase4 they are
evaluated on the speed sets and saved next to the Phase 6 runs, so they
count as extra runs (their seed was not controlled, see RESULTS.md).

The KTH runs (scripts/phase6_train_kth.py) get their own table, next to
the Phase 2 runs (results/reeval_kth.json, results/kth_motion_eval.json).

Usage:
    python scripts/phase6_summary.py --import-phase4 \
        --mnist-root data/mnist --ckpt-root checkpoints     # once, on CPU
    python scripts/phase6_summary.py                         # after Colab
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts.phase6_train import build_compact, eval_speed_groups, SPEEDS


PHASE4_RUNS = {
    "compact-abl-latent_dim=64": "n2",
    "compact-abl-translator_hidden=256": "n2",
    "compact-abl-n_layers=2": "n2",
    "compact-abl-n_layers=3": "n3",
}
ORDER = ["n2", "n3", "skip", "l1", "dil2"]
LABELS = {
    "n2": "n_layers=2 (baseline)",
    "n3": "n_layers=3",
    "skip": "n_layers=2 + skip",
    "l1": "n_layers=2, L1 loss",
    "dil2": "n_layers=2, dilation 2",
}
COLORS = {"n2": "#2a78d6", "n3": "#eb6834", "skip": "#1baf7a",
          "l1": "#eda100", "dil2": "#e87ba4"}
MARKERS = {"n2": "o", "n3": "s", "skip": "^", "l1": "D", "dil2": "v"}


def import_phase4(ckpt_root, mnist_root, out_dir):
    for name, variant in PHASE4_RUNS.items():
        print(f"Speed test for {name} ...", flush=True)
        abl = json.loads((Path("ablations/compact") / f"{name}.json").read_text())
        model = build_compact(variant)
        ckpt = torch.load(Path(ckpt_root) / name / "best.pt",
                          map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        mse, ssim = abl["test_mse_per_step"], abl["test_ssim_per_step"]
        summary = {
            "run_name": f"phase4-{name}",
            "variant": variant,
            "seed": None,
            "source": "Phase 4 ablation run (seed not controlled)",
            "n_epochs": 100,
            "n_params": abl["n_params"],
            "best_val_loss": abl["best_val_loss"],
            "best_epoch": abl["best_epoch"],
            "toronto_test": {
                "mse_per_step": mse, "ssim_per_step": ssim,
                "mse_avg": sum(mse) / len(mse), "ssim_avg": sum(ssim) / len(ssim),
            },
            "speed_test": eval_speed_groups(model, mnist_root, torch.device("cpu")),
        }
        out = Path(out_dir) / f"phase4-{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))


def mean_std(values):
    values = np.array(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else float("nan")
    return float(values.mean()), float(std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase6")
    parser.add_argument("--import-phase4", action="store_true")
    parser.add_argument("--mnist-root", default="data/mnist")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out", default="results/phase6_summary.json")
    parser.add_argument("--fig", default="figures/phase6_variants.png")
    args = parser.parse_args()

    if args.import_phase4:
        import_phase4(args.ckpt_root, args.mnist_root, args.results_dir)

    runs = {}
    for path in sorted(Path(args.results_dir).glob("*.json")):
        r = json.loads(path.read_text())
        if r.get("dataset") != "kth":
            runs.setdefault(r["variant"], []).append(r)

    summary = {}
    print("\n| Variant | Runs | Test MSE | Test SSIM | "
          + " | ".join(f"SSIM t=10, {s:g} px/fr" for s in SPEEDS) + " |")
    print("|---" * (4 + len(SPEEDS)) + "|")
    for variant in [v for v in ORDER if v in runs]:
        rs = runs[variant]
        row = {
            "n_runs": len(rs),
            "runs": [r["run_name"] for r in rs],
            "test_mse_avg": mean_std([r["toronto_test"]["mse_avg"] for r in rs]),
            "test_ssim_avg": mean_std([r["toronto_test"]["ssim_avg"] for r in rs]),
            "test_ssim_per_step": [mean_std([r["toronto_test"]["ssim_per_step"][t] for r in rs])
                                   for t in range(10)],
            "speed_ssim_t10": {str(s): mean_std([r["speed_test"][str(s)]["ssim_per_step"][9]
                                                  for r in rs]) for s in SPEEDS},
        }
        summary[variant] = row
        fmt = lambda ms, d: f"{ms[0]:.{d}f} +/- {ms[1]:.{d}f}"
        print(f"| {LABELS[variant]} | {len(rs)} | {fmt(row['test_mse_avg'], 4)} | "
              f"{fmt(row['test_ssim_avg'], 3)} | "
              + " | ".join(fmt(row["speed_ssim_t10"][str(s)], 3) for s in SPEEDS) + " |")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {args.out}")

    # Per-step SSIM on the Toronto test set, mean +/- std over runs
    fig, ax = plt.subplots(figsize=(10, 4.5))
    steps = np.arange(1, 11)
    for variant, row in summary.items():
        mean = np.array([m for m, _ in row["test_ssim_per_step"]])
        std = np.nan_to_num(np.array([s for _, s in row["test_ssim_per_step"]]))
        ax.plot(steps, mean, color=COLORS[variant], marker=MARKERS[variant],
                markersize=5, linewidth=2,
                label=f"{LABELS[variant]} ({row['n_runs']} run{'s' if row['n_runs'] > 1 else ''})")
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[variant], alpha=0.15)
    reeval = json.loads(Path("results/reeval_mmnist.json").read_text())
    for key, label, style in [("copy-last-frame", "Copy last frame (no model)", "--"),
                              ("black-frame", "Black frame (no model)", ":")]:
        if key in reeval:
            ax.plot(steps, reeval[key]["test_ssim_per_step"], color="gray",
                    linestyle=style, linewidth=1.5, label=label)
    ax.set_xlabel("Predicted step")
    ax.set_ylabel("Test SSIM")
    ax.set_xticks([1, 5, 10])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("Phase 6 variants of Compact, 100 epochs (band = +/- 1 std)")
    plt.tight_layout()
    Path(args.fig).parent.mkdir(exist_ok=True)
    plt.savefig(args.fig, dpi=200, bbox_inches="tight")
    print(f"Saved {args.fig}")
    kth_table(args.results_dir)


def kth_table(results_dir):
    """Phase 6 on KTH: full frame and moving pixels, averaged over the 20 steps."""
    rows = []
    for path in sorted(Path(results_dir).glob("phase6-kth-*.json")):
        r = json.loads(path.read_text())
        t, m = r["kth_test"], r["kth_moving"]
        rows.append((r["run_name"], t["mse_avg"], t["ssim_avg"], m["moving_mse_avg"],
                     m["sharpness_per_step"][-1], r["best_epoch"] + 1))
    if not rows:
        return
    full = json.loads(Path("results/reeval_kth.json").read_text())
    moving = json.loads(Path("results/kth_motion_eval.json").read_text())
    for name in ["compact-kth", "compact-kth-n3", "simvpv2-kth", "copy-last-frame"]:
        f, m = full[name], moving[name]
        rows.append((f"Phase 2: {name}", np.mean(f["test_mse_per_step"]),
                     np.mean(f["test_ssim_per_step"]), m["moving_mse_avg"],
                     m["sharpness_per_step"][-1], None))
    print("\nKTH, averages over the 20 predicted steps\n")
    print("| Run | Full-frame MSE | Full-frame SSIM | Moving-pixel MSE | Sharpness t=20 | Best epoch |")
    print("|---|---|---|---|---|---|")
    for name, mse, ssim, mov, sharp, ep in rows:
        print(f"| {name} | {mse:.5f} | {ssim:.3f} | {mov:.4f} | {sharp:.2f} | {ep or '-'} |")


if __name__ == "__main__":
    main()
