"""
Phase 6: does Compact's blur depend on digit speed?

With n_layers=2, the one 3x3 conv of the latent moves information by
about 4 px, while a digit can move ~40 px over 10 frames. If that short
reach causes the blur, the SSIM gap SimVPv2 - Compact n2 at t=10 should
be at least 2x larger at 4 px/frame than at 1 px/frame.

CPU, existing checkpoints.

Usage:
    python scripts/phase6_speed_test.py --mnist-root data/mnist --ckpt-root checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt

from src.models.simvpv2 import SimVPv2
from scripts.phase6_train import build_compact, eval_speed_groups, SPEEDS
from scripts.reevaluate_checkpoints import CopyLastFrame


MODELS = {
    # name in the figure: (checkpoint folder, builder)
    "SimVPv2": ("simvpv2-mmnist-otf", lambda: SimVPv2(10, 10, (10, 1, 64, 64))),
    "Compact n_layers=3": ("compact-mmnist-otf-n3", lambda: build_compact("n3")),
    "Compact n_layers=2": ("compact-mmnist-otf", lambda: build_compact("n2")),
    "Copy last frame": (None, lambda: CopyLastFrame(10, 10)),
}
COLORS = {
    "SimVPv2": "#2a78d6",
    "Compact n_layers=3": "#eb6834",
    "Compact n_layers=2": "#1baf7a",
    "Copy last frame": "gray",
}
MARKERS = {"SimVPv2": "o", "Compact n_layers=3": "s",
           "Compact n_layers=2": "^", "Copy last frame": None}


def plot(results, out_path):
    fig, axes = plt.subplots(2, len(SPEEDS), figsize=(13, 7), sharey="row")
    steps = range(1, 11)
    for col, speed in enumerate(SPEEDS):
        for row, metric in enumerate(["ssim", "mse"]):
            ax = axes[row, col]
            for name, per_speed in results.items():
                values = per_speed[str(speed)][f"{metric}_per_step"]
                ax.plot(steps, values, color=COLORS[name], marker=MARKERS[name],
                        markersize=5, linewidth=2,
                        linestyle="--" if name == "Copy last frame" else "-",
                        label=name)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([1, 5, 10])
            if row == 0:
                ax.set_title(f"{speed:g} px / frame", fontsize=12)
            else:
                ax.set_xlabel("Predicted step")
            if col == 0:
                ax.set_ylabel("SSIM" if metric == "ssim" else "MSE")
    axes[0, 0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Moving MNIST with fixed digit speed (1000 test sequences per speed)",
                 fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", default="data/mnist")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out", default="results/phase6_speed_test.json")
    parser.add_argument("--fig", default="figures/phase6_speed_test.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    results = {}
    for name, (ckpt_dir, build) in MODELS.items():
        print(f"Evaluating {name} ...", flush=True)
        model = build()
        if ckpt_dir is not None:
            ckpt = torch.load(Path(args.ckpt_root) / ckpt_dir / "best.pt",
                              map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        results[name] = eval_speed_groups(model, args.mnist_root, device)
        for speed in SPEEDS:
            r = results[name][str(speed)]
            print(f"  {speed:g} px/frame: SSIM t=10 {r['ssim_per_step'][9]:.3f}  "
                  f"SSIM avg {r['ssim_avg']:.3f}  MSE avg {r['mse_avg']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")

    # SSIM gap per speed at t=10
    gap = {s: results["SimVPv2"][str(s)]["ssim_per_step"][9]
              - results["Compact n_layers=2"][str(s)]["ssim_per_step"][9] for s in SPEEDS}
    gain = {s: results["Compact n_layers=3"][str(s)]["ssim_per_step"][9]
               - results["Compact n_layers=2"][str(s)]["ssim_per_step"][9] for s in SPEEDS}
    print("\nSSIM gap SimVPv2 - Compact n2 at t=10:",
          {s: round(g, 3) for s, g in gap.items()})
    print("SSIM gain Compact n3 - n2 at t=10:   ",
          {s: round(g, 3) for s, g in gain.items()})
    ratio = gap[SPEEDS[-1]] / max(gap[SPEEDS[0]], 1e-6)
    print(f"Fast / slow gap ratio: {ratio:.2f} "
          f"({'consistent with' if ratio >= 2 else 'rejects'} the hypothesis)")

    plot(results, Path(args.fig))


if __name__ == "__main__":
    main()
