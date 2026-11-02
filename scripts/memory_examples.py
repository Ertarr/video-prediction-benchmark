"""
What the memory models draw on one test sequence, with digit B never
hidden (k=0) and hidden for the last 20 observed frames (k=20). The red
square is the scored zone. One figure per seed.

The sequence is the first one where the two digits don't overlap in the
frames to predict (rare: 5 of the 1000 test sequences).

Usage:
    python scripts/memory_examples.py --mnist-root data/mnist --ckpt-root checkpoints --seed 1
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.datasets.moving_mnist_occlusion import MovingMNISTOcclusion
from scripts.eval_memory import build_memory_model, TEST_SEED, DIGIT


MODELS = ["window", "rmt", "knn"]
LABELS = {"window": "Window", "rmt": "RMT", "knn": "kNN"}
OBSERVED = [40, 49]        # first and last frame of the window (0-indexed)
STEPS = [0, 2, 4, 9]       # predicted steps shown: +1, +3, +5, +10
KS = [0, 20]


def pick_sequence(ds, n_try=1000):
    """First sequence where the two digits never overlap in the frames to predict."""
    for idx in range(n_try):
        pos = ds[idx][2]["positions"].numpy()[ds.n_observed:]
        if np.abs(pos[:, 0] - pos[:, 1]).max(axis=1).min() >= DIGIT:
            return idx
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", default="data/mnist")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    args.out = args.out or f"figures/memory_examples_seed{args.seed}.png"

    models = {}
    for name in MODELS:
        model = build_memory_model(name)
        ckpt = torch.load(Path(args.ckpt_root) / f"memory-{name}-seed{args.seed}" / "best.pt",
                          map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        models[name] = model.eval()

    columns = [("obs", t) for t in OBSERVED] + [("pred", s) for s in STEPS]
    n_cols = len(columns)
    fig, axes = plt.subplots(
        1 + len(MODELS), 2 * n_cols + 1, figsize=(2 * n_cols * 1.3 + 0.5, (1 + len(MODELS)) * 1.4),
        gridspec_kw={"width_ratios": [1] * n_cols + [0.3] + [1] * n_cols})
    for ax in axes[:, n_cols]:
        ax.axis("off")

    idx = None
    for block, k in enumerate(KS):
        ds = MovingMNISTOcclusion(args.mnist_root, n_samples=1000, k=k, seed=TEST_SEED,
                                  mnist_train=False, return_meta=True)
        idx = pick_sequence(ds) if idx is None else idx
        inp, tgt, meta = ds[idx]
        pos = meta["positions"].numpy()
        with torch.no_grad():
            preds = {name: m(inp[None]).clamp(0, 1)[0] for name, m in models.items()}

        off = block * (n_cols + 1)
        for row in range(1 + len(MODELS)):
            for c, (kind, t) in enumerate(columns):
                ax = axes[row, off + c]
                ax.set_xticks([])
                ax.set_yticks([])
                if kind == "obs":
                    if row > 0:
                        ax.axis("off")
                        continue
                    img, frame = inp[t, 0], t
                else:
                    img = tgt[t, 0] if row == 0 else preds[MODELS[row - 1]][t, 0]
                    frame = ds.n_observed + t
                ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
                x, y = pos[frame, 1]
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), DIGIT, DIGIT, fill=False,
                                       edgecolor="red", linewidth=0.8))
                if row == 0:
                    ax.set_title(f"frame {frame + 1}" if kind == "obs" else f"+{t + 1}",
                                 fontsize=8)
        axes[0, off].set_ylabel("Input /\ntruth", fontsize=8)
        for row, name in enumerate(MODELS, start=1):
            axes[row, off + len(OBSERVED)].set_ylabel(LABELS[name], fontsize=8)

    plt.subplots_adjust(wspace=0.08, hspace=0.12, top=0.80)
    for block, title in enumerate(["k = 0: digit B never hidden",
                                   "k = 20: B hidden in the last 20 observed frames"]):
        first, last = axes[0, block * (n_cols + 1)], axes[0, block * (n_cols + 1) + n_cols - 1]
        x = (first.get_position().x0 + last.get_position().x1) / 2
        fig.text(x, 0.93, title, ha="center", fontsize=10)
    fig.suptitle(f"Test sequence {idx}, seed {args.seed} of each model "
                 "(red square: digit B, the zone that is scored)", fontsize=10, y=1.0)
    Path(args.out).parent.mkdir(exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out} (sequence {idx})")


if __name__ == "__main__":
    main()
