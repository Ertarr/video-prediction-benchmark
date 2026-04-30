"""
Generate qualitative comparison figures on the Moving MNIST test set.

Produces two versions of each figure:
    - 'absolute' (vmax=1.0): shows true pixel intensities, makes the
      regression-to-the-mean blur of the Compact model very visible
      (the digits become dim because the model averages over plausible
      futures).
    - 'normalized' (vmax = per-image max): rescales each panel to its
      own range, which reveals the residual structure that the Compact
      model still produces under the dim output.

Both views are useful: the absolute one is honest about the SSIM gap,
the normalized one helps see what the Compact model is actually doing
internally. Saves the figures as PNGs. Uses the OTF-trained checkpoints
(Phase 1b).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.models.simvpv2 import SimVPv2
from src.models.compact.compact_predictor import CompactPredictor


SEQUENCE_INDICES = [0, 100, 500]


def load_simvpv2(cfg, device):
    """Load SimVPv2 with the OTF checkpoint."""
    model = SimVPv2(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
    ).to(device)
    ckpt_path = Path(cfg.checkpoint_dir) / "simvpv2-mmnist-otf" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_compact(cfg, device):
    """Load CompactPredictor with the OTF checkpoint."""
    model = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
    ).to(device)
    ckpt_path = Path(cfg.checkpoint_dir) / "compact-mmnist-otf" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, input_frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run a model on one input sequence (no batch). Returns (T, 1, H, W)."""
    x = input_frames.unsqueeze(0).to(device)
    pred = model(x).clamp(0.0, 1.0).squeeze(0)
    return pred.cpu()


def figure_per_sequence(test_ds, simvpv2, compact, device, normalize: bool):
    """
    4-panel side-by-side figure (input / GT / SimVPv2 / Compact) for
    each selected sequence, using either absolute (vmax=1) or
    per-image normalized intensities.
    """
    n_seq = len(SEQUENCE_INDICES)
    n_frames = 10
    n_cols = n_frames * 4
    
    fig = plt.figure(figsize=(n_cols * 0.6, n_seq * 2.4 + 0.6))
    gs = GridSpec(n_seq, 4, figure=fig, wspace=0.3, hspace=0.15)
    
    panel_titles = ["Input (t=1..10)", "Ground truth (t=11..20)",
                    "SimVPv2 prediction", "Compact prediction"]
    
    for row, idx in enumerate(SEQUENCE_INDICES):
        print(f"  processing sequence {idx}...")
        inp, tgt = test_ds[idx]
        pred_simvp = predict(simvpv2, inp, device)
        pred_compact = predict(compact, inp, device)
        
        panels = [inp, tgt, pred_simvp, pred_compact]
        
        for col, frames in enumerate(panels):
            sub_gs = gs[row, col].subgridspec(1, n_frames, wspace=0.05)
            for t in range(n_frames):
                ax = fig.add_subplot(sub_gs[0, t])
                img = frames[t, 0].numpy()
                if normalize:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0 and t == 0:
                    ax.set_title(panel_titles[col], fontsize=10, loc="left", pad=8)
                if col == 0 and t == 0:
                    ax.set_ylabel(f"seq {idx}", fontsize=9)
    
    if normalize:
        suffix = "normalized"
        title = ("Qualitative comparison on Moving MNIST test set "
                 "(per-image normalized intensities, t=11..20 horizon)")
    else:
        suffix = "absolute"
        title = ("Qualitative comparison on Moving MNIST test set "
                 "(absolute intensities, vmax=1.0, t=11..20 horizon)")
    
    fig.suptitle(title, fontsize=11, y=0.995)
    
    if normalize:
        out_path = Path("figures/qualitative_comparison_normalized.png")
    else:
        out_path = Path("figures/qualitative_comparison.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


@torch.no_grad()
def figure_long_horizon_zoom(test_ds, simvpv2, compact, device, normalize: bool):
    """
    Last predicted frame (t=20) for several sequences, where the
    Compact regression-to-the-mean is most visible.
    """
    indices = [0, 50, 100, 200, 500, 800]
    n = len(indices)
    
    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5.0))
    
    for col, idx in enumerate(indices):
        inp, tgt = test_ds[idx]
        pred_simvp = predict(simvpv2, inp, device)
        pred_compact = predict(compact, inp, device)
        
        for row, (frames, label) in enumerate([
            (tgt, "Ground truth"),
            (pred_simvp, "SimVPv2"),
            (pred_compact, "Compact"),
        ]):
            ax = axes[row, col]
            img = frames[-1, 0].numpy()
            if normalize:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f"seq {idx}", fontsize=9)
    
    if normalize:
        title = ("Long-horizon prediction (t=20, last frame, per-image "
                 "normalized): Compact still preserves coarse structure")
        out_path = Path("figures/long_horizon_zoom_normalized.png")
    else:
        title = ("Long-horizon prediction (t=20, last frame, absolute "
                 "intensities): Compact regresses to the mean, output is dim")
        out_path = Path("figures/long_horizon_zoom.png")
    
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    cfg = MMNISTConfig(device="cpu")
    device = torch.device(cfg.device)
    
    test_ds = MovingMNIST(
        cfg.npy_path, split="test",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )
    
    print("Loading SimVPv2 OTF checkpoint...")
    simvpv2 = load_simvpv2(cfg, device)
    
    print("Loading Compact OTF checkpoint...")
    compact = load_compact(cfg, device)
    
    print("\nFigure 1: per-sequence comparison (absolute intensities)")
    figure_per_sequence(test_ds, simvpv2, compact, device, normalize=False)
    
    print("\nFigure 2: per-sequence comparison (normalized intensities)")
    figure_per_sequence(test_ds, simvpv2, compact, device, normalize=True)
    
    print("\nFigure 3: long-horizon zoom (absolute intensities)")
    figure_long_horizon_zoom(test_ds, simvpv2, compact, device, normalize=False)
    
    print("\nFigure 4: long-horizon zoom (normalized intensities)")
    figure_long_horizon_zoom(test_ds, simvpv2, compact, device, normalize=True)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
