"""
Generate qualitative comparison figures on KTH Actions test set.

Sequence selection: high-variance clips (where actions are clearly visible)
are selected automatically to avoid background-only clips.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import KTHConfig
from src.datasets.kth_actions import KTHActions
from src.models.simvpv2 import SimVPv2
from src.models.compact.compact_predictor import CompactPredictor


# Frames to display in figure 1
INPUT_FRAMES_TO_SHOW = [0, 3, 6, 9]
TARGET_FRAMES_TO_SHOW = [10, 14, 18, 22, 26, 29]


def load_model(model_class, ckpt_subdir, cfg, device, **kwargs):
    in_shape = (cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size)
    model = model_class(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=in_shape, **kwargs,
    ).to(device)
    ckpt_path = Path(cfg.checkpoint_dir) / ckpt_subdir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, input_frames, device):
    x = input_frames.unsqueeze(0).to(device)
    pred = model(x).clamp(0.0, 1.0).squeeze(0)
    return pred.cpu()


def find_active_sequences(test_ds, n_select, sample_every=20):
    """Sample some clips, compute temporal variance, return top n_select."""
    print(f"  scanning {len(test_ds) // sample_every} clips for motion...")
    variances = []
    for idx in range(0, len(test_ds), sample_every):
        inp, tgt = test_ds[idx]
        full = torch.cat([inp, tgt], dim=0)
        var = full.var(dim=0).mean().item()
        variances.append((idx, var))
    variances.sort(key=lambda x: -x[1])
    selected = [idx for idx, v in variances[:n_select]]
    print(f"  selected indices: {selected}")
    return selected


def figure_per_sequence(test_ds, simvpv2, compact, device, indices):
    n_seq = len(indices)
    n_input = len(INPUT_FRAMES_TO_SHOW)
    n_target = len(TARGET_FRAMES_TO_SHOW)
    
    fig = plt.figure(figsize=(20, n_seq * 2.6 + 0.5))
    gs = GridSpec(n_seq, 4, figure=fig, wspace=0.25, hspace=0.2)
    
    panel_titles = [
        f"Input ({n_input} of 10 frames)",
        f"Ground truth ({n_target} of 20 frames)",
        "SimVPv2 prediction",
        "Compact prediction",
    ]
    
    for row, idx in enumerate(indices):
        print(f"  processing sequence {idx}...")
        inp, tgt = test_ds[idx]
        pred_simvp = predict(simvpv2, inp, device)
        pred_compact = predict(compact, inp, device)
        
        panels = [
            (inp, INPUT_FRAMES_TO_SHOW),
            (tgt, [t - 10 for t in TARGET_FRAMES_TO_SHOW]),
            (pred_simvp, [t - 10 for t in TARGET_FRAMES_TO_SHOW]),
            (pred_compact, [t - 10 for t in TARGET_FRAMES_TO_SHOW]),
        ]
        
        for col, (frames, frame_indices) in enumerate(panels):
            n = len(frame_indices)
            sub_gs = gs[row, col].subgridspec(1, n, wspace=0.05)
            for i, t in enumerate(frame_indices):
                ax = fig.add_subplot(sub_gs[0, i])
                ax.imshow(frames[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                
                label = f"t={INPUT_FRAMES_TO_SHOW[i]+1}" if col == 0 else f"t={TARGET_FRAMES_TO_SHOW[i]+1}"
                ax.set_xlabel(label, fontsize=8)
                
                if row == 0 and i == 0:
                    ax.set_title(panel_titles[col], fontsize=10, loc="left", pad=8)
                if col == 0 and i == 0:
                    ax.set_ylabel(f"seq {idx}", fontsize=9)
    
    fig.suptitle(
        "Qualitative comparison on KTH Actions test set "
        "(SimVPv2 vs Compact, 10 -> 20 horizon, high-motion sequences)",
        fontsize=11, y=0.995,
    )
    out = Path("figures/qualitative_comparison_kth.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved to {out}")


@torch.no_grad()
def figure_long_horizon_zoom(test_ds, simvpv2, compact, device, indices):
    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5.4))
    
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
            ax.imshow(frames[-1, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f"seq {idx}", fontsize=9)
    
    fig.suptitle(
        "Long-horizon prediction (t=30, last predicted frame): "
        "SimVPv2 vs Compact on KTH (high-motion sequences)",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()
    out = Path("figures/long_horizon_zoom_kth.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved to {out}")


def main():
    cfg = KTHConfig(device="cpu")
    device = torch.device(cfg.device)
    
    test_ds = KTHActions(
        cfg.frames_root, cfg.test_clips,
        input_len=cfg.input_len, target_len=cfg.target_len,
        image_size=cfg.image_size,
    )
    print(f"Test set: {len(test_ds)} clips")
    
    print("\nLoading checkpoints...")
    simvpv2 = load_model(SimVPv2, "simvpv2-kth", cfg, device)
    compact = load_model(
        CompactPredictor, "compact-kth", cfg, device,
        latent_dim=64, translator_hidden=128, n_layers=2,
    )
    
    print("\nFinding sequences with most motion...")
    active_indices = find_active_sequences(test_ds, n_select=6)
    
    print("\nFigure 1: per-sequence comparison (top 3 most active)")
    figure_per_sequence(test_ds, simvpv2, compact, device, active_indices[:3])
    
    print("\nFigure 2: long-horizon zoom (6 active sequences)")
    figure_long_horizon_zoom(test_ds, simvpv2, compact, device, active_indices[:6])
    
    print("\nDone.")


if __name__ == "__main__":
    main()
