"""
KTH: error and sharpness only where something moves (CPU, existing
checkpoints). The background is static, so full-frame metrics mostly
score it (copying the last frame gets SSIM 0.81).

Motion mask: pixels where the true frame differs from the last observed
frame by more than 0.1. Per predicted step:
    - MSE inside the mask
    - sharpness: mean gradient magnitude of the prediction in the mask,
      divided by the same for the true frame (1 = as sharp as the truth)
    - share of pixels in the mask
The test clips are also split in three groups by how much they move.

Usage:
    python scripts/kth_motion_eval.py --kth-root path/to/kth --ckpt-root checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.kth_actions import KTHActions
from src.models.simvpv2 import SimVPv2
from src.models.compact.compact_predictor import CompactPredictor
from scripts.reevaluate_checkpoints import CopyLastFrame


THRESHOLD = 0.1
KTH_SHAPE = (10, 1, 128, 128)
MODELS = {
    "copy-last-frame": None,
    "simvpv2-kth": lambda: SimVPv2(10, 20, KTH_SHAPE),
    "compact-kth": lambda: CompactPredictor(10, 20, KTH_SHAPE, latent_dim=64,
                                            translator_hidden=128, n_layers=2),
    "compact-kth-n3": lambda: CompactPredictor(10, 20, KTH_SHAPE, latent_dim=64,
                                               translator_hidden=128, n_layers=3),
}


def grad_mag(x):
    """Mean of |dx| + |dy| per pixel, same shape as x (last two dims)."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    gy[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return gx + gy


def moving_region_metrics(model, loader, device=torch.device("cpu")):
    """Moving-region MSE and relative sharpness per step (no motion groups)."""
    T = None
    model.eval()
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp).clamp(0, 1)
            if T is None:
                T = tgt.shape[1]
                se, n_mask, g_pred, g_true = (np.zeros(T) for _ in range(4))
            mask = ((tgt - inp[:, -1:]).abs() > THRESHOLD).float()
            se += ((pred - tgt) ** 2 * mask).sum(dim=(0, 2, 3, 4)).cpu().numpy()
            n_mask += mask.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            g_pred += (grad_mag(pred) * mask).sum(dim=(0, 2, 3, 4)).cpu().numpy()
            g_true += (grad_mag(tgt) * mask).sum(dim=(0, 2, 3, 4)).cpu().numpy()
    mse = se / np.maximum(n_mask, 1)
    return {"moving_mse_per_step": mse.tolist(), "moving_mse_avg": float(mse.mean()),
            "sharpness_per_step": (g_pred / np.maximum(g_true, 1e-12)).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kth-root", required=True)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="results/kth_motion_eval.json")
    args = parser.parse_args()

    ds = KTHActions(Path(args.kth_root) / "frames", Path(args.kth_root) / "test_clips.txt",
                    input_len=10, target_len=20, image_size=128)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # motion of each clip, from its input frames only
    motion = []
    for inp, _ in loader:
        motion.append((inp[:, 1:] - inp[:, :-1]).abs().mean(dim=(1, 2, 3, 4)))
    motion = torch.cat(motion).numpy()
    edges = np.quantile(motion, [1 / 3, 2 / 3])
    group = np.digitize(motion, edges)                 # 0 = least motion, 2 = most
    print(f"{len(ds)} clips, motion terciles at {edges.round(4).tolist()}")

    results = {"n_clips": len(ds), "threshold": THRESHOLD,
               "motion_tercile_edges": edges.tolist()}
    for name, build in MODELS.items():
        print(f"Evaluating {name} ...", flush=True)
        if build is None:
            model = CopyLastFrame(10, 20)
        else:
            model = build()
            ckpt = torch.load(Path(args.ckpt_root) / name / "best.pt",
                              map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        se = np.zeros(20)                 # summed squared error in the mask, per step
        n_mask = np.zeros(20)             # masked pixels per step
        n_pix = np.zeros(20)
        g_pred = np.zeros(20)
        g_true = np.zeros(20)
        se_g = np.zeros((3, 20))
        n_g = np.zeros((3, 20))
        i0 = 0
        with torch.no_grad():
            for inp, tgt in loader:
                pred = model(inp).clamp(0, 1)
                last = inp[:, -1:]                                # (B, 1, 1, H, W)
                mask = ((tgt - last).abs() > THRESHOLD).float()   # (B, 20, 1, H, W)
                err = (pred - tgt) ** 2 * mask
                se += err.sum(dim=(0, 2, 3, 4)).numpy()
                n_mask += mask.sum(dim=(0, 2, 3, 4)).numpy()
                n_pix += np.full(20, mask[:, 0].numel())
                g_pred += (grad_mag(pred) * mask).sum(dim=(0, 2, 3, 4)).numpy()
                g_true += (grad_mag(tgt) * mask).sum(dim=(0, 2, 3, 4)).numpy()
                g = group[i0:i0 + inp.shape[0]]
                for k in range(3):
                    sel = torch.from_numpy(g == k)
                    if sel.any():
                        se_g[k] += err[sel].sum(dim=(0, 2, 3, 4)).numpy()
                        n_g[k] += mask[sel].sum(dim=(0, 2, 3, 4)).numpy()
                i0 += inp.shape[0]

        mse_moving = se / np.maximum(n_mask, 1)
        entry = {
            "moving_mse_per_step": mse_moving.tolist(),
            "moving_mse_avg": float(mse_moving.mean()),
            "sharpness_per_step": (g_pred / np.maximum(g_true, 1e-12)).tolist(),
            "moving_share_per_step": (n_mask / n_pix).tolist(),
            "moving_mse_by_motion_tercile": {
                str(k): (se_g[k] / np.maximum(n_g[k], 1)).tolist() for k in range(3)
            },
        }
        results[name] = entry
        print(f"  moving MSE t=1 {mse_moving[0]:.4f}  t=10 {mse_moving[9]:.4f}  "
              f"t=20 {mse_moving[19]:.4f}  | sharpness t=1 "
              f"{entry['sharpness_per_step'][0]:.2f}  t=20 {entry['sharpness_per_step'][19]:.2f}"
              f"  | moving share t=20 {entry['moving_share_per_step'][19]:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
