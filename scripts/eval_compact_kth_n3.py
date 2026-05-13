"""Eval Compact n_layers=3 on KTH test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import KTHConfig
from src.datasets.kth_actions import KTHActions
from src.models.compact.compact_predictor import CompactPredictor
from src.evaluation.metrics import evaluate


def main():
    cfg = KTHConfig()

    test_ds = KTHActions(
        cfg.frames_root, cfg.test_clips,
        input_len=cfg.input_len, target_len=cfg.target_len,
        image_size=cfg.image_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=0,
    )

    device = torch.device("cuda")
    model = CompactPredictor(
        input_len=cfg.input_len,
        target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=128,
        n_layers=3,
    ).to(device)

    ckpt = torch.load('checkpoints/compact-kth-n3/best.pt', map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.5f}")

    res = evaluate(model, test_loader, device)
    mse = res['mse_per_step']
    ssim = res['ssim_per_step']
    if hasattr(mse, 'tolist'): mse = mse.tolist()
    if hasattr(ssim, 'tolist'): ssim = ssim.tolist()

    print(f"\n{'Step':<6}{'MSE':<14}{'SSIM':<10}")
    for t in range(len(mse)):
        print(f"{t+1:<6}{mse[t]:<14.5f}{ssim[t]:<10.4f}")

    print("\n=== Comparaison vs Compact n_layers=2 (baseline KTH Phase 2) ===")
    # Baselines from RESULTS.md
    print(f"{'Horizon':<10}{'n_layers=2 (baseline)':<32}{'n_layers=3 (this run)'}")
    print(f"{'t=1':<10}{'MSE 0.00093 / SSIM 0.86':<32}{f'MSE {mse[0]:.5f} / SSIM {ssim[0]:.4f}'}")
    print(f"{'t=10':<10}{'MSE 0.00324 / SSIM 0.79':<32}{f'MSE {mse[9]:.5f} / SSIM {ssim[9]:.4f}'}")
    print(f"{'t=20':<10}{'MSE 0.00394 / SSIM 0.77':<32}{f'MSE {mse[19]:.5f} / SSIM {ssim[19]:.4f}'}")
    print(f"\nAvg MSE: {sum(mse)/len(mse):.5f}")
    print(f"Avg SSIM: {sum(ssim)/len(ssim):.4f}")


if __name__ == "__main__":
    main()
