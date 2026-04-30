"""Evaluate SimVPv2 on KTH test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import KTHConfig
from src.datasets.kth_actions import KTHActions
from src.models.simvpv2 import SimVPv2
from src.evaluation.metrics import evaluate


def main():
    cfg = KTHConfig(batch_size=4, num_workers=0, device="cpu")
    
    test_ds = KTHActions(
        cfg.frames_root, cfg.test_clips,
        input_len=cfg.input_len, target_len=cfg.target_len,
        image_size=cfg.image_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )
    
    device = torch.device(cfg.device)
    model = SimVPv2(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
    ).to(device)
    
    ckpt_path = Path(cfg.checkpoint_dir) / "simvpv2-kth" / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with val_loss={checkpoint['val_loss']:.5f}")
    
    print("Running evaluation on test set...")
    results = evaluate(model, test_loader, device)
    
    print("\n=== Per-step results on test set ===")
    print(f"{'Step':<8}{'MSE':<14}{'SSIM':<12}")
    print("-" * 34)
    for t in range(len(results["mse_per_step"])):
        print(f"{t+1:<8}{results['mse_per_step'][t]:<14.5f}"
              f"{results['ssim_per_step'][t]:<12.4f}")
    
    print("\n=== Key horizons (1/10/20-step) ===")
    for h in [1, 10, 20]:
        idx = h - 1
        print(f"  t={h:2d}-step: MSE={results['mse_per_step'][idx]:.5f}  "
              f"SSIM={results['ssim_per_step'][idx]:.4f}")


if __name__ == "__main__":
    main()
