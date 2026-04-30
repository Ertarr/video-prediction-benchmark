"""
Generic evaluation runner for video prediction models on Moving MNIST.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.models.base import VideoPredictor
from src.evaluation.metrics import evaluate


def run_evaluation_mmnist(
    model: VideoPredictor,
    cfg: MMNISTConfig,
    checkpoint_path: Path,
) -> dict:
    """
    Loads a checkpoint, evaluates the model on the test set, and 
    prints per-step + key-horizon results.
    """
    test_ds = MovingMNIST(
        cfg.npy_path, split="test",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    device = torch.device(cfg.device)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with val_loss={checkpoint['val_loss']:.5f}")

    results = evaluate(model, test_loader, device)

    # Per-step display
    print("\n=== Per-step results on test set ===")
    print(f"{'Step':<8}{'MSE':<14}{'SSIM':<12}")
    print("-" * 34)
    for t in range(len(results["mse_per_step"])):
        print(f"{t+1:<8}{results['mse_per_step'][t]:<14.5f}"
              f"{results['ssim_per_step'][t]:<12.4f}")

    # Key horizons
    print("\n=== Key horizons (1/5/10-step) ===")
    for h in [1, 5, 10]:
        idx = h - 1
        print(f"  t={h:2d}-step: MSE={results['mse_per_step'][idx]:.5f}  "
              f"SSIM={results['ssim_per_step'][idx]:.4f}")
    
    return results