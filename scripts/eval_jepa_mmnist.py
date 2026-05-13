"""
Evaluate JEPAPredictor on Moving MNIST test set.

Reports both:
    - Latent-space prediction error (smooth L1) - the JEPA training metric
    - Pixel-space MSE/SSIM - via the auxiliary decoder, for comparison
      with Compact (Phase 1, 1b) and SimVPv2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn
import numpy as np

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.models.compact_jepa.jepa_predictor import JEPAPredictor


@torch.no_grad()
def evaluate_jepa(model, loader, device):
    model.eval()
    
    n_steps = None
    sum_mse_per_step = None
    sum_ssim_per_step = None
    sum_latent_l1_per_step = None
    n_samples = 0
    
    for input_frames, future_frames in loader:
        input_frames = input_frames.to(device)
        future_frames = future_frames.to(device)
        
        out = model(input_frames, future_frames=future_frames)
        pred_latents = out['predicted_latents']
        target_latents = out['target_latents']
        pred_pixels = out['predicted_pixels'].clamp(0.0, 1.0)
        
        B, T = pred_pixels.shape[:2]
        if n_steps is None:
            n_steps = T
            sum_mse_per_step = torch.zeros(T)
            sum_ssim_per_step = torch.zeros(T)
            sum_latent_l1_per_step = torch.zeros(T)
        
        # Pixel MSE per step
        mse = ((pred_pixels - future_frames) ** 2).mean(dim=(2, 3, 4))
        sum_mse_per_step += mse.sum(dim=0).cpu()
        
        # Latent L1 per step
        l_l1 = F.smooth_l1_loss(
            pred_latents, target_latents, reduction='none',
        ).mean(dim=(2, 3, 4))
        sum_latent_l1_per_step += l_l1.sum(dim=0).cpu()
        
        # SSIM per step
        for b in range(B):
            for t in range(T):
                p = pred_pixels[b, t, 0].cpu().numpy()
                g = future_frames[b, t, 0].cpu().numpy()
                s = ssim_fn(g, p, data_range=1.0)
                sum_ssim_per_step[t] += s
        
        n_samples += B
    
    return {
        'mse_per_step': (sum_mse_per_step / n_samples).numpy().tolist(),
        'ssim_per_step': (sum_ssim_per_step / n_samples).numpy().tolist(),
        'latent_l1_per_step': (sum_latent_l1_per_step / n_samples).numpy().tolist(),
    }


def main():
    cfg = MMNISTConfig(batch_size=16, num_workers=0, device="cpu")
    
    test_ds = MovingMNIST(
        cfg.npy_path, split="test",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    device = torch.device(cfg.device)
    model = JEPAPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
        ema_momentum=0.996,
        use_decoder=True,
    ).to(device)
    
    ckpt_path = Path(cfg.checkpoint_dir) / "jepa-mmnist-otf" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"with val_loss={ckpt['val_loss']:.5f}")
    
    print("Running evaluation on test set...")
    results = evaluate_jepa(model, test_loader, device)
    
    print("\n=== Per-step results ===")
    print(f"{'Step':<6}{'MSE':<14}{'SSIM':<12}{'Latent L1':<14}")
    print("-" * 50)
    for t in range(len(results['mse_per_step'])):
        print(
            f"{t+1:<6}"
            f"{results['mse_per_step'][t]:<14.5f}"
            f"{results['ssim_per_step'][t]:<12.4f}"
            f"{results['latent_l1_per_step'][t]:<14.5f}"
        )
    
    print("\n=== Key horizons ===")
    for h in [1, 5, 10]:
        idx = h - 1
        print(
            f"  t={h:2d}: MSE={results['mse_per_step'][idx]:.5f}  "
            f"SSIM={results['ssim_per_step'][idx]:.4f}  "
            f"latent_L1={results['latent_l1_per_step'][idx]:.5f}"
        )


if __name__ == "__main__":
    main()
