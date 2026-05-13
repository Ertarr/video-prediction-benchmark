"""Eval JEPA optimal + Compact n3 et affiche les métriques côte à côte."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.models.compact.compact_predictor import CompactPredictor
from src.models.compact_jepa.jepa_predictor import JEPAPredictor
from src.evaluation.metrics import evaluate


def eval_compact_n3():
    cfg = MMNISTConfig(batch_size=16, num_workers=0, device="cuda")
    test_ds = MovingMNIST(cfg.npy_path, split="test",
                          input_len=cfg.input_len, target_len=cfg.target_len)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    
    print("=" * 70)
    print("Compact n_layers=3 (200 epochs)")
    print("=" * 70)
    
    model = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64, translator_hidden=256, n_layers=3,
    )
    ckpt = torch.load('checkpoints/compact-mmnist-otf-n3/best.pt', map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.5f}")
    
    res = evaluate(model, test_loader, device)
    mse = res['mse_per_step']
    ssim = res['ssim_per_step']
    if hasattr(mse, 'tolist'): mse = mse.tolist()
    if hasattr(ssim, 'tolist'): ssim = ssim.tolist()
    
    print(f"\n{'Step':<6}{'MSE':<14}{'SSIM':<10}")
    for t in range(len(mse)):
        print(f"{t+1:<6}{mse[t]:<14.5f}{ssim[t]:<10.4f}")
    
    avg_mse = sum(mse) / len(mse)
    avg_ssim = sum(ssim) / len(ssim)
    print(f"\nAvg MSE: {avg_mse:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"MSE t=10: {mse[9]:.4f}")
    print(f"SSIM t=10: {ssim[9]:.4f}")
    return mse, ssim


def eval_jepa_best():
    cfg = MMNISTConfig(batch_size=16, num_workers=0, device="cuda")
    test_ds = MovingMNIST(cfg.npy_path, split="test",
                          input_len=cfg.input_len, target_len=cfg.target_len)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    
    print("\n" + "=" * 70)
    print("JEPA optimal (var_coef=0, cov_coef=0, ema_momentum=0.999, 200 epochs)")
    print("=" * 70)
    
    model = JEPAPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64, translator_hidden=256, n_layers=2,
        ema_momentum=0.999, use_decoder=True,
    )
    ckpt = torch.load('checkpoints/jepa-mmnist-otf-best/best.pt', map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.5f}")
    
    sum_mse = torch.zeros(10)
    sum_ssim = torch.zeros(10)
    n = 0
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp, future_frames=tgt)
            pred_pix = out['predicted_pixels'].clamp(0, 1)
            B, T = pred_pix.shape[:2]
            mse_batch = ((pred_pix - tgt) ** 2).mean(dim=(2,3,4))
            sum_mse += mse_batch.sum(dim=0).cpu()
            for b in range(B):
                for t in range(T):
                    p = pred_pix[b, t, 0].cpu().numpy()
                    g = tgt[b, t, 0].cpu().numpy()
                    sum_ssim[t] += ssim_fn(g, p, data_range=1.0)
            n += B
    
    mse = (sum_mse / n).tolist()
    ssim = (sum_ssim / n).tolist()
    
    print(f"\n{'Step':<6}{'MSE':<14}{'SSIM':<10}")
    for t in range(len(mse)):
        print(f"{t+1:<6}{mse[t]:<14.5f}{ssim[t]:<10.4f}")
    
    avg_mse = sum(mse) / len(mse)
    avg_ssim = sum(ssim) / len(ssim)
    print(f"\nAvg MSE: {avg_mse:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"MSE t=10: {mse[9]:.4f}")
    print(f"SSIM t=10: {ssim[9]:.4f}")
    return mse, ssim


def main():
    mse_compact, ssim_compact = eval_compact_n3()
    mse_jepa, ssim_jepa = eval_jepa_best()
    
    # Summary final
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL — toutes les variantes du projet")
    print("=" * 70)
    print(f"{'Model':<40}{'MSE avg':<12}{'SSIM avg':<12}")
    print("-" * 64)
    print(f"{'SimVPv2 (4.72M, ref)':<40}{0.0168:<12.4f}{0.84:<12.4f}")
    print(f"{'Compact main (0.98M, n_layers=2)':<40}{0.0316:<12.4f}{0.46:<12.4f}")
    print(f"{'Compact n_layers=3 (0.99M, 200ep)':<40}{sum(mse_compact)/len(mse_compact):<12.4f}{sum(ssim_compact)/len(ssim_compact):<12.4f}")
    print(f"{'JEPA defaults (0.98M, 200ep)':<40}{0.0356:<12.4f}{0.30:<12.4f}")
    print(f"{'JEPA optimal (0.98M, 200ep)':<40}{sum(mse_jepa)/len(mse_jepa):<12.4f}{sum(ssim_jepa)/len(ssim_jepa):<12.4f}")


if __name__ == "__main__":
    main()
