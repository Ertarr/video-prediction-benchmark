"""
Phase 1/2 ablations on the Compact model (Moving MNIST OTF).

Sweeps three axes to build a Pareto curve and isolate the impact of
each design choice:
    - latent_dim:           [16, 32, 64, 128]
    - translator_hidden:    [64, 128, 256, 512]
    - n_layers:             [2, 3]   (3 = 8x downsampling, latent 8x8 at 64x64)

Each run uses the same training hyperparameters as the main Phase 1b
experiments (200 epochs, batch=32, lr=1e-3, OTF generation). Results
are stored as JSON files under `ablations/compact/` so they can be
loaded by `generate_pareto_plot.py`.

Usage:
    python scripts/ablations_compact.py --axis latent_dim
    python scripts/ablations_compact.py --axis translator_hidden
    python scripts/ablations_compact.py --axis n_layers
    python scripts/ablations_compact.py --axis all          # all sweeps
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.datasets.moving_mnist_on_the_fly import MovingMNISTOnTheFly
from src.models.compact.compact_predictor import CompactPredictor
from src.training.trainer import train
from src.evaluation.metrics import evaluate


SWEEPS = {
    'latent_dim':         [16, 32, 64, 128],
    'translator_hidden':  [64, 128, 256, 512],
    'n_layers':           [2, 3],
}

DEFAULTS = {
    'latent_dim': 64,
    'translator_hidden': 256,
    'n_layers': 2,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(cfg, hparams: dict, run_name: str, train_loader, val_loader,
            test_loader, device):
    """Train one config and return its summary (params, best_val, test metrics)."""
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Hparams: {hparams}")
    print(f"{'='*60}")
    
    model = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        **hparams,
    )
    n_params = count_params(model)
    print(f"Trainable params: {n_params / 1e6:.3f}M")
    
    # train() returns {"best_val_loss": ..., "best_epoch": ...} in this repo
    train_result = train(
        model, cfg, train_loader, val_loader, run_name=run_name,
    )
    best_val = train_result['best_val_loss']
    best_epoch = train_result['best_epoch']
    
    # Load best for evaluation
    ckpt_path = Path(cfg.checkpoint_dir) / run_name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    
    test_results = evaluate(model, test_loader, device)
    
    # test_results['mse_per_step'] is a torch.Tensor or numpy array
    mse_list = test_results['mse_per_step']
    ssim_list = test_results['ssim_per_step']
    if hasattr(mse_list, 'tolist'):
        mse_list = mse_list.tolist()
    if hasattr(ssim_list, 'tolist'):
        ssim_list = ssim_list.tolist()
    
    summary = {
        'run_name': run_name,
        'hparams': hparams,
        'n_params': n_params,
        'n_params_M': n_params / 1e6,
        'best_val_loss': float(best_val),
        'best_epoch': int(best_epoch),
        'test_mse_per_step': list(mse_list),
        'test_ssim_per_step': list(ssim_list),
        'test_mse_avg': float(sum(mse_list) / len(mse_list)),
        'test_ssim_avg': float(sum(ssim_list) / len(ssim_list)),
    }
    
    out_path = Path("ablations/compact") / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--axis', choices=list(SWEEPS.keys()) + ['all'], default='all',
    )
    parser.add_argument('--n_epochs', type=int, default=100,
                        help="Reduced epochs for ablations (default 100)")
    args = parser.parse_args()
    
    cfg = MMNISTConfig(
        batch_size=32, n_epochs=args.n_epochs, lr=1e-3,
        num_workers=4, device="cuda", log_every_n_steps=100,
    )
    
    train_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path, mnist_root=cfg.mnist_root,
        split="train", n_samples=cfg.otf_n_samples_train,
        input_len=cfg.input_len, target_len=cfg.target_len,
        seed=cfg.otf_seed,
    )
    val_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path, mnist_root=cfg.mnist_root,
        split="val", n_samples=cfg.otf_n_samples_val,
        input_len=cfg.input_len, target_len=cfg.target_len,
        seed=cfg.otf_seed,
    )
    test_ds = MovingMNIST(
        cfg.npy_path, split="test",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0,
    )
    
    device = torch.device(cfg.device)
    
    axes = list(SWEEPS.keys()) if args.axis == 'all' else [args.axis]
    all_summaries = []
    
    for axis in axes:
        for value in SWEEPS[axis]:
            hparams = DEFAULTS.copy()
            hparams[axis] = value
            run_name = f"compact-abl-{axis}={value}"
            summary = run_one(
                cfg, hparams, run_name,
                train_loader, val_loader, test_loader, device,
            )
            all_summaries.append(summary)
    
    # Aggregate summary
    print("\n\n=== Summary of ablations ===")
    print(f"{'Run':<35}{'Params (M)':<12}{'MSE avg':<12}{'SSIM avg':<12}")
    print("-" * 71)
    for s in all_summaries:
        print(
            f"{s['run_name']:<35}"
            f"{s['n_params_M']:<12.3f}"
            f"{s['test_mse_avg']:<12.4f}"
            f"{s['test_ssim_avg']:<12.4f}"
        )


if __name__ == "__main__":
    main()
