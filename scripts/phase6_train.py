"""
Phase 6 on Moving MNIST: one change to Compact n_layers=2 per variant,
100 epochs, same data and loop as the Phase 4 ablations.

    skip    SimVP's skip connection (first encoder block added back
            before the last decoder block)
    l1      L1 loss instead of MSE
    dil2    translator 3x3 with dilation 2 (the window covers a digit)
    n3      n_layers=3, more seeds for the Phase 4 result
    n2      the baseline

The best checkpoint is scored on the Toronto test set and on the
fixed-speed test sets, in results/phase6/<run_name>.json.

Usage:
    WANDB_MODE=disabled python scripts/phase6_train.py --variant dil2 --seed 1 \
        --data-root data --ckpt-dir /content/drive/MyDrive/vpb/checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.datasets.moving_mnist_on_the_fly import MovingMNISTOnTheFly
from src.datasets.moving_mnist_speed import MovingMNISTFixedSpeed
from src.models.compact.compact_predictor import CompactPredictor
from src.training.trainer import train, set_seed
from src.evaluation.metrics import evaluate


VARIANTS = {
    "n2":   dict(n_layers=2, translator_dilation=1, skip=False, loss="mse"),
    "n3":   dict(n_layers=3, translator_dilation=1, skip=False, loss="mse"),
    "skip": dict(n_layers=2, translator_dilation=1, skip=True,  loss="mse"),
    "l1":   dict(n_layers=2, translator_dilation=1, skip=False, loss="l1"),
    "dil2": dict(n_layers=2, translator_dilation=2, skip=False, loss="mse"),
}

SPEEDS = [1.0, 2.5, 4.0]
SPEED_SET_SEED = 12345       # same sequences for every model and every run
SPEED_SET_SIZE = 1000


def build_compact(variant: str) -> CompactPredictor:
    v = VARIANTS[variant]
    return CompactPredictor(
        input_len=10, target_len=10, in_shape=(10, 1, 64, 64),
        latent_dim=64, translator_hidden=256,
        n_layers=v["n_layers"], translator_dilation=v["translator_dilation"],
        skip=v["skip"],
    )


def summarize(res: dict) -> dict:
    mse = [float(x) for x in res["mse_per_step"]]
    ssim = [float(x) for x in res["ssim_per_step"]]
    return {
        "mse_per_step": mse, "ssim_per_step": ssim,
        "mse_avg": sum(mse) / len(mse), "ssim_avg": sum(ssim) / len(ssim),
    }


def eval_speed_groups(model, mnist_root, device, batch_size=32, download=False,
                      n_samples=SPEED_SET_SIZE):
    """Per-step MSE / SSIM on each fixed-speed test set."""
    out = {}
    for speed in SPEEDS:
        ds = MovingMNISTFixedSpeed(
            mnist_root, speed=speed, n_samples=n_samples,
            seed=SPEED_SET_SEED, download=download,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        out[str(speed)] = summarize(evaluate(model, loader, device))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="results/phase6")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    # smaller values only for a quick check that the script runs
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=None,
                        help="limit the Toronto test set (default: all 1000)")
    parser.add_argument("--speed-set-size", type=int, default=SPEED_SET_SIZE)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cfg = MMNISTConfig(
        npy_path=data_root / "moving_mnist" / "mnist_test_seq.npy",
        mnist_root=str(data_root / "mnist"),
        batch_size=32, n_epochs=args.n_epochs, lr=1e-3,
        num_workers=args.num_workers, device=args.device,
        log_every_n_steps=100, seed=args.seed,
        checkpoint_dir=Path(args.ckpt_dir),
        otf_n_samples_train=args.n_train, otf_n_samples_val=args.n_val,
    )
    run_name = f"phase6-{args.variant}-seed{args.seed}"
    variant = VARIANTS[args.variant]

    # seed before building the model, so it also fixes the init
    set_seed(args.seed)
    model = build_compact(args.variant)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_name}: {variant}, {n_params / 1e6:.3f}M params")

    train_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path, mnist_root=cfg.mnist_root,
        split="train", n_samples=cfg.otf_n_samples_train,
        input_len=cfg.input_len, target_len=cfg.target_len, seed=cfg.otf_seed,
    )
    val_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path, mnist_root=cfg.mnist_root,
        split="val", n_samples=cfg.otf_n_samples_val,
        input_len=cfg.input_len, target_len=cfg.target_len, seed=cfg.otf_seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    criterion = nn.L1Loss() if variant["loss"] == "l1" else None
    result = train(model, cfg, train_loader, val_loader,
                   run_name=run_name, criterion=criterion)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg.checkpoint_dir) / run_name / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    test_ds = MovingMNIST(cfg.npy_path, split="test",
                          input_len=cfg.input_len, target_len=cfg.target_len)
    if args.n_test is not None:
        test_ds = torch.utils.data.Subset(test_ds, range(args.n_test))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    summary = {
        "run_name": run_name,
        "variant": args.variant,
        "config": variant,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "n_params": n_params,
        "best_val_loss": float(result["best_val_loss"]),
        "best_epoch": int(result["best_epoch"]),
        "toronto_test": summarize(evaluate(model, test_loader, device)),
        "speed_test": eval_speed_groups(model, cfg.mnist_root, device, download=True,
                                        n_samples=args.speed_set_size),
    }
    out = Path(args.out_dir) / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    print(f"Toronto test: MSE {summary['toronto_test']['mse_avg']:.5f}  "
          f"SSIM {summary['toronto_test']['ssim_avg']:.4f}")


if __name__ == "__main__":
    main()
