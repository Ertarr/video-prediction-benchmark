"""
Phase 6 on KTH: the same one-change variants of Compact (see
phase6_train.py), scored on the full frame and on the moving pixels
(scripts/kth_motion_eval.py). Same setup as scripts/train_compact_kth.py:
128x128, 10 -> 20 frames, batch 16, 100 epochs.

--eval-only skips training and scores the saved best.pt.

Usage (Colab, KTH frames in data/kth):
    WANDB_MODE=disabled python scripts/phase6_train_kth.py --variant skip --seed 1 \
        --kth-root data/kth --ckpt-dir /content/drive/MyDrive/vpb/checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import KTHConfig
from src.datasets.kth_actions import KTHActions
from src.models.compact.compact_predictor import CompactPredictor
from src.training.trainer import train, set_seed
from src.evaluation.metrics import evaluate
from scripts.phase6_train import VARIANTS, summarize
from scripts.kth_motion_eval import moving_region_metrics


def build_compact_kth(variant: str) -> CompactPredictor:
    v = VARIANTS[variant]
    return CompactPredictor(
        input_len=10, target_len=20, in_shape=(10, 1, 128, 128),
        latent_dim=64, translator_hidden=128,
        n_layers=v["n_layers"], translator_dilation=v["translator_dilation"],
        skip=v["skip"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--kth-root", default="data/kth")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="results/phase6")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-only", action="store_true")
    # smaller values only for a quick check that the script runs
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.kth_root)
    cfg = KTHConfig(
        frames_root=str(root / "frames"),
        train_clips=str(root / "train_clips.txt"),
        val_clips=str(root / "val_clips.txt"),
        test_clips=str(root / "test_clips.txt"),
        batch_size=16, n_epochs=args.n_epochs, lr=1e-3,
        num_workers=args.num_workers, device=args.device, seed=args.seed,
        checkpoint_dir=args.ckpt_dir,
    )
    run_name = f"phase6-kth-{args.variant}-seed{args.seed}"
    variant = VARIANTS[args.variant]

    set_seed(args.seed)
    model = build_compact_kth(args.variant)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_name}: {variant}, {n_params / 1e6:.3f}M params")

    def kth(clips, n):
        ds = KTHActions(cfg.frames_root, clips, input_len=10, target_len=20, image_size=128)
        return ds if n is None else Subset(ds, range(n))

    if not args.eval_only:
        train_loader = DataLoader(kth(cfg.train_clips, args.n_train), batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(kth(cfg.val_clips, args.n_val), batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        criterion = nn.L1Loss() if variant["loss"] == "l1" else None
        train(model, cfg, train_loader, val_loader, run_name=run_name, criterion=criterion)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg.checkpoint_dir) / run_name / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    test_loader = DataLoader(kth(cfg.test_clips, args.n_test), batch_size=16,
                             shuffle=False, num_workers=cfg.num_workers)

    summary = {
        "run_name": run_name,
        "dataset": "kth",
        "variant": args.variant,
        "config": variant,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "n_params": n_params,
        "best_val_loss": float(ckpt["val_loss"]),
        "best_epoch": int(ckpt["epoch"]),
        "kth_test": summarize(evaluate(model, test_loader, device)),
        "kth_moving": moving_region_metrics(model, test_loader, device),
    }
    if args.eval_only:
        summary["note"] = "training interrupted before epoch 100, best.pt evaluated"
    out = Path(args.out_dir) / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    m = summary["kth_moving"]
    print(f"KTH test: MSE {summary['kth_test']['mse_avg']:.5f}  SSIM "
          f"{summary['kth_test']['ssim_avg']:.4f} | moving MSE t=20 "
          f"{m['moving_mse_per_step'][19]:.4f}  sharpness t=20 {m['sharpness_per_step'][19]:.2f}")


if __name__ == "__main__":
    main()
