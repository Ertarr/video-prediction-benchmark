"""
Train one memory model on occluded Moving MNIST, then compute its
forgetting curve (scripts/eval_memory.py).

    window  no memory, attention over the last 10 frames
    rmt     + 16 recurrent memory tokens
    knn     + kNN memory over the frames that left the window
    full    attention over all 50 observed frames (control)

Training: k uniform in [0, 40], fresh sequences every epoch, same loop
as the rest of the repo (src/training/trainer.py).

Usage (Colab):
    WANDB_MODE=disabled python scripts/train_memory.py --model rmt --seed 1 \
        --data-root data --ckpt-dir /content/drive/MyDrive/vpb/checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist_occlusion import MovingMNISTOcclusion
from src.training.trainer import train, set_seed
from scripts.eval_memory import build_memory_model, evaluate_model


VAL_SEED = 7_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["window", "rmt", "knn", "full"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="results/memory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    # smaller values only for a quick check that the script runs
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    args = parser.parse_args()

    mnist_root = str(Path(args.data_root) / "mnist")
    cfg = MMNISTConfig(
        batch_size=32, n_epochs=args.n_epochs, lr=1e-3,
        num_workers=args.num_workers, device=args.device,
        log_every_n_steps=100, seed=args.seed,
        checkpoint_dir=Path(args.ckpt_dir),
    )
    run_name = f"memory-{args.model}-seed{args.seed}"

    set_seed(args.seed)
    model = build_memory_model(args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_name}: {n_params / 1e6:.3f}M params")

    train_ds = MovingMNISTOcclusion(mnist_root, n_samples=args.n_train, seed=None,
                                    mnist_train=True, download=True)
    val_ds = MovingMNISTOcclusion(mnist_root, n_samples=args.n_val, seed=VAL_SEED,
                                  mnist_train=True, download=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    result = train(model, cfg, train_loader, val_loader, run_name=run_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.ckpt_dir) / run_name / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    summary = evaluate_model(args.model, model, mnist_root, device, args.n_test)
    summary.update({
        "run_name": run_name, "seed": args.seed, "n_epochs": args.n_epochs,
        "n_params": n_params,
        "best_val_loss": float(result["best_val_loss"]),
        "best_epoch": int(result["best_epoch"]),
    })
    out = Path(args.out_dir) / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
