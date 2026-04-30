"""
Entry point for training the Compact predictor on Moving MNIST.

Usage:
    python scripts/train_compact_mmnist.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist import MovingMNIST
from src.models.compact.compact_predictor import CompactPredictor
from src.training.trainer import train


def main():
    cfg = MMNISTConfig(
        batch_size=32,
        n_epochs=200,
        lr=1e-3,
        num_workers=4,
        device="cuda",
        log_every_n_steps=50,
    )

    train_ds = MovingMNIST(
        cfg.npy_path, split="train",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )
    val_ds = MovingMNIST(
        cfg.npy_path, split="val",
        input_len=cfg.input_len, target_len=cfg.target_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = CompactPredictor(
        input_len=cfg.input_len,
        target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
    )

    train(model, cfg, train_loader, val_loader, run_name="compact-mmnist")


if __name__ == "__main__":
    main()
