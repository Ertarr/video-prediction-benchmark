"""Train Compact with n_layers=3 on KTH Actions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.config import KTHConfig
from src.datasets.kth_actions import KTHActions
from src.models.compact.compact_predictor import CompactPredictor
from src.training.trainer import train


def main():
    cfg = KTHConfig()  # all defaults

    train_ds = KTHActions(
        cfg.frames_root, cfg.train_clips,
        input_len=cfg.input_len, target_len=cfg.target_len,
        image_size=cfg.image_size,
    )
    val_ds = KTHActions(
        cfg.frames_root, cfg.val_clips,
        input_len=cfg.input_len, target_len=cfg.target_len,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # KTH = 128x128, n_layers=3 → latent 16x16
    model = CompactPredictor(
        input_len=cfg.input_len,
        target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=128,   # KTH default (256 on MMNIST)
        n_layers=3,              # instead of 2
    )

    result = train(model, cfg, train_loader, val_loader, run_name="compact-kth-n3")
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
