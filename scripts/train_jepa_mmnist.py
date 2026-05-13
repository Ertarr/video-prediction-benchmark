"""Train JEPAPredictor on Moving MNIST with on-the-fly generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.config import MMNISTConfig
from src.datasets.moving_mnist_on_the_fly import MovingMNISTOnTheFly
from src.models.compact_jepa.jepa_predictor import JEPAPredictor
from src.training.jepa_trainer import train_jepa


def main():
    cfg = MMNISTConfig(
        batch_size=32, n_epochs=200, lr=1e-3,
        num_workers=4, device="cuda", log_every_n_steps=50,
    )
    
    train_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path,
        mnist_root=cfg.mnist_root,
        split="train",
        n_samples=cfg.otf_n_samples_train,
        input_len=cfg.input_len, target_len=cfg.target_len,
        seed=cfg.otf_seed,
    )
    val_ds = MovingMNISTOnTheFly(
        npy_test_path=cfg.npy_path,
        mnist_root=cfg.mnist_root,
        split="val",
        n_samples=cfg.otf_n_samples_val,
        input_len=cfg.input_len, target_len=cfg.target_len,
        seed=cfg.otf_seed,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    
    model = JEPAPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
        ema_momentum=0.996,
        use_decoder=True,
    )
    
    train_jepa(
        model, cfg, train_loader, val_loader, run_name="jepa-mmnist-otf",
        var_coef=1.0, cov_coef=0.04, decoder_coef=1.0,
    )


if __name__ == "__main__":
    main()
