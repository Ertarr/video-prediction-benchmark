"""
Evaluate Compact predictor (trained on the fixed Toronto split) on the
standard Moving MNIST test set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MMNISTConfig
from src.models.compact.compact_predictor import CompactPredictor
from src.evaluation.runner import run_evaluation_mmnist


def main():
    cfg = MMNISTConfig(batch_size=16, num_workers=0, device="cpu")
    
    model = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
    )
    
    ckpt_path = Path(cfg.checkpoint_dir) / "compact-mmnist" / "best.pt"
    run_evaluation_mmnist(model, cfg, ckpt_path)


if __name__ == "__main__":
    main()
