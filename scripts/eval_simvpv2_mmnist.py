import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MMNISTConfig
from src.models.simvpv2 import SimVPv2
from src.evaluation.runner import run_evaluation_mmnist


def main():
    cfg = MMNISTConfig(batch_size=16, num_workers=0, device="cpu")
    
    model = SimVPv2(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=(cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size),
    )
    
    ckpt_path = Path(cfg.checkpoint_dir) / "simvpv2-mmnist" / "best.pt"
    run_evaluation_mmnist(model, cfg, ckpt_path)


if __name__ == "__main__":
    main()