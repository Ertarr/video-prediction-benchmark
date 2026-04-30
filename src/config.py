from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "moving_mnist"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class MMNISTConfig:
    # Data
    npy_path: Path = DATA_DIR / "mnist_test_seq.npy"
    input_len: int = 10
    target_len: int = 10
    image_size: int = 64
    n_channels: int = 1
    
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    n_epochs: int = 200
    weight_decay: float = 0.0
    num_workers: int = 4           # set 0 on Windows; 4 on Linux/Colab
    device: str = "cuda"
    
    # On-the-fly generation (used only by *_otf scripts)
    mnist_root: str = "./data/mnist"
    otf_n_samples_train: int = 10000
    otf_n_samples_val: int = 1000
    otf_seed: int = 0
    
    # Logging + checkpoints
    log_every_n_steps: int = 50
    checkpoint_dir: Path = CHECKPOINTS_DIR
    
    # Reproducibility
    seed: int = 7
    
    # W&B
    wandb_project: str = "video-prediction-benchmark"


@dataclass
class KTHConfig:
    """Configuration for KTH Actions experiments."""
    frames_root: str = "data/kth/frames"
    train_clips: str = "data/kth/train_clips.txt"
    val_clips: str = "data/kth/val_clips.txt"
    test_clips: str = "data/kth/test_clips.txt"
    
    input_len: int = 10
    target_len: int = 20
    image_size: int = 128
    n_channels: int = 1
    
    batch_size: int = 16
    seed: int = 7
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 100
    num_workers: int = 4
    
    device: str = "cuda"
    log_every_n_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "video-prediction-benchmark"
