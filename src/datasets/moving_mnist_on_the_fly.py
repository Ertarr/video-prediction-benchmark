"""
Moving MNIST with on-the-fly generation, following the standard protocol
(Srivastava et al., 2015; SimVP, PredRNN, ConvLSTM benchmarks).

Train/val sequences are generated dynamically from the MNIST source images.
The test set always uses the official Toronto .npy file for comparability.

Protocol:
    - 2 MNIST digits per sequence
    - Each digit has a random initial position and a random velocity sampled
      once at sequence init (constant velocity throughout the sequence)
    - Digits bounce off the canvas walls with classical reflection
    - Pixel overlap between digits handled with element-wise max
    - Default: 20 frames at 64x64 resolution, 1 channel
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from src.datasets.moving_mnist import MovingMNIST


class MovingMNISTOnTheFly(Dataset):
    """
    Generates Moving MNIST sequences on the fly from MNIST source images.
    
    For split='test', delegates to MovingMNIST (Toronto .npy file)
    to ensure comparability with published benchmarks.
    """
    def __init__(
        self,
        npy_test_path: str | Path,     # used only for split='test'
        mnist_root: str = "./data/mnist",
        split: str = "train",          # 'train', 'val', or 'test'
        n_samples: int = 10000,        # number of sequences for train/val
        input_len: int = 10,
        target_len: int = 10,
        image_size: int = 64,
        digit_size: int = 28,
        max_speed: float = 4.0,
        seed: int = 0,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.input_len = input_len
        self.target_len = target_len
        self.seq_len = input_len + target_len
        self.image_size = image_size
        self.digit_size = digit_size
        self.max_speed = max_speed
        self.seed = seed
        self.n_samples = n_samples
        
        # For test, delegate to the standard Toronto-based dataset
        if split == "test":
            self.test_ds = MovingMNIST(
                npy_test_path, split="test",
                input_len=input_len, target_len=target_len,
            )
        else:
            # Load MNIST source images, normalized to [0, 1]
            mnist = datasets.MNIST(root=mnist_root, train=True, download=True)
            self.mnist_imgs = np.array(mnist.data, dtype=np.float32) / 255.0
    
    def __len__(self) -> int:
        if self.split == "test":
            return len(self.test_ds)
        return self.n_samples
    
    def __getitem__(self, idx: int):
        if self.split == "test":
            return self.test_ds[idx]
        
        # Val: deterministic per idx so the val set is the same every epoch.
        # Train: stochastic, fresh sequences every call.
        if self.split == "val":
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random.RandomState()
        
        sequence = self._generate_sequence(rng)
        input_frames = sequence[:self.input_len]
        target_frames = sequence[self.input_len:]
        return torch.from_numpy(input_frames), torch.from_numpy(target_frames)
    
    def _generate_sequence(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate one sequence of self.seq_len frames with 2 moving digits.
        
        Returns:
            np.ndarray of shape (seq_len, 1, image_size, image_size), float32, in [0, 1]
        """
        canvas = np.zeros(
            (self.seq_len, 1, self.image_size, self.image_size),
            dtype=np.float32,
        )
        max_pos = self.image_size - self.digit_size
        
        # 2 digits placed independently, overlapping pixels combined via max
        for _ in range(2):
            digit_idx = rng.randint(self.mnist_imgs.shape[0])
            digit_img = self.mnist_imgs[digit_idx]   # (28, 28)
            
            # Random initial position (top-left corner) and constant velocity
            pos = rng.uniform(0, max_pos, size=2).astype(np.float32)
            vel = rng.uniform(-self.max_speed, self.max_speed, size=2).astype(np.float32)
            
            for t in range(self.seq_len):
                x, y = int(pos[0]), int(pos[1])
                region = canvas[t, 0, y:y + self.digit_size, x:x + self.digit_size]
                canvas[t, 0, y:y + self.digit_size, x:x + self.digit_size] = np.maximum(
                    region, digit_img,
                )
                
                # Update position with wall reflection on each axis
                new_pos = pos + vel
                for axis in range(2):
                    if new_pos[axis] < 0:
                        new_pos[axis] = -new_pos[axis]
                        vel[axis] = -vel[axis]
                    elif new_pos[axis] > max_pos:
                        new_pos[axis] = 2 * max_pos - new_pos[axis]
                        vel[axis] = -vel[axis]
                pos = new_pos
        
        return canvas
