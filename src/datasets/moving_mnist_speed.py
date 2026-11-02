"""
Moving MNIST test sequences where the digit speed is fixed.

Used in Phase 6 to check whether the Compact model's blur depends on how
fast the digits move. Same rendering as MovingMNISTOnTheFly (2 digits,
element-wise max on overlap, wall reflection, int() of the position when
drawing), with two differences:
    - both digits move at the same speed (pixels per frame) in a random
      direction, instead of a random velocity in [-4, 4] on each axis
    - digits come from the MNIST test split by default, so the models
      never saw these exact digits during training

Each sequence only depends on (seed, idx), and the random draws do not
depend on the speed. So with the same seed, two speed groups contain the
same digits, start points and directions: only the speed changes.

With speed=None, each digit gets a random velocity in [-max_speed,
max_speed] on each axis, like MovingMNISTOnTheFly. That is what the JEPA
probe uses, with return_positions=True.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MovingMNISTFixedSpeed(Dataset):
    def __init__(
        self,
        mnist_root: str | Path,
        speed: float | None,           # pixels per frame, same for both digits
        n_samples: int = 1000,
        input_len: int = 10,
        target_len: int = 10,
        image_size: int = 64,
        digit_size: int = 28,
        seed: int = 0,
        mnist_train: bool = False,     # False -> MNIST test digits
        download: bool = False,
        return_positions: bool = False,
        max_speed: float = 4.0,        # only used when speed is None
    ):
        self.speed = speed
        self.max_speed = max_speed
        self.n_samples = n_samples
        self.input_len = input_len
        self.target_len = target_len
        self.seq_len = input_len + target_len
        self.image_size = image_size
        self.digit_size = digit_size
        self.seed = seed
        self.return_positions = return_positions

        mnist = datasets.MNIST(root=str(mnist_root), train=mnist_train, download=download)
        self.mnist_imgs = np.array(mnist.data, dtype=np.float32) / 255.0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self.seed + idx)
        frames, positions = self.generate(rng)
        inp = torch.from_numpy(frames[:self.input_len])
        tgt = torch.from_numpy(frames[self.input_len:])
        if self.return_positions:
            return inp, tgt, torch.from_numpy(positions)
        return inp, tgt

    def generate(self, rng: np.random.RandomState):
        """
        Returns:
            frames:    (seq_len, 1, image_size, image_size) float32 in [0, 1]
            positions: (seq_len, 2, 2) int64, top-left (x, y) of each digit
                       as drawn in each frame
        """
        canvas = np.zeros(
            (self.seq_len, 1, self.image_size, self.image_size), dtype=np.float32,
        )
        positions = np.zeros((self.seq_len, 2, 2), dtype=np.int64)
        max_pos = self.image_size - self.digit_size

        for d in range(2):
            digit_img = self.mnist_imgs[rng.randint(self.mnist_imgs.shape[0])]
            pos = rng.uniform(0, max_pos, size=2).astype(np.float32)
            if self.speed is None:
                vel = rng.uniform(-self.max_speed, self.max_speed, size=2).astype(np.float32)
            else:
                angle = rng.uniform(0, 2 * np.pi)
                vel = (self.speed * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)

            for t in range(self.seq_len):
                x, y = int(pos[0]), int(pos[1])
                positions[t, d] = (x, y)
                region = canvas[t, 0, y:y + self.digit_size, x:x + self.digit_size]
                canvas[t, 0, y:y + self.digit_size, x:x + self.digit_size] = np.maximum(
                    region, digit_img,
                )
                # Same wall reflection as MovingMNISTOnTheFly
                new_pos = pos + vel
                for axis in range(2):
                    if new_pos[axis] < 0:
                        new_pos[axis] = -new_pos[axis]
                        vel[axis] = -vel[axis]
                    elif new_pos[axis] > max_pos:
                        new_pos[axis] = 2 * max_pos - new_pos[axis]
                        vel[axis] = -vel[axis]
                pos = new_pos

        return canvas, positions
