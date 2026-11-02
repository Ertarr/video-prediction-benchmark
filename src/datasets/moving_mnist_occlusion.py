"""
Moving MNIST with an occluded digit, for the memory experiments.

Two digits move as in MovingMNISTOnTheFly. Digit B is hidden during the
last k of the n_observed frames, then visible again in the n_predict
frames to predict.
    mode="mask"    a gray square (0.5) follows B and covers it: the model
                   still sees where B is, not which digit it is
    mode="vanish"  B is not drawn at all

k is fixed (test sets) or uniform in [0, k_max] for each sequence
(training). With seed=None every call draws a fresh sequence; with an
int seed, sequence idx only depends on (seed, idx). return_meta=True also
returns k and the positions of both digits (top-left corners).
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MovingMNISTOcclusion(Dataset):
    def __init__(
        self,
        mnist_root: str | Path,
        n_samples: int = 10000,
        n_observed: int = 50,
        n_predict: int = 10,
        k: int | None = None,           # None -> uniform in [0, k_max]
        k_max: int = 40,
        mode: str = "mask",
        seed: int | None = None,
        mnist_train: bool = True,
        image_size: int = 64,
        digit_size: int = 28,
        max_speed: float = 4.0,
        mask_value: float = 0.5,
        download: bool = False,
        return_meta: bool = False,
    ):
        assert mode in ("mask", "vanish")
        assert (k if k is not None else k_max) <= n_observed
        self.n_samples = n_samples
        self.n_observed = n_observed
        self.n_predict = n_predict
        self.seq_len = n_observed + n_predict
        self.k = k
        self.k_max = k_max
        self.mode = mode
        self.seed = seed
        self.image_size = image_size
        self.digit_size = digit_size
        self.max_speed = max_speed
        self.mask_value = mask_value
        self.return_meta = return_meta

        mnist = datasets.MNIST(root=str(mnist_root), train=mnist_train, download=download)
        self.mnist_imgs = np.array(mnist.data, dtype=np.float32) / 255.0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.RandomState() if self.seed is None else \
            np.random.RandomState(self.seed + idx)
        k = rng.randint(self.k_max + 1) if self.k is None else self.k
        frames, positions = self.generate(rng, k)
        inp = torch.from_numpy(frames[:self.n_observed])
        tgt = torch.from_numpy(frames[self.n_observed:])
        if self.return_meta:
            return inp, tgt, {"k": k, "positions": torch.from_numpy(positions)}
        return inp, tgt

    def trajectory(self, rng):
        """Top-left positions (seq_len, 2) of one digit, as drawn."""
        max_pos = self.image_size - self.digit_size
        pos = rng.uniform(0, max_pos, size=2).astype(np.float32)
        vel = rng.uniform(-self.max_speed, self.max_speed, size=2).astype(np.float32)
        out = np.zeros((self.seq_len, 2), dtype=np.int64)
        for t in range(self.seq_len):
            out[t] = (int(pos[0]), int(pos[1]))
            new_pos = pos + vel
            for axis in range(2):
                if new_pos[axis] < 0:
                    new_pos[axis] = -new_pos[axis]
                    vel[axis] = -vel[axis]
                elif new_pos[axis] > max_pos:
                    new_pos[axis] = 2 * max_pos - new_pos[axis]
                    vel[axis] = -vel[axis]
            pos = new_pos
        return out

    def generate(self, rng, k):
        """
        Returns:
            frames:    (seq_len, 1, H, W) float32 in [0, 1]
            positions: (seq_len, 2, 2) int64, [t, digit, (x, y)], digit 1 is B
        """
        size, d = self.image_size, self.digit_size
        frames = np.zeros((self.seq_len, 1, size, size), dtype=np.float32)
        imgs = [self.mnist_imgs[rng.randint(len(self.mnist_imgs))] for _ in range(2)]
        positions = np.stack([self.trajectory(rng) for _ in range(2)], axis=1)
        hidden = range(self.n_observed - k, self.n_observed)    # frames where B is hidden

        for t in range(self.seq_len):
            for digit in range(2):
                if digit == 1 and t in hidden:
                    continue
                x, y = positions[t, digit]
                region = frames[t, 0, y:y + d, x:x + d]
                frames[t, 0, y:y + d, x:x + d] = np.maximum(region, imgs[digit])
            if self.mode == "mask" and t in hidden:
                # the square is drawn on top of everything, like a real occluder
                x, y = positions[t, 1]
                frames[t, 0, y:y + d, x:x + d] = self.mask_value
        return frames, positions
