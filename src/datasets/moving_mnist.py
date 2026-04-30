from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

class MovingMNIST(Dataset):

    split_ranges = {
        "train": (0, 8000),
        "val":   (8000, 9000),
        "test":  (9000, 10000)
    }

    def __init__(self, npy_path: str | Path, split="train", input_len=10, target_len=10) -> None:

        if split not in self.split_ranges:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(self.split_ranges.keys())}.")
        
        assert(input_len + target_len <= 20)

        self.data = np.load(npy_path, mmap_mode='r')
        self.data = np.transpose(self.data, [1, 0, 2, 3])
        self.split_range = self.split_ranges[split]
        self.input_len = input_len
        self.target_len = target_len


    def __len__(self):
        return self.split_range[1] - self.split_range[0]
        
    def __getitem__(self, idx):

        assert(idx < self.__len__())
        item = self.data[idx + self.split_range[0]].astype(np.float32) / 255.
        item = torch.from_numpy(item)
        # add the channel dimension
        item = torch.unsqueeze(item, 1)
        input_frames, target_frames = item[:self.input_len], item[self.input_len:self.input_len+self.target_len]
        return input_frames, target_frames