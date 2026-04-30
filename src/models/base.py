from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn


class VideoPredictor(nn.Module):
    """
    Base class for all video prediction models in this project.
    
    Input  : tensor of shape (B, T_in, C, H, W), float32 in [0, 1]
    Output : tensor of shape (B, T_out, C, H, W), float32

    Subclasses must implement function "forward"
    """

    def __init__(self, input_len: int, target_len: int) -> None:
        super().__init__()
        self.input_len = input_len
        self.target_len = target_len

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
