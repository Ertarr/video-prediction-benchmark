from ..base import VideoPredictor
from ..simvpv2.simvp_model import SimVP_Model
import torch

class SimVPv2(VideoPredictor):
    """
    Wrapper around SimVP_Model from OpenSTL, adapted to our VideoPredictor interface.
    """

    def __init__(self, input_len: int, target_len: int, in_shape: tuple, **kwargs) -> None:
        super().__init__(input_len, target_len)
        self.model = SimVP_Model(in_shape=in_shape, **kwargs)

    def forward(self, x):
        """
        Args:
            x: (B, input_len, C, H, W)
        Returns:
            (B, target_len, C, H, W)
        """
        # If target_len == input_len, single forward pass
        if self.target_len == self.input_len:
            return self.model(x)
        
        # Rolling prediction
        n_rolls = (self.target_len + self.input_len - 1) // self.input_len
        predictions = []
        current_input = x
        
        for _ in range(n_rolls):
            out = self.model(current_input)  # (B, input_len, C, H, W)
            predictions.append(out)
            current_input = out
        
        full = torch.cat(predictions, dim=1)
        return full[:, :self.target_len]

