"""
Compact video prediction model with latent dynamics.

Architecture:
    Encoder (CNN) -> Latent Transition -> Decoder (CNN)
    
Key design choices:
    - Downsampling controlled by n_layers (2 -> /4, 3 -> /8)
    - Latent transition operates on (B, T*latent_dim, H', W') by fusing
      the temporal dimension into channels and applying 2D convolutions
      (similar in spirit to SimVPv2 but simpler)
    - GroupNorm + GELU throughout, matching the SimVPv2 baseline for 
      fair comparison
"""
import torch
import torch.nn as nn

from ..base import VideoPredictor


class CompactEncoder(nn.Module):
    """
    Encodes each frame independently to a low-resolution latent.
    
    Input:  (B*T, in_channels, H, W)
    Output: (B*T, latent_dim, H/2^n_layers, W/2^n_layers)
    """
    def __init__(self, in_channels: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = latent_dim // (2 ** (n_layers - 1 - i))
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(2, out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class LatentTransition(nn.Module):
    """
    Applies temporal modeling on the sequence of latents.
    
    Input:  (B, T, latent_dim, H', W')
    Output: (B, T, latent_dim, H', W')
    
    Internally: reshapes to (B, T*latent_dim, H', W'), applies 2D convs
    that mix temporal information through the channel dimension, then 
    reshapes back.
    """
    def __init__(self, n_steps: int, latent_dim: int, hidden_dim: int = 128,
                 dilation: int = 1):
        super().__init__()
        self.n_steps = n_steps
        self.latent_dim = latent_dim
        in_channels = n_steps * latent_dim   # ex: 10 * 64 = 640

        # 1x1 -> 3x3 -> 1x1 bottleneck: drastically reduces params while
        # still keeping a 3x3 spatial mixing in the middle.
        # dilation > 1 spreads the 3x3 taps apart: same params and FLOPs,
        # but the conv reaches further in the latent (Phase 6).
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, latent_dim, H', W')
        Returns:
            (B, T, latent_dim, H', W')
        """
        B, T, C, H, W = x.shape
        x = x.reshape([B, T * C, H, W])
        x = self.layers(x)
        x = x.reshape([B, T, C, H, W])
        return x


class CompactDecoder(nn.Module):
    """
    Decodes each latent back to a frame, symmetrical to the encoder.
    """
    def __init__(self, latent_dim: int, out_channels: int, n_layers: int = 2):
        super().__init__()
        layers = []
        ch = latent_dim
        for i in range(n_layers):
            # halve channels at each upsampling step, except the last 1x1 conv
            out_ch = latent_dim // (2 ** (i + 1))
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(2, out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, out_channels, kernel_size=1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class CompactPredictor(VideoPredictor):
    """
    Full video prediction model:
        Encoder -> LatentTransition -> Decoder
    """
    def __init__(
        self,
        input_len: int,
        target_len: int,
        in_shape: tuple,                 # (T, C, H, W)
        latent_dim: int = 64,
        translator_hidden: int = 128,
        n_layers: int = 2,
        translator_dilation: int = 1,
        skip: bool = False,
    ):
        super().__init__(input_len=input_len, target_len=target_len)
        T, C, H, W = in_shape
        self.n_layers = n_layers
        # skip=True copies SimVP's skip connection (Phase 6): the output of
        # the first encoder block is added back before the last decoder
        # block. Frame i of the output gets the features of input frame i,
        # like in SimVP. No extra parameters.
        self.skip = skip

        self.encoder = CompactEncoder(
            in_channels=C, latent_dim=latent_dim, n_layers=n_layers,
        )
        self.transition = LatentTransition(
            n_steps=T, latent_dim=latent_dim, hidden_dim=translator_hidden,
            dilation=translator_dilation,
        )
        self.decoder = CompactDecoder(
            latent_dim=latent_dim, out_channels=C, n_layers=n_layers,
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, input_len, C, H, W)
        Returns:
            (B, target_len, C, H, W) where target_len may differ from input_len.
        """
        B, T_in, C, H, W = x.shape
        assert T_in == self.input_len
        
        # Same-length case: single forward pass
        if self.target_len == self.input_len:
            return self._forward_single(x)
        
        # target_len > input_len: autoregressive rolling
        n_rolls = (self.target_len + self.input_len - 1) // self.input_len
        predictions = []
        current_input = x
        
        for _ in range(n_rolls):
            out = self._forward_single(current_input)
            predictions.append(out)
            current_input = out
        
        full = torch.cat(predictions, dim=1)
        return full[:, :self.target_len]
    
    def _forward_single(self, x):
        """One pass of encoder -> transition -> decoder, T_out = T_in."""
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        if self.skip:
            # each encoder / decoder block is (conv, GroupNorm, GELU)
            enc1 = self.encoder.layers[:3](x_flat)
            latents_flat = self.encoder.layers[3:](enc1)
        else:
            latents_flat = self.encoder(x_flat)
        _, latent_dim, H_, W_ = latents_flat.shape
        latents = latents_flat.reshape(B, T, latent_dim, H_, W_)
        latents = self.transition(latents)
        latents_flat = latents.reshape(B * T, latent_dim, H_, W_)
        if self.skip:
            # the decoder block before the last one outputs the same shape
            # as the first encoder block (latent_dim / 2^(n-1) channels, H/2)
            k = 3 * (self.n_layers - 1)
            hid = self.decoder.layers[:k](latents_flat)
            pred = self.decoder.layers[k:](hid + enc1)
        else:
            pred = self.decoder(latents_flat)
        pred = pred.reshape(B, T, C, H, W)
        return pred
