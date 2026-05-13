"""
JEPA-style video prediction model.

Adapts the V-JEPA training approach to the supervised video prediction
setting of this benchmark. Key differences vs the pixel-space Compact
predictor:

    1. Two encoders (online + EMA target) instead of one
    2. The translator predicts FUTURE LATENTS (not future pixels)
    3. The loss is computed in latent space (smooth L1) with optional
       variance/covariance regularization to prevent representation
       collapse
    4. A small auxiliary decoder maps predicted latents back to pixels
       for visualization only (its gradients do NOT flow back into the
       latent predictor)

The architecture deliberately reuses the same encoder/translator
modules as the Compact predictor (Phase 1, 2), so any improvement
attributable to JEPA training comes from the training strategy, not
from a different architecture. This makes the comparison fair.
"""
import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import VideoPredictor
from ..compact.compact_predictor import (
    CompactEncoder,
    LatentTransition,
    CompactDecoder,
)


class EMATargetEncoder(nn.Module):
    """
    Target encoder updated by exponential moving average of the
    online encoder's weights. No gradient flows through this module.
    
    Args:
        online_encoder: the encoder being trained normally
        momentum: EMA decay (0.996 default, schedule possible)
    """
    def __init__(self, online_encoder: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.target_encoder = copy.deepcopy(online_encoder)
        self.momentum = momentum
        # Freeze: no gradient through target
        for p in self.target_encoder.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """EMA update: target = m * target + (1 - m) * online."""
        for p_target, p_online in zip(
            self.target_encoder.parameters(),
            online_encoder.parameters(),
        ):
            p_target.data.mul_(self.momentum).add_(
                p_online.data, alpha=1.0 - self.momentum,
            )
    
    @torch.no_grad()
    def forward(self, x):
        return self.target_encoder(x)


class JEPAPredictor(VideoPredictor):
    """
    JEPA-style video predictor.
    
    Forward pass returns BOTH:
        - predicted_latents: (B, T_out, latent_dim, H', W')
            -> used for the JEPA latent-space loss
        - predicted_pixels: (B, T_out, C, H, W)
            -> used only for visualization, decoder trained with
               stop-grad on latents
    
    Args:
        input_len, target_len: as in VideoPredictor
        in_shape: (T, C, H, W)
        latent_dim, translator_hidden, n_layers: same as CompactPredictor
        ema_momentum: EMA decay for target encoder
        use_decoder: whether to attach the auxiliary visualization decoder
    """
    def __init__(
        self,
        input_len: int,
        target_len: int,
        in_shape: tuple,
        latent_dim: int = 64,
        translator_hidden: int = 256,
        n_layers: int = 2,
        ema_momentum: float = 0.996,
        use_decoder: bool = True,
    ):
        super().__init__(input_len=input_len, target_len=target_len)
        T, C, H, W = in_shape
        
        # Online encoder (trained by gradient)
        self.online_encoder = CompactEncoder(
            in_channels=C, latent_dim=latent_dim, n_layers=n_layers,
        )
        
        # Target encoder (EMA, frozen)
        self.target_encoder = EMATargetEncoder(
            self.online_encoder, momentum=ema_momentum,
        )
        
        # Predictor: same as Compact translator, but predicts future
        # latents from past latents
        self.predictor = LatentTransition(
            n_steps=T, latent_dim=latent_dim, hidden_dim=translator_hidden,
        )
        
        # Auxiliary decoder for visualization (latents -> pixels)
        # Trained with stop-grad on latents to keep the JEPA training
        # purely in latent space
        self.use_decoder = use_decoder
        if use_decoder:
            self.aux_decoder = CompactDecoder(
                latent_dim=latent_dim, out_channels=C, n_layers=n_layers,
            )
    
    def encode_online(self, frames):
        """Apply online encoder frame-by-frame. Input: (B, T, C, H, W)."""
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        latents_flat = self.online_encoder(flat)
        _, latent_dim, H_, W_ = latents_flat.shape
        return latents_flat.reshape(B, T, latent_dim, H_, W_)
    
    def encode_target(self, frames):
        """Apply target encoder frame-by-frame (no grad)."""
        B, T, C, H, W = frames.shape
        with torch.no_grad():
            flat = frames.reshape(B * T, C, H, W)
            latents_flat = self.target_encoder(flat)
            _, latent_dim, H_, W_ = latents_flat.shape
            return latents_flat.reshape(B, T, latent_dim, H_, W_)
    
    def forward(
        self,
        x: torch.Tensor,
        future_frames: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, input_len, C, H, W) - past frames
            future_frames: (B, target_len, C, H, W) - GT future frames,
                used to compute target latents during training. At
                inference, can be None.
        
        Returns:
            dict with:
                'predicted_latents': (B, target_len, latent_dim, H', W')
                'target_latents':    (B, target_len, latent_dim, H', W') or None
                'online_latents':    (B, input_len, latent_dim, H', W')
                'predicted_pixels':  (B, target_len, C, H, W) (if use_decoder)
        """
        # 1. Encode past with online encoder
        online_latents = self.encode_online(x)
        
        # 2. Predict future latents
        # Note: predictor expects (B, T, latent_dim, H', W'), same shape in/out
        predicted_latents = self.predictor(online_latents)
        
        # 3. If we have future frames (training), encode them with the
        #    target encoder to get the supervision signal
        target_latents = None
        if future_frames is not None:
            target_latents = self.encode_target(future_frames)
        
        # 4. Optional: decode predicted latents to pixels for visualization
        predicted_pixels = None
        if self.use_decoder:
            B, T, D, H_, W_ = predicted_latents.shape
            # Stop-grad on latents so the decoder doesn't influence the
            # latent predictor's training
            latents_for_decode = predicted_latents.detach()
            flat = latents_for_decode.reshape(B * T, D, H_, W_)
            pixels_flat = self.aux_decoder(flat)
            _, C, H, W = pixels_flat.shape
            predicted_pixels = pixels_flat.reshape(B, T, C, H, W)
        
        return {
            'predicted_latents': predicted_latents,
            'target_latents': target_latents,
            'online_latents': online_latents,
            'predicted_pixels': predicted_pixels,
        }
    
    @torch.no_grad()
    def update_target(self):
        """Call after each training step to update the EMA target."""
        self.target_encoder.update(self.online_encoder)


def jepa_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
    use_huber: bool = True,
) -> dict:
    """
    JEPA loss with VICReg-style anti-collapse regularization.
    
    Total loss = prediction_loss + var_coef * variance_loss + cov_coef * covariance_loss
    
    Args:
        predicted_latents: (B, T, D, H', W')
        target_latents: (B, T, D, H', W')
        var_coef: weight on variance loss (anti-collapse)
        cov_coef: weight on covariance loss (decorrelation)
        use_huber: smooth L1 vs L2
    
    Returns:
        dict with 'total', 'pred', 'var', 'cov' for logging
    """
    # Prediction loss: smooth L1 between predicted and target embeddings
    if use_huber:
        pred_loss = F.smooth_l1_loss(predicted_latents, target_latents)
    else:
        pred_loss = F.mse_loss(predicted_latents, target_latents)
    
    # Reshape for VICReg: (B*T*H*W, D)
    B, T, D, H_, W_ = predicted_latents.shape
    z = predicted_latents.permute(0, 1, 3, 4, 2).reshape(-1, D)
    
    # Variance loss: encourage std >= 1 along each latent dim
    eps = 1e-4
    std_z = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std_z).mean()
    
    # Covariance loss: penalize off-diagonal entries of feature covariance
    z_centered = z - z.mean(dim=0)
    n = z.size(0)
    cov = (z_centered.T @ z_centered) / (n - 1)
    # Off-diagonal squared and sum, normalized by D
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag.pow(2).sum()) / D
    
    total = pred_loss + var_coef * var_loss + cov_coef * cov_loss
    
    return {
        'total': total,
        'pred': pred_loss.detach(),
        'var': var_loss.detach(),
        'cov': cov_loss.detach(),
    }


def pixel_loss(predicted_pixels: torch.Tensor, target_pixels: torch.Tensor):
    """Auxiliary pixel-space loss for the visualization decoder only."""
    return F.mse_loss(predicted_pixels, target_pixels)
