"""
Evaluation metrics for video prediction models.

Provides per-timestep MSE and SSIM, and an evaluate() function
that runs over a dataloader and returns aggregated metrics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn


def compute_mse_per_step(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error separately for each timestep.
    
    Args:
        pred:   (B, T, C, H, W) float tensor in [0, 1]
        target: (B, T, C, H, W) float tensor in [0, 1]
    
    Returns:
        Tensor of shape (T,) with per-timestep MSE averaged over batch and pixels.
    """

    return ((pred - target) ** 2).mean(dim = [0, 2, 3, 4])


def compute_ssim_per_step(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Compute Structural Similarity Index (SSIM) separately for each timestep.
    
    SSIM is computed per-image then averaged over the batch.
    Uses scikit-image's reference implementation with data_range=1.0.
    
    Args:
        pred:   (B, T, C, H, W) float tensor in [0, 1]
        target: (B, T, C, H, W) float tensor in [0, 1]
    
    Returns:
        np.ndarray of shape (T,) with per-timestep SSIM averaged over batch.
    """
    # Move to CPU + convert to numpy
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    
    B, T, C, H, W = pred_np.shape
    
    # Output accumulator
    ssim_per_step = np.zeros(T)
    
    # for each timestep, average SSIM over batch
    for t in range(T):

        for b in range(B):
            ssim_per_step[t] += ssim_fn(pred_np[b, t, 0], tgt_np[b, t, 0], data_range=1.0)

        ssim_per_step[t] /= B
    
    return ssim_per_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate a model on a dataloader, computing per-step MSE and SSIM.
    
    Args:
        model:  trained model with a forward(x) -> y interface
        loader: dataloader yielding (input, target) batches
        device: torch device for inference
    
    Returns:
        dict with keys:
            - 'mse_per_step':  np.ndarray of shape (T,)
            - 'ssim_per_step': np.ndarray of shape (T,)
    """
    model.eval()
    
    # Accumulators (sample-weighted to handle the last batch correctly)
    total_mse = None      # will be a tensor of shape (T,)
    total_ssim = None      # will be a numpy array of shape (T,)
    n_samples = 0
    
    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        
        # Forward pass
        pred = model(inp)
        
        # Clamp predictions to [0, 1] before computing metrics
        # (model outputs are not guaranteed to be in [0, 1])
        pred = pred.clamp(0.0, 1.0)
        
        B = inp.size(0)
        
        # MSE per step: shape (T,)
        mse_step = compute_mse_per_step(pred, tgt) * B
        total_mse = mse_step if total_mse is None else total_mse + mse_step
        
        
        # SSIM per step: shape (T,)
        ssim_step = compute_ssim_per_step(pred, tgt) * B
        total_ssim = ssim_step if total_ssim is None else total_ssim + ssim_step
        
        n_samples += B
    
    # Final weighted averages
    assert total_mse is not None and total_ssim is not None, "Empty loader"
    return {
        "mse_per_step": (total_mse / n_samples).cpu().numpy(),
        "ssim_per_step": total_ssim / n_samples,
    }