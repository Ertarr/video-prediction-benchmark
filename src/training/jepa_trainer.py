"""
Training loop for JEPAPredictor.

Differs from the standard pixel-space trainer in three ways:
    1. Loss is in latent space (smooth L1 + variance + covariance)
    2. EMA target encoder is updated after each step
    3. Auxiliary visualization decoder has its own pixel-space loss
       (with stop-grad on the latent predictor side)
"""
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.models.compact_jepa.jepa_predictor import (
    JEPAPredictor, jepa_loss, pixel_loss,
)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_jepa(
    model: JEPAPredictor,
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
    decoder_coef: float = 1.0,
    ema_schedule: bool = True,
):
    """
    Train a JEPAPredictor.
    
    Args:
        model: JEPAPredictor (must have use_decoder=True for visualization)
        cfg: config dataclass with batch_size, lr, n_epochs, device, etc.
        train_loader, val_loader: yield (input_frames, future_frames)
        run_name: subdir under cfg.checkpoint_dir
        var_coef, cov_coef: VICReg-style regularization weights
        decoder_coef: weight on auxiliary decoder pixel loss
        ema_schedule: cosine schedule from cfg ema_momentum -> 1.0
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    
    # Optimizer: only parameters that require grad
    # Note: target encoder params have requires_grad=False so they're skipped
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs,
    )
    
    n_params = count_params(model)
    print(f"Model params (trainable): {n_params / 1e6:.2f}M")
    print(f"Run: {run_name}")
    print(f"Loss weights: var={var_coef}, cov={cov_coef}, decoder={decoder_coef}")
    
    # W&B init
    use_wandb = HAS_WANDB and getattr(cfg, 'wandb_project', None) is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project, name=run_name,
            config={
                **{k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                'var_coef': var_coef, 'cov_coef': cov_coef,
                'decoder_coef': decoder_coef, 'n_params_M': n_params / 1e6,
            },
        )
    
    # Cosine EMA schedule from initial momentum to 1.0
    initial_momentum = model.target_encoder.momentum
    
    ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    best_epoch = -1
    global_step = 0
    
    for epoch in range(cfg.n_epochs):
        # EMA momentum schedule (cosine: initial -> 1.0)
        if ema_schedule:
            progress = epoch / max(cfg.n_epochs - 1, 1)
            current_momentum = initial_momentum + (1.0 - initial_momentum) * (
                0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159265)))
            ).item()
            model.target_encoder.momentum = current_momentum
        
        # === Train ===
        model.train()
        t0 = time.time()
        train_losses = {'total': 0.0, 'pred': 0.0, 'var': 0.0,
                       'cov': 0.0, 'decoder': 0.0}
        n_batches = 0
        
        for batch_idx, (input_frames, future_frames) in enumerate(train_loader):
            input_frames = input_frames.to(device, non_blocking=True)
            future_frames = future_frames.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward
            out = model(input_frames, future_frames=future_frames)
            
            # JEPA latent-space loss (this is the main one)
            losses = jepa_loss(
                out['predicted_latents'], out['target_latents'],
                var_coef=var_coef, cov_coef=cov_coef,
            )
            
            # Auxiliary decoder loss (pixels)
            # The decoder receives detached latents inside the forward,
            # so its gradients only update the decoder
            dec_loss_value = torch.tensor(0.0, device=device)
            if out['predicted_pixels'] is not None:
                dec_loss_value = pixel_loss(
                    out['predicted_pixels'], future_frames,
                )
            
            total = losses['total'] + decoder_coef * dec_loss_value
            total.backward()
            optimizer.step()
            
            # EMA update
            model.update_target()
            
            train_losses['total'] += total.item()
            train_losses['pred'] += losses['pred'].item()
            train_losses['var'] += losses['var'].item()
            train_losses['cov'] += losses['cov'].item()
            train_losses['decoder'] += dec_loss_value.item()
            n_batches += 1
            global_step += 1
            
            if use_wandb and global_step % cfg.log_every_n_steps == 0:
                wandb.log({
                    'step/total': total.item(),
                    'step/pred': losses['pred'].item(),
                    'step/var': losses['var'].item(),
                    'step/cov': losses['cov'].item(),
                    'step/decoder': dec_loss_value.item(),
                    'step/lr': scheduler.get_last_lr()[0],
                    'step/ema_momentum': model.target_encoder.momentum,
                }, step=global_step)
        
        for k in train_losses:
            train_losses[k] /= n_batches
        
        # === Val ===
        model.eval()
        val_losses = {'total': 0.0, 'pred': 0.0, 'decoder': 0.0}
        n_val = 0
        with torch.no_grad():
            for input_frames, future_frames in val_loader:
                input_frames = input_frames.to(device)
                future_frames = future_frames.to(device)
                out = model(input_frames, future_frames=future_frames)
                losses = jepa_loss(
                    out['predicted_latents'], out['target_latents'],
                    var_coef=var_coef, cov_coef=cov_coef,
                )
                dec_loss_v = torch.tensor(0.0, device=device)
                if out['predicted_pixels'] is not None:
                    dec_loss_v = pixel_loss(
                        out['predicted_pixels'], future_frames,
                    )
                val_losses['total'] += losses['total'].item()
                val_losses['pred'] += losses['pred'].item()
                val_losses['decoder'] += dec_loss_v.item()
                n_val += 1
        
        for k in val_losses:
            val_losses[k] /= n_val
        
        scheduler.step()
        
        elapsed = time.time() - t0
        print(
            f"[{epoch+1:3d}/{cfg.n_epochs}] "
            f"train_pred={train_losses['pred']:.5f} "
            f"train_var={train_losses['var']:.4f} "
            f"train_cov={train_losses['cov']:.4f} "
            f"train_dec={train_losses['decoder']:.5f} | "
            f"val_pred={val_losses['pred']:.5f} "
            f"val_dec={val_losses['decoder']:.5f} | "
            f"{elapsed:.0f}s"
        )
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **{f'train/{k}': v for k, v in train_losses.items()},
                **{f'val/{k}': v for k, v in val_losses.items()},
            }, step=global_step)
        
        # Save best on val pred loss (latent-space prediction quality)
        if val_losses['pred'] < best_val_loss:
            best_val_loss = val_losses['pred']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, ckpt_dir / 'best.pt')
    
    print(f"\nBest val pred loss: {best_val_loss:.5f} at epoch {best_epoch}")
    if use_wandb:
        wandb.finish()
    return best_val_loss, best_epoch
