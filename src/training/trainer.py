"""
Training utilities for video prediction models.

Provides generic train / validate functions that work with any model
inheriting from VideoPredictor, using MSE loss and Adam optimizer.

Set WANDB_MODE=disabled in the environment to skip W&B logging
(useful when reproducing without a W&B account).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across numpy, torch CPU and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_every_n_steps: int = 50,
) -> float:
    """
    Run one training epoch.
    
    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    
    for step, (inp, tgt) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        

        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log step-level loss periodically
        if step % log_every_n_steps == 0:
            wandb.log({
                "train/step_loss": loss.item(),
                "train/step": epoch * n_batches + step,
            })
    
    return running_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run validation loop without gradient computation.
    
    Returns:
        Mean validation loss over the loader.
    """
    model.eval()
    total_loss = 0.0
    
    for inp, tgt in loader:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        
        pred = model(inp)
        loss = criterion(pred, tgt)
        total_loss += loss.item()

    
    return total_loss / len(loader)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """Save model + optimizer + metadata to disk."""
    torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)


def train(
    model: nn.Module,
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str = "simvpv2-mmnist",
) -> dict:
    """
    Full training loop with W&B logging, checkpointing, and cosine LR schedule.
    
    Returns:
        dict with best_val_loss and best_epoch.
    """
    set_seed(cfg.seed)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Optimizer + scheduler + loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.n_epochs,
    )
    criterion = nn.MSELoss()
    
    # W&B init. The mode is read from the WANDB_MODE environment variable
    # (set WANDB_MODE=disabled to skip W&B logging entirely).
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config=vars(cfg),
    )
    wandb.watch(model, log="gradients", log_freq=100)
    
    # Checkpoint dir
    ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(cfg.n_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            device, epoch, cfg.log_every_n_steps,
        )
        val_loss = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
            "lr": current_lr,
        })
        
        print(f"Epoch {epoch+1}/{cfg.n_epochs} | "
              f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
              f"lr={current_lr:.2e}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                ckpt_dir / "best.pt",
            )
        
    
    wandb.finish()
    print(f"Best val loss: {best_val_loss:.5f} at epoch {best_epoch+1}")
    
    return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}