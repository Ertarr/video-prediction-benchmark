"""
Check that an interrupted run resumes from last.pt and ends with the same
weights as a run that was never interrupted.

    pytest tests/test_trainer.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["WANDB_MODE"] = "disabled"

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import train


class CrashOnSecondPass:
    """Validation loader that fails the second time it is used (a disconnect)."""

    def __init__(self, loader):
        self.loader, self.passes = loader, 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.passes += 1
        if self.passes == 2:
            raise RuntimeError("disconnected")
        return iter(self.loader)


def run(ckpt_dir, crash=False):
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(64, 4), torch.randn(64, 4))
    model = nn.Linear(4, 4)
    cfg = SimpleNamespace(seed=1, device="cpu", lr=1e-2, weight_decay=0.0, n_epochs=3,
                          wandb_project="test", checkpoint_dir=str(ckpt_dir),
                          log_every_n_steps=100)
    train_loader = DataLoader(data, batch_size=8, shuffle=True)
    val_loader = DataLoader(data, batch_size=8)
    if crash:
        val_loader = CrashOnSecondPass(val_loader)
    result = train(model, cfg, train_loader, val_loader, run_name="run")
    return model, result


def test_resume_gives_the_same_run(tmp_path):
    full_model, full_result = run(tmp_path / "full")

    with pytest.raises(RuntimeError):
        run(tmp_path / "cut", crash=True)
    last = torch.load(tmp_path / "cut" / "run" / "last.pt", weights_only=True)
    assert last["epoch"] == 0

    resumed_model, resumed_result = run(tmp_path / "cut")
    assert torch.equal(resumed_model.weight, full_model.weight)
    assert torch.equal(resumed_model.bias, full_model.bias)
    assert resumed_result == full_result
