"""
Checks for the memory models and the occlusion dataset.

    pytest tests/test_memory.py

The dataset tests need MNIST: set VPB_MNIST_ROOT to the folder that
contains MNIST/raw (they are skipped otherwise).
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from src.models.transformer_window import WindowTransformer
from src.models.transformer_window.core import attention, TOKENS_PER_FRAME
from src.models.transformer_rmt import RMTTransformer
from src.models.transformer_knn import KNNTransformer
from src.models.transformer_knn.knn_predictor import knn_attention


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 4, 5, 8).unbind(0)
    assert torch.allclose(attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-6)


def test_knn_attention_with_k_at_least_memory_size_is_full_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 8)
    mem_k, mem_v = torch.randn(2, 2, 4, 7, 8).unbind(0)
    assert torch.allclose(knn_attention(q, mem_k, mem_v, topk=7), attention(q, mem_k, mem_v), atol=1e-6)
    assert torch.allclose(knn_attention(q, mem_k, mem_v, topk=100), attention(q, mem_k, mem_v), atol=1e-6)


def test_knn_attention_only_uses_top_k():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 4)
    mem_k = torch.randn(1, 1, 6, 4)
    mem_v = torch.randn(1, 1, 6, 4)
    scores = (q @ mem_k.transpose(-2, -1)).squeeze()
    best = scores.argmax()
    out = knn_attention(q, mem_k, mem_v, topk=1)
    assert torch.allclose(out.squeeze(), mem_v[0, 0, best])


@pytest.mark.parametrize("cls", [WindowTransformer, RMTTransformer, KNNTransformer])
def test_shapes(cls):
    model = cls(input_len=25, target_len=3, window=5)
    out = model(torch.rand(2, 25, 1, 64, 64))
    assert out.shape == (2, 3, 1, 64, 64)


def test_rmt_gradient_reaches_first_segment():
    torch.manual_seed(0)
    model = RMTTransformer(input_len=25, target_len=2, window=5, n_mem=4)
    x = torch.rand(1, 25, 1, 64, 64, requires_grad=True)
    model(x).pow(2).mean().backward()
    assert model.mem_init.grad.abs().sum() > 0
    # the first frame is only seen through the memory (it is never in the window)
    assert x.grad[:, 0].abs().sum() > 0


def test_window_model_ignores_frames_before_the_window():
    torch.manual_seed(0)
    model = WindowTransformer(input_len=25, target_len=2, window=5).eval()
    x = torch.rand(1, 25, 1, 64, 64)
    x2 = x.clone()
    x2[:, :20] = torch.rand(1, 20, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x2))


def test_knn_cache_size_and_trim():
    model = KNNTransformer(input_len=25, target_len=3, window=5)
    with torch.no_grad():
        window, state = model.start(torch.rand(2, 25, 1, 64, 64))
        assert state["k"].shape[2] == 20 * TOKENS_PER_FRAME
        _, window, state = model.step(window, state)
        assert state["k"].shape[2] == 21 * TOKENS_PER_FRAME
        model.cache_frames = 3
        _, window, state = model.step(window, state)
        assert state["k"].shape[2] == 3 * TOKENS_PER_FRAME


def test_rmt_memory_frames_only_reads_the_last_frames():
    torch.manual_seed(0)
    model = RMTTransformer(input_len=25, target_len=2, window=5, memory_frames=5).eval()
    x = torch.rand(1, 25, 1, 64, 64)
    x2 = x.clone()
    x2[:, :15] = torch.rand(1, 15, 1, 64, 64)      # frames 15..19 are still read
    x3 = x.clone()
    x3[:, 15:20] = torch.rand(1, 5, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x2))
        assert not torch.allclose(model(x), model(x3))


MNIST_ROOT = os.environ.get("VPB_MNIST_ROOT")


@pytest.mark.skipif(MNIST_ROOT is None, reason="set VPB_MNIST_ROOT")
def test_occlusion_hides_digit_b_in_the_last_k_observed_frames():
    from src.datasets.moving_mnist_occlusion import MovingMNISTOcclusion
    ds = MovingMNISTOcclusion(MNIST_ROOT, n_samples=4, k=20, seed=0, return_meta=True)
    inp, tgt, meta = ds[0]
    x, y = meta["positions"][35, 1]                  # an occluded frame (30..49)
    assert torch.all(inp[35, 0, y:y + 28, x:x + 28] == 0.5)
    x, y = meta["positions"][20, 1]                  # before the occlusion
    assert not torch.all(inp[20, 0, y:y + 28, x:x + 28] == 0.5)
    assert tgt.shape == (10, 1, 64, 64)


@pytest.mark.skipif(MNIST_ROOT is None, reason="set VPB_MNIST_ROOT")
def test_occlusion_is_deterministic_with_a_seed():
    from src.datasets.moving_mnist_occlusion import MovingMNISTOcclusion
    ds = MovingMNISTOcclusion(MNIST_ROOT, n_samples=4, seed=3)
    assert torch.equal(ds[1][0], ds[1][0])
