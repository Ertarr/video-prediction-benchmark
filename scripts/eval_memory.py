"""
Forgetting curve for the memory models: for k in (0, 5, 10, 20, 40), 1000
test sequences with digit B hidden during the last k observed frames.
Reports MSE and SSIM on B's 28x28 square (zone) and on the full frame,
per predicted step.

kNN and RMT are also scored with their memory limited at test time to
the last 10 or 20 frames before the window (cache_frames, memory_frames),
without retraining.

Usage:
    python scripts/eval_memory.py --model knn --ckpt checkpoints/memory-knn-seed1/best.pt \
        --mnist-root data/mnist
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

from src.datasets.moving_mnist_occlusion import MovingMNISTOcclusion
from src.models.compact.compact_predictor import CompactPredictor
from src.models.transformer_window import WindowTransformer
from src.models.transformer_rmt import RMTTransformer
from src.models.transformer_knn import KNNTransformer


KS = [0, 5, 10, 20, 40]
TEST_SEED = 8_000_000
DIGIT = 28


def build_memory_model(name: str) -> nn.Module:
    if name == "window":
        return WindowTransformer(input_len=50, target_len=10, window=10)
    if name == "rmt":
        return RMTTransformer(input_len=50, target_len=10, window=10, n_mem=16)
    if name == "knn":
        return KNNTransformer(input_len=50, target_len=10, window=10, mem_layer=2, topk=32)
    if name == "full":      # control: attention over all 50 observed frames
        return WindowTransformer(input_len=50, target_len=10, window=50)
    raise ValueError(name)


class CompactLastFrames(nn.Module):
    """Compact (trained on plain Moving MNIST) fed with the last 10 frames."""

    def __init__(self, compact):
        super().__init__()
        self.compact = compact

    def forward(self, x):
        return self.compact(x[:, -10:])


def evaluate_occlusion(model, mnist_root, k, device, n_samples=1000, batch_size=32):
    ds = MovingMNISTOcclusion(mnist_root, n_samples=n_samples, k=k, seed=TEST_SEED,
                              mnist_train=False, return_meta=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    n_pred = ds.n_predict
    sums = {"mse": np.zeros(n_pred), "ssim": np.zeros(n_pred),
            "zone_mse": np.zeros(n_pred), "zone_ssim": np.zeros(n_pred)}
    n = 0
    model.eval()
    with torch.no_grad():
        for inp, tgt, meta in loader:
            pred = model(inp.to(device)).clamp(0, 1).cpu().numpy()
            tgt = tgt.numpy()
            pos = meta["positions"].numpy()                 # (B, 60, 2, 2)
            for b in range(pred.shape[0]):
                for t in range(n_pred):
                    p, g = pred[b, t, 0], tgt[b, t, 0]
                    x, y = pos[b, ds.n_observed + t, 1]
                    pz, gz = p[y:y + DIGIT, x:x + DIGIT], g[y:y + DIGIT, x:x + DIGIT]
                    sums["mse"][t] += ((p - g) ** 2).mean()
                    sums["ssim"][t] += ssim_fn(g, p, data_range=1.0)
                    sums["zone_mse"][t] += ((pz - gz) ** 2).mean()
                    sums["zone_ssim"][t] += ssim_fn(gz, pz, data_range=1.0)
            n += pred.shape[0]
    out = {f"{key}_per_step": (v / n).tolist() for key, v in sums.items()}
    for key in sums:
        out[f"{key}_avg"] = float(np.mean(out[f"{key}_per_step"]))
    return out


def forgetting_curve(model, mnist_root, device, n_samples=1000, ks=KS):
    curve = {}
    for k in ks:
        curve[str(k)] = evaluate_occlusion(model, mnist_root, k, device, n_samples)
        print(f"  k={k:2d}: zone MSE {curve[str(k)]['zone_mse_avg']:.4f}  "
              f"zone SSIM {curve[str(k)]['zone_ssim_avg']:.3f}  "
              f"full MSE {curve[str(k)]['mse_avg']:.4f}", flush=True)
    return curve


def evaluate_model(name, model, mnist_root, device, n_samples=1000):
    result = {"model": name, "curve": forgetting_curve(model, mnist_root, device, n_samples)}
    if isinstance(model, KNNTransformer):
        result["curve_by_cache"] = {}
        for frames in [20, 10]:
            print(f"  kNN cache limited to {frames} frames")
            model.cache_frames = frames
            result["curve_by_cache"][str(frames)] = forgetting_curve(
                model, mnist_root, device, n_samples)
        model.cache_frames = None
    if isinstance(model, RMTTransformer):
        result["curve_by_memory"] = {}
        for frames in [20, 10]:
            print(f"  RMT memory reading only the last {frames} frames")
            model.memory_frames = frames
            result["curve_by_memory"][str(frames)] = forgetting_curve(
                model, mnist_root, device, n_samples)
        model.memory_frames = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["window", "rmt", "knn", "full", "compact"],
                        required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mnist-root", default="data/mnist")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.model == "compact":
        compact = CompactPredictor(10, 10, (10, 1, 64, 64), latent_dim=64,
                                   translator_hidden=256, n_layers=3)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        compact.load_state_dict(ckpt["model_state_dict"])
        model = CompactLastFrames(compact)
    else:
        model = build_memory_model(args.model)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    result = evaluate_model(args.model, model, args.mnist_root, device, args.n_samples)
    result["ckpt"] = str(args.ckpt)
    out = Path(args.out or f"results/memory/eval-{args.model}-{Path(args.ckpt).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
