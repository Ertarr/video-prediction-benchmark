"""
Benchmark computational efficiency of video prediction models.

Reports for each model:
    - Total trainable parameters (M)
    - FLOPs per forward pass (G)
    - Mean / std CPU inference latency over N runs (ms)
    - Throughput (samples / second on CPU)

Compares SimVPv2 baseline against the Compact predictor.
All measurements done with batch_size=1 on CPU, after warm-up runs.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import MMNISTConfig
from src.models.simvpv2 import SimVPv2
from src.models.compact.compact_predictor import CompactPredictor


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """
    Measure forward pass FLOPs using fvcore.
    
    Args:
        model: model in eval mode
        input_shape: (B, T, C, H, W)
    
    Returns:
        Total FLOPs as integer.
    """
    from fvcore.nn import FlopCountAnalysis
    
    dummy = torch.randn(*input_shape)
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total()


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple,
    n_warmup: int = 5,
    n_runs: int = 30,
) -> dict:
    """
    Measure CPU inference latency.
    
    Args:
        model: model in eval mode, on CPU
        input_shape: (B, T, C, H, W) - typically batch_size=1 for latency
        n_warmup: warmup runs (CPU caches, JIT compilation)
        n_runs: actual measurement runs
    
    Returns:
        dict with 'mean_ms', 'std_ms', 'throughput_samples_per_sec'
    """
    model.eval()
    dummy = torch.randn(*input_shape)
    
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    batch_size = input_shape[0]
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "throughput_samples_per_sec": batch_size / (times.mean() / 1000),
    }


def benchmark_model(
    name: str,
    model: torch.nn.Module,
    input_shape: tuple,
) -> dict:
    """Run all benchmarks on a single model and return a summary dict."""
    model.eval()
    
    print(f"\n=== Benchmarking {name} ===")
    
    n_params = count_params(model)
    print(f"  Params: {n_params / 1e6:.2f}M")
    
    flops = measure_flops(model, input_shape)
    print(f"  FLOPs: {flops / 1e9:.2f}G (per forward pass, batch_size={input_shape[0]})")
    
    lat = measure_latency(model, input_shape)
    print(f"  CPU latency: {lat['mean_ms']:.1f} +/- {lat['std_ms']:.1f} ms")
    print(f"  Throughput: {lat['throughput_samples_per_sec']:.1f} samples/sec")
    
    return {
        "name": name,
        "params_M": n_params / 1e6,
        "flops_G": flops / 1e9,
        "latency_mean_ms": lat["mean_ms"],
        "latency_std_ms": lat["std_ms"],
        "throughput": lat["throughput_samples_per_sec"],
    }


def main():
    cfg = MMNISTConfig()
    in_shape = (cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size)
    bench_input_shape = (1, *in_shape)
    
    print(f"Benchmark setup:")
    print(f"  Input shape: {bench_input_shape}")
    print(f"  Device: CPU")
    print(f"  Latency: 5 warmup + 30 measurement runs")
    
    results = []
    
    # SimVPv2 baseline (default OpenSTL config)
    simvpv2 = SimVPv2(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=in_shape,
    )
    results.append(benchmark_model("SimVPv2 (gSTA)", simvpv2, bench_input_shape))
    
    # Compact: same config as on Moving MNIST (translator_hidden=256, n_layers=2)
    compact = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=in_shape,
        latent_dim=64,
        translator_hidden=256,
        n_layers=2,
    )
    results.append(benchmark_model("Compact (ours)", compact, bench_input_shape))
    
    # Final summary table
    print("\n\n=== Summary ===")
    print(f"{'Model':<20}{'Params (M)':<14}{'FLOPs (G)':<14}"
          f"{'Latency (ms)':<18}{'Throughput':<14}")
    print("-" * 80)
    for r in results:
        lat_str = f"{r['latency_mean_ms']:.1f} +/- {r['latency_std_ms']:.1f}"
        through_str = f"{r['throughput']:.1f} s/s"
        print(f"{r['name']:<20}{r['params_M']:<14.2f}{r['flops_G']:<14.2f}"
              f"{lat_str:<18}{through_str:<14}")
    
    if len(results) == 2:
        ratio_params = results[0]["params_M"] / results[1]["params_M"]
        ratio_flops = results[0]["flops_G"] / results[1]["flops_G"]
        ratio_latency = results[0]["latency_mean_ms"] / results[1]["latency_mean_ms"]
        print("\n=== Compact gains vs SimVPv2 ===")
        print(f"  Params:  {ratio_params:.2f}x smaller")
        print(f"  FLOPs:   {ratio_flops:.2f}x fewer")
        print(f"  Latency: {ratio_latency:.2f}x faster")


if __name__ == "__main__":
    main()
