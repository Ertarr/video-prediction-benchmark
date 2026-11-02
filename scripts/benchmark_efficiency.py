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
        dict with 'mean_ms', 'std_ms', 'median_ms', 'p95_ms',
        'throughput_samples_per_sec'
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
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
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
    print(f"  CPU latency: {lat['mean_ms']:.1f} +/- {lat['std_ms']:.1f} ms "
          f"(median {lat['median_ms']:.1f}, p95 {lat['p95_ms']:.1f})")
    print(f"  Throughput: {lat['throughput_samples_per_sec']:.1f} samples/sec")

    return {
        "name": name,
        "params_M": n_params / 1e6,
        "flops_G": flops / 1e9,
        "latency_mean_ms": lat["mean_ms"],
        "latency_std_ms": lat["std_ms"],
        "latency_median_ms": lat["median_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "throughput": lat["throughput_samples_per_sec"],
    }


def main():
    # One CPU thread, so the numbers do not depend on how many cores the
    # machine has.
    torch.set_num_threads(1)

    cfg = MMNISTConfig()
    in_shape = (cfg.input_len, cfg.n_channels, cfg.image_size, cfg.image_size)
    bench_input_shape = (1, *in_shape)

    print("Benchmark setup:")
    print(f"  Input shape: {bench_input_shape}")
    print(f"  Device: CPU, {torch.get_num_threads()} thread(s)")
    print("  Latency: 5 warmup + 30 measurement runs")

    results = []

    # SimVPv2 baseline (SimVP_Model code defaults, a small config:
    # hid_S=16, hid_T=256, N_S=4, N_T=4)
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

    # Compact with n_layers=3 (best ablation finding)
    compact_n3 = CompactPredictor(
        input_len=cfg.input_len, target_len=cfg.target_len,
        in_shape=in_shape,
        latent_dim=64,
        translator_hidden=256,
        n_layers=3,
    )
    results.append(benchmark_model("Compact n_layers=3", compact_n3, bench_input_shape))

    # Final summary table
    print("\n\n=== Summary ===")
    print(f"{'Model':<20}{'Params (M)':<12}{'FLOPs (G)':<11}"
          f"{'Mean (ms)':<17}{'Median':<9}{'p95':<9}{'Throughput':<14}")
    print("-" * 92)
    for r in results:
        lat_str = f"{r['latency_mean_ms']:.1f} +/- {r['latency_std_ms']:.1f}"
        through_str = f"{r['throughput']:.1f} s/s"
        print(f"{r['name']:<20}{r['params_M']:<12.2f}{r['flops_G']:<11.2f}"
              f"{lat_str:<17}{r['latency_median_ms']:<9.1f}"
              f"{r['latency_p95_ms']:<9.1f}{through_str:<14}")

    ref = results[0]
    for r in results[1:]:
        print(f"\n=== {r['name']} vs {ref['name']} ===")
        print(f"  Params:  {ref['params_M'] / r['params_M']:.2f}x smaller")
        print(f"  FLOPs:   {ref['flops_G'] / r['flops_G']:.2f}x fewer")
        print(f"  Latency: {ref['latency_median_ms'] / r['latency_median_ms']:.2f}x faster (median)")


if __name__ == "__main__":
    main()
