"""
Cost of the memory models as the video gets longer.

For each model and each number of observed frames L, on one CPU thread:
    - latency of one predicted frame, once the L observed frames have been
      read (median and p95 over 30 runs, after 5 warmup runs)
    - FLOPs of one predicted frame (torch.utils.flop_counter, which counts
      a multiply-add as 2 FLOPs, unlike fvcore which counts it as 1)
    - memory kept outside the window, in bytes (the window itself is the
      same for every model: 10 frames x 16 tokens x 128 floats)
    - time to read the L observed frames (encode them and build the memory)

The weights don't change the cost, so the models are at random init.
"full" is the control where the window covers all L frames.

Usage:
    python scripts/memory_cost.py               # measure, then plot
    python scripts/memory_cost.py --plot-only   # redraw from results/memory_cost.json
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.flop_counter import FlopCounterMode

from src.models.transformer_window import WindowTransformer, state_size_bytes
from src.models.transformer_rmt import RMTTransformer
from src.models.transformer_knn import KNNTransformer


LENGTHS = [10, 20, 50, 100, 200, 400]
FULL_MAX_LENGTH = 200          # beyond that, full attention gets too slow on one core
MODELS = {
    "window": lambda L: WindowTransformer(L, 1, window=10),
    "rmt": lambda L: RMTTransformer(L, 1, window=10, n_mem=16),
    "knn": lambda L: KNNTransformer(L, 1, window=10, mem_layer=2, topk=32),
    "full": lambda L: WindowTransformer(L, 1, window=L),
}
LABELS = {"window": "Window (no memory)", "rmt": "RMT (16 memory tokens)",
          "knn": "kNN memory (full cache)", "full": "Full context (window = L)"}
COLORS = {"window": "#2a78d6", "rmt": "#eb6834", "knn": "#1baf7a", "full": "#eda100"}
MARKERS = {"window": "o", "rmt": "s", "knn": "^", "full": "D"}


def measure(model, L, n_warmup=5, n_runs=30):
    x = torch.rand(1, L, 1, 64, 64)
    with torch.no_grad():
        start_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            window, state = model.start(x)
            start_times.append((time.perf_counter() - t0) * 1000)
        for _ in range(n_warmup):
            model.step(window, state)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.step(window, state)
            times.append((time.perf_counter() - t0) * 1000)
        with FlopCounterMode(display=False) as counter:
            model.step(window, state)
    times = np.array(times)
    return {
        "step_median_ms": float(np.median(times)),
        "step_p95_ms": float(np.percentile(times, 95)),
        "step_gflops": counter.get_total_flops() / 1e9,
        "memory_bytes": state_size_bytes(state),
        "window_bytes": window.numel() * window.element_size(),
        "read_observed_ms": float(np.median(start_times)),
    }


def plot(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, per_length in results.items():
        Ls = [int(L) for L in per_length]
        lat = [per_length[str(L)]["step_median_ms"] for L in Ls]
        style = dict(color=COLORS[name], marker=MARKERS[name], markersize=6,
                     linewidth=2, label=LABELS[name])
        axes[0].plot(Ls, lat, **style)
        # no point for models or lengths that keep nothing (log scale)
        kept = [(L, per_length[str(L)]["memory_bytes"] / 1e6) for L in Ls
                if per_length[str(L)]["memory_bytes"] > 0]
        if kept:
            axes[1].plot(*zip(*kept), **style)
    axes[0].set_ylabel("Latency per predicted frame (ms, median)")
    axes[1].set_ylabel("Memory outside the window (MB)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Observed frames L")
        ax.set_xticks(LENGTHS)
        ax.set_xticklabels([str(L) for L in LENGTHS])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    axes[1].set_title("Window and full context keep nothing outside the window", fontsize=10)
    fig.suptitle("Cost on one CPU thread as the video gets longer", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    if "--plot-only" in sys.argv:          # redraw from the saved measurements
        plot(json.loads(Path("results/memory_cost.json").read_text()),
             Path("figures/memory_cost.png"))
        return
    torch.set_num_threads(1)
    torch.manual_seed(0)
    results = {}
    for name, build in MODELS.items():
        results[name] = {}
        for L in LENGTHS:
            if name == "full" and L > FULL_MAX_LENGTH:
                continue
            model = build(L).eval()
            r = measure(model, L)
            results[name][str(L)] = r
            print(f"{name:7s} L={L:3d}: step {r['step_median_ms']:7.2f} ms "
                  f"(p95 {r['step_p95_ms']:7.2f})  {r['step_gflops']:6.2f} GFLOPs  "
                  f"memory {r['memory_bytes'] / 1e3:8.1f} kB  "
                  f"read {r['read_observed_ms']:8.1f} ms", flush=True)
    out = Path("results/memory_cost.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")
    plot(results, Path("figures/memory_cost.png"))


if __name__ == "__main__":
    main()
