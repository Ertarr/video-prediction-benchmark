"""
Generate a size vs quality plot summarizing all ablation runs.

Reads JSON summaries from `ablations/compact/` and `ablations/jepa/`,
plots params vs test MSE on log-x scale, with markers indicating model
type (Compact / JEPA). Also shows the main Compact and SimVPv2 runs and
the copy-last-frame and black-frame baselines, read from
results/reeval_mmnist.json.

Output: figures/pareto_plot.png
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt


REEVAL_PATH = Path("results/reeval_mmnist.json")
REFERENCE_RUNS = {
    'SimVPv2 (Phase 1b)': 'simvpv2-mmnist-otf',
    'Compact (Phase 1b)': 'compact-mmnist-otf',
}


def load_reference_points():
    """Phase 1b reference points and the copy-last-frame baseline."""
    reeval = json.loads(REEVAL_PATH.read_text())
    points = {
        name: {
            'n_params_M': reeval[run]['n_params'] / 1e6,
            'test_mse_avg': reeval[run]['test_mse_avg'],
            'test_ssim_avg': reeval[run]['test_ssim_avg'],
        }
        for name, run in REFERENCE_RUNS.items()
    }
    return points, reeval['copy-last-frame'], reeval.get('black-frame')


def load_summaries(directory: Path):
    """Load all JSON files from a directory."""
    summaries = []
    if not directory.exists():
        print(f"Warning: {directory} does not exist, skipping")
        return summaries
    for json_path in sorted(directory.glob("*.json")):
        with open(json_path) as f:
            s = json.load(f)
        if 'test_mse_avg' in s:
            summaries.append(s)
    return summaries


def main():
    compact_summaries = load_summaries(Path("ablations/compact"))
    jepa_summaries = load_summaries(Path("ablations/jepa"))
    
    print(f"Loaded {len(compact_summaries)} Compact runs")
    print(f"Loaded {len(jepa_summaries)} JEPA runs")

    REFERENCE_POINTS, copy_baseline, black_baseline = load_reference_points()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---- Subplot 1: params vs MSE ----
    ax = axes[0]
    
    # Compact ablation cloud
    if compact_summaries:
        xs = [s['n_params_M'] for s in compact_summaries]
        ys = [s['test_mse_avg'] for s in compact_summaries]
        ax.scatter(xs, ys, s=60, alpha=0.6, c='tab:blue',
                   label='Compact ablations', marker='o')
    
    # JEPA ablation cloud
    if jepa_summaries:
        xs = [s['n_params_M'] for s in jepa_summaries]
        ys = [s['test_mse_avg'] for s in jepa_summaries]
        ax.scatter(xs, ys, s=60, alpha=0.6, c='tab:orange',
                   label='JEPA ablations', marker='s')
    
    # Reference points
    for name, ref in REFERENCE_POINTS.items():
        marker = '*' if 'SimVPv2' in name else 'D'
        color = 'tab:red' if 'SimVPv2' in name else 'tab:green'
        ax.scatter(
            ref['n_params_M'], ref['test_mse_avg'],
            s=200, c=color, marker=marker, edgecolors='black',
            linewidths=1.5, zorder=10, label=name,
        )
    
    ax.axhline(copy_baseline['test_mse_avg'], color='gray', linestyle='--',
               linewidth=1, label='Copy last frame (no model)')

    ax.set_xscale('log')
    ax.set_xlabel('Params (M, log scale)', fontsize=11)
    ax.set_ylabel('Test MSE (avg over 10 steps)', fontsize=11)
    ax.set_title('Model size vs prediction error', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    
    # ---- Subplot 2: params vs SSIM ----
    ax = axes[1]
    
    if compact_summaries:
        xs = [s['n_params_M'] for s in compact_summaries]
        ys = [s['test_ssim_avg'] for s in compact_summaries]
        ax.scatter(xs, ys, s=60, alpha=0.6, c='tab:blue',
                   label='Compact ablations', marker='o')
    
    if jepa_summaries:
        xs = [s['n_params_M'] for s in jepa_summaries]
        ys = [s['test_ssim_avg'] for s in jepa_summaries]
        ax.scatter(xs, ys, s=60, alpha=0.6, c='tab:orange',
                   label='JEPA ablations', marker='s')
    
    for name, ref in REFERENCE_POINTS.items():
        marker = '*' if 'SimVPv2' in name else 'D'
        color = 'tab:red' if 'SimVPv2' in name else 'tab:green'
        ax.scatter(
            ref['n_params_M'], ref['test_ssim_avg'],
            s=200, c=color, marker=marker, edgecolors='black',
            linewidths=1.5, zorder=10, label=name,
        )
    
    ax.axhline(copy_baseline['test_ssim_avg'], color='gray', linestyle='--',
               linewidth=1, label='Copy last frame (no model)')
    if black_baseline is not None:
        ax.axhline(black_baseline['test_ssim_avg'], color='gray', linestyle=':',
                   linewidth=1.5, label='Black frame (no model)')

    ax.set_xscale('log')
    ax.set_xlabel('Params (M, log scale)', fontsize=11)
    ax.set_ylabel('Test SSIM (avg over 10 steps)', fontsize=11)
    ax.set_title('Model size vs SSIM', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    fig.suptitle(
        'Ablations on Moving MNIST OTF (single seed per run)',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = Path("figures/pareto_plot.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved size vs quality plot to {out}")


if __name__ == "__main__":
    main()
