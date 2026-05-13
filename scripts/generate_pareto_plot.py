"""
Generate a Pareto plot summarizing all ablation runs.

Reads JSON summaries from `ablations/compact/` and `ablations/jepa/`,
plots params vs test MSE on log-x scale, with markers indicating model
type (Compact / JEPA). Also overlays the Phase 1b "main" Compact and
SimVPv2 results as reference points.

Output: figures/pareto_plot.png
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt


# Phase 1b reference points (from RESULTS.md, OTF eval)
REFERENCE_POINTS = {
    'SimVPv2 (Phase 1b)': {
        'n_params_M': 4.72,
        'test_mse_avg': 0.01685,
        'test_ssim_avg': 0.84,
    },
    'Compact (Phase 1b)': {
        'n_params_M': 0.98,
        'test_mse_avg': 0.02628,
        'test_ssim_avg': 0.46,
    },
}


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
    
    ax.set_xscale('log')
    ax.set_xlabel('Params (M, log scale)', fontsize=11)
    ax.set_ylabel('Test MSE (avg over 10 steps)', fontsize=11)
    ax.set_title('Pareto: model size vs prediction error', fontsize=12)
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
    
    ax.set_xscale('log')
    ax.set_xlabel('Params (M, log scale)', fontsize=11)
    ax.set_ylabel('Test SSIM (avg over 10 steps)', fontsize=11)
    ax.set_title('Pareto: model size vs perceptual quality', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    
    fig.suptitle(
        'Phase 3+ ablations: Compact and JEPA on Moving MNIST OTF',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    
    out = Path("figures/pareto_plot.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved Pareto plot to {out}")


if __name__ == "__main__":
    main()
