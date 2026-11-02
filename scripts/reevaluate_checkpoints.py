"""
Re-evaluate every Moving MNIST checkpoint on the Toronto test split, with
the copy-last-frame and black-frame baselines. The JSON files of the
first Colab evaluations were lost, so everything is recomputed from the
checkpoints.

Outputs:
    results/reeval_mmnist.json          main runs + baselines
    ablations/compact/<run_name>.json   same format as ablations_compact.py
    ablations/jepa/<run_name>.json      same format as ablations_jepa.py
    results/reeval_kth.json             KTH runs + baseline (if --kth-root)

Usage:
    python scripts/reevaluate_checkpoints.py \
        --data-root path/to/data --ckpt-root path/to/checkpoints \
        [--kth-root path/to/kth] [--skip-mmnist]
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets.moving_mnist import MovingMNIST
from src.datasets.kth_actions import KTHActions
from src.models.base import VideoPredictor
from src.models.simvpv2 import SimVPv2
from src.models.compact.compact_predictor import CompactPredictor
from src.models.compact_jepa.jepa_predictor import JEPAPredictor
from src.evaluation.metrics import evaluate
from scripts.eval_jepa_mmnist import evaluate_jepa
from scripts.ablations_compact import SWEEPS as COMPACT_SWEEPS
from scripts.ablations_compact import DEFAULTS as COMPACT_DEFAULTS
from scripts.ablations_jepa import SWEEPS as JEPA_SWEEPS
from scripts.ablations_jepa import DEFAULTS as JEPA_DEFAULTS


IN_SHAPE = (10, 1, 64, 64)


class CopyLastFrame(VideoPredictor):
    """Trivial baseline: repeat the last input frame for every future step."""

    def forward(self, x):
        return x[:, -1:].repeat(1, self.target_len, 1, 1, 1)


class BlackFrame(VideoPredictor):
    """Trivial baseline: predict an empty (all black) frame at every step."""

    def forward(self, x):
        return torch.zeros_like(x[:, -1:]).repeat(1, self.target_len, 1, 1, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def build_compact(hparams):
    return CompactPredictor(10, 10, IN_SHAPE, **hparams)


def build_jepa(hparams):
    return JEPAPredictor(
        10, 10, IN_SHAPE, latent_dim=64, translator_hidden=256, n_layers=2,
        ema_momentum=hparams["ema_momentum"],
        use_decoder=hparams["decoder_coef"] > 0,
    )


def summarize(mse, ssim):
    mse, ssim = list(map(float, mse)), list(map(float, ssim))
    return {
        "test_mse_per_step": mse,
        "test_ssim_per_step": ssim,
        "test_mse_avg": sum(mse) / len(mse),
        "test_ssim_avg": sum(ssim) / len(ssim),
    }


def eval_pixel_model(model, loader):
    res = evaluate(model, loader, torch.device("cpu"))
    return summarize(res["mse_per_step"], res["ssim_per_step"])


def eval_jepa_model(model, loader):
    if not model.use_decoder:
        # No decoder -> no pixel metrics, as in ablations_jepa.py
        return {"test_mse_per_step": None, "test_ssim_per_step": None}
    res = evaluate_jepa(model, loader, torch.device("cpu"))
    return summarize(res["mse_per_step"], res["ssim_per_step"])


def ablation_runs(prefix, sweeps, defaults):
    """Rebuild (run_name, hparams) exactly as the ablation scripts did."""
    for axis, values in sweeps.items():
        for value in values:
            hparams = defaults.copy()
            hparams[axis] = value
            yield f"{prefix}-abl-{axis}={value}", hparams


def reevaluate_kth(kth_root, ckpt_root, batch_size):
    """KTH test set: 10 input frames, 20 predicted frames, 128x128."""
    test_ds = KTHActions(
        Path(kth_root) / "frames", Path(kth_root) / "test_clips.txt",
        input_len=10, target_len=20, image_size=128,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    kth_shape = (10, 1, 128, 128)
    runs = {
        "copy-last-frame": ("baseline, no training", None),
        "simvpv2-kth": ("SimVPv2, 100 ep", lambda: SimVPv2(10, 20, kth_shape)),
        "compact-kth": ("Compact n_layers=2, translator_hidden=128, 100 ep",
                        lambda: CompactPredictor(10, 20, kth_shape, latent_dim=64,
                                                 translator_hidden=128, n_layers=2)),
        "compact-kth-n3": ("Compact n_layers=3, translator_hidden=128, 100 ep",
                           lambda: CompactPredictor(10, 20, kth_shape, latent_dim=64,
                                                    translator_hidden=128, n_layers=3)),
    }
    results = {"n_test_clips": len(test_ds)}
    for run_name, (description, build) in runs.items():
        print(f"Evaluating KTH {run_name} ...", flush=True)
        if build is None:
            model, entry = CopyLastFrame(10, 20), {"description": description, "n_params": 0}
        else:
            model, ckpt = load_checkpoint(build(), Path(ckpt_root) / run_name / "best.pt")
            entry = {
                "description": description,
                "n_params": count_params(model),
                "best_epoch_saved": int(ckpt["epoch"]),
                "best_val_loss": float(ckpt["val_loss"]),
            }
        entry.update(eval_pixel_model(model, loader))
        results[run_name] = entry
        print(f"  MSE avg {entry['test_mse_avg']:.5f}  SSIM avg {entry['test_ssim_avg']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--kth-root", default=None,
                        help="folder with frames/ and test_clips.txt (optional)")
    parser.add_argument("--skip-mmnist", action="store_true")
    parser.add_argument("--out", default="results/reeval_mmnist.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.kth_root is not None:
        kth = reevaluate_kth(args.kth_root, args.ckpt_root, args.batch_size)
        out = Path("results/reeval_kth.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(kth, indent=2))
        print(f"Saved {out}")
    if args.skip_mmnist:
        return

    npy = Path(args.data_root) / "moving_mnist" / "mnist_test_seq.npy"
    test_ds = MovingMNIST(npy, split="test", input_len=10, target_len=10)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    ckpt_root = Path(args.ckpt_root)
    torch.manual_seed(0)  # SimVPv2 / Compact eval is deterministic anyway

    # ---- Main runs + trivial baseline ----
    main_runs = {
        "copy-last-frame": ("baseline, no training", None),
        "black-frame": ("baseline, predicts an empty frame", "black"),
        "simvpv2-mmnist": ("SimVPv2, fixed split, 200 ep", lambda: SimVPv2(10, 10, IN_SHAPE)),
        "simvpv2-mmnist-otf": ("SimVPv2, on-the-fly, 200 ep", lambda: SimVPv2(10, 10, IN_SHAPE)),
        "compact-mmnist": ("Compact n_layers=2, fixed split, 200 ep",
                           lambda: build_compact(COMPACT_DEFAULTS)),
        "compact-mmnist-otf": ("Compact n_layers=2, on-the-fly, 200 ep",
                               lambda: build_compact(COMPACT_DEFAULTS)),
        "compact-mmnist-otf-n3": ("Compact n_layers=3, on-the-fly, 200 ep",
                                  lambda: build_compact({**COMPACT_DEFAULTS, "n_layers": 3})),
        "jepa-mmnist-otf": ("JEPA defaults, on-the-fly, 200 ep",
                            lambda: build_jepa(JEPA_DEFAULTS)),
        "jepa-mmnist-otf-best": ("JEPA after ablations, on-the-fly, 200 ep",
                                 lambda: build_jepa({**JEPA_DEFAULTS, "ema_momentum": 0.999,
                                                     "var_coef": 0.0, "cov_coef": 0.0})),
    }

    results = {}
    for run_name, (description, build) in main_runs.items():
        print(f"Evaluating {run_name} ...", flush=True)
        if build is None or build == "black":
            model = CopyLastFrame(10, 10) if build is None else BlackFrame(10, 10)
            entry = {"description": description, "n_params": 0}
            entry.update(eval_pixel_model(model, loader))
        else:
            model, ckpt = load_checkpoint(build(), ckpt_root / run_name / "best.pt")
            entry = {
                "description": description,
                "n_params": count_params(model),
                "best_epoch_saved": int(ckpt["epoch"]),
                "best_val_loss": float(ckpt["val_loss"]),
            }
            is_jepa = isinstance(model, JEPAPredictor)
            entry.update((eval_jepa_model if is_jepa else eval_pixel_model)(model, loader))
        results[run_name] = entry
        if entry.get("test_mse_avg") is not None:
            print(f"  MSE avg {entry['test_mse_avg']:.5f}  SSIM avg {entry['test_ssim_avg']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")

    # ---- Ablation summaries, same format as the ablation scripts ----
    groups = [
        ("compact", COMPACT_SWEEPS, COMPACT_DEFAULTS, build_compact, eval_pixel_model),
        ("jepa", JEPA_SWEEPS, JEPA_DEFAULTS, build_jepa, eval_jepa_model),
    ]
    for prefix, sweeps, defaults, build, run_eval in groups:
        for run_name, hparams in ablation_runs(prefix, sweeps, defaults):
            print(f"Evaluating {run_name} ...", flush=True)
            model, ckpt = load_checkpoint(build(hparams), ckpt_root / run_name / "best.pt")
            n_params = count_params(model)
            summary = {
                "run_name": run_name,
                "hparams": hparams,
                "n_params": n_params,
                "n_params_M": n_params / 1e6,
                "best_val_loss": float(ckpt["val_loss"]),
                "best_epoch": int(ckpt["epoch"]),
                "source": "rebuilt from the checkpoint by scripts/reevaluate_checkpoints.py",
            }
            summary.update(run_eval(model, loader))
            path = Path("ablations") / prefix / f"{run_name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
