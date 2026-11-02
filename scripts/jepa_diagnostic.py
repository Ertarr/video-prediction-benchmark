"""
Phase 7: did the JEPA encoders collapse, and what do the predicted
latents contain? CPU only, existing checkpoints.

collapse: share of the representation that depends on the input, and
effective rank (RankMe, Garrido et al. 2023), on 2000 test frames, for
every JEPA, the Compact encoders and a random encoder. Complete collapse
if the share is below 1% or the rank below 5, dimensional collapse if
the rank is below half of Compact's.

probe: ridge regression from the frozen predicted latents to the future
digit centers at horizons 1, 5 and 10, with the latents centered and
divided by one global scale. Floor: the same JEPA at random init.
Ceiling: the latents of the true future frame. The two digits are
interchangeable, so the error keeps the better of the two assignments.

Usage:
    python scripts/jepa_diagnostic.py collapse --data-root data --ckpt-root checkpoints
    python scripts/jepa_diagnostic.py probe --data-root data --ckpt-root checkpoints
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.datasets.moving_mnist import MovingMNIST
from src.datasets.moving_mnist_speed import MovingMNISTFixedSpeed
from src.models.compact.compact_predictor import CompactPredictor, CompactEncoder
from src.models.compact_jepa.jepa_predictor import JEPAPredictor
from scripts.reevaluate_checkpoints import (
    build_jepa, ablation_runs, JEPA_SWEEPS, JEPA_DEFAULTS, COMPACT_DEFAULTS,
)


JEPA_MAIN = {
    "jepa-mmnist-otf": JEPA_DEFAULTS,
    "jepa-mmnist-otf-best": {**JEPA_DEFAULTS, "ema_momentum": 0.999,
                             "var_coef": 0.0, "cov_coef": 0.0},
}
COMPACT_N2 = ["compact-mmnist-otf", "compact-abl-latent_dim=64",
              "compact-abl-translator_hidden=256", "compact-abl-n_layers=2"]
HORIZONS = [1, 5, 10]
DIGIT_HALF = 14     # positions are top-left corners of 28x28 digits


def load(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def jepa_checkpoints():
    """(name, hparams) for the 2 main JEPA runs and the 11 ablations."""
    runs = list(JEPA_MAIN.items())
    runs += list(ablation_runs("jepa", JEPA_SWEEPS, JEPA_DEFAULTS))
    return runs


# ---- collapse ----

def effective_rank(Z: torch.Tensor) -> float:
    """RankMe: exp(entropy) of the normalized singular values of Z (N, D)."""
    s = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def collapse_stats(encoder, frames):
    with torch.no_grad():
        z = encoder(frames)                      # (N, 64, 16, 16)
    N, C, H, W = z.shape
    Z = z.reshape(N, -1).double()
    rms = Z.pow(2).mean().sqrt()
    std = Z.std(dim=0)
    tokens = z.permute(0, 2, 3, 1).reshape(-1, C).double()
    return {
        "rms": float(rms),
        "var_share": float(Z.var(dim=0).sum() / Z.pow(2).sum(dim=1).mean()),
        "median_dim_std_rel": float(std.median() / rms),
        "dead_dims_frac": float((std < 0.01 * rms).double().mean()),
        "effective_rank": effective_rank(Z.float()),
        "channel_effective_rank": effective_rank(tokens.float()),
        "last_groupnorm_abs_gamma": float(encoder.layers[-2].weight.abs().mean()),
    }


def run_collapse(args):
    test = MovingMNIST(Path(args.data_root) / "moving_mnist" / "mnist_test_seq.npy",
                       split="test", input_len=10, target_len=10)
    # 2 frames per test sequence (first input frame, 6th target frame)
    frames = torch.cat([torch.stack([test[i][0][0], test[i][1][5]])
                        for i in range(len(test))])
    print(f"{frames.shape[0]} frames")
    ckpt_root = Path(args.ckpt_root)
    reeval = json.loads(Path("results/reeval_mmnist.json").read_text())

    results = {}
    torch.manual_seed(0)
    results["random encoder"] = collapse_stats(CompactEncoder(1, 64, 2).eval(), frames)

    for name in COMPACT_N2:
        model, ckpt = load(CompactPredictor(10, 10, (10, 1, 64, 64), **COMPACT_DEFAULTS),
                           ckpt_root / name / "best.pt")
        results[name] = collapse_stats(model.encoder, frames)

    for name, hp in jepa_checkpoints():
        model, ckpt = load(build_jepa(hp), ckpt_root / name / "best.pt")
        entry = {
            "hparams": hp,
            "best_val_latent_loss": float(ckpt["val_loss"]),
            "target": collapse_stats(model.target_encoder.target_encoder, frames),
            "online": collapse_stats(model.online_encoder, frames),
        }
        summ = reeval.get(name) or json.loads(
            (Path("ablations/jepa") / f"{name}.json").read_text())
        entry["test_ssim_avg"] = summ.get("test_ssim_avg")
        results[name] = entry

    compact_rank = results["compact-mmnist-otf"]["effective_rank"]
    print(f"\n{'Encoder':38s} {'var_share':>9} {'rel std':>8} {'eRank':>7} "
          f"{'ch eRank':>8} {'|gamma|':>8} {'verdict':>12}")
    for name, r in results.items():
        rows = [(name, r)] if "target" not in r else \
               [(name + " [target]", r["target"]), (name + " [online]", r["online"])]
        for label, s in rows:
            if s["var_share"] < 0.01 or s["effective_rank"] < 5:
                verdict = "complete"
            elif s["effective_rank"] < 0.5 * compact_rank:
                verdict = "dimensional"
            else:
                verdict = "none"
            s["verdict"] = verdict
            print(f"{label:38s} {s['var_share']:9.3f} {s['median_dim_std_rel']:8.3f} "
                  f"{s['effective_rank']:7.1f} {s['channel_effective_rank']:8.1f} "
                  f"{s['last_groupnorm_abs_gamma']:8.3f} {verdict:>12}")

    out = Path("results/jepa_collapse.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


# ---- probe ----

def probe_data(mnist_root, n, seed, mnist_train):
    ds = MovingMNISTFixedSpeed(mnist_root, speed=None, n_samples=n, seed=seed,
                               mnist_train=mnist_train, return_positions=True)
    loader = DataLoader(ds, batch_size=100, shuffle=False)
    inps, tgts, poss = [], [], []
    for inp, tgt, pos in loader:
        inps.append(inp); tgts.append(tgt); poss.append(pos)
    inp, tgt, pos = torch.cat(inps), torch.cat(tgts), torch.cat(poss)
    # digit centers, ordered by x at the last observed frame (index 9)
    centers = pos.double() + DIGIT_HALF                   # (N, 20, 2, 2)
    swap = centers[:, 9, 0, 0] > centers[:, 9, 1, 0]
    centers[swap] = centers[swap].flip(dims=[2])
    return inp, tgt, centers


def features(kind, model, inp, tgt):
    """
    (N, len(HORIZONS), D) float32 features.
    kind = 'pred': predicted latents for the future frames
    kind = 'ceil': target latents of the true future frames
    """
    out = []
    with torch.no_grad():
        for i in range(0, inp.shape[0], 100):
            x, y = inp[i:i + 100], tgt[i:i + 100]
            if isinstance(model, JEPAPredictor):
                z = (model.predictor(model.encode_online(x)) if kind == "pred"
                     else model.encode_target(y))
            else:   # Compact: transition output, or encoder on the true frames
                enc = lambda f: model.encoder(f.reshape(-1, *f.shape[2:])).reshape(
                    f.shape[0], f.shape[1], -1)
                if kind == "pred":
                    lat = enc(x).reshape(x.shape[0], 10, 64, 16, 16)
                    z = model.transition(lat)
                else:
                    z = enc(y)
            z = z.reshape(z.shape[0], 10, -1)[:, [h - 1 for h in HORIZONS]]
            out.append(z.float())
    return torch.cat(out)


def assignment_errors(pred, true):
    """Squared error per sequence, keeping the better digit assignment."""
    direct = ((pred - true) ** 2).sum(dim=1)
    swapped = ((pred - true[:, [2, 3, 0, 1]]) ** 2).sum(dim=1)
    return torch.minimum(direct, swapped)


def ridge_probe(Xtr, Ytr, Xva, Yva, Xte, Yte):
    """
    Dual ridge, weight picked on validation. Features are centered and
    divided by one global scale (not per dimension): the probe doesn't
    care about the overall size of the latents, and nearly constant
    dimensions are not blown up by a tiny standard deviation.
    """
    mu = Xtr.mean(0)
    scale = Xtr.var(0).mean().sqrt().clamp_min(1e-12)
    Xtr, Xva, Xte = [(X - mu) / scale for X in (Xtr, Xva, Xte)]
    y_mean = Ytr.mean(0)
    K = (Xtr @ Xtr.T).double()
    evals, U = torch.linalg.eigh(K)
    UtY = U.T @ (Ytr - y_mean)
    Kva, Kte = (Xva @ Xtr.T).double(), (Xte @ Xtr.T).double()
    best = None
    for c in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        lam = c * K.diagonal().mean()
        alpha = U @ (UtY / (evals + lam)[:, None])
        val_err = assignment_errors(Kva @ alpha + y_mean, Yva).mean()
        if best is None or val_err < best[0]:
            best = (val_err, c, alpha)
    _, c, alpha = best
    err = assignment_errors(Kte @ alpha + y_mean, Yte)
    sst = ((Yte - Yte.mean(0)) ** 2).sum()
    return {
        "rmse_px": float((err.mean() / 4).sqrt()),
        "r2": float(1 - err.sum() / sst),
        "ridge_c": c,
    }


def run_probe(args):
    mnist_root = Path(args.data_root) / "mnist"
    splits = {
        "train": probe_data(mnist_root, 4000, 100000, mnist_train=True),
        "val": probe_data(mnist_root, 1000, 200000, mnist_train=True),
        "test": probe_data(mnist_root, 2000, 300000, mnist_train=False),
    }
    targets = {s: torch.stack([c[:, 9 + h].reshape(-1, 4) for h in HORIZONS], dim=1)
               for s, (_, _, c) in splits.items()}       # (N, len(H), 4)
    ckpt_root = Path(args.ckpt_root)

    models = {}
    torch.manual_seed(0)
    models["Random init (floor)"] = build_jepa(JEPA_DEFAULTS).eval()
    models["Compact n_layers=2"] = load(CompactPredictor(10, 10, (10, 1, 64, 64),
                                                         **COMPACT_DEFAULTS),
                                        ckpt_root / "compact-mmnist-otf" / "best.pt")[0]
    for name, hp in jepa_checkpoints():
        models[name] = load(build_jepa(hp), ckpt_root / name / "best.pt")[0]
    with_ceiling = {"Random init (floor)", "Compact n_layers=2", *JEPA_MAIN}

    results = {"horizons": HORIZONS}
    # reference without any model: the digit stays where it was last seen
    still = {}
    for k, h in enumerate(HORIZONS):
        last = splits["test"][2][:, 9].reshape(-1, 4)
        err = assignment_errors(last, targets["test"][:, k])
        still[str(h)] = float((err.mean() / 4).sqrt())
    results["no motion (last position)"] = {"pred": {h: {"rmse_px": v} for h, v in still.items()}}
    print("no motion RMSE px:", still)

    for name, model in models.items():
        entry = {}
        for kind in (["pred", "ceil"] if name in with_ceiling else ["pred"]):
            feats = {s: features(kind, model, inp, tgt)
                     for s, (inp, tgt, _) in splits.items()}
            entry[kind] = {}
            for k, h in enumerate(HORIZONS):
                entry[kind][str(h)] = ridge_probe(
                    feats["train"][:, k], targets["train"][:, k],
                    feats["val"][:, k], targets["val"][:, k],
                    feats["test"][:, k], targets["test"][:, k])
        results[name] = entry
        line = "  ".join(f"h={h}: {entry['pred'][str(h)]['rmse_px']:.2f}px "
                         f"(R2 {entry['pred'][str(h)]['r2']:.2f})" for h in HORIZONS)
        print(f"{name:38s} pred  {line}", flush=True)
        if "ceil" in entry:
            line = "  ".join(f"h={h}: {entry['ceil'][str(h)]['rmse_px']:.2f}px"
                             for h in HORIZONS)
            print(f"{'':38s} ceil  {line}", flush=True)

    out = Path("results/jepa_probe.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")
    plot_probe(results, Path("figures/jepa_probe.png"))


def plot_probe(results, out_path):
    curves = {
        "JEPA best": ("jepa-mmnist-otf-best", "#2a78d6", "o"),
        "JEPA defaults": ("jepa-mmnist-otf", "#eb6834", "s"),
        "Compact n_layers=2": ("Compact n_layers=2", "#1baf7a", "^"),
        "Random init": ("Random init (floor)", "#eda100", "D"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (key, color, marker) in curves.items():
        r = results[key]
        ax.plot(HORIZONS, [r["pred"][str(h)]["rmse_px"] for h in HORIZONS],
                color=color, marker=marker, linewidth=2, markersize=6,
                label=f"{label}, predicted latents")
        ax.plot(HORIZONS, [r["ceil"][str(h)]["rmse_px"] for h in HORIZONS],
                color=color, linestyle=":", linewidth=1.5)
    ax.plot(HORIZONS, [results["no motion (last position)"]["pred"][str(h)]["rmse_px"]
                       for h in HORIZONS],
            color="gray", linestyle="--", linewidth=1.5, label="Digit assumed not to move")
    ax.set_xlabel("Horizon (frames after the last input)")
    ax.set_ylabel("Position error of the linear probe (px, RMSE)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Linear probe of digit positions (dotted: true future frame)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["collapse", "probe"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-root", default="checkpoints")
    args = parser.parse_args()
    run_collapse(args) if args.part == "collapse" else run_probe(args)


if __name__ == "__main__":
    main()
