# Modern Video Prediction Benchmark with Compact Latent Dynamics

This is a small PyTorch benchmark for future frame prediction.
The goal was to compare a strong modern baseline, **SimVPv2**, with a
much smaller **compact latent dynamics model** that I designed myself.

I started with **Moving MNIST**, then extended the comparison to
**KTH Actions**.

The question I had in mind was simple:

> how much quality can I keep if I make the prediction model much smaller
> and faster?

## Main result on Moving MNIST

On Moving MNIST, the compact model is much cheaper than SimVPv2:

- **4.82x fewer parameters** (0.98M vs 4.72M)
- **3.26x fewer FLOPs** (0.46G vs 1.50G)
- about **4x lower CPU latency**

The downside is that it gets blurrier at long horizons, which shows up
clearly in SSIM.

| Model    | Params | Test MSE (avg) | Test SSIM (avg) |
|----------|--------|----------------|-----------------|
| SimVPv2  | 4.72M  | 0.0215         | 0.74            |
| Compact  | 0.98M  | 0.0316         | 0.40            |

So roughly a 5x parameter reduction for about 1.5x worse MSE on average.
Full per-step metrics and discussion are in `RESULTS.md`.

## Main result on KTH Actions

On KTH the picture is very different. I used a smaller version of the
Compact model here (translator_hidden=128 instead of 256), which gives
0.37M params - about 13x smaller than SimVPv2 at 128x128 - and the gap
with SimVPv2 becomes much narrower:

| Horizon | SimVPv2 MSE | Compact MSE | MSE ratio |
|---------|-------------|-------------|-----------|
| t=1     | 0.00093     | 0.00093     | 1.00x     |
| t=10    | 0.00245     | 0.00324     | 1.32x     |
| t=20    | 0.00359     | 0.00394     | 1.10x     |

At t=1 the two models give the same MSE. Even at t=20 (after
autoregressive rolling), the compact model is only about 10% worse.

## Computational efficiency

Single-threaded forward pass, batch_size=1, 64x64 input, on a laptop CPU
(Intel i7).

| Metric          | SimVPv2 | Compact  |
|-----------------|---------|----------|
| Parameters      | 4.72M   | 0.98M    |
| FLOPs           | 1.50G   | 0.46G    |
| CPU latency     | 25 ms   | 5.9 ms   |

The ~4x speedup is consistent across two CPU setups (server Xeon and
laptop i7), so the gain comes from the model design, not from one
specific machine.

## What is in this repo

- **SimVPv2 baseline** integrated into a clean PyTorch training/eval pipeline
  (the SimVPv2 model code is taken from OpenSTL with attribution, see below)
- **Compact latent dynamics model** (`encoder -> latent transition -> decoder`)
- Multi-step evaluation with **MSE** and **SSIM**
- Two Moving MNIST setups:
  - fixed Toronto split
  - on-the-fly sequence generation
- Extension to **KTH Actions**
- Qualitative comparisons and simple efficiency measurements

## Project structure

    src/
      config.py
      datasets/         # MovingMNIST, MovingMNISTOnTheFly, KTHActions
      models/
        base.py
        simvpv2/        # baseline (from OpenSTL, see ATTRIBUTION.md)
        compact/        # my compact model
      training/
      evaluation/
    scripts/            # download, train, eval, benchmark scripts
    data/               # datasets (not committed)
    checkpoints/        # not committed
    figures/
    RESULTS.md

## How to reproduce

Tested with Python 3.10. Training scripts log to Weights & Biases by
default. To run them without a W&B account, set `WANDB_MODE=disabled`
in your environment.

```bash
git clone <repo>
cd video-prediction-benchmark
pip install -r requirements.txt

# Moving MNIST (fixed split)
python scripts/download_mmnist.py
python scripts/train_simvpv2_mmnist.py
python scripts/train_compact_mmnist.py
python scripts/eval_simvpv2_mmnist.py
python scripts/eval_compact_mmnist.py

# Moving MNIST (on-the-fly)
python scripts/train_simvpv2_mmnist_otf.py
python scripts/train_compact_mmnist_otf.py
python scripts/eval_simvpv2_mmnist_otf.py
python scripts/eval_compact_mmnist_otf.py

# KTH Actions
python scripts/download_kth.py
python scripts/extract_kth_frames.py
python scripts/generate_kth_splits.py
python scripts/train_simvpv2_kth.py
python scripts/train_compact_kth.py
python scripts/eval_simvpv2_kth.py
python scripts/eval_compact_kth.py

# Efficiency benchmark
python scripts/benchmark_efficiency.py
```

Most training was done on a single H100 GPU through Colab.

## Visual comparison - Moving MNIST

![Per-step comparison](figures/qualitative_comparison.png)

For 3 fixed test sequences, this figure shows the input frames, ground
truth, SimVPv2 predictions, and Compact predictions side by side over
the 10-step horizon (raw intensities, same scale as ground truth).

![Long-horizon detail](figures/long_horizon_zoom.png)

At t=20, SimVPv2 still keeps clear digit structure, while the Compact
model becomes very dark. This is not a rendering bug: the Compact model
regresses toward the mean of the training distribution at long horizons,
so its outputs have very low amplitude. This matches the SSIM gap
(0.34 vs 0.75 at t=10 in RESULTS.md) and is what you'd expect from a
low-capacity model on a benchmark with many possible futures.

To make sure it isn't just collapsing to noise, here are the same
figures with per-image contrast normalization. You can see the Compact
predictions still keep a coarse memory of where the digits were:

![Per-step comparison, contrast-normalized](figures/qualitative_comparison_normalized.png)

![Long-horizon detail, contrast-normalized](figures/long_horizon_zoom_normalized.png)

So the model isn't predicting noise, it just has very low pixel
amplitude. The normalized view also shows the checkerboard pattern
typical of ConvTranspose2d (Odena et al. 2016). One natural fix for a
v2 would be to replace transposed convolutions with Upsample + Conv2d.
For the regression-to-the-mean issue itself, JEPA-style training in
latent space looks like the most direct next step.

## Visual comparison - KTH Actions

![KTH per-sequence comparison](figures/qualitative_comparison_kth.png)

For 3 high-motion KTH test clips, this figure shows the same side-by-side
comparison over the 20-step horizon.

![KTH long-horizon detail](figures/long_horizon_zoom_kth.png)

At the last predicted frame (after autoregressive rolling), both models
still preserve the overall human silhouette. The Compact model is
softer, but it stays competitive even though it's much smaller.

## Short takeaway

The compact model works surprisingly well when motion is smooth and
fairly predictable (like KTH), but struggles when the future is more
multi-modal (like Moving MNIST, where two digits move independently).
In the harder cases, MSE stays acceptable but SSIM drops a lot, which
shows the model is averaging plausible futures and producing blurrier
predictions.

## Attribution

The SimVPv2 model code (with the gSTA token-mixer) is taken from the
OpenSTL benchmark by Tan et al. (2023), under their Apache 2.0
license. See `ATTRIBUTION.md` for details. Everything else
(datasets, training pipeline, compact model, evaluation, benchmark
scripts, figures) is mine.

## Notes

This is a focused benchmark, not an exhaustive survey. I tried to keep
the codebase short and reproducible while still making the comparison
meaningful.

A few natural next steps:

- testing on more complex datasets (UCF-101, BAIR Robot Pushing)
- conditioning on actions / control inputs for a more world-model-like
  setup
- trying JEPA-style training in latent space, which would directly
  address the regression-to-the-mean blur on Moving MNIST
- pushing the compact model further for embedded or robotics settings
