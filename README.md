# Modern Video Prediction Benchmark with Compact Latent Dynamics

Small PyTorch benchmark for future frame prediction. I compared SimVPv2
(a strong modern baseline) against a compact latent dynamics model I
designed myself, first on Moving MNIST and then on KTH Actions. Later
I added ablations and a JEPA-style variant.

The main question I wanted to answer: how much prediction quality can
I keep while making the model much smaller and faster?

## Main result on Moving MNIST

On the on-the-fly setup (200 epochs), the compact model is about 4-5x
smaller than SimVPv2 (4.72M -> 0.98M params) and ~3x cheaper in FLOPs.
It pays for this in long-horizon SSIM. The ablations later showed that
just bumping n_layers from 2 to 3 in the encoder recovers most of the
gap, almost for free.

| Model                          | Params | Test MSE | Test SSIM |
|--------------------------------|--------|----------|-----------|
| SimVPv2 (gSTA, OpenSTL)        | 4.72M  | 0.0168   | 0.84      |
| Compact (n_layers=2, baseline) | 0.98M  | 0.0316   | 0.46      |
| Compact (n_layers=3)           | 0.99M  | 0.0239   | 0.65      |

Going from n_layers=2 to n_layers=3 only adds 0.01M params, but SSIM
jumps from 0.46 to 0.65 (+40%). Full per-step metrics, ablations and
JEPA experiments are in `RESULTS.md`.

## Main result on KTH Actions

KTH tells a very different story. I used a smaller Compact config here
(translator_hidden=128 instead of 256), which puts it at 0.37M params,
roughly 13x smaller than SimVPv2 at 128x128. The gap basically
disappears:

| Horizon | SimVPv2 MSE | Compact MSE | MSE ratio |
|---------|-------------|-------------|-----------|
| t=1     | 0.00093     | 0.00093     | 1.00x     |
| t=10    | 0.00245     | 0.00324     | 1.32x     |
| t=20    | 0.00359     | 0.00394     | 1.10x     |

Same MSE at t=1, and even after 20 autoregressive steps the Compact
model is only ~10% behind.

I also tried n_layers=3 on KTH (since it worked so well on MMNIST),
but it didn't help here. Both Compact configs end up around SSIM
~0.77. KTH futures are much less multi-modal than MMNIST, so the
extra compression doesn't really buy anything.

## Computational efficiency

Single-threaded forward pass, batch_size=1, 64x64 input, on a laptop
CPU (Intel i7):

| Metric        | SimVPv2 | Compact  |
|---------------|---------|----------|
| Parameters    | 4.72M   | 0.98M    |
| FLOPs         | 1.50G   | 0.46G    |
| CPU latency   | 25 ms   | 5.9 ms   |

I re-ran the benchmark on a different machine (server Xeon) and got a
similar 4x speedup, so it's not just a quirk of one CPU.

## What's in this repo

- SimVPv2 baseline plugged into a clean PyTorch training/eval pipeline
  (model code is from OpenSTL with attribution, see below)
- Compact latent dynamics model (encoder -> latent transition -> decoder)
- JEPA-style variant of the Compact model, with EMA target encoder
  and VICReg-style regularization
- Multi-step evaluation with MSE and SSIM
- Two Moving MNIST setups: fixed Toronto split and on-the-fly generation
- Extension to KTH Actions
- 21 ablation runs (10 Compact + 11 JEPA)
- Pareto plot of model size vs prediction quality
- Qualitative figures and a simple efficiency benchmark

## Project structure

    src/
      config.py
      datasets/         # MovingMNIST, MovingMNISTOnTheFly, KTHActions
      models/
        base.py
        simvpv2/        # baseline (from OpenSTL, see ATTRIBUTION.md)
        compact/        # my compact model
        compact_jepa/   # JEPA-style variant
      training/
        trainer.py
        jepa_trainer.py # JEPA-specific training loop
      evaluation/
    scripts/            # download, train, eval, benchmark scripts
    ablations/          # JSON results from 21 ablation runs
      compact/          # 10 Compact ablations
      jepa/             # 11 JEPA ablations
    data/               # datasets (not committed)
    checkpoints/        # not committed
    figures/            # qualitative comparisons + Pareto plot
    RESULTS.md

## How to reproduce

Tested with Python 3.10. Training scripts log to Weights & Biases by
default; set `WANDB_MODE=disabled` in your env to skip that.

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

# JEPA-style training
python scripts/train_jepa_mmnist.py          # default hyperparameters
python scripts/train_jepa_mmnist_best.py     # best hyperparameters from ablations
python scripts/eval_jepa_mmnist.py

# Ablations
python scripts/ablations_compact.py --axis all --n_epochs 100
python scripts/ablations_jepa.py --axis all --n_epochs 100
python scripts/generate_pareto_plot.py

# Compact with n_layers=3 (best ablation finding)
python scripts/train_compact_mmnist_otf_n3.py
python scripts/eval_final.py    # evaluates both JEPA-best and Compact-n3
```

Most of the training was done on a single H100 GPU through Colab.

## Visual comparison - Moving MNIST

![Per-step comparison](figures/qualitative_comparison.png)

For 3 fixed test sequences, this figure shows input frames, ground
truth, SimVPv2 predictions and Compact predictions side by side over
the 10-step horizon (raw intensities, same scale as ground truth).

![Long-horizon detail](figures/long_horizon_zoom.png)

At t=20, SimVPv2 still keeps the digit structure intact while the
Compact predictions go very dark. It's not a rendering issue: the
Compact model regresses toward the dataset mean at long horizons, so
the output amplitude collapses. This is consistent with the SSIM gap
at t=10 (~0.34 Compact vs ~0.75 SimVPv2 on the OTF split) and is
basically what happens when you push a small model on a benchmark
with many possible futures.

To check that the model wasn't just outputting noise, I re-rendered
the same figures with per-image contrast normalization:

![Per-step comparison, contrast-normalized](figures/qualitative_comparison_normalized.png)

![Long-horizon detail, contrast-normalized](figures/long_horizon_zoom_normalized.png)

The normalized view shows the predictions still hold a coarse memory
of where the digits were, so the model is actually predicting
something - the amplitude is just very low. You can also see the
typical ConvTranspose2d checkerboard artifact (Odena et al. 2016).
Replacing transposed convs with Upsample + Conv2d would be a clean
fix for a v2.

For the regression-to-the-mean issue I tried two different things: a
JEPA-style training scheme (loss in latent space), and a simple
architectural tweak (n_layers=3). The architectural one worked much
better, which honestly I wasn't expecting.

## JEPA-style training

The idea was to attack the regression-to-the-mean problem by training
in latent space instead of pixel space, following the JEPA recipe
(Assran et al. 2023): online encoder + EMA target encoder, prediction
loss between latents, VICReg-style anti-collapse term.

I ran 11 ablations to find decent hyperparameters, then retrained the
best config. It did not beat the pixel-space Compact baseline. Best
SSIM was 0.39, against 0.46 for Compact n_layers=2 and 0.65 for
n_layers=3.

A few things I noticed from the ablations:

- The VICReg-style variance regularization actually hurt performance
  here. The best run had var_coef=0.
- Higher EMA momentum helped (0.999 > 0.996 > 0.99). The standard
  recipe seems too reactive for a setup this small.
- JEPA was designed for large-scale self-supervised pretraining
  (V-JEPA uses 200M+ params), and I'm running a 0.98M supervised
  model. That's well outside the regime it was designed for, which
  is probably part of why it underperformed.

In this setup, changing the architecture mattered more than changing
the training objective. Pareto plot for all 21 runs:

![Pareto plot of all 21 ablations + main runs](figures/pareto_plot.png)

## Visual comparison - KTH Actions

![KTH per-sequence comparison](figures/qualitative_comparison_kth.png)

For 3 high-motion KTH test clips, this figure shows the same
side-by-side comparison over the 20-step horizon.

![KTH long-horizon detail](figures/long_horizon_zoom_kth.png)

At the last predicted frame, both models still preserve the human
silhouette. The Compact predictions are softer but stay competitive
even though the model is much smaller.

## Takeaway

The Compact model works really well on smooth real video (KTH: same
MSE as SimVPv2 at t=1, within 10% at t=20, with ~13x fewer params),
but struggles on synthetic multi-modal benchmarks like Moving MNIST,
where it blurs toward the mean at long horizons.

On MMNIST, the simple n_layers 2 -> 3 tweak recovered 40% of the SSIM
gap. JEPA-style training didn't help. The same tweak did not transfer
to KTH, so the sweet spot really depends on the dataset.

## Attribution

The SimVPv2 model code (with the gSTA token-mixer) comes from the
OpenSTL benchmark by Tan et al. (2023), Apache 2.0. See
`ATTRIBUTION.md`. Everything else - datasets, training pipeline,
Compact model, evaluation, benchmark scripts, figures - is mine.

## Notes

This is meant to be a focused benchmark, not an exhaustive survey.
I tried to keep the codebase short while making the comparison
meaningful.

A few natural next steps:

- INT8 quantization and benchmarking on embedded ARM hardware, to
  actually validate the embedded deployment claim.
- Hybrid latent + pixel loss for JEPA. The strict stop-gradient on
  the auxiliary decoder might be too restrictive; letting the latent
  predictor and the decoder co-adapt could help.
- RGB datasets (UCF-101, BAIR Robot Pushing) to check whether the
  architectural findings transfer beyond grayscale.
- Action-conditioned video prediction, with control inputs feeding
  into the latent transition (relevant for drone-style deployment).
