# Results

This file has the detailed metrics, training settings and discussion
for each phase of the benchmark. The high-level summary is in
`README.md`.

The benchmark compares SimVPv2 (gSTA, the strongest baseline from the
OpenSTL benchmark) against a compact latent dynamics model I designed.
Both models share the same encoder-translator-decoder structure, so
the comparison stays as direct as I could make it. The real difference
is the translator: MetaFormer with gSTA on one side, a small 1x1 ->
3x3 -> 1x1 bottleneck on the other.

The phases reported below are:

1. Phase 1: Moving MNIST with the fixed Toronto split (8000 train).
2. Phase 1b: Moving MNIST with on-the-fly sequence generation.
3. Phase 2: KTH Actions, a real-video benchmark.
4. Phase 3: JEPA-style training (Moving MNIST).
5. Phase 4: architecture and training ablations (21 runs).
6. Phase 5: cross-dataset validation of the best ablation finding.

There's also a separate section for computational efficiency
(parameters, FLOPs, CPU latency).

Training was done in PyTorch on a single H100 GPU (Colab).

### Note on Compact configurations

I used two slightly different Compact configurations across phases:

- Moving MNIST (Phase 1, 1b): `latent_dim=64, translator_hidden=256,
  n_layers=2` -> 0.98M params at 64x64.
- KTH Actions (Phase 2): `latent_dim=64, translator_hidden=128,
  n_layers=2` -> 0.37M params at 128x128.

I went with a smaller translator on KTH because human motion is more
deterministic than two random digits bouncing around, so I figured a
smaller predictor should still be enough.

## Phase 1 - Moving MNIST: Comparison

### Configuration
- 200 epochs, batch_size=32, lr=1e-3, cosine LR schedule, Adam + MSE
- 8000/1000/1000 train/val/test split (Toronto .npy)
- H100 GPU
- Best checkpoint selected on validation loss

### Best validation loss

| Model | Params | Best val loss | Best epoch |
|-------|--------|---------------|------------|
| SimVPv2 (gSTA) | 4.72M | 0.02127 | 10 |
| Compact (ours) | 0.98M | 0.03098 | 10 |

### Test set metrics - per-step

| Step | SimVPv2 MSE | SimVPv2 SSIM | Compact MSE | Compact SSIM |
|------|-------------|--------------|-------------|--------------|
| 1 | 0.01071 | 0.8643 | 0.01421 | 0.8234 |
| 2 | 0.01400 | 0.8295 | 0.02125 | 0.7088 |
| 3 | 0.01705 | 0.7874 | 0.02740 | 0.5462 |
| 4 | 0.01970 | 0.7494 | 0.03238 | 0.3597 |
| 5 | 0.02227 | 0.7215 | 0.03499 | 0.2492 |
| 6 | 0.02414 | 0.7038 | 0.03639 | 0.2187 |
| 7 | 0.02533 | 0.6830 | 0.03714 | 0.2057 |
| 8 | 0.02634 | 0.6691 | 0.03741 | 0.1921 |
| 9 | 0.02741 | 0.6504 | 0.03732 | 0.1938 |
| 10 | 0.02812 | 0.6238 | 0.03712 | 0.1989 |

### Key horizons

| Horizon | SimVPv2 | Compact | Compact / SimVPv2 |
|---------|---------|---------|-------------------|
| t=1  | MSE 0.0107 / SSIM 0.86 | MSE 0.0142 / SSIM 0.82 | MSE x 1.33 |
| t=5  | MSE 0.0223 / SSIM 0.72 | MSE 0.0350 / SSIM 0.25 | MSE x 1.57 |
| t=10 | MSE 0.0281 / SSIM 0.62 | MSE 0.0371 / SSIM 0.20 | MSE x 1.32 |

### Discussion

In this first Moving MNIST setup the Compact model stays reasonably
close to SimVPv2 in MSE, which is decent given that it uses about 5x
fewer parameters. The clear weak point is SSIM: after t=5 or so, the
gap really opens up. That matched what I saw visually too - the
Compact model gets blurry at long horizons while SimVPv2 keeps the
digits looking sharp.

One thing bothered me here: both models hit their best validation
loss extremely early (epoch 10 out of 200). That looked more like a
data-diversity issue than under-training - the fixed 8000-sequence
split probably wasn't enough. So I ran a second phase with on-the-fly
sequence generation to check.

## Phase 1b - Moving MNIST with on-the-fly generation

### Setup
On-the-fly generation following Srivastava et al. (2015): 2 MNIST digits per sequence, random initial position, constant velocity sampled at init, wall reflection. 10000 train + 1000 val sequences per epoch. Test set remains the official Toronto .npy file for comparability.

### Best validation loss

| Model | Params | Best val loss | Best epoch |
|-------|--------|---------------|------------|
| SimVPv2 (gSTA) | 4.72M | 0.01125 | 195 |
| Compact (ours) | 0.98M | 0.02505 | 195 |

### Test set metrics - per-step

| Step | SimVPv2 MSE | SimVPv2 SSIM | Compact MSE | Compact SSIM |
|------|-------------|--------------|-------------|--------------|
| 1 | 0.00722 | 0.9235 | 0.01101 | 0.8506 |
| 2 | 0.00977 | 0.9023 | 0.01696 | 0.7550 |
| 3 | 0.01286 | 0.8771 | 0.02370 | 0.6109 |
| 4 | 0.01500 | 0.8578 | 0.02895 | 0.4703 |
| 5 | 0.01703 | 0.8391 | 0.03133 | 0.4065 |
| 6 | 0.01859 | 0.8231 | 0.03247 | 0.3784 |
| 7 | 0.02011 | 0.8062 | 0.03312 | 0.3636 |
| 8 | 0.02152 | 0.7893 | 0.03368 | 0.3528 |
| 9 | 0.02275 | 0.7710 | 0.03404 | 0.3452 |
| 10 | 0.02370 | 0.7483 | 0.03416 | 0.3390 |

### Comparison to literature

For SimVPv2, my number is in the same range as what's reported in
OpenSTL. The mean MSE x 100 over the 10 predicted steps is 16.9,
which is close to the SimVPv2 range from Tan et al. (2023) (~15-18),
even though I only trained for 200 epochs instead of 2000.

### Key observations

The early-overfitting effect from Phase 1 is gone. With the fixed
split, both models peaked at epoch 10; with on-the-fly generation
the best epoch moves to 195/200. So the bottleneck really was data
diversity, not optimization.

SimVPv2 benefits more from the extra data than Compact does. Its
validation loss drops by 47% while Compact only improves by 19%. That
makes sense - the bigger model has more room to use the extra
diversity.

The MSE/SSIM gap of the Compact model is still there though. At t=10,
MSE is only ~1.4x higher than SimVPv2, but SSIM is much worse. So the
Compact model is still close pixel-wise, but the predictions look
blurry.

### Qualitative results

See `figures/qualitative_comparison.png` and `figures/long_horizon_zoom.png`
for the visual comparisons (raw intensities, same scale across all panels).
At t=20, SimVPv2 still keeps recognizable digit shapes while the Compact
output goes very dark - the model's mean output value at long horizons
collapses toward the global pixel mean of the training distribution.
That fits with the SSIM gap and with the model averaging multiple
plausible futures.

For reference, `figures/qualitative_comparison_normalized.png` and
`figures/long_horizon_zoom_normalized.png` show the same predictions
with per-image contrast normalization. There you can see the Compact
model still keeps a coarse spatial memory of where the digits were,
even at t=20. So the failure is regression-to-the-mean (amplitude
collapse), not complete prediction collapse. The normalized view also
makes the ConvTranspose2d checkerboard artifacts visible (Odena et
al. 2016) - one easy v2 fix would be to swap transposed conv for
Upsample + Conv2d.

The regression-to-the-mean part is the main reason JEPA-style
training shows up later in Phase 3.

## Computational efficiency

Single-threaded forward pass, batch_size=1, input shape (1, 10, 1, 64, 64).
Latency averaged over 30 runs after 5 warmup runs, with `torch.no_grad()`.
FLOPs computed with fvcore.

### Hardware A: Intel Xeon (Google Colab CPU instance)

| Model           | Params  | FLOPs  | CPU Latency      | Throughput   |
|-----------------|---------|--------|------------------|--------------|
| SimVPv2 (gSTA)  | 4.72M   | 1.50G  | 7.7 +/- 0.1 ms   | 129.8 s/s    |
| Compact (ours)  | 0.98M   | 0.46G  | 1.9 +/- 0.0 ms   | 528.7 s/s    |

### Hardware B: laptop CPU (Intel i7, x86)

| Model           | Params  | FLOPs  | CPU Latency      | Throughput   |
|-----------------|---------|--------|------------------|--------------|
| SimVPv2 (gSTA)  | 4.72M   | 1.50G  | 25.0 +/- 5.0 ms  | 40.0 s/s     |
| Compact (ours)  | 0.98M   | 0.46G  | 5.9 +/- 0.5 ms   | 169.9 s/s    |

### Compact model gains vs SimVPv2 baseline

| Metric | Hardware A | Hardware B |
|--------|------------|------------|
| Parameters | 4.82x smaller | 4.82x smaller |
| FLOPs | 3.26x fewer | 3.26x fewer |
| Latency speedup | 4.07x | 4.25x |

The speedup is consistent across two CPU setups, which suggests the
gain comes from the model design itself rather than from one specific
machine. The latency speedup is even slightly larger than the FLOPs
reduction - I suspect that's because of the lighter 1x1 bottleneck
and reduced framework overhead in the Compact translator.

Note on embedded deployment: these measurements are on x86 CPUs
(server / laptop), not on actual embedded ARM hardware. The absolute
latency numbers would obviously differ on a drone-class platform,
but I'd expect the relative speedup to hold.

## Phase 2 - KTH Actions

### Setup

KTH Actions dataset (Schuldt et al., 2004): 6 action classes, 25 subjects.
Standard split (persons 1-13 train, 14-16 val, 17-25 test) at 128x128 grayscale. Protocol: 10 input -> 20 target frames via autoregressive rolling. Stride=20 between clips, batch_size=16, 100 epochs. Trained on a single H100 GPU through Colab.

The Compact model on KTH uses a smaller translator hidden dimension
(`translator_hidden=128` instead of 256 on Moving MNIST). At 128x128
input with `n_layers=2`, the latent is 32x32 and the model totals
0.37M parameters (vs 0.98M on Moving MNIST), about 13x smaller than
SimVPv2 at this resolution.

### Best validation loss

| Model | Params | Best val loss | Best epoch |
|-------|--------|---------------|------------|
| SimVPv2 (gSTA) | 4.72M | 0.00444 | 24 |
| Compact (ours) | 0.37M | 0.00477 | 40 |

### Test set metrics - per-step

| Step | SimVPv2 MSE | SimVPv2 SSIM | Compact MSE | Compact SSIM |
|------|-------------|--------------|-------------|--------------|
| 1 | 0.00093 | 0.8738 | 0.00093 | 0.8613 |
| 2 | 0.00116 | 0.8648 | 0.00124 | 0.8494 |
| 3 | 0.00139 | 0.8569 | 0.00156 | 0.8375 |
| 4 | 0.00160 | 0.8510 | 0.00191 | 0.8277 |
| 5 | 0.00178 | 0.8458 | 0.00214 | 0.8188 |
| 6 | 0.00195 | 0.8416 | 0.00240 | 0.8115 |
| 7 | 0.00209 | 0.8377 | 0.00269 | 0.8045 |
| 8 | 0.00219 | 0.8349 | 0.00286 | 0.8007 |
| 9 | 0.00230 | 0.8324 | 0.00303 | 0.7967 |
| 10 | 0.00245 | 0.8290 | 0.00324 | 0.7881 |
| 11 | 0.00300 | 0.8016 | 0.00326 | 0.7866 |
| 12 | 0.00309 | 0.8003 | 0.00336 | 0.7857 |
| 13 | 0.00316 | 0.7980 | 0.00345 | 0.7834 |
| 14 | 0.00323 | 0.7965 | 0.00355 | 0.7811 |
| 15 | 0.00331 | 0.7947 | 0.00359 | 0.7792 |
| 16 | 0.00338 | 0.7927 | 0.00369 | 0.7768 |
| 17 | 0.00343 | 0.7915 | 0.00376 | 0.7750 |
| 18 | 0.00346 | 0.7901 | 0.00382 | 0.7737 |
| 19 | 0.00350 | 0.7888 | 0.00386 | 0.7722 |
| 20 | 0.00359 | 0.7864 | 0.00394 | 0.7673 |

### Key horizons

| Horizon | SimVPv2 | Compact | MSE ratio |
|---------|---------|---------|-----------|
| t=1  | MSE 0.00093 / SSIM 0.87 | MSE 0.00093 / SSIM 0.86 | 1.00x |
| t=10 | MSE 0.00245 / SSIM 0.83 | MSE 0.00324 / SSIM 0.79 | 1.32x |
| t=20 | MSE 0.00359 / SSIM 0.79 | MSE 0.00394 / SSIM 0.77 | 1.10x |

### Discussion

A few things stood out on KTH.

Compact matches SimVPv2 at t=1 - same MSE (0.00093) at the first
predicted step, even though it's much smaller. SSIM is also basically
identical.

It also stays competitive at t=20. The gap is way smaller than on
Moving MNIST: ~10% higher MSE at the last step. KTH is a much more
physically regular prediction problem - human motion is continuous
and the future is much less ambiguous than two digits bouncing
around.

There's a visible jump at t=10/t=11 from the autoregressive rolling.
SimVPv2 shows a sharper discontinuity there, while Compact stays
almost flat. I think the Compact model produces smoother predictions
overall, which hurts sharpness but also makes the handoff between
rollout segments less abrupt.

### Qualitative results

See `figures/qualitative_comparison_kth.png` and `figures/long_horizon_zoom_kth.png`. I picked high-motion clips on purpose - clips with mostly static backgrounds aren't really informative for visual comparison.

The qualitative results line up with the metrics: both models keep
the main body silhouette over the 20-step horizon. The Compact model
is a bit blurrier, especially near the end of the rollout, but the
predictions stay clearly interpretable.

## Phase 3 - JEPA-style training (Moving MNIST)

### Motivation

Phase 1b made the regression-to-the-mean failure mode on Moving MNIST
very visible: at long horizons, the Compact model's predicted pixels
collapse toward the global pixel mean. The idea for Phase 3 was that
JEPA-style training (Assran et al. 2023; Bardes et al. 2024) might
help by predicting in latent space instead of pixel space, since
multiple plausible futures can be semantically close in a learned
representation even when they look visually different.

### Setup

Architecture:
- Online encoder: same as Compact (latent_dim=64, n_layers=2)
- EMA target encoder: deep copy of online encoder, momentum updated
  (target_w = momentum * target_w + (1-momentum) * online_w)
- Predictor: same LatentTransition as Compact (translator_hidden=256)
- Auxiliary decoder: Compact decoder receiving `predicted_latents.detach()`
  (stop-gradient), so the predictor trains purely on latent-space loss

Loss components (VICReg-inspired):
- Prediction loss: smooth L1 between predicted and target latents
- Variance loss: penalizes std < 1 in any latent dimension
- Covariance loss: penalizes off-diagonal entries of the latent covariance matrix
- Auxiliary decoder loss: pixel MSE on decoder output (separate gradient flow)

200 epochs, Adam, lr=1e-3, batch_size=32, on-the-fly Moving MNIST.

### Results

Two JEPA runs:

| Config | EMA mom. | var_coef | cov_coef | Best epoch | MSE avg | SSIM avg |
|--------|----------|----------|----------|------------|---------|----------|
| JEPA defaults | 0.996 | 1.0 | 0.04 | 79 | 0.0356 | 0.30 |
| JEPA optimal (post-ablation) | 0.999 | 0.0 | 0.0 | 197 | 0.0308 | 0.39 |
| Compact main (n_layers=2, ref) | - | - | - | 195 | 0.0316 | 0.46 |
| Compact n_layers=3 (Phase 4) | - | - | - | 195 | 0.0239 | 0.65 |

JEPA optimal beats JEPA defaults by +30% SSIM, mostly by removing the
VICReg regularization and using a higher EMA momentum. But it still
underperforms direct pixel-space training: 0.39 SSIM vs 0.46 for
Compact n_layers=2 and 0.65 for Compact n_layers=3.

### Discussion

This is a negative result for JEPA in this particular setting. A few
possible reasons:

The architecture is probably too small for representation learning to
really kick in. V-JEPA uses 200M+ parameters; my Compact JEPA has
0.98M. Representation learning generally benefits from scale, and I'm
running well below the regime JEPA was designed for.

The setting is also fully supervised, which weakens JEPA's main
advantage. V-JEPA's strength is leveraging unlabeled data to learn
transferable representations. Here, future frames are fully
supervised, so the regression-to-the-mean problem is better attacked
by architectural choices (Phase 4) than by a representation-learning
objective.

There's also a decoder bottleneck. The auxiliary decoder gets
stop-gradded latents - if the predicted latents are off geometrically,
the decoder can't correct them. With a single shared encoder-decoder
(Compact), gradients flow end-to-end and the model can co-adapt.

The Phase 4 ablations confirm that VICReg regularization was
counterproductive here (var_coef=0 was best) and that higher EMA
momentum helped (0.999 > 0.996 > 0.99). My read is that JEPA is
pretty sensitive to hyperparameters in small-supervised regimes -
basically outside its design intent.

### Qualitative results

See `figures/qualitative_comparison.png` for the Compact baselines.
I didn't regenerate qualitative figures specifically for the JEPA
optimal run, but visually the predictions are in the same range as
Compact main: coarse spatial structure preserved, pixel amplitude
collapsing at long horizons.

## Phase 4 - Ablations (Moving MNIST)

### Setup

10 Compact ablations + 11 JEPA ablations on Moving MNIST on-the-fly,
100 epochs each. One-At-a-Time (OAT) sweep: each axis is varied while
the others are fixed at defaults.

### Compact ablations

| Axis | Values | Best | SSIM range |
|------|--------|------|------------|
| latent_dim | 16, 32, 64, 128 | 64 | 0.41 -> 0.47 (saturates) |
| translator_hidden | 64, 128, 256, 512 | 256 | 0.41 -> 0.47 (saturates) |
| n_layers | 2, 3 | 3 | 0.46 -> 0.62 (+35%, key finding) |

The standout result is n_layers=3, which improves SSIM by 35% over
n_layers=2 at basically the same parameter count (0.99M vs 0.98M).
At 100 epochs (the ablation budget) it reaches SSIM 0.62; trained
for the full 200 epochs it reaches SSIM 0.65, which is the new
Compact best (see Phase 3 table).

### JEPA ablations

| Axis | Values | Best | SSIM range |
|------|--------|------|------------|
| ema_momentum | 0.99, 0.996, 0.999 | 0.999 | 0.27 -> 0.32 |
| var_coef | 0.0, 1.0, 5.0 | 0.0 | 0.36 -> 0.29 (regularization hurts) |
| cov_coef | 0.0, 0.04, 0.2 | insensitive | 0.32 -> 0.33 (negligible) |
| decoder_coef | 0.0, 1.0 | required | n/a without decoder |

Main things I noticed:

- The variance regularization (var_coef) was counter-productive in
  this setting. The best run had no regularization at all.
- Higher EMA momentum (0.999) was clearly preferred - JEPA seems to
  need a more stable target encoder to learn anything useful here.
- Covariance regularization had basically no impact.

### Methodology note

The ablations are run One-At-a-Time (OAT): each axis is varied while
the others are kept at defaults. That's a budget-friendly choice - it
surfaces the marginal contribution of each axis but doesn't capture
pairwise interactions. A full grid search would have been more
rigorous but cost roughly 3x more compute. OAT is enough here to
confirm that the default configuration sits on the Pareto frontier of
params vs MSE.

### Pareto curve

See `figures/pareto_plot.png` for the full landscape of all 21
ablations plus the main runs (SimVPv2, Compact main), on params vs
MSE and params vs SSIM axes. A few things are visible there:

- JEPA ablations cluster in the high-MSE/low-SSIM region
- Compact n_layers=3 is an isolated win on SSIM
- SimVPv2 stays Pareto-optimal in absolute terms but at ~5x cost

## Phase 5 - Cross-dataset architecture validation

### Setup

Phase 4 ablations on Moving MNIST showed that going from n_layers=2
to n_layers=3 boosts SSIM by 40% (0.46 -> 0.65). To check whether
that finding generalizes, I retrained the Compact model with
n_layers=3 on KTH Actions.

### Results

| Horizon | Compact n=2 (baseline) | Compact n=3 | SimVPv2 |
|---------|----------------------|-------------|---------|
| t=1     | MSE 0.00093 / SSIM 0.86 | MSE 0.00132 / SSIM 0.81 | - |
| t=10    | MSE 0.00324 / SSIM 0.79 | MSE 0.00306 / SSIM 0.76 | - |
| t=20    | MSE 0.00394 / SSIM 0.77 | MSE 0.00395 / SSIM 0.74 | MSE 0.00359 / SSIM 0.79 |
| Avg     | - / 0.79             | 0.00297 / 0.77        | - / 0.79 |

### Discussion

Unlike on Moving MNIST, increasing n_layers does *not* improve
performance on KTH - both configs perform basically the same (SSIM
0.77 vs 0.79). A few reasons that probably matter:

Spatial resolution: KTH (128x128) with n_layers=2 already gives a
32x32 latent, which seems sufficient for the spatial complexity of
the dataset. The n_layers=3 variant compresses further to 16x16,
which probably loses useful spatial information without bringing the
matching benefit (escaping regression-to-the-mean) that helps so much
on Moving MNIST.

Multi-modal futures: KTH features are largely deterministic - a person
walking continues walking. There isn't really a regression-to-the-mean
problem to escape. On Moving MNIST, two digits with independent
trajectories produce genuinely multi-modal futures, and that's where
the extra latent compression helps.

Convergence: KTH n_layers=3 also reaches its best val loss at epoch
25/100, which suggests the model has enough capacity but the task
itself just doesn't benefit from more depth.

So this is a useful negative finding: architectural choices that help
on a synthetic, multi-modal benchmark (Moving MNIST) don't
automatically transfer to real-world video (KTH). The architecture
really has to match the dataset.

## Take-aways across phases

Looking at all phases together:

- On the fixed Moving MNIST split, both models hit their best val
  loss at epoch 10/200. That pointed to a data-diversity bottleneck,
  not a capacity issue, which is what motivated Phase 1b.
- With on-the-fly Moving MNIST the bottleneck disappears: best val
  moves to epoch 195/200, and SimVPv2 improves more than Compact
  (47% vs 19%), which fits its larger capacity.
- On KTH the picture changes a lot. The MSE/SSIM gap mostly closes
  and Compact becomes very competitive with ~13x fewer parameters.
  KTH is much less multi-modal than Moving MNIST, so small models
  are penalized less by their tendency to predict averaged futures.
- JEPA-style training did not improve over pixel-space training in
  this small-supervised regime. Phase 3 + Phase 4 trace it to the
  small model size (0.98M vs 200M+ in V-JEPA) and the stop-gradient
  decoder bottleneck.
- The most impactful improvement came from the architecture, not
  the training strategy: bumping n_layers from 2 to 3 on Moving
  MNIST boosted SSIM by 40% (0.46 -> 0.65) at basically no extra
  cost in parameters.
- But architectural wins don't always transfer across datasets - the
  n_layers=3 finding from Moving MNIST didn't generalize to KTH
  (Phase 5), where n_layers=2 stayed optimal.

## Limitations

Some honest limitations:

- Only two datasets, both grayscale (Moving MNIST and KTH). Extending
  to RGB datasets like UCF-101 or BAIR Robot Pushing would give a
  stronger picture.
- Latency was measured on x86 CPUs (server / laptop), not on actual
  embedded ARM hardware. The relative trend should hold but absolute
  numbers would change.
- No comparison against recurrent baselines like ConvLSTM or PredRNN,
  which would put SimVPv2 in a wider context.
- The JEPA experiments are limited to one model size (0.98M).
  Scaling up could reveal benefits that disappear at this small
  scale, but that was outside the compute budget for this project.
- Ablations are OAT (see Phase 4 methodology note). A full grid
  search would catch pairwise interactions between axes but wasn't
  feasible within the compute budget.

## Future work

- Quantization and ARM-CPU benchmarking, to make the embedded
  deployment claim concrete (most direct next step).
- Hybrid latent + pixel loss for JEPA: removing the strict
  stop-gradient on the auxiliary decoder may let the predictor learn
  representations that are simultaneously good for latent prediction
  and pixel reconstruction.
- Action-conditioned video prediction, with control inputs feeding
  into the latent transition (relevant for drone-like deployment).
- RGB benchmarks (UCF-101, BAIR Robot Pushing) to check whether the
  findings transfer beyond grayscale.
- Recurrent baselines (ConvLSTM, PredRNN) to widen the comparison.
- A short tech report based on these results.
