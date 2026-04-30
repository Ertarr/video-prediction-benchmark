# Results

This file has the detailed metrics, training settings, and discussion
for each phase of the benchmark. The high-level summary is in
`README.md`.

The benchmark compares SimVPv2 (gSTA, the strongest baseline from the
OpenSTL benchmark) to a compact latent dynamics model that I designed.
Both models share the same encoder-translator-decoder structure, so the
comparison stays as direct as possible. The main thing that really
differs is the translator (MetaFormer with gSTA on one side, a small
1x1 -> 3x3 -> 1x1 bottleneck on the other side).

Three phases are reported:

1. **Phase 1**: Moving MNIST with the fixed Toronto split (8000 train).
2. **Phase 1b**: Moving MNIST with on-the-fly sequence generation.
3. **Phase 2**: KTH Actions, a real-video benchmark.

A separate section reports the **computational efficiency** numbers
(parameters, FLOPs, CPU latency).

Training was done in PyTorch on a single H100 GPU (Colab).

### Note on Compact configurations

I used two slightly different Compact configurations across phases:

- **Moving MNIST (Phase 1, 1b)**: `latent_dim=64, translator_hidden=256,
  n_layers=2` -> 0.98M params at 64x64.
- **KTH Actions (Phase 2)**: `latent_dim=64, translator_hidden=128,
  n_layers=2` -> 0.37M params at 128x128.

I went with a smaller translator on KTH because human motion is more
deterministic than two random digits, so I figured less capacity should
be enough.

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

On this first Moving MNIST setup, the compact model stays reasonably close to SimVPv2 in terms of MSE, especially considering that it uses about 5x fewer parameters. The main weakness is clearly SSIM: after around t=5, the gap becomes large, which matches what I saw visually as well. The compact model gets blurrier at long horizons, while SimVPv2 keeps sharper digit shapes.

One thing that bothered me here: both models reach their best validation loss very early (epoch 10/200). To me that pointed to a data-diversity issue rather than under-training - the fixed 8000-sequence split is probably the bottleneck. So I ran a second phase with on-the-fly sequence generation to check.

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

For SimVPv2, my result is in the same range as the numbers reported in OpenSTL. The mean MSE x 100 over the 10 predicted steps is 16.9, which is close to the SimVPv2 range from Tan et al. (2023) (~15-18), even though I only trained for 200 epochs instead of 2000.

### Key observations

1. **On-the-fly generation removes the early overfitting effect.**
   With the fixed split, both models peaked at epoch 10. With on-the-fly generation, the best epoch moves to 195/200. So the original bottleneck really was dataset size/diversity, not optimization.

2. **SimVPv2 benefits more from the extra data.**
   Its validation loss drops by 47%, compared to 19% for the Compact model. That makes sense: the larger model has more room to use the extra training diversity.

3. **The MSE/SSIM gap of the Compact model is still there.**
   At t=10, MSE is only about 1.4x higher than SimVPv2, but SSIM is much lower. So the compact model is still close pixel-wise, but visually blurrier.

### Qualitative results

See `figures/qualitative_comparison.png` and `figures/long_horizon_zoom.png`
for visual comparisons (raw intensities, same scale across all panels).
At t=20, SimVPv2 still keeps recognizable digit shapes, while the Compact
model output becomes very dark - its mean output value at long horizons
collapses toward the global pixel mean of the training distribution.
This matches the SSIM gap and fits with the model averaging plausible
futures when several are possible.

For reference, `figures/qualitative_comparison_normalized.png` and
`figures/long_horizon_zoom_normalized.png` show the same predictions
with contrast normalized per-image. You can see the Compact model still
keeps a coarse spatial memory of where the digits were, even at t=20.
So the failure is regression-to-the-mean (output amplitude collapse),
not complete prediction collapse. The normalized view also makes
visible the checkerboard artifacts typical of ConvTranspose2d (Odena
et al. 2016) - one easy v2 fix would be to swap transposed conv for
Upsample + Conv2d.

The regression-to-the-mean part is the main reason JEPA-style training
shows up in Future work.

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
| **Latency speedup** | **4.07x** | **4.25x** |

The speedup is consistent across two CPU setups, so the gain comes from
the model design itself and not from one specific machine. The latency
speedup is even slightly larger than the FLOPs reduction - I suspect
that comes from the lighter 1x1 bottleneck and reduced framework
overhead in the Compact translator.

**Note on embedded deployment:** these measurements are on x86 CPUs
(server / laptop), not on embedded ARM hardware. Absolute latency
numbers would change on a drone platform, but I would expect the
relative speedup to hold.

## Phase 2 - KTH Actions

### Setup

KTH Actions dataset (Schuldt et al., 2004): 6 action classes, 25 subjects.
Standard split (persons 1-13 train, 14-16 val, 17-25 test) at 128x128 grayscale. Protocol: 10 input -> 20 target frames via autoregressive rolling. Stride=20 between clips, batch_size=16, 100 epochs. Trained on a single H100 GPU through Colab.

The Compact model on KTH uses a smaller translator hidden dimension
(`translator_hidden=128` instead of 256 on Moving MNIST). At 128x128
input with `n_layers=2`, the latent is 32x32 and the model totals
**0.37M parameters** (vs 0.98M on Moving MNIST), about 13x smaller than
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
| t=1  | MSE 0.00093 / SSIM 0.87 | MSE 0.00093 / SSIM 0.86 | **1.00x** |
| t=10 | MSE 0.00245 / SSIM 0.83 | MSE 0.00324 / SSIM 0.79 | 1.32x |
| t=20 | MSE 0.00359 / SSIM 0.79 | MSE 0.00394 / SSIM 0.77 | 1.10x |

### Discussion

A few things stand out on KTH:

1. **Compact matches SimVPv2 at t=1.**
   MSE is the same at the first predicted step (0.00093), even though
   the Compact model is much smaller. SSIM is also very close.

2. **Compact stays competitive even at t=20.**
   The gap is much smaller than on Moving MNIST. At the last step, MSE
   is only about 10% higher. KTH is a much more physically regular
   prediction problem - human motion is continuous and the future is
   less ambiguous than in Moving MNIST.

3. **There's a visible break at t=10 / t=11 from the autoregressive rolling.**
   SimVPv2 has a sharper jump there, while Compact stays almost flat.
   I think the compact model produces smoother predictions overall,
   which hurts sharpness but also makes the handoff between rollout
   segments less abrupt.

### Qualitative results

See `figures/qualitative_comparison_kth.png` and `figures/long_horizon_zoom_kth.png`. I picked high-motion clips on purpose, because clips with almost static backgrounds aren't really informative for visual comparison.

The qualitative results match the metrics: both models keep the main body silhouette over the 20-step horizon. Compact is still a bit blurrier, especially near the end of the rollout, but the predictions stay clearly interpretable.

## Take-aways across phases

Looking at the three phases together:

- On the **fixed Moving MNIST split**, both models hit their best
  validation loss at epoch 10/200, which clearly pointed to a
  data-diversity bottleneck rather than a capacity issue. That's
  what motivated Phase 1b.
- With **on-the-fly Moving MNIST**, the bottleneck disappears: best
  val moves to epoch 195/200, and SimVPv2 improves more than the
  compact model (47% vs 19%), which fits with its larger capacity.
- On **KTH**, the picture changes a lot. The MSE/SSIM gap mostly
  closes, and the compact model becomes very competitive even with
  ~13x fewer parameters. KTH is much less multi-modal than Moving
  MNIST, so capacity-limited models are penalized less by their
  tendency to predict averaged futures.

## Limitations

A few honest limitations:

- Only two datasets, both grayscale (Moving MNIST and KTH). Extending
  to RGB datasets like UCF-101 or BAIR Robot Pushing would give a
  stronger picture.
- The compact model design wasn't formally ablated. A Pareto curve
  over `latent_dim` or `translator_hidden` would make the design
  choices more rigorous.
- Latency was measured on x86 CPUs (server / laptop), not on embedded
  ARM hardware. Relative trend should hold but absolute numbers would
  differ.
- No comparison against recurrent baselines like ConvLSTM or PredRNN,
  which would put the SimVPv2 family in a wider context.

## Future work

- Action-conditioned video prediction, with control inputs feeding
  into the latent transition.
- JEPA-style training (predicting in latent space rather than
  pixel space) is the most direct way to address the long-horizon
  blur observed in the compact model on Moving MNIST.
- Quantization and ARM-CPU benchmarking, to make the embedded
  deployment claim concrete.
- A short tech report based on these results.
