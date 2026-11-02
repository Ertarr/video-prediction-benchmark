# Attribution

This repository includes code from third-party projects.

## SimVPv2 (OpenSTL)

The SimVPv2 model implementation in `src/models/simvpv2/` is adapted
from the OpenSTL benchmark:

> Tan, C., Gao, Z., Wu, L., Xu, Y., Xia, J., Li, S., & Li, S. Z. (2023).
> OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive
> Learning. arXiv:2306.11249.

Original code: https://github.com/chengtan9907/OpenSTL

License: Apache 2.0 (the corresponding `LICENSE` file is in
`src/models/simvpv2/`).

The integration into this repository (config wrapping, `VideoPredictor`
base class, training/eval glue, rolling forward for KTH) is mine.

## Methods re-implemented from papers (no code copied)

These parts are my own code, written from the papers:

- Recurrent memory tokens (`src/models/transformer_rmt/`): Bulatov, A.,
  Kuratov, Y., & Burtsev, M. (2022). Recurrent Memory Transformer.
  NeurIPS 2022.
- kNN memory with a learned gate (`src/models/transformer_knn/`),
  simplified: Wu, Y., Rabe, M. N., Hutchins, D., & Szegedy, C. (2022).
  Memorizing Transformers. ICLR 2022.
- Effective rank (`scripts/jepa_diagnostic.py`): Garrido, Q.,
  Balestriero, R., Najman, L., & LeCun, Y. (2023). RankMe. ICML 2023.
- The optional skip connection of Compact (Phase 6) reproduces the
  encoder-to-decoder skip of SimVP (see the SimVPv2 section above).

The OpenSTL numbers quoted in `RESULTS.md` come from the OpenSTL model
zoo tables (https://github.com/chengtan9907/OpenSTL).

## Datasets

- **Moving MNIST**: official Toronto split released by Srivastava et al. (2015),
  http://www.cs.toronto.edu/~nitish/unsupervised_video/
- **KTH Actions**: Schuldt, C., Laptev, I., & Caputo, B. (2004).
  Recognizing human actions: a local SVM approach.
  https://www.csc.kth.se/cvap/actions/
