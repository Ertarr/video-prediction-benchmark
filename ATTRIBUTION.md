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

## Datasets

- **Moving MNIST**: official Toronto split released by Srivastava et al. (2015),
  http://www.cs.toronto.edu/~nitish/unsupervised_video/
- **KTH Actions**: Schuldt, C., Laptev, I., & Caputo, B. (2004).
  Recognizing human actions: a local SVM approach.
  https://www.csc.kth.se/cvap/actions/
