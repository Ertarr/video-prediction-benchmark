# Attribution

The SimVPv2 baseline used in this project is adapted from OpenSTL:
https://github.com/chengtan9907/OpenSTL

Original paper:
Tan et al., "SimVP: Towards Simple yet Powerful Spatiotemporal
Predictive Learning", 2022.

The DWConv and MixMlp modules (in modules.py) originate from VAN:
https://github.com/Visual-Attention-Network/VAN-Classification

License: Apache 2.0 (see LICENSE file in this directory).

### Modifications from original OpenSTL code:
- Extracted only the modules needed for the 'gsta' variant of SimVPv2.
- Simplified MetaBlock to support only the gSTA variant.
- Removed MidIncepNet (SimVPv1 translator) which is unused in this project.
- Added a VideoPredictor wrapper (simvpv2.py) to match this project's 
  model interface.
- Added None-checks on bias parameters in `_init_weights` methods for 
  robustness when convolutional layers are instantiated with bias=False.