# Third-party notices

## OpenPathSampling

This repository includes a **vendored copy** of [OpenPathSampling](https://github.com/openpathsampling/openpathsampling) under `src/python/openpathsampling/`, for path sampling (TPS, TIS, etc.) as the driver layer for generative trajectory analysis.

- **Upstream:** https://github.com/openpathsampling/openpathsampling  
- **License:** MIT License  
- **Upstream license file:** `src/python/openpathsampling/LICENSE.openpathsampling-upstream` (copy of upstream `LICENSE` at vendor time)

Copyright and permission notices from the MIT license apply to that subtree. The genai-tps project may modify vendored files for integration; track upstream releases when updating the vendor tree.

## RLDiff (reference submodule and derived training code)

The repository may include **[RLDiff](https://github.com/oxpig/RLDiff)** as a **git submodule** under `RLDiff/` (MIT License, University of Oxford). The paper *Teaching Diffusion Models Physics: Reinforcement Learning for Physically Valid Diffusion-Based Docking* (bioRxiv, DOI [10.64898/2026.03.25.714128](https://doi.org/10.64898/2026.03.25.714128)) describes that framework.

A small portion of the offline **PPO-style clipped surrogate** in `src/python/genai_tps/rl/ppo_surrogate.py` is **derived from** RLDiff’s `utils/train_utils.py` (`compute_loss` and reward-normalization patterns), adapted for Boltz-2. The MIT license and copyright notice from `RLDiff/LICENSE` apply to those derived parts. The submodule is provided for attribution and comparison; runtime training uses `genai_tps.rl` and does not import DiffDock or the upstream `posebusters` package on the hot path.
