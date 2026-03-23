# Vendored OpenPathSampling

The repository previously tracked **OpenPathSampling** as a **git submodule**; it is now **vendored** under:

`src/python/openpathsampling/`

## Contributor migration

- **Cloning:** You no longer need `git submodule update --init openpathsampling`.
- **Install:** From the repo root, `pip install -e .` installs both `openpathsampling` and `genai_tps` from `src/python`.
- **Updating upstream:** Fetch a release from [openpathsampling/openpathsampling](https://github.com/openpathsampling/openpathsampling), diff against `src/python/openpathsampling/`, and merge selectively. Preserve `LICENSE.openpathsampling-upstream` and note the upstream commit or tag in the commit message.

## Why vendor?

- Single distribution (**genai-tps**) for generative TPS workflows without a PyPI dependency pin on the full upstream package.
- Freedom to ship small integration patches if needed (prefer upstreaming when possible).
