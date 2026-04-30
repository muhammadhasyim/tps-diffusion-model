# Local model weights (not in git)

Place large checkpoints here on each machine. Filenames are ignored by `.gitignore`; only empty layout is tracked.

| Path | Purpose |
|------|--------|
| `boltz/` | Mirror of the Boltz cache: same layout as `~/.boltz` (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`, `mols.tar`, …). |
| `finetuned/` | Your training runs (e.g. WDSM `boltz2_wdsm_*.pt`). |

**Boltz:** set an absolute cache path, e.g.

```bash
export BOLTZ_CACHE="$(pwd)/weights/boltz"
```

If `BOLTZ_CACHE` is unset, project scripts default to `$SCRATCH/.boltz` when `SCRATCH`
is defined (typical HPC), otherwise `~/.boltz`.

Or pass `--cache /path/to/.../weights/boltz` to scripts that support it.

No Git LFS: keep weights local or sync them yourself (tarball, shared disk, artifact store).
