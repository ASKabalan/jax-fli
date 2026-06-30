# fli-infer

Run **full-field** MCMC inference: jointly sample the initial conditions and cosmological
parameters of the forward model, conditioned on an observed convergence (κ) map. This is the
most expensive — and most information-rich — inference mode.

## Usage

```bash
# Local observable parquet; optional warm-start IC from another parquet.
fli-infer --input kappa.parquet --path chains/ \
          --ic-input ic.parquet \
          --num-warmup 200 --num-samples 500 --sampler NUTS --checkpoints 10

# Observable streamed from a HuggingFace dataset repo.
fli-infer --repo user/repo --data-files 'obs/kappa.parquet' --path chains/ \
          --num-warmup 200 --num-samples 500
```

## Arguments

The observable uses the shared `add_source_args` interface (local `--input` **XOR** HuggingFace
`--repo` + `--data-files`, same as [`fli-born-rt`](fli-born-rt.md)). It must be a **single-row**
catalog whose field type matches `--lensing-output`: a convergence (κ) field for `convergence`, or a
shear field for `shear` / `reduced_shear`. The optional initial condition uses the parallel **prefixed**
source `--ic-input` / `--ic-repo` / `--ic-data-files` and must be a single-row `DensityField`; it fixes
the IC (when `ic` is not in `--sample`) or warm-starts a sampled IC.

| Flag | Meaning |
|------|---------|
| `--input` / `--repo` + `--data-files` | single-row observable source (κ or shear, per `--lensing-output`) |
| `--ic-input` / `--ic-repo` + `--ic-data-files` | optional single-row `DensityField` initial condition |
| `--path` | output directory for the MCMC chains |
| `--num-warmup` / `--num-samples` | sampler iterations |
| `--sampler {NUTS,HMC,MCLMC}` | inference kernel |
| `--checkpoints` | gradient checkpoints (memory vs compute for the reversible solver) |
| `--sigma-e` | shape-noise dispersion of the likelihood |

Plus the shared **cosmology**, **simulation**, **integration**, **lensing** and **prior** groups.
The reversible `DoubleKickDrift` solver makes the gradient through the whole N-body run memory
efficient. Post-process the chains with [`fli-extract`](fli-extract.md); the (work-in-progress)
walk-through lives under [Sampling & inference](../3-sampling-and-inference/).
