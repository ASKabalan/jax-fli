# Experiment 00 — CosmoGrid reference

**Goal.** Provide the full-resolution ground truth that the accuracy experiments (01–07) validate
against: the native CosmoGrid density lightcone, CosmoGrid's **own** Stage-3 forecast κ, and two
convergence (κ) references computed from the density — a dorian **ray-traced** κ (post-Born exact) and
a **Born** κ — all published to the HuggingFace dataset `ASKabalan/jax-fli-experiments`.

## Data

Built from `cosmo_000001` (density from `raw/cosmo_000001/run_0`; the forecast κ from the matching
`stage3_forecast/cosmo_000001/perm_0000`, so it shares the same cosmology). HuggingFace dataset configs:

| config | field | nside | dtype | contents |
|--------|-------|------:|-------|----------|
| `00-cosmogrid-density-NN`     | `SphericalDensity`    | 2048 | float32 | density (ρ = N/V), ~67 shells covering the source n(z) (z≲3.0), split into **N set configs** (`-00`,`-01`,…) |
| `00-cosmogrid-kappa`          | `SphericalKappaField` |  512 | float32 | 4 bins, CosmoGrid's own **Stage-3 forecast** κ (the simulation's published convergence maps) |
| `00-cosmogrid-kappa-raytrace` | `SphericalKappaField` | 2048 | float32 | 4 bins, **ray-traced** (dorian, full distortion matrix) from the density |
| `00-cosmogrid-kappa-born`     | `SphericalKappaField` | 2048 | float32 | 4 bins, **Born** approximation from the same density |

The 2048 density is too large to be a single config: serializing all shells at once OOMs (the `datasets`
ArrayND path balloons ~10×), and even if it fit, `load_dataset` combining ~67·npix ≈ 3.4 G elements
overflows arrow's INT32 list offset. So the shell axis is split into a few **set configs**
`00-cosmogrid-density-NN`, each small enough to write and to load independently — reassemble the full
lightcone by loading each and concatenating along the shell axis. It is saved as **density** (ρ = N/V,
float32) covering the source redshift support of **both** shears in `jfli.data` (Stage-3 z≈1.7, DES Y3
z≈3.0). The computed κ (ray-traced + Born) use the Stage-3 source n(z) by default (DES Y3 is a one-line
`--nz des` switch); `00-cosmogrid-kappa` keeps CosmoGrid's own forecast κ at its native nside 512.

```python
import re
import jax
import jax.numpy as jnp
from datasets import get_dataset_config_names, load_dataset
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
# the density is split into set configs 00-cosmogrid-density-NN — load each, concatenate along shells
cfgs = sorted(n for n in get_dataset_config_names(REPO) if re.fullmatch(r"00-cosmogrid-density-\d+", n))
fields = []
for n in cfgs:
    fields += Catalog.from_dataset(load_dataset(REPO, n, split="train").with_format("numpy")).field
density = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *fields)   # (~67, npix) float32, nside 2048
```

## How to run

Five scripts. The density and forecast-κ scripts rebuild from the local CosmoGrid files and publish
directly; the two computed-κ scripts run on the cluster and **only save parquet**, and
`publish_local.py` uploads them afterwards (so `HF_TOKEN` is only needed where you publish). Each
publisher also takes `--check`, which loads its **published** config from HuggingFace and prints the
stored field attributes (no rebuild) — i.e. it inspects what is actually live on the Hub.

**1 · Density — `publish_density_2048.py`** (CPU, ~16 GB RAM/set). Rebuilds the 2048 lightcone from the
local raw npz, converts COUNTS→**density** (ρ = N/V, float32), splits the shell axis into `--n-sets`
sets (default 4) and publishes each as its own config `00-cosmogrid-density-NN` (one config for all
shells would OOM on write and overflow arrow's INT32 offset on load). Keeps only the shells covering the
`jfli.data` source n(z) by default (`--all-shells` for all 69):

```bash
python publish_density_2048.py --self-test                 # validate the serializer round-trip (fast)
python publish_density_2048.py --out /scratch/d.parquet     # build the set parquets locally (no upload) to verify
python publish_density_2048.py --check                      # inspect the published 00-cosmogrid-density-NN configs on HF
python publish_density_2048.py --publish                    # build sets + upload as 00-cosmogrid-density-NN configs
```

**2 · CosmoGrid forecast κ — `publish_kappa_512.py`** (CPU). Loads CosmoGrid's own Stage-3 forecast κ
(nside 512, float32, 4 bins) from `stage3_forecast/cosmo_000001/perm_0000` via `load_cosmogrid_kappa`,
asserts it shares the density's cosmology, and overwrites `00-cosmogrid-kappa`:

```bash
python publish_kappa_512.py --self-test      # validate the κ parquet round-trip (fast)
python publish_kappa_512.py --check          # inspect the published 00-cosmogrid-kappa config on HF
python publish_kappa_512.py --publish        # build + overwrite the HuggingFace config
```

**3 · Ray-traced κ — `raytrace_kappa.py`** (dorian + **MPI**). dorian is replicated numpy+MPI — every
rank holds the full lightcone in host RAM (~14 GB at float32, 69 shells @ nside 2048) and MPI splits
the work; only rank 0 writes:

```bash
srun -n <ranks> python raytrace_kappa.py --nz s3 --out kappa_raytrace.parquet
mpirun --oversubscribe -np 64 python raytrace_kappa.py --smoke-test   # MPI plumbing test (synthetic, no HF)
```

**4 · Born κ — `born_kappa.py`** (JAX-distributed, like `docs/2-advanced-usage/11-multi-host-pm.py`).
Loads the density **sharded** across the device mesh (`P("x","y")`, npix per device) and runs the
fully-JAX Born approximation — one process per GPU, `jax.distributed` coordinates them:

```bash
srun -n $SLURM_NTASKS python born_kappa.py --multihost --nz s3 --out kappa_born.parquet
XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
    python born_kappa.py --smoke-test --nbins 2     # sharded-load + Born test on fake devices
```

**5 · Publish computed κ — `publish_local.py`** (local, `HF_TOKEN`). Uploads both computed-κ parquets
and registers their configs in the dataset card (ray-traced → `00-cosmogrid-kappa-raytrace`, Born →
`00-cosmogrid-kappa-born`), leaving CosmoGrid's forecast κ at `00-cosmogrid-kappa` untouched:

```bash
python publish_local.py          # dry run: print exactly what would be uploaded
python publish_local.py --yes    # upload both + update the card
```

**Offline cluster (Jean Zay — no internet on compute nodes).** Pre-cache the density on a login node,
then run offline (the same applies to any experiment that compares against CosmoGrid, e.g. 06, 07):

```bash
HF_HOME=$WORK/hf_cache python -c \
  "from datasets import load_dataset; load_dataset('ASKabalan/jax-fli-experiments', '00-cosmogrid-density', split='train')"
# then on compute nodes:
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```
