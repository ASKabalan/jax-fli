# Experiment 00 — CosmoGrid reference

**Goal.** Provide the full-resolution ground truth that the accuracy experiments (01–07) validate
against: the native CosmoGrid density lightcone, CosmoGrid's **own** Stage-3 forecast κ, and two
convergence (κ) references computed from the density — a dorian **ray-traced** κ (post-Born exact) and
a **Born** κ — all published to the HuggingFace dataset `ASKabalan/jax-fli-experiments`.

## Data

Built from `cosmo_000001` (density from `raw/cosmo_000001/run_0`; the forecast κ from the matching
`stage3_forecast/cosmo_000001/perm_0000`, so it shares the same cosmology). Four HuggingFace dataset
configs:

| config | field | nside | dtype | contents |
|--------|-------|------:|-------|----------|
| `00-cosmogrid-density`        | `SphericalDensity`    | 2048 | uint16  | 69 lightcone shells, particle counts, z < 3.5 (native resolution) |
| `00-cosmogrid-kappa`          | `SphericalKappaField` |  512 | float32 | 4 bins, CosmoGrid's own **Stage-3 forecast** κ (the simulation's published convergence maps) |
| `00-cosmogrid-kappa-raytrace` | `SphericalKappaField` | 2048 | float32 | 4 bins, **ray-traced** (dorian, full distortion matrix) from the density |
| `00-cosmogrid-kappa-born`     | `SphericalKappaField` | 2048 | float32 | 4 bins, **Born** approximation from the same density |

The density is stored at the **raw uint16 precision** of CosmoGrid's `compressed_shells.npz` (≈ 7 GB,
lossless) rather than upcast to float32 — the jax-fli catalog serializer preserves the array dtype.
The computed κ (ray-traced + Born) use the Stage-3 source n(z) by default (DES Y3 is a one-line
`--nz des` switch); `00-cosmogrid-kappa` keeps CosmoGrid's own forecast κ at its native nside 512 as an
independent reference.

```python
from datasets import load_dataset
from jax_fli.io import Catalog

ds = load_dataset("ASKabalan/jax-fli-experiments", "00-cosmogrid-density", split="train").with_format("numpy")
density = Catalog.from_dataset(ds).field[0]   # SphericalDensity (69, npix) uint16, nside 2048
```

## How to run

Five scripts. The density and forecast-κ scripts rebuild from the local CosmoGrid files and publish
directly; the two computed-κ scripts run on the cluster and **only save parquet**, and
`publish_local.py` uploads them afterwards (so `HF_TOKEN` is only needed where you publish). Each
publisher also takes `--check`, which loads its **published** config from HuggingFace and prints the
stored field attributes (no rebuild) — i.e. it inspects what is actually live on the Hub.

**1 · Density — `publish_density_2048.py`** (CPU, ~16 GB RAM). Rebuilds the 2048 uint16 lightcone from
the local raw npz and overwrites `00-cosmogrid-density`:

```bash
python publish_density_2048.py --self-test   # validate the >2 GB parquet split round-trip (fast)
python publish_density_2048.py --check       # inspect the published 00-cosmogrid-density config on HF
python publish_density_2048.py --publish     # build + overwrite the HuggingFace config
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
