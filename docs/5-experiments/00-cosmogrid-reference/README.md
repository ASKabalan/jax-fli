# Experiment 00 — CosmoGrid reference

**Goal.** Provide the full-resolution ground truth that the accuracy experiments (01–07) validate
against: the native CosmoGrid density lightcone and two convergence (κ) references computed from it —
a dorian **ray-traced** κ (post-Born exact) and a **Born** κ — all published to the HuggingFace
dataset `ASKabalan/jax-fli-experiments`.

## Data

Built from `CosmoGrid/raw/cosmo_000001/run_0`. Three HuggingFace dataset configs:

| config | field | nside | dtype | contents |
|--------|-------|------:|-------|----------|
| `00-cosmogrid-density`    | `SphericalDensity`    | 2048 | uint16  | 69 lightcone shells, particle counts, z < 3.5 (native resolution) |
| `00-cosmogrid-kappa`      | `SphericalKappaField` | 2048 | float32 | 4 bins, **ray-traced** (dorian, full distortion matrix) from the density |
| `00-cosmogrid-kappa-born` | `SphericalKappaField` | 2048 | float32 | 4 bins, **Born** approximation from the same density |

The density is stored at the **raw uint16 precision** of CosmoGrid's `compressed_shells.npz` (≈ 7 GB,
lossless) rather than upcast to float32 — the jax-fli catalog serializer preserves the array dtype.
Both κ maps use the Stage-3 source n(z) by default (DES Y3 is a one-line `--nz des` switch); the
ray-traced κ replaces the earlier Stage-3 *forecast* κ, which was only nside 512.

```python
from datasets import load_dataset
from jax_fli.io import Catalog

ds = load_dataset("ASKabalan/jax-fli-experiments", "00-cosmogrid-density", split="train").with_format("numpy")
density = Catalog.from_dataset(ds).field[0]   # SphericalDensity (69, npix) uint16, nside 2048
```

## How to run

Four scripts. The two κ compute scripts run on the cluster and **only save parquet**;
`publish_local.py` uploads them afterwards (so `HF_TOKEN` is only needed where you publish). All take
`--nside` (omit for native/raw) and `--nz {s3,des}`.

**1 · Density — `publish_density_2048.py`** (CPU, ~16 GB RAM). Rebuilds the 2048 uint16 lightcone from
the local raw npz and overwrites `00-cosmogrid-density`:

```bash
python publish_density_2048.py --self-test   # validate the >2 GB parquet split round-trip (fast)
python publish_density_2048.py --publish     # build + overwrite the HuggingFace config
```

**2 · Ray-traced κ — `raytrace_kappa.py`** (dorian + **MPI**). dorian is replicated numpy+MPI — every
rank holds the full lightcone in host RAM (~14 GB at float32, 69 shells @ nside 2048) and MPI splits
the work; only rank 0 writes:

```bash
srun -n <ranks> python raytrace_kappa.py --nz s3 --out kappa_raytrace.parquet
mpirun --oversubscribe -np 64 python raytrace_kappa.py --smoke-test   # MPI plumbing test (synthetic, no HF)
```

**3 · Born κ — `born_kappa.py`** (JAX-distributed, like `docs/2-advanced-usage/11-multi-host-pm.py`).
Loads the density **sharded** across the device mesh (`P("x","y")`, npix per device) and runs the
fully-JAX Born approximation — one process per GPU, `jax.distributed` coordinates them:

```bash
srun -n $SLURM_NTASKS python born_kappa.py --multihost --nz s3 --out kappa_born.parquet
XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
    python born_kappa.py --smoke-test --nbins 2     # sharded-load + Born test on fake devices
```

**4 · Publish — `publish_local.py`** (local, `HF_TOKEN`). Uploads both κ parquets and registers their
configs in the dataset card (ray-traced → `00-cosmogrid-kappa`, Born → `00-cosmogrid-kappa-born`):

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
