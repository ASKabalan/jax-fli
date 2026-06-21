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
| `00-cosmogrid-density`        | `SphericalDensity`    | 2048 | float32 | particle **counts**, ~56 shells out to **z ≤ 1.6** (DES Y3 depth) — **one config, one row per shell** (load by streaming) |
| `00-cosmogrid-kappa`          | `SphericalKappaField` |  512 | float32 | 4 bins, CosmoGrid's own **Stage-3 forecast** κ (the simulation's published convergence maps) |
| `00-cosmogrid-kappa-raytrace` | `SphericalKappaField` | 2048 | float32 | 4 bins, **ray-traced** (dorian, full distortion matrix) from the density |
| `00-cosmogrid-kappa-born`     | `SphericalKappaField` | 2048 | float32 | 4 bins, **Born** approximation from the same density |

The density is loaded with `jfli.io.load_cosmogrid_lc(max_redshift=1.6)` and kept out to **z ≤ 1.6**:
that is the **DES Y3** source depth (its deepest bin ends z≈1.45, see [experiment 06](../06-cosmogrid-shells/)),
and the small Stage-3 tail beyond 1.6 is negligible for the lensing kernel, so the one reference serves
both the DES and Stage-3 κ paths. It is saved in the loader's native **COUNTS** unit (float32) — what
every `load_cosmogrid_lc` consumer gets; `jfli.born`/`jfli.raytrace` convert to overdensity internally.
The 2048 lightcone is too large to be one parquet — stacking it to write OOMs and `load_dataset`
combining ~56·npix elements overflows arrow's INT32 list offset — so `load_cosmogrid_lc(output=folder)`
writes **one `(1, npix)` parquet per shell** and they are published as **one config `00-cosmogrid-density`**.
Reassemble by **streaming** the config (a non-streaming load would combine the rows and overflow) and
concatenating along the shell axis. The computed κ (ray-traced + Born)
use the Stage-3 source n(z) by default (DES Y3 is a one-line `--nz des` switch); `00-cosmogrid-kappa`
keeps CosmoGrid's own forecast κ at its native nside 512.

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
from jax_fli import SphericalDensity
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
GLOB = "00-cosmogrid/catalogs/cosmogrid_density_nside2048_shell*.parquet"
# one config, one (npix,) row per shell. Snapshot the files once (idempotent; offline-capable), then
# STREAM from the LOCAL dir (a non-streaming load overflows arrow's int32 list-offset; and offline,
# load_dataset(REPO, ...) cannot resolve the config — point it at the snapshot dir instead).
local = snapshot_download(REPO, repo_type="dataset", allow_patterns=[GLOB, "README.md"])
ds = load_dataset(local, "00-cosmogrid-density", split="train", streaming=True)
density = SphericalDensity.stack([Catalog.from_dataset(row).field[0] for row in ds])   # (~56, npix) float32, nside 2048
```

## How to run

Five scripts, plus `download.py` to pre-cache the HuggingFace data. The density and forecast-κ
scripts rebuild from the local CosmoGrid files and publish directly; the two computed-κ scripts run
on the cluster and **only save parquet**, and `publish_local.py` uploads them afterwards (so
`HF_TOKEN` is only needed where you publish). Each publisher also takes `--check`, which loads its
**published** config from HuggingFace and prints the stored field attributes (no rebuild) — i.e. it
inspects what is actually live on the Hub.

**0 · Pre-cache — `download.py`** (online, one time). Snapshots the per-shell density parquets and
caches the forecast-κ config into `HF_HOME`. Run it once on a node with internet (a login node);
`raytrace_kappa.py` / `born_kappa.py` then stream from that warm cache and run offline (set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` on compute nodes — see below):

```bash
python download.py                  # snapshot density shells + cache the kappa config into HF_HOME
```

**1 · Density — `publish_density_2048.py`** (CPU). Loads the 2048 lightcone with
`jfli.io.load_cosmogrid_lc(max_redshift=--max-z)` (default **z ≤ 1.6**, DES Y3 depth), converts
COUNTS→**density** (ρ = N/V, float32), and writes **one parquet per shell** (each a `(1, npix)` row),
publishing them all under **one config `00-cosmogrid-density`** (one parquet for all shells would OOM on
write and overflow arrow's INT32 offset on load). The per-shell write keeps peak RAM low; load the config
by **streaming**:

```bash
python publish_density_2048.py --self-test                 # validate the serializer round-trip (fast)
python publish_density_2048.py --out /scratch/d.parquet     # write the per-shell parquets locally (no upload)
python publish_density_2048.py --check                      # STREAM + inspect the published 00-cosmogrid-density config
python publish_density_2048.py --publish                    # write shells + upload as one config 00-cosmogrid-density
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
uv run mpirun -n 8 python raytrace_kappa.py --nside 32                # local small-nside run (CPU + MPI)
mpirun --oversubscribe -np 64 python raytrace_kappa.py --smoke-test   # MPI plumbing test (synthetic, no HF)
```

**4 · Born κ — `born_kappa.py`** (JAX-distributed, like `docs/2-advanced-usage/11-multi-host-pm.py`).
Loads the density **sharded** across the device mesh (`P("x","y")`, npix per device) and runs the
fully-JAX Born approximation — one process per GPU, `jax.distributed` coordinates them:

```bash
srun -n $SLURM_NTASKS python born_kappa.py --nz s3 --out kappa_born.parquet
uv run mpirun -n 8 -x JAX_PLATFORMS=cpu python born_kappa.py --nside 32   # local CPU run (1 GPU ≠ 8 procs)
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
# Snapshot the per-shell parquets + the dataset README (the streaming loader resolves the config's
# data_files glob from it). A non-streaming load_dataset cannot be used to pre-cache — it overflows
# arrow's int32 list-offset across the 56 shells. download.py does exactly this snapshot.
HF_HOME=$WORK/hf_cache python download.py
# then on compute nodes (streaming reads the local cache):
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

`born_kappa.py` and `raytrace_kappa.py` already do this snapshot internally and stream from the local
dir, so once the cache is warm they run unchanged on offline compute nodes (with the env above set).
