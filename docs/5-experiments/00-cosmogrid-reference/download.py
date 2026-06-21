"""Pre-cache the Experiment 00 reference data from HuggingFace for offline use.

The density (``00-cosmogrid-density``) is **56 per-shell parquet rows at nside 2048**. A plain
``load_dataset(REPO, "00-cosmogrid-density")`` cannot materialize it: concatenating the 56 shells into
one Arrow column overflows its int32 list-offset (56 × 12·2048² ≈ 2.8e9 > 2.1e9 elements). So the
density must be read by **streaming**, and offline the streaming loader must point at a **local dir**
(``load_dataset(REPO, CONFIG, streaming=True)`` cannot resolve the config under ``HF_HUB_OFFLINE=1``).
We therefore snapshot the per-shell parquet files (plus the dataset ``README.md``, which carries the
config→files glob) and stream from that local snapshot — exactly what ``born_kappa.py`` /
``raytrace_kappa.py`` do internally.

    python download.py                       # snapshot the density files + cache the kappa config
    HF_HUB_OFFLINE=1 python ...               # then stream offline from the local dir (see snippet)
"""

from datasets import load_dataset
from huggingface_hub import snapshot_download

REPO = "ASKabalan/jax-fli-experiments"
DENSITY_CONFIG = "00-cosmogrid-density"
DENSITY_GLOB = "00-cosmogrid/density/cosmogrid_density_nside2048_shell*.parquet"
KAPPA_CONFIG = "00-cosmogrid-kappa"

# Density: snapshot the per-shell parquet files (one (npix,) row each) + the README (the streaming
# loader resolves the config's data_files glob from it). snapshot_download caches into HF_HOME and is
# idempotent — re-runs only fetch what is missing, and it returns the cached path under HF_HUB_OFFLINE=1.
local = snapshot_download(REPO, repo_type="dataset", allow_patterns=[DENSITY_GLOB, "README.md"])
print(f"Density shells cached under: {local}")

# Kappa: small (nside 512, 4 bins) — a normal (non-streaming) load_dataset caches it fine.
kappa_data = load_dataset(REPO, KAPPA_CONFIG)
print(f"Kappa config cached: {kappa_data}")
