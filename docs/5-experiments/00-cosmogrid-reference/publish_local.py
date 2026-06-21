#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — publish the ray-traced and Born κ parquets to HuggingFace.

The compute scripts run on the cluster and only SAVE parquet (raytrace_kappa.py -> kappa_raytrace.parquet,
born_kappa.py -> kappa_born.parquet). Run THIS locally (where ``HF_TOKEN`` lives) to upload them:

  - ray-traced κ          -> ``00-cosmogrid-kappa-raytrace``  (CosmoGrid's forecast κ stays at ``00-cosmogrid-kappa``)
  - Born κ (DES Y3 n(z))  -> ``00-cosmogrid-born-des``        (new config; registered in the dataset card YAML)
  - Born κ (Stage-3 n(z)) -> ``00-cosmogrid-born-s3``         (new config; registered in the dataset card YAML)

Whichever parquet exists is published; the card's ``configs:`` are updated so both load with
``datasets.load_dataset(REPO, <config>)``.

    python publish_local.py            # dry run — print exactly what would be uploaded
    python publish_local.py --yes      # upload both + update the dataset card (HF_TOKEN required)
    python publish_local.py --yes --born-des kappa_born_des.parquet --born-s3 kappa_born_s3.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = "ASKabalan/jax-fli-experiments"
HERE = Path(__file__).resolve().parent

# kind -> (config_name, default local parquet, repo basename under the dataset's kappa dir)
TARGETS = {
    "raytrace": ("00-cosmogrid-kappa-raytrace", "kappa_raytrace.parquet", "kappa_raytrace.parquet"),
    "born-des": ("00-cosmogrid-born-des", "kappa_born_des.parquet", "kappa_born_des.parquet"),
    "born-s3": ("00-cosmogrid-born-s3", "kappa_born_s3.parquet", "kappa_born_s3.parquet"),
}


def _load_card(api):
    """Return (meta, body): the dataset card's parsed YAML front-matter and the markdown body."""
    import yaml
    from huggingface_hub import hf_hub_download

    text = open(hf_hub_download(REPO, "README.md", repo_type="dataset")).read()
    parts = text.split("---")
    if len(parts) >= 3:
        return (yaml.safe_load(parts[1]) or {}), "---".join(parts[2:])
    return {}, text


def _config_path(meta, name):
    for cfg in meta.get("configs") or []:
        if cfg.get("config_name") == name:
            df = cfg["data_files"]
            return df[0] if isinstance(df, list) else df
    return None


def _ensure_config(meta, name, path):
    cfgs = meta.setdefault("configs", [])
    for cfg in cfgs:
        if cfg.get("config_name") == name:
            cfg["data_files"] = path
            return
    cfgs.append({"config_name": name, "data_files": path})


def _save_card(api, meta, body):
    import yaml

    text = "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---" + body
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raytrace", default=str(HERE / TARGETS["raytrace"][1]), help="ray-traced κ parquet")
    ap.add_argument(
        "--born-des", dest="born_des", default=str(HERE / TARGETS["born-des"][1]), help="Born κ parquet (DES Y3 n(z))"
    )
    ap.add_argument(
        "--born-s3", dest="born_s3", default=str(HERE / TARGETS["born-s3"][1]), help="Born κ parquet (Stage-3 n(z))"
    )
    ap.add_argument("--yes", action="store_true", help="actually upload (otherwise dry run)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    meta, body = _load_card(api)

    # All κ variants land next to CosmoGrid's forecast κ (00-cosmogrid-kappa) under distinct filenames.
    # The forecast config is owned by publish_kappa_512.py and is left untouched here.
    forecast_kappa = _config_path(meta, "00-cosmogrid-kappa") or "00-cosmogrid/kappa/cosmogrid_sample_kappa.parquet"
    kappa_dir = str(Path(forecast_kappa).parent)

    plan = []  # (kind, config, local, repo_path, is_new)
    for kind, local in (("raytrace", args.raytrace), ("born-des", args.born_des), ("born-s3", args.born_s3)):
        config, _default_local, repo_basename = TARGETS[kind]
        lp = Path(local)
        if not lp.exists():
            print(f"skip {kind}: {lp} not found")
            continue
        repo_path = _config_path(meta, config) or f"{kappa_dir}/{repo_basename}"
        plan.append((kind, config, str(lp), repo_path, _config_path(meta, config) is None))

    if not plan:
        print("nothing to publish (no parquet files found).")
        return

    print(f"Publish plan for {REPO}:")
    for kind, config, lp, rp, is_new in plan:
        print(
            f"  {kind:9s} {lp} ({Path(lp).stat().st_size / 1e9:.2f} GB)  ->  "
            f"config {config} {'(NEW)' if is_new else '(overwrite)'}  at  {rp}"
        )
    if not args.yes:
        print("\nDry run — re-run with --yes to upload and update the dataset card.")
        return

    for kind, config, lp, rp, _is_new in plan:
        print(f"uploading {kind} -> {rp} …")
        api.upload_file(path_or_fileobj=lp, path_in_repo=rp, repo_id=REPO, repo_type="dataset")
        _ensure_config(meta, config, rp)
    _save_card(api, meta, body)
    print(f"done. configs now: {[c['config_name'] for c in meta.get('configs', [])]}")


if __name__ == "__main__":
    main()
