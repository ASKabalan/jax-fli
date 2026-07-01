# jax-fli — Analysis app

A Streamlit app for browsing and plotting `jax-fli` catalogs (local Parquet files or the
`ASKabalan/jax-fli-experiments` HuggingFace dataset): field maps, angular `C_ℓ`, 3-D `P(k)`,
peak counts, PDFs, and starlet coefficients. This is the analysis-only half of what used to be a
combined simulate-and-analyze app — the SLURM/launcher forms were dropped since they only ever
produced a command string to copy-paste, which has no meaning inside a hosted Space.

This is prep for a future Hugging Face Space — no Space has been created yet.

## Run locally

```bash
uv sync --extra app          # add --extra starlet for the starlet-wavelet tab
cd app && uv run streamlit run app.py
```

Run from inside `app/` (not the repo root): Streamlit resolves `.streamlit/config.toml` relative
to the current working directory at invocation, not relative to the script path — this also
matches how it behaves once `app/` becomes a Space root on its own.

## Environment

- `JAX_FLI_COMPARE_RTOL` / `JAX_FLI_COMPARE_ATOL` (default `1e-1` each) — tolerance used by the
  spherical/density comparison plots.
- The HuggingFace dataset repo (`ASKabalan/jax-fli-experiments` by default) is an editable text
  field in the UI, not an env var.

## Future HF Space metadata (reference only)

When this becomes an actual Space, its README front-matter (in the Space repo, not this one) would
be:

```yaml
sdk: streamlit
app_file: app.py
python_version: "3.12"
```

**Open risk to solve at that point, not here:** HF's Streamlit SDK installs from
`requirements.txt` via plain `pip`, which does not read this repo's `[tool.uv.sources]` — and
`jax_fli` depends on git-branch forks of `jaxpm` and `jax-cosmo`, not their PyPI releases. A naive
`pip install jax-fli[app]` in a Space would silently pull the wrong upstream code instead. The
committed `app/requirements.txt` in this directory sidesteps that today (it's resolved from
`uv.lock`, so the forks come out pinned as `git+https://...@branch` lines) — regenerate it with
`uv export --extra app --no-hashes -o app/requirements.txt` whenever the `app` extra or the
lockfile changes, and re-verify it still resolves once a real Space is stood up.
