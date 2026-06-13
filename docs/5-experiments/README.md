# Experiments

Reproduction studies for specific analyses and papers. Each experiment is a single runnable
Python script that *saves* its figures (SVG for the web, PDF for a paper) plus a hand-written
page that embeds and explains them — so the same assets feed both the docs and a manuscript.

- [**Experiment 08 — Masked shear**](08-masked-shear/README.md). Kaiser–Squires κ → γ on a cut
  sky (DES Y3 footprint and observer visibility masks), comparing masked vs full-sky shear maps
  and mask-decoupled `EE` spectra. Loads the CosmoGrid convergence from the `00-cosmogrid-kappa`
  HuggingFace config (Experiment 0). Script: [`08-masked-shear.py`](08-masked-shear/08-masked-shear.py).

  [![Masked shear footprints](08-masked-shear/assets/fig01-masks.svg)](08-masked-shear/README.md)
