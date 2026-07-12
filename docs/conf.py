project = "jax-fli"
copyright = "2025, Wassim Kabalan"
author = "Wassim Kabalan"

# jax_fli itself is deliberately NOT installed in the docs build environment (see
# .readthedocs.yaml: --no-install-project) — notebooks ship their own outputs and are
# never re-executed here, so the heavy jax/jaxpm/healpy/s2fft stack is unnecessary.
version = release = ""

extensions = [
    "myst_nb",
    "sphinx_copybutton",
]

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

# RTD builders have no GPU/cluster and hard resource/time limits — notebooks are
# committed with their outputs (some from full-scale GPU/cluster runs) and must never
# be re-executed at build time.
nb_execution_mode = "off"

root_doc = "index"

exclude_patterns = [
    "_build",
    "000_RUNS",
    "WORK_IN_PROGRESS",
    "**/.ipynb_checkpoints",
    # agent-instructions files, not documentation content
    "**/CLAUDE.md",
    # git-ignored notebook runtime artifacts, not documentation content
    "3-sampling-and-inference/output",
    "3-sampling-and-inference/rosenbrock_samples_nuts",
]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/DifferentiableUniverseInitiative/jax-fli",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
}
