"""Everything the animation of experiment 13 draws, computed once from the 1200^3 DES Y3 MAP run and cached in
.cache/map_mesh_1200_DES.npz (map_anim.py renders from it). CPU is enough (~20 GB of RAM):

    JAX_PLATFORMS=cpu uv run --no-sync python map_data.py [--force]

Run: the initial conditions (IC) of a 1200^3 mesh are reconstructed by maximum a posteriori (Adam, 400 steps,
cosine decay from 0.05) from two DES Y3 convergence maps (source bins 2 and 3, nside 1024, sigma_e = 0.26,
1.48 galaxies per arcmin^2 and bin), with 2LPT on 22 equal-volume shells from 300 Mpc/h to z = 1 and a pixel
likelihood on kappa band-limited to 2 <= l <= 700 (taper 64), on 32 GPUs (15-LPT-Multihost-MassMapping.py).
21 steps are saved, one every 20 (0, 20, ..., 400); the 3-D IC is stored at 1024^3.

Per saved step and for the truth:
  proj    the IC projected on the sky with the lensing efficiency of the two bins, averaged
          (DensityField.sky_projection) at nside 1024 -- how lensing sees the IC
  kappa   the convergence of both bins
both stored at nside 256 for display.
Per saved step, against the truth, at nside 1024:
  r_kappa, r_proj           Pearson correlation of the maps
  coh, trans                cross-correlation coefficient C_l^{MAP,truth} / sqrt(C_l^MAP C_l^truth) and
                            sqrt(C_l^MAP / C_l^truth) of kappa, 2 <= l < 700 (above, the forward model has no kappa)
  l1                        starlet l1 norm of kappa (5 scales at nside 256, nu = w_j / sigma_j^truth,
                            40 bins over [-5, 5]); l1_truth for the truth
plus the joint Wiener-filter cross-correlation coefficient of both bins (the best a linear reconstruction reaches), and
the steps and the loss.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / ".cache" / "map_mesh_1200_DES.npz"
if NPZ.exists() and "--force" not in sys.argv:
    print(f"{NPZ.name} exists, pass --force to recompute")
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp
import jax_cosmo as jc
from huggingface_hub import snapshot_download

import jax_fli as jfli
from jax_fli.data.nz import get_des_y3_nz_shear
from jax_fli.summary_statistics import starlet_coefficients_spherical

REPO, RUN = "ASKabalan/jax-fli-sampling", "13-LPT-MassMapping/mesh_1200_DES"
out = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{RUN}/**"])) / RUN

# the likelihood band of the run (--ell-max 700 --ell-taper 64); the correlation is averaged up to ELL_MAX - ELL_TAPER
ELL_MIN, ELL_MAX, ELL_TAPER = 2, 700, 64
DES_BINS, MAX_Z, SIGMA_E, R_MIN = (1, 2), 1.0, 0.26, 300.0
ST_NSIDE, N_SCALES = int(2 ** np.ceil(np.log2(ELL_MAX / 3))), 5
NU_EDGES = np.linspace(-5.0, 5.0, 41)
SHOW_NSIDE = 256  # the maps the animation draws

rows = sorted(json.loads((out / "metrics.json").read_text()), key=lambda r: r["frame"])
ic_files = sorted((out / "ic_evolution").glob("ic_*.parquet"), key=lambda p: int(p.stem.split("_")[-1]))
kappa_files = sorted((out / "kappa_evolution").glob("kappa_*.parquet"), key=lambda p: int(p.stem.split("_")[-1]))
cosmo = jc.Planck18()
nz = [get_des_y3_nz_shear(zmax=MAX_Z)[i] for i in DES_BINS]
kappa_truth = jfli.io.Catalog.from_parquet(str(out / "truth_kappa.parquet")).field[0]
NSIDE = int(kappa_truth.nside)


def project(ic):
    return np.asarray(ic.sky_projection(cosmo, nz, nside=NSIDE, r_min=R_MIN).array, dtype=np.float64).mean(axis=0)


def starlet(m):
    m = hp.alm2map(hp.map2alm(np.asarray(m, dtype=np.float64), lmax=ELL_MAX), ST_NSIDE, lmax=ELL_MAX)
    return np.asarray(starlet_coefficients_spherical(m, nside=ST_NSIDE, nscales=N_SCALES)[0])


def l1_norm(coef, sigma):
    res = np.zeros((N_SCALES, len(NU_EDGES) - 1))
    for j in range(N_SCALES):
        nu = coef[j] / sigma[j]
        i = np.digitize(nu, NU_EDGES) - 1
        ok = (i >= 0) & (i < res.shape[1])
        res[j] = np.bincount(i[ok], weights=np.abs(nu[ok]), minlength=res.shape[1])
    return res


kt = np.asarray(kappa_truth.array, dtype=np.float64)
proj_t = project(jfli.io.Catalog.from_parquet(str(out / "true_ic.parquet")).field[0])
st_t = [starlet(kt[b]) for b in range(len(DES_BINS))]
sig_t = [c.std(axis=1) for c in st_t]
ell = np.arange(ELL_MAX + 1)
sel = (ell >= ELL_MIN) & (ell < ELL_MAX)
ctt = [hp.anafast(kt[b], lmax=ELL_MAX) for b in range(len(DES_BINS))]

# the joint Wiener filter of both bins: r_a^2 = [C (C + N)^-1 C]_aa / C_aa
C = np.array([[hp.anafast(kt[a], kt[b], lmax=ELL_MAX) for b in range(2)] for a in range(2)]).transpose(2, 0, 1)
N_ell = SIGMA_E**2 / (np.array([float(n.gals_per_arcmin2) for n in nz]) * (180 * 60 / np.pi) ** 2)
M = np.einsum("lab,lbc,lcd->lad", C[sel], np.linalg.inv(C[sel] + np.diag(N_ell)[None]), C[sel])
wiener = np.array([np.sqrt(M[:, b, b] / C[sel, b, b]) for b in range(2)])

proj, kappa, r_kappa, r_proj, coh, trans, l1 = [], [], [], [], [], [], []
for f_ic, f_k in zip(ic_files, kappa_files, strict=True):
    pp = project(jfli.io.Catalog.from_parquet(str(f_ic)).field[0])
    kp = np.asarray(jfli.io.Catalog.from_parquet(str(f_k)).field[0].array, dtype=np.float64)
    proj.append(hp.ud_grade(pp, SHOW_NSIDE).astype(np.float32))
    kappa.append(np.array(hp.ud_grade(kp, SHOW_NSIDE), dtype=np.float32))
    r_proj.append(np.corrcoef(proj_t, pp)[0, 1])
    r_kappa.append([np.corrcoef(kt[b], kp[b])[0, 1] for b in range(2)])
    c_, t_ = [], []
    for b in range(2):
        cpp, ctp = hp.anafast(kp[b], lmax=ELL_MAX), hp.anafast(kt[b], kp[b], lmax=ELL_MAX)
        c_.append((ctp / np.sqrt(ctt[b] * cpp))[sel])
        t_.append(np.sqrt(cpp / ctt[b])[sel])
    coh.append(c_)
    trans.append(t_)
    l1.append([l1_norm(starlet(kp[b]), sig_t[b]) for b in range(2)])
    print(f"{f_ic.stem}: r(kappa) {np.round(r_kappa[-1], 3)}, r(projected IC) {r_proj[-1]:.3f}", flush=True)

keep = ell[sel] <= ELL_MAX - ELL_TAPER
print(
    f"last step, 2 <= l <= {ELL_MAX - ELL_TAPER}: kappa cross-correlation coefficient {np.round(np.mean(np.array(coh[-1])[:, keep], axis=1), 3)}"
    f" (joint Wiener filter {np.round(wiener[:, keep].mean(axis=1), 3)}), sqrt(C_MAP / C_truth)"
    f" {np.round(np.mean(np.array(trans[-1])[:, keep], axis=1), 3)}"
)
l1_truth = np.array([l1_norm(st_t[b], sig_t[b]) for b in range(2)])
print(f"starlet l1 MAP / truth, fine -> coarse: {np.round(np.array(l1[-1]).sum(-1) / l1_truth.sum(-1), 2).tolist()}")

NPZ.parent.mkdir(exist_ok=True)
np.savez(
    NPZ,
    steps=np.array([r["step"] for r in rows]),
    loss=np.array([r["loss"] for r in rows]),
    proj_truth=hp.ud_grade(proj_t, SHOW_NSIDE).astype(np.float32),
    proj=np.array(proj),
    kappa_truth=np.array(hp.ud_grade(kt, SHOW_NSIDE), dtype=np.float32),
    kappa=np.array(kappa),
    r_kappa=np.array(r_kappa),
    r_proj=np.array(r_proj),
    ell=ell[sel],
    coh=np.array(coh),
    trans=np.array(trans),
    wiener=wiener,
    l1=np.array(l1),
    l1_truth=l1_truth,
    nu=0.5 * (NU_EDGES[1:] + NU_EDGES[:-1]),
    st_nside=ST_NSIDE,
    ell_max=ELL_MAX,
)
print(f"wrote {NPZ}")
