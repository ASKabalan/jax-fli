"""832^3 CosmoGrid IC vs a spectrally resampled copy -- P(k), transfer, coherence.

Heavy prep step; writes a small committed CSV that build.py turns into a figure. ``NT`` picks
the target mesh and the direction: ``NT=1024`` upsamples, ``NT=512`` downsamples.

STAGE 1  ``STAGE=dump``  parquet -> white_832.npy   (the parquet load alone peaks ~12 GB,
                                                     so it gets its own process)
STAGE 2  (default)       white_832.npy -> ic_spectra_<NS>_<NT>.csv

Why this does not call ``jfli.resample_white_field`` directly
------------------------------------------------------------
It would, but the shipped function works on FULL complex grids and materialises the 1024^3
real field: peak ~46 GB, which OOM-kills a 62 GB box. Everything the figure needs lives in
Fourier space, so this builds the target's HALF-complex (``rfftn``) k-array instead and never
inverse-transforms -- peak ~11 GB. ``STAGE=check`` asserts, at a small size, that both this
path and ``jfli.resample_white_field`` satisfy the same contract: every shared mode of the
target equals the source's exactly.

Mode sets -- keeping these apart is the whole correctness question
-----------------------------------------------------------------
*_blk   the SHARED block (|n_i| <= min(NS, NT)/2 - 1), the modes BOTH grids carry. Transfer and
        coherence are only defined here, and P11/P22/P12 must all be averaged over this one
        identical set; averaging each over its own grid's modes breaks Cauchy-Schwarz (r > 1).
*_all   each grid's own full mode set, for the P(k) panel.

``rfftn`` keeps only kz >= 0, so each mode with 0 < nz < N/2 stands for two real modes. Every
sum below carries that weight (1 on the nz = 0 and nz = N/2 planes, 2 elsewhere).

Normalization: W = rfftn(w)/N^1.5, so E|W|^2 = 1 per mode for a unit-variance white field and
the colored spectrum is simply P_lin(k) * <|W(k)|^2>.
"""

import os
import resource

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gc

import numpy as np
import scipy.fft

NS = int(os.environ.get("NS", 832))
NT = int(os.environ.get("NT", 1024))
BOX = 900.0
STAGE = os.environ.get("STAGE", "run")
NPY = os.environ.get("NPY", "white_832.npy")
OUT = os.environ.get("OUT", f"ic_spectra_{NS}_{NT}.csv")
REPO = "ASKabalan/jax-fli-experiments"
IC_IN_REPO = "14-inference-cosmogrid/truth/input_cg.parquet"
IC = os.environ.get("IC")  # set to a local parquet to bypass the Hub
SEED = 12345


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def block_indices(n_src, n_tgt):
    """Signed integer wavevectors both grids carry, as index arrays into each.

    In fftfreq order these are NOT a centred slice: n = 0..h-1 sit at the start of the axis
    and n = -1..-(h-1) at its end. h is the SMALLER grid's half-extent (so this works in both
    directions), and index h itself is dropped so the set stays symmetric under n -> -n, which
    is what keeps the inverse transform real.
    """
    h = min(n_src, n_tgt) // 2
    low = np.arange(h)
    high = np.arange(1, h)
    return (np.concatenate([low, n_src - high]), np.concatenate([low, n_tgt - high]))


def resample_rfft(ws, n_src, n_tgt, rng):
    """Half-complex k-array of the resampled field: fresh modes, then the shared block on top."""
    fresh = rng.standard_normal((n_tgt,) * 3, dtype=np.float32)
    wt = (scipy.fft.rfftn(fresh, workers=-1) / n_tgt**1.5).astype(np.complex64)
    del fresh
    gc.collect()
    si, ti = block_indices(n_src, n_tgt)
    kz = np.arange(min(n_src, n_tgt) // 2)  # rfft keeps kz >= 0 only
    wt[np.ix_(ti, ti, kz)] = ws[np.ix_(si, si, kz)]
    return wt


# --------------------------------------------------------------------------------------
if STAGE == "dump":
    from huggingface_hub import snapshot_download

    import jax_fli as jfli

    # The published truth IC, the same file run 4 reads via --ic-repo/--ic-data-files.
    ic = IC or f"{snapshot_download(REPO, repo_type='dataset', allow_patterns=IC_IN_REPO)}/{IC_IN_REPO}"
    print(f"[{rss():6.1f} GB] loading {ic}", flush=True)
    cat = jfli.io.Catalog.from_parquet(ic)
    w = np.asarray(cat.field[0].array, dtype=np.float32)
    c = cat.cosmology[0]
    np.save(NPY, w)
    np.save("cosmo_832.npy", np.array([float(c.Omega_c), float(c.Omega_b), float(c.h),
                                       float(c.n_s), float(c.sigma8), float(c.w0),
                                       float(c.wa), float(c.Omega_k), float(c.Omega_nu)]))
    print(f"[{rss():6.1f} GB] wrote {NPY} {w.shape} mean={w.mean():.3e} var={w.var():.8f}", flush=True)
    raise SystemExit(0)

if STAGE == "check":
    # Both paths must satisfy the same contract: every shared mode equals the source's.
    import jax
    from jaxpm.distributed import normal_field

    import jax_fli as jfli

    ns, nt = 48, 64
    src = np.asarray(normal_field(seed=jax.random.key(5), shape=(ns,) * 3), dtype=np.float32)
    Fs = np.fft.fftn(src) / ns**1.5
    si, ti = block_indices(ns, nt)

    shipped = np.asarray(jfli.resample_white_field(src, jax.random.key(6), (nt,) * 3))
    Fj = np.fft.fftn(shipped) / nt**1.5
    dj = np.max(np.abs(Fs[np.ix_(si, si, si)] - Fj[np.ix_(ti, ti, ti)]))

    ws = (scipy.fft.rfftn(src) / ns**1.5).astype(np.complex64)
    wt = resample_rfft(ws, ns, nt, np.random.default_rng(0))
    mine = scipy.fft.irfftn(wt * nt**1.5, s=(nt,) * 3)
    Fm = np.fft.fftn(mine) / nt**1.5
    dm = np.max(np.abs(Fs[np.ix_(si, si, si)] - Fm[np.ix_(ti, ti, ti)]))

    print(f"shipped jfli.resample_white_field : shared-block max|d| = {dj:.3e}")
    print(f"this script's rfft path           : shared-block max|d| = {dm:.3e}")
    print(f"rfft path stays real              : max|imag-equivalent| = {np.abs(mine.imag).max() if np.iscomplexobj(mine) else 0.0:.3e}")
    print(f"rfft path variance                : {mine.var():.6f}")
    # Tolerance is the float32 FFT round-off floor, ~sqrt(N)*eps for O(1) mode amplitudes,
    # not an algorithmic allowance: both paths copy modes exactly, in float32.
    tol = 5e-6
    ok = dj < tol and dm < tol
    print(f"VERDICT: {'both paths satisfy the contract' if ok else 'MISMATCH'} (float32 floor, tol={tol:.0e})")
    raise SystemExit(0 if ok else 1)

# --------------------------------------------------------------------------------------
import jax_cosmo as jc  # noqa: E402

_c = np.load("cosmo_832.npy")  # the truth catalog's own cosmology, saved by STAGE=dump
cosmo = jc.Cosmology(Omega_c=_c[0], Omega_b=_c[1], h=_c[2], n_s=_c[3], sigma8=_c[4],
                     w0=_c[5], wa=_c[6], Omega_k=_c[7], Omega_nu=_c[8])

print(f"[{rss():6.1f} GB] rfftn source ...", flush=True)
w = np.load(NPY, mmap_mode="r")
assert w.shape == (NS,) * 3, w.shape
WS = (scipy.fft.rfftn(np.ascontiguousarray(w), workers=-1) / NS**1.5).astype(np.complex64)
del w
gc.collect()
print(f"[{rss():6.1f} GB] WS {WS.shape}", flush=True)

print(f"[{rss():6.1f} GB] building target k-array ({NT}^3, rfft) ...", flush=True)
WT = resample_rfft(WS, NS, NT, np.random.default_rng(SEED))
gc.collect()
print(f"[{rss():6.1f} GB] WT {WT.shape}", flush=True)

kf = 2 * np.pi / BOX
NBIN = 140
# max(NS, NT), not NT: when downsampling, the SOURCE grid reaches further in k than the target
# and its own P(k) would be silently truncated by a target-sized upper edge.
kedges = np.logspace(np.log10(kf * 0.9), np.log10(kf * max(NS, NT) / 2 * np.sqrt(3) * 1.01), NBIN + 1)
kcen = np.sqrt(kedges[1:] * kedges[:-1])
kgrid = np.logspace(-4, 1, 512)
pkgrid = np.asarray(jc.power.linear_matter_power(cosmo, kgrid))

z = lambda: np.zeros(NBIN)
s11b, s22b, s12b, cb = z(), z(), z(), z()
s11a, c11a, pk1a = z(), z(), z()
s22a, c22a, pk2a = z(), z(), z()


def hweight(n, nz):
    """Hermitian multiplicity of each rfft mode: 1 on the nz=0 and nz=N/2 planes, else 2."""
    return np.where((nz == 0) | (nz == n // 2), 1.0, 2.0)


def binsum(dst, b, ok, v):
    dst += np.bincount(b[ok], weights=v[ok], minlength=NBIN)


for tag, W, n, (sa, ca, pa) in (("target", WT, NT, (s22a, c22a, pk2a)),
                                ("source", WS, NS, (s11a, c11a, pk1a))):
    nn = np.fft.fftfreq(n, d=1.0 / n).astype(np.int64)
    nz = np.arange(W.shape[2], dtype=np.int64)
    wz = hweight(n, nz)
    print(f"[{rss():6.1f} GB] binning {tag} grid ...", flush=True)
    for i in range(n):
        kmag = np.sqrt((nn[i] * kf) ** 2 + (nn[:, None] * kf) ** 2 + (nz[None, :] * kf) ** 2).ravel()
        b = np.digitize(kmag, kedges) - 1
        ok = (b >= 0) & (b < NBIN)
        wgt = np.broadcast_to(wz, W.shape[1:]).ravel()
        p = (np.abs(W[i]) ** 2).astype(np.float64).ravel() * wgt
        binsum(sa, b, ok, p)
        binsum(ca, b, ok, wgt)
        binsum(pa, b, ok, p * np.interp(kmag, kgrid, pkgrid))
        if i % 256 == 0:
            print(f"  {tag} slab {i}/{n}  [{rss():.1f} GB]", flush=True)

# shared block -- ONE mode set for P11, P22 and P12
si, ti = block_indices(NS, NT)
h = min(NS, NT) // 2  # the smaller grid's half-extent: works up AND down
kzb = np.arange(h, dtype=np.int64)
nsg = np.fft.fftfreq(NS, d=1.0 / NS).astype(np.int64)[si]  # signed n of the block, both grids
wzb = hweight(min(NS, NT), kzb)  # kzb never reaches either grid's Nyquist, so only kz=0 weighs 1
print(f"[{rss():6.1f} GB] binning shared block ...", flush=True)
for a, isrc in enumerate(si):
    a1 = WS[isrc][np.ix_(si, kzb)]
    a2 = WT[ti[a]][np.ix_(ti, kzb)]
    kmag = np.sqrt((nsg[a] * kf) ** 2 + (nsg[:, None] * kf) ** 2 + (kzb[None, :] * kf) ** 2).ravel()
    b = np.digitize(kmag, kedges) - 1
    ok = (b >= 0) & (b < NBIN)
    wgt = np.broadcast_to(wzb, a1.shape).ravel()
    binsum(s11b, b, ok, (np.abs(a1) ** 2).astype(np.float64).ravel() * wgt)
    binsum(s22b, b, ok, (np.abs(a2) ** 2).astype(np.float64).ravel() * wgt)
    binsum(s12b, b, ok, np.real(a1 * np.conj(a2)).astype(np.float64).ravel() * wgt)
    binsum(cb, b, ok, wgt)
    if a % 256 == 0:
        print(f"  block slab {a}/{si.size}  [{rss():.1f} GB]", flush=True)

# CSV, not .npz: `*.npz` and `*.npy` are gitignored repo-wide, and this artifact is meant to be
# committed so build.py can redraw the figure without the 2.3 GB IC. The theory curve is not
# stored -- build.py rebuilds it from the cosmology in the header.
cols = dict(kcen=kcen, s11b=s11b, s22b=s22b, s12b=s12b, cb=cb,
            s11a=s11a, c11a=c11a, pk1a=pk1a, s22a=s22a, c22a=c22a, pk2a=pk2a)
names = ["Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"]
with open(OUT, "w") as f:
    f.write("# generated by prep_ic_spectra.py -- DO NOT EDIT. Read by build.py.\n")
    f.write(f"# box={BOX} ns={NS} nt={NT} nbin={NBIN}\n")
    # float(v), not {v!r}: a numpy scalar reprs as "np.float64(0.25)", which float() cannot parse.
    f.write("# cosmo " + " ".join(f"{n}={float(v):.12g}" for n, v in zip(names, _c)) + "\n")
    f.write("# *b = shared-block modes (one set for 11/22/12); *a = each grid's own modes\n")
    f.write(",".join(cols) + "\n")
    for row in zip(*cols.values()):
        f.write(",".join(f"{v:.10e}" for v in row) + "\n")
print(f"[{rss():6.1f} GB] wrote {OUT}", flush=True)
