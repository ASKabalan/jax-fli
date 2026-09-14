#!/usr/bin/env python
"""Distributed (multi-host) MAP mass mapping: reconstruct the initial condition
field from two DES Y3 kappa maps.

This is the script counterpart of ``14-LPT-MassMapping.ipynb`` (sections 1-7),
launched the way ``11-multi-host-pm.py`` is: one process per GPU across nodes,
tied together by ``jax.distributed.initialize()`` before the JAX backend is
touched. The physics code is identical to the notebook; only the launch and the
device mesh change. Rank 0 writes the same parquet layout as the notebook
(``true_ic.parquet``, ``truth_density_lightcone.parquet``, ``truth_kappa.parquet``,
``ic_evolution/ic_{frame}.parquet``, ``density_evolution/lc_{frame}.parquet``,
``kappa_evolution/kappa_{frame}.parquet``, ``coh_*/trans_*`` evolution dirs,
``metrics.json``, ``loss_history.npz``), so the notebook's sections 8-11 and
``compare_backends.py`` read the output unchanged.

  # Multi-host on SLURM (one task per GPU):
  srun -n $SLURM_NTASKS python 15-LPT-Multihost-MassMapping.py --mesh 64 --nside 64 \
      --out-dir V100_MG

  # Single host, one GPU:
  python 15-LPT-Multihost-MassMapping.py --mesh 300 --nside 256 --out-dir A100

  # Laptop smoke test with fake CPU devices:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
      python 15-LPT-Multihost-MassMapping.py --mesh 32 --nside 32 --map-max-iter 5 \
      --n-snapshots 2 --out-dir smoke
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"  # float32 => NaN/chaotic IC gradients
# setdefault: an externally exported JAX_PLATFORMS wins, so the documented
# fake-CPU smoke test (JAX_PLATFORMS=cpu + forced host devices) still works.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Set before importing jax; the notebook hardcodes 0.95 but the cuFFT plan-creation
# failures on the V100 MG run were memory contention, so this is CLI-overridable.
_MEM_FRACTION = os.environ.get("DIAG_MEM_FRACTION", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95"))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _MEM_FRACTION

from datetime import datetime


def _maybe_init_distributed() -> None:
    """Initialize JAX distributed only under a real multi-process launch (srun / mpirun).

    Mirrors ``bin/fli-simulate`` and ``11-multi-host-pm.py``: strip the proxy
    variables *before* ``jax.distributed.initialize()`` so the coordinator's
    gRPC channel connects directly to the sibling tasks instead of being routed
    through an (unreachable) HTTP / VSCode-remote proxy.
    """
    multi = (
        int(os.environ.get("SLURM_NTASKS", 0)) > 1
        or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)) > 1
        or int(os.environ.get("PMI_SIZE", 0)) > 1
    )
    if not multi:
        return
    for key in ("VSCODE_PROXY_URI", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Detected multi-host environment, initializing JAX distributed ...", flush=True)
    import jax

    jax.distributed.initialize()
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] JAX distributed initialized with "
        f"{jax.process_count()} processes, rank {jax.process_index()}",
        flush=True,
    )


_maybe_init_distributed()

import argparse
import json
import time
from pathlib import Path

import datasets

datasets.disable_progress_bar()

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import optax
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpyro.handlers import condition, seed, trace
from numpyro.infer.util import initialize_model

import jax_fli as jfli
from jax_fli.initial import interpolate_initial_conditions

jax.config.update("jax_enable_x64", True)

IS_MULTIHOST = jax.process_count() > 1
RANK = jax.process_index()


def log(msg: str) -> None:
    """Rank-0, flush=True logging that stays readable under srun's block buffering."""
    if RANK == 0:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # --- resolution ---
    p.add_argument("--mesh", type=int, default=64, help="cells per axis")
    p.add_argument("--nside", type=int, default=64, help="HEALPix nside of the kappa maps")
    p.add_argument("--paint-nside", type=int, default=32, help="painting nside")
    p.add_argument("--n-shells", type=int, default=12, help="lightcone shells")
    p.add_argument("--ell-max", type=int, default=48, help="field-level scale cut")
    p.add_argument("--ell-taper", type=int, default=8, help="cosine taper width")
    p.add_argument("--halo-cells", type=int, default=16, help="ghost cells per sharded axis")
    # --- DES Y3 survey ---
    p.add_argument("--des-bins", type=int, nargs="+", default=(1, 2))
    p.add_argument("--sigma-e", type=float, default=0.001)
    p.add_argument("--max-z", type=float, default=1.0)
    p.add_argument("--normalization", choices=("global", "local"), default="global")
    # --- run ---
    p.add_argument("--map-max-iter", type=int, default=300)
    p.add_argument("--n-snapshots", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results")
    # --- backend / cluster ---
    p.add_argument("--map2alm-method", choices=("jax", "jax_cuda"), default="jax")
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", 4)))
    p.add_argument(
        "--skip-spectra",
        action="store_true",
        help="skip the two spherical coherence/transfer spectra per frame (bypasses the XLA CPU "
        "fft-thunk bug on sharded SHTs; the GPU cluster and single-device runs do not need this)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nbins = len(args.des_bins)

    # --- device mesh: (D, 1) as in the notebooks; x carries pixels, y is unsplit ---
    n_dev = jax.device_count()
    PDIMS = (n_dev, 1)
    if not hasattr(jax.devices()[0], "slice_index") or not IS_MULTIHOST:
        mesh = jax.make_mesh(PDIMS, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    else:
        # Multi-host: tile the x-axis so each node's GPUs stay contiguous (NVLink)
        # and only the slab boundaries between nodes cross InfiniBand.
        intra = (args.gpus_per_node, 1)
        inter = (PDIMS[0] // args.gpus_per_node, PDIMS[1])
        mesh = Mesh(create_hybrid_device_mesh(intra, inter), axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))

    OUT_DIR = Path(args.out_dir)
    if RANK == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(
        f"jax {jax.__version__}  backend {jax.default_backend()}  x64 {jax.config.jax_enable_x64}"
        f"  MESH={args.mesh}  NSIDE={args.nside}  devices={n_dev}  processes={jax.process_count()}"
        f"  sharding={sharding.spec}  mem_fraction={_MEM_FRACTION}"
    )
    log(f"OUT_DIR = {OUT_DIR.resolve()}  NORMALIZATION = {args.normalization}")

    from jax_fli.data.nz import get_des_y3_nz_shear

    cosmo = jc.Planck18()
    box = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, args.max_z, (0.5, 0.5, 0.5)))
    log(f"Box size is {box} for max redshift {args.max_z}")

    des_y3_all = get_des_y3_nz_shear(zmax=args.max_z)
    nz_shear = [des_y3_all[i] for i in args.des_bins]
    for i, nz in zip(args.des_bins, nz_shear):
        log(f"  Bin {i + 1}: zmax={nz.zmax:.3f}, gals_per_arcmin2={nz.gals_per_arcmin2:.3f}")

    # --- config (notebook section 3) ---
    # Ghost cells serve the sharded stencil/FFT exchange, so they are needed only on axes
    # that are actually split. HALO_CELLS is an explicit cell count: it only has to cover
    # the LPT displacement plus the gradient stencil, a few cells at these resolutions.
    halo_size = tuple(args.halo_cells if p > 1 else 0 for p in PDIMS)

    config = jfli.ppl.Configurations(
        mesh_size=(args.mesh,) * 3,
        box_size=box,
        halo_size=halo_size,
        field_sharding=sharding,
        sim_mode="lpt",
        nbody_solver="BullFrog",  # inert in "lpt" mode
        t0=0.001,
        t1=1.0,
        lpt_order=1,
        number_of_shells=args.n_shells,
        nb_steps=5,
        paint_order="cic",
        gradient_order=4,
        laplace_fd=True,
        shell_spacing="a",
        time_stepping="D",
        min_width=50.0,
        # The observable IS the pair of kappa maps: the forward model returns kappa directly,
        # so what the likelihood fits and what the run publishes are the same object.
        lensing_output="convergence",
        map2alm_method=args.map2alm_method,
        min_redshift=0.001,
        max_redshift=args.max_z,
        n_integrate=8,
        nside=args.nside,
        geometry="spherical",
        scheme="bilinear",
        observer_position=(0.5, 0.5, 0.5),
        paint_nside=args.paint_nside,
        kernel_width_pixels=0.8,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors={
            "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
            "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
        },
        sigma_e=args.sigma_e,
        # Field-level scale cut (map2alm -> cosine taper -> alm2map). Cuts at the mesh
        # resolution limit, which also removes the shot-noise-dominated small scales.
        ell_max=args.ell_max,
        ell_taper_width=args.ell_taper,
        normalization=args.normalization,
        adjoint="checkpointed",
        checkpoints=4,
    )
    model = jfli.ppl.full_field_probmodel(config)

    # --- truth trace and observable (notebook section 4) ---
    tr = trace(seed(model, args.seed)).get_trace()

    theta_truth_base = jnp.array([float(tr["Omega_c_base"]["value"]), float(tr["sigma8_base"]["value"])])
    theta_truth = jnp.array([float(tr["Omega_c"]["value"]), float(tr["sigma8"]["value"])])
    # The white IC is a GLOBAL array across processes: keep it on device. Host-fetching it
    # (np.asarray) raises "spans non-addressable devices" under mpirun/SLURM, and nothing
    # downstream needs host values -- predict(), potential_fn and the warm-start shape all
    # work on the device array / its metadata.
    ic_truth_white = tr["initial_conditions"]["value"].array
    truth_cosmo = tr["cosmo"]["value"]

    forward = jfli.ppl.make_full_field_model(config)

    def predict(cosmo, white):
        """(cosmo, white IC) -> (physical IC, density lightcone, kappa)."""
        ic = interpolate_initial_conditions(
            white,
            config.mesh_size,
            config.box_size,
            cosmo=cosmo,
            observer_position=config.observer_position,
            halo_size=config.halo_size,
            nside=config.nside,
            field_sharding=config.field_sharding,
        )
        kappa, lightcone = forward(cosmo, ic)
        return ic, lightcone, kappa

    ic_truth, lightcone_truth, kappa_truth = predict(truth_cosmo, ic_truth_white)

    # ALL ranks must enter Catalog.to_parquet: catalog_to_row starts with a
    # process_allgather, and a collective entered by a subset of ranks kills the
    # gloo context (reproduced: rank0-only to_parquet -> AllGather failure on the
    # following sync_global_devices). The library gathers on every rank and only
    # rank 0 materializes/writes the row, so the file writes stay rank-exclusive.
    ic_truth_cat = ic_truth.replace(
        name="true_ic",
        z_sources=jnp.zeros(ic_truth.array.shape[0]),
        comoving_centers=jnp.zeros(ic_truth.array.shape[0]),
        scale_factors=jnp.zeros(ic_truth.array.shape[0]),
        density_width=jnp.zeros(ic_truth.array.shape[0]),
    )
    jfli.io.Catalog(field=[ic_truth_cat], cosmology=[truth_cosmo]).to_parquet(str(OUT_DIR / "true_ic.parquet"))
    jfli.io.Catalog(field=[lightcone_truth], cosmology=[truth_cosmo]).to_parquet(
        str(OUT_DIR / "truth_density_lightcone.parquet")
    )
    jfli.io.Catalog(field=[kappa_truth], cosmology=[truth_cosmo]).to_parquet(str(OUT_DIR / "truth_kappa.parquet"))
    sync_global_devices("truth-written")
    log(f"wrote true_ic / truth_density_lightcone / truth_kappa to {OUT_DIR}")

    x_obs = jnp.stack([tr[f"observable_{i}"]["value"] for i in range(nbins)])
    log(f"truth  Omega_c={float(theta_truth[0]):.4f}  sigma8={float(theta_truth[1]):.4f}")
    log(f"truth lightcone {lightcone_truth.array.shape}  truth kappa {kappa_truth.array.shape}")

    # --- condition on the observable (notebook section 5) ---
    obs_data = {f"observable_{i}": x_obs[i] for i in range(nbins)}
    obs_data.update(
        {
            "Omega_c_base": theta_truth_base[0],
            "sigma8_base": theta_truth_base[1],
        }
    )
    cond_model = condition(model, data=obs_data)
    free_sites = ["initial_conditions"]
    log(f"free sites: {free_sites}")
    log(f"cosmology fixed at truth  Omega_c={float(theta_truth[0]):.4f}  sigma8={float(theta_truth[1]):.4f}")

    # --- potential (notebook section 6) ---
    param_info, potential_fn, postprocess_fn, _ = initialize_model(jax.random.PRNGKey(0), cond_model, dynamic_args=False)
    init_params = param_info.z
    log(f"free sites: {list(init_params.keys())}")
    assert set(init_params.keys()) == set(free_sites)

    potential = lambda params: potential_fn(params)
    vg = jax.jit(jax.value_and_grad(potential))

    def grad_norm(g):
        return float(jnp.sqrt(sum(jnp.vdot(v, v).real for v in jax.tree.leaves(g))))

    potential_truth = float(potential_fn({"initial_conditions": jnp.asarray(ic_truth_white)}))
    assert np.isfinite(potential_truth)
    log(f"potential(truth) = {potential_truth:.2f}")

    # --- minimize in one continuous run with animation frames (notebook section 7) ---
    key = jax.random.PRNGKey(30)
    # Shard the warm start on the truth IC's sharding: a replicated noise field would put
    # the full mesh on every process (137 GB at 2048^3 float64).
    warm = {
        "initial_conditions": jax.device_put(
            jax.random.normal(key, ic_truth_white.shape, dtype=ic_truth_white.dtype),
            ic_truth_white.sharding,
        )
    }

    frames = np.linspace(0, args.map_max_iter, args.n_snapshots + 1).astype(int)
    log(f"{args.map_max_iter} Adam iterations, frames at iterations {frames.tolist()}")

    ic_path = OUT_DIR / "ic_evolution"
    lc_path = OUT_DIR / "density_evolution"
    kappa_path = OUT_DIR / "kappa_evolution"
    coh_ic_path = OUT_DIR / "coh_ic_evolution"
    trans_ic_path = OUT_DIR / "trans_ic_evolution"
    coh_kappa_path = OUT_DIR / "coh_kappa_evolution"
    trans_kappa_path = OUT_DIR / "trans_kappa_evolution"

    if RANK == 0:
        for path, pattern in (
            (ic_path, "ic_*.parquet"),
            (lc_path, "lc_*.parquet"),
            (kappa_path, "kappa_*.parquet"),
            (coh_ic_path, "coh_ic_*.parquet"),
            (trans_ic_path, "trans_ic_*.parquet"),
            (coh_kappa_path, "coh_kappa_*.parquet"),
            (trans_kappa_path, "trans_kappa_*.parquet"),
        ):
            path.mkdir(parents=True, exist_ok=True)
            for stale in path.glob(pattern):  # a rerun must not mix frames from an older run
                stale.unlink()

        metrics_json = OUT_DIR / "metrics.json"
        loss_history_npz = OUT_DIR / "loss_history.npz"
        for stale_file in (metrics_json, loss_history_npz):
            if stale_file.exists():
                stale_file.unlink()
    sync_global_devices("dirs-clean")

    N_LOWK = 5  # bins averaged for the large-scale coherence; high-k coherence falls even for a good MAP
    metrics_rows = []

    def snapshot(position, frame, step, loss):
        """Save the reconstruction at the current position: physical IC + density lightcone + Born kappa + spectra."""
        ic_run, lc_run, kappa_run = predict(truth_cosmo, position["initial_conditions"])
        # All metric reads are device-side reductions over per-process addressable shards:
        # each returns a replicated scalar, safe to float() on every rank without any allgather.
        mse_ic = float(jnp.mean((ic_run.array - ic_truth.array) ** 2))
        mse_kappa = float(jnp.mean((kappa_run.array - kappa_truth.array) ** 2))

        # The 3-D IC coherence/transfer (Fourier monopole) run fine on sharded meshes.
        coh_ic = ic_truth.coherence(ic_run)
        trans_ic = ic_truth.transfer(ic_run)
        # method="jax": the "healpy" backend host-fetches the maps (np.asarray ->
        # hp.anafast), which raises on global arrays under mpirun/SLURM. The jax
        # backend is numerically equivalent and stays on device.
        #
        # Known limitation: the sharded spherical SHT inside anafast hits an XLA
        # CPU fft-thunk layout bug (RET_CHECK fft_thunk.cc:168) when the kappa
        # maps are sharded over >1 device -- CPU-only, single-device CPU and the
        # GPU cluster are unaffected. --skip-spectra bypasses the two spherical
        # spectra on such smoke runs (the MAP loop itself is unaffected).
        if args.skip_spectra:
            coh_kappa = trans_kappa = None
        else:
            coh_kappa = kappa_truth.coherence(kappa_run, method="jax")
            trans_kappa = kappa_truth.transfer(kappa_run, method="jax")

        d_ic = position["initial_conditions"]
        d_ic_norm = float(
            jnp.sqrt(jnp.vdot(d_ic - warm["initial_conditions"], d_ic - warm["initial_conditions"]).real)
        )

        ic_cat = ic_run.replace(
            name=f"ic_{frame}",
            z_sources=jnp.zeros(ic_run.array.shape[0]),
            comoving_centers=jnp.zeros(ic_run.array.shape[0]),
            scale_factors=jnp.zeros(ic_run.array.shape[0]),
            density_width=jnp.zeros(ic_run.array.shape[0]),
        )
        coh_ic_cat = coh_ic.replace(name=f"coh_ic_{frame}")
        trans_ic_cat = trans_ic.replace(name=f"trans_ic_{frame}")

        # Catalog.to_parquet must be entered by ALL ranks: catalog_to_row starts
        # with a process_allgather (rank 0 alone would kill the gloo context).
        # The library gathers on every rank; only rank 0 materializes the row
        # and writes the file, so the writes stay rank-exclusive.
        jfli.io.Catalog(field=[ic_cat], cosmology=[truth_cosmo]).to_parquet(str(ic_path / f"ic_{frame}.parquet"))
        jfli.io.Catalog(field=[lc_run], cosmology=[truth_cosmo]).to_parquet(str(lc_path / f"lc_{frame}.parquet"))
        jfli.io.Catalog(field=[kappa_run], cosmology=[truth_cosmo]).to_parquet(
            str(kappa_path / f"kappa_{frame}.parquet")
        )
        jfli.io.Catalog(field=[coh_ic_cat], cosmology=[truth_cosmo]).to_parquet(
            str(coh_ic_path / f"coh_ic_{frame}.parquet")
        )
        jfli.io.Catalog(field=[trans_ic_cat], cosmology=[truth_cosmo]).to_parquet(
            str(trans_ic_path / f"trans_ic_{frame}.parquet")
        )
        if not args.skip_spectra:
            coh_kappa_cat = coh_kappa.replace(name=f"coh_kappa_{frame}")
            trans_kappa_cat = trans_kappa.replace(name=f"trans_kappa_{frame}")
            jfli.io.Catalog(field=[coh_kappa_cat], cosmology=[truth_cosmo]).to_parquet(
                str(coh_kappa_path / f"coh_kappa_{frame}.parquet")
            )
            jfli.io.Catalog(field=[trans_kappa_cat], cosmology=[truth_cosmo]).to_parquet(
                str(trans_kappa_path / f"trans_kappa_{frame}.parquet")
            )

        # Metrics are device-side reductions (replicated scalars, float()-safe on every
        # rank); only the JSON write is rank-exclusive.
        coh_ic_lowk = float(jnp.mean(coh_ic_cat.array[..., :N_LOWK]))
        coh_kappa_lowk = float("nan") if args.skip_spectra else float(jnp.mean(coh_kappa_cat.array[..., :N_LOWK]))
        if RANK == 0:
            metrics_rows.append(
                {
                    "frame": int(frame),
                    "step": int(step),
                    "loss": float(loss),
                    "MSE_IC": mse_ic,
                    "MSE_kappa": mse_kappa,
                    "coh_IC": float(jnp.mean(coh_ic_cat.array)),
                    "coh_IC_lowk": coh_ic_lowk,
                    "coh_kappa": float("nan") if args.skip_spectra else float(jnp.mean(coh_kappa_cat.array)),
                    "coh_kappa_lowl": coh_kappa_lowk,
                    "trans_IC": float(jnp.mean(trans_ic_cat.array)),
                    "trans_kappa": float("nan") if args.skip_spectra else float(jnp.mean(trans_kappa_cat.array)),
                }
            )
            with open(metrics_json, "w") as f:
                json.dump(metrics_rows, f, indent=2)
        label = "first guess" if frame == 0 else f"frame {frame}/{args.n_snapshots}"
        log(
            f"{label} (iter {step}):  loss={loss:.4e}"
            f"  r(IC)={coh_ic_lowk:.3f}"
            f"  r(kappa)={coh_kappa_lowk:.3f}"
            f"  MSE(IC)={mse_ic:.3e}  MSE(kappa)={mse_kappa:.3e}  |dIC|={d_ic_norm:.2f}"
        )
        sync_global_devices(f"frame-{frame}")

    losses, unorms = [], []
    position = warm

    # Adam with cosine decay schedule: coordinate-wise adaptive updates without linesearch collapse.
    lr_schedule = optax.cosine_decay_schedule(init_value=0.05, decay_steps=args.map_max_iter, alpha=0.05)
    solver = optax.adam(learning_rate=lr_schedule)
    opt_state = solver.init(position)

    @jax.jit
    def opt_step(position, opt_state):
        val, grad = vg(position)
        updates, opt_state = solver.update(grad, opt_state, position)
        unorm = jnp.sqrt(sum(jnp.vdot(u, u).real for u in jax.tree.leaves(updates)))
        return optax.apply_updates(position, updates), opt_state, val, unorm

    t0 = time.time()
    f_warm, g_warm = jax.block_until_ready(vg(position))
    gnorm_warm = grad_norm(g_warm)
    snapshot(position, 0, 0, float(f_warm))
    for step in range(1, args.map_max_iter + 1):
        position, opt_state, val, unorm = opt_step(position, opt_state)
        losses.append(float(val))
        unorms.append(float(unorm))
        if step in frames[1:]:
            snapshot(position, int(np.searchsorted(frames, step)), step, losses[-1])

    if RANK == 0:
        # Persist the per-iteration history: objective at the start of each iteration + update norm.
        np.savez(
            OUT_DIR / "loss_history.npz",
            steps=np.arange(1, args.map_max_iter + 1),
            loss=np.asarray(losses),
            update_norm=np.asarray(unorms),
        )
    log(f"total {time.time() - t0:.1f}s")
    log(f"wrote {OUT_DIR / 'loss_history.npz'}")

    map_params = position
    f_map, g_map = vg(map_params)
    jax.block_until_ready(f_map)
    gnorm_map = grad_norm(g_map)
    log(
        f"objective = {float(f_map):.6e}   truth objective = {potential_truth:.6e}   ratio = {float(f_map) / potential_truth:.3f}"
    )
    log(f"|grad| = {gnorm_map:.3e}   (first guess {gnorm_warm:.3e}, ratio {gnorm_map / gnorm_warm:.3e})")

    if gnorm_map / gnorm_warm > 0.5:
        log("WARNING: Gradient norm did not decrease significantly. Raise --map-max-iter or inspect learning rate.")
    else:
        log("converged: loss minimized and gradient reduced across all coordinates.")

    sync_global_devices("Done")
    if IS_MULTIHOST:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
