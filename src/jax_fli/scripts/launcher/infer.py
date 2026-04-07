"""infer subcommand — wraps fli-infer."""
from __future__ import annotations

import os

from .parser import (
    add_common_sim_args,
    add_lensing_args,
    add_lightcone_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "infer",
        help="Submit a single fli-infer MCMC inference job",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_lensing_args(p)
    add_lightcone_args(p)

    g = p.add_argument_group("infer")
    g.add_argument("--observable-dir", default="observables")
    g.add_argument("--observable", required=True, help="Filename within --observable-dir (e.g. BORN_SMALL.parquet)")
    g.add_argument("--output-dir", default="results/inference_runs")
    g.add_argument("--mesh-size", nargs=3, type=int, default=[16, 16, 16], metavar=("MX", "MY", "MZ"))
    g.add_argument("--box-size", nargs=3, type=float, default=[1000.0, 1000.0, 1000.0], metavar=("BX", "BY", "BZ"))
    g.add_argument("--nb-shells", type=int, default=8, help="Number of lightcone shells (default: 8)")
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")
    g.add_argument("--chain-index", type=int, default=0)
    g.add_argument("--adjoint", choices=["checkpointed", "recursive"], default="checkpointed")
    g.add_argument("--checkpoints", type=int, default=10)
    g.add_argument("--num-warmup", type=int, default=1)
    g.add_argument("--num-samples", type=int, default=1)
    g.add_argument("--batch-count", type=int, default=2)
    g.add_argument("--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS")
    g.add_argument("--backend", choices=["numpyro", "blackjax"], default="blackjax")
    g.add_argument("--sigma-e", type=float, default=0.26)
    g.add_argument("--sample", nargs="+", default=["cosmo", "ic"], help="What to sample, e.g. cosmo ic")
    g.add_argument("--initial-condition", default=None, help="Path to IC parquet; omitted if not set")
    g.add_argument("--init-cosmo", action="store_true", help="Warm-start cosmology from observable")
    g.add_argument("--omega-c", type=float, default=0.2589)
    g.add_argument("--sigma8", type=float, default=0.8159)
    g.add_argument("--h", type=float, default=0.6774)
    g.add_argument("--seed", type=int, default=0)

    # infer_script.bash: GPUS_PER_NODE=1, NODES=1, NB_STEPS=40, ENABLE_X64=true
    p.set_defaults(
        gpus_per_node=1,
        nodes=1,
        nb_steps=40,
        func=run,
        enable_x64=True,
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_TRACEBACK_FILTERING": "off"}

    px, py = args.pdim
    obs_path = f"{args.observable_dir}/{args.observable}"
    obs_name = os.path.splitext(args.observable)[0]
    job_name = f"{args.constraint}_infer_{obs_name}_Oc{args.omega_c}_S8{args.sigma8}_s{args.seed}"
    out_path = f"{args.output_dir}/chain_{args.chain_index}"

    print(f"Submitting {job_name}")
    print(f"  -> Observable: {obs_path} | Seed: {args.seed} | Mesh: {args.mesh_size}")

    fli_cmd = [
        "fli-infer",
        "--observable",
        obs_path,
        "--path",
        out_path,
        "--mesh-size",
        *[str(m) for m in args.mesh_size],
        "--box-size",
        *[str(b) for b in args.box_size],
        "--nb-shells",
        str(args.nb_shells),
        "--pdim",
        str(px),
        str(py),
        "--nodes",
        str(args.nodes),
        "--halo-multiplier",
        str(args.halo_multiplier),
        "--t0",
        str(args.t0),
        "--t1",
        str(args.t1),
        "--nb-steps",
        str(args.nb_steps),
        "--lpt-order",
        str(args.lpt_order),
        "--interp",
        args.interp,
        "--scheme",
        args.scheme,
    ]
    if args.paint_nside is not None:
        fli_cmd += ["--paint-nside", str(args.paint_nside)]
    if args.kernel_width_arcmin is not None:
        fli_cmd += ["--kernel-width-arcmin", str(args.kernel_width_arcmin)]
    if args.drift_on_lightcone:
        fli_cmd.append("--drift-on-lightcone")
    if args.equal_vol:
        fli_cmd.append("--equal-vol")
    if args.dealiased:
        fli_cmd.append("--dealiased")
    if args.exact_growth:
        fli_cmd.append("--exact-growth")
    fli_cmd += ["--gradient-order", str(args.gradient_order)]
    if args.laplace_fd:
        fli_cmd.append("--laplace-fd")
    fli_cmd += [
        "--min-width",
        str(args.min_width),
        "--nz-shear",
        *[str(v) for v in args.nz_shear],
        "--min-z",
        str(args.min_z),
        "--max-z",
        str(args.max_z),
        "--n-integrate",
        str(args.n_integrate),
        "--adjoint",
        args.adjoint,
        "--checkpoints",
        str(args.checkpoints),
        "--num-warmup",
        str(args.num_warmup),
        "--num-samples",
        str(args.num_samples),
        "--batch-count",
        str(args.batch_count),
        "--sampler",
        args.sampler,
        "--backend",
        args.backend,
        "--sigma-e",
        str(args.sigma_e),
        "--sample",
        *args.sample,
    ]
    if args.initial_condition:
        fli_cmd += ["--initial-condition", args.initial_condition]
    if args.init_cosmo:
        fli_cmd.append("--init-cosmo")
    if args.enable_x64:
        fli_cmd.append("--enable-x64")
    fli_cmd += ["--seed", str(args.seed)]

    dispatch(args, job_name, "FLI_INFERENCE", fli_cmd, env=env)
