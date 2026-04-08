"""Top-level argparse entry point for the fli-launcher command."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="fli-launcher",
        description="jax-fli job launcher",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    from . import born_rt, dorian_rt, extract, grid, infer, samples, simulate

    for mod in [simulate, grid, samples, infer, born_rt, extract, dorian_rt]:
        mod.add_subparser(sub)

    args = parser.parse_args()
    # Pretty-print resolved configuration
    width = 60
    print("=" * width)
    print(f"  fli-launcher · {args.subcommand}")
    print("=" * width)
    skip = {"func", "subcommand"}
    for key, val in sorted(vars(args).items()):
        if key in skip:
            continue
        print(f"  {key:<30} {val}")
    print("=" * width)
    print()

    args.func(args)
