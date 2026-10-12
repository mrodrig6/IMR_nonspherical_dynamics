"""Command-line interface.

    imrns run examples/free_viscoelastic_single.toml
    imrns list
    imrns info examples/lic_viscoelastic_multi.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_case
from .runner import run_case

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _cmd_run(args) -> int:
    config = load_case(args.case)
    if args.backend:
        config.numerics.backend = args.backend
    if args.xN:
        config.numerics.xN = args.xN
    if args.steps:
        config.drive.n_steps = args.steps

    print(f"{config.name}: {config.description}")
    print(f"  modes {config.modes}   {config.scales().summary()}")
    print(f"  {config.numerics.model}/{config.numerics.closure} "
          f"xN={config.numerics.xN} steps={config.drive.n_steps} backend={config.numerics.backend}")

    result = run_case(config, compare_irrotational=not args.no_compare, progress=args.progress)
    rot = result.rotational
    print(f"  {rot.termination}; peak |eps| per mode: "
          + ", ".join(f"n={n}: {np.abs(rot.ep[i]).max():.4g}" for i, n in enumerate(rot.modes)))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out)
        print(f"  wrote {out}")
    return 0


def _cmd_list(_args) -> int:
    cases = sorted(EXAMPLES.glob("*.toml"))
    if not cases:
        print(f"no cases found in {EXAMPLES}")
        return 1
    for path in cases:
        try:
            print(f"  {path.stem:36s} {load_case(path).description}")
        except Exception as exc:  # pragma: no cover
            print(f"  {path.stem:36s} <could not load: {exc}>")
    return 0


def _cmd_info(args) -> int:
    config = load_case(args.case)
    print(f"name        {config.name}")
    print(f"description {config.description}")
    print(f"modes       {config.modes}")
    print(f"ep0         {config.ep0}")
    print(f"drive       {config.drive.kind}, Req={config.drive.Req:g} m, "
          f"t_end={config.drive.t_end:g}, steps={config.drive.n_steps}")
    m = config.material
    print(f"material    mu={m.mu:g} Pa s, G={m.G:g} Pa, alpha={m.alpha:g}, sigma={m.sigma:g} N/m")
    print(f"groups      {config.scales().summary()}")
    print(f"numerics    {config.numerics}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="imrns", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a case from a TOML file")
    run.add_argument("case", help="path to a case TOML file")
    run.add_argument("-o", "--output", help="write trajectories to this .npz")
    run.add_argument("--backend", choices=["numpy", "numba", "jax"])
    run.add_argument("--xN", type=int, help="override the radial grid size")
    run.add_argument("--steps", type=int, help="override the number of time steps")
    run.add_argument("--no-compare", action="store_true",
                     help="skip the irrotational comparison run")
    run.add_argument("--progress", action="store_true")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list the bundled cases")
    lst.set_defaults(func=_cmd_list)

    info = sub.add_parser("info", help="show a case's parameters and dimensionless groups")
    info.add_argument("case")
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        # Configuration problems are the user's to fix, so report them plainly
        # rather than as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
