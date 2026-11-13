"""
harmonize/hpsweep/submit.py
Expand a sweep grid YAML and emit one `sbatch` command per combo.

Usage (dry-run, prints commands without submitting):

    python -m harmonize.hpsweep.submit configs/sweeps/initial.yaml --dry-run

Real submit:

    python -m harmonize.hpsweep.submit configs/sweeps/initial.yaml \
        --partition gpu-single --gres gpu:A40:1 --time 02:00:00 \
        --wrapper jobs/run_sweep_one.sh

The wrapper script is expected to be `jobs/run_sweep_one.sh`, which
takes the run-name + all model CLI args as positional arguments and
forwards them to `harmonize.train_forecast`.
"""

from __future__ import annotations
import os

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .grid import (
    combos_as_cli_args,
    expand_grid,
    format_run_name,
    load_grid_yaml,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Submit a sweep of train_forecast jobs.")
    p.add_argument("grid_yaml", type=Path,
                   help="YAML file with fixed: / axes: blocks (see grid.py docstring).")
    p.add_argument("--wrapper", type=str,
                   default="jobs/run_sweep_one.sh",
                   help="SLURM wrapper script invoked once per combo.")
    p.add_argument("--chdir", type=str,
                   default=os.environ.get("PROJECT_ROOT", "."),
                   help="Working directory passed to sbatch --chdir.")
    p.add_argument("--partition", type=str, default="gpu-single")
    p.add_argument("--gres",      type=str, default="gpu:A40:1")
    p.add_argument("--time",      type=str, default="08:00:00")
    p.add_argument("--mem",       type=str, default="48G")
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--dry-run",   action="store_true",
                   help="Print sbatch commands without submitting.")
    p.add_argument("--limit",     type=int, default=None,
                   help="Cap the number of submitted jobs (debugging).")
    p.add_argument("--manifest",  type=Path, default=None,
                   help="Write JSON manifest of submitted jobs to this path.")
    args = p.parse_args()

    cfg = load_grid_yaml(args.grid_yaml)
    sweep_name = cfg.get("name", args.grid_yaml.stem)
    combos = expand_grid(cfg)
    if args.limit is not None:
        combos = combos[: args.limit]

    print(f"Sweep '{sweep_name}': {len(combos)} combinations")
    submitted: list[dict] = []

    for combo in combos:
        run_name = format_run_name(combo, sweep_name)
        cli_args = combos_as_cli_args(combo)

        # sbatch <slurm args> <wrapper> <run_name> <cli args...>
        sbatch_args = [
            "sbatch",
            "--job-name", run_name,
            "--chdir", args.chdir,
            "--parsable",
            "--partition", args.partition,
            "--gres", args.gres,
            "--time", args.time,
            "--mem", args.mem,
            "--cpus-per-task", str(args.cpus_per_task),
            args.wrapper,
            run_name,
            *cli_args,
        ]

        if args.dry_run:
            print(" ".join(shlex.quote(a) for a in sbatch_args))
            submitted.append({"run_name": run_name, "combo": combo})
            continue

        try:
            out = subprocess.check_output(sbatch_args, text=True).strip()
        except subprocess.CalledProcessError as e:
            print(f"sbatch FAILED for {run_name}: {e}", file=sys.stderr)
            continue
        # `sbatch` prints "Submitted batch job <id>"
        job_id = out.rsplit(None, 1)[-1]
        print(f"  {job_id}  {run_name}")
        submitted.append({"job_id": job_id, "run_name": run_name, "combo": combo})

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(
            {"sweep": sweep_name, "submitted": submitted}, indent=2
        ))
        print(f"\nManifest written to {args.manifest}")


if __name__ == "__main__":
    main()
