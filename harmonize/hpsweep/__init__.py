"""
harmonize/hpsweep/
HP-sweep tooling: take a small YAML grid file and emit one SLURM job per
hyperparameter combination. See `configs/sweeps/initial.yaml` for the
config format and `docs/2605_comprehensive_documentation.md` §3.4 for the design.
"""

from .grid import expand_grid, load_grid_yaml, format_run_name

__all__ = ["expand_grid", "load_grid_yaml", "format_run_name"]
