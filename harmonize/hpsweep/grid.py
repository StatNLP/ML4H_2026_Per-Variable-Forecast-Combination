"""
harmonize/hpsweep/grid.py
YAML grid → list of HP dicts (Cartesian product) + run-name formatter.

Grid YAML format (see configs/sweeps/initial.yaml):

    name: initial_grid
    description: "Paper-1 reproduction first sweep."

    fixed:                  # applied to every combination
        train_h5: "data/dataset/{dataset}_seed42_train__{tag}.h5"
        val_h5:   "data/dataset/{dataset}_seed42_val__{tag}.h5"
        epochs: 60

    axes:                   # Cartesian product over these
        dataset:    [umm, mimic_iii]
        tag:        [sofa_first_fwd]
        model:      [dlinear, dense_autoregressive]
        loss:       [mse, huber_per_var]
        lr:         [3e-4, 1e-3]
        schedule:   [schedule_di]

`expand_grid()` returns a list of dicts, one per combination.
`format_run_name(combo, name_template)` produces a deterministic short id.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import yaml


def load_grid_yaml(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "axes" not in cfg or not isinstance(cfg["axes"], dict):
        raise ValueError(f"{path}: missing `axes:` block")
    return cfg


def expand_grid(cfg: dict) -> list[dict[str, Any]]:
    """Cartesian product over `axes`, merged with `fixed` constants.

    Returned dicts have `fixed` values formatted using `{key}` placeholders
    against the combo, so paths can reference axis vars
    (e.g. `train_h5: "data/dataset/{dataset}_..._{tag}.h5"`).
    """
    axes = cfg["axes"]
    fixed = cfg.get("fixed", {}) or {}

    keys = list(axes.keys())
    value_lists = [axes[k] if isinstance(axes[k], list) else [axes[k]] for k in keys]

    combos: list[dict[str, Any]] = []
    for vals in itertools.product(*value_lists):
        combo = dict(zip(keys, vals))
        # Format any string template values in `fixed` against the combo
        for k, v in fixed.items():
            if isinstance(v, str):
                try:
                    combo[k] = v.format(**combo)
                except KeyError:
                    combo[k] = v
            else:
                combo[k] = v
        combos.append(combo)
    return combos


def format_run_name(combo: dict[str, Any], sweep_name: str,
                    abbrev_axes: list[str] | None = None) -> str:
    """Build a short, deterministic run name. Includes the sweep name,
    the values of the chosen axes (abbreviated), and a 6-char hash of the
    full combo (so even non-abbreviated runs stay unique)."""
    if abbrev_axes is None:
        abbrev_axes = ["dataset", "model", "loss", "lr"]
    parts = [sweep_name]
    for ax in abbrev_axes:
        if ax in combo:
            v = str(combo[ax])
            parts.append(_short(v))
    h = hashlib.sha1(str(sorted(combo.items())).encode()).hexdigest()[:6]
    parts.append(h)
    return "_".join(parts)


def _short(s: str) -> str:
    """Lossy short-form for paths and model names."""
    s = s.replace("dense_autoregressive", "denseAR").replace("persistence", "pers")
    s = s.replace("schedule_", "sch")
    s = s.replace(".", "").replace("/", "-").replace(":", "")
    return s[:20]


def combos_as_cli_args(combo: dict[str, Any],
                       *, model_key: str = "model",
                       passthrough_keys: tuple[str, ...] = (
                           "model", "train_h5", "val_h5", "epochs",
                           "batch_size", "lr", "loss", "schedule",
                           "d_model", "d_ff", "n_heads", "n_layers",
                           "dropout", "huber_delta",
                           "patience_lr", "patience_stop", "seed",
                       )) -> list[str]:
    """Turn one combo into the argv list for `harmonize.train_forecast`."""
    args: list[str] = []
    for k in passthrough_keys:
        if k in combo:
            flag = "--" + k.replace("_", "-")
            args += [flag, str(combo[k])]
    return args
