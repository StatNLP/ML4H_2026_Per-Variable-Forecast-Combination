"""
harmonize/train_forecast.py
Stage 4 — forecast training loop (masked MSE, scheduled sampling,
per-feature / per-horizon validation metrics).

See docs/2605_comprehensive_documentation.md

Usage:
    python -m harmonize.train_forecast \
        --model dense_autoregressive \
        --train-h5 data/dataset/umm_seed42_train__sofa_first_fwd.h5 \
        --val-h5   data/dataset/umm_seed42_val__sofa_first_fwd.h5 \
        --epochs 30 --batch-size 256 --lr 3e-4 \
        --schedule schedule_di \
        --run-name umm_sofa_denseAR_di
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from harmonize.icu_dataset import ICUDataset
from harmonize.models import build_model
from harmonize.triplet_dataset import TripletICUDataset, triplet_collate

log = logging.getLogger("train_forecast")


# ---------------------------------------------------------------------------
# Dataset representation: dense hourly grid (ICUDataset) vs raw triplet set
# (TripletICUDataset). Auto-detected from the HDF5 `representation` attr.
# ---------------------------------------------------------------------------

def _is_triplet_h5(h5_path: str) -> bool:
    import h5py
    with h5py.File(h5_path, "r") as f:
        return str(f.attrs.get("representation", "")) == "triplet"


def make_dataset(h5_path: str):
    """Return (dataset, collate_fn) for either representation."""
    if _is_triplet_h5(h5_path):
        return TripletICUDataset(h5_path), triplet_collate
    return ICUDataset(h5_path), None


def model_forward(model, batch, device, *,
                  x_target=None, teacher_forcing_ratio: float = 0.0):
    """Call the model with the right inputs for dense or triplet batches."""
    static = batch["static"].to(device, non_blocking=True)
    if "tri_val" in batch:
        return model(
            batch["tri_val"].to(device, non_blocking=True),
            batch["tri_time"].to(device, non_blocking=True),
            batch["tri_var"].to(device, non_blocking=True),
            batch["tri_pad"].to(device, non_blocking=True),
            static, x_target=x_target, teacher_forcing_ratio=teacher_forcing_ratio,
        )
    return model(
        batch["x_in"].to(device, non_blocking=True),
        batch["mask_in"].to(device, non_blocking=True),
        static, x_target=x_target, teacher_forcing_ratio=teacher_forcing_ratio,
    )


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------
#
# All losses are masked (only positions where `mask_target` is True contribute)
# and aggregated in one of two ways:
#
#   global        ∑_{i,t,f} L_{itf} M_{itf} / ∑_{i,t,f} M_{itf}
#                 — every observed position counted equally; variables with
#                   more measurements dominate the gradient.
#
#   per_variable  mean_f [ ∑_{i,t} L_{itf} M_{itf} / ∑_{i,t} M_{itf} ]
#                 — every feature counted equally regardless of measurement
#                   density (recommended for cross-variable forecasting).
#
# Base losses available: mse (L2), mae (L1), huber (Smooth L1, δ-parametrised).
# Huber is the recommended default for z-scored physiological data: quadratic
# inside ±δ standard deviations, linear outside (robust to outliers).
# ---------------------------------------------------------------------------


def _pointwise(pred: torch.Tensor, target: torch.Tensor, base: str,
               huber_delta: float, loss_power: float = 2.0) -> torch.Tensor:
    """Element-wise loss (no masking, no reduction). Replaces NaN target with 0."""
    target = torch.nan_to_num(target, nan=0.0)
    if base == "mse":
        return (pred - target).pow(2)
    if base == "power":
        # |e|^p with p in (2,3): a milder, stabler version of the cubic idea — penalises
        # large errors (deterioration excursions) more than MSE without the cubic's
        # vanishing gradient near 0. p=2.0 recovers MSE. (#6 deterioration-loss experiment.)
        return (pred - target).abs().pow(loss_power)
    if base == "mae":
        return (pred - target).abs()
    if base == "huber":
        # PyTorch's huber_loss uses (pred - target) internally; matches our convention.
        return torch.nn.functional.huber_loss(
            pred, target, reduction="none", delta=huber_delta
        )
    raise ValueError(f"unknown base loss: {base!r}")


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    base: str = "mse",
    per_variable: bool = False,
    unnormalised: bool = False,
    huber_delta: float = 1.0,
    loss_power: float = 2.0,
    feature_weights: "torch.Tensor | None" = None,
    sample_weights: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Masked + aggregated loss. See module docstring for the aggregation modes.

    Shapes:  pred, target, mask  →  (B, T, F).  Returns a scalar.

    `unnormalised=True` reproduces the Paper-1 convention: plain sum of the
    masked element-wise errors, *not* divided by the observation count. Mostly
    useful for reporting; for training it is sensitive to batch size and the
    learning rate must be scaled accordingly.
    """
    obs = mask.float()
    if sample_weights is not None:
        # per-window weight (e.g. ×K on deterioration windows): scaling obs makes every
        # downstream reduction a proper weighted mean (num & denom weighted alike).
        obs = obs * sample_weights.view(-1, 1, 1).to(obs.device, obs.dtype)
    e   = _pointwise(pred, target, base, huber_delta, loss_power) * obs

    def _feat_mean(per_feat: torch.Tensor) -> torch.Tensor:
        # Mean over the F axis, optionally weighted (clinically-weighted loss).
        if feature_weights is not None:
            w = feature_weights.to(per_feat.device, per_feat.dtype)
            return (per_feat * w).sum() / w.sum().clamp(min=1e-8)
        return per_feat.mean()

    if unnormalised:
        if per_variable:
            # sum per feature, then (weighted) mean across features.
            return _feat_mean(e.sum(dim=(0, 1)))
        return e.sum()

    if per_variable:
        num = e.sum(dim=(0, 1))
        den = obs.sum(dim=(0, 1)).clamp(min=1.0)
        return _feat_mean(num / den)

    return e.sum() / obs.sum().clamp(min=1.0)


# Backwards-compatible thin wrappers (used by smoke scripts / evaluation reports)
def masked_mse(pred, target, mask):
    return masked_loss(pred, target, mask, base="mse", per_variable=False)


def masked_mae(pred, target, mask):
    return masked_loss(pred, target, mask, base="mae", per_variable=False)


def masked_huber(pred, target, mask, delta=1.0):
    return masked_loss(pred, target, mask, base="huber",
                       per_variable=False, huber_delta=delta)


# ---------------------------------------------------------------------------
# Scheduled-sampling schedules — return p_teacher in [0, 1] for an epoch
# ---------------------------------------------------------------------------

def teacher_prob(schedule: str, epoch: int, max_epochs: int) -> float:
    if schedule in ("none", "teacher"):
        return 1.0
    if schedule == "student":
        return 0.0
    frac = epoch / max(1, max_epochs - 1)
    if schedule == "schedule_di":   # direct incremental: linear decay 1→0
        return max(0.0, 1.0 - frac)
    if schedule == "schedule_dd":   # direct decay: exponential
        return 0.5 ** (frac * 5.0)
    if schedule == "schedule_ri":   # random incremental: sigmoid
        return 1.0 / (1.0 + np.exp(10.0 * (frac - 0.5)))
    if schedule == "schedule_rd":   # random decay: 1 / (1 + epoch/k)
        return 1.0 / (1.0 + frac * 5.0)
    raise ValueError(f"unknown schedule: {schedule}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             feature_names: list[str], window_out: int,
             huber_delta: float = 1.0,
             day1_cutoff_h: float = 24.0) -> dict:
    """Compute all four loss aggregations + MAE on the loader, plus per-feature
    and per-horizon MSE for diagnostics.

    Returned scalars (lower-is-better):
        mse / mae / huber / rmse / mse_per_var / huber_per_var / mae_per_var
            — over ALL validation windows
        mse_per_var_day1 / huber_per_var_day1 / mae_per_var_day1
            — over "first-day" windows only (t0_hour < day1_cutoff_h = 24h).
              For stride=first all windows qualify and these equal the _all metrics.
              For sliding-window datasets these cover only early-admission windows.
        n_windows_all / n_windows_day1  — window counts for both subsets
        n_obs               — total observed positions across val set
    """
    model.eval()
    n_feat = len(feature_names)

    # Per-feature accumulators — all windows
    sq_per_feat    = torch.zeros(n_feat)
    abs_per_feat   = torch.zeros(n_feat)
    huber_per_feat = torch.zeros(n_feat)
    obs_per_feat   = torch.zeros(n_feat)

    # Per-feature accumulators — first-day windows (t0_hour < day1_cutoff_h)
    sq_per_feat_d1    = torch.zeros(n_feat)
    huber_per_feat_d1 = torch.zeros(n_feat)
    abs_per_feat_d1   = torch.zeros(n_feat)
    obs_per_feat_d1   = torch.zeros(n_feat)

    # Per-horizon (sum over B, F; one value per timestep)
    sq_per_horizon  = torch.zeros(window_out)
    obs_per_horizon = torch.zeros(window_out)

    n_windows_all = 0
    n_windows_d1  = 0

    for batch in loader:
        x_tgt   = batch["x_target"].to(device)
        m_tgt   = batch["mask_target"].to(device)
        t0      = batch["t0_hour"]          # (B,) on CPU
        # Physiology is student-forced (teacher_forcing_ratio=0 ⇒ use_teacher=False),
        # but x_target IS passed so the EXOGENOUS channels (dense_pteacher
        # medications) get teacher-forced — physician decisions are known, not
        # forecast. No-op for models without exogenous_idx. (Fixes a train/eval
        # mismatch: pteacher trained WITH true meds but was evaluated WITHOUT.)
        pred    = model_forward(model, batch, device,
                                x_target=(torch.from_numpy(
                                              AR_PRED_VAL[batch["row_idx"].numpy()]
                                          ).to(device) if AR_PRED_VAL is not None else x_tgt),
                                teacher_forcing_ratio=1.0 if AR_PRED_VAL is not None else 0.0)
        obs     = m_tgt.float()
        tgt     = torch.nan_to_num(x_tgt, nan=0.0)

        sq    = (pred - tgt).pow(2) * obs
        ab    = (pred - tgt).abs()  * obs
        hb    = torch.nn.functional.huber_loss(
                    pred, tgt, reduction="none", delta=huber_delta) * obs

        sq_per_feat    += sq.sum(dim=(0, 1)).cpu()
        abs_per_feat   += ab.sum(dim=(0, 1)).cpu()
        huber_per_feat += hb.sum(dim=(0, 1)).cpu()
        obs_per_feat   += obs.sum(dim=(0, 1)).cpu()

        sq_per_horizon  += sq.sum(dim=(0, 2)).cpu()
        obs_per_horizon += obs.sum(dim=(0, 2)).cpu()

        n_windows_all += t0.shape[0]

        # First-day subset: broadcast (B,) mask → (B, 1, 1)
        is_d1 = (t0 < day1_cutoff_h).float()[:, None, None]   # CPU
        sq_cpu = sq.cpu();  ab_cpu = ab.cpu()
        hb_cpu = hb.cpu();  obs_cpu = obs.cpu()
        sq_per_feat_d1    += (sq_cpu  * is_d1).sum(dim=(0, 1))
        abs_per_feat_d1   += (ab_cpu  * is_d1).sum(dim=(0, 1))
        huber_per_feat_d1 += (hb_cpu  * is_d1).sum(dim=(0, 1))
        obs_per_feat_d1   += (obs_cpu * is_d1).sum(dim=(0, 1))
        n_windows_d1      += int((t0 < day1_cutoff_h).sum().item())

    obs_total = obs_per_feat.sum().item()
    denom_global = max(obs_total, 1.0)

    # Per-feature averages (each feature equally weighted in *_per_var)
    feat_denom = obs_per_feat.clamp(min=1.0)
    feat_mse_t   = sq_per_feat    / feat_denom
    feat_mae_t   = abs_per_feat   / feat_denom
    feat_huber_t = huber_per_feat / feat_denom

    feat_denom_d1   = obs_per_feat_d1.clamp(min=1.0)
    feat_mse_d1     = sq_per_feat_d1    / feat_denom_d1
    feat_huber_d1   = huber_per_feat_d1 / feat_denom_d1
    feat_mae_d1     = abs_per_feat_d1   / feat_denom_d1

    metrics: dict = {
        "mse":                  sq_per_feat.sum().item()    / denom_global,
        "mae":                  abs_per_feat.sum().item()   / denom_global,
        "huber":                huber_per_feat.sum().item() / denom_global,
        "rmse":                 (sq_per_feat.sum().item() / denom_global) ** 0.5,
        "mse_per_var":          feat_mse_t.mean().item(),
        "huber_per_var":        feat_huber_t.mean().item(),
        "mae_per_var":          feat_mae_t.mean().item(),
        # First-day subset metrics
        "mse_per_var_day1":     feat_mse_d1.mean().item(),
        "huber_per_var_day1":   feat_huber_d1.mean().item(),
        "mae_per_var_day1":     feat_mae_d1.mean().item(),
        # Window counts
        "n_windows_all":        n_windows_all,
        "n_windows_day1":       n_windows_d1,
        # Paper-1 un-normalised sums (no division by observation count)
        "mse_old":              sq_per_feat.sum().item(),
        "huber_old":            huber_per_feat.sum().item(),
        "mse_old_per_var":      sq_per_feat.mean().item(),
        "n_obs":                int(obs_total),
    }
    metrics["feat_mse"]   = {n: float(v) for n, v in zip(feature_names, feat_mse_t.tolist())}
    metrics["feat_huber"] = {n: float(v) for n, v in zip(feature_names, feat_huber_t.tolist())}
    metrics["horizon_mse"] = [
        float(s / max(o, 1.0))
        for s, o in zip(sq_per_horizon.tolist(), obs_per_horizon.tolist())
    ]
    return metrics


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def make_loader(h5_path: str, batch_size: int, shuffle: bool,
                num_workers: int) -> DataLoader:
    ds = ICUDataset(h5_path)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info("Seed: %d", args.seed)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir) / args.run_name
    from harmonize.common import archive_existing_run
    archive_existing_run(out_dir)   # never silently overwrite a previous run's model
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data (dense hourly grid or raw triplet set, auto-detected) ---
    train_ds, collate = make_dataset(args.train_h5)
    val_ds,   _       = make_dataset(args.val_h5)
    if collate is not None:
        log.info("Representation: triplet (raw set-function input)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, collate_fn=collate)
    # Overfit mode (--select-on train): a non-shuffled pass over the training set
    # used purely for best-checkpoint selection / early stopping on the train loss.
    train_eval_loader = (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True,
                   persistent_workers=args.num_workers > 0, collate_fn=collate)
        if args.select_on == "train" else None
    )
    # Effective steps per epoch: min(full loader length, max_steps_per_epoch).
    # max_steps_per_epoch=0 means unlimited (full pass over training set).
    _full_steps = len(train_loader)
    max_steps   = _full_steps if args.max_steps_per_epoch <= 0 else min(args.max_steps_per_epoch, _full_steps)
    log.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))
    log.info("F_in=%d  F_target=%d  F_static=%d  T_in=%d  T_out=%d",
             train_ds.n_features_in, train_ds.n_features_target,
             train_ds.n_features_static, train_ds.window_in, train_ds.window_out)
    log.info("Steps/epoch: %d of %d batches (batch_size=%d, max_steps_per_epoch=%s)",
             max_steps, _full_steps, args.batch_size,
             args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else "unlimited")

    target_idx = train_ds.target_indices_in_input()

    # Partial teacher forcing (dense_pteacher): medication targets are exogenous —
    # fed from ground truth in the AR rollout and excluded from the loss.
    exogenous_idx: list[int] = []
    if args.model == "dense_pteacher":
        from harmonize.common import FeatureCatalog
        _cat = FeatureCatalog.from_yaml(
            Path(args.train_h5).resolve().parents[2] / "configs" / "features.yaml")
        for i, f in enumerate(train_ds.features_target):
            try:
                if _cat.measurement_type(f) == "medication":
                    exogenous_idx.append(i)
            except KeyError:
                pass
        log.info("Partial teacher forcing: %d exogenous (medication) targets: %s",
                 len(exogenous_idx),
                 [train_ds.features_target[i] for i in exogenous_idx])

    # --- model ---
    # classical-baseline hyperparameters, forwarded only for the relevant model
    extra_model_kwargs: dict = {}
    if args.model == "arima":
        extra_model_kwargs["order"] = tuple(int(x) for x in args.arima_order.split(","))
    elif args.model == "ets":
        extra_model_kwargs.update(alpha=args.ets_alpha, beta=args.ets_beta, phi=args.ets_phi)
    elif args.model == "mamba":
        extra_model_kwargs.update(d_state=args.d_state, d_conv=args.d_conv, expand=args.expand)
    elif args.model == "patchtst":
        extra_model_kwargs.update(patch_len=args.patch_len, stride=args.stride)
    elif args.model == "dlinear":
        # DLinear IGNORES d_model (112 800 params at d128 and d512 alike): its capacity is the
        # per-channel window_in->window_out map. Its real hyperparameters are the decomposition
        # kernel and whether channels share weights — neither of which was sweepable before, which
        # is why the family has 26 runs at "2 configs" that differ only in irrelevant fields.
        extra_model_kwargs.update(kernel_size=args.kernel_size, individual=args.individual)
    if args.model in ("dense_autoregressive", "dense_pteacher"):
        extra_model_kwargs["detach_rollout"] = args.detach_rollout
    model = build_model(
        args.model,
        n_features_in=train_ds.n_features_in,
        n_features_target=train_ds.n_features_target,
        n_features_static=train_ds.n_features_static,
        window_in=train_ds.window_in,
        window_out=train_ds.window_out,
        target_idx_in_input=target_idx,
        d_model=args.d_model, d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers_enc=args.n_layers,
        n_layers_dec=(args.n_layers_dec if args.n_layers_dec is not None else args.n_layers),
        dropout=args.dropout,
        exogenous_idx=exogenous_idx,
        **extra_model_kwargs,
    ).to(device)
    log.info("Model: %s  params=%s", args.model, f"{model.n_params():,}")

    # --- optional fine-tuning init: load weights from a pretrained checkpoint ---
    if getattr(args, "init_from", None):
        ckpt = Path(args.init_from)
        if ckpt.is_dir():
            ckpt = ckpt / "best.pt"
        state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("Fine-tune init from %s (missing=%d, unexpected=%d) — same feature space "
                 "+ d_model required", ckpt, len(missing), len(unexpected))

    # --- optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_grad = len(trainable_params) > 0
    if has_grad:
        optim = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5,
            patience=args.patience_lr, min_lr=1e-7,
        )
    else:
        log.info("Model has 0 trainable parameters — skipping optimizer.")
        optim = sched = None

    # --- training loop ---
    best_val    = float("inf")   # measured on the metric named by --loss
    best_epoch  = -1
    no_improve  = 0
    train_log: list[dict] = []
    log.info("Loss: --loss=%s  (base=%s  per_variable=%s  unnormalised=%s)  "
             "huber_delta=%.2f",
             args.loss_name, args.loss_base,
             args.loss_per_var, args.loss_unnormalised, args.huber_delta)

    # Save run config + start time
    import datetime
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    cfg = vars(args).copy()
    cfg["features_target"] = train_ds.features_target
    cfg["features_in"]     = train_ds.features_in
    cfg["features_static"] = train_ds.features_static
    cfg["started_at"]      = started_at
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # Per-feature loss weighting: up-weight SOFA-relevant features (--sofa-weight) and/or
    # exclude exogenous medication channels (dense_pteacher) from the loss.
    feature_weights = None
    if abs(args.sofa_weight - 1.0) > 1e-9 or exogenous_idx:
        if not args.loss_per_var:
            log.warning("per-feature loss weighting (sofa-weight / exogenous meds) needs a "
                        "*_per_var loss (got --loss=%s); ignoring.", args.loss_name)
        else:
            from harmonize.common import FeatureCatalog
            root = Path(args.train_h5).resolve().parents[2]
            cat = FeatureCatalog.from_yaml(root / "configs" / "features.yaml")
            def _is_sofa(f: str) -> bool:
                try:
                    return cat.sofa_relevant(f)
                except KeyError:
                    return False
            w = [args.sofa_weight if _is_sofa(f) else 1.0
                 for f in train_ds.features_target]
            for i in exogenous_idx:
                w[i] = 0.0     # exogenous (medication) channels fed from GT → not scored
            feature_weights = torch.tensor(w, dtype=torch.float32, device=device)
            up = [f for f, v in zip(train_ds.features_target, w) if v > 1.0]
            if up:
                log.info("Clinically-weighted loss ×%.1f on %d SOFA-relevant targets: %s",
                         args.sofa_weight, len(up), up)
            if exogenous_idx:
                log.info("Excluded %d exogenous (medication) targets from the loss.",
                         len(exogenous_idx))

    for epoch in range(1, args.epochs + 1):
        p_tf = teacher_prob(args.schedule, epoch - 1, args.epochs)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            if n_steps >= max_steps:
                break
            x_tgt  = batch["x_target"].to(device, non_blocking=True)
            m_tgt  = batch["mask_target"].to(device, non_blocking=True)

            if optim is not None:
                optim.zero_grad(set_to_none=True)
            # AR-INPUT INJECTION (§27.11): the autoregressive decoder normally consumes its own
            # previous prediction. With --ar-pred-file it instead consumes the ENSEMBLE's value on
            # the channels the ensemble wins (NaN elsewhere -> own live output), while the LOSS
            # stays against the true target. That is what lets the decoder LEARN to use another
            # model's lines rather than merely being fed them at inference, where every injection
            # variant loses (Stage D: masked -0.09 %, full +1.33 %, live hybrid +3.37 %).
            ar_in, tf = x_tgt, p_tf
            if AR_PRED is not None:
                idx = batch.get("row_idx")
                if idx is None:
                    raise RuntimeError("--ar-pred-file needs row_idx from ICUDataset")
                ar_in = torch.from_numpy(AR_PRED[idx.numpy()]).to(device, non_blocking=True)
                # The decoder only READS x_target when teacher forcing fires. Under the default
                # --schedule student p_tf is exactly 0, so injection was a silent no-op: the run
                # logged "AR injection: ..." and trained an ordinary denseAR. Injection is not
                # scheduled sampling against the truth — it is the decoder's input by construction,
                # so force it on. NaN channels still fall back to the model's own live output.
                tf = 1.0
            pred = model_forward(model, batch, device,
                                 x_target=ar_in, teacher_forcing_ratio=tf)
            sw = None
            if args.event_weight != 1.0 and "dsofa2" in batch:
                d = batch["dsofa2"].to(device)                 # (B,) {0,1,NaN}
                sw = torch.where(d == 1.0, torch.full_like(d, args.event_weight),
                                 torch.ones_like(d))           # NaN/0 → weight 1
            loss = masked_loss(
                pred, x_tgt, m_tgt,
                base=args.loss_base,
                per_variable=args.loss_per_var,
                unnormalised=args.loss_unnormalised,
                huber_delta=args.huber_delta,
                loss_power=args.loss_power,
                feature_weights=feature_weights,
                sample_weights=sw,
            )
            if optim is not None:
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optim.step()
            running_loss += loss.item()
            n_steps += 1

        train_loss = running_loss / max(n_steps, 1)
        elapsed = time.time() - t0
        val = evaluate(model, val_loader, device,
                       train_ds.features_target, train_ds.window_out,
                       huber_delta=args.huber_delta)
        # Early-stopping / best-checkpoint selection key — match the training loss.
        # In overfit mode the selection metric is measured on the TRAIN split, so
        # training continues until the train loss stops improving (deliberate
        # over-fitting for the MIA setup); val is still computed for reporting.
        sel_key = args.loss_name
        # `power`/`power_per_var` is a training loss only (no val metric of that name);
        # select the best checkpoint on the comparable MSE metric instead.
        if sel_key == "power":
            sel_key = "mse"
        elif sel_key == "power_per_var":
            sel_key = "mse_per_var"
        if args.select_on == "train":
            sel_metrics = evaluate(model, train_eval_loader, device,
                                   train_ds.features_target, train_ds.window_out,
                                   huber_delta=args.huber_delta)
            sel_val = sel_metrics[sel_key]
        else:
            sel_val = val[sel_key]
        row = {
            "epoch": epoch, "p_teacher": round(p_tf, 4),
            "train_loss": train_loss,
            "val_mse": val["mse"], "val_mae": val["mae"],
            "val_huber": val["huber"], "val_rmse": val["rmse"],
            "val_mse_per_var":        val["mse_per_var"],
            "val_huber_per_var":      val["huber_per_var"],
            "val_mae_per_var":        val["mae_per_var"],
            "val_mse_per_var_day1":   val["mse_per_var_day1"],
            "val_huber_per_var_day1": val["huber_per_var_day1"],
            "val_mae_per_var_day1":   val["mae_per_var_day1"],
            f"sel_{args.select_on}({sel_key})": sel_val,
            "elapsed_s": round(elapsed, 1),
        }
        train_log.append(row)
        current_lr = optim.param_groups[0]["lr"] if optim is not None else 0.0
        log.info(
            "Ep %2d/%2d  p_tf=%.2f  train_%s=%.4f  | "
            "val mse_pv=%.4f  huber_pv=%.4f  mae_pv=%.4f  "
            "[day1: mse_pv=%.4f  huber_pv=%.4f  mae_pv=%.4f]  "
            "lr=%.2e  no_improve=%d  (%.1fs)",
            epoch, args.epochs, p_tf, args.loss_name, train_loss,
            val["mse_per_var"], val["huber_per_var"], val["mae_per_var"],
            val["mse_per_var_day1"], val["huber_per_var_day1"], val["mae_per_var_day1"],
            current_lr, no_improve, elapsed,
        )
        if sel_val < best_val or not (out_dir / "best.pt").exists():
            best_val = sel_val
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            log.info("   -> new best %s_%s=%.4f saved", args.select_on, sel_key, best_val)
        else:
            no_improve += 1
        if sched is not None:
            sched.step(sel_val)
        row["lr"] = current_lr
        row["no_improve"] = no_improve
        with open(out_dir / "train_log.jsonl", "w") as f:
            for r in train_log:
                f.write(json.dumps(r) + "\n")
        if no_improve >= args.patience_stop:
            log.info("Early stopping at epoch %d (no_improve=%d >= patience_stop=%d).",
                     epoch, no_improve, args.patience_stop)
            break

    # --- final eval on best checkpoint ---
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    final_val = evaluate(model, val_loader, device,
                         train_ds.features_target, train_ds.window_out)
    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    final = {
        "run_name":     args.run_name,
        "started_at":   started_at,
        "finished_at":  finished_at,
        "best_epoch":   best_epoch,
        "epochs_run":   epoch,
        "selection":    args.loss_name,
        "best_val_sel": best_val,
        "val":          final_val,
        "combo": {k: getattr(args, k) for k in (
            "model", "loss", "lr", "epochs", "seed",
            "patience_lr", "patience_stop", "schedule",
            "d_model", "d_ff", "n_heads", "n_layers", "n_layers_dec", "dropout",
            "patch_len", "stride", "detach_rollout",
            "max_steps_per_epoch", "train_h5",
        ) if hasattr(args, k)},
    }
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    log.info("Done. best val_%s=%.4f at epoch %d (started %s, finished %s)",
             args.loss_name, best_val, best_epoch, started_at, finished_at)
    from harmonize.common import append_run_manifest
    append_run_manifest(args.run_name, out_dir, started_at, finished_at,
                        model=args.model, dataset=Path(args.train_h5).stem.split("_seed")[0],
                        epochs_run=epoch, best_metric=f"val_{args.loss_name}", best_val=best_val)

    # Append one row to the central results CSV (created if absent).
    import csv, fcntl
    combo = final["combo"]

    # Parse dataset, features, stride, imputation from train_h5 filename.
    # Filename convention: {ds}_seed{N}_{split}__{features}_{stride}_{imputation}.h5
    import re as _re
    h5_stem = Path(combo.get("train_h5", "")).stem
    ds_match  = _re.match(r"([a-z_]+)_seed\d+", h5_stem)
    tag_match = _re.search(r"__(.+)$", h5_stem)
    _dataset = ds_match.group(1) if ds_match else ""
    if tag_match:
        tag_parts = tag_match.group(1).split("_")
        # tag = {features}_{stride}_{imputation} — last two are stride and imputation
        _imputation = tag_parts[-1]  if len(tag_parts) >= 1 else ""
        _stride     = tag_parts[-2]  if len(tag_parts) >= 2 else ""
        _features   = "_".join(tag_parts[:-2]) if len(tag_parts) >= 3 else ""
    else:
        _features = _stride = _imputation = ""

    # Duration in minutes
    try:
        _dt_start = datetime.datetime.fromisoformat(final["started_at"])
        _dt_end   = datetime.datetime.fromisoformat(final["finished_at"])
        _duration_min = round((_dt_end - _dt_start).total_seconds() / 60, 2)
    except Exception:
        _duration_min = ""

    row = {
        "run_name":           final["run_name"],
        "started_at":         final["started_at"],
        "finished_at":        final["finished_at"],
        "duration_min":       _duration_min,
        "dataset":            _dataset,
        "model":              combo.get("model", ""),
        "features":           _features,
        "stride":             _stride,
        "imputation":         _imputation,
        "train_h5":           combo.get("train_h5", ""),
        "seed":               combo.get("seed", ""),
        "d_model":            combo.get("d_model", ""),
        "n_heads":            combo.get("n_heads", ""),
        "n_layers":           combo.get("n_layers", ""),
        "dropout":            combo.get("dropout", ""),
        "lr":                 combo.get("lr", ""),
        "loss":               combo.get("loss", ""),
        "schedule":           combo.get("schedule", ""),
        "max_steps_per_epoch": combo.get("max_steps_per_epoch", ""),
        "epochs_run":         final["epochs_run"],
        "best_epoch":         final["best_epoch"],
        # Global metrics (sum / count_observed — single scalar across all vars)
        "val_mse":            final["val"].get("mse", ""),
        "val_mae":            final["val"].get("mae", ""),
        "val_rmse":           final["val"].get("rmse", ""),
        "val_n_obs":          final["val"].get("n_obs", ""),
        # Per-variable metrics (each variable equally weighted)
        "val_mse_per_var":        final["val"].get("mse_per_var", ""),
        "val_huber_per_var":      final["val"].get("huber_per_var", ""),
        "val_mae_per_var":        final["val"].get("mae_per_var", ""),
        # First-day subset
        "val_mse_per_var_day1":   final["val"].get("mse_per_var_day1", ""),
        "val_huber_per_var_day1": final["val"].get("huber_per_var_day1", ""),
        "val_mae_per_var_day1":   final["val"].get("mae_per_var_day1", ""),
        "val_n_windows_all":      final["val"].get("n_windows_all", ""),
        "val_n_windows_day1":     final["val"].get("n_windows_day1", ""),
    }

    results_csv = Path(args.train_h5).parent.parent.parent / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="") as fcsv:
        fcntl.flock(fcsv, fcntl.LOCK_EX)
        writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(fcsv, fcntl.LOCK_UN)

    # Log top-5 hardest features
    worst = sorted(final_val["feat_mse"].items(), key=lambda kv: -kv[1])[:5]
    log.info("Hardest features (val MSE):")
    for n, v in worst:
        log.info("  %-22s  %.4f", n, v)


AR_PRED = None       # train-split AR input (--ar-pred-file)
AR_PRED_VAL = None   # val-split AR input (--ar-pred-file-val): without it, early
                     # stopping scores the FREE-RUNNING model and the whole point of
                     # validating in the injected regime is lost


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4 forecast training")
    p.add_argument("--model",      default="dense_autoregressive",
                   choices=["zeroforecast", "persistence", "dlinear", "ets", "arima",
                            "dense_autoregressive", "dense_pteacher", "dense_dms",
                            "gru", "lstm", "mamba",
                            "informer", "patchtst", "autoformer", "strats", "strats_raw"])
    p.add_argument("--train-h5",   required=True, type=str)
    p.add_argument("--val-h5",     required=True, type=str)
    p.add_argument("--output-dir", default="checkpoints", type=str)
    p.add_argument("--run-name",   default="run", type=str)
    p.add_argument("--init-from",  default=None, type=str,
                   help="Fine-tune: initialise weights from a pretrained checkpoint dir or "
                        "best.pt before training (requires the same feature space + d_model). "
                        "Used for the single-hospital local-adaptation privacy experiment.")
    p.add_argument("--device",     default=None)
    p.add_argument("--seed",       type=int,   default=42,
                   help="Global RNG seed (torch, numpy, random).")

    p.add_argument("--epochs",        type=int,   default=200,
                   help="Hard upper limit on epochs (early stopping will usually trigger first).")
    p.add_argument("--max-steps-per-epoch", type=int, default=0,
                   help="Max gradient steps per epoch (0 = full dataset pass). "
                        "Batches are drawn without replacement from a fresh random permutation "
                        "each epoch (seeded by --seed). Useful to normalise epoch cost across "
                        "datasets of different sizes and for large sliding-window datasets.")
    p.add_argument("--patience-lr",   type=int,   default=3,
                   help="Epochs without improvement before halving the LR.")
    p.add_argument("--patience-stop", type=int,   default=6,
                   help="Epochs without improvement before early stopping.")
    p.add_argument("--select-on", default="val", choices=["val", "train"],
                   help="Split used for best-checkpoint selection, early stopping "
                        "and the LR scheduler. 'val' (default) = standard "
                        "generalisation training. 'train' = drive the model to "
                        "overfit: select/stop on the TRAINING loss (no gains left). "
                        "Use 'train' to build the over-fitted base model required "
                        "for the Paper-2 membership-inference setup.")
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--num-workers",   type=int,   default=2)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-5)
    p.add_argument("--grad-clip",     type=float, default=1.0)

    # Scheduled-sampling schedules. DEFAULT student (free-running from epoch 0): it is the
    # protocol of the paper-repro headline and the only one robust at large capacity — a
    # decaying schedule (schedule_di) leaves big AR models reliant on teacher forcing and
    # collapses the free-running val (cf. the d1024 incident). Set explicitly for the
    # scheduled-sampling ablation only.
    p.add_argument("--schedule", default="student",
                   choices=["none", "teacher", "student",
                            "schedule_di", "schedule_dd",
                            "schedule_ri", "schedule_rd"])

    # Loss function. One CLI flag covers all three axes (base × aggregation
    # × normalisation). Allowed values map onto the keys returned by
    # `evaluate()`, so the same string drives training, selection and reporting.
    p.add_argument("--loss", default="mse",
                   choices=[
                       "mse", "mae", "huber", "power",
                       "mse_per_var", "huber_per_var", "mae_per_var", "power_per_var",
                       "mse_old", "huber_old", "mse_old_per_var",
                   ],
                   help="Training + best-checkpoint selection metric. "
                        "All other metrics are still computed for reporting.")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="δ for Huber loss; data are z-scored, so 1.0 ≈ 1 σ.")
    p.add_argument("--loss-power", type=float, default=2.0,
                   help="Exponent p for --loss power (|e|^p); p in (2,3) up-weights large "
                        "errors more than MSE without the cubic's near-0 gradient collapse. "
                        "2.0 = MSE. (#6 deterioration-loss experiment.)")
    p.add_argument("--event-weight", type=float, default=1.0,
                   help="Per-window loss multiplier on deterioration windows (dsofa2==1), "
                        "so the rare ΔSOFA≥2 events dominate the gradient less weakly. "
                        "1.0 = off. Works with any --loss base. Record it in --run-name.")
    p.add_argument("--sofa-weight", type=float, default=1.0,
                   help="Clinically-weighted loss: multiply the per-feature loss of "
                        "SOFA-relevant target features (catalog sofa_relevant=true) by "
                        "this factor before the cross-feature mean. 1.0 = off; requires a "
                        "*_per_var loss. Record it in --run-name (e.g. _sw5); it is not a "
                        "results.csv column.")

    # Common model HPs (ignored by some models — pass via REGISTRY)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff",    type=int, default=-1,
                   help="Feed-forward dim; ≤0 auto-sets to 4×d_model.")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--detach-rollout", action="store_true",
                   help="dense_autoregressive/pteacher: feed student predictions back .detach()'d "
                        "→ no backprop through the AR rollout (the Staniek BP-through-predictions ablation).")
    p.add_argument("--ar-pred-file", default=None,
                   help="npy of ensemble predictions aligned to the TRAIN split's row order, used "
                        "as the AR decoder input (NaN = keep the model's own live output). The "
                        "loss is unchanged. NB the predictions must be OUT-OF-SAMPLE for the base "
                        "models, or the decoder learns to trust an input quality it will never "
                        "see at inference")
    p.add_argument("--ar-pred-file-val", default=None,
                   help="AR input for the VAL split. Required alongside --ar-pred-file: without "
                        "it early stopping scores the free-running model, not the injected one")
    p.add_argument("--kernel-size", type=int, default=5,
                   help="DLinear moving-average decomposition kernel (odd; must be <= window_in, "
                        "or the trend collapses to the window mean as in the Autoformer bug)")
    p.add_argument("--individual", action=argparse.BooleanOptionalAction, default=True,
                   help="DLinear: one linear map per channel (--individual) or shared "
                        "(--no-individual)")
    p.add_argument("--patch-len", type=int, default=8,
                   help="PatchTST patch length (window split). Capped at window_in.")
    p.add_argument("--stride", type=int, default=4,
                   help="PatchTST patch stride. n_patch = (window_in - patch_len)//stride + 1.")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-layers-dec", type=int, default=None,
                   help="decoder depth; if unset, = --n-layers (so 2/2). Set 1 for the Paper-1 "
                        "2-encoder/1-decoder architecture (Staniek). Selection is on val (--select-on).")
    p.add_argument("--dropout", type=float, default=0.1)
    # Mamba (selective state-space) hyperparameters (ignored by other models)
    p.add_argument("--d-state", type=int, default=16,
                   help="Mamba SSM latent size N (mamba only)")
    p.add_argument("--d-conv",  type=int, default=4,
                   help="Mamba causal-conv kernel width (mamba only)")
    p.add_argument("--expand",  type=int, default=2,
                   help="Mamba inner-width expansion factor (mamba only)")
    # classical-baseline hyperparameters (ignored by the neural models)
    p.add_argument("--arima-order", default="1,1,0",
                   help="ARIMA (p,d,q) order, comma-separated e.g. '2,1,0' (arima model only)")
    p.add_argument("--ets-alpha", type=float, default=0.5, help="ETS level smoothing (ets only)")
    p.add_argument("--ets-beta",  type=float, default=0.1, help="ETS trend smoothing (ets only)")
    p.add_argument("--ets-phi",   type=float, default=0.9, help="ETS trend damping φ (ets only)")
    args = p.parse_args()

    # logging is configured inside train(), which runs AFTER this block — without configuring it
    # here the "AR injection" confirmations below go to an unconfigured logger at WARNING level
    # and vanish, which is indistinguishable from the injection not loading at all.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # AR-input injection (§27.11): load the ensemble predictions used as the decoder's
    # autoregressive input. NaN entries mean "keep the model's own live output" on that channel.
    # The loss is untouched — this changes what the decoder CONSUMES, not what it is scored on.
    global AR_PRED, AR_PRED_VAL
    if getattr(args, "ar_pred_file_val", None):
        AR_PRED_VAL = np.load(args.ar_pred_file_val).astype(np.float32)
        log.info("AR injection (val): %s %s", args.ar_pred_file_val, AR_PRED_VAL.shape)
    if getattr(args, "ar_pred_file", None):
        AR_PRED = np.load(args.ar_pred_file).astype(np.float32)
        n_nan = int(np.isnan(AR_PRED[0]).all(axis=0).sum())
        log.info("AR injection: %s %s — %d/%d channels left to the decoder",
                 args.ar_pred_file, AR_PRED.shape, n_nan, AR_PRED.shape[2])

    if args.d_ff <= 0:
        args.d_ff = 4 * args.d_model

    # Decompose --loss into the three axes used internally.
    args.loss_name        = args.loss
    args.loss_unnormalised = args.loss.endswith("_old") or "_old_" in args.loss
    args.loss_per_var      = args.loss.endswith("_per_var")
    head = args.loss
    for suffix in ("_old_per_var", "_per_var", "_old"):
        if head.endswith(suffix):
            head = head[: -len(suffix)]
            break
    args.loss_base = head  # mse | mae | huber

    train(args)


if __name__ == "__main__":
    main()
