"""
10_scripts/paper1_analyses.py

Consolidated Paper-1 robustness / representation / cross-cohort analyses (doc §R6, §12.1).
One committed entry point for every §R6 statement (replaces the one-off scratch scripts).

  --analysis counterfactual   §R6.1  input +3σ perturbation → forecast reaction (absorption/localisation/couplings)
  --analysis severity         §R6.2  per-subscore SOFA per dataset (+ total-ex-CNS) [+ --eicu-hospital, --over-time]
  --analysis similarity       §R6.3  encoder-rep t-SNE colored by loss + loss-structure (k-NN) correlation
  --analysis prototypes       §R6.3  k-means elbow/silhouette over encoder reps + prototype loss/SOFA characterisation
  --analysis missingness      §R6.4  does the act of measuring forecast deterioration? (obs-count + per-variable AUROC)
  --analysis embedding        §12.1  embedding density + per-hour norm vs #observed (dense despite sparse input)
  --analysis hospital_cov            eICU per-hospital variable coverage (for the by-hospital split)

Usage:  python 10_scripts/paper1_analyses.py --analysis counterfactual [--dataset mimic_iii] [--checkpoint <dir>]
GPU analyses (counterfactual/similarity/prototypes/embedding) need the forecaster; run via run_train_gpu/sbatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = "all_4h_fwd"
DATASETS = ["mimic_iii", "mimic_iv", "eicu", "hirid", "umm"]
WINNER = {"mimic_iii": "hpsf_mimic_iii_095a5e97", "mimic_iv": "hpsf_mimic_iv_88fb076a",
          "eicu": "hpsf_eicu_0fc6127a", "hirid": "hpsf_hirid_f4e3a6c7", "umm": "hpsf_umm_6a82f5b0"}
SOFA_NAMES = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal", "total"]
NONCNS = [0, 1, 2, 3, 5]


# ---------------------------------------------------------------------------
def _load_model(ds, device):
    import torch
    from harmonize.icu_dataset import ICUDataset
    from harmonize.eval_forecast import load_model
    d = ICUDataset(str(ROOT / f"data/dataset/{ds}_seed42_val__{TAG}.h5"))
    m, cfg = load_model(ROOT / "checkpoints" / WINNER[ds], d, device)
    m.eval()
    return d, m


def _encode_and_loss(ds, cap=4000):
    """Pooled encoder reps + per-window forecast MSE + per-window SOFA total (GPU)."""
    import torch
    from torch.utils.data import DataLoader
    d, m = _load_model(ds, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    R, L, SF, n = [], [], [], 0
    for b in DataLoader(d, batch_size=256, shuffle=False):
        dev = next(m.parameters()).device
        x, mi, st = b["x_in"].to(dev), b["mask_in"].to(dev), b["static"].to(dev)
        xt, mt = b["x_target"].numpy(), b["mask_target"].numpy().astype(bool)
        sf = (np.nanmean(b["sofa_scores"][:, :, 6].numpy(), 1) if "sofa_scores" in b
              else np.full(len(x), np.nan))
        with torch.no_grad():
            h = m.encode(x, mi, st).mean(1).cpu().numpy()
            pred = m(x, mi, st).cpu().numpy()
        se = ((pred - np.nan_to_num(xt)) ** 2 * mt).sum((1, 2)) / mt.sum((1, 2)).clip(1)
        R.append(h); L.append(se); SF.append(sf); n += len(h)
        if n >= cap:
            break
    return np.concatenate(R)[:cap], np.concatenate(L)[:cap], np.concatenate(SF)[:cap]


# ---------------------------------------------------------------------------
def counterfactual(ds="mimic_iii", sigma=3.0):
    import torch
    from torch.utils.data import DataLoader
    d, m = _load_model(ds, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dev = next(m.parameters()).device
    ft, fi = list(d.features_target), list(d.features_in)
    clin = [c for c in ["bilirubin_total", "creatinine", "platelets", "lactate", "gcs_motor",
                        "map", "potassium", "sodium", "bun", "wbc", "sao2", "pao2", "fio2",
                        "hemoglobin", "heart_rate"] if c in ft and c in fi]
    b = next(iter(DataLoader(d, batch_size=512, shuffle=False)))
    x, mi, st = b["x_in"].to(dev), b["mask_in"].to(dev), b["static"].to(dev)
    mt = b["mask_target"].numpy().astype(bool)
    with torch.no_grad():
        base = m(x, mi, st).cpu().numpy()
    infl = lambda p: (np.abs(p - base) * mt).sum((0, 1)) / mt.sum((0, 1)).clip(1)
    rows = []
    for v in clin:
        iv = fi.index(v); xp = x.clone(); obs = mi[:, :, iv].bool()
        xp[:, :, iv] = torch.where(obs, xp[:, :, iv] + sigma, xp[:, :, iv])
        with torch.no_grad():
            row = infl(m(xp, mi, st).cpu().numpy())
        s, tot = float(row[ft.index(v)]), float(row.sum())
        oth = [(ft[i], row[i]) for i in np.argsort(-row) if ft[i] != v][:4]
        rows.append({"perturbed": v, "self": round(s, 3), "self/Nσ": round(s / sigma, 2),
                     "spill": round(tot - s, 3), "self_frac": round(s / max(tot, 1e-9), 2),
                     "top_other": ";".join(f"{n}:{x:.2f}" for n, x in oth)})
    df = pd.DataFrame(rows).sort_values("self", ascending=False)
    df.to_csv(ROOT / f"docs/tables/counterfactual_influence_{ds}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nmean absorption self/Nσ={df['self/Nσ'].mean():.2f} | mean localisation self_frac={df['self_frac'].mean():.2f}")


def severity(eicu_hospital=False, over_time=False):
    def stay_sub(ds):
        S, I = [], []
        for sp in ("train", "val", "test"):
            p = ROOT / f"data/dataset/{ds}_seed42_{sp}__{TAG}.h5"
            if not p.exists():
                continue
            with h5py.File(p) as f:
                if "sofa_scores" not in f:
                    continue
                S.append(np.nanmean(f["sofa_scores"][:], 1)); I.append(f["stay_id"][:])
        df = pd.DataFrame(np.concatenate(S), columns=SOFA_NAMES); df["stay_id"] = np.concatenate(I)
        return df.groupby("stay_id").mean()
    rows = []
    cache = {}
    for ds in DATASETS:
        g = stay_sub(ds); cache[ds] = g
        rows.append({"dataset": ds, **{n: round(g[n].mean(), 2) for n in SOFA_NAMES},
                     "total_ex_cns": round(g[[SOFA_NAMES[i] for i in NONCNS]].sum(1).mean(), 2)})
    print("=== per-subscore SOFA + total-ex-CNS ===")
    print(pd.DataFrame(rows).to_string(index=False))
    if eicu_hospital:
        coh = pd.read_parquet(ROOT / "data/long/eicu_cohort.parquet", columns=["stay_id", "hospitalid"]).dropna()
        s2h = dict(zip(coh.stay_id.astype(int), coh.hospitalid.astype(int)))
        e = cache["eicu"][[SOFA_NAMES[i] for i in NONCNS]].sum(1).rename("sofa").reset_index()
        e["h"] = e.stay_id.map(lambda s: s2h.get(int(s)))
        perh = e.dropna(subset=["h"]).groupby("h").agg(n=("sofa", "size"), sofa=("sofa", "mean"))
        perh = perh[perh.n >= 50]
        print(f"\n=== eICU per-hospital ({len(perh)} hosps): SOFA-ex-CNS mean={perh.sofa.mean():.2f} "
              f"range {perh.sofa.min():.2f}-{perh.sofa.max():.2f} std={perh.sofa.std():.2f} ===")


def _per_stay_dead(ds, coh):
    """Verified ICU-mortality per stay (§9.2 / explore_datasets._get_mortality_rate):
    death within [intime, outtime+1h]; eICU uses unitdischargestatus=Expired."""
    cols = set(coh.columns)
    if ds == "eicu" and "unitdischargestatus" in cols:
        return coh["unitdischargestatus"].astype(str).str.strip().str.lower().eq("expired")
    inc = next((c for c in ("intime", "icu_start") if c in cols), None)
    outc = next((c for c in ("outtime", "icu_end") if c in cols), None)
    dc = next((c for c in ("deathtime", "death_time", "DOD", "dod") if c in cols), None)
    if inc and outc and dc:
        ti, to, td = (pd.to_datetime(coh[c], errors="coerce") for c in (inc, outc, dc))
        return td.notna() & (td >= ti) & (td <= to + pd.Timedelta(hours=1))
    dsc = next((c for c in ("discharge_status", "dischargestatus") if c in cols), None)
    return coh[dsc].astype(str).str.strip().str.lower().isin(["expired", "dead"]) if dsc else None


def mortality():
    """§R6.2(iv): verify ICU mortality vs doc §9.2, then SOFA-stratify (gated on the match)."""
    # Documented ICU mortality per database (§9.2), used as a self-consistency gate: a database
    # whose computed rate disagrees by >1 pp is EXCLUDED from the stratification below rather than
    # reported from a number we cannot reproduce.
    # hirid corrected 6.1 -> 8.3 on 2026-10-26 (§37.58 D34): the stale 6.1 failed the gate by
    # 2.15 pp and silently dropped HiRID from this analysis. Two independent routes agree on ~8.3:
    # `discharge_status in {dead, expired}` gives 8.25 % (1 307/15 841) and the Stage-1
    # `mortality_icu` static gives 8.35 % (on its 15 655 non-null stays).
    DOC = {"mimic_iii": 7.4, "mimic_iv": 7.5, "eicu": 5.2, "hirid": 8.3, "umm": 19.4}
    print("verify ICU mortality vs doc §9.2 (gate stratification):")
    ok, dm = {}, {}
    for ds in DOC:
        coh = pd.read_parquet(ROOT / f"data/long/{ds}_cohort.parquet")
        d = _per_stay_dead(ds, coh)
        idc = next((c for c in ("stay_id", "patientid", "patient_id") if c in coh.columns), None)
        if d is None or idc is None:
            continue
        rate = 100 * d.mean(); ok[ds] = abs(rate - DOC[ds]) < 1.0
        dm[ds] = pd.Series(d.values, index=coh[idc].astype("int64", errors="ignore"))
        print(f"  {ds:10s} {rate:5.1f}% vs doc {DOC[ds]:.1f}%  {'MATCH' if ok[ds] else 'MISMATCH (excluded)'}")
    print("\nSOFA-stratified ICU mortality (matched only):")
    for ds in DOC:
        if not ok.get(ds):
            continue
        S, I = [], []
        for sp in ("train", "val", "test"):
            p = ROOT / f"data/dataset/{ds}_seed42_{sp}__{TAG}.h5"
            if not p.exists():
                continue
            with h5py.File(p) as f:
                S.append(np.nanmean(f["sofa_scores"][:], 1)[:, 6]); I.append(f["stay_id"][:])
        sf = pd.DataFrame({"stay_id": np.concatenate(I), "s": np.concatenate(S)}).groupby("stay_id").s.mean()
        df = pd.DataFrame({"sofa": sf}); df["m"] = [dm[ds].get(int(i), np.nan) for i in df.index]; df = df.dropna()
        df["b"] = (df.sofa // 3 * 3).clip(0, 15).astype(int); g = df.groupby("b").m.agg(["mean", "size"])
        print(f"  {ds:9s} overall={100*df.m.mean():4.1f}% | " +
              " ".join(f"S{b}:{100*g.loc[b,'mean']:.1f}%" for b in g.index if g.loc[b, "size"] >= 15))


def sapsii_mortality():
    """Validate SAPS-II (incl. admission_type + chronic_disease, Le Gall 1993) against observed ICU
    mortality. SAPS-II is an *admission* score → computed on the first-24h window per stay (t0=0).
    Reports discrimination (AUROC of the score → death) base (physiology+age) vs full (+adm+chronic),
    and calibration (Le Gall logistic predicted-mortality vs observed rate)."""
    from sklearn.metrics import roc_auc_score
    from harmonize.scores.sapsii import sapsii_batch, mortality_probability, load_sapsii_static_maps
    print("=== SAPS-II → Le Gall mortality (first-24h window/stay; AUROC = score→death) ===")
    for ds in ["mimic_iii", "mimic_iv", "eicu", "hirid", "umm"]:
        coh = pd.read_parquet(ROOT / f"data/long/{ds}_cohort.parquet")
        dead = _per_stay_dead(ds, coh)
        idc = next((c for c in ("stay_id", "patientid", "patient_id") if c in coh.columns), None)
        if dead is None or idc is None:
            print(f"  {ds}: no mortality field"); continue
        dmap = pd.Series(dead.values, index=coh[idc].astype("int64", errors="ignore"))
        norm = pd.read_parquet(ROOT / f"data/splits/{ds}_norm_stats.parquet").set_index("variable")
        am, cm = load_sapsii_static_maps(ds)
        parts = []
        for sp in ("train", "val", "test"):
            p = ROOT / f"data/dataset/{ds}_seed42_{sp}__{TAG}.h5"
            if not p.exists():
                continue
            with h5py.File(p) as f:
                sel = f["t0_hour"][:] == 0          # one first-24h window per stay
                fi = [x.decode() if isinstance(x, bytes) else x for x in f.attrs["features_in"]]
                fs = [x.decode() if isinstance(x, bytes) else x for x in f.attrs["features_static"]]
                xi, mk = f["x_in"][:][sel], f["mask_in"][:][sel].astype(bool)
                st, sid = f["static"][:][sel], f["stay_id"][:][sel]
            mean = np.array([norm.loc[v, "train_mean"] if v in norm.index else 0. for v in fi], np.float32)
            std = np.array([norm.loc[v, "train_std"] if v in norm.index else 1. for v in fi], np.float32)
            xp = np.where(mk, xi * std + mean, np.nan)            # denorm; unobserved → NaN
            age = st[:, fs.index("age")] if "age" in fs else None
            adm = np.array([am.get(int(s), 0.) for s in sid], np.float32)
            chr_ = np.array([cm.get(int(s), 0.) for s in sid], np.float32)
            parts.append(pd.DataFrame({
                "base": sapsii_batch(xp, fi, age, None, None),    # physiology + age
                "full": sapsii_batch(xp, fi, age, adm, chr_),     # + adm + chronic
                "dead": [dmap.get(int(s), np.nan) for s in sid]}))
        df = pd.concat(parts).dropna()
        if df.dead.nunique() < 2:
            print(f"  {ds}: no mortality variation"); continue
        pf = np.array([mortality_probability(s) for s in df.full])
        print(f"  {ds:9s} n={len(df):6d} obs-mort={100*df.dead.mean():4.1f}% | "
              f"AUROC base(physio+age)={roc_auc_score(df.dead, df.base):.3f} "
              f"full(+adm+chronic)={roc_auc_score(df.dead, df.full):.3f} | "
              f"mean SAPS={df.full.mean():4.1f} → Le Gall pred-mort {100*pf.mean():4.1f}%")


def missingness():
    from sklearn.metrics import roc_auc_score
    print("=== informative missingness: does measurement forecast deterioration? ===")
    for ds in ["mimic_iv", "eicu", "umm"]:
        p = ROOT / f"data/dataset/{ds}_seed42_val__{TAG}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            mk = f["mask_in"][:]; feats = [x.decode() if isinstance(x, bytes) else x for x in f.attrs["features_in"]]
            y = f["dsofa2"][:] if "dsofa2" in f else None
        if y is None:
            continue
        v = ~np.isnan(y); yy = y[v].astype(int)
        if len(np.unique(yy)) < 2:
            continue
        oc = mk[v].sum((1, 2)).astype(float); ov = mk[v].any(1).astype(float)
        det, sta = ov[yy == 1].mean(0), ov[yy == 0].mean(0)
        print(f"  {ds}: AUROC(total #meas → ΔSOFA2)={roc_auc_score(yy, oc):.3f}; "
              f"top vars measured more when deteriorating: "
              f"{', '.join(f'{feats[i]}({(det-sta)[i]:+.2f})' for i in np.argsort(-(det-sta))[:4])}")


def embedding(ds="umm"):
    import torch
    from torch.utils.data import DataLoader
    d, m = _load_model(ds, torch.device("cpu"))
    b = next(iter(DataLoader(d, batch_size=1024, shuffle=False)))
    with torch.no_grad():
        emb = m.embed_inputs(b["x_in"], b["mask_in"], b["static"])
    obs = b["mask_in"].float().sum(-1); norm = emb.norm(dim=-1)
    print(f"=== embedding density ({ds}) ===")
    print(f"  input frac masked={(~b['mask_in'].bool()).float().mean():.3f} | "
          f"frac all-zero emb vectors={(emb.abs().sum(-1) < 1e-8).float().mean():.4f} | NaN={torch.isnan(emb).any().item()}")
    print(f"  corr(emb_norm, #obs)={np.corrcoef(norm.flatten().numpy(), obs.flatten().numpy())[0, 1]:.3f} "
          f"(→ info in DIRECTION, not magnitude)")


def similarity(ds="mimic_iii"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    R, L, _ = _encode_and_loss(ds)
    nn = NearestNeighbors(n_neighbors=21).fit(R); _, idx = nn.kneighbors(R)
    nbr = L[idx[:, 1:]].mean(1); r = np.corrcoef(L, nbr)[0, 1]
    print(f"=== similarity cloud ({ds}): loss-structure corr={r:.3f} (shuffled "
          f"{np.corrcoef(L, np.random.permutation(nbr))[0, 1]:.3f}) ===")
    emb = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(R)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(L, 0, np.percentile(L, 98)), s=7, cmap="viridis")
    plt.colorbar(sc, label="forecast MSE"); ax.set_title(f"{ds}: encoder t-SNE by loss (corr={r:.2f})")
    ax.set_xticks([]); ax.set_yticks([]); fig.savefig(ROOT / "docs/figures/fig_similarity_cloud.png", dpi=120, bbox_inches="tight")
    print("  wrote docs/figures/fig_similarity_cloud.png")


def prototypes(ds="mimic_iii"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    R, L, SF = _encode_and_loss(ds, cap=5000)
    Rn = (R - R.mean(0)) / (R.std(0) + 1e-8)
    ks = list(range(2, 16)); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=5, random_state=0).fit(Rn)
        inertia.append(km.inertia_); sil.append(silhouette_score(Rn, km.labels_, sample_size=2000, random_state=0))
    kb = ks[int(np.argmax(sil))]
    km = KMeans(kb, n_init=10, random_state=0).fit(Rn)
    proto = pd.DataFrame({"c": km.labels_, "loss": L, "sofa": SF}).groupby("c").agg(
        n=("loss", "size"), loss=("loss", "mean"), sofa=("sofa", "mean")).round(3)
    print(f"=== prototypes ({ds}): best-silhouette k={kb} (max sil={max(sil):.3f} → continuum, weak clustering) ===")
    print(proto.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set(xlabel="k", ylabel="inertia", title="Elbow")
    ax[1].plot(ks, sil, "o-", color="C1"); ax[1].set(xlabel="k", ylabel="silhouette", title=f"Silhouette (best k={kb})")
    fig.tight_layout(); fig.savefig(ROOT / "docs/figures/fig_prototype_elbow.png", dpi=120)
    print("  wrote docs/figures/fig_prototype_elbow.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True,
                    choices=["counterfactual", "severity", "mortality", "sapsii_mortality", "missingness", "embedding", "similarity", "prototypes"])
    ap.add_argument("--dataset", default="mimic_iii")
    ap.add_argument("--eicu-hospital", action="store_true")
    a = ap.parse_args()
    if a.analysis == "counterfactual":
        counterfactual(a.dataset)
    elif a.analysis == "severity":
        severity(eicu_hospital=a.eicu_hospital)
    elif a.analysis == "mortality":
        mortality()
    elif a.analysis == "sapsii_mortality":
        sapsii_mortality()
    elif a.analysis == "missingness":
        missingness()
    elif a.analysis == "embedding":
        embedding(a.dataset)
    elif a.analysis == "similarity":
        similarity(a.dataset)
    elif a.analysis == "prototypes":
        prototypes(a.dataset)


if __name__ == "__main__":
    main()
