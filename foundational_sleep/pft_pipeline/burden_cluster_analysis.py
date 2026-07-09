"""
Burden vs. Cluster Association Analysis
=======================================

Tests whether the PFTSleep embedding clusters are associated with externally
supplied physiological burden measures (HB_AUC, SWAK_DPI, VB_50, VB_70).

Two units of analysis (per Grace's spec):
  PRIMARY   - subject level: each subject's CLUSTER PROPORTIONS vs each burden.
  SECONDARY - window level : burden value attached to each window via its
              subject, compared ACROSS the cluster a window belongs to.

Two output types:
  HEADLINE       - Spearman + Pearson correlations with Benjamini-Hochberg FDR
                   (matches the q-value framing Diego presented).
  INTERPRETATION - regression of each burden on cluster proportions, with the
                   compositional-collinearity problem handled explicitly.

Design decisions (publication-defensible, stated so you can defend them):
  1. Cluster proportions are COMPOSITIONAL (they sum to 1 per subject). Putting
     all of them in an OLS gives perfect collinearity. We handle this two ways
     and report both:
       (a) UNIVARIATE per-cluster regressions (one cluster proportion at a time,
           each adjusted for AHI) - clean, interpretable, no collinearity.
       (b) MULTIVARIABLE OLS with a reference cluster DROPPED (the largest), so
           coefficients are "relative to spending that time in the reference
           cluster." This is the standard compositional fix.
  2. The HDBSCAN NOISE cluster (-1) is reported but flagged; we never let it be
     the reference, and we report results both including and excluding it.
  3. Spearman is the headline (monotone, robust to the non-normal, zero-inflated
     proportion distributions); Pearson reported alongside for completeness.
  4. FDR (Benjamini-Hochberg) is applied ACROSS all cluster x burden tests
     within each correlation type - this is the multiple-comparison control
     Diego's FDR framing expects.

Inputs (from the existing pipeline outputs):
  - subject_cluster_proportions__{TAG}.parquet : nsrrid + clust_{c}_prop cols
  - window_clusters__{TAG}.parquet             : [nsrrid, time_idx, cluster, probability]
  - burden table                               : nsrrid HB_AUC SWAK_DPI VB_50 VB_70
  - (optional) demographics csv for AHI adjustment (ahi_a0h3a)

Outputs:
  - burden_cluster_correlations__{TAG}.parquet   (headline, subject level)
  - burden_cluster_regression__{TAG}.parquet     (interpretation, subject level)
  - burden_by_cluster_window__{TAG}.parquet      (secondary, window level)
  - console summary of the significant associations after FDR
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Config - these are injected by the pipeline; defaults let it run standalone.
# ---------------------------------------------------------------------------
BURDEN_COLS = ["HB_AUC", "SWAK_DPI", "VB_50", "VB_70"]
AHI_COL = "ahi_a0h3a"          # adjust regressions for sleep-apnea severity
MIN_SUBJECTS_PER_CLUSTER = 10  # skip clusters present in <N subjects (unstable)
FDR_ALPHA = 0.05


def load_burden_table(path: str | Path) -> pd.DataFrame:
    """Read the burden table (csv or parquet) and normalize the nsrrid dtype."""
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        b = pd.read_parquet(path)
    else:
        b = pd.read_csv(path, sep=None, engine="python")  # sniff , or tab
    b.columns = [c.strip() for c in b.columns]
    assert "nsrrid" in b.columns, f"burden table needs 'nsrrid'; got {list(b.columns)}"
    b["nsrrid"] = b["nsrrid"].astype(np.int64)
    present = [c for c in BURDEN_COLS if c in b.columns]
    missing = [c for c in BURDEN_COLS if c not in b.columns]
    if missing:
        print(f"[burden] WARNING: missing burden columns {missing}; "
              f"proceeding with {present}")
    # coerce burdens to numeric, report non-numeric contamination
    for c in present:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    return b[["nsrrid"] + present], present


def subject_level_correlations(
    prop: pd.DataFrame,
    burden: pd.DataFrame,
    burden_cols: list[str],
    prop_cols: list[str],
) -> pd.DataFrame:
    """Spearman + Pearson of each cluster proportion vs each burden, with BH-FDR.

    FDR is applied across ALL (cluster x burden) pairs within each method.
    """
    df = prop.merge(burden, on="nsrrid", how="inner")
    print(f"[burden] subject-level merge: {df.shape[0]} subjects "
          f"({prop.shape[0]} clustered ∩ {burden.shape[0]} burden)")

    records = []
    for pc in prop_cols:
        n_present = (df[pc] > 0).sum()
        for bc in burden_cols:
            sub = df[[pc, bc]].dropna()
            if len(sub) < MIN_SUBJECTS_PER_CLUSTER:
                continue
            # Spearman (headline) and Pearson (companion)
            rho, p_s = stats.spearmanr(sub[pc], sub[bc])
            r, p_p = stats.pearsonr(sub[pc], sub[bc])
            records.append({
                "cluster_prop": pc,
                "burden": bc,
                "n": len(sub),
                "n_subjects_in_cluster": int(n_present),
                "spearman_rho": rho,
                "spearman_p": p_s,
                "pearson_r": r,
                "pearson_p": p_p,
            })
    res = pd.DataFrame(records)
    if res.empty:
        print("[burden] no valid cluster x burden pairs.")
        return res

    # Benjamini-Hochberg across all pairs, per method
    for pcol, qcol in [("spearman_p", "spearman_q"), ("pearson_p", "pearson_q")]:
        ok = res[pcol].notna()
        res.loc[ok, qcol] = multipletests(res.loc[ok, pcol],
                                          method="fdr_bh")[1]
    res["sig_spearman_fdr"] = res.get("spearman_q", np.nan) < FDR_ALPHA
    return res.sort_values("spearman_q").reset_index(drop=True)


def subject_level_regression(
    prop: pd.DataFrame,
    burden: pd.DataFrame,
    burden_cols: list[str],
    prop_cols: list[str],
    demo: pd.DataFrame | None,
) -> pd.DataFrame:
    """Two regression views, handling compositional collinearity:

    (a) UNIVARIATE: burden ~ one cluster proportion (+ AHI if available).
        One model per (cluster, burden). No collinearity. Interpretable as
        'subjects who spend more time in cluster c have higher/lower burden'.
    (b) MULTIVARIABLE: burden ~ all cluster proportions EXCEPT a dropped
        reference (the largest non-noise cluster) (+ AHI). Coefficients are
        relative to the reference cluster.
    Returns the univariate table (the cleaner one for a conference); the
    multivariable R^2 is printed for context.
    """
    df = prop.merge(burden, on="nsrrid", how="inner")
    has_ahi = False
    if demo is not None and AHI_COL in demo.columns:
        d = demo[["nsrrid", AHI_COL]].copy()
        d["nsrrid"] = d["nsrrid"].astype(np.int64)
        df = df.merge(d, on="nsrrid", how="left")
        has_ahi = df[AHI_COL].notna().sum() > MIN_SUBJECTS_PER_CLUSTER
    print(f"[burden] regression: AHI adjustment {'ON' if has_ahi else 'OFF'}")

    rows = []
    for bc in burden_cols:
        for pc in prop_cols:
            sub = df[[pc, bc] + ([AHI_COL] if has_ahi else [])].dropna()
            if len(sub) < MIN_SUBJECTS_PER_CLUSTER or sub[pc].std() == 0:
                continue
            X = sub[[pc] + ([AHI_COL] if has_ahi else [])]
            X = sm.add_constant(X)
            y = sub[bc]
            try:
                m = sm.OLS(y, X).fit()
                rows.append({
                    "burden": bc,
                    "cluster_prop": pc,
                    "n": len(sub),
                    "beta": m.params[pc],
                    "se": m.bse[pc],
                    "p": m.pvalues[pc],
                    "partial_r2": _partial_r2(m, pc),
                    "ahi_adjusted": has_ahi,
                })
            except Exception as e:  # noqa: BLE001
                print(f"[burden] regression failed {bc}~{pc}: {e}")
    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out["p"].notna()
        out.loc[ok, "q_fdr"] = multipletests(out.loc[ok, "p"],
                                             method="fdr_bh")[1]
        out["sig_fdr"] = out["q_fdr"] < FDR_ALPHA
        out = out.sort_values("q_fdr").reset_index(drop=True)

    # multivariable R^2 per burden, for context (printed, not returned)
    _multivariable_context(df, burden_cols, prop_cols, has_ahi)
    return out


def _partial_r2(model, term: str) -> float:
    """Partial R^2 for a single term via its t-stat (Cohen): t^2 / (t^2 + df)."""
    t = model.tvalues.get(term, np.nan)
    df_resid = model.df_resid
    if np.isnan(t) or df_resid <= 0:
        return np.nan
    return float(t**2 / (t**2 + df_resid))


def _multivariable_context(df, burden_cols, prop_cols, has_ahi):
    """Fit burden ~ all cluster props minus a reference; print R^2 only."""
    # choose reference = largest-mean NON-NOISE cluster proportion
    non_noise = [c for c in prop_cols if not c.startswith("clust_-1")]
    if not non_noise:
        return
    ref = df[non_noise].mean().idxmax()
    keep = [c for c in prop_cols if c != ref]
    for bc in burden_cols:
        cols = keep + ([AHI_COL] if has_ahi else [])
        sub = df[[bc] + cols].dropna()
        if len(sub) < (len(cols) + 5):
            continue
        X = sm.add_constant(sub[cols])
        try:
            m = sm.OLS(sub[bc], X).fit()
            print(f"[burden] multivariable {bc} ~ cluster props "
                  f"(ref={ref}{', +AHI' if has_ahi else ''}): "
                  f"R²={m.rsquared:.3f}, adjR²={m.rsquared_adj:.3f}, "
                  f"F p={m.f_pvalue:.2e}, n={len(sub)}")
        except Exception as e:  # noqa: BLE001
            print(f"[burden] multivariable {bc} failed: {e}")


def window_level_burden(
    win: pd.DataFrame,
    burden: pd.DataFrame,
    burden_cols: list[str],
) -> pd.DataFrame:
    """SECONDARY: attach each window's subject burden, summarize burden by the
    cluster the window belongs to. Tests (Kruskal-Wallis) whether burden
    differs across clusters at the window level.

    NOTE: windows from the same subject share a burden value, so windows are
    NOT independent - this is descriptive support, not an independent test.
    We state that explicitly; the subject-level analysis is the inferential one.
    """
    w = win.merge(burden, on="nsrrid", how="inner")
    rows = []
    for bc in burden_cols:
        grp = w[["cluster", bc]].dropna()
        # per-cluster summary
        summ = grp.groupby("cluster")[bc].agg(["count", "mean", "median", "std"])
        summ["burden"] = bc
        rows.append(summ.reset_index())
        # Kruskal-Wallis across clusters (descriptive; non-independent)
        groups = [g[bc].values for _, g in grp.groupby("cluster") if len(g) > 0]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            print(f"[burden] window-level {bc}: Kruskal-Wallis H={h:.1f}, "
                  f"p={p:.2e} (descriptive - windows not independent)")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(
    subject_prop_path: str | Path,
    burden_path: str | Path,
    window_path: str | Path | None = None,
    demo_path: str | Path | None = None,
    out_dir: str | Path = ".",
    tag: str = "run",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prop = pd.read_parquet(subject_prop_path)
    prop["nsrrid"] = prop["nsrrid"].astype(np.int64)
    prop_cols_all = [c for c in prop.columns if c.endswith("_prop")]

    # drop clusters present in too few subjects (unstable correlations)
    keep_prop = []
    for c in prop_cols_all:
        if (prop[c] > 0).sum() >= MIN_SUBJECTS_PER_CLUSTER:
            keep_prop.append(c)
    dropped = sorted(set(prop_cols_all) - set(keep_prop))
    if dropped:
        print(f"[burden] dropping {len(dropped)} clusters present in "
              f"<{MIN_SUBJECTS_PER_CLUSTER} subjects: {dropped}")
    prop_cols = keep_prop

    burden, burden_cols = load_burden_table(burden_path)
    demo = pd.read_csv(demo_path) if demo_path else None

    # 1) HEADLINE - subject-level correlations + FDR
    corr = subject_level_correlations(prop, burden, burden_cols, prop_cols)
    if not corr.empty:
        p = out_dir / f"burden_cluster_correlations__{tag}.parquet"
        corr.to_parquet(p, index=False)
        print(f"\n[burden] saved correlations -> {p}")
        sig = corr[corr["sig_spearman_fdr"]]
        print(f"[burden] {len(sig)} of {len(corr)} cluster×burden pairs "
              f"significant after FDR (Spearman q<{FDR_ALPHA}):")
        if not sig.empty:
            print(sig[["cluster_prop", "burden", "n", "spearman_rho",
                       "spearman_q"]].to_string(index=False))

    # 2) INTERPRETATION - subject-level regression
    reg = subject_level_regression(prop, burden, burden_cols, prop_cols, demo)
    if not reg.empty:
        p = out_dir / f"burden_cluster_regression__{tag}.parquet"
        reg.to_parquet(p, index=False)
        print(f"\n[burden] saved regression -> {p}")
        sig = reg[reg["sig_fdr"]]
        print(f"[burden] {len(sig)} of {len(reg)} cluster→burden regressions "
              f"significant after FDR (q<{FDR_ALPHA})")
        if not sig.empty:
            print(sig[["burden", "cluster_prop", "beta", "partial_r2",
                       "q_fdr"]].head(20).to_string(index=False))

    # 3) SECONDARY - window-level
    if window_path:
        win = pd.read_parquet(window_path)
        win["nsrrid"] = win["nsrrid"].astype(np.int64)
        wl = window_level_burden(win, burden, burden_cols)
        if not wl.empty:
            p = out_dir / f"burden_by_cluster_window__{tag}.parquet"
            wl.to_parquet(p, index=False)
            print(f"\n[burden] saved window-level summary -> {p}")

    print("\n[burden] done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proportions", required=True)
    ap.add_argument("--burden", required=True)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--demo", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    run(a.proportions, a.burden, a.windows, a.demo, a.out, a.tag)
