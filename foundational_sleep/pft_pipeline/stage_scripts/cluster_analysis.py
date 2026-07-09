# Databricks notebook source
# =============================================================================
# PFTSleep Cluster Analysis
# =============================================================================
# Joins HDBSCAN cluster labels to per-window metadata and to subject-level
# demographics / CVD outcomes. Produces:
#   STEP 1  Verify what is on disk (labels + metadata)
#   STEP 2  Window-level table  : [nsrrid, time_idx, cluster, probability]
#   STEP 3  Subject-level table : cluster proportions per subject (1219 rows)
#   STEP 4  Associate cluster proportions with AHI / HypoxicBurden / CVD
#
# Cluster labels are row-aligned 1:1 to the metadata arrays by window index.
# =============================================================================

# COMMAND ----------

# DBTITLE 1,Install
# MAGIC %pip install statsmodels scikit-learn -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Paths
from pathlib import Path
import numpy as np
import pandas as pd

TAG       = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
OUT_DIR   = Path(globals().get("OUT_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
CACHE_DIR = Path(globals().get("CACHE_DIR", "/dbfs/tmp/new_pftsleep_cache"))
DEMO_CSV  = globals().get("DEMO_CSV", "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel.csv")

# COMMAND ----------

# DBTITLE 1,STEP 1 - verify labels + metadata are on disk
# Cluster labels (saved after HDBSCAN)
labels_path = OUT_DIR / f"cluster_labels__{TAG}.npy"
assert labels_path.exists(), f"Missing cluster labels: {labels_path}"
cluster_labels = np.load(labels_path)
print(f"cluster_labels: {cluster_labels.shape}")

# Probabilities (optional but useful)
prob_path = OUT_DIR / f"cluster_probabilities__{TAG}.npy"
cluster_probabilities = np.load(prob_path) if prob_path.exists() else None
print(f"cluster_probabilities: "
      f"{None if cluster_probabilities is None else cluster_probabilities.shape}")

# Metadata arrays. These live in the extraction cache. The nsrrid file holds
# REAL subject ids (e.g. 200002); nsrrid_intidx is the sequential index - we
# want the real ids for joining to demographics.
def _load_meta(name):
    p = CACHE_DIR / f"{name}__{TAG}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing metadata array: {p}\n"
            f"This is produced by the extraction step's final concatenation. "
            f"If it is gone (ephemeral /dbfs/tmp wiped), you must re-run the "
            f"concatenation or restore from the cache backup."
        )
    return np.load(p, allow_pickle=True)

night_id      = _load_meta("night_id")
time_idx      = _load_meta("time_idx")
window_nsrrid = _load_meta("nsrrid").astype(np.int64)

# Alignment check - all arrays must be the same length as the labels.
n = len(cluster_labels)
assert len(night_id) == n,      f"night_id {len(night_id)} != labels {n}"
assert len(time_idx) == n,      f"time_idx {len(time_idx)} != labels {n}"
assert len(window_nsrrid) == n, f"nsrrid {len(window_nsrrid)} != labels {n}"
print(f"All metadata aligned to {n:,} windows.")
print(f"Unique subjects: {len(np.unique(window_nsrrid)):,}")
print(f"First 5 nsrrids: {window_nsrrid[:5]}")   # expect real ids 200002...

# COMMAND ----------

# DBTITLE 1,STEP 2 - window-level table (joinable to events by nsrrid + time)
win = pd.DataFrame({
    "nsrrid":      window_nsrrid,
    "time_idx":    time_idx,        # seconds from lights-off (window start)
    "cluster":     cluster_labels,  # -1 = noise
    "probability": cluster_probabilities if cluster_probabilities is not None
                   else np.nan,
})
print(f"Window table: {win.shape}")
print(win.head(10).to_string())

# Save it so you can join to the events Excel later (by nsrrid + time window).
win_path = OUT_DIR / f"window_clusters__{TAG}.parquet"
win.to_parquet(win_path, index=False)
print(f"Saved window table: {win_path}")

# Cluster size summary
sizes = win["cluster"].value_counts().sort_values(ascending=False)
print("\nTop clusters by window count:")
print(sizes.head(15).to_string())
print(f"\nNoise (-1): {(win['cluster']==-1).mean()*100:.1f}% of windows")

# COMMAND ----------

# DBTITLE 1,STEP 3 - subject-level cluster proportions (1219 x n_clusters)
# For each subject, what fraction of their windows fell in each cluster?
# This turns per-window clusters into per-subject features for the CVD analysis.
# Noise (-1) is kept as its own column so proportions sum to 1 per subject.
clusters_sorted = sorted(win["cluster"].unique())

# crosstab: rows = subjects, cols = clusters, values = window counts
ct = pd.crosstab(win["nsrrid"], win["cluster"])
# normalize each subject's row to proportions (fraction of that subject's windows)
prop = ct.div(ct.sum(axis=1), axis=0)
prop.columns = [f"clust_{c}_prop" for c in prop.columns]
prop = prop.reset_index()   # nsrrid becomes a column

print(f"Subject-level proportions: {prop.shape}  (subjects x cluster-proportion cols)")
print(prop.head().to_string())

subj_path = OUT_DIR / f"subject_cluster_proportions__{TAG}.parquet"
prop.to_parquet(subj_path, index=False)
print(f"Saved subject proportions: {subj_path}")

# COMMAND ----------

# DBTITLE 1,STEP 4 - join to demographics + associate with outcomes
# DBTITLE 1,STEP 4 - join to demographics + associate with outcomes

demo = pd.read_csv(DEMO_CSV)

demo["nsrrid"] = demo["nsrrid"].astype(np.int64)
 
merged = prop.merge(demo, on="nsrrid", how="inner")

print(f"Merged: {merged.shape[0]} subjects matched to demographics")
 
# ---- 4a. Correlate each cluster proportion with continuous severity measures

from scipy.stats import spearmanr

from statsmodels.stats.multitest import multipletests
 
prop_cols = [c for c in merged.columns if c.endswith("_prop")]

targets = ["ahi_a0h3a", "HypoxicBurden", "pctlt90", "MinSat", "bmi_s1", "age_s1"]

targets = [t for t in targets if t in merged.columns]
 
print("\nSpearman correlation: cluster proportion vs severity measures")

print(f"{'cluster':>16} | " + " | ".join(f"{t:>14}" for t in targets))

cor_records = []

for pc in prop_cols:

    cors = []

    for t in targets:

        sub = merged[[pc, t]].dropna()

        if len(sub) > 30 and sub[pc].std() > 0 and sub[t].std() > 0:

            r, p = spearmanr(sub[pc], sub[t]); n_t = len(sub)

        else:

            r, p, n_t = np.nan, np.nan, len(sub)

        cors.append((r, p))

        cor_records.append({"cluster_prop": pc, "target": t,

                            "spearman_rho": r, "p_value": p, "n": n_t})

    line = f"{pc:>16} | " + " | ".join(

        f"{r:>+7.3f}{'*' if (pp is not None and pp < 0.05) else ' '}     "

        for r, pp in cors)

    print(line)
 
cor_df = pd.DataFrame(cor_records)

finite = cor_df["p_value"].notna()

cor_df["p_fdr_bh"] = np.nan

if finite.sum() > 0:

    cor_df.loc[finite, "p_fdr_bh"] = multipletests(

        cor_df.loc[finite, "p_value"], method="fdr_bh")[1]

cor_path = OUT_DIR / f"cluster_severity_correlations__{TAG}.parquet"

cor_df.to_parquet(cor_path, index=False)

cor_df.to_csv(cor_path.with_suffix(".csv"), index=False)

print(f"\nSaved correlations ({cor_df.shape[0]} tests): {cor_path}")
 
# ---- 4b. Logistic association with incident CVD (adjusting for AHI)

import statsmodels.api as sm
 
cvd_outcome = "inci_cvd_all_01"

cvd_records = []

if cvd_outcome in merged.columns:

    print(f"\nLogistic: {cvd_outcome} ~ cluster_prop + ahi_a0h3a + age + bmi")

    print(f"{'cluster':>16} | {'OR(prop)':>9} | {'p(prop)':>9} | {'n':>6}")

    base = ["ahi_a0h3a", "age_s1", "bmi_s1"]

    for pc in prop_cols:

        cols = [pc] + base + [cvd_outcome]

        sub = merged[cols].apply(pd.to_numeric, errors="coerce").dropna()

        if len(sub) < 100 or sub[pc].std() == 0:

            cvd_records.append({"cluster_prop": pc, "odds_ratio": np.nan,

                                "coef": np.nan, "p_value": np.nan,

                                "ci_low": np.nan, "ci_high": np.nan,

                                "n": len(sub), "note": "skipped (n<100 or constant)"})

            continue

        X = sm.add_constant(sub[[pc] + base]); y = sub[cvd_outcome]

        try:

            res = sm.Logit(y, X).fit(disp=0)

            ci = res.conf_int().loc[pc]

            rec = {"cluster_prop": pc,

                   "odds_ratio": float(np.exp(res.params[pc])),

                   "coef": float(res.params[pc]),

                   "p_value": float(res.pvalues[pc]),

                   "ci_low": float(np.exp(ci[0])),

                   "ci_high": float(np.exp(ci[1])),

                   "n": int(len(sub)), "note": ""}

            cvd_records.append(rec)

            print(f"{pc:>16} | {rec['odds_ratio']:>9.3f} | "

                  f"{rec['p_value']:>9.4f} | {rec['n']:>6}")

        except Exception as e:

            cvd_records.append({"cluster_prop": pc, "odds_ratio": np.nan,

                                "coef": np.nan, "p_value": np.nan,

                                "ci_low": np.nan, "ci_high": np.nan,

                                "n": len(sub), "note": f"failed: {str(e)[:40]}"})

            print(f"{pc:>16} | model failed: {str(e)[:30]}")
 
    cvd_df = pd.DataFrame(cvd_records)

    finite = cvd_df["p_value"].notna()

    cvd_df["p_fdr_bh"] = np.nan

    if finite.sum() > 0:

        cvd_df.loc[finite, "p_fdr_bh"] = multipletests(

            cvd_df.loc[finite, "p_value"], method="fdr_bh")[1]

    cvd_path = OUT_DIR / f"cluster_cvd_logistic__{TAG}.parquet"

    cvd_df.to_parquet(cvd_path, index=False)

    cvd_df.to_csv(cvd_path.with_suffix(".csv"), index=False)

    print(f"\nSaved CVD logistic results ({cvd_df.shape[0]} models): {cvd_path}")

else:

    print(f"\n{cvd_outcome} not found in demographics columns.")
 
print("\np_fdr_bh columns hold Benjamini-Hochberg FDR-corrected p-values. "

      "Use those, not raw p_value, when judging significance across clusters.")
 
