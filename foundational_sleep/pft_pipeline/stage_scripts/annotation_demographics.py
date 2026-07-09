# Databricks notebook source
# =============================================================================
# Per-cluster annotation composition + demographic statistics
# =============================================================================
# TABLE 1: for each cluster, % of its windows overlapping each annotation type
#          (sleep stages AND respiratory events; a window may overlap several).
# TABLE 2: for each cluster, demographic stats (mean/median/mode/std/min/max)
#          computed over the DISTINCT SUBJECTS appearing in that cluster.
#
# Demographics are per-subject, so TABLE 2 is per-subject (each subject counted
# once per cluster), not per-window - window-weighting would bias toward long
# recordings.
# =============================================================================

# COMMAND ----------

from pathlib import Path
import numpy as np
import pandas as pd

TAG        = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
OUT_DIR    = Path(globals().get("OUT_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
EVENTS_DIR = Path(globals().get("EVENTS_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep/cohort_1219_events"))
MASTER_CSV = "/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep/master_study_duration.csv"
DEMO_CSV   = globals().get("DEMO_CSV", "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel.csv")
WINDOW_SEC = 6

# Demographic columns to summarize (numeric + a couple categorical for mode).
DEMO_NUM = ["age_s1", "bmi_s1", "ahi_a0h3a", "HypoxicBurden", "pctlt90",
            "MinSat", "hdl_s1", "chol_s1", "trig_s1", "slptime"]
DEMO_CAT = ["gender", "race_s1", "smokecat_s1", "OSA_Sev_Cat"]

# COMMAND ----------

# DBTITLE 1,Load window-cluster table + lights-off offsets
win = pd.read_parquet(OUT_DIR / f"window_clusters__{TAG}.parquet")
print(f"Window table: {win.shape}")

master = pd.read_csv(MASTER_CSV)
master.columns = [c.strip() for c in master.columns]
lights_off = dict(zip(master["nsrrid"].astype(np.int64),
                      master["Lights Off"].astype(float)))

# COMMAND ----------

# DBTITLE 1,Annotate each window with EVERY overlapping annotation type
def load_events(nsrrid: int):
    matches = (list(EVENTS_DIR.glob(f"*{nsrrid}.YMTEvents.csv")) +
               list(EVENTS_DIR.glob(f"*{nsrrid}_YMTEvents.csv")))
    if not matches:
        return None
    df = pd.read_csv(matches[0]); df.columns = [c.strip() for c in df.columns]
    return df

# Annotations we ignore entirely (position, EEG markers, lights).
IGNORE = {"Supine", "Upright", "Alpha Intrusion Index", "Eye Movement",
          "Lights Off", "Lights On", "Bad Oxygen", "Missed REM Period"}

# Build a long table: one row per (window_row_index, annotation_name) overlap.
# We accumulate per subject to keep memory bounded.
overlap_rows = []   # (win_global_index, annotation_name)
win = win.reset_index(drop=True)
win["gidx"] = np.arange(len(win))

subjects = win["nsrrid"].unique()
for si, nsrrid in enumerate(subjects):
    ev = load_events(int(nsrrid))
    if ev is None:
        continue
    sub = win[win["nsrrid"] == nsrrid]
    off = lights_off.get(int(nsrrid), 0.0)
    w_start = sub["time_idx"].to_numpy() + off
    w_end   = w_start + WINDOW_SEC
    g       = sub["gidx"].to_numpy()

    for _, r in ev.iterrows():
        name = str(r["Name"])
        if name in IGNORE:
            continue
        e0 = float(r["Start"]); e1 = e0 + float(r["Duration"])
        m = (w_start < e1) & (w_end > e0)     # window-event overlap
        if m.any():
            for gi in g[m]:
                overlap_rows.append((gi, name))
    if (si + 1) % 100 == 0:
        print(f"  annotated {si+1}/{len(subjects)} subjects")

overlaps = pd.DataFrame(overlap_rows, columns=["gidx", "annotation"])
overlaps = overlaps.merge(win[["gidx", "cluster"]], on="gidx", how="left")
print(f"Total window-annotation overlaps: {len(overlaps):,}")

# COMMAND ----------

# DBTITLE 1,TABLE 1 - per-cluster annotation composition (% of windows)
# For each cluster: of its windows, what % overlap >=1 annotation of each type.
windows_per_cluster = win.groupby("cluster").size().rename("n_windows")

# count distinct windows (gidx) per (cluster, annotation) - distinct so a
# window with two Desats in it counts once for "Desat".
distinct = overlaps.drop_duplicates(["gidx", "cluster", "annotation"])
counts = distinct.groupby(["cluster", "annotation"]).size().rename("n").reset_index()

# pivot to cluster x annotation, then divide by windows in that cluster
pivot = counts.pivot(index="cluster", columns="annotation", values="n").fillna(0)
pct = pivot.div(windows_per_cluster, axis=0) * 100.0
pct = pct.round(1)
pct.insert(0, "n_windows", windows_per_cluster)

print("TABLE 1 - annotation composition per cluster "
      "(% of cluster's windows overlapping each annotation type):")
print(pct.to_string())

t1_path = OUT_DIR / f"cluster_annotation_composition__{TAG}.parquet"
pct.to_parquet(t1_path)
pct.to_csv(t1_path.with_suffix(".csv"))
print(f"\nSaved TABLE 1: {t1_path}")

# COMMAND ----------

# DBTITLE 1,TABLE 2 - per-cluster demographic statistics (per distinct subject)
demo = pd.read_csv(DEMO_CSV)
demo["nsrrid"] = demo["nsrrid"].astype(np.int64)

# distinct (cluster, subject) pairs - which subjects appear in each cluster
cluster_subj = win[["cluster", "nsrrid"]].drop_duplicates()
cs = cluster_subj.merge(demo, on="nsrrid", how="left")

num_cols = [c for c in DEMO_NUM if c in cs.columns]
cat_cols = [c for c in DEMO_CAT if c in cs.columns]

stat_rows = []
for clust, grp in cs.groupby("cluster"):
    rec = {"cluster": clust, "n_subjects": grp["nsrrid"].nunique()}
    for col in num_cols:
        v = pd.to_numeric(grp[col], errors="coerce").dropna()
        if len(v) == 0:
            continue
        rec[f"{col}_mean"]   = v.mean()
        rec[f"{col}_median"] = v.median()
        rec[f"{col}_std"]    = v.std()
        rec[f"{col}_min"]    = v.min()
        rec[f"{col}_max"]    = v.max()
        m = v.mode()
        rec[f"{col}_mode"]   = m.iloc[0] if len(m) else np.nan
    for col in cat_cols:
        m = grp[col].mode()
        rec[f"{col}_mode"] = m.iloc[0] if len(m) else np.nan
    stat_rows.append(rec)

demo_stats = pd.DataFrame(stat_rows).sort_values("cluster")
# round numeric stat columns for readability
num_stat_cols = [c for c in demo_stats.columns
                 if c not in ("cluster", "n_subjects")
                 and demo_stats[c].dtype != object]
demo_stats[num_stat_cols] = demo_stats[num_stat_cols].round(2)

print("TABLE 2 - demographic statistics per cluster (over distinct subjects):")
print(demo_stats.to_string(index=False))

t2_path = OUT_DIR / f"cluster_demographic_stats__{TAG}.parquet"
demo_stats.to_parquet(t2_path, index=False)
demo_stats.to_csv(t2_path.with_suffix(".csv"), index=False)
print(f"\nSaved TABLE 2: {t2_path}")

# COMMAND ----------

display(pct)

# COMMAND ----------

display(demo_stats)

# COMMAND ----------



# COMMAND ----------


