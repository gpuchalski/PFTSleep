# Databricks notebook source
# =============================================================================
# Map clustered windows -> scored sleep (stages + respiratory events)
# =============================================================================
# For every clustered window, determine:
#   (a) the sleep STAGE active at that window's time
#   (b) whether a respiratory EVENT (apnea/hypopnea/desat) overlaps it
# then cross-tabulate clusters against stages and event-overlap.
#
# TIME ALIGNMENT (critical):
#   window time_idx = seconds from LIGHTS OFF (zarrs were trimmed to the
#                     Lights Off -> Lights On scored interval)
#   events Start    = seconds from RECORDING START
#   => shift events by each subject's Lights Off so both share one clock.
# =============================================================================

# COMMAND ----------

# DBTITLE 1,Paths + config
from pathlib import Path
import numpy as np
import pandas as pd

TAG        = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
OUT_DIR    = Path(globals().get("OUT_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
EVENTS_DIR = Path(globals().get("EVENTS_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep/cohort_1219_events"))  # <-- folder of per-subject *_YMTEvents.csv
MASTER_CSV = "/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep/master_study_duration.csv"

WINDOW_SEC = 6   # each window spans 6 seconds

# Which Name values are sleep STAGES (everything else that's respiratory is an event).
STAGE_NAMES = {"Wake", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "REM"}
# Respiratory event names contain these tokens. 'Desat' plus the slash-coded
# apnea/hypopnea types (A, H/D, HO/D, C/D, O/D, RERA/A/D, ...).
def is_resp_event(name: str) -> bool:
    n = str(name)
    if n in STAGE_NAMES:
        return False
    # respiratory markers: apnea (A), hypopnea (H), desat (D/Desat),
    # obstructive/central (O/C), RERA. Exclude pure position/EEG annotations.
    tokens = {"Desat", "A", "H", "D", "O", "C", "RERA", "HO"}
    parts = set(n.replace("/", " ").split())
    return bool(parts & tokens) and n not in {
        "Supine", "Upright", "Alpha Intrusion Index", "Eye Movement",
        "Lights Off", "Lights On", "Bad Oxygen", "Missed REM Period"
    }

# COMMAND ----------

# DBTITLE 1,Load window-cluster table + per-subject Lights Off
win = pd.read_parquet(OUT_DIR / f"window_clusters__{TAG}.parquet")
print(f"Window table: {win.shape}")

master = pd.read_csv(MASTER_CSV)
# normalize column names to find Lights Off + nsrrid
master.columns = [c.strip() for c in master.columns]
lights_off = dict(zip(master["nsrrid"].astype(np.int64),
                      master["Lights Off"].astype(float)))
print(f"Lights Off loaded for {len(lights_off)} subjects")

# COMMAND ----------

# DBTITLE 1,Build stage + event lookup per subject, annotate each window
def load_events(nsrrid: int) -> pd.DataFrame:
    # SHHS event filenames look like shhs1-200002_YMTEvents.csv
    matches = list(EVENTS_DIR.glob(f"*{nsrrid}.YMTEvents.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    df.columns = [c.strip() for c in df.columns]
    return df

def annotate_subject(sub_win: pd.DataFrame, nsrrid: int) -> pd.DataFrame:
    ev = load_events(nsrrid)
    out = sub_win.copy()
    out["stage"] = "Unknown"
    out["resp_event"] = False
    if ev is None:
        return out

    off = lights_off.get(nsrrid, 0.0)
    # window absolute time (from recording start) = time_idx + lights_off
    w_start = out["time_idx"].to_numpy() + off
    w_end   = w_start + WINDOW_SEC

    stages = ev[ev["Name"].isin(STAGE_NAMES)]
    resp   = ev[ev["Name"].apply(is_resp_event)]

    # ---- assign sleep stage: the stage interval containing the window start
    stage_name = np.array(["Unknown"] * len(out), dtype=object)
    for _, r in stages.iterrows():
        s0 = float(r["Start"]); s1 = s0 + float(r["Duration"])
        m = (w_start >= s0) & (w_start < s1)
        stage_name[m] = r["Name"]
    out["stage"] = stage_name

    # ---- flag respiratory-event overlap: any resp event intersecting the window
    has_event = np.zeros(len(out), dtype=bool)
    for _, r in resp.iterrows():
        e0 = float(r["Start"]); e1 = e0 + float(r["Duration"])
        # overlap if window_start < event_end AND window_end > event_start
        m = (w_start < e1) & (w_end > e0)
        has_event |= m
    out["resp_event"] = has_event
    return out

# Process subject by subject (keeps memory bounded; events are small per file).
annotated = []
subjects = win["nsrrid"].unique()
for i, nsrrid in enumerate(subjects):
    sub = win[win["nsrrid"] == nsrrid]
    annotated.append(annotate_subject(sub, int(nsrrid)))
    if (i + 1) % 100 == 0:
        print(f"  annotated {i+1}/{len(subjects)} subjects")

win_annot = pd.concat(annotated, ignore_index=True)
print(f"Annotated windows: {win_annot.shape}")
print(win_annot.head(10).to_string())

win_annot.to_parquet(OUT_DIR / f"window_clusters_scored__{TAG}.parquet", index=False)
print("Saved annotated window table.")

# COMMAND ----------

# DBTITLE 1,Cluster vs sleep stage
# Row-normalized: for each cluster, what % of its windows are each stage?
stage_ct = pd.crosstab(win_annot["cluster"], win_annot["stage"], normalize="index")
print("Cluster composition by sleep stage (row %):")
print((stage_ct * 100).round(1).to_string())

# COMMAND ----------

display(win_annot)

# COMMAND ----------

import os

CLUSTER_OUT_DIR = OUT_DIR / "cluster_specific_win_annot"
os.makedirs(CLUSTER_OUT_DIR, exist_ok=True)

for cluster_id in win_annot["cluster"].unique():
    df_cluster = win_annot[win_annot["cluster"] == cluster_id]
    df_cluster.to_parquet(CLUSTER_OUT_DIR / f"cluster_{cluster_id}_win_annot.parquet", index=False)


for cluster_id in win_annot["cluster"].unique():
    df_cluster = win_annot[win_annot["cluster"] == cluster_id]
    display(df_cluster)

for cluster_id in win_annot["cluster"].unique():
    df_cluster = win_annot[win_annot["cluster"] == cluster_id]
    seqs = []
    for nsrrid, sub in df_cluster.groupby("nsrrid"):
        t = np.sort(sub["time_idx"].to_numpy())
        if len(t) == 0:
            continue
        # Find breaks in the sequence
        diff = np.diff(t)
        breaks = np.where(diff != WINDOW_SEC)[0]
        # Start indices of sequences
        starts = np.insert(breaks + 1, 0, 0)
        # End indices of sequences
        ends = np.append(breaks, len(t) - 1)
        for s, e in zip(starts, ends):
            starting_idx = t[s]
            ending_idx = t[e] + WINDOW_SEC
            duration = ending_idx - starting_idx
            seqs.append({
                "nsrrid": nsrrid,
                "starting_idx": starting_idx,
                "ending_idx": ending_idx,
                "cluster": cluster_id,
                "duration": duration
            })
    df_seq = pd.DataFrame(seqs)
    print(f"\n=== Cluster {cluster_id} window sequences ===")
    display(df_seq)
    df_seq.to_parquet(f"/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100/cluster_specific_win_annot/cluster_{cluster_id}_win_sequence.parquet", index=False)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Cluster vs respiratory-event overlap
# For each cluster, what fraction of its windows overlap a respiratory event?
evt_rate = win_annot.groupby("cluster")["resp_event"].mean().sort_values(ascending=False)
overall  = win_annot["resp_event"].mean()
print(f"Overall respiratory-event window rate: {overall*100:.1f}%\n")
print("Cluster respiratory-event overlap rate (sorted):")
for c, rate in evt_rate.items():
    flag = "  <-- enriched" if rate > 2 * overall else ""
    print(f"  cluster {c:>4}: {rate*100:>5.1f}%{flag}")

print("\nClusters far above the overall rate are candidate 'respiratory-event' "
      "states - the structure your hypothesis predicts the embedding encodes.")

# COMMAND ----------



# COMMAND ----------

from pathlib import Path
EVENTS_DIR = Path(globals().get("EVENTS_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep/cohort_1219_events"))  # your path
 
# Does the directory exist and what's in it?
print("Dir exists:", EVENTS_DIR.exists())
all_csv = list(EVENTS_DIR.glob("*.csv"))
print(f"CSV files in dir: {len(all_csv)}")
print("First few:", [p.name for p in all_csv[:5]])
 
# Does the glob for subject 200002 match?
matches = list(EVENTS_DIR.glob("*200002.YMTEvents.csv"))
print("Match for 200002:", matches)

# COMMAND ----------


