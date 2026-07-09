# Databricks notebook source
# =============================================================================
# Collect scored-sleep event files for the 1219-subject cohort
# =============================================================================
# Searches the four source folders for *{nsrrid}.YMTEvents.csv (and the
# underscore variant), copies the matches for the 1219 cohort nsrrids into one
# destination folder, and reports exactly what was found / missing / duplicated.
#
# SAFETY: source folders are read only here; we COPY (never move), and skip
# files that already exist in the destination so nothing is overwritten.
# =============================================================================

# COMMAND ----------

from pathlib import Path
import shutil
import re
import numpy as np

TAG       = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
CACHE_DIR = Path(globals().get("CACHE_DIR", "/dbfs/tmp/new_pftsleep_cache"))

PARENT = Path("/Volumes/kumc_sleep/sleep_studies/shhs_data/scored_sleep")
SOURCE_DIRS = [
    # TODO: replace the next three with your actual folder names
    PARENT / "SHHS-1 204001-205000/",
    PARENT / "SHHS-1 205001-205804/",
    PARENT / "SHHS-1 EVs 00-201000/",
    PARENT / "SHHS-1 202001-203000/",
    PARENT / "SHHS-1 201001-202000/",
    PARENT / "SHHS-CSV files/SHHS-1 203001-204000/SHHS-1 203001-204000/",
]
DEST_DIR = PARENT / "cohort_1219_events"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Load the 1219 cohort nsrrids
# Use the extraction metadata as the source of truth for which subjects are in
# the cohort (the same nsrrids the clustering used).
nsrrid_arr = np.load(CACHE_DIR / f"nsrrid__{TAG}.npy", allow_pickle=True)
cohort = sorted({int(x) for x in np.unique(nsrrid_arr)})
print(f"Cohort size: {len(cohort)} nsrrids")
print(f"Range: {cohort[0]} .. {cohort[-1]}")

# COMMAND ----------

# DBTITLE 1,Index every event file across the four folders by nsrrid
# Map nsrrid -> list of source paths. Filenames look like
# shhs1-204001.YMTEvents.csv  (dot)  or  shhs1-200002_YMTEvents.csv (underscore)
id_re = re.compile(r"shhs1-(\d+)[._]YMTEvents\.csv$", re.IGNORECASE)

found = {}   # nsrrid -> [paths]
for d in SOURCE_DIRS:
    if not d.exists():
        print(f"WARNING: source folder does not exist: {d}")
        continue
    n_here = 0
    for p in d.glob("*.csv"):
        m = id_re.search(p.name)
        if m:
            nid = int(m.group(1))
            found.setdefault(nid, []).append(p)
            n_here += 1
    print(f"{d.name}: indexed {n_here} event files")

print(f"\nTotal distinct nsrrids found across folders: {len(found)}")

# COMMAND ----------

# DBTITLE 1,Copy the cohort's files into the destination
copied, missing, dupes, already = [], [], [], []

for nid in cohort:
    paths = found.get(nid, [])
    if not paths:
        missing.append(nid)
        continue
    if len(paths) > 1:
        dupes.append((nid, [str(p) for p in paths]))
    src = paths[0]               # take the first if duplicated across folders
    dst = DEST_DIR / src.name
    if dst.exists():
        already.append(nid)
        continue
    shutil.copy2(str(src), str(dst))   # copy2 preserves metadata; never moves
    copied.append(nid)

print(f"Copied        : {len(copied)}")
print(f"Already there : {len(already)}")
print(f"Missing       : {len(missing)}")
print(f"Duplicated    : {len(dupes)} (had files in >1 folder; used first)")

if missing:
    print(f"\nFirst 20 MISSING nsrrids (no event file found): {missing[:20]}")
if dupes:
    print(f"\nFirst 5 duplicated: {dupes[:5]}")

# COMMAND ----------

# DBTITLE 1,Verify destination contents
dest_files = list(DEST_DIR.glob("*.csv"))
dest_ids = set()
for p in dest_files:
    m = id_re.search(p.name)
    if m:
        dest_ids.add(int(m.group(1)))

print(f"Destination folder: {DEST_DIR}")
print(f"Files in destination : {len(dest_files)}")
print(f"Cohort nsrrids covered: {len(dest_ids & set(cohort))} / {len(cohort)}")
covered_pct = len(dest_ids & set(cohort)) / len(cohort) * 100
print(f"Coverage: {covered_pct:.1f}%")

# Save the list of missing nsrrids for the record
if missing:
    import pandas as pd
    pd.DataFrame({"nsrrid_missing_events": missing}).to_csv(
        DEST_DIR / "MISSING_event_files.csv", index=False)
    print(f"Wrote MISSING_event_files.csv ({len(missing)} ids)")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
