# Databricks notebook source
# DBTITLE 1,pips
# MAGIC %pip install \
# MAGIC numpy \
# MAGIC scikit-learn \
# MAGIC umap-learn \
# MAGIC optuna \
# MAGIC joblib \
# MAGIC psutil \
# MAGIC pandas \
# MAGIC tqdm
# MAGIC %pip install hdbscan
# MAGIC %pip install faiss-cpu
# MAGIC %pip install optuna-dashboard
# MAGIC
# MAGIC # GPU stages (final UMAP + final HDBSCAN) use NVIDIA RAPIDS cuML.
# MAGIC # On a Databricks GPU ML runtime, cuML is often preinstalled. If the
# MAGIC # import in the "GPU setup" cell fails, install the matching wheel:
# MAGIC #   %pip install cuml-cu12   (for CUDA 12 runtimes)
# MAGIC #   %pip install cuml-cu11   (for CUDA 11 runtimes)
# MAGIC # Check your runtime's CUDA version with: !nvcc --version
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import cupy as cp
import cuml
from cuml.manifold import UMAP as cuUMAP
from cuml.cluster import HDBSCAN as cuHDBSCAN
print(f"cuML {cuml.__version__} ready; GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# COMMAND ----------

import os

for p in ["/dbfs/tmp/new_hdbscan_umap.csv", "/local_disk0/new_hdbscan_umap.csv"]:

    if os.path.exists(p): os.remove(p)
 

# COMMAND ----------

# DBTITLE 1,installs
# ============================================================

# STANDARD LIBRARIES

# ============================================================
 
import os

import glob

import time

from pathlib import Path

from multiprocessing import Manager
 
# ============================================================

# NUMERICAL / DATA

# ============================================================
 
import numpy as np

import pandas as pd

import psutil
 
# ============================================================

# VISUALIZATION

# ============================================================
 
import matplotlib.pyplot as plt
 
import plotly.io as pio

import plotly.graph_objects as go

from plotly.subplots import make_subplots
 
pio.renderers.default = "browser"
 
# ============================================================

# PROGRESS BARS

# ============================================================
 
from tqdm.auto import tqdm
 
# ============================================================

# MACHINE LEARNING

# ============================================================
 
from umap import UMAP
 
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score
 
# ============================================================

# HYPERPARAMETER TUNING

# ============================================================
 
import optuna
 
# ============================================================

# OPTIONAL PARALLEL UTILITIES

# ============================================================
 
from joblib import Parallel, delayed
 



import time
import psutil
import os
import shutil
import faiss
import hdbscan

# COMMAND ----------

# DBTITLE 1,configs
# CONFIG (same as your tag logic)

# ----------------------------
# UPDATED to consume the new representation_extracting.py outputs:
#   - cache_dir = /dbfs/tmp/new_pftsleep_cache (matches new extraction)
#   - num_files = 1219 (1224 master CSV rows - 5 nsrrid-not-in-CSV)
#   - cache_tag ends with "__concat7ch" because embeddings are now the
#     7-channel-concatenated 3584-dim vectors
#   - D = 3584, not 512
# ----------------------------

cache_dir = Path(globals().get("CACHE_DIR", "/dbfs/tmp/new_pftsleep_cache"))

encoder_name = globals().get("ENCODER_NAME", "PFTSleep")

num_files = globals().get("NUM_FILES", globals().get("num_files", 1219))

frequency = globals().get("FREQUENCY", 125)

win_length = globals().get("WIN_LENGTH", 750)

hop_length = globals().get("HOP_LENGTH", 750)

max_seq_len_sec = globals().get("MAX_SEQ_LEN_SEC", 8 * 3600)

def cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec):

    # __concat7ch must match what representation_extracting.py writes.
    return (f"{encoder_name}__files{num_files}__freq{frequency}"
            f"__win{win_length}__hop{hop_length}__max{max_seq_len_sec}__concat7ch")

tag = cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec)

shard_dir = cache_dir / f"{tag}_shards"

shard_files = sorted(glob.glob(str(shard_dir / "Z_part_*.npy")))

assert len(shard_files) > 0, f"No shards found in {shard_dir}"

print(f"Found {len(shard_files)} shards")

print(f"RAM now: {psutil.virtual_memory().available / 1e9:.2f} GB free")

# ----------------------------

# 1) Compute total rows cheaply (mmap_mode avoids loading)

# ----------------------------

# 3584 = 7 channels x 512, concatenated per-window embedding dimension.
D = 3584

total_rows = 0

first = np.load(shard_files[0], mmap_mode="r")

print("Example shard shape/dtype:", first.shape, first.dtype)

# Fail loudly if pointed at old 512-dim shards rather than silently producing
# wrong cluster results.
assert first.shape[1] == D, (
    f"Shard width {first.shape[1]} != expected {D}. The shards in {shard_dir} "
    f"are not the 7-channel-concatenated extraction outputs. Re-run "
    f"representation_extracting.py, or point cache_dir / tag at the right run."
)

for f in shard_files:

    total_rows += np.load(f, mmap_mode="r").shape[0]

print(f"Total rows: {total_rows:,}  (expected ~5,899,200)")

print(f"Approx raw size float16: {total_rows * D * 2 / 1e9:.2f} GB")

print(f"Approx raw size float32: {total_rows * D * 4 / 1e9:.2f} GB")

# ----------------------------

# 2) Build a memmap on local NVMe (fast + avoids RAM ceilings)

# ----------------------------

local_dir = Path("/local_disk0/pftsleep_memmap")

local_dir.mkdir(parents=True, exist_ok=True)

mm_path = local_dir / f"X__{tag}__l2norm_f32.memmap"

shape_path = local_dir / f"X__{tag}__shape.txt"

# Create/overwrite memmap file

X_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(total_rows, D))

# ----------------------------

# 3) Fill memmap sequentially (no vstack, no giant allocations)

# ----------------------------

t0 = time.time()

offset = 0

for f in tqdm(shard_files, desc="Writing memmap", unit="file"):

    shard = np.load(f)  # should be float16

    if shard.dtype != np.float16:

        # still fine; we cast below, but this warns you if storage isn't what you expect

        pass

    n = shard.shape[0]

    X_mm[offset:offset+n, :] = shard.astype(np.float32, copy=False)

    offset += n

X_mm.flush()

t1 = time.time()

with open(shape_path, "w") as s:

    s.write(f"{total_rows},{D}\n")

print(f"\nMemmap written: {mm_path}")

print(f"Write time: {t1 - t0:.2f} sec")

print(f"RAM now: {psutil.virtual_memory().available / 1e9:.2f} GB free")

# ----------------------------

# 4) In-place L2 normalize in chunks (still memmap-backed)

# ----------------------------

print("\nNormalizing memmap in chunks...")

chunk_rows = 250_000  # tune if you want (100k–500k is fine)

eps = 1e-8

t2 = time.time()

for start in tqdm(range(0, total_rows, chunk_rows), desc="L2 normalize", unit="chunk"):

    end = min(total_rows, start + chunk_rows)

    block = X_mm[start:end, :]  # view into memmap (does not load everything)

    norms = np.linalg.norm(block, axis=1, keepdims=True)

    block /= np.maximum(norms, eps)

X_mm.flush()

t3 = time.time()

print(f"Normalization time: {t3 - t2:.2f} sec")

print("✅ Memmap X is ready. Use X_mm like a normal array: X_mm[i:j]")

print(f"RAM now: {psutil.virtual_memory().available / 1e9:.2f} GB free")

# ----------------------------

# 5) Load your metadata (small) - NEW LAYOUT

# ----------------------------
# UPDATED for the new representation_extracting.py outputs.
#
# OLD pipeline produced one array, zarr_file_idx, mapping each window to its
# zero-based source-file index. Downstream code then looked up subject IDs via
# np.array(zarr_nsrrids)[zarr_file_idx]. That indirection was fragile because
# zarr_nsrrids was rebuilt from a folder glob whose order could drift.
#
# NEW pipeline produces two metadata files directly:
#   nsrrid__<tag>.npy         per-window nsrrid (string, e.g. "200002")
#   nsrrid_intidx__<tag>.npy  per-window nsrrid (int32, e.g. 200002)
#
# We now keep `zarr_file_idx` as a NAME for backward compatibility with
# everything downstream that already references it -- but it now holds the
# per-window nsrrid INTEGER directly. This means anywhere old code did
#     window_nsrrid = np.array(zarr_nsrrids)[zarr_file_idx]
# can now use `zarr_file_idx` itself. We also define
# `window_nsrrid` as the explicit name for clarity in new code.

night_id = np.load(cache_dir / f"night_id__{tag}.npy")

time_idx = np.load(cache_dir / f"time_idx__{tag}.npy")

# FIX: load the REAL nsrrid (e.g. 200002), not nsrrid_intidx. The intidx file
# holds a sequential integer label (0,1,2,...) assigned at extraction time and
# does NOT match the demographics CSV - loading it caused "Matched 0 / 1219".
# The string nsrrid file holds the actual subject IDs.
window_nsrrid = np.load(cache_dir / f"nsrrid__{tag}.npy",
                        allow_pickle=True).astype(np.int64)

# Backward-compatible alias for downstream cells.
zarr_file_idx = window_nsrrid

print("Metadata loaded:",
      f"night_id={night_id.shape}",
      f"time_idx={time_idx.shape}",
      f"window_nsrrid={window_nsrrid.shape}")
print(f"Unique subjects in metadata: {len(np.unique(window_nsrrid)):,}")


# COMMAND ----------

# DBTITLE 1,Derive ordered subject list from extraction metadata
# -------------------------
# UPDATED: derive zarr_nsrrids from EXTRACTION METADATA (the authoritative
# source) instead of re-globbing a zarr folder. The previous folder-glob
# approach was a real source of misalignment bugs because:
#   - zarrs may exist on disk for subjects skipped during extraction
#   - the glob order is OS-dependent
#   - the order MUST match the order of files extraction actually processed,
#     which is what defines the integer file index in night_id
# Deriving subject order from the metadata is exact by construction.
# -------------------------

# Per-window night_id is the file-position index at extraction time. To get
# the ordered subject list, group by night_id and take the FIRST nsrrid in
# each group (every window from the same file has the same nsrrid).
_order = pd.DataFrame({"night_id": night_id, "nsrrid": window_nsrrid})
zarr_nsrrids = (_order.drop_duplicates("night_id")
                       .sort_values("night_id")["nsrrid"]
                       .astype(int)
                       .tolist())
print(f"Derived ordered subject list: {len(zarr_nsrrids)} subjects")
print(f"First 10 nsrrids: {zarr_nsrrids[:10]}")

# Load and align demographics to the extraction-derived order.
demographics_df = pd.read_csv(
    "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel.csv"
)
demographics_df["nsrrid"] = demographics_df["nsrrid"].astype(int)
print(f"Demographics CSV shape: {demographics_df.shape}")

order_df = pd.DataFrame({
    "nsrrid": zarr_nsrrids,
    "order":  range(len(zarr_nsrrids)),
})
demographics_ordered = (demographics_df.merge(order_df, on="nsrrid", how="inner")
                                       .sort_values("order")
                                       .drop("order", axis=1))
print(f"Matched {len(demographics_ordered)} / {len(zarr_nsrrids)} subjects in demographics")

# Persist the aligned demographics so analysis notebooks see the same order.
output_path = "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel_ordered.csv"
demographics_ordered.to_csv(output_path, index=False)
print(f"Saved aligned demographics: {output_path}")
print(f"Order check (first 5):"
      f" extraction={zarr_nsrrids[:5]}"
      f" demographics={demographics_ordered['nsrrid'].head(5).tolist()}")

# COMMAND ----------

# DBTITLE 1,umap, hdbscan, composite score tuning
import numpy as np

import pandas as pd

import optuna

import hdbscan

import os

import shutil

import time

import psutil
 
from umap import UMAP

from optuna.exceptions import TrialPruned

from sklearn.metrics import silhouette_score, davies_bouldin_score

from hdbscan.validity import validity_index
 
# Ensure numpy array

night_id = np.asarray(night_id)
 
# Sort indices by night_id

sorted_idx = np.argsort(night_id)

sorted_nights = night_id[sorted_idx]
 
# Find boundaries where night changes

split_points = np.where(np.diff(sorted_nights) != 0)[0] + 1
 
# Split indices into groups

groups = np.split(sorted_idx, split_points)
 
# Map night_id -> row indices

unique_nights = sorted_nights[np.concatenate(([0], split_points))]

night_to_rows = dict(zip(unique_nights, groups))
 
print(f"✅ Built mapping for {len(night_to_rows)} nights")
 
# --------------------------------------------------

# BALANCED NIGHT SAMPLER (unchanged)

# --------------------------------------------------
 
def sample_balanced_rows_by_night(night_to_rows, rng, n_nights, max_rows_per_night):
 
    night_keys = np.array(list(night_to_rows.keys()))
 
    chosen_nights = rng.choice(

        night_keys,

        size=min(n_nights, len(night_keys)),

        replace=False

    )
 
    sampled_rows = []
 
    for nid in chosen_nights:
 
        rows = night_to_rows[nid]
 
        if len(rows) > max_rows_per_night:
 
            rows = rng.choice(rows, size=max_rows_per_night, replace=False)
 
        sampled_rows.append(rows)
 
    return np.concatenate(sampled_rows)
 
 
# --------------------------------------------------

# METRICS

# --------------------------------------------------
 
def compute_metrics(X_emb, labels, clusterer):
 
    mask = labels != -1
 
    if mask.sum() < 500:

        return None
 
    unique_clusters = np.unique(labels[mask])
 
    if len(unique_clusters) < 2:

        return None
 
    noise_fraction = float((labels == -1).mean())
 
    X_core = X_emb[mask]

    y_core = labels[mask]
 
    sample_idx = np.random.choice(len(X_core),

                                 min(20000, len(X_core)),

                                 replace=False)
 
    sil = float(silhouette_score(X_core[sample_idx], y_core[sample_idx]))
 
    db = float(davies_bouldin_score(X_core, y_core))
 
    # cuML HDBSCAN returns cluster_persistence_ and probabilities_ as cupy
    # device arrays. Convert to host numpy before reducing so compute_metrics
    # works with both CPU hdbscan and GPU cuHDBSCAN.
    _persist = getattr(clusterer, "cluster_persistence_", np.array([]))
    try:
        import cupy as _cp; _persist = _cp.asnumpy(_persist)
    except (TypeError, AttributeError, ImportError):
        _persist = np.asarray(_persist)

    _probs = getattr(clusterer, "probabilities_", np.zeros(len(labels)))
    try:
        import cupy as _cp; _probs = _cp.asnumpy(_probs)
    except (TypeError, AttributeError, ImportError):
        _probs = np.asarray(_probs)

    persistence = float(np.mean(_persist)) if len(_persist) > 0 else 0.0

    probabilities = _probs[mask]

    mean_prob = float(probabilities.mean())

    sample_idx = np.random.choice(len(X_emb), min(20000, len(X_emb)), replace=False)

    try:

        dbcv = validity_index(X_emb[sample_idx].astype(np.float64), labels[sample_idx])

    except ValueError:

        dbcv = 0.0
 
    _, counts = np.unique(y_core, return_counts=True)
 
    largest_cluster_fraction = float(counts.max() / len(y_core))
 
    small_cluster_fraction = float(np.mean(counts < 20))
 
    return dict(

        silhouette=sil,

        davies_bouldin=db,

        persistence=persistence,

        mean_probability=mean_prob,

        dbcv=dbcv,

        n_clusters=len(unique_clusters),

        noise_fraction=noise_fraction,

        largest_cluster_fraction=largest_cluster_fraction,

        small_cluster_fraction=small_cluster_fraction,

    )
 
 
# --------------------------------------------------

# COMPOSITE SCORE (NEW)

# --------------------------------------------------
 
def composite_score(metrics, trial):
 
    if metrics is None:
        return -1e9
 
    # --------------------------------------------------
    # LEARNABLE WEIGHTS
    # --------------------------------------------------
 
    w_dbcv = trial.suggest_float("w_dbcv", 0.25, 0.55)
    w_persistence = trial.suggest_float("w_persistence", 0.05, 0.30)
    w_probability = trial.suggest_float("w_probability", 0.05, 0.25)
    w_silhouette = trial.suggest_float("w_silhouette", 0.00, 0.20)
    w_noise = trial.suggest_float("w_noise", 0.00, 0.15)
    w_fragmentation = trial.suggest_float("w_fragmentation", 0.00, 0.10)
 
    # normalize weights
 
    total = (
        w_dbcv
        + w_persistence
        + w_probability
        + w_silhouette
        + w_noise
        + w_fragmentation
    )
 
    w_dbcv /= total
    w_persistence /= total
    w_probability /= total
    w_silhouette /= total
    w_noise /= total
    w_fragmentation /= total
 
    # --------------------------------------------------
    # NORMALIZE METRICS
    # --------------------------------------------------
 
    dbcv_norm = (metrics["dbcv"] + 1) / 2
 
    noise_target = 0.25
    noise_score = 1 - abs(metrics["noise_fraction"] - noise_target)
 
    collapse_penalty = max(
        0,
        metrics["largest_cluster_fraction"] - 0.80
    )
 
    fragmentation_penalty = metrics["small_cluster_fraction"]
 
    # --------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------
 
    score = (
        w_dbcv * dbcv_norm
        + w_persistence * metrics["persistence"]
        + w_probability * metrics["mean_probability"]
        + w_silhouette * metrics["silhouette"]
        + w_noise * noise_score
        - w_fragmentation * fragmentation_penalty
        - 0.10 * collapse_penalty
        - 0.10 * metrics["davies_bouldin"]
    )
 
    # prevent 1-cluster collapse solutions
 
    cluster_penalty = min(metrics["n_clusters"] / 8, 1)
 
    return score * cluster_penalty 
 
# --------------------------------------------------

# OPTUNA OBJECTIVE

# --------------------------------------------------

FREEZE_AFTER_N_TRIALS = 40

WEIGHT_STD_THRESHOLD = 0.015
 
def get_frozen_weights_if_ready(study):
 
    completed = [

        t for t in study.trials

        if t.state.name == "COMPLETE"

        and "w_dbcv" in t.params

    ]
 
    if len(completed) < FREEZE_AFTER_N_TRIALS:

        return None
 
    import numpy as np
 
    weight_matrix = np.array([

        [

            t.params["w_dbcv"],

            t.params["w_persistence"],

            t.params["w_probability"],

            t.params["w_silhouette"],

            t.params["w_noise"],

            t.params["w_fragmentation"],

        ]

        for t in completed[-FREEZE_AFTER_N_TRIALS:]

    ])
 
    weight_std = weight_matrix.std(axis=0)
 
    if np.max(weight_std) < WEIGHT_STD_THRESHOLD:
 
        frozen = weight_matrix.mean(axis=0)
 
        frozen /= frozen.sum()
 
        print("\n🔒 Freezing metric weights:", frozen, "\n")
 
        study.set_user_attr("frozen_weights", frozen.tolist())
 
        return frozen
 
    return None
 

def objective(trial):
 
    t0_trial = time.time()
 
    # ---------------- UMAP ----------------
 
    umap_n_neighbors = int(trial.suggest_float("umap_n_neighbors", 10, 30))
 
    umap_n_components = 3
 
    umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 0.5)
 
    # ---------------- HDBSCAN ----------------
 
    hdb_min_cluster_size = trial.suggest_int("hdb_min_cluster_size", 20, 400)
 
    hdb_min_samples = int(trial.suggest_float("hdb_min_samples", 5, 80))
 
    hdb_cluster_selection_method = trial.suggest_categorical(

        "hdb_cluster_selection_method",

        ["eom", "leaf"]

    )
 
    n_resamples = 3
 
    scores = []
 
    metrics_all = []
 
    for rep in range(n_resamples):
 
        rng = np.random.default_rng(1000 + rep)
 
        row_idx = sample_balanced_rows_by_night(

            night_to_rows,

            rng,

            n_nights=len(night_to_rows),

            max_rows_per_night=500

        )
 
        X_sub = np.asarray(X_mm[row_idx], dtype=np.float64)
 
        # ---------------- UMAP ----------------
 
        # GPU UMAP (cuML). Note: cuML does not accept transform_seed,
        # low_memory, or n_jobs - those are CPU-umap args. random_state=None
        # keeps trials stochastic, matching the original sweep intent.
        X_sub_gpu = cp.asarray(np.asarray(X_mm[row_idx], dtype=np.float32))
        reducer = cuUMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=None,
        )
        X_umap_gpu = reducer.fit_transform(X_sub_gpu)
        X_umap = cp.asnumpy(X_umap_gpu).astype(np.float64)
        del X_sub_gpu, X_umap_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # GPU HDBSCAN (cuML) on the compact 3D embedding from UMAP.
        X_umap_gpu2 = cp.asarray(X_umap.astype(np.float32))
        clusterer = cuHDBSCAN(
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            metric="euclidean",
            cluster_selection_method=hdb_cluster_selection_method,
            prediction_data=False,
        )
        labels_gpu = clusterer.fit_predict(X_umap_gpu2)
        labels = cp.asnumpy(labels_gpu)
        del X_umap_gpu2, labels_gpu
        cp.get_default_memory_pool().free_all_blocks()
 
        metrics = compute_metrics(X_umap, labels, clusterer)
 
        score = composite_score(metrics, trial)
 
        scores.append(score)
 
        metrics_all.append(metrics)
 
        trial.report(score, step=rep)
 
        # pruning guards
 
        if metrics is None:

            raise TrialPruned()
 
        if metrics["noise_fraction"] > 0.85:

            raise TrialPruned()
 
        if metrics["n_clusters"] < 2:

            raise TrialPruned()
 
        if trial.should_prune():

            raise TrialPruned()
 
    scores = np.array(scores)
 
    valid_metrics = [m for m in metrics_all if m is not None]
 
    if len(valid_metrics) == 0:

        return -1e9
 
    final_score = float(np.mean(scores) - 0.5 * np.std(scores))
 
    trial.set_user_attr("score_mean", float(np.mean(scores)))
 
    trial.set_user_attr("score_std", float(np.std(scores)))
 
    trial.set_user_attr(

        "mean_clusters",

        float(np.mean([m["n_clusters"] for m in valid_metrics]))

    )
 
    trial.set_user_attr(

        "mean_noise",

        float(np.mean([m["noise_fraction"] for m in valid_metrics]))

    )
 
    trial.set_user_attr(

        "mean_dbcv",

        float(np.mean([m["dbcv"] for m in valid_metrics]))

    )
 
    trial.set_user_attr(

        "mean_persistence",

        float(np.mean([m["persistence"] for m in valid_metrics]))

    )
    trial.set_user_attr("w_dbcv", trial.params["w_dbcv"])
    trial.set_user_attr("w_persistence", trial.params["w_persistence"])
    trial.set_user_attr("w_probability", trial.params["w_probability"])
    trial.set_user_attr("w_silhouette", trial.params["w_silhouette"])
    trial.set_user_attr("w_noise", trial.params["w_noise"])
    trial.set_user_attr("w_fragmentation", trial.params["w_fragmentation"])
    t1_trial = time.time()
 
    trial_total = t1_trial - t0_trial
 
    print(

        f"[Trial {trial.number}] "

        f"score={final_score:.4f} | "

        f"clusters={trial.user_attrs['mean_clusters']:.1f} | "

        f"noise={trial.user_attrs['mean_noise']:.2f} | "

        f"dbcv={trial.user_attrs['mean_dbcv']:.3f} | "

        f"time={trial_total/60:.2f}m"

    )
 
    print(

        f"RAM available: "

        f"{psutil.virtual_memory().available/1e9:.1f} GB"

    )
    # --------------------------------------------------

    # PER-TRIAL CSV LOGGING (FULL METRICS EXPORT)

    # --------------------------------------------------
 
    cluster_counts = [m["n_clusters"] for m in valid_metrics]

    noise_vals = [m["noise_fraction"] for m in valid_metrics]

    dbcv_vals = [m["dbcv"] for m in valid_metrics]

    persist_vals = [m["persistence"] for m in valid_metrics]

    prob_vals = [m["mean_probability"] for m in valid_metrics]

    sil_vals = [m["silhouette"] for m in valid_metrics]

    db_vals = [m["davies_bouldin"] for m in valid_metrics]

    largest_vals = [m["largest_cluster_fraction"] for m in valid_metrics]

    small_vals = [m["small_cluster_fraction"] for m in valid_metrics]
 
    trial_record = {
 
        # trial info

        "trial": trial.number,

        "score": final_score,
 
        # UMAP params

        "umap_n_neighbors": umap_n_neighbors,

        "umap_n_components": umap_n_components,

        "umap_min_dist": umap_min_dist,
 
        # HDBSCAN params

        "hdb_min_cluster_size": hdb_min_cluster_size,

        "hdb_min_samples": hdb_min_samples,

        "hdb_cluster_selection_method": hdb_cluster_selection_method,
 
        # metric means

        "mean_n_clusters": np.mean(cluster_counts),

        "mean_noise_fraction": np.mean(noise_vals),

        "mean_dbcv": np.mean(dbcv_vals),

        "mean_persistence": np.mean(persist_vals),

        "mean_probability": np.mean(prob_vals),

        "mean_silhouette": np.mean(sil_vals),

        "mean_davies_bouldin": np.mean(db_vals),

        "mean_largest_cluster_fraction": np.mean(largest_vals),

        "mean_small_cluster_fraction": np.mean(small_vals),
 
        # metric variability (important for stability)

        "std_n_clusters": np.std(cluster_counts),

        "std_noise_fraction": np.std(noise_vals),

        "std_dbcv": np.std(dbcv_vals),

        "std_persistence": np.std(persist_vals),

        "std_probability": np.std(prob_vals),

        "std_silhouette": np.std(sil_vals),
 
        # learned weights

        "w_dbcv": trial.params.get("w_dbcv"),

        "w_persistence": trial.params.get("w_persistence"),

        "w_probability": trial.params.get("w_probability"),

        "w_silhouette": trial.params.get("w_silhouette"),

        "w_noise": trial.params.get("w_noise"),

        "w_fragmentation": trial.params.get("w_fragmentation"),
 
        # runtime

        "trial_total_sec": trial_total,
 
        # frozen scoring weights snapshot

        "frozen_weights": trial.study.user_attrs.get("frozen_weights"),

    }
 
    try:
 
        local_csv = "/local_disk0/new_hdbscan_umap.csv"
 
        pd.DataFrame([trial_record]).to_csv(
        local_csv,
        mode="a",
        header=not os.path.exists(local_csv),
        index=False
        )
        shutil.copy(local_csv, "/dbfs/tmp/new_hdbscan_umap.csv")
 
    except Exception as e:
 
        print(f"CSV logging failed: {e}")
 
    return final_score
 
 
# --------------------------------------------------

# STUDY

# --------------------------------------------------
 
study = optuna.create_study(

    study_name="umap_hdbscan_joint_density_v4",

    storage="sqlite:////local_disk0/umap_hdbscan_optuna_new.db",

    load_if_exists=True,

    direction="maximize",

    sampler=optuna.samplers.TPESampler(seed=42),

    pruner=optuna.pruners.MedianPruner(

        n_startup_trials=3,

        n_warmup_steps=1,

        interval_steps=1

    )

)
 
# --- Optuna sweep runs ONLY when config says so (cfg.clustering.run_optuna_sweep).
#     When False, we skip the 50-trial search entirely and use the fixed params
#     from config.yaml below (reproducible re-runs). ---
_RUN_OPTUNA = bool(cfg.clustering.run_optuna_sweep)
_N_TRIALS = int(getattr(cfg.clustering, "optuna_n_trials", 50))
if _RUN_OPTUNA:
    print(f"Optuna sweep ENABLED: running {_N_TRIALS} trials.")
    study.optimize(objective, n_trials=_N_TRIALS)
else:
    print("Optuna sweep DISABLED (run_optuna_sweep=false): "
          "skipping search, using fixed params from config.yaml.")

# COMMAND ----------

# DBTITLE 1,best trial selection manual
# ============================================================

# PARAM SOURCE depends on cfg.clustering.run_optuna_sweep:
#   True  -> auto-select the best trial from the sweep's trial CSV
#   False -> use the FIXED params from config.yaml (reproducible re-runs)

# ============================================================

if _RUN_OPTUNA:
    TRIAL_CSV_PATH = "/dbfs/tmp/new_hdbscan_umap.csv"
    print("\nLoading trials from CSV:", TRIAL_CSV_PATH)
    trial_df = pd.read_csv(TRIAL_CSV_PATH)

    # AUTO-SELECT the best trial = highest composite score across all completed
    # trials. Drop non-finite scores (failed/pruned) before choosing the max.
    trial_df = trial_df[np.isfinite(trial_df["score"])].copy()
    if len(trial_df) == 0:
        raise ValueError("No completed trials with a finite score in the CSV.")

    best_idx              = trial_df["score"].idxmax()
    row                   = trial_df.loc[best_idx]
    SELECTED_TRIAL_NUMBER = int(row["trial"])

    print(f"Auto-selected best trial by highest score: "
          f"trial {SELECTED_TRIAL_NUMBER} (score={row['score']:.6f})")
    print(f"  ({len(trial_df)} completed trials considered; "
          f"score range {trial_df['score'].min():.4f}..{trial_df['score'].max():.4f})")

    best_params = {
        "umap_n_neighbors": int(row["umap_n_neighbors"]),
        "umap_n_components": int(row["umap_n_components"]),
        "umap_min_dist": float(row["umap_min_dist"]),
        "hdb_min_cluster_size": int(row["hdb_min_cluster_size"]),
        "hdb_min_samples": int(row["hdb_min_samples"]),
        "hdb_cluster_selection_method": row["hdb_cluster_selection_method"],
    }
    print("\n========== SELECTED TRIAL (from Optuna) ==========")
    print("Trial:", SELECTED_TRIAL_NUMBER, "| Score:", row["score"])
else:
    # Fixed, locked params from config.yaml - no search, fully reproducible.
    cl = cfg.clustering
    best_params = {
        "umap_n_neighbors": int(cl.umap_n_neighbors),
        "umap_n_components": int(cl.umap_n_components),
        "umap_min_dist": float(getattr(cl, "umap_min_dist", 0.0)),
        "hdb_min_cluster_size": int(cl.hdbscan_min_cluster_size),
        "hdb_min_samples": int(cl.hdbscan_min_samples),
        "hdb_cluster_selection_method": cl.hdbscan_cluster_selection_method,
    }
    SELECTED_TRIAL_NUMBER = None
    print("\n========== FIXED PARAMS (from config.yaml) ==========")

print("Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
 

# COMMAND ----------

# DBTITLE 1,GPU setup (cuML)
# ============================================================
# GPU SETUP - cuML (RAPIDS). GPU-only, no CPU fallback (per request).
# Requires a GPU cluster with cuML installed. If this import fails, the
# notebook is on the wrong cluster or cuML is not installed - see the pip
# cell at the top for the install command matching your CUDA version.
# ============================================================
import cuml
import cupy as cp
from cuml.manifold import UMAP as cuUMAP
from cuml.cluster import HDBSCAN as cuHDBSCAN

print(f"cuML version: {cuml.__version__}")
# Confirm a GPU is actually visible.
import subprocess
print(subprocess.check_output(["nvidia-smi",
    "--query-gpu=name,memory.total,memory.used",
    "--format=csv,noheader"]).decode().strip())

# COMMAND ----------

# DBTITLE 1,umap on full data (GPU)
# ============================================================

# FULL DATASET UMAP EMBEDDING - GPU (cuML)

# ============================================================
#
# Strategy: fit cuML UMAP on a random subsample that fits comfortably in
# T4 VRAM (16 GB), then transform ALL windows in chunks. This keeps the
# full 3584-dim input (no PCA) and assigns a coordinate to every window.
#
# VRAM math: 500k x 3584 float32 = ~7 GB for the fit subset; each transform
# chunk of 200k x 3584 = ~2.9 GB. Both fit a T4 with headroom.
#
# cuML UMAP notes:
#   - cosine metric is supported by cuML UMAP.
#   - cuML does not use n_jobs / low_memory (those are CPU-umap args).
#   - random_state makes the embedding reproducible.

print("\nRunning full dataset UMAP embedding (GPU)...")
t0 = time.time()

N_FIT  = 500_000     # windows used to LEARN the manifold (fits T4 VRAM)
n_rows = X_mm.shape[0]
n_comp = int(best_params["umap_n_components"])

final_reducer = cuUMAP(
    n_neighbors=int(best_params["umap_n_neighbors"]),
    n_components=n_comp,
    min_dist=float(best_params["umap_min_dist"]),
    metric="cosine",
    random_state=42,
)

if n_rows <= N_FIT:
    X_gpu = cp.asarray(np.asarray(X_mm), dtype=cp.float32)
    embedding_full = cp.asnumpy(final_reducer.fit_transform(X_gpu)).astype(np.float32)
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()
else:
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(n_rows, size=N_FIT, replace=False)
    fit_idx.sort()   # sorted access is faster on a memmap
    print(f"  Fitting cuML UMAP on {N_FIT:,} of {n_rows:,} windows...")

    # Move the fit subset to GPU, fit, then free it.
    X_fit_gpu = cp.asarray(np.asarray(X_mm[fit_idx]), dtype=cp.float32)
    final_reducer.fit(X_fit_gpu)
    del X_fit_gpu
    cp.get_default_memory_pool().free_all_blocks()

    embedding_full = np.empty((n_rows, n_comp), dtype=np.float32)
    chunk = 200_000
    print(f"  Transforming all {n_rows:,} windows in chunks of {chunk:,}...")
    for i in range(0, n_rows, chunk):
        j = min(i + chunk, n_rows)
        X_chunk_gpu = cp.asarray(np.asarray(X_mm[i:j]), dtype=cp.float32)
        embedding_full[i:j] = cp.asnumpy(
            final_reducer.transform(X_chunk_gpu)
        ).astype(np.float32)
        del X_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print(f"    {j:,} / {n_rows:,}")

print(f"UMAP (GPU) finished in {(time.time()-t0)/60:.2f} minutes")
print("Embedding shape:", embedding_full.shape)
 

# COMMAND ----------

# DBTITLE 1,load umap embeddings
# ============================================================
# LOAD PRECOMPUTED UMAP EMBEDDING
# ============================================================
 
embedding_path = "/dbfs/tmp/final_hdbscan_outputs/new_hdbscan_outputs/new/umap_embedding_full.npy"
 
print("\nLoading precomputed UMAP embedding...")
 
embedding_full = np.load(embedding_path)
 
print("Embedding loaded.")
print("Shape:", embedding_full.shape)
print("dtype:", embedding_full.dtype)

# COMMAND ----------

# DBTITLE 1,hdbscan on full data
# ============================================================
# FINAL HDBSCAN ON FULL EMBEDDING
# ============================================================
 
print("\nRunning final HDBSCAN clustering (GPU)...")
 
t0 = time.time()
 
# GPU HDBSCAN via cuML. Input is the compact 3D UMAP embedding (5.2M x 3),
# only ~62 MB, so it fits any GPU trivially - this is where GPU gives a
# large speedup over CPU HDBSCAN with no VRAM concern.
# NOTE: cuML HDBSCAN is not bit-identical to the CPU hdbscan library; cluster
# boundaries may differ marginally. Document this in methods.
final_clusterer = cuHDBSCAN(
    min_cluster_size=int(best_params["hdb_min_cluster_size"]),
    min_samples=int(best_params["hdb_min_samples"]),
    cluster_selection_method=best_params["hdb_cluster_selection_method"],
    metric="euclidean",
    prediction_data=True,
)

embedding_gpu  = cp.asarray(embedding_full, dtype=cp.float32)
cluster_labels = cp.asnumpy(final_clusterer.fit_predict(embedding_gpu))
# cuML exposes probabilities_ as a device array; bring it to host.
cluster_probabilities = cp.asnumpy(final_clusterer.probabilities_)
del embedding_gpu
cp.get_default_memory_pool().free_all_blocks()
# ============================================================

# FULL-DATASET CLUSTERING METRICS (INCLUDING DBCV)

# ============================================================
 
from hdbscan.validity import validity_index
 
print("\n===== FULL DATASET METRICS =====")
 
labels = cluster_labels

mask = labels != -1
 
# ----------------------------

# BASIC COUNTS

# ----------------------------
 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

noise_fraction = np.mean(labels == -1)
 
# ----------------------------

# PROBABILITY

# ----------------------------
 
mean_probability = (

    cluster_probabilities[mask].mean()

    if mask.sum() > 0 else 0

)
 
# ----------------------------

# PERSISTENCE

# ----------------------------
 
# cuML exposes cluster_persistence_ as a device array; bring it to host
# before reducing. Guard for both cuML (cupy) and CPU (numpy) cases.
_persist = getattr(final_clusterer, "cluster_persistence_", None)
if _persist is not None:
    try:
        _persist = cp.asnumpy(_persist)   # cuML device array -> host
    except (TypeError, AttributeError):
        _persist = np.asarray(_persist)   # already host (CPU fallback case)

if _persist is not None and len(_persist) > 0:

    mean_persistence = float(np.mean(_persist))

else:

    mean_persistence = 0
 
# ----------------------------

# CLUSTER SIZE STRUCTURE

# ----------------------------
 
cluster_sizes = np.unique(labels[mask], return_counts=True)[1]
 
largest_cluster_fraction = (

    cluster_sizes.max() / mask.sum()

    if len(cluster_sizes) > 0 else 0

)
 
small_cluster_fraction = np.mean(cluster_sizes < 20)
 
# ----------------------------

# SAMPLED DBCV (SAFE FOR LARGE DATASETS)

# ----------------------------
 
dbcv_sample_size = min(20000, len(embedding_full))
 
sample_idx = np.random.choice(

    len(embedding_full),

    dbcv_sample_size,

    replace=False

)
 
try:

    sampled_dbcv = validity_index(

        embedding_full[sample_idx].astype(np.float64),

        labels[sample_idx]

    )

except Exception:

    sampled_dbcv = np.nan
 
# ----------------------------

# PRINT RESULTS

# ----------------------------
 
print(f"Clusters: {n_clusters}")

print(f"Noise fraction: {noise_fraction:.4f}")

print(f"Mean assignment probability: {mean_probability:.4f}")

print(f"Mean persistence: {mean_persistence:.4f}")

print(f"Largest cluster fraction: {largest_cluster_fraction:.4f}")

print(f"Small cluster fraction: {small_cluster_fraction:.4f}")

print(f"Sampled DBCV: {sampled_dbcv:.4f}")
  
 
print(f"HDBSCAN finished in {(time.time()-t0):.2f} seconds")
 
unique, counts = np.unique(cluster_labels, return_counts=True)
 
print("\nCluster counts:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c:,} windows")

# COMMAND ----------

# DBTITLE 1,save embeddings and labels
# ============================================================

# SAVE FULL EMBEDDING + LABELS

# ============================================================
 
LOCAL_OUT = "/local_disk0/final_hdbscan_outputs/new_hdbscan_outputs"

DBFS_OUT = "/dbfs/tmp/final_hdbscan_outputs/new_hdbscan_outputs/"
 
os.makedirs(LOCAL_OUT, exist_ok=True)

os.makedirs(DBFS_OUT, exist_ok=True)
 
embedding_path_local = f"{LOCAL_OUT}/umap_embedding_full.npy"

labels_path_local = f"{LOCAL_OUT}/cluster_labels.npy"

prob_path_local = f"{LOCAL_OUT}/cluster_probabilities.npy"
 
np.save(embedding_path_local, embedding_full)

np.save(labels_path_local, cluster_labels)

np.save(prob_path_local, cluster_probabilities)
 
shutil.copy(embedding_path_local, DBFS_OUT)

shutil.copy(labels_path_local, DBFS_OUT)

shutil.copy(prob_path_local, DBFS_OUT)
 
print("Saved embeddings and labels.")
 

# COMMAND ----------

# DBTITLE 1,Conditional refinement (save baseline, refine if triggered)
# =============================================================================
# CONDITIONAL REFINEMENT - save baseline, then refine if triggered
# =============================================================================
# Inserted AFTER the full HDBSCAN (cluster_labels, cluster_probabilities,
# final_clusterer, embedding_full all exist). This is now the ONLY refinement
# path; the old recursive_refine_hdbscan has been removed.
#
# Behaviour (per your spec):
#   1. ALWAYS save the ORIGINAL clustering + per-cluster metrics (baseline).
#   2. TRIGGER refinement if  noise_fraction > NOISE_TRIGGER  OR
#                             largest_cluster_fraction > DOMINANCE_TRIGGER.
#   3. Refinement does BOTH: split dominant cluster(s) and reassign noise
#      points to their nearest cluster.
#   4. Save the REFINED clustering + per-cluster metrics SEPARATELY so the two
#      can be compared / the better one chosen.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

# ---- thresholds (could be promoted to config later) ------------------------
NOISE_TRIGGER     = 0.35
DOMINANCE_TRIGGER = 0.70

# ---- where to save (mirror the script's existing output dirs) --------------
BASE_OUT = Path("/dbfs/tmp/final_hdbscan_outputs")
(BASE_OUT / "baseline").mkdir(parents=True, exist_ok=True)
(BASE_OUT / "refined").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def per_cluster_metrics(labels, embedding, probabilities=None,
                        persistence=None):
    """Per-cluster metric table: size, fraction, mean probability, and
    persistence (if available). One row per cluster (noise -1 included)."""
    out = []
    n_total = len(labels)
    uniq = sorted(set(int(c) for c in labels))
    # persistence is indexed by cluster id 0..K-1 (noise excluded)
    for c in uniq:
        mask = labels == c
        n = int(mask.sum())
        row = {
            "cluster": c,
            "n_windows": n,
            "fraction": n / n_total if n_total else 0.0,
            "mean_probability": (float(probabilities[mask].mean())
                                 if probabilities is not None and n > 0 else np.nan),
        }
        if persistence is not None and c >= 0 and c < len(persistence):
            row["persistence"] = float(persistence[c])
        else:
            row["persistence"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def overall_metrics(labels, probabilities=None, persistence=None):
    """Whole-clustering summary metrics."""
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float(np.mean(labels == -1))
    if mask.sum() > 0:
        _, sizes = np.unique(labels[mask], return_counts=True)
        largest = float(sizes.max() / mask.sum())
        small = float(np.mean(sizes < 20))
    else:
        largest = small = 0.0
    mean_persist = (float(np.mean(persistence))
                    if persistence is not None and len(persistence) > 0 else np.nan)
    mean_prob = (float(probabilities[mask].mean())
                 if probabilities is not None and mask.sum() > 0 else np.nan)
    return {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "largest_cluster_fraction": largest,
        "small_cluster_fraction": small,
        "mean_persistence": mean_persist,
        "mean_assignment_probability": mean_prob,
        "n_windows_total": int(len(labels)),
        "n_windows_clustered": int(mask.sum()),
    }


def save_clustering(tag_dir, labels, embedding, probabilities, persistence):
    """Save labels + embedding + both metric tables under tag_dir."""
    d = BASE_OUT / tag_dir
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "cluster_labels.npy", labels)
    np.save(d / "cluster_probabilities.npy", probabilities)
    # per-cluster metrics
    pcm = per_cluster_metrics(labels, embedding, probabilities, persistence)
    pcm.to_csv(d / "per_cluster_metrics.csv", index=False)
    # overall metrics (single-row)
    om = overall_metrics(labels, probabilities, persistence)
    pd.DataFrame([om]).to_csv(d / "overall_metrics.csv", index=False)
    print(f"  saved {tag_dir}: {om['n_clusters']} clusters, "
          f"noise={om['noise_fraction']:.3f}, "
          f"largest={om['largest_cluster_fraction']:.3f}")
    return om, pcm


# --------------------------------------------------------------------------- #
# STEP 1 - ALWAYS save the baseline (original full-HDBSCAN result)
# --------------------------------------------------------------------------- #
print("\n===== SAVING BASELINE CLUSTERING =====")
_persist_arr = getattr(final_clusterer, "cluster_persistence_", None)
if _persist_arr is not None:
    try:
        import cupy as cp
        _persist_arr = cp.asnumpy(_persist_arr)
    except Exception:
        _persist_arr = np.asarray(_persist_arr)

baseline_overall, baseline_pcm = save_clustering(
    "baseline", cluster_labels, embedding_full, cluster_probabilities, _persist_arr)


# --------------------------------------------------------------------------- #
# STEP 2 - decide whether to refine
# --------------------------------------------------------------------------- #
_noise = baseline_overall["noise_fraction"]
_largest = baseline_overall["largest_cluster_fraction"]
trigger = (_noise > NOISE_TRIGGER) or (_largest > DOMINANCE_TRIGGER)

print(f"\n===== REFINEMENT TRIGGER CHECK =====")
print(f"  noise_fraction           = {_noise:.3f}  (trigger if > {NOISE_TRIGGER})")
print(f"  largest_cluster_fraction = {_largest:.3f}  (trigger if > {DOMINANCE_TRIGGER})")
print(f"  -> refinement {'TRIGGERED' if trigger else 'NOT triggered'}")


# --------------------------------------------------------------------------- #
# STEP 3 - refinement: split dominant cluster(s) AND reassign noise
# --------------------------------------------------------------------------- #
def refine_split_and_denoise(labels, embedding,
                             dominance_thresh=DOMINANCE_TRIGGER,
                             min_cluster_size=None, min_samples=None):
    """One refinement pass that does BOTH:
      (a) splits any cluster larger than dominance_thresh of clustered points
          by re-running HDBSCAN on just that cluster's points, and
      (b) reassigns noise points to their nearest cluster centroid.
    Returns refined_labels and a log DataFrame.
    """
    import hdbscan
    refined = labels.copy()
    log = []
    next_label = (refined.max() + 1) if (refined >= 0).any() else 0

    # ---- (a) split dominant clusters ----
    clustered_mask = refined != -1
    n_clustered = int(clustered_mask.sum())
    if n_clustered > 0:
        ids, sizes = np.unique(refined[clustered_mask], return_counts=True)
        for cid, sz in zip(ids, sizes):
            frac = sz / n_clustered
            if frac > dominance_thresh:
                idx = np.where(refined == cid)[0]
                sub_emb = embedding[idx]
                mcs = min_cluster_size or max(50, int(0.02 * len(idx)))
                ms = min_samples or 10
                try:
                    sub = hdbscan.HDBSCAN(
                        min_cluster_size=int(mcs), min_samples=int(ms),
                        cluster_selection_method="eom", metric="euclidean",
                    )
                    sub_labels = sub.fit_predict(sub_emb)
                    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
                    if n_sub >= 2:
                        # remap sub-clusters to fresh global ids; keep sub-noise
                        # as noise (will be picked up by denoise step below)
                        for s in sorted(set(sub_labels)):
                            if s == -1:
                                refined[idx[sub_labels == -1]] = -1
                            else:
                                refined[idx[sub_labels == s]] = next_label
                                next_label += 1
                        log.append({"action": "split", "cluster": int(cid),
                                    "size": int(sz), "fraction": float(frac),
                                    "n_subclusters": int(n_sub),
                                    "status": "accepted_split"})
                    else:
                        log.append({"action": "split", "cluster": int(cid),
                                    "size": int(sz), "fraction": float(frac),
                                    "n_subclusters": int(n_sub),
                                    "status": "no_acceptable_split"})
                except Exception as e:
                    log.append({"action": "split", "cluster": int(cid),
                                "size": int(sz), "fraction": float(frac),
                                "status": f"failed: {str(e)[:60]}"})

    # ---- (b) reassign noise points to their nearest cluster, but ONLY if the
    #          point lies within that cluster's characteristic radius. We use a
    #          pre-specified, data-defined rule (NOT a tuned target):
    #
    #            absorb a noise point into cluster c iff its distance to c's
    #            centroid <= the 75th percentile of c's own members' distances
    #            to that centroid.
    #
    #          Points beyond every cluster's characteristic radius REMAIN noise.
    #          The resulting noise fraction is therefore an OUTCOME we report,
    #          not a value we tune toward - which is the defensible framing for
    #          publication (no circular "loosen until noise <= X").
    DENOISE_RADIUS_PCTL = 75   # within-cluster distance percentile defining "close"
    noise_idx = np.where(refined == -1)[0]
    clustered_ids = sorted(c for c in set(refined) if c != -1)
    n_reassigned = 0
    if len(noise_idx) > 0 and len(clustered_ids) > 0:
        centroids = np.vstack([embedding[refined == c].mean(axis=0)
                               for c in clustered_ids])
        # characteristic radius per cluster = the chosen percentile of member
        # distances to the centroid (the cluster's own spread).
        thresholds = []
        for k, c in enumerate(clustered_ids):
            members = embedding[refined == c]
            dists = np.linalg.norm(members - centroids[k], axis=1)
            thresholds.append(np.percentile(dists, DENOISE_RADIUS_PCTL)
                              if len(dists) else 0.0)
        thresholds = np.asarray(thresholds)

        CH = 100000
        for start in range(0, len(noise_idx), CH):
            chunk = noise_idx[start:start + CH]
            pts = embedding[chunk]
            d = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
            nearest_k = d.argmin(axis=1)
            nearest_dist = d[np.arange(len(chunk)), nearest_k]
            # absorb only points within the nearest cluster's characteristic radius
            accept = nearest_dist <= thresholds[nearest_k]
            assign_to = np.array(clustered_ids)[nearest_k]
            refined[chunk[accept]] = assign_to[accept]
            n_reassigned += int(accept.sum())
        kept_noise = int((refined == -1).sum())
        log.append({"action": "denoise", "cluster": -1,
                    "size": int(len(noise_idx)), "fraction": np.nan,
                    "n_reassigned": int(n_reassigned),
                    "n_kept_as_noise": kept_noise,
                    "radius_percentile": DENOISE_RADIUS_PCTL,
                    "status": "noise_reduced"})

    return refined, pd.DataFrame(log)


if trigger:
    print("\n===== RUNNING REFINEMENT (split dominant + denoise) =====")
    refined_labels, refine_log = refine_split_and_denoise(
        cluster_labels, embedding_full)

    # refined clustering has no native persistence (labels were reassigned),
    # so persistence is left NaN in refined metrics; sizes/fractions/probs
    # remain meaningful. Probabilities for reassigned points are set to NaN.
    refined_probs = cluster_probabilities.copy().astype(float)
    refined_probs[cluster_labels == -1] = np.nan  # reassigned points: unknown prob

    refined_overall, refined_pcm = save_clustering(
        "refined", refined_labels, embedding_full, refined_probs, None)

    # save the refinement action log
    refine_log.to_csv(BASE_OUT / "refined" / "refinement_log.csv", index=False)

    # side-by-side comparison so you can pick the better one
    comparison = pd.DataFrame([
        {"version": "baseline", **baseline_overall},
        {"version": "refined", **refined_overall},
    ])
    comparison.to_csv(BASE_OUT / "baseline_vs_refined_comparison.csv", index=False)
    print("\n===== BASELINE vs REFINED =====")
    print(comparison[["version", "n_clusters", "noise_fraction",
                      "largest_cluster_fraction", "mean_persistence"]].to_string(index=False))
    print(f"\nTarget check (refined): "
          f"noise {refined_overall['noise_fraction']:.3f} "
          f"({'OK' if refined_overall['noise_fraction'] <= NOISE_TRIGGER else 'still high'}), "
          f"largest {refined_overall['largest_cluster_fraction']:.3f} "
          f"({'OK' if refined_overall['largest_cluster_fraction'] <= DOMINANCE_TRIGGER else 'still high'})")
else:
    print("\nBaseline within thresholds - no refinement performed.")


# COMMAND ----------

# DBTITLE 1,Save final outputs to config out_dir (embeddings, labels, persistence)
# ---------------------------------------------------------------------------
# Save the canonical outputs the downstream pipeline stages consume, into the
# config-driven OUT_DIR, keyed by TAG. We save the chosen clustering: the
# refined labels if refinement was triggered, otherwise the baseline labels.
# (Both versions remain available under BASE_OUT/baseline and BASE_OUT/refined
# from the conditional-refinement block above, for comparison.)
# ---------------------------------------------------------------------------
import numpy as np
from pathlib import Path

OUT_DIR = Path(globals().get("OUT_DIR",
            "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = globals().get("TAG",
        "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")

# choose which labels are "final": refined if it ran, else baseline
if "trigger" in dir() and trigger and "refined_labels" in dir():
    final_labels = refined_labels
    print("Saving REFINED clustering as the final output.")
else:
    final_labels = cluster_labels
    print("Saving BASELINE clustering as the final output (no refinement).")

# bring persistence to host (cuML stores it on GPU)
_persist = getattr(final_clusterer, "cluster_persistence_", None)
if _persist is not None:
    try:
        import cupy as cp
        _persist = cp.asnumpy(_persist)
    except Exception:
        _persist = np.asarray(_persist)

# --- save the three canonical artifacts to OUT_DIR, keyed by TAG ---
emb_path   = OUT_DIR / f"umap_embedding_3d__{TAG}.npy"
label_path = OUT_DIR / f"cluster_labels__{TAG}.npy"
prob_path  = OUT_DIR / f"cluster_probabilities__{TAG}.npy"
pers_path  = OUT_DIR / f"cluster_persistence__{TAG}.npy"

np.save(emb_path,   embedding_full)
np.save(label_path, final_labels)
np.save(prob_path,  cluster_probabilities)
if _persist is not None:
    np.save(pers_path, _persist)

# also save a tidy per-cluster persistence table (cluster id + persistence)
if _persist is not None:
    import pandas as pd
    uniq = sorted(c for c in set(int(x) for x in final_labels) if c != -1)
    rows = []
    for c in uniq:
        rows.append({
            "cluster": c,
            "n_windows": int((final_labels == c).sum()),
            "persistence": float(_persist[c]) if c < len(_persist) else float("nan"),
        })
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"cluster_persistence__{TAG}.csv", index=False)

print("Saved to OUT_DIR:")
print("  embedding   ->", emb_path.name)
print("  labels      ->", label_path.name)
print("  probabilities ->", prob_path.name)
if _persist is not None:
    print("  persistence ->", pers_path.name, "(+ .csv table)")
print(f"OUT_DIR: {OUT_DIR}")

