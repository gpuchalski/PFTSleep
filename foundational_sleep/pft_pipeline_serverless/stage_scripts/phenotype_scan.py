# Databricks notebook source
# =============================================================================
# PHENOTYPE SCAN - compute all candidate k with every selection metric
# =============================================================================
# Runs BEFORE the phenotypes stage. For each candidate number of subject
# phenotypes k, computes the standard cluster-count metrics so you can decide
# which k value(s) to put in config.yaml (phenotypes.k_values), which the
# phenotypes stage then uses.
#
# Metrics per k:
#   silhouette          (higher better; >0.5 strong, 0.25-0.5 reasonable, <0.25 weak)
#   calinski_harabasz   (higher better; relative - look for the peak)
#   davies_bouldin      (lower  better; <1 good, look for the minimum)
#   inertia             (lower better; look for the "elbow", not the minimum)
#   gap / gap_sk        (gap higher better; choose smallest k where
#                        gap[k] >= gap[k+1] - gap_sk[k+1])
#
# Injected config names used: OUT_DIR, RANDOM_STATE, TAG, and the scan range
# cfg.phenotypes.k_scan_min .. k_scan_max.
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from scipy.cluster.hierarchy import linkage, dendrogram

OUT_DIR = Path(globals().get("OUT_DIR",
            "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
TAG = globals().get("TAG",
        "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
RANDOM_STATE = int(globals().get("RANDOM_STATE", 42))
FIG_DIR = OUT_DIR / "figures_extended"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# scan range from config (fallback 2..15)
K_MIN = int(getattr(cfg.phenotypes, "k_scan_min", 2)) if "cfg" in dir() else 2
K_MAX = int(getattr(cfg.phenotypes, "k_scan_max", 15)) if "cfg" in dir() else 15
K_RANGE = list(range(K_MIN, K_MAX + 1))

# COMMAND ----------

# DBTITLE 1,Load subject cluster proportions
prop = pd.read_parquet(OUT_DIR / f"subject_cluster_proportions__{TAG}.parquet")
pcols = [c for c in prop.columns if c.endswith("_prop")]
X = prop[pcols].fillna(0).values
Xs = StandardScaler().fit_transform(X)
print(f"Subjects: {X.shape[0]}, cluster-proportion features: {X.shape[1]}")
print(f"Scanning k = {K_MIN}..{K_MAX}")

# COMMAND ----------

# DBTITLE 1,Gap statistic helper
def gap_statistic(Xz, k_range, n_ref=10, seed=42):
    rng = np.random.default_rng(seed)
    mins, maxs = Xz.min(0), Xz.max(0)
    gaps, sks = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xz)
        wk = np.log(km.inertia_ + 1e-12)
        refs = []
        for _ in range(n_ref):
            ref = rng.uniform(mins, maxs, size=Xz.shape)
            refs.append(np.log(
                KMeans(n_clusters=k, n_init=5, random_state=0).fit(ref).inertia_ + 1e-12))
        gaps.append(float(np.mean(refs) - wk))
        sks.append(float(np.std(refs) * np.sqrt(1 + 1 / n_ref)))
    return np.array(gaps), np.array(sks)

# COMMAND ----------

# DBTITLE 1,Compute every metric across all k
sil, ch, db, inertia = [], [], [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    lab = km.labels_
    sil.append(float(silhouette_score(Xs, lab)))
    ch.append(float(calinski_harabasz_score(Xs, lab)))
    db.append(float(davies_bouldin_score(Xs, lab)))
    inertia.append(float(km.inertia_))

gaps, sks = gap_statistic(Xs, K_RANGE, seed=RANDOM_STATE)

scan = pd.DataFrame({
    "k": K_RANGE,
    "silhouette": np.round(sil, 4),
    "calinski_harabasz": np.round(ch, 1),
    "davies_bouldin": np.round(db, 4),
    "inertia": np.round(inertia, 1),
    "gap": np.round(gaps, 4),
    "gap_sk": np.round(sks, 4),
})

# each metric's own suggested k
sil_k = K_RANGE[int(np.argmax(sil))]
ch_k  = K_RANGE[int(np.argmax(ch))]
db_k  = K_RANGE[int(np.argmin(db))]
gap_k = None
for i in range(len(gaps) - 1):
    if gaps[i] >= gaps[i + 1] - sks[i + 1]:
        gap_k = K_RANGE[i]; break

# COMMAND ----------

# DBTITLE 1,Print the scan table + metric legend
print("=" * 78)
print("PHENOTYPE K SCAN - metrics for every candidate k")
print("=" * 78)
print(scan.to_string(index=False))

print("\n" + "-" * 78)
print("METRIC LEGEND - what values are acceptable")
print("-" * 78)
print("""\
silhouette        higher = better separation.
                    > 0.50  strong structure
                    0.25-0.50 reasonable / moderate
                    < 0.25  weak - groups not well separated
calinski_harabasz higher = better. No absolute cutoff; pick the k at the PEAK.
davies_bouldin    lower  = better. < 1.0 is good; pick the k at the MINIMUM.
inertia           lower = tighter, but always falls as k rises. Do NOT pick the
                    minimum; look for the "elbow" where the drop flattens.
gap / gap_sk      higher gap = better. Standard rule: choose the SMALLEST k where
                    gap[k] >= gap[k+1] - gap_sk[k+1]. gap_sk is the error bar.

HOW TO CHOOSE k:
  - Look for AGREEMENT across metrics. If silhouette, CH, and gap all point near
    the same k, that's a strong choice.
  - If they disagree widely, the phenotypes are not sharply separated (the data
    may be more of a continuum) - prefer a small k and report it as exploratory.
  - You may pick MORE THAN ONE k to compare; put them all in
    config.yaml -> phenotypes.k_values, e.g. [3, 4, 5].
""")

print("-" * 78)
print("EACH METRIC'S SUGGESTED k")
print("-" * 78)
print(f"  silhouette (max)        -> k = {sil_k}  (value {max(sil):.3f})")
print(f"  calinski_harabasz (max) -> k = {ch_k}")
print(f"  davies_bouldin (min)    -> k = {db_k}  (value {min(db):.3f})")
print(f"  gap statistic (rule)    -> k = {gap_k}")
print("\n  -> Put your chosen k value(s) into config.yaml: phenotypes.k_values")

# COMMAND ----------

# DBTITLE 1,Save scan table + figures (E15 criteria panel, E16 dendrogram)
scan.to_csv(FIG_DIR / "phenotype_selection_criteria.csv", index=False)

ks = K_RANGE
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
def mark(ax, kbest):
    if kbest is not None: ax.axvline(kbest, color="red", ls="--", alpha=0.6)
axes[0,0].plot(ks, sil, "o-"); axes[0,0].set_title("Silhouette (higher better)"); mark(axes[0,0], sil_k)
axes[0,1].plot(ks, ch, "o-");  axes[0,1].set_title("Calinski-Harabasz (higher better)"); mark(axes[0,1], ch_k)
axes[0,2].plot(ks, db, "o-");  axes[0,2].set_title("Davies-Bouldin (lower better)"); mark(axes[0,2], db_k)
axes[1,0].plot(ks, inertia, "o-"); axes[1,0].set_title("Inertia (elbow)")
axes[1,1].errorbar(ks, gaps, yerr=sks, fmt="o-"); axes[1,1].set_title("Gap statistic"); mark(axes[1,1], gap_k)
axes[1,2].axis("off")
axes[1,2].text(0.02, 0.9,
    f"Suggested k:\n\n silhouette: {sil_k}\n calinski-harabasz: {ch_k}\n"
    f" davies-bouldin: {db_k}\n gap: {gap_k}\n\n"
    "Put chosen k(s) in\n config.yaml phenotypes.k_values",
    fontsize=12, family="monospace", va="top")
for ax in axes.ravel():
    if ax.get_title(): ax.set_xlabel("k (number of phenotypes)")
fig.suptitle("Phenotype-count selection metrics", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "E15_phenotype_count_selection.png", bbox_inches="tight")
plt.close(fig)

Z = linkage(Xs, method="ward")
fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z, truncate_mode="level", p=6, no_labels=True, ax=ax,
           color_threshold=0.7 * max(Z[:, 2]))
ax.set_title("Ward dendrogram of subjects (where you cut = number of phenotypes)")
ax.set_ylabel("merge distance")
fig.savefig(FIG_DIR / "E16_phenotype_dendrogram.png", bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: phenotype_selection_criteria.csv, "
      f"E15_phenotype_count_selection.png, E16_phenotype_dendrogram.png")
print(f"  in {FIG_DIR}")

