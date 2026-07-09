# Databricks notebook source
# =============================================================================
# PHENOTYPES - assign subjects to phenotype groups at the chosen k value(s)
# =============================================================================
# Runs AFTER phenotype_scan. Uses the k value(s) you chose (based on the scan's
# metrics + legend) from config.yaml -> phenotypes.k_values. For each chosen k
# it saves the per-subject phenotype labels and a profile heatmap.
#
# The k-selection metrics (silhouette/CH/Davies-Bouldin/inertia/gap) and the
# E15/E16 figures are produced by the phenotype_scan stage - not here. This
# stage only ASSIGNS at the k(s) you settled on.
# =============================================================================

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUT_DIR = Path(globals().get("OUT_DIR",
            "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
TAG = globals().get("TAG",
        "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
RANDOM_STATE = int(globals().get("RANDOM_STATE", 42))
FIG_DIR = OUT_DIR / "figures_extended"; FIG_DIR.mkdir(parents=True, exist_ok=True)
PHENO_DIR = FIG_DIR / "E11"; PHENO_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = list(cfg.phenotypes.k_values) if "cfg" in dir() else [3, 4, 5]

prop = pd.read_parquet(OUT_DIR / f"subject_cluster_proportions__{TAG}.parquet")
pcols = [c for c in prop.columns if c.endswith("_prop")]
X = prop[pcols].fillna(0).values
Xs = StandardScaler().fit_transform(X)

print(f"Assigning phenotypes at k = {K_VALUES} "
      f"(from config.yaml phenotypes.k_values).")
print("If you have not reviewed the phenotype_scan output yet, do that first "
      "to choose these k values.")

# COMMAND ----------

# DBTITLE 1,Assign phenotypes at each chosen k + profile heatmap
for k in K_VALUES:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Xs)
    out = prop[["nsrrid"]].copy()
    out["phenotype"] = km.labels_
    out.to_parquet(PHENO_DIR / f"E11_subject_phenotypes_{k}.parquet", index=False)

    profile = prop.set_index("nsrrid")[pcols].copy()
    profile["phenotype"] = km.labels_
    prof = profile.groupby("phenotype")[pcols].mean()
    fig, ax = plt.subplots(figsize=(max(10, len(pcols) * 0.3), max(4, k * 0.5)))
    sns.heatmap(prof, cmap="viridis", ax=ax, cbar_kws={"label": "mean proportion"})
    ax.set_title(f"Phenotype profiles (k={k})")
    ax.set_xlabel("cluster"); ax.set_ylabel("phenotype")
    fig.savefig(FIG_DIR / f"E11b_phenotype_profiles_chosen_k_{k}.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  k={k}: sizes {pd.Series(km.labels_).value_counts().sort_index().to_dict()}")

print("Phenotype assignment complete. Files in:", PHENO_DIR)

