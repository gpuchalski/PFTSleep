# Databricks notebook source
# =============================================================================
# PFTSleep Clustering - Extended Exploratory Plots
# =============================================================================
# Exploration-focused. Emphasis on correlation heatmaps, composition heatmaps,
# and temporal / sequence structure. Each plot is independent (try/except) and
# saves to FIG_DIR. Companion to all_cluster_plots.py.
# =============================================================================

# COMMAND ----------

# DBTITLE 1,Install
# MAGIC %pip install plotly matplotlib seaborn scipy scikit-learn pyarrow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config + load
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":200,"font.size":10})

TAG      = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
OUT_DIR  = Path(globals().get("OUT_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
DEMO_CSV = globals().get("DEMO_CSV", "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel.csv")
FIG_DIR  = OUT_DIR / "figures_extended"; FIG_DIR.mkdir(parents=True, exist_ok=True)

def L(loader, path):
    try: return loader(path)
    except Exception as e: print(f"  missing {path.name}: {str(e)[:50]}"); return None

win    = L(pd.read_parquet, OUT_DIR / f"window_clusters__{TAG}.parquet")
scored = L(pd.read_parquet, OUT_DIR / f"window_clusters_scored__{TAG}.parquet")
annot  = L(pd.read_parquet, OUT_DIR / f"cluster_annotation_composition__{TAG}.parquet")
prop   = L(pd.read_parquet, OUT_DIR / f"subject_cluster_proportions__{TAG}.parquet")
demo   = pd.read_csv(DEMO_CSV); demo["nsrrid"] = demo["nsrrid"].astype("int64")

def save(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {name}")

print("Loaded.")

# COMMAND ----------

# DBTITLE 1,E1 - full annotation-composition heatmap (clusters x all annotation types)
try:
    if annot is not None:
        A = annot.copy()
        if "n_windows" in A.columns: A = A.drop(columns=["n_windows"])
        A = A.set_index("cluster") if "cluster" in A.columns else A
        fig, ax = plt.subplots(figsize=(max(10, A.shape[1]*0.5), max(8, len(A)*0.28)))
        sns.heatmap(A, cmap="rocket_r", ax=ax, cbar_kws={"label":"% of cluster windows"})
        ax.set_title("Annotation composition per cluster (% of windows)")
        ax.set_xlabel("annotation type"); ax.set_ylabel("cluster")
        save(fig, "E01_annotation_composition_heatmap.png")
except Exception as _e:
    print(f"  E1 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E2 - annotation composition, column-normalized (which cluster owns each annotation)
try:
    if annot is not None:
        A = annot.copy()
        if "n_windows" in A.columns: A = A.drop(columns=["n_windows"])
        A = A.set_index("cluster") if "cluster" in A.columns else A
        # normalize each annotation column to sum to 100 across clusters
        Acol = A.div(A.sum(axis=0).clip(lower=1e-9), axis=1) * 100
        fig, ax = plt.subplots(figsize=(max(10, A.shape[1]*0.5), max(8, len(A)*0.28)))
        sns.heatmap(Acol, cmap="mako_r", ax=ax,
                    cbar_kws={"label":"% of this annotation's windows in cluster"})
        ax.set_title("Where each annotation type concentrates (column-normalized)")
        save(fig, "E02_annotation_column_normalized.png")
except Exception as _e:
    print(f"  E2 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E3 - cluster proportion correlation matrix (which clusters co-occur in subjects)
try:
    if prop is not None:
        pcols = [c for c in prop.columns if c.endswith("_prop")]
        C = prop[pcols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(9,len(C)*0.3), max(8,len(C)*0.3)))
        sns.heatmap(C, cmap="coolwarm", center=0, ax=ax, square=True,
                    cbar_kws={"label":"Spearman r between cluster proportions"})
        ax.set_title("Cluster co-occurrence: correlation of per-subject proportions")
        save(fig, "E03_cluster_cooccurrence_corr.png")
except Exception as _e:
    print(f"  E3 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E4 - clustered correlation matrix (same as E3 but hierarchically ordered)
try:
    if prop is not None:
        pcols = [c for c in prop.columns if c.endswith("_prop")]
        C = prop[pcols].corr(method="spearman").fillna(0)
        g = sns.clustermap(C, cmap="coolwarm", center=0, figsize=(12,12),
                           cbar_kws={"label":"Spearman r"})
        g.fig.suptitle("Cluster co-occurrence (hierarchically clustered)", y=1.02)
        g.savefig(FIG_DIR / "E04_cluster_cooccurrence_clustered.png", bbox_inches="tight")
        plt.close(g.fig); print("  saved E04_cluster_cooccurrence_clustered.png")
except Exception as _e:
    print(f"  E4 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E5 - cluster proportions vs FULL demographic panel (big correlation heatmap)
try:
    if prop is not None:
        pcols = [c for c in prop.columns if c.endswith("_prop")]
        demo_vars = ["age_s1","bmi_s1","ahi_a0h3a","ahi_a0h4_s1","HypoxicBurden",
                     "pctlt90","MinSat","NDes3pH","hdl_s1","chol_s1","trig_s1",
                     "ess_s1","slptime","total_ap_hyp_count"]
        demo_vars = [v for v in demo_vars if v in demo.columns]
        m = prop.merge(demo[["nsrrid"]+demo_vars], on="nsrrid")
        for v in demo_vars: m[v] = pd.to_numeric(m[v], errors="coerce")
        R = pd.DataFrame(index=pcols, columns=demo_vars, dtype=float)
        from scipy.stats import spearmanr
        for pc in pcols:
            for v in demo_vars:
                s = m[[pc,v]].dropna()
                R.loc[pc,v] = spearmanr(s[pc], s[v])[0] if len(s)>30 else np.nan
        R = R.astype(float)
        fig, ax = plt.subplots(figsize=(max(10,len(demo_vars)*0.7), max(8,len(pcols)*0.3)))
        sns.heatmap(R, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label":"Spearman r"})
        ax.set_title("Cluster proportions vs full demographic / severity panel")
        save(fig, "E05_prop_vs_full_demographics.png")
except Exception as _e:
    print(f"  E5 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E6 - cluster prevalence across the night (time-of-night dynamics)
try:
    if win is not None:
        w = win[win["cluster"]>=0].copy()
        # bin time-of-night into 5-min bins; fraction of windows in each cluster per bin
        w["tbin"] = (w["time_idx"] // 300).astype(int)   # 300s = 5 min
        # focus on the top 12 clusters by size for readability
        top = w["cluster"].value_counts().head(12).index.tolist()
        wt = w[w["cluster"].isin(top)]
        ct = pd.crosstab(wt["tbin"], wt["cluster"], normalize="index") * 100
        ct = ct[ct.index < 96]   # first 8 hours (96 * 5min)
        fig, ax = plt.subplots(figsize=(15, 6))
        for c in top:
            if c in ct.columns:
                ax.plot(ct.index*5/60, ct[c], label=f"C{c}", lw=1.5)
        ax.set_xlabel("hours from lights-off"); ax.set_ylabel("% of windows in bin")
        ax.set_title("Cluster prevalence across the night (top 12 clusters)")
        ax.legend(ncol=2, fontsize=7, loc="upper right")
        save(fig, "E06_cluster_prevalence_over_night.png")
except Exception as _e:
    print(f"  E6 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E7 - cluster prevalence heatmap (time-of-night x cluster)
try:
    if win is not None:
        w = win[win["cluster"]>=0].copy()
        w["tbin"] = (w["time_idx"] // 600).astype(int)   # 10-min bins
        ct = pd.crosstab(w["tbin"], w["cluster"], normalize="index") * 100
        ct = ct[ct.index < 48]   # first 8 hours
        fig, ax = plt.subplots(figsize=(max(10, ct.shape[1]*0.3), 8))
        sns.heatmap(ct.T, cmap="viridis", ax=ax,
                    cbar_kws={"label":"% of windows in time-bin"})
        ax.set_xlabel("time bin (10-min, from lights-off)"); ax.set_ylabel("cluster")
        ax.set_title("Cluster prevalence over the night (heatmap)")
        save(fig, "E07_prevalence_heatmap.png")
except Exception as _e:
    print(f"  E7 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E8 - transition probability matrix (row-normalized)
try:
    if win is not None:
        w = win.sort_values(["nsrrid","time_idx"])
        cur = w["cluster"].values; nxt = np.roll(cur,-1)
        ss = w["nsrrid"].values == np.roll(w["nsrrid"].values,-1)
        mask = ss & (cur>=0) & (nxt>=0)
        cur_m, nxt_m = cur[mask], nxt[mask]
        cl = sorted(np.unique(np.concatenate([cur_m,nxt_m])))
        idx = {c:i for i,c in enumerate(cl)}
        T = np.zeros((len(cl),len(cl)))
        for a,b in zip(cur_m,nxt_m): T[idx[a],idx[b]] += 1
        Tn = T / T.sum(axis=1,keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(max(10,len(cl)*0.35), max(9,len(cl)*0.35)))
        sns.heatmap(Tn, cmap="magma", ax=ax, xticklabels=cl, yticklabels=cl,
                    cbar_kws={"label":"P(next | current)"})
        ax.set_xlabel("next cluster"); ax.set_ylabel("current cluster")
        ax.set_title("Cluster transition probabilities")
        save(fig, "E08_transition_probabilities.png")
except Exception as _e:
    print(f"  E8 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E9 - self-transition (stickiness) and mean dwell time per cluster
try:
    if win is not None:
        w = win.sort_values(["nsrrid","time_idx"])
        cur = w["cluster"].values; nxt = np.roll(cur,-1)
        ss = w["nsrrid"].values == np.roll(w["nsrrid"].values,-1)
        mask = ss & (cur>=0)
        # stickiness = P(stay in same cluster)
        stick = {}
        for c in np.unique(cur[cur>=0]):
            m = mask & (cur==c)
            if m.sum()>0: stick[c] = (nxt[m]==c).mean()
        s = pd.Series(stick).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14,5))
        ax.bar(s.index.astype(str), s.values*100, color="#6a51a3")
        ax.set_ylabel("% self-transition (stickiness)")
        ax.set_xlabel("cluster"); ax.set_title("Cluster stickiness: P(next window = same cluster)")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "E09_cluster_stickiness.png")
except Exception as _e:
    print(f"  E9 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E10 - first-half vs second-half of night: cluster shift
try:
    if win is not None:
        w = win[win["cluster"]>=0].copy()
        # per subject, split each night at its own midpoint
        w["half"] = w.groupby("nsrrid")["time_idx"].transform(
            lambda s: (s > s.median()).astype(int))   # 0=first half, 1=second
        h0 = w[w["half"]==0]["cluster"].value_counts(normalize=True)*100
        h1 = w[w["half"]==1]["cluster"].value_counts(normalize=True)*100
        comp = pd.DataFrame({"first_half":h0,"second_half":h1}).fillna(0)
        comp["shift"] = comp["second_half"] - comp["first_half"]
        comp = comp.sort_values("shift")
        fig, ax = plt.subplots(figsize=(14,5))
        colors = ["#c0504d" if v<0 else "#4f81bd" for v in comp["shift"]]
        ax.bar(comp.index.astype(str), comp["shift"], color=colors)
        ax.axhline(0,color="black",lw=0.8)
        ax.set_ylabel("Δ % (second half − first half)")
        ax.set_xlabel("cluster")
        ax.set_title("Cluster shift: second half vs first half of night")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "E10_first_vs_second_half.png")
except Exception as _e:
    print(f"  E10 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E13 - stage x cluster, but as enrichment (obs/expected) not raw %
try:
    if scored is not None:
        sc = scored[scored["stage"]!="Unknown"]
        obs = pd.crosstab(sc["cluster"], sc["stage"])
        # expected under independence
        row = obs.sum(axis=1); col = obs.sum(axis=0); tot = obs.values.sum()
        exp = np.outer(row, col)/tot
        enr = np.log2((obs.values+1)/(exp+1))   # log2 obs/exp, smoothed
        E = pd.DataFrame(enr, index=obs.index, columns=obs.columns)
        fig, ax = plt.subplots(figsize=(10, max(8,len(E)*0.28)))
        sns.heatmap(E, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label":"log2(observed/expected)"})
        ax.set_title("Cluster x stage ENRICHMENT (red = over-represented)")
        save(fig, "E13_stage_enrichment.png")
except Exception as _e:
    print(f"  E13 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E14 - per-subject cluster entropy (sleep fragmentation proxy) vs AHI
try:
    if prop is not None:
        pcols = [c for c in prop.columns if c.endswith("_prop") and "noise" not in c
                 and "-1" not in c]
        P = prop[pcols].values
        Pn = P / P.sum(axis=1, keepdims=True).clip(min=1e-9)
        ent = -(Pn * np.log(Pn + 1e-12)).sum(axis=1)   # Shannon entropy
        e = prop[["nsrrid"]].copy(); e["cluster_entropy"] = ent
        m = e.merge(demo[["nsrrid","ahi_a0h3a"]], on="nsrrid")
        m["ahi_a0h3a"] = pd.to_numeric(m["ahi_a0h3a"], errors="coerce")
        m = m.dropna()
        from scipy.stats import spearmanr
        r,p = spearmanr(m["cluster_entropy"], m["ahi_a0h3a"])
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(m["cluster_entropy"], m["ahi_a0h3a"], s=10, alpha=0.4, color="#3a6ea5")
        ax.set_xlabel("cluster entropy (higher = more varied night)")
        ax.set_ylabel("AHI"); ax.set_title(f"Cluster diversity vs AHI (rho={r:.2f}, p={p:.1e})")
        save(fig, "E14_entropy_vs_ahi.png")
        e.to_parquet(FIG_DIR / "subject_cluster_entropy.parquet", index=False)
except Exception as _e:
    print(f"  E14 FAILED: {type(_e).__name__}: {str(_e)[:120]}")

# COMMAND ----------

# DBTITLE 1,E15 - summary of what was produced
try:
    import os
    print("Extended figures written to:", FIG_DIR)
    for f in sorted(os.listdir(FIG_DIR)):
        print("  ", f)

except Exception as _e:
    print(f"  [skip] E15 failed: {type(_e).__name__}: {str(_e)[:100]}")
