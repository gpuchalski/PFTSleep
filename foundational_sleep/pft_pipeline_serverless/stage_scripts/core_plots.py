# Databricks notebook source
# =============================================================================
# PFTSleep Clustering - 20 Visualizations
# =============================================================================
# Reads the saved clustering outputs and produces 20 plots covering cluster
# structure, physiology (sleep stages / respiratory events), patient
# characteristics, and cardiovascular outcomes. Each plot saves to FIG_DIR as
# PNG (static) or HTML (interactive).
#
# Inputs expected in OUT_DIR (produced by the earlier analysis scripts):
#   cluster_labels__{TAG}.npy
#   cluster_probabilities__{TAG}.npy
#   umap_embedding_3d__{TAG}.npy
#   cluster_persistence__{TAG}.parquet           (optional)
#   window_clusters__{TAG}.parquet
#   window_clusters_scored__{TAG}.parquet
#   cluster_annotation_composition__{TAG}.parquet
#   cluster_demographic_stats__{TAG}.parquet
#   subject_cluster_proportions__{TAG}.parquet
#   cluster_severity_correlations__{TAG}.parquet
#   cluster_cvd_logistic__{TAG}.parquet
# plus the demographics CSV for outcomes.
# =============================================================================

# COMMAND ----------

# DBTITLE 1,Install
# MAGIC %pip install lifelines plotly matplotlib seaborn scipy scikit-learn -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config + load everything available
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200,
                     "font.size": 10, "axes.titlesize": 12,
                     "figure.autolayout": True})

TAG      = globals().get("TAG", "PFTSleep__files1219__freq125__win750__hop750__max28800__concat7ch")
OUT_DIR  = Path(globals().get("OUT_DIR", "/Volumes/kumc_sleep/sleep_studies/shhs_data/clustering_outputs_new/A100"))
DEMO_CSV = globals().get("DEMO_CSV", "/Volumes/kumc_sleep/sleep_studies/shhs_data/excel_sheet/sleep_excel.csv")
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CVD_EVENT = "inci_cvd_all_01"
CVD_TIME  = "inci_cvd_all_time"

def _try_load(loader, path, label):
    try:
        obj = loader(path)
        print(f"  loaded {label}: {path.name}")
        return obj
    except Exception as e:
        print(f"  MISSING {label}: {path.name}  ({str(e)[:50]})")
        return None

print("Loading inputs...")
labels   = _try_load(np.load, OUT_DIR / f"cluster_labels__{TAG}.npy", "labels")
probs    = _try_load(np.load, OUT_DIR / f"cluster_probabilities__{TAG}.npy", "probs")
emb      = _try_load(np.load, OUT_DIR / f"umap_embedding_3d__{TAG}.npy", "embedding")
win      = _try_load(pd.read_parquet, OUT_DIR / f"window_clusters__{TAG}.parquet", "window table")
scored   = _try_load(pd.read_parquet, OUT_DIR / f"window_clusters_scored__{TAG}.parquet", "scored windows")
annot    = _try_load(pd.read_parquet, OUT_DIR / f"cluster_annotation_composition__{TAG}.parquet", "annotation comp")
demostat = _try_load(pd.read_parquet, OUT_DIR / f"cluster_demographic_stats__{TAG}.parquet", "demo stats")
prop     = _try_load(pd.read_parquet, OUT_DIR / f"subject_cluster_proportions__{TAG}.parquet", "subject props")
corr     = _try_load(pd.read_parquet, OUT_DIR / f"cluster_severity_correlations__{TAG}.parquet", "severity corr")
cvd      = _try_load(pd.read_parquet, OUT_DIR / f"cluster_cvd_logistic__{TAG}.parquet", "cvd logistic")
try:
    persist = pd.read_parquet(OUT_DIR / f"cluster_persistence__{TAG}.parquet")
    print("  loaded persistence")
except Exception:
    persist = None
    print("  persistence not found (plot 4 will be skipped)")

demo = pd.read_csv(DEMO_CSV)
demo["nsrrid"] = demo["nsrrid"].astype(np.int64)

# A consistent categorical palette big enough for many clusters.
def cluster_palette(n):
    base = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) +
            list(plt.cm.tab20c.colors))
    return [base[i % len(base)] for i in range(n)]

def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")

print("Ready.")

# COMMAND ----------

# DBTITLE 1,Subsample helper for scatter plots (5.2M points won't render)
def stratified_subsample(emb_arr, lab_arr, per=800, drop_noise=True):
    idx_keep = []
    rng = np.random.default_rng(42)
    for c in np.unique(lab_arr):
        if drop_noise and c == -1:
            continue
        idx = np.where(lab_arr == c)[0]
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        idx_keep.append(idx)
    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)
    return idx_keep

# COMMAND ----------

# DBTITLE 1,Plot 1 - 3D UMAP scatter colored by cluster (interactive HTML)
try:
    import plotly.express as px
    if emb is not None and labels is not None:
        k = stratified_subsample(emb, labels, per=800)
        df = pd.DataFrame({"UMAP1": emb[k,0], "UMAP2": emb[k,1], "UMAP3": emb[k,2],
                           "cluster": labels[k].astype(str)})
        fig = px.scatter_3d(df, x="UMAP1", y="UMAP2", z="UMAP3", color="cluster",
                            opacity=0.6, title="UMAP 3D embedding by cluster")
        fig.update_traces(marker=dict(size=2))
        fig.write_html(FIG_DIR / "01_umap_3d.html")
        print("  saved 01_umap_3d.html")

except Exception as _e:
    print(f"  [skip] Plot 1 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 2 - 2D UMAP scatter (three axis-pairs)
try:
    if emb is not None and labels is not None:
        k = stratified_subsample(emb, labels, per=800)
        pal = cluster_palette(len(np.unique(labels[labels>=0])))
        cmap = {c: pal[i] for i, c in enumerate(sorted(np.unique(labels[labels>=0])))}
        cols = [cmap[c] for c in labels[k]]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (i, j, t) in zip(axes, [(0,1,"1 vs 2"),(0,2,"1 vs 3"),(1,2,"2 vs 3")]):
            ax.scatter(emb[k,i], emb[k,j], c=cols, s=2, alpha=0.5, linewidths=0)
            ax.set_title(f"UMAP axes {t}"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("UMAP 2D projections (colored by cluster)")
        save(fig, "02_umap_2d_pairs.png")

except Exception as _e:
    print(f"  [skip] Plot 2 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 3 - cluster size bar chart (log scale)
try:
    if win is not None:
        sizes = win[win["cluster"]>=0]["cluster"].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(sizes)), sizes.values, color="#3a6ea5")
        ax.set_yscale("log")
        ax.set_xlabel("cluster (sorted by size)"); ax.set_ylabel("windows (log)")
        ax.set_title("Cluster sizes")
        save(fig, "03_cluster_sizes.png")

except Exception as _e:
    print(f"  [skip] Plot 3 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 4 - cluster persistence bar chart
try:
    if persist is not None:
        p = persist.sort_values("persistence", ascending=False)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(p["cluster"].astype(str), p["persistence"], color="#2c8c5a")
        ax.set_xlabel("cluster"); ax.set_ylabel("persistence")
        ax.set_title("Cluster persistence (higher = more stable)")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "04_cluster_persistence.png")

except Exception as _e:
    print(f"  [skip] Plot 4 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 5 - assignment-probability histograms per cluster (top 12)
try:
    if win is not None and probs is not None:
        wp = win.copy(); wp["prob"] = probs
        top = wp[wp["cluster"]>=0]["cluster"].value_counts().index
        n = len(top)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        axes = axes.ravel()
        for i, c in enumerate(top):
            ax = axes[i]
            ax.hist(wp[wp["cluster"]==c]["prob"], bins=30, color="#3a6ea5")
            ax.set_title(f"cluster {c}", fontsize=9); ax.set_xlim(0,1)
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Assignment-probability distribution (all clusters)")
        save(fig, "05_probability_histograms.png")

except Exception as _e:
    print(f"  [skip] Plot 5 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 6 - cluster x sleep-stage heatmap
try:
    STAGE_ORDER = ["Wake", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "REM"]
    if scored is not None:
        sc = scored[scored["stage"] != "Unknown"]
        ct = pd.crosstab(sc["cluster"], sc["stage"], normalize="index") * 100
        cols = [c for c in STAGE_ORDER if c in ct.columns] + \
               [c for c in ct.columns if c not in STAGE_ORDER]
        ct = ct[cols]
        fig, ax = plt.subplots(figsize=(10, max(6, len(ct)*0.25)))
        sns.heatmap(ct, cmap="YlGnBu", annot=False, ax=ax, cbar_kws={"label": "% of cluster"})
        ax.set_title("Cluster composition by sleep stage (%)")
        ax.set_xlabel("sleep stage"); ax.set_ylabel("cluster")
        save(fig, "06_cluster_stage_heatmap.png")

except Exception as _e:
    print(f"  [skip] Plot 6 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 7 - stacked bar: stage composition per cluster
try:
    if scored is not None:
        sc = scored[scored["stage"] != "Unknown"]
        ct = pd.crosstab(sc["cluster"], sc["stage"], normalize="index") * 100
        cols = [c for c in STAGE_ORDER if c in ct.columns] + \
               [c for c in ct.columns if c not in STAGE_ORDER]
        ct = ct[cols]
        fig, ax = plt.subplots(figsize=(14, 6))
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Spectral", width=0.9)
        ax.set_ylabel("% of cluster windows"); ax.set_title("Stage composition per cluster")
        ax.legend(title="stage", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "07_stage_stacked_bar.png")

except Exception as _e:
    print(f"  [skip] Plot 7 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 8 - respiratory-event rate per cluster
try:
    if scored is not None and "resp_event" in scored.columns:
        rate = scored.groupby("cluster")["resp_event"].mean().sort_values(ascending=False) * 100
        rate = rate[rate.index >= 0]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(rate.index.astype(str), rate.values, color="#c0504d")
        ax.set_ylabel("% windows overlapping a respiratory event")
        ax.set_title("Respiratory-event rate by cluster")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "08_resp_event_rate.png")

except Exception as _e:
    print(f"  [skip] Plot 8 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 9 - event enrichment (observed / expected)
try:
    if scored is not None and "resp_event" in scored.columns:
        base = scored["resp_event"].mean()
        rate = scored[scored["cluster"]>=0].groupby("cluster")["resp_event"].mean()
        enr = (rate / base).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 5))
        colors_e = ["#c0504d" if v > 1 else "#9bbb59" for v in enr.values]
        ax.bar(enr.index.astype(str), enr.values, color=colors_e)
        ax.axhline(1.0, color="black", lw=1, ls="--")
        ax.set_ylabel("event rate / cohort baseline")
        ax.set_title(f"Respiratory-event enrichment (baseline={base*100:.1f}%)")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        save(fig, "09_event_enrichment.png")

except Exception as _e:
    print(f"  [skip] Plot 9 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 10 - Sankey: cluster -> dominant stage
try:
    import plotly.graph_objects as go
    if scored is not None:
        sc = scored[scored["stage"] != "Unknown"]
        dom = (sc.groupby("cluster")["stage"]
                 .agg(lambda s: s.value_counts().idxmax()))
        clusters = [f"C{c}" for c in dom.index]
        stages = sorted(sc["stage"].unique())
        nodes = clusters + stages
        nidx = {n: i for i, n in enumerate(nodes)}
        src = [nidx[f"C{c}"] for c in dom.index]
        tgt = [nidx[dom[c]] for c in dom.index]
        val = [int((sc["cluster"]==c).sum()) for c in dom.index]
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, pad=12, thickness=14),
            link=dict(source=src, target=tgt, value=val)))
        fig.update_layout(title_text="Cluster -> dominant sleep stage", font_size=9)
        fig.write_html(FIG_DIR / "10_sankey_cluster_stage.html")
        print("  saved 10_sankey_cluster_stage.html")

except Exception as _e:
    print(f"  [skip] Plot 10 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 11 - cluster x demographic heatmap (z-scored vs cohort)
try:
    DEMO_VARS = ["age_s1", "bmi_s1", "ahi_a0h3a", "HypoxicBurden", "pctlt90",
                 "MinSat", "hdl_s1", "chol_s1", "trig_s1"]
    if demostat is not None:
        mean_cols = [f"{v}_mean" for v in DEMO_VARS if f"{v}_mean" in demostat.columns]
        M = demostat.set_index("cluster")[mean_cols].copy()
        M.columns = [c.replace("_mean", "") for c in M.columns]
        # z-score each column across clusters so colors are comparable
        Z = (M - M.mean()) / M.std()
        fig, ax = plt.subplots(figsize=(10, max(6, len(Z)*0.25)))
        sns.heatmap(Z, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label": "z-score vs cohort"})
        ax.set_title("Cluster demographic profile (z-scored cluster means)")
        ax.set_xlabel("variable"); ax.set_ylabel("cluster")
        save(fig, "11_demographic_heatmap.png")

except Exception as _e:
    print(f"  [skip] Plot 11 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 12 - AHI distribution by dominant cluster (box/violin)
try:
    if prop is not None:
        import os
        cluster_plot_dir = FIG_DIR / "ahi_by_cluster"
        cluster_plot_dir.mkdir(parents=True, exist_ok=True)
        pcols = [c for c in prop.columns if c.endswith("_prop")]
        nz = [c for c in pcols if "clust_-1" not in c and "noise" not in c]
        dom_cluster = prop[nz].idxmax(axis=1).str.extract(r"clust_(\-?\d+)_prop")[0]
        dsub = pd.DataFrame({"nsrrid": prop["nsrrid"], "dom": dom_cluster})
        dsub = dsub.merge(demo[["nsrrid", "ahi_a0h3a"]], on="nsrrid")
        dsub["ahi_a0h3a"] = pd.to_numeric(dsub["ahi_a0h3a"], errors="coerce")
        keep = dsub["dom"].value_counts()[lambda s: s >= 10].index
        dsub = dsub[dsub["dom"].isin(keep)].dropna()
        order = sorted(dsub["dom"].unique(), key=lambda x: int(x))
        for cl in order:
            subcl = dsub[dsub["dom"] == cl]
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.violinplot(data=subcl, x="dom", y="ahi_a0h3a", order=[cl], ax=ax,
                           inner="quartile", cut=0)
            ax.set_xlabel("dominant cluster"); ax.set_ylabel("AHI (a0h3a)")
            ax.set_title(f"AHI distribution for dominant cluster {cl}")
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            fig.savefig(cluster_plot_dir / f"ahi_by_dominant_cluster_{cl}.png", bbox_inches="tight")
            plt.close(fig)
        print(f"  saved per-cluster plots to {cluster_plot_dir}")

except Exception as _e:
    print(f"  [skip] Plot 12 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 13 - scatter: cluster proportion vs HypoxicBurden (top 6 by |corr|)
try:
    if prop is not None and corr is not None:
        m = prop.merge(demo[["nsrrid", "HypoxicBurden"]], on="nsrrid")
        m["HypoxicBurden"] = pd.to_numeric(m["HypoxicBurden"], errors="coerce")
        hb = corr[corr["target"] == "HypoxicBurden"].copy()
        hb["abs_rho"] = hb["spearman_rho"].abs()
        top = hb.sort_values("abs_rho", ascending=False).head(6)["cluster_prop"].tolist()
        top = [c for c in top if c in m.columns]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, pc in zip(axes.ravel(), top):
            sub = m[[pc, "HypoxicBurden"]].dropna()
            ax.scatter(sub[pc], sub["HypoxicBurden"], s=8, alpha=0.4, color="#3a6ea5")
            ax.set_title(pc, fontsize=9); ax.set_xlabel("proportion"); ax.set_ylabel("HB")
        fig.suptitle("Cluster proportion vs Hypoxic Burden (top |correlation|)")
        save(fig, "13_prop_vs_hypoxicburden.png")

except Exception as _e:
    print(f"  [skip] Plot 13 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 14 - correlation heatmap: cluster proportions x severity measures
try:
    if corr is not None:
        piv = corr.pivot(index="cluster_prop", columns="target", values="spearman_rho")
        fig, ax = plt.subplots(figsize=(10, max(6, len(piv)*0.25)))
        sns.heatmap(piv, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label": "Spearman rho"})
        ax.set_title("Cluster-proportion correlations with severity measures")
        save(fig, "14_correlation_heatmap.png")

except Exception as _e:
    print(f"  [skip] Plot 14 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 15 - forest plot of CVD odds ratios (HEADLINE)
try:
    if cvd is not None:
        c = cvd.dropna(subset=["odds_ratio"]).copy()
        c = c.sort_values("odds_ratio")
        fig, ax = plt.subplots(figsize=(9, max(6, len(c)*0.3)))
        y = range(len(c))
        ax.errorbar(c["odds_ratio"], y,
                    xerr=[c["odds_ratio"]-c["ci_low"], c["ci_high"]-c["odds_ratio"]],
                    fmt="o", color="#1a3a5c", ecolor="#888", capsize=3, ms=5)
        ax.axvline(1.0, color="red", ls="--", lw=1)
        ax.set_yticks(list(y)); ax.set_yticklabels(c["cluster_prop"], fontsize=7)
        ax.set_xlabel("Odds ratio for incident CVD (adjusted for AHI, age, BMI)")
        ax.set_title("Cluster proportion vs incident CVD - forest plot")
        save(fig, "15_cvd_forest_plot.png")

except Exception as _e:
    print(f"  [skip] Plot 15 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 16 - Kaplan-Meier by cluster-time tertile (top CVD clusters)
try:
    from lifelines import KaplanMeierFitter
    if prop is not None and cvd is not None and CVD_EVENT in demo.columns:
        m = prop.merge(demo[["nsrrid", CVD_EVENT, CVD_TIME]], on="nsrrid")
        m[CVD_EVENT] = pd.to_numeric(m[CVD_EVENT], errors="coerce")
        m[CVD_TIME]  = pd.to_numeric(m[CVD_TIME], errors="coerce")
        # pick the 4 clusters with most extreme (lowest p) CVD association
        cand = cvd.dropna(subset=["p_value"]).sort_values("p_value").head(4)["cluster_prop"]
        cand = [c for c in cand if c in m.columns]
        if len(cand) == 0:
            print("  [skip] Plot 16: no candidate CVD clusters available")
        else:
            fig, axes = plt.subplots(1, len(cand), figsize=(5*len(cand), 5), squeeze=False)
            for ax, pc in zip(axes[0], cand):
                sub = m[[pc, CVD_EVENT, CVD_TIME]].dropna()
                # tertiles of time-in-cluster
                try:
                    sub["grp"] = pd.qcut(sub[pc], 3, labels=["low","med","high"], duplicates="drop")
                except ValueError:
                    sub["grp"] = pd.cut(sub[pc], 3, labels=["low","med","high"])
                kmf = KaplanMeierFitter()
                for g in ["low","med","high"]:
                    d = sub[sub["grp"]==g]
                    if len(d) > 5:
                        kmf.fit(d[CVD_TIME], d[CVD_EVENT], label=f"{g} (n={len(d)})")
                        kmf.plot_survival_function(ax=ax, ci_show=False)
                ax.set_title(pc, fontsize=9); ax.set_xlabel("time"); ax.set_ylabel("CVD-free")
            fig.suptitle("CVD-free survival by cluster-time tertile")
            save(fig, "16_kaplan_meier.png")

except Exception as _e:
    print(f"  [skip] Plot 16 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 17 - volcano: effect size vs significance
try:
    if cvd is not None:
        c = cvd.dropna(subset=["odds_ratio", "p_value"]).copy()
        c["log_or"] = np.log2(c["odds_ratio"].replace(0, np.nan))
        c["neglogp"] = -np.log10(c["p_value"].replace(0, 1e-300))
        fig, ax = plt.subplots(figsize=(9, 7))
        sig = c["p_fdr_bh"] < 0.05 if "p_fdr_bh" in c.columns else c["p_value"] < 0.05
        ax.scatter(c.loc[~sig,"log_or"], c.loc[~sig,"neglogp"], c="#999", s=25, label="ns")
        ax.scatter(c.loc[sig,"log_or"],  c.loc[sig,"neglogp"],  c="#c0504d", s=35, label="FDR<0.05")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axhline(-np.log10(0.05), color="blue", lw=0.8, ls=":")
        ax.set_xlabel("log2(odds ratio)"); ax.set_ylabel("-log10(p)")
        ax.set_title("Volcano: cluster CVD effect vs significance"); ax.legend()
        save(fig, "17_volcano.png")

except Exception as _e:
    print(f"  [skip] Plot 17 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 18 - cluster timeline (hypnogram-style) for one patient
try:
    if win is not None:
        example = win["nsrrid"].iloc[0]
        one = win[win["nsrrid"]==example].sort_values("time_idx")
        fig, ax = plt.subplots(figsize=(16, 3))
        # color each window by cluster
        uniq = sorted(one["cluster"].unique())
        pal = cluster_palette(len(uniq)); cmap = {c: pal[i] for i,c in enumerate(uniq)}
        ax.scatter(one["time_idx"]/3600, one["cluster"],
                   c=[cmap[c] for c in one["cluster"]], s=4)
        ax.set_xlabel("hours from lights-off"); ax.set_ylabel("cluster")
        ax.set_title(f"Cluster timeline across the night - patient {example}")
        save(fig, "18_patient_timeline.png")

except Exception as _e:
    print(f"  [skip] Plot 18 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 19 - patient x cluster-proportion heatmap (rows clustered)
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    if prop is not None:
        pcols = [c for c in prop.columns if c.endswith("_prop")]
        P = prop.set_index("nsrrid")[pcols].fillna(0).values
        # order rows by hierarchical clustering so similar patients group together
        try:
            order = leaves_list(linkage(P, method="ward"))
        except Exception:
            order = np.arange(len(P))
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(P[order], cmap="magma", ax=ax,
                    cbar_kws={"label": "proportion of night"})
        ax.set_xlabel("cluster"); ax.set_ylabel("patients (hierarchically ordered)")
        ax.set_yticks([])
        ax.set_title("Patient x cluster-proportion phenotype map")
        save(fig, "19_patient_phenotype_heatmap.png")

except Exception as _e:
    print(f"  [skip] Plot 19 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Plot 20 - cluster transition matrix
try:
    if win is not None:
        w = win.sort_values(["nsrrid", "time_idx"]).copy()
        cur = w["cluster"].values
        nxt = np.roll(cur, -1)
        same_subj = w["nsrrid"].values == np.roll(w["nsrrid"].values, -1)
        # only count transitions within the same subject, between non-noise states
        mask = same_subj & (cur >= 0) & (nxt >= 0)
        cur_m, nxt_m = cur[mask], nxt[mask]
        clusters = sorted(np.unique(np.concatenate([cur_m, nxt_m])))
        cidx = {c: i for i, c in enumerate(clusters)}
        T = np.zeros((len(clusters), len(clusters)))
        for a, b in zip(cur_m, nxt_m):
            T[cidx[a], cidx[b]] += 1
        # row-normalize to transition probabilities
        T = T / T.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(T, cmap="viridis", ax=ax, cbar_kws={"label": "P(next | current)"})
        ax.set_xlabel("next cluster"); ax.set_ylabel("current cluster")
        ax.set_title("Cluster transition matrix (consecutive windows)")
        save(fig, "20_transition_matrix.png")

except Exception as _e:
    print(f"  [skip] Plot 20 failed: {type(_e).__name__}: {str(_e)[:100]}")
# COMMAND ----------

# DBTITLE 1,Summary - list everything generated
import os
print("Figures written to:", FIG_DIR)
for f in sorted(os.listdir(FIG_DIR)):
    print("  ", f)

# COMMAND ----------

for name, obj in [("labels",labels),("probs",probs),("emb",emb),("win",win),
                  ("scored",scored),("annot",annot),("demostat",demostat),
                  ("prop",prop),("corr",corr),("cvd",cvd)]:
    status = "None/missing" if obj is None else (
        f"{obj.shape}" if hasattr(obj,"shape") else type(obj).__name__)
    print(f"  {name:10s}: {status}")
