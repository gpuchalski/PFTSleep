# Databricks notebook source
# =============================================================================
# PFTSleep Pipeline - Driver Notebook
# =============================================================================
# The single entry point. Run this top to bottom. It installs dependencies once,
# loads the typed config, and runs the pipeline (all enabled stages, in order,
# with verification gates).
#
# Edit settings in config.yaml, NOT here. This notebook just drives.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -e "/Workspace/Users/gpuchalski@kumc.edu/PFTSleep/"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,One-time installs (must be before importing the pipeline)
# MAGIC %uv pip install "zarr<3" "numcodecs<0.16" -q
# MAGIC %uv pip install umap-learn hdbscan optuna optuna-dashboard faiss-cpu lifelines statsmodels -q
# MAGIC %pip install "huggingface-hub>=0.34.0,<1.0"
# MAGIC %uv pip install plotly matplotlib seaborn scipy scikit-learn pandas pyarrow joblib psutil tqdm pyyaml -q
# MAGIC %uv pip install --prerelease=allow --index-strategy=unsafe-best-match --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.10.*" "cudf-cu12==24.10.*" "cupy-cuda12x==13.*"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

# DBTITLE 1,Point Python at the pipeline package
import sys
# adjust to wherever you put the pft_pipeline folder
PIPELINE_DIR = "/Workspace/Users/gpuchalski@kumc.edu/foundational_sleep/pft_pipeline"
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

# COMMAND ----------

# DBTITLE 1,Load + inspect the config (sanity check before a long run)
from pft_config import load_config
cfg = load_config("/Workspace/Users/gpuchalski@kumc.edu/foundational_sleep/pft_pipeline/config.yaml")   # reads config.yaml next to pft_config.py
print("tag      :", cfg.tag)
print("embed dim:", cfg.embedding.D)
print("out_dir  :", cfg.paths.out_dir)
print("stages on:", [k for k, v in vars(cfg.stages).items() if v])

# COMMAND ----------

# DBTITLE 1,Serverless GPU: stage volume data locally (FUSE workaround)
from volume_compat import localize_for_serverless, sync_outputs_back
sync_info = localize_for_serverless(cfg, dbutils)

# COMMAND ----------

# DBTITLE 1,Run the whole pipeline (all enabled stages, in order, with gates)
from run_pipeline import run
run(cfg, only=["extract"])
sync_outputs_back(sync_info, dbutils) 

run(cfg, only=["cluster", "collect_events", "cluster_analysis",
               "scored_sleep", "annotation_demographics",
               "core_plots", "extended_plots", "phenotype_scan"])

# COMMAND ----------

# DBTITLE 1,Choose k's from last step, update config and run this
run(cfg, only=["phenotypes"]) 

# COMMAND ----------

# DBTITLE 1,(Optional) Back up the cache off ephemeral /dbfs/tmp
# Sync pipeline outputs back to volume (handles both FUSE and non-FUSE)
import shutil
if sync_info.active:
    # Copy cache to the local cache_backup staging dir, then sync everything
    shutil.copytree(str(cfg.paths.cache_dir), str(cfg.paths.cache_backup),
                    dirs_exist_ok=True)
    sync_outputs_back(sync_info, dbutils)
else:
    shutil.copytree(str(cfg.paths.cache_dir), cfg.paths.cache_backup,
                    dirs_exist_ok=True)
    print("\u2713 Cache backed up to", cfg.paths.cache_backup)
