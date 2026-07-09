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

# DBTITLE 1,One-time installs (must be before importing the pipeline)
# MAGIC %pip install "zarr<3" "numcodecs<0.16" -q
# MAGIC %pip install /Workspace/Users/gpuchalski@kumc.edu/PFTSleep/ -q
# MAGIC %pip install umap-learn hdbscan optuna optuna-dashboard faiss-cpu lifelines statsmodels -q
# MAGIC %pip install plotly matplotlib seaborn scipy scikit-learn pandas pyarrow joblib psutil tqdm pyyaml -q
# MAGIC %pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.10.*" "cudf-cu12==24.10.*" "cupy-cuda12x==13.*"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Point Python at the pipeline package
import sys
# adjust to wherever you put the pft_pipeline folder
PIPELINE_DIR = "/Workspace/Users/gpuchalski@kumc.edu/pft_pipeline"
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

# COMMAND ----------

# DBTITLE 1,Load + inspect the config (sanity check before a long run)
from pft_config import load_config
cfg = load_config()   # reads config.yaml next to pft_config.py
print("tag      :", cfg.tag)
print("embed dim:", cfg.embedding.D)
print("out_dir  :", cfg.paths.out_dir)
print("stages on:", [k for k, v in vars(cfg.stages).items() if v])

# COMMAND ----------

# DBTITLE 1,Run the whole pipeline (all enabled stages, in order, with gates)
from run_pipeline import run
run(cfg)

# COMMAND ----------

# DBTITLE 1,(Optional) Re-run only specific stages
# run(cfg, only=["core_plots", "extended_plots"])

# COMMAND ----------

# DBTITLE 1,(Optional) Back up the cache off ephemeral /dbfs/tmp
import shutil
shutil.copytree(str(cfg.paths.cache_dir), cfg.paths.cache_backup,
                dirs_exist_ok=True)
print("Cache backed up to", cfg.paths.cache_backup, "- remember to DETACH the A100.")

