# PFTSleep Pipeline

One program, one config file, one pipeline. This replaces the old collection of
notebooks chained by `%run` with a single config-driven pipeline where each
stage is an importable module.

## Layout

```
pft_pipeline/
├── config.yaml              # ALL settings live here - edit this, not code
├── pft_config.py            # typed loader: validates config.yaml -> PFTConfig
├── gates.py                 # hard verification gates between stages
├── nb_exec.py               # runs the bundled notebook-export scripts as stages
├── run_pipeline.py          # the driver / single entry point
├── stages/                  # one thin module per stage (NAME + run(cfg))
│   ├── __init__.py          #   STAGE_ORDER (canonical execution order)
│   ├── _inject.py           #   builds the config namespace each script needs
│   ├── extract.py
│   ├── cluster.py
│   ├── collect_events.py
│   ├── cluster_analysis.py
│   ├── scored_sleep.py
│   ├── annotation_demographics.py
│   ├── core_plots.py
│   ├── extended_plots.py
│   └── phenotypes.py
└── stage_scripts/           # your original, validated stage bodies (unchanged)
    └── *.py
```

The key idea: **`config.yaml` is the single source of truth.** Every stage reads
its settings from there via `pft_config.py`. The path inconsistencies that
existed across the old notebooks (e.g. `A100_clustering_outputs_new` vs
`clustering_outputs_new/A100`) can no longer happen, because every stage derives
paths from one validated config object.

## One-time cluster setup (installs)

`%pip` installs cannot run inside an imported module, so do them once in the
first cell of your driver notebook (or a cluster init script), BEFORE importing
the pipeline:

```python
%pip install "zarr<3" "numcodecs<0.16" -q
%pip install /Workspace/Users/gpuchalski@kumc.edu/PFTSleep/ -q
%pip install umap-learn hdbscan optuna optuna-dashboard faiss-cpu lifelines statsmodels -q
%pip install plotly matplotlib seaborn scipy scikit-learn pandas pyarrow joblib psutil tqdm pyyaml -q
%pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.10.*" "cudf-cu12==24.10.*" "cupy-cuda12x==13.*"
dbutils.library.restartPython()
```

## Running it

### From a Databricks notebook (recommended)
```python
import sys; sys.path.append("/Workspace/Users/<you>/pft_pipeline")
from pft_config import load_config
from run_pipeline import run

cfg = load_config()      # reads config.yaml next to pft_config.py
run(cfg)                 # runs every enabled stage, in order, with gates
```

### Run only some stages (e.g. re-do just the plots)
```python
run(cfg, only=["core_plots", "extended_plots"])
```

### From the command line
```bash
python run_pipeline.py                       # all enabled stages
python run_pipeline.py --only cluster_analysis core_plots
python run_pipeline.py --config other.yaml   # a different config
python run_pipeline.py --no-gates            # debugging only
```

## Turning stages on/off

Two ways:
- **Persistent:** set the flags under `stages:` in `config.yaml` to true/false.
- **Per-run:** pass `--only ...` (CLI) or `only=[...]` (notebook). This overrides
  the flags for that run without editing the file.

The canonical order is fixed in `stages/__init__.py` (`STAGE_ORDER`); `--only`
always runs in that order regardless of how you list them.

## The verification gates

After a stage runs, its gate (if any) checks the on-disk output and HALTS the
pipeline on failure, so a long run never proceeds on bad data:

- **after `extract`** - GATE 1: metadata exists, shard width == 3584, NaN-free
  sample (catches the FP16 overflow), real nsrrids (not sequential indices).
- **after `cluster`** - GATE 2: label count == window count (catches
  label/window misalignment).
- **before `cluster_analysis`** - required clustering outputs present.

Gates live in `gates.py`; add one by writing `gate_<stage>(cfg)` and registering
it in the `GATES` dict.

## Changing settings (common edits)

| Want to change | Edit in `config.yaml` |
|---|---|
| Cohort size | `num_files` |
| Output location | `paths.out_dir` |
| FP16 vs FP32 | `compute.use_fp16` |
| Batch size | `compute.batch_size` |
| HDBSCAN params | `clustering.hdbscan_*` |
| CVD outcome column | `analysis.cvd_event_col` / `cvd_time_col` |
| Phenotype k values | `phenotypes.k_values` |
| Which stages run | `stages.*` |

After any edit, the next run uses the new values - no code changes needed.

## Adding a new stage

1. Put the stage body in `stage_scripts/my_stage.py` (a plain or notebook-export
   script that reads injected names like `OUT_DIR`, `TAG`, `cfg`).
2. Create `stages/my_stage.py` with `NAME = "my_stage"` and a `run(cfg)` that
   calls `exec_notebook(script_path("my_stage"), common_namespace(cfg))`.
3. Add `"my_stage"` to `STAGE_ORDER` in `stages/__init__.py`.
4. Add a `my_stage: true` flag under `stages:` in `config.yaml` and a field to
   the `Stages` dataclass in `pft_config.py`.
5. (Optional) add a `gate_my_stage(cfg)` to `gates.py` and register it.

## Notes

- The original stage bodies in `stage_scripts/` are kept **unchanged** in logic;
  the pipeline strips only Databricks magics (`%pip`, `%run`, `%md`,
  `restartPython`) so they run as importable code. Installs move to the one-time
  setup cell above.
- `pft_config.py` computes the `TAG` and embedding dim `D` in one place, so they
  are always consistent across stages.
- Run `python pft_config.py` alone to print the resolved config and sanity-check
  your edits before a full run.
