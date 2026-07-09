"""
stages/_inject.py - build the variable namespace the original stage scripts
expect, derived from the typed config.

The original notebooks declared module-level constants (TAG, OUT_DIR, CACHE_DIR,
DEMO_CSV, EVENTS_DIR, frequency, win_length, batch_size, ...). Rather than edit
hundreds of references inside those validated bodies, each stage seeds these
names from cfg before executing the script. This is what makes config.yaml the
single source of truth without rewriting the stage internals.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from pft_config import PFTConfig


def common_namespace(cfg: PFTConfig) -> Dict[str, Any]:
    """Names shared across most stage scripts."""
    p = cfg.paths
    return {
        # identity / tag
        "TAG": cfg.tag, "tag": cfg.tag,
        "ENCODER_NAME": cfg.encoder_name,
        "NUM_FILES": cfg.num_files, "num_files": cfg.num_files,
        "RANDOM_STATE": cfg.random_state,
        # paths (both UPPER and lower aliases the scripts have used)
        "OUT_DIR": p.out_dir, "out_dir": p.out_dir,
        "CACHE_DIR": p.cache_dir, "cache_dir": p.cache_dir,
        "DEMO_CSV": p.demo_csv, "demo_csv": p.demo_csv,
        "EVENTS_DIR": p.events_dir, "events_dir": p.events_dir,
        "EVENTS_PARENT": p.events_parent, "PARENT": p.events_parent,
        "ZARR_DIR": p.zarr_dir, "zarr_dir": p.zarr_dir,
        "MODEL_CKPT": p.model_ckpt, "pft_ckpt_path": p.model_ckpt,
        # signal / windowing
        "FREQUENCY": cfg.signal.frequency, "frequency": cfg.signal.frequency,
        "WIN_LENGTH": cfg.signal.win_length, "win_length": cfg.signal.win_length,
        "HOP_LENGTH": cfg.signal.hop_length, "hop_length": cfg.signal.hop_length,
        "MAX_SEQ_LEN_SEC": cfg.signal.max_seq_len_sec,
        "max_seq_len_sec": cfg.signal.max_seq_len_sec,
        "WINDOW_SIZE_SEC": cfg.signal.window_size_sec,
        "window_size_sec": cfg.signal.window_size_sec,
        "expected_windows": cfg.signal.expected_windows,
        # embedding
        "CHANNEL_NAMES": cfg.embedding.channel_names,
        "N_CHANNELS": cfg.embedding.n_channels,
        "CH_DIM": cfg.embedding.channel_dim,
        "D": cfg.embedding.D,
        # compute
        "USE_FP16": cfg.compute.use_fp16, "use_fp16": cfg.compute.use_fp16,
        "BATCH_SIZE": cfg.compute.batch_size, "batch_size": cfg.compute.batch_size,
        "WORKERS": cfg.compute.workers, "workers": cfg.compute.workers,
        # clustering
        "N_FIT_UMAP": cfg.clustering.n_fit_umap, "N_FIT": cfg.clustering.n_fit_umap,
        # analysis
        "PRIMARY_AHI": cfg.analysis.primary_ahi,
        "SEVERITY_MEASURES": cfg.analysis.severity_measures,
        "CVD_EVENT": cfg.analysis.cvd_event_col,
        "CVD_TIME": cfg.analysis.cvd_time_col,
        "FDR_ALPHA": cfg.analysis.fdr_alpha,
        # give scripts the cfg object too, for anything not covered above
        "cfg": cfg,
    }


def script_path(name: str) -> Path:
    """Path to the bundled original stage script."""
    return Path(__file__).resolve().parent.parent / "stage_scripts" / f"{name}.py"
