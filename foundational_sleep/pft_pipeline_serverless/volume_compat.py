"""
volume_compat.py - UC Volume compatibility for Serverless GPU.

On Databricks Serverless GPU compute, the FUSE mount for UC Volumes
(/Volumes/...) is intermittently non-functional: Python's os, pathlib, and
open() cannot read or write files on volumes.  This module auto-detects the
situation and transparently works around it.

Bypass mechanisms (both avoid FUSE entirely):
  - INPUT staging:  Spark binaryFile reader  (volume -> local /tmp)
  - OUTPUT sync:    Databricks SDK Files API (local /tmp -> volume)

Usage (in the driver notebook):
    from volume_compat import localize_for_serverless, sync_outputs_back
    sync_info = localize_for_serverless(cfg, dbutils)
    # ... run pipeline stages ...
    sync_outputs_back(sync_info, dbutils)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

_VOLUME_PREFIX = "/Volumes/"
_LOCAL_ROOT = Path("/tmp/_vol_local")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _is_volume_path(p) -> bool:
    return str(p).startswith(_VOLUME_PREFIX)


def fuse_works(test_path: str | Path) -> bool:
    """Return True if the FUSE layer can actually list contents of a volume dir.

    We rely on the fact that dbutils.fs.ls shows items but os.listdir does not
    when FUSE is broken.  test_path should be a known-non-empty volume directory.
    """
    p = str(test_path)
    if not p.startswith(_VOLUME_PREFIX):
        return True  # not a volume path at all
    try:
        if os.path.isdir(p):
            return len(os.listdir(p)) > 0
        return os.path.exists(p)
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Sync info
# --------------------------------------------------------------------------- #

@dataclass
class SyncInfo:
    """Tracks original volume paths so outputs can be pushed back."""
    active: bool = False
    output_mappings: Dict[str, str] = field(default_factory=dict)
    # local_path -> volume_path


# --------------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------------- #

def localize_for_serverless(cfg, dbutils) -> SyncInfo:
    """Stage volume data locally and redirect cfg paths.

    Parameters
    ----------
    cfg : PFTConfig
        The loaded pipeline config (will be mutated in-place).
    dbutils : DBUtils
        The Databricks dbutils handle (available in notebooks).

    Returns
    -------
    SyncInfo
        Pass this to sync_outputs_back() after the pipeline completes.
    """
    info = SyncInfo()

    # ---- Quick gate: is FUSE working? ---
    vol_root = str(cfg.paths.out_dir)
    while vol_root.count("/") > 4 and vol_root.startswith(_VOLUME_PREFIX):
        vol_root = str(Path(vol_root).parent)

    if fuse_works(vol_root):
        print("\u2713 Volume FUSE functional - no staging needed.")
        return info

    print("\u26A0 Volume FUSE is broken on this Serverless GPU instance.")
    print("  Staging volume data to local /tmp via Spark ...")
    info.active = True
    _LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    # ---- Stage INPUTS (read-only volume paths) ----

    # 1. demo_csv (single file, ~1 MB)
    if _is_volume_path(cfg.paths.demo_csv):
        local_csv = _LOCAL_ROOT / "inputs" / Path(cfg.paths.demo_csv).name
        local_csv.parent.mkdir(parents=True, exist_ok=True)
        _stage_file_via_spark(spark, cfg.paths.demo_csv, local_csv)
        cfg.paths.demo_csv = str(local_csv)
        print(f"  \u2713 demo_csv staged ({local_csv.name})")

    # 2. events_dir (directory, ~40 MB / 1220 files)
    local_events = _LOCAL_ROOT / "inputs" / "scored_sleep" / "cohort_1219_events"
    if _is_volume_path(cfg.paths.events_dir):
        local_events.mkdir(parents=True, exist_ok=True)
        n = _stage_dir_via_spark(spark, str(cfg.paths.events_dir), local_events)
        cfg.paths.events_dir = local_events
        print(f"  \u2713 events_dir staged ({n} files, {_dir_size_mb(local_events):.1f} MB)")

    # 3. events_parent - set to the parent of the staged events_dir
    if _is_volume_path(cfg.paths.events_parent):
        cfg.paths.events_parent = local_events.parent

    # ---- Redirect OUTPUTS ----

    # out_dir
    if _is_volume_path(cfg.paths.out_dir):
        original_out = str(cfg.paths.out_dir)
        local_out = _LOCAL_ROOT / "outputs" / "pipeline_run"
        local_out.mkdir(parents=True, exist_ok=True)
        cfg.paths.out_dir = local_out
        info.output_mappings[str(local_out)] = original_out
        print(f"  \u2713 out_dir redirected to {local_out}")

    # cache_backup
    if isinstance(cfg.paths.cache_backup, str) and cfg.paths.cache_backup.startswith(_VOLUME_PREFIX):
        original_backup = cfg.paths.cache_backup
        local_backup = _LOCAL_ROOT / "outputs" / "cache_backup"
        local_backup.mkdir(parents=True, exist_ok=True)
        cfg.paths.cache_backup = str(local_backup)
        info.output_mappings[str(local_backup)] = original_backup
        print(f"  \u2713 cache_backup redirected to {local_backup}")

    print("  Staging complete.  Pipeline will use local paths.")
    return info


# --------------------------------------------------------------------------- #
# Output sync
# --------------------------------------------------------------------------- #

def sync_outputs_back(info: SyncInfo, dbutils, *, verbose: bool = True) -> None:
    """Copy local output directories back to their volume destinations.

    Uses the Databricks SDK Files API (bypasses FUSE entirely).
    """
    if not info.active:
        if verbose:
            print("Nothing to sync (FUSE was functional).")
        return

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    for local_path, vol_path in info.output_mappings.items():
        if not os.path.isdir(local_path):
            continue
        files = []
        for root, _, filenames in os.walk(local_path):
            for fname in filenames:
                files.append(os.path.join(root, fname))

        if verbose:
            print(f"Syncing {len(files)} files: {local_path} -> {vol_path}")

        for fpath in files:
            rel = os.path.relpath(fpath, local_path)
            dest = f"{vol_path}/{rel}".replace("\\", "/")
            with open(fpath, "rb") as f:
                w.files.upload(dest, f, overwrite=True)

        if verbose:
            print(f"  \u2713 Done ({len(files)} files).")

    if verbose:
        print("All outputs synced to volume.")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _stage_file_via_spark(spark, vol_path: str, local_path: Path) -> None:
    """Read a single file from a volume via Spark and write locally."""
    df = spark.read.format("binaryFile").load(vol_path)
    row = df.first()
    with open(local_path, "wb") as f:
        f.write(row.content)


def _stage_dir_via_spark(spark, vol_dir: str, local_dir: Path) -> int:
    """Read all files from a volume directory via Spark and write locally."""
    df = spark.read.format("binaryFile").load(vol_dir + "/")
    rows = df.select("path", "content").collect()
    for row in rows:
        fname = row.path.split("/")[-1]
        with open(local_dir / fname, "wb") as f:
            f.write(row.content)
    return len(rows)


def _dir_size_mb(path: Path) -> float:
    """Total size of a local directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024
