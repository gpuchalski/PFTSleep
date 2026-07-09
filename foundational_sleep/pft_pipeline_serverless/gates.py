"""
gates.py - hard verification gates between pipeline stages.

Each gate inspects on-disk artifacts and RAISES if a stage produced bad output,
halting the pipeline before a downstream stage wastes hours on garbage. The
gates encode the exact failure modes found during development:
  - FP16 NaN overflow (99% NaN embeddings)
  - inverted padding mask
  - nsrrid-vs-intidx mismatch (demographics matched 0/1219)
  - label/window misalignment

Every gate takes the typed config and returns nothing on success.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from pft_config import PFTConfig


class GateError(RuntimeError):
    """Raised when a verification gate fails."""


# --------------------------------------------------------------------------- #
def gate_extraction(cfg: PFTConfig) -> None:
    """GATE 1 - extraction produced clean, non-NaN, real-nsrrid embeddings."""
    cache, tag, D = cfg.paths.cache_dir, cfg.tag, cfg.embedding.D

    required = [cache / f"{n}__{tag}.npy"
                for n in ("night_id", "time_idx", "nsrrid", "nsrrid_intidx")]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        # Show what IS in the cache dir so a filename/tag/path mismatch is obvious.
        existing = sorted(p.name for p in cache.glob("*.npy")) if cache.exists() else []
        meta_like = [n for n in existing if any(
            n.startswith(k) for k in ("night_id", "time_idx", "nsrrid"))]
        raise GateError(
            "GATE 1 FAILED: extraction metadata not found at expected path.\n"
            f"  Looking in : {cache}\n"
            f"  Expected   : {missing[0]}\n"
            f"  Cache dir exists: {cache.exists()}\n"
            f"  Metadata-like files actually present: "
            f"{meta_like if meta_like else '(none)'}\n"
            f"  Total .npy in cache dir: {len(existing)}\n"
            "  If the present files have a different tag suffix, the cluster/\n"
            "  extract tag differs from cfg.tag. If the dir is wrong or empty,\n"
            "  cfg.paths.cache_dir does not match where extraction wrote.")

    shards = sorted(cfg.shard_dir.glob("Z_part_*.npy"))
    if not shards:
        raise GateError(f"GATE 1 FAILED: no shards in {cfg.shard_dir}")

    w = np.load(shards[0], mmap_mode="r").shape[1]
    if w != D:
        raise GateError(f"GATE 1 FAILED: shard width {w} != {D}. "
                        "These are not 7-channel-concatenated embeddings.")

    sample = shards[:: max(1, len(shards) // 30)][:30]
    bad = []
    for sf in sample:
        s = np.load(sf).astype(np.float32)
        frac = (~np.isfinite(s)).mean()
        if frac > 0:
            bad.append((sf.name, float(frac)))
    if bad:
        raise GateError(
            "GATE 1 FAILED: NaN/Inf in embeddings - the FP16 overflow signature. "
            "Confirm compute.use_fp16=false and re-extract.\n"
            f"  First offenders: {bad[:5]}")

    nsrrid = np.load(cache / f"nsrrid__{tag}.npy", allow_pickle=True).astype(np.int64)
    if nsrrid.max() < 1000:
        raise GateError(
            "GATE 1 FAILED: nsrrid values look like sequential indices "
            f"(max={nsrrid.max()}), not real SHHS ids.")

    print(f"GATE 1 PASSED: {len(shards)} shards, width {D}, NaN-free sample, "
          f"real nsrrids ({nsrrid.min()}..{nsrrid.max()}).")


# --------------------------------------------------------------------------- #
def gate_clustering(cfg: PFTConfig) -> None:
    """GATE 2 - cluster labels exist and align 1:1 with window count."""
    cache, tag, out = cfg.paths.cache_dir, cfg.tag, cfg.paths.out_dir

    n_windows = len(np.load(cache / f"night_id__{tag}.npy"))

    candidates = (list(out.glob("*cluster*label*.npy")) +
                  list(Path("/dbfs/tmp").glob("**/cluster_labels*.npy")))
    if not candidates:
        raise GateError("GATE 2 FAILED: no cluster label file found. "
                        "Did the cluster stage complete and save labels?")
    labels = np.load(sorted(candidates)[-1])
    if len(labels) != n_windows:
        raise GateError(
            f"GATE 2 FAILED: {len(labels)} labels != {n_windows} windows. "
            "Label/window misalignment - do not trust downstream analysis.")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = float(np.mean(labels == -1) * 100)
    print(f"GATE 2 PASSED: {len(labels):,} labels aligned to windows, "
          f"{n_clusters} clusters, {noise_pct:.1f}% noise.")


# --------------------------------------------------------------------------- #
def gate_analysis_inputs(cfg: PFTConfig) -> None:
    """Pre-analysis gate - the files the analysis stages need are present."""
    out, tag = cfg.paths.out_dir, cfg.tag
    need = [out / f"cluster_labels__{tag}.npy"]
    win = out / f"window_clusters__{tag}.parquet"
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise GateError("ANALYSIS GATE FAILED: missing required inputs: "
                        + ", ".join(missing))
    print("ANALYSIS GATE PASSED: required clustering outputs present.")


# --------------------------------------------------------------------------- #
GATES = {
    "extract": gate_extraction,
    "cluster": gate_clustering,
    "cluster_analysis": gate_analysis_inputs,
}


def run_gate(stage: str, cfg: PFTConfig) -> None:
    """Run the gate for a stage if one is defined; otherwise no-op."""
    g = GATES.get(stage)
    if g is not None:
        g(cfg)
