"""
pft_config.py - typed configuration loader for the PFTSleep pipeline.

Loads config.yaml into a validated dataclass tree. Every stage imports
`load_config()` and reads typed attributes instead of re-declaring constants.
This is the single source of truth: change config.yaml, and every stage sees
the change on the next run.

Why a dataclass over a raw dict:
  - validation at load time (missing/!typed keys fail fast, not mid-run)
  - the derived TAG and embedding dim D are computed in ONE place
  - editors autocomplete cfg.paths.out_dir, cfg.clustering.hdbscan_min_samples
  - the path inconsistencies that existed across the old notebooks
    (A100_clustering_outputs_new vs clustering_outputs_new/A100) cannot recur,
    because every stage derives paths from this object.

Usage:
    from pft_config import load_config
    cfg = load_config()                 # default: config.yaml beside this file
    cfg = load_config("/path/to/other_config.yaml")
    print(cfg.tag, cfg.embedding.D, cfg.paths.out_dir)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


# --------------------------------------------------------------------------- #
# Typed sections
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    frequency: int
    win_length: int
    hop_length: int
    max_seq_len_sec: int
    window_size_sec: int

    @property
    def expected_windows(self) -> int:
        return int(self.max_seq_len_sec / self.window_size_sec)

    @property
    def max_seq_len(self) -> int:
        return self.max_seq_len_sec * self.frequency


@dataclass
class Embedding:
    channel_names: List[str]
    n_channels: int
    channel_dim: int

    @property
    def D(self) -> int:
        return self.n_channels * self.channel_dim


@dataclass
class Paths:
    zarr_dir: Path
    model_ckpt: Path
    cache_dir: Path
    out_dir: Path
    demo_csv: str
    events_dir: Path
    events_parent: Path
    cache_backup: str

    def __post_init__(self):
        # coerce the directory-like ones to Path
        for f in ("zarr_dir", "model_ckpt", "cache_dir", "out_dir",
                  "events_dir", "events_parent"):
            setattr(self, f, Path(getattr(self, f)))


@dataclass
class Compute:
    use_fp16: bool
    batch_size: int
    workers: int
    require_a100: bool


@dataclass
class Clustering:
    n_fit_umap: int
    umap_n_neighbors: int
    umap_n_components: int
    umap_build_algo: str
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_cluster_selection_method: str
    run_optuna_sweep: bool
    optuna_n_trials: int = 50
    umap_min_dist: float = 0.0


@dataclass
class Analysis:
    primary_ahi: str
    severity_measures: List[str]
    cvd_event_col: str
    cvd_time_col: str
    fdr_alpha: float


@dataclass
class Phenotypes:
    k_values: List[int]
    k_scan_min: int
    k_scan_max: int


@dataclass
class Stages:
    extract: bool
    cluster: bool
    collect_events: bool
    cluster_analysis: bool
    scored_sleep: bool
    annotation_demographics: bool
    core_plots: bool
    extended_plots: bool
    phenotype_scan: bool
    phenotypes: bool


@dataclass
class PFTConfig:
    encoder_name: str
    num_files: int
    random_state: int
    signal: Signal
    embedding: Embedding
    paths: Paths
    compute: Compute
    clustering: Clustering
    analysis: Analysis
    phenotypes: Phenotypes
    stages: Stages

    @property
    def tag(self) -> str:
        """The cache/output tag that keys every artifact on disk.
        Identical formula to the original scripts, derived in ONE place."""
        s = self.signal
        return (f"{self.encoder_name}__files{self.num_files}"
                f"__freq{s.frequency}__win{s.win_length}__hop{s.hop_length}"
                f"__max{s.max_seq_len_sec}__concat{self.embedding.n_channels}ch")

    @property
    def shard_dir(self) -> Path:
        return self.paths.cache_dir / f"{self.tag}_shards"

    def ensure_dirs(self) -> None:
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.out_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Fail fast on inconsistent settings before a multi-hour run."""
        errs = []
        if self.embedding.D != self.embedding.n_channels * self.embedding.channel_dim:
            errs.append("embedding dim mismatch")
        if self.signal.win_length != self.signal.hop_length:
            errs.append("win_length != hop_length (non-overlapping patches expected)")
        if self.signal.win_length != int(self.signal.window_size_sec * self.signal.frequency):
            errs.append("win_length != window_size_sec * frequency")
        if len(self.embedding.channel_names) != self.embedding.n_channels:
            errs.append("channel_names length != n_channels")
        if self.clustering.umap_n_components not in (2, 3):
            errs.append("umap_n_components should be 2 or 3")
        if errs:
            raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errs))


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> PFTConfig:
    """Load and validate config.yaml into a typed PFTConfig."""
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = PFTConfig(
        encoder_name=raw["encoder_name"],
        num_files=raw["num_files"],
        random_state=raw["random_state"],
        signal=Signal(**raw["signal"]),
        embedding=Embedding(**raw["embedding"]),
        paths=Paths(**raw["paths"]),
        compute=Compute(**raw["compute"]),
        clustering=Clustering(**raw["clustering"]),
        analysis=Analysis(**raw["analysis"]),
        phenotypes=Phenotypes(**raw["phenotypes"]),
        stages=Stages(**raw["stages"]),
    )
    cfg.validate()
    return cfg


if __name__ == "__main__":
    # Quick self-test / inspection when run directly.
    c = load_config()
    print("Config loaded and validated.")
    print(f"  tag       : {c.tag}")
    print(f"  embed dim : {c.embedding.D}")
    print(f"  out_dir   : {c.paths.out_dir}")
    print(f"  stages on : {[k for k,v in vars(c.stages).items() if v]}")
