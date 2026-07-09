"""
stages/__init__.py - the stage registry.

Each stage is a module in this package exposing:
    NAME: str                      # matches a key in cfg.stages and the order list
    def run(cfg) -> None           # executes the stage using typed config

The driver imports STAGES (ordered) and runs each whose flag is enabled in
cfg.stages, calling the matching gate afterward (see gates.py).

To add a stage: create stages/my_stage.py with NAME and run(cfg), then add its
NAME to STAGE_ORDER below and a flag to the Stages dataclass + config.yaml.
"""
from __future__ import annotations

# Canonical execution order. The driver honors this sequence.
STAGE_ORDER = [
    "extract",
    "cluster",
    "collect_events",
    "cluster_analysis",
    "scored_sleep",
    "annotation_demographics",
    "core_plots",
    "extended_plots",
    "phenotype_scan",
    "phenotypes",
]
