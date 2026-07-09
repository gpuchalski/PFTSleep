"""stages/core_plots.py - runs the core_plots stage from its bundled script, config-driven."""
from __future__ import annotations
from pft_config import PFTConfig
from nb_exec import exec_notebook
from stages._inject import common_namespace, script_path

NAME = "core_plots"


def run(cfg: PFTConfig) -> None:
    inject = common_namespace(cfg)
    exec_notebook(script_path("core_plots"), inject)
