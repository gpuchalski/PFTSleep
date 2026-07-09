"""stages/extended_plots.py - runs the extended_plots stage from its bundled script, config-driven."""
from __future__ import annotations
from pft_config import PFTConfig
from nb_exec import exec_notebook
from stages._inject import common_namespace, script_path

NAME = "extended_plots"


def run(cfg: PFTConfig) -> None:
    inject = common_namespace(cfg)
    exec_notebook(script_path("extended_plots"), inject)
