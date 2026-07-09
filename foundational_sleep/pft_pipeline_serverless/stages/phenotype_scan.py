"""stages/phenotype_scan.py - all-k metric scan to guide phenotype k selection."""
from __future__ import annotations
from pft_config import PFTConfig
from nb_exec import exec_notebook
from stages._inject import common_namespace, script_path

NAME = "phenotype_scan"


def run(cfg: PFTConfig) -> None:
    exec_notebook(script_path("phenotype_scan"), common_namespace(cfg))
