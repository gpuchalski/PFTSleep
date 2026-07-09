"""
run_pipeline.py - the single entry point for the PFTSleep pipeline.

Loads the typed config, then runs each enabled stage in canonical order,
invoking the stage's verification gate after it completes. A stage failure or
a gate failure halts the pipeline with a clear message, so a long run never
silently proceeds on bad data.

Run the whole thing:
    python run_pipeline.py

Run with a different config:
    python run_pipeline.py --config /path/to/config.yaml

Run only specific stages (overrides the flags in config.yaml for this run):
    python run_pipeline.py --only cluster_analysis core_plots

Skip the gates (NOT recommended; for debugging only):
    python run_pipeline.py --no-gates

In a Databricks notebook, instead of the CLI you can do:
    from pft_config import load_config
    from run_pipeline import run
    run(load_config())
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
import traceback
from typing import List, Optional

from pft_config import load_config, PFTConfig
from gates import run_gate, GateError
from stages import STAGE_ORDER


def _enabled_stages(cfg: PFTConfig, only: Optional[List[str]]) -> List[str]:
    if only:
        unknown = [s for s in only if s not in STAGE_ORDER]
        if unknown:
            raise SystemExit(f"Unknown stage(s): {unknown}\n"
                             f"Valid stages: {STAGE_ORDER}")
        # preserve canonical order even if --only lists them out of order
        return [s for s in STAGE_ORDER if s in only]
    flags = vars(cfg.stages)
    return [s for s in STAGE_ORDER if flags.get(s, False)]


def run(cfg: PFTConfig, only: Optional[List[str]] = None,
        use_gates: bool = True) -> None:
    """Execute the pipeline. Importable from a notebook."""
    cfg.ensure_dirs()
    stages = _enabled_stages(cfg, only)

    print("=" * 70)
    print("PFTSleep pipeline")
    print(f"  tag    : {cfg.tag}")
    print(f"  out    : {cfg.paths.out_dir}")
    print(f"  stages : {stages}")
    print(f"  gates  : {'on' if use_gates else 'OFF'}")
    print("=" * 70)

    for name in stages:
        t0 = time.time()
        print(f"\n----- STAGE: {name}  -----")
        try:
            mod = importlib.import_module(f"stages.{name}")
        except ModuleNotFoundError:
            raise SystemExit(
                f"Stage module 'stages/{name}.py' not found. "
                "Create it (NAME + run(cfg)) or disable the stage in config.yaml.")
        if not hasattr(mod, "run"):
            raise SystemExit(f"stages/{name}.py has no run(cfg) function.")

        try:
            mod.run(cfg)
        except Exception:
            print(f"\nSTAGE '{name}' FAILED:\n{traceback.format_exc()}")
            raise SystemExit(f"Pipeline halted at stage '{name}'.")

        if use_gates:
            try:
                run_gate(name, cfg)
            except GateError as e:
                print(f"\n{e}")
                raise SystemExit(f"Pipeline halted by gate after stage '{name}'.")

        print(f"----- STAGE '{name}' done in {time.time()-t0:.0f}s -----")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run the PFTSleep pipeline.")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these stages (space-separated)")
    p.add_argument("--no-gates", action="store_true",
                   help="Skip verification gates (debugging only)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    run(cfg, only=args.only, use_gates=not args.no_gates)


if __name__ == "__main__":
    main()
