"""
nb_exec.py - run a Databricks notebook-exported .py as a stage body.

The original stages are Databricks notebook exports containing magics
(%pip, %run, %md) and dbutils calls. To run them as importable stages while
preserving the working code, this helper:
  - strips '# MAGIC' lines and '# COMMAND ----------' separators
  - removes %pip / %run / %md / dbutils.library.restartPython lines
    (installs and restarts are handled once by the driver environment, not
     per-stage)
  - executes the remaining code in a namespace pre-seeded with the config-
    derived constants the script expects (TAG, OUT_DIR, CACHE_DIR, ...)

This lets the large, validated extraction and clustering notebooks run
unchanged in body while still being driven by the single config. Smaller
analysis stages are refactored directly instead of going through this path.

NOTE: %pip installs must be done in the cluster/driver setup (see SETUP.md),
because pip + restartPython cannot run mid-process inside an imported module.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, Any


_DROP_PATTERNS = (
    re.compile(r"^\s*# MAGIC\s+%pip"),
    re.compile(r"^\s*# MAGIC\s+%run"),
    re.compile(r"^\s*# MAGIC\s+%md"),
    re.compile(r"^\s*# MAGIC\s+%sh"),
    re.compile(r"dbutils\.library\.restartPython"),
)


def clean_notebook_source(path: Path) -> str:
    r"""Return the script source with Databricks magics/separators removed.

    Handles multi-line magic commands: a a multi-line MAGIC pip install followed by
    `# MAGIC numpy \\` continuation lines - all continuation lines are dropped,
    not just the first, so no stray `numpy \` code survives.
    """
    out_lines = []
    in_magic_continuation = False
    in_md_cell = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()

        if stripped == "# COMMAND ----------":
            in_magic_continuation = False
            in_md_cell = False
            continue
        if stripped.startswith("# DBTITLE"):
            continue
        if stripped == "# MAGIC":
            continue

        # Is this a "# MAGIC ..." line? Extract its payload.
        m = re.match(r"^(\s*)# MAGIC ?(.*)$", line)
        if m:
            payload = m.group(2)
            # Inside a %md cell: every MAGIC line is markdown -> drop until
            # the cell boundary (handled above).
            if in_md_cell:
                continue
            # If we're mid-continuation of a dropped backslash magic, drop it.
            if in_magic_continuation:
                in_magic_continuation = payload.rstrip().endswith("\\")
                continue
            # Start of a %md cell -> drop this and all following MAGIC lines.
            if payload.lstrip().startswith("%md"):
                in_md_cell = True
                continue
            # Start of another magic to DROP (%pip/%run/%sh/restartPython)?
            if (any(p.search(line) for p in _DROP_PATTERNS)
                    or payload.lstrip().startswith(("%pip", "%run", "%sh"))
                    or "restartPython" in payload):
                in_magic_continuation = payload.rstrip().endswith("\\")
                continue
            # A non-magic "# MAGIC <code>" line -> keep as real code.
            if payload.lstrip().startswith("%"):
                continue
            out_lines.append(m.group(1) + payload)
            continue

        # Plain (non-MAGIC) line.
        in_magic_continuation = False
        in_md_cell = False
        out_lines.append(line)
    return "\n".join(out_lines)


def exec_notebook(path: Path, inject: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a cleaned notebook script with `inject` pre-seeded in globals.

    Returns the resulting namespace so callers can pull out values if needed.
    """
    src = clean_notebook_source(path)
    ns: Dict[str, Any] = dict(inject)
    ns["__name__"] = "__pft_stage__"
    code = compile(src, str(path), "exec")
    exec(code, ns)  # noqa: S102 - intentional: runs the validated stage body
    return ns
