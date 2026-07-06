#!/usr/bin/env python3
"""Execute calorie_pipeline.ipynb locally with progress logging."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parent
NOTEBOOK_IN = PROJECT / "calorie_pipeline.ipynb"
NOTEBOOK_OUT = PROJECT / "calorie_pipeline_executed.ipynb"


def main() -> int:
    print(f"Reading {NOTEBOOK_IN}", flush=True)
    with NOTEBOOK_IN.open() as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=7200,
        kernel_name="py312-local",
        resources={"metadata": {"path": str(PROJECT)}},
    )

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"Executing {n_code} code cells (SMOKE_TEST dry-run, CPU)...", flush=True)
    client.execute()

    NOTEBOOK_OUT.write_text(nbformat.writes(nb))
    print(f"\nDone. Wrote {NOTEBOOK_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
