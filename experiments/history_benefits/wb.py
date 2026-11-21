"""History-refinement adapter over the shared experiments/exp_wandb.py (same Run(cfg, out, name, job_type) API).

Runs go to project "flash-ansr-history" (override with WANDB_PROJECT), grouped by the run's output folder.
"""

from pathlib import Path
from typing import Any
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_wandb as _shared  # noqa: E402  (experiments/exp_wandb.py)

PROJECT = "flash-ansr-history"
logged_in = _shared.logged_in
_mode = _shared.mode


class Run(_shared.Run):
    def __init__(self, cfg: Any, out: Any, name: Any, job_type: Any, extra: Any = None) -> None:
        super().__init__(group=os.path.basename(str(cfg["output"])), name=name, job_type=job_type,
                         config={**cfg, **(extra or {})}, out=out, project=PROJECT)
