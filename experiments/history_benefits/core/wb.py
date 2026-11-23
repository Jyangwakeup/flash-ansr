"""History-refinement adapter over the shared experiments/exp_wandb.py (same Run(cfg, out, name, job_type) API).

Runs go to project "flash-ansr-history" (override with WANDB_PROJECT).

How a run is labelled on W&B, so runs of different rounds can be told apart in one list:
* group: the config's ``wandb_group`` (e.g. ``r3_near_correct_history``); without it, the
  output folder name as before (e.g. ``mix``).
* name: round prefix + the script's name, e.g. ``r3 train-A`` or ``r3-smoke train-A``.
* tags: the round (``r3``), plus ``smoke`` for smoke configs (output folder or group ending in
  ``_smoke``), so smoke runs can be filtered out with one click.
"""

from pathlib import Path
from typing import Any
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import exp_wandb as _shared  # noqa: E402  (experiments/exp_wandb.py)

PROJECT = "flash-ansr-history"
logged_in = _shared.logged_in
_mode = _shared.mode


def labels(cfg: Any, name: Any) -> tuple[str, str, list[str]]:
    """(group, display name, tags) for one run of this config."""
    folder = os.path.basename(str(cfg["output"]))
    group = str(cfg.get("wandb_group") or folder)
    smoke = folder.endswith("_smoke") or group.endswith("_smoke")
    base = group.removesuffix("_smoke")
    short = base.split("_")[0] if cfg.get("wandb_group") else base   # "r3_near_..." -> "r3"
    prefix = f"{short}-smoke" if smoke else short
    return group, f"{prefix} {name}", [short] + (["smoke"] if smoke else [])


class Run(_shared.Run):
    def __init__(self, cfg: Any, out: Any, name: Any, job_type: Any, extra: Any = None) -> None:
        group, display, tags = labels(cfg, name)
        super().__init__(group=group, name=display, job_type=job_type, tags=tags,
                         config={**cfg, **(extra or {})}, out=out, project=PROJECT)
