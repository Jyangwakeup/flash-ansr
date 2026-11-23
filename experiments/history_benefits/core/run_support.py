"""Run bookkeeping shared by every round: provenance records, loading step-0 and checkpoints,
jsonl progress logs. Moved here unchanged from the round-2 script (r2_run.py)."""

from typing import Any
import json
import subprocess
import sys
import time

import torch
from flash_ansr.model.flash_ansr_model import FlashANSRModel

from common import ROOT, sha


def provenance(cfg: Any, out: Any, stage: Any) -> None:
    """Record config, code version and file hashes of one stage (append-only)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    files = sorted((ROOT / "experiments/history_benefits").rglob("*.py"))
    record = dict(
        stage=stage,
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        argv=sys.argv,
        git_commit=git("rev-parse", "HEAD").strip(),
        git_dirty=bool(git("status", "--porcelain").strip()),
        hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
        torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
    with (out / "provenance.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")


def load_step0(cfg: Any, out: Any) -> Any:
    m = FlashANSRModel.from_config(json.loads((out / "model_config.json").read_text()))
    m.load_state_dict(torch.load(out / "step0.pt", weights_only=True))
    return m


def load_checkpoint(m: Any, path: Any) -> Any:
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m


def jsonl_logger(path: Any) -> Any:
    def log(row: Any) -> None:
        with path.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(f"{path.parent.name}/{path.stem} pid={row['pid']}", flush=True)
    return log
