"""Equal-time budget (正式实验规则.md §2.2). RUN ALONE: these timings decide the budget.

On the first ``n`` validation problems, time (encoding + history choice + generation + fitting):

- the original T8 with ``baseline_draws`` draws          -> the per-problem budget T
- the original T8 with ``phase0_draws`` draws            -> phase-0 cost t0
- every final condition's model with ``phase1_draws``    -> phase-1 cost per draw

and give each condition the phase-1 draws that fit the same time:
``n1 = round((T - t0) / (t1 / phase1_draws))`` (medians over the timed problems), at least 0.
The reruns at that budget may then run in parallel (their timing no longer decides anything).
"""

from typing import Any
import json
import platform
import time

import numpy as np

from common import dump
from medium_eval import CONDITIONS, run_condition, run_original

N_TIMED = 20


def measure(cfg: Any, out: Any, m: Any, val_ids: Any) -> None:
    from medium import load_checkpoint

    path = out / "equal_time_budget.json"
    if path.exists():
        return
    ids = val_ids[:N_TIMED]
    t_budget = [r["seconds_total"] for r in run_original(cfg, out, ids, cfg["baseline_draws"])]
    t_phase0 = [r["seconds_total"] for r in run_original(cfg, out, ids, cfg["phase0_draws"])]
    budget_T = float(np.median(t_budget))
    phase0_t = float(np.median(t_phase0))
    per_draw = {}
    draws = {}
    for name, (group, hist) in CONDITIONS.items():
        load_checkpoint(m, out / "cooldown" / group / f"checkpoint_{cfg['steps']}.pt")
        rows = run_condition(cfg, out, m, ids, hist, cfg["phase1_draws"], phase0=False)
        per_draw[name] = float(np.median([r["seconds_total"] for r in rows])) / cfg["phase1_draws"]
        draws[name] = max(0, int(round((budget_T - phase0_t) / per_draw[name])))
    dump(path, dict(
        timed_problems=ids,
        budget_seconds_per_problem=budget_T,
        phase0_seconds_per_problem=phase0_t,
        phase1_seconds_per_draw=per_draw,
        phase1_draws=draws,
        original_draws=cfg["baseline_draws"],
        machine=platform.platform(),
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        note="medians over the timed problems; measured with no other experiment running",
    ))
    print(json.dumps(draws), flush=True)
