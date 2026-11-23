"""Paired comparison of solve rates (bootstrap 95% CI), shared by every round's report.
Moved here unchanged from the round-2 report (r2_report.py)."""

from typing import Any

import numpy as np

N_BOOT = 10_000


def solved(rows: Any, key: Any = "top1_val") -> dict:
    return {r["pid"]: bool(r["merged"] and r["merged"][key]) for r in rows}


def paired(a: Any, b: Any, seed: Any) -> dict:
    """Paired difference of success rates (a − b) over common problems, bootstrap 95% CI."""
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1) if len(d) else np.array([np.nan])
    return dict(n=len(ids), diff=float(d.mean()) if len(d) else float("nan"),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))
