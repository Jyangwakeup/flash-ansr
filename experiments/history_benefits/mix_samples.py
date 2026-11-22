"""Mixed training batches and the checkpoint-selection loss of the mix run (正式实验规则.md §12).

Sample kinds (fractions per group in ``cfg["mix"]``):

    none   no history; a fresh on-the-fly problem
    near   a near-correct history (corrupt.py) of a fresh problem's true formula
    real   one of the first history_top candidates of a stored model pool (5,000 problems)

Fresh problem j of step s is drawn with seed SeedSequence([data_seed_fresh, s, j]), so every
group sees the same fresh problems (A's are a superset of C's) and a resumed run repeats them.
"""

from typing import Any
import warnings

import numpy as np

from common import as_problem, collate, eval_seed, history_ids, history_sample, loss, read_pool, read_problem
from corrupt import LEVELS, near_correct

KINDS = ("none", "near", "real")


def kind_counts(fractions: dict, batch: int) -> dict:
    """Integer sample counts per kind (largest remainder), summing to ``batch``."""
    raw = {k: float(fractions.get(k, 0)) * batch for k in KINDS}
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"mix fractions must sum to 1: {fractions}")
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(KINDS, key=lambda k: (-(raw[k] - counts[k]), KINDS.index(k)))[: batch - sum(counts.values())]:
        counts[k] += 1
    return counts


class MixSampler:
    def __init__(self, cfg: Any, out: Any, model: Any, wc: Any, catalog: Any, fresh_source: Any,
                 group: Any, pool_ids: Any) -> None:
        self.cfg, self.out, self.model, self.wc, self.catalog = cfg, out, model, wc, catalog
        self.fresh_source = fresh_source
        self.group = group
        self.fractions = dict(cfg["mix"][group])
        self.counts = kind_counts(self.fractions, cfg["batch_size"])
        self.levels = [float(cfg["near_levels"][lv]) for lv in LEVELS]
        self.pool_ids = list(pool_ids)
        self._pools: dict = {}
        self.engine = model.simplipy_engine

    # -- sources -------------------------------------------------------------------------
    def fresh(self, *key: int) -> Any:
        seq = np.random.SeedSequence([int(self.cfg["data_seed_fresh"]), *map(int, key)])
        self.fresh_source._rng = np.random.default_rng(seq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for problem in iter(self.fresh_source):
                if not problem.is_placeholder:
                    return as_problem(problem)
        raise RuntimeError("fresh source ended")

    def pool(self, pid: int) -> tuple:
        """(problem, first history_top candidates, their predictions), cached."""
        if pid not in self._pools:
            rows, yh = read_pool(self.out, pid, "model")
            top = self.cfg["history_top"]
            self._pools[pid] = (read_problem(self.out, pid), rows[:top], yh[:top])
        return self._pools[pid]

    def tokenizable(self, expr: list) -> bool:
        try:
            history_ids(self.model, {"expression_prefix": expr})
        except Exception:  # noqa: BLE001
            return False
        return True

    # -- samples -------------------------------------------------------------------------
    def near(self, problem: Any, rng: Any, level: Any = None) -> dict | None:
        drawn = LEVELS[int(rng.choice(len(LEVELS), p=self.levels))]  # always drawn: same rng path
        return near_correct(problem, level or drawn, rng, self.engine, self.tokenizable)

    def make(self, kind: str, rng: Any, fresh_key: tuple = (), validation: bool = False,
             pid: Any = None, level: Any = None, drop_history: bool = False) -> Any:
        """One sample of ``kind``. ``fresh_key`` seeds the fresh problem (none/near); ``pid``
        names a stored problem instead (selection set, and every real sample).

        Ablations (ablate.py): ``level`` forces the near-correct level; ``drop_history`` makes
        every choice exactly as with history (same rng path, same problem and rows) but builds
        the sample without it, so the two differ only in the history."""
        if kind == "real":
            for _ in range(20):
                if pid is None:
                    pid = int(self.pool_ids[int(rng.integers(len(self.pool_ids)))])
                problem, rows, yh = self.pool(pid)
                if rows:
                    break
                pid = None  # an empty pool: draw another problem
            chosen = int(rng.integers(len(rows)))
            s = history_sample(self.cfg, self.model, self.wc, self.catalog, problem, rng,
                               None if drop_history else (rows[chosen]["expression_prefix"], yh[chosen]),
                               validation)
            s["info"] = dict(pid=pid, candidate=chosen, history_fvu=rows[chosen]["fvu"])
            s["chosen"] = chosen if s["metadata"]["history_valid"] else None
        else:
            for attempt in range(10):
                problem = (read_problem(self.out, pid) if pid is not None
                           else self.fresh(*fresh_key, *([attempt] if attempt else [])))
                hist = self.near(problem, rng, level) if kind == "near" else None
                if kind == "none" or hist is not None:
                    break
                # a stored problem retries with the advanced rng; a fresh one takes a new problem
            else:
                raise RuntimeError(f"no near-correct history for {pid if pid is not None else fresh_key}")
            history = (hist["expression"], hist["predictions"]) if hist and not drop_history else None
            s = history_sample(self.cfg, self.model, self.wc, self.catalog, problem, rng,
                               history, validation)
            s["info"] = dict(pid=pid) if pid is not None else dict(fresh=[*fresh_key, attempt])
            if hist:
                s["info"].update(level=hist["level"], actual=hist["actual"], edits=hist["edits"])
        s["kind"] = kind
        s["planned_history"] = kind != "none"
        return s

    def __call__(self, step: int, rng: Any) -> list:
        kinds = [k for k in KINDS for _ in range(self.counts[k])]
        kinds = [kinds[i] for i in rng.permutation(len(kinds))]
        seeds = rng.integers(0, 2**32 - 1, size=len(kinds))
        out, j = [], 0
        for kind, seed in zip(kinds, seeds):
            sample_rng = np.random.default_rng(int(seed))
            key: tuple = ()
            if kind != "real":
                key, j = (step, j), j + 1
            s = self.make(kind, sample_rng, fresh_key=key)
            s["info"]["seed"] = int(seed)
            out.append(s)
        return out


def selection_set(sampler: MixSampler, ids: Any) -> dict:
    """Fixed checkpoint-selection samples: per problem one sample of every kind the group
    trains on (seeded by the eval seed), built once."""
    samples: dict = {}
    for kind in KINDS:
        if not sampler.fractions.get(kind):
            continue
        samples[kind] = [sampler.make(kind, np.random.default_rng([eval_seed(sampler.cfg, pid), KINDS.index(kind)]),
                                      validation=True, pid=pid) for pid in ids]
    return samples


def selection_loss(model: Any, samples: dict, fractions: dict, micro_batch: int) -> dict:
    """Target CE per kind on the selection set and their mix-weighted sum (``loss``, the
    pre-registered checkpoint criterion)."""
    result: dict[str, Any] = {}
    for kind, rows in samples.items():
        ces = []
        for begin in range(0, len(rows), micro_batch):
            part = rows[begin: begin + micro_batch]
            _, metrics = loss(model, collate(part, model.tokenizer))
            ces += metrics["ce_rows"]
        result[f"ce_{kind}"] = float(np.mean(ces))
        result[f"history_valid_{kind}"] = float(np.mean([s["metadata"]["history_valid"] for s in rows]))
    result["loss"] = float(sum(fractions[k] * result[f"ce_{k}"] for k in samples))
    result["n"] = {k: len(v) for k, v in samples.items()}
    return result
