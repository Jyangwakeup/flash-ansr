"""Mixed training batches and the fixed evaluation set of round 4 (设计与预注册.md §14).

Copied from round 3's r3_samples.py (rounds do not import each other). Changes:

* ``cfg["near_mode"][group]``: ``levels`` builds near-correct histories exactly as round 3
  (same rng path, so group I's samples equal round-3 C's); ``fvu_bins`` (group IC) aims each
  one at an FVU bin drawn uniformly (r4_corrupt.near_in_bin).
* every history sample records its support FVU and bin in ``info``;
* ``make(..., bin=, candidate=)`` force the bin / pool candidate (evaluation set);
* ``eval_set`` / ``eval_loss``: one fixed set for both groups -- per problem one none sample,
  one near-correct history per FVU bin and one real candidate per FVU bin.

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
from r4_corrupt import LEVELS, bin_of, near_correct, near_in_bin

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
        self.near_mode = cfg.get("near_mode", {}).get(group, "levels")
        self.edges = tuple(cfg.get("fvu_edges", (1e-3, 0.1)))
        self.pool_ids = list(pool_ids)
        self._pools: dict = {}
        self.eval_dropped = 0  # history samples dropped from the last eval_set
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
            # copy: a numpy slice is a view that would keep the whole predictions array in memory
            self._pools[pid] = (read_problem(self.out, pid), rows[:top], yh[:top].copy())
        return self._pools[pid]

    def tokenizable(self, expr: list) -> bool:
        try:
            history_ids(self.model, {"expression_prefix": expr})
        except Exception:  # noqa: BLE001
            return False
        return True

    # -- samples -------------------------------------------------------------------------
    def near(self, problem: Any, rng: Any, level: Any = None, bin: Any = None) -> dict | None:
        """``level`` forces round 3's construction at that level; ``bin`` forces the FVU bin
        of the binned construction; otherwise the group's ``near_mode`` decides."""
        if level is None and (bin is not None or self.near_mode == "fvu_bins"):
            drawn_bin = int(rng.integers(3))  # always drawn: same rng path
            return near_in_bin(problem, drawn_bin if bin is None else int(bin), rng, self.engine, self.tokenizable,
                               self.edges, float(self.cfg.get("near_exact_in_low", 0.5)),
                               tuple(self.cfg.get("near_constant_scale", (0.001, 0.5))))
        drawn = LEVELS[int(rng.choice(len(LEVELS), p=self.levels))]  # always drawn: same rng path
        return near_correct(problem, level or drawn, rng, self.engine, self.tokenizable)

    def make(self, kind: str, rng: Any, fresh_key: tuple = (), validation: bool = False,
             pid: Any = None, level: Any = None, drop_history: bool = False,
             bin: Any = None, candidate: Any = None) -> Any:
        """One sample of ``kind``. ``fresh_key`` seeds the fresh problem (none/near); ``pid``
        names a stored problem instead (selection set, and every real sample).

        Ablations (ablate.py): ``level`` forces the near-correct level; ``drop_history`` makes
        every choice exactly as with history (same rng path, same problem and rows) but builds
        the sample without it, so the two differ only in the history. ``bin`` forces the FVU
        bin of a near-correct history, ``candidate`` the pool candidate of a real one."""
        if kind == "real":
            for _ in range(20):
                if pid is None:
                    pid = int(self.pool_ids[int(rng.integers(len(self.pool_ids)))])
                problem, rows, yh = self.pool(pid)
                if rows:
                    break
                pid = None  # an empty pool: draw another problem
            chosen = int(rng.integers(len(rows)))  # always drawn: same rng path
            if candidate is not None:
                chosen = int(candidate)
            s = history_sample(self.cfg, self.model, self.wc, self.catalog, problem, rng,
                               None if drop_history else (rows[chosen]["expression_prefix"], yh[chosen]),
                               validation)
            f = rows[chosen]["fvu"]
            s["info"] = dict(pid=pid, candidate=chosen, history_fvu=f,
                             history_bin=bin_of(f if f is not None else float("inf"), self.edges))
            s["chosen"] = chosen if s["metadata"]["history_valid"] else None
        else:
            for attempt in range(10):
                problem = (read_problem(self.out, pid) if pid is not None
                           else self.fresh(*fresh_key, *([attempt] if attempt else [])))
                hist = self.near(problem, rng, level, bin) if kind == "near" else None
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
                s["info"].update(level=hist["level"], actual=hist["actual"], edits=hist["edits"],
                                 history_fvu=hist["fvu"], history_bin=hist["bin"])
                if "target" in hist:
                    s["info"].update(target_bin=hist["target"], fallback=hist["fallback"])
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


BIN_NAMES = ("fvu<1e-3", "fvu 1e-3–0.1", "fvu≥0.1")


def eval_specs(sampler: MixSampler, pid: int) -> list:
    """The history samples of one evaluation problem: (kind, bin, candidate). Near-correct: one
    per FVU bin. Real: per FVU bin one of the first history_top pool candidates in that bin
    (chosen by the eval seed), when the pool has one."""
    specs: list = [("near", b, None) for b in range(3)]
    _, rows, _ = sampler.pool(pid)
    rng = np.random.default_rng([eval_seed(sampler.cfg, pid), 41])
    for b in range(3):
        inside = [i for i, r in enumerate(rows)
                  if bin_of(r["fvu"] if r["fvu"] is not None else float("inf"), sampler.edges) == b]
        if inside:
            specs.append(("real", b, int(rng.choice(inside))))
    return specs


def spec_seed(sampler: MixSampler, pid: int, kind: str, b: Any) -> list:
    return [eval_seed(sampler.cfg, pid), KINDS.index(kind), 9 if b is None else int(b)]


def eval_set(sampler: MixSampler, ids: Any, with_base: bool = False) -> list:
    """The fixed evaluation set (same for every group and checkpoint; seeded by the eval seed):
    dicts with pid, kind, bin (actual, by support FVU), level, fvu, ``full`` (the sample) and,
    with ``with_base``, ``base`` -- the same sample built without its history (same rng path).
    Every problem also gets one none sample (bin None). A history sample whose problem was
    replaced (empty pool) or whose base has a different target is dropped and counted."""
    rows: list = []
    dropped = 0
    for pid in ids:
        rows.append(dict(pid=pid, kind="none", bin=None, level=None, fvu=None,
                         full=sampler.make("none", np.random.default_rng(spec_seed(sampler, pid, "none", None)),
                                           validation=True, pid=pid)))
        for kind, b, cand in eval_specs(sampler, pid):
            seed = spec_seed(sampler, pid, kind, b)
            kw = dict(validation=True, pid=pid, bin=b if kind == "near" else None, candidate=cand)
            full = sampler.make(kind, np.random.default_rng(seed), **kw)
            row = dict(pid=pid, kind=kind, bin=full["info"].get("history_bin"), level=full["info"].get("actual"),
                       fvu=full["info"].get("history_fvu"), target_bin=b, full=full)
            if full["info"].get("pid") != pid or not full["metadata"]["history_valid"]:
                dropped += 1
                continue
            if with_base:
                base = sampler.make(kind, np.random.default_rng(seed), drop_history=True, **kw)
                if not same_target(sampler.model, full, base):
                    dropped += 1
                    continue
                row["base"] = base
            rows.append(row)
    sampler.eval_dropped = dropped
    return rows


def same_target(model: Any, a: Any, b: Any) -> bool:
    def labels(s: Any) -> list:
        lab = collate([s], model.tokenizer)["labels"][0]
        return lab[lab != model.tokenizer["<pad>"]].tolist()
    return labels(a) == labels(b)


def target_ces(model: Any, samples: list, micro_batch: int) -> list:
    ces: list = []
    for begin in range(0, len(samples), micro_batch):
        _, metrics = loss(model, collate(samples[begin: begin + micro_batch], model.tokenizer))
        ces += metrics["ce_rows"]
    return ces


def group_name(kind: str, b: Any) -> str:
    return kind if b is None else f"{kind} {BIN_NAMES[int(b)]}"


def eval_loss(model: Any, rows: list, fractions: dict, micro_batch: int) -> dict:
    """Target CE on the fixed evaluation set: per kind (``ce_<kind>``), per kind and FVU bin
    (``ce_<kind>_b<bin>``) and the mix-weighted sum ``loss`` (reported, not used to pick a
    checkpoint: round 4 compares fixed steps). Also the input-weight norms of the data columns,
    to watch the history column grow."""
    ces = target_ces(model, [r["full"] for r in rows], micro_batch)
    result: dict[str, Any] = {}
    for kind in KINDS:
        vals = [c for r, c in zip(rows, ces) if r["kind"] == kind]
        if vals:
            result[f"ce_{kind}"] = float(np.mean(vals))
        for b in range(3):
            vals = [c for r, c in zip(rows, ces) if r["kind"] == kind and r["bin"] == b]
            if vals:
                result[f"ce_{kind}_b{b}"] = float(np.mean(vals))
    result["loss"] = float(sum(fractions.get(k, 0) * result.get(f"ce_{k}", 0.0) for k in KINDS))
    result["n"] = {group_name(r["kind"], r["bin"]): 0 for r in rows}
    for r in rows:
        result["n"][group_name(r["kind"], r["bin"])] += 1
    result.update(column_norms(model))
    return result


def column_norms(model: Any) -> dict:
    """Frobenius norms of the encoder's input weights per data column group: x (mean over the
    x columns), y and the history column (the last one)."""
    w = model.encoder.embedding.weight.detach()
    e = model.pre_encoder.encoding_size
    cols = w.reshape(w.shape[0], -1, e)
    return dict(norm_x=float(cols[:, :-2].norm(dim=(0, 2)).mean()), norm_y=float(cols[:, -2].norm()),
                norm_history=float(cols[:, -1].norm()))
