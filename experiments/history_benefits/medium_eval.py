"""Medium-scale evaluation (训练评估流程.md §6–§7; 正式实验规则.md §1–§2).

Every candidate is scored on the support split (fNRR) and on the held-out split (vNRR), which
never enters generation, fitting or ranking. Success: FVU <= float32 eps (srbf is_perfect_fit).
"""

from typing import Any
import time

import numpy as np
import torch
from flash_ansr import FlashANSR, SoftmaxSamplingConfig
from flash_ansr.flash_ansr import Generation
from flash_ansr.inference import evaluate_prefix
from flash_ansr.preprocessing import PromptPrefix

from common import eval_seed, fit_seed, history_ids, load_base, read_pool, read_problem, seed_all
from pool import fvu

EPS32 = float(np.finfo(np.float32).eps)

#: The six final-evaluation conditions (流程 §7.2): name -> (group, history source or None).
CONDITIONS = {
    "1_A_none": ("A", None),
    "2_C_model": ("C", "model"),
    "3_B_random": ("B", "random"),
    "4_C_none": ("C", None),
    "5_B_none": ("B", None),
    "6_B_model": ("B", "model"),
}


def wrapper_for(cfg: Any, model: Any) -> Any:
    return FlashANSR(
        model.simplipy_engine,
        model,
        model.tokenizer,
        generation_config=SoftmaxSamplingConfig(draws=1, emission="fittable"),
        refine={"n_restarts": cfg["restarts"]},
        ranking="mdl",
        compute={"device": "cpu", "workers": cfg["workers"]},
    )


def held_out_fvu(engine: Any, row: Any, problem: Any) -> float:
    """FVU of a fitted candidate on the held-out split; inf when it cannot be evaluated."""
    if problem.x_validation is None or len(problem.x_validation) == 0:
        return float("nan")
    try:
        pred = evaluate_prefix(
            engine, row["expression_prefix"], problem.x_validation,
            n_variables=problem.x_validation.shape[1],
        )
    except Exception:  # noqa: BLE001 -- an unevaluable candidate is simply not a solution
        return float("inf")
    return fvu(problem.y_validation, pred)


def choose_history(cfg: Any, pid: Any, pool: Any) -> Any:
    """Index of the history candidate: uniform over the first history_top, by the eval seed."""
    if not pool:
        return None
    return int(np.random.default_rng(eval_seed(cfg, pid)).integers(min(cfg["history_top"], len(pool))))


def generate(cfg: Any, model: Any, wrapper: Any, problem: Any, pid: Any, history: Any, draws: Any) -> Any:
    """Draw ``draws`` expressions (optionally conditioned on one history candidate), fit them and
    return candidate rows plus timing. ``history`` is ``(row, support_predictions)`` or None."""
    tok = model.tokenizer
    h = history_ids(model, history[0]) if history is not None else []
    prefix = [tok["<bos>"], tok["<mask_fittable>"], *h, tok["<expression>"]]
    reason = "included" if h else "absent"
    if len(prefix) + cfg["generation_tokens"] > cfg["max_length"]:
        h, reason = [], "length"
        prefix = [tok["<bos>"], tok["<mask_fittable>"], tok["<expression>"]]
    x, y = problem.x_support, problem.y_support_noisy
    # the original T8 has no history slot: it reads (x, y) only and takes no history mask
    has_slot = getattr(model, "history_slots", 0) == 1
    if not has_slot and h:
        raise ValueError("a model without a history slot cannot take history")
    columns = [x, y] + ([history[1] if h else np.zeros(len(x))] if has_slot else [])
    data = torch.tensor(np.column_stack(columns), dtype=torch.float64).unsqueeze(0)
    seed_all(eval_seed(cfg, pid))
    t0 = time.perf_counter()
    prompt = PromptPrefix(prefix, [float("nan")] * len(prefix), [True] * len(prefix), {})
    with torch.no_grad():
        if has_slot:
            memory = model._create_memory(data, history_mask=torch.tensor([bool(h)]))
        else:
            memory = model._create_memory(data)
        beams, logs = model.sample_top_kp(
            data,
            draws=draws,
            max_len=len(prefix) + cfg["generation_tokens"],
            batch_size=min(draws, 256),
            return_raw=True,
            static_decode=False,
            use_cache=True,
            prompt_prefix=prompt,
            memory=memory,
        )
    accepted = []
    for beam, logprob in zip(beams, logs):
        try:
            tok.extract_expression_from_beam(beam)
        except (ValueError, IndexError):
            continue
        accepted.append((beam, logprob))
    t_gen = time.perf_counter() - t0
    gen = Generation(
        [b for b, _ in accepted], [v for _, v in accepted], x, y, float(np.var(y)), len(x),
        prompt, memory, torch.device("cpu"), {}, None, t_gen, draws, eval_seed(cfg, pid),
    )
    outcome = wrapper._fit_refine(gen, refine_seed=fit_seed(cfg, pid), allow_empty=True)
    result = wrapper._build_result(gen, outcome)
    rows = []
    for rank, c in enumerate(result.candidates):
        try:
            pred = np.asarray(result.predict(x, rank=rank)).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(pred).all():
            continue
        row = dict(
            expression_prefix=list(c.expression_prefix),
            skeleton_prefix=list(c.skeleton_prefix),
            constants=[float(v) for v in c.constants],
            score=float(c.score),
            fvu=fvu(y, pred),
            log_prob=float(c.log_prob),
        )
        row["fvu_val"] = held_out_fvu(model.simplipy_engine, row, problem)
        rows.append(row)
    return dict(
        candidates=rows,
        reason=reason,
        raw=len(beams),
        accepted=len(accepted),
        seconds_generate=t_gen,
        seconds_total=time.perf_counter() - t0,
    )


def outcome(candidates: Any) -> dict:
    """fNRR / vNRR of the score-ranked first candidate, and the held-out Oracle."""
    def key(r: Any) -> float:
        # a missing score (NaN, stored as null by dump) ranks last, as in the shipped ranking
        s = r.get("score")
        return float("inf") if s is None or s != s else float(s)

    ranked = sorted(candidates, key=key)
    top = ranked[0] if ranked else None
    return dict(
        n_candidates=len(candidates),
        top1_fit=bool(top is not None and top["fvu"] <= EPS32),
        top1_val=bool(top is not None and top["fvu_val"] <= EPS32),
        oracle_val=any(r["fvu_val"] <= EPS32 for r in candidates),
        top1=top,
    )


def pool_rows(cfg: Any, out: Any, pid: Any, problem: Any, engine: Any, kind: Any = "model") -> Any:
    """Stored pool candidates with their held-out FVU (phase 0 of the final evaluation)."""
    pool, yh = read_pool(out, pid, kind)
    rows = [dict(r, fvu_val=held_out_fvu(engine, r, problem)) for r in pool]
    return pool, yh, rows


def run_condition(cfg: Any, out: Any, model: Any, ids: Any, source: Any, draws: Any,
                  phase0: Any = True, log: Any = None) -> list:
    """One evaluation condition over ``ids``: optional history from ``source`` ('model' /
    'random' / None), ``draws`` new expressions, merged with the phase-0 model pool if asked."""
    model.eval()
    wrapper = wrapper_for(cfg, model)
    engine = model.simplipy_engine
    results = []
    try:
        for pid in ids:
            problem = read_problem(out, pid)
            phase0_rows = pool_rows(cfg, out, pid, problem, engine, "model")[2] if phase0 else []
            history = None
            chosen = None
            if source is not None:
                pool, yh = read_pool(out, pid, source)
                chosen = choose_history(cfg, pid, pool)
                if chosen is not None:
                    history = (pool[chosen], yh[chosen])
            new = generate(cfg, model, wrapper, problem, pid, history, draws)
            merged = (phase0_rows if phase0 else []) + new["candidates"]
            phase0_keys = {tuple(r["skeleton_prefix"]) for r in phase0_rows}
            history_row = history[0] if history is not None else None
            row = dict(
                pid=pid,
                history_source=source,
                history_index=chosen,
                history_reason=new["reason"] if source is not None and chosen is not None
                else ("empty_pool" if source is not None else "absent"),
                history=None if history_row is None else dict(
                    expression_prefix=history_row["expression_prefix"],
                    fvu=history_row["fvu"],
                    fvu_val=held_out_fvu(engine, history_row, problem),
                ),
                draws=draws,
                raw=new["raw"],
                accepted=new["accepted"],
                seconds_generate=new["seconds_generate"],
                seconds_total=new["seconds_total"],
                phase0=outcome(phase0_rows) if phase0 else None,
                phase1=outcome(new["candidates"]),
                merged=outcome(merged),
                phase1_new_skeletons=sum(
                    tuple(r["skeleton_prefix"]) not in phase0_keys for r in new["candidates"]
                ),
                phase1_same_as_history=sum(
                    history_row is not None
                    and r["skeleton_prefix"] == history_row.get("skeleton_prefix")
                    for r in new["candidates"]
                ),
            )
            results.append(row)
            if log is not None:
                log(row)
    finally:
        wrapper.close()
    return results


def run_original(cfg: Any, out: Any, ids: Any, draws: Any, log: Any = None) -> list:
    """The frozen original T8 through the shipped fit pipeline (same-draws baseline)."""
    base = load_base(dict(cfg, draws=draws))
    engine = base.simplipy_engine
    results = []
    try:
        for pid in ids:
            problem = read_problem(out, pid)
            t0 = time.perf_counter()
            result = base.fit(problem.x_support, problem.y_support_noisy,
                              seed=fit_seed(cfg, pid), on_empty="return")
            seconds = time.perf_counter() - t0
            rows = []
            for rank, c in enumerate(result.candidates):
                try:
                    pred = np.asarray(result.predict(problem.x_support, rank=rank)).reshape(-1)
                except Exception:  # noqa: BLE001
                    continue
                if not np.isfinite(pred).all():
                    continue
                r = dict(expression_prefix=list(c.expression_prefix),
                         skeleton_prefix=list(c.skeleton_prefix), score=float(c.score),
                         fvu=fvu(problem.y_support_noisy, pred))
                r["fvu_val"] = held_out_fvu(engine, r, problem)
                rows.append(r)
            row = dict(pid=pid, draws=draws, seconds_total=seconds, merged=outcome(rows))
            results.append(row)
            if log is not None:
                log(row)
    finally:
        base.close()
    return results


def summarize(rows: Any, part: Any = "merged") -> dict:
    """Rates over all problems (every problem stays in the denominator, 规则 §3)."""
    n = len(rows)
    get = lambda key: sum(bool(r[part] and r[part][key]) for r in rows)  # noqa: E731
    return dict(
        n=n,
        vNRR=get("top1_val") / n if n else float("nan"),
        fNRR=get("top1_fit") / n if n else float("nan"),
        oracle_val=get("oracle_val") / n if n else float("nan"),
        mean_seconds=float(np.mean([r["seconds_total"] for r in rows])) if n else float("nan"),
    )


def own_pool(engine: Any, problem: Any, candidates: Any) -> tuple[list, list]:
    """A model's own phase-0 candidates as a history pool, built like the stored pools:
    sorted by support FVU, one per skeleton, with their support predictions."""
    ranked = sorted(candidates, key=lambda r: (r["fvu"], str(r["expression_prefix"])))
    seen, rows, preds = set(), [], []
    for r in ranked:
        key = tuple(r["skeleton_prefix"])
        if key in seen:
            continue
        try:
            pred = np.asarray(evaluate_prefix(engine, r["expression_prefix"], problem.x_support,
                                              n_variables=problem.x_support.shape[1])).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(pred).all():
            continue
        seen.add(key)
        rows.append(r)
        preds.append(pred)
    return rows, preds


def run_self(cfg: Any, out: Any, model: Any, ids: Any, two_stage: Any, log: Any = None) -> list:
    """Full-pipeline evaluation (规则 §2.3): the model alone, no frozen-T8 phase.

    two_stage: phase 0 = phase0_draws without history; history = one of the first history_top
    of the model's OWN phase-0 candidates (support FVU order, eval seed); phase 1 = phase1_draws
    with that history; merged and ranked. Otherwise: baseline_draws without history.
    """
    model.eval()
    wrapper = wrapper_for(cfg, model)
    engine = model.simplipy_engine
    results = []
    try:
        for pid in ids:
            problem = read_problem(out, pid)
            first = generate(cfg, model, wrapper, problem, pid, None,
                             cfg["phase0_draws"] if two_stage else cfg["baseline_draws"])
            row = dict(pid=pid, two_stage=bool(two_stage), seconds_total=first["seconds_total"],
                       phase0=outcome(first["candidates"]), raw0=first["raw"], accepted0=first["accepted"])
            merged = list(first["candidates"])
            if two_stage:
                pool, preds = own_pool(engine, problem, first["candidates"])
                chosen = choose_history(cfg, pid, pool)
                history = (pool[chosen], preds[chosen]) if chosen is not None else None
                second = generate(cfg, model, wrapper, problem, pid, history, cfg["phase1_draws"])
                merged += second["candidates"]
                keys = {tuple(r["skeleton_prefix"]) for r in first["candidates"]}
                hrow = history[0] if history is not None else None
                row.update(
                    history_index=chosen,
                    history_reason=second["reason"] if history is not None else "empty_pool",
                    history=None if hrow is None else dict(
                        expression_prefix=hrow["expression_prefix"], fvu=hrow["fvu"],
                        fvu_val=hrow["fvu_val"]),
                    phase1=outcome(second["candidates"]),
                    phase1_new_skeletons=sum(tuple(r["skeleton_prefix"]) not in keys
                                             for r in second["candidates"]),
                    phase1_same_as_history=sum(hrow is not None and r["skeleton_prefix"] == hrow["skeleton_prefix"]
                                               for r in second["candidates"]),
                    seconds_total=first["seconds_total"] + second["seconds_total"],
                )
            row["merged"] = outcome(merged)
            results.append(row)
            if log is not None:
                log(row)
    finally:
        wrapper.close()
    return results
