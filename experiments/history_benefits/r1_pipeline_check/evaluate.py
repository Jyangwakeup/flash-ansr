"""Small generation/fitting gate using the existing estimator's refinement and scoring."""

from typing import Any
import time
from dataclasses import replace
import numpy as np
import torch
from flash_ansr import FlashANSR, SoftmaxSamplingConfig
from flash_ansr.flash_ansr import Generation
from flash_ansr.preprocessing import PromptPrefix
from common import dump, eval_seed, fit_seed, history_ids, read_pool, read_problem, seed_all
from pool import result_rows
from flash_ansr.refine import refinement_slots
from symbolic_data.token_ops import tagged_canonical


def cache_replay(wrapper: Any, gen: Any, pool: Any, candidates: Any, seed: Any) -> Any:
    """Independent second refinement pass: cache is local to this problem/settings.

    Preserve typed literals in the key. Exact token equality is conservative: equivalent
    but differently written structures are misses, never an unsafe cross-formula hit.
    """

    def key(expression: Any) -> Any:
        # History serialization canonicalizes the fitted formula. Apply the same
        # conversion to both cache entries and proposals before comparing them.
        canonical = tagged_canonical(wrapper.simplipy_engine, list(expression))
        tokens = list(wrapper.simplipy_engine.to_prefix(canonical))
        for index in refinement_slots(tokens, wrapper.simplipy_engine, "fittable"):
            tokens[index] = "<constant>"
        return tuple(tokens)

    start = time.perf_counter()
    cache = {key(row["expression_prefix"]): row for row in pool}
    reused = []
    misses = []
    paired = []
    for beam, lp in zip(gen.raw_beams, gen.log_probs):
        expr_ids = wrapper.tokenizer.extract_expression_from_beam(beam)[0]
        try:
            realized = wrapper.flash_ansr_model._realize_ieee754_spans(expr_ids)
            explicit = wrapper._ensure_explicit_dialect(realized) if realized else None
            hit = cache.get(key(explicit)) if explicit else None
        except (ValueError, TypeError, IndexError):
            hit = None
        if hit is None:
            misses.append((beam, lp))
        else:
            reused.append(dict(hit, cache_reused=True))
    remaining = replace(
        gen, raw_beams=[b for b, _ in misses], log_probs=[lp for _, lp in misses]
    )
    outcome = wrapper._fit_refine(remaining, refine_seed=seed, allow_empty=True)
    result = wrapper._build_result(remaining, outcome)
    fresh = [r for r, _ in result_rows(result, gen.X, gen.y, "model")]
    for row in candidates:
        old = cache.get(key(row["expression_prefix"]))
        if old is not None:
            paired.append(dict(old_fvu=old["fvu"], refitted_fvu=row["fvu"]))
    return dict(
        hits=len(reused),
        misses=len(misses),
        seconds=time.perf_counter() - start,
        candidates=reused + fresh,
        repeat_fit_comparison=paired,
        key_scope="same problem data and refinement settings; fittable slots masked, typed constants retained",
    )


def evaluate(cfg: Any, out: Any, model: Any, group: Any, ids: Any, phase: Any) -> Any:
    wrapper = FlashANSR(
        model.simplipy_engine,
        model,
        model.tokenizer,
        generation_config=SoftmaxSamplingConfig(
            draws=cfg["evaluation_draws"], emission="fittable"
        ),
        refine={"n_restarts": cfg["restarts"]},
        ranking="mdl",
        compute={"device": "cpu", "workers": cfg["workers"]},
    )
    rows = []
    model.eval()
    try:
        for pid in ids:
            p = read_problem(out, pid)
            pool, yh = read_pool(out, pid, "model")
            chosen = (
                int(
                    np.random.default_rng(eval_seed(cfg, pid)).integers(
                        min(10, len(pool))
                    )
                )
                if pool
                else None
            )
            for requested in ([0] if group == "A" else [0, 1]):
                h = (
                    history_ids(model, pool[chosen])
                    if requested and chosen is not None
                    else []
                )
                tok = model.tokenizer
                prefix = [tok["<bos>"], tok["<mask_fittable>"], *h, tok["<expression>"]]
                reason = "included" if h else ("empty_pool" if requested else "absent")
                if len(prefix) + cfg["generation_tokens"] > cfg["max_length"]:
                    h = []
                    prefix = [tok["<bos>"], tok["<mask_fittable>"], tok["<expression>"]]
                    reason = "length"
                data = torch.tensor(
                    np.column_stack(
                        (
                            p.x_support,
                            p.y_support_noisy,
                            yh[chosen] if h else np.zeros(len(p.x_support)),
                        )
                    ),
                    dtype=torch.float64,
                ).unsqueeze(0)
                seed_all(eval_seed(cfg, pid))
                t0 = time.perf_counter()
                prompt = PromptPrefix(
                    prefix, [float("nan")] * len(prefix), [True] * len(prefix), {}
                )
                with torch.no_grad():
                    memory = model._create_memory(
                        data, history_mask=torch.tensor([bool(h)])
                    )
                    beams, logs = model.sample_top_kp(
                        data,
                        draws=cfg["evaluation_draws"],
                        max_len=len(prefix) + cfg["generation_tokens"],
                        batch_size=cfg["evaluation_draws"],
                        return_raw=True,
                        static_decode=False,
                        use_cache=True,
                        prompt_prefix=prompt,
                        memory=memory,
                    )
                raw_beams, raw_logs = beams, logs
                accepted = []
                rejections = []
                for index, (beam, logprob) in enumerate(zip(beams, logs)):
                    try:
                        tok.extract_expression_from_beam(beam)
                    except (ValueError, IndexError) as exc:
                        rejections.append(dict(index=index, reason=str(exc)))
                        continue
                    accepted.append((beam, logprob))
                beams = [b for b, _ in accepted]
                logs = [v for _, v in accepted]
                gen = Generation(
                    beams,
                    logs,
                    p.x_support,
                    p.y_support_noisy,
                    float(np.var(p.y_support_noisy)),
                    len(p.x_support),
                    prompt,
                    memory,
                    torch.device("cpu"),
                    {},
                    None,
                    time.perf_counter() - t0,
                    cfg["evaluation_draws"],
                    eval_seed(cfg, pid),
                )
                outcome = wrapper._fit_refine(
                    gen, refine_seed=fit_seed(cfg, pid), allow_empty=True
                )
                result = wrapper._build_result(gen, outcome)
                candidates = [
                    r
                    for r, _ in result_rows(
                        result, p.x_support, p.y_support_noisy, "model"
                    )
                ]
                original_seconds = time.perf_counter() - t0
                cached = cache_replay(wrapper, gen, pool, candidates, fit_seed(cfg, pid))
                merged = [dict(r, origin="phase0") for r in pool] + [
                    dict(r, origin="phase1") for r in candidates
                ]
                merged.sort(
                    key=lambda r: r["score"] if r["score"] is not None else float("inf")
                )
                keys = {tuple(r["skeleton_prefix"]) for r in pool}
                row = dict(
                    pid=pid,
                    group=group,
                    requested_history=requested,
                    actual_history=bool(h),
                    reason=reason,
                    draws=cfg["evaluation_draws"],
                    raw_count=len(raw_beams),
                    raw_beams=raw_beams,
                    raw_log_probs=raw_logs,
                    extraction_failures=rejections,
                    candidates=candidates,
                    merged_candidates=merged,
                    top1=merged[0] if merged else None,
                    structural_repeats=sum(
                        tuple(r["skeleton_prefix"]) in keys for r in candidates
                    ),
                    seconds=original_seconds,
                    cache_replay=cached,
                )
                rows.append(row)
                dump(out / phase / group / "evaluation.json", rows)
                print(
                    f"eval {phase}/{group}/{pid}/K{requested}: {len(candidates)} fitted",
                    flush=True,
                )
    finally:
        wrapper.close()
    return rows
