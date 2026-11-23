"""CPU micro-batch training with explicit, replayable checkpoint state."""

from typing import Any
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from common import (
    ROOT,
    dump,
    seed_all,
    eval_seed,
    train_seed,
    assemble,
    collate,
    loss,
    read_problem,
    read_pool,
)
from flash_ansr.train.optimizers import build_optimizer
from flash_ansr.utils.config_io import load_config


def warmup_constant(warmup: Any, cooldown: Any = None) -> Any:
    """Learning-rate factor after ``step`` completed updates: linear warmup to 1 over ``warmup``
    updates, then constant; ``cooldown=(start, end)`` decays linearly to 0 between those steps."""

    def factor(step: Any) -> float:
        if step < warmup:
            return step / warmup
        if cooldown is not None and step >= cooldown[0]:
            start, end = cooldown
            return max(0.0, (end - step) / (end - start))
        return 1.0

    return factor


def lr_factor(step: Any, total: Any) -> Any:
    """The pilot schedule (kept for the pilot's artifacts)."""
    warm = 2 if total == 20 else 10
    decay = int(0.8 * total)
    if step < warm:
        return step / warm
    if step <= decay:
        return 1.0
    return max(0.0, (total - step) / (total - decay))


def checkpoint(
    path: Any, m: Any, opt: Any, sched: Any, step: Any, rng: Any, usage: Any, group: Any
) -> Any:
    state = dict(
        model=m.state_dict(),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        step=step,
        torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        sample_rng=rng.bit_generator.state,
        usage=dict(usage),
        group=group,
    )
    torch.save(state, path)


def restore(path: Any, m: Any, opt: Any, sched: Any, rng: Any) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    rng.bit_generator.state = state["sample_rng"]
    return state


def train(
    cfg: Any,
    out: Any,
    m: Any,
    wc: Any,
    catalog: Any,
    group: Any,
    ids: Any,
    val_ids: Any,
    total: Any,
    phase: Any,
    *,
    schedule: Any = None,
    checkpoints: Any = None,
    resume_check_step: Any = "default",
    resume: Any = None,
    folder: Any = None,
    on_checkpoint: Any = None,
    on_step: Any = None,
    batch_maker: Any = None,
    validator: Any = None,
) -> Any:
    """Train ``group`` to ``total`` updates.

    ``schedule(completed_steps) -> lr factor`` (default: the pilot schedule); ``checkpoints``:
    steps saved in full (default: half-way and the end); ``resume``: a checkpoint to continue
    from (its weights, optimizer, RNG and sample position; the schedule is this call's);
    ``on_checkpoint(step)``: called after each checkpoint's validation.

    ``batch_maker(step, rng) -> samples`` replaces the pool sampling (round 3): each sample
    carries ``kind`` and a JSON-able ``info``; per-kind target CE is logged as ``ce_<kind>``.
    ``validator(step) -> dict`` (with a ``loss`` key) replaces the validation-set loss; it is
    then also the only use of ``val_ids`` (pools are loaded lazily, only for pool samples).
    """
    folder = Path(folder) if folder is not None else out / phase / group
    folder.mkdir(parents=True, exist_ok=True)
    if checkpoints is None:
        checkpoints = {total // 2, total}
    if resume_check_step == "default":
        resume_check_step = total // 2
    m.load_state_dict(torch.load(out / "step0.pt", weights_only=True))
    seed_all(train_seed(cfg))
    spec = load_config(str(ROOT / cfg["train_recipe"]))["optimizer"]
    spec = copy.deepcopy(spec)
    spec["kwargs"]["lr"] = 1e-4
    opt = build_optimizer(spec, m)
    factor = schedule if schedule is not None else (lambda step: lr_factor(step, total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    rng = np.random.default_rng(train_seed(cfg))
    usage: Counter[str] = Counter()
    first = 1
    if resume is not None:
        state = restore(resume, m, opt, sched, rng)
        first = int(state["step"]) + 1
        usage.update(state["usage"])
        # LambdaLR stores no lambda: re-apply this call's schedule at the restored position.
        for group_lr, base in zip(opt.param_groups, sched.base_lrs):
            group_lr["lr"] = base * factor(state["step"])
    reasons: Counter[str] = Counter()
    planned = actual = 0
    problems: dict = {}
    pools: dict = {}
    if batch_maker is None:
        problems = {pid: read_problem(out, pid) for pid in ids + val_ids}
        pools = {
            pid: read_pool(out, pid, "random" if group == "B" else "model")
            for pid in ids + val_ids
        }

    def make(pid: Any, seed: Any, validation: Any = False) -> Any:
        pool, yh = pools[pid]
        return assemble(
            cfg, m, wc, catalog, problems[pid], pool, yh, seed, group, validation
        )

    def validate(step: Any) -> Any:
        if validator is not None:
            m.eval()
            with torch.no_grad():
                result = validator(step)
            dump(folder / f"validation_{step}.json", dict(step=step, **result))
            return
        m.eval()
        values = []
        with torch.no_grad():
            for pid in val_ids:
                sample = make(pid, eval_seed(cfg, pid), True)
                _, metrics = loss(m, collate([sample], m.tokenizer))
                values.append(metrics["loss"])
        dump(
            folder / f"validation_{step}.json",
            dict(step=step, loss=float(np.mean(values)), n=len(values)),
        )

    if resume is None:
        validate(0)
        if on_checkpoint is not None:
            on_checkpoint(0)
    start = time.perf_counter()
    resume_checked = False
    with (
        (folder / "metrics.jsonl").open("a" if resume is not None else "w") as log,
        (folder / "samples.jsonl").open("a" if resume is not None else "w") as sample_log,
    ):
        for step in range(first, total + 1):
            m.train()
            opt.zero_grad(set_to_none=True)
            values = []
            t0 = time.perf_counter()
            if batch_maker is not None:
                made = batch_maker(step, rng)
                if len(made) != cfg["batch_size"]:
                    raise ValueError(f"batch_maker returned {len(made)} samples")
            else:
                picks = rng.choice(ids, size=cfg["batch_size"], replace=True)
                seeds = rng.integers(0, 2**32 - 1, size=len(picks))
                made = []
                for pid, seed in zip(picks, seeds):
                    s = make(int(pid), int(seed))
                    s["kind"] = "pool"
                    s["info"] = dict(pid=int(pid), seed=int(seed), candidate=s["chosen"])
                    made.append(s)
            by_kind: dict[str, list[float]] = {}
            for begin in range(0, len(made), cfg["micro_batch"]):
                samples = made[begin: begin + cfg["micro_batch"]]
                for s in samples:
                    planned += int(s.get("planned_history", s["metadata"]["history_valid"]))
                    actual += int(s["metadata"]["history_valid"])
                    reasons[s["metadata"]["history_reason"]] += 1
                    sample_log.write(
                        json.dumps(
                            dict(
                                step=step,
                                kind=s["kind"],
                                **s["info"],
                                planned_k=int(s.get("planned_history", s["metadata"]["history_valid"])),
                                actual_k=int(s["metadata"]["history_valid"]),
                                unconditional=not s["metadata"].get(
                                    "condition_mask", True
                                ),
                                reason=s["metadata"]["history_reason"],
                                sequence_length=len(s["input_ids"]),
                            )
                        )
                        + "\n"
                    )
                    if s.get("chosen") is not None and "pid" in s["info"]:
                        usage[f'{s["info"]["pid"]}:{s["chosen"]}'] += 1
                batch = collate(samples, m.tokenizer)
                value, metrics = loss(m, batch)
                for s, ce in zip(samples, metrics.pop("ce_rows")):
                    by_kind.setdefault(s["kind"], []).append(ce)
                if not torch.isfinite(value):
                    raise FloatingPointError(f"nonfinite loss {group}/{step}")
                (value / (cfg["batch_size"] // cfg["micro_batch"])).backward()
                values.append(metrics)
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            opt.step()
            sched.step()
            if not all(torch.isfinite(p).all() for p in m.parameters()):
                raise FloatingPointError("nonfinite weights")
            row = dict(
                step=step,
                seconds=time.perf_counter() - t0,
                grad_norm=norm.item(),
                lr=opt.param_groups[0]["lr"],
                **{k: float(np.mean([v[k] for v in values])) for k in values[0]},
                **({f"ce_{k}": float(np.mean(v)) for k, v in by_kind.items()}
                   if batch_maker is not None else {}),
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
            if on_step is not None:
                on_step(row)
            print(f"{phase} {group} {row}", flush=True)
            sample_log.flush()
            if step in checkpoints:
                checkpoint(
                    folder / f"checkpoint_{step}.pt",
                    m,
                    opt,
                    sched,
                    step,
                    rng,
                    usage,
                    group,
                )
                if step == resume_check_step:
                    probe = collate([make(ids[0], 123)], m.tokenizer)
                    m.eval()
                    with torch.no_grad():
                        before = loss(m, probe)[0].clone()
                    before_weights = {k: v.clone() for k, v in m.state_dict().items()}
                    restore(
                        folder / f"checkpoint_{step}.pt", m, opt, sched, rng
                    )
                    with torch.no_grad():
                        after = loss(m, probe)[0]
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                    assert all(
                        torch.equal(v, m.state_dict()[k])
                        for k, v in before_weights.items()
                    )
                    # The probe consumed no sampling RNG; restore all stochastic states again.
                    restore(folder / f"checkpoint_{step}.pt", m, opt, sched, rng)
                    resume_checked = True
                    # Replay an actual stochastic update twice, including optimizer moments,
                    # scheduler, dropout RNG and sample selection, then undo the probe.
                    replay = []
                    for _ in range(2):
                        restore(folder / f"checkpoint_{step}.pt", m, opt, sched, rng)
                        probe_pid = int(rng.choice(ids))
                        probe_seed = int(rng.integers(2**32 - 1))
                        m.train()
                        opt.zero_grad(set_to_none=True)
                        probe_loss, _ = loss(
                            m, collate([make(probe_pid, probe_seed)], m.tokenizer)
                        )
                        probe_loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                        opt.step()
                        sched.step()
                        replay.append(
                            (
                                probe_pid,
                                probe_seed,
                                {k: v.clone() for k, v in m.state_dict().items()},
                            )
                        )
                    assert replay[0][:2] == replay[1][:2]
                    for key, value in replay[0][2].items():
                        torch.testing.assert_close(
                            value, replay[1][2][key], rtol=0, atol=0
                        )
                    restore(folder / f"checkpoint_{step}.pt", m, opt, sched, rng)
                    dump(
                        folder / "resume_check.json",
                        dict(step=step, stochastic_update_replay=True),
                    )
                validate(step)
                if on_checkpoint is not None:
                    on_checkpoint(step)
    report = dict(
        group=group,
        steps=total,
        first_step=first,
        seconds=time.perf_counter() - start,
        planned_history=planned,
        actual_history=actual,
        history_fraction=actual / ((total - first + 1) * cfg["batch_size"]),
        reasons=dict(reasons),
        usage=dict(usage),
        resume_checked=resume_checked,
    )
    if group != "A" and actual == 0:
        raise AssertionError("All samples fell back to K=0")
    dump(folder / "summary.json", report)
    return report
