"""Persist paired problems and FVU-ranked random/model history pools."""

from typing import Any
import time
from pathlib import Path
import numpy as np
from flash_ansr import SoftmaxSamplingConfig
from flash_ansr.utils.generation import PriorSamplingConfig
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns
from common import VALIDATION_ID_OFFSET, ROOT, dump, fit_seed, load_base, source, split_seeds


def fvu(y: Any, pred: Any) -> Any:
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if not np.isfinite(pred).all():
        return float("inf")
    var = np.var(y)
    err = np.mean((y - pred) ** 2)
    return float(err / var) if var else (0.0 if err == 0 else float("inf"))


def draw_problems(cfg: Any, split: Any, count: Any) -> Any:
    """The first ``count`` real problems of one split's seeded stream, in order."""
    src, _ = source(cfg, split)
    iterator = iter(src)
    problems: list[Any] = []
    while len(problems) < count:
        problem = next(iterator)
        if not problem.is_placeholder:
            problems.append(problem)
    return problems


def split_of(cfg: Any, pid: Any) -> tuple[str, int]:
    """(split, index within the split) of a problem id.

    Split-seed configs: train ids are 0.., validation ids VALIDATION_ID_OFFSET.., independent of
    the training-set size. Legacy configs: train ids first, then validation.
    """
    if split_seeds(cfg):
        if pid >= VALIDATION_ID_OFFSET:
            return "validation", int(pid) - VALIDATION_ID_OFFSET
        if pid >= cfg["train_problems"]:
            raise ValueError(f"problem {pid} is past the {cfg['train_problems']} training problems")
        return "train", int(pid)
    if pid < cfg["train_problems"]:
        return "train", int(pid)
    return "validation", int(pid) - int(cfg["train_problems"])


def prepare(cfg: Any, out: Any, ids: Any) -> Any:
    out = Path(out)
    (out / "problems").mkdir(parents=True, exist_ok=True)
    if split_seeds(cfg):
        wanted = [pid for pid in ids if not (out / "problems" / f"{pid}.json").exists()]
        by_split: dict[str, list[int]] = {}
        for pid in wanted:
            split, index = split_of(cfg, pid)
            by_split.setdefault(split, []).append(index)
        for split, indices in by_split.items():
            drawn = draw_problems(cfg, split, max(indices) + 1)
            offset = 0 if split == "train" else VALIDATION_ID_OFFSET
            for index in indices:
                save_problem(cfg, out, offset + index, drawn[index])
        return None
    src, _ = source(cfg)
    iterator = iter(src)
    for pid in range(max(ids) + 1):
        problem = next(iterator)
        while problem.is_placeholder:
            problem = next(iterator)
        if pid not in ids or (out / "problems" / f"{pid}.json").exists():
            continue
        save_problem(cfg, out, pid, problem)
    return src


def save_problem(cfg: Any, out: Any, pid: Any, problem: Any) -> None:
    split, index = split_of(cfg, pid)
    x = problem.x_support.copy()
    x_val = problem.x_validation
    has_val = x_val is not None and len(x_val) > 0
    x_val = x_val.copy() if has_val else np.empty((0, x.shape[1]))
    mask_unused_variable_columns(
        arrays=(x, x_val) if has_val else (x,),
        variables=list(problem.variables),
        skeleton_tokens=list(problem.skeleton),
        padding="zero",
    )
    np.savez_compressed(
        out / "problems" / f"{pid}.npz",
        x=x,
        y=problem.y_support_noisy,
        clean=problem.y_support,
        outlier=(
            problem.outlier_mask_support
            if problem.outlier_mask_support is not None
            else np.zeros(len(x))
        ),
        # held-out split (srbf protocol): never used for generation, fitting or ranking
        x_validation=x_val,
        y_validation=problem.y_validation if has_val else np.empty(0),
    )
    dump(
        out / "problems" / f"{pid}.json",
        dict(
            problem_id=pid,
            split=split,
            split_index=index,
            expression=list(problem.expression),
            skeleton=list(problem.skeleton),
            constants=list(problem.constants),
            variables=list(problem.variables),
            noise=problem.noise,
        ),
    )


def result_rows(result: Any, x: Any, y: Any, kind: Any) -> Any:
    records = []
    for rank, c in enumerate(result.candidates):
        try:
            pred = np.asarray(result.predict(x, rank=rank)).reshape(-1)
        except Exception:
            continue
        if not np.isfinite(pred).all():
            continue
        records.append(
            (
                dict(
                    expression_prefix=list(c.expression_prefix),
                    skeleton_prefix=list(c.skeleton_prefix),
                    constants=[float(v) for v in c.constants],
                    fvu=fvu(y, pred),
                    score=float(c.score),
                    log_prob=float(c.log_prob) if kind != "random" else None,
                    rank=rank,
                ),
                pred,
            )
        )
    records.sort(key=lambda item: (item[0]["fvu"], str(item[0]["expression_prefix"])))
    seen = set()
    selected = []
    for row, pred in records:
        key = tuple(row["skeleton_prefix"])
        if key in seen:
            continue
        seen.add(key)
        selected.append((row, pred))
    return selected


def build(cfg: Any, out: Any, ids: Any, rebuild: Any = False, kinds: Any = ("random", "model")) -> Any:
    from common import read_problem

    out = Path(out)
    if not all((out / "problems" / f"{pid}.json").exists() for pid in ids):
        prepare(cfg, out, ids)
    if not rebuild and all(
        (out / "pools" / kind / f"{pid}.json").exists()
        for kind in kinds
        for pid in ids
    ):
        return
    model = load_base(cfg)
    try:
        for kind in kinds:
            model.generation_config = (
                PriorSamplingConfig(
                    draws=cfg["draws"],
                    max_tries=cfg["draws"],
                    catalog=str(ROOT / "configs/v25.0-T8.1-3M/catalog_train.yaml"),
                    match_variables=True,
                )
                if kind == "random"
                else SoftmaxSamplingConfig(draws=cfg["draws"], emission="fittable")
            )
            folder = out / "pools" / kind
            folder.mkdir(parents=True, exist_ok=True)
            for pid in ids:
                dest = folder / f"{pid}.json"
                if dest.exists() and not rebuild:
                    continue
                p = read_problem(out, pid)
                start = time.perf_counter()
                error = None
                try:
                    result = model.fit(
                        p.x_support,
                        p.y_support_noisy,
                        seed=fit_seed(cfg, pid),
                        on_empty="return",
                    )
                    # keep: null keeps every skeleton-unique candidate (formal rules);
                    # history selection still reads only the first history_top of them.
                    rows = result_rows(result, p.x_support, p.y_support_noisy, kind)[
                        : cfg.get("keep")
                    ]
                    status_counts: dict[str, int] = {}
                    for status in result.ledger.fit_status:
                        status_counts[str(status)] = (
                            status_counts.get(str(status), 0) + 1
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    rows = []
                    status_counts = {}
                duration = time.perf_counter() - start
                predictions = (
                    np.stack([v for _, v in rows])
                    if rows
                    else np.empty((0, len(p.x_support)))
                )
                np.savez_compressed(dest.with_suffix(".npz"), predictions=predictions)
                dump(
                    dest,
                    dict(
                        problem_id=pid,
                        kind=kind,
                        attempt_budget=cfg["draws"],
                        seconds=duration,
                        seed=fit_seed(cfg, pid),
                        error=error,
                        status_counts=status_counts,
                        candidates=[r for r, _ in rows],
                    ),
                )
                print(
                    f"pool {kind} {pid}: {len(rows)} candidates {duration:.2f}s error={error}",
                    flush=True,
                )
    finally:
        model.close()
