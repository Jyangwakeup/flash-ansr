"""Measured persistence/recomputation costs on exactly the same candidate pools."""

from typing import Any
import io
import json
import resource
import time
import tracemalloc
import numpy as np
from common import dump, read_problem, read_pool
from flash_ansr.inference import evaluate_prefix
from pool import build


def formula_constants_only(rows: Any) -> Any:
    return [
        {"expression_prefix": row["expression_prefix"], "constants": row["constants"]}
        for row in rows
    ]


def measure_persistence(
    cfg: Any,
    out: Any,
    model: Any,
    modes: tuple[str, ...] = ("full", "formula_constants"),
) -> Any:
    folder = out / "storage"
    folder.mkdir(parents=True, exist_ok=True)
    originals = []
    # Both strategies see the same already-loaded training data. Do not charge
    # only the slim strategy for rereading the problem's X from disk.
    inputs = {pid: read_problem(out, pid).x_support for pid in range(cfg["problems"])}
    for kind in ("random", "model"):
        for pid in range(cfg["problems"]):
            rows, yh = read_pool(out, pid, kind)
            originals.append((kind, pid, rows, yh))
    reports: dict[str, Any] = {}
    for mode in modes:
        dest = folder / mode
        dest.mkdir(exist_ok=True)
        start = time.perf_counter()
        paths = []
        for kind, pid, rows, yh in originals:
            path = dest / f"{kind}_{pid}.json"
            dump(path, rows if mode == "full" else formula_constants_only(rows))
            paths.append(path)
            if mode == "full":
                np.savez_compressed(path.with_suffix(".npz"), predictions=yh)
        write = time.perf_counter() - start
        results = []
        for repetitions in (1, 2, 10):
            tracemalloc.start()
            start = time.perf_counter()
            read_seconds = decode_seconds = recompute_seconds = 0.0
            for repeat in range(repetitions):
                for path, (_, pid, expected, yh) in zip(paths, originals):
                    t = time.perf_counter()
                    raw = path.read_bytes()
                    blob = (
                        path.with_suffix(".npz").read_bytes()
                        if mode == "full"
                        else None
                    )
                    read_seconds += time.perf_counter() - t
                    t = time.perf_counter()
                    rows = json.loads(raw)
                    if blob is not None:
                        with np.load(io.BytesIO(blob)) as a:
                            values = a["predictions"].copy()
                    decode_seconds += time.perf_counter() - t
                    assert rows == (
                        expected if mode == "full" else formula_constants_only(expected)
                    )
                    if mode != "full":
                        t = time.perf_counter()
                        values = (
                            np.stack(
                                [
                                    evaluate_prefix(
                                        model.simplipy_engine,
                                        r["expression_prefix"],
                                        inputs[pid],
                                    ).reshape(-1)
                                    for r in rows
                                ]
                            )
                            if rows
                            else np.empty_like(yh)
                        )
                        recompute_seconds += time.perf_counter() - t
                    np.testing.assert_allclose(values, yh, rtol=1e-10, atol=1e-10)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results.append(
                dict(
                    repetitions=repetitions,
                    total_seconds=time.perf_counter() - start,
                    read_seconds=read_seconds,
                    deserialize_seconds=decode_seconds,
                    recompute_seconds=recompute_seconds,
                    python_peak_bytes=peak,
                    process_lifetime_peak_rss=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                )
            )
        reports[mode] = dict(
            bytes=sum(p.stat().st_size for p in dest.iterdir()),
            write_seconds=write,
            reads=results,
        )
    return reports, originals


def compare(cfg: Any, out: Any, model: Any) -> Any:
    folder = out / "storage"
    reports, originals = measure_persistence(cfg, out, model)
    dump(folder / "costs.json", reports)
    rebuild = folder / "rebuild"
    rebuild.mkdir(exist_ok=True)
    # Reuse the immutable problems; regenerate both sources once, not merely refit saved formulas.
    link = rebuild / "problems"
    if not link.exists():
        link.symlink_to((out / "problems").resolve(), target_is_directory=True)
    t = time.perf_counter()
    build(cfg, rebuild, list(range(cfg["problems"])), rebuild=True)
    elapsed = time.perf_counter() - t
    identical = 0
    failures = []
    for kind, pid, rows, yh in originals:
        new, ny = read_pool(rebuild, pid, kind)
        same = [r["expression_prefix"] for r in new] == [
            r["expression_prefix"] for r in rows
        ]
        if same:
            try:
                np.testing.assert_allclose(ny, yh, rtol=1e-10, atol=1e-10)
            except AssertionError:
                same = False
        identical += int(same)
        if not same:
            failures.append([kind, pid])
    reports["regenerate"] = dict(
        measured_rebuild_seconds=elapsed,
        identical_pools=identical,
        total_pools=len(originals),
        different_pools=failures,
        estimated_seconds_by_replays={str(n): elapsed * n for n in (1, 2, 10)},
    )
    usage = {}
    for group in ("A", "B", "C"):
        path = out / f"pilot/{group}/summary.json"
        if path.exists():
            usage[group] = sum(json.loads(path.read_text())["usage"].values())
    reports["actual_training_history_uses"] = usage
    for mode in ("full", "formula_constants"):
        read = reports[mode]["reads"][0]["total_seconds"]
        write = reports[mode]["write_seconds"]
        reports[mode]["estimated_break_even_full_pool_replays"] = (
            write / (elapsed - read) if elapsed > read else None
        )
    reports["limitations"] = [
        "First read is not an OS cold-cache measurement.",
        "Both strategies share preloaded problem inputs; compiled formula functions are cached within the process.",
        "Python peak excludes native allocations; RSS is the process lifetime peak (bytes on macOS).",
        "Regenerate repeats 2 and 10 are linear estimates from one measured rebuild.",
        "Replay includes equality verification; compare component timings for storage alone.",
    ]
    dump(folder / "costs.json", reports)
    return reports
