"""Did stage 2 refine its history, or find the answer on its own? (exploratory, reads existing results)

For a two-stage condition, lists the problems that stage 2 newly solved (stage-1 top-1 wrong,
stage-2 top-1 right) and how far each stage-2 answer is from the history it was given:
skeleton edit distance (operators, variables, constant slots) divided by the longer length.
Also counts the draft "修正成功率" (0-实验规则.md §13): problems whose stage-1 top-1 is wrong and
whose history fits well (support FVU < 1e-3), solved by stage 2 within the distance threshold.

No simplification yet: two ways of writing the same formula count as different, so distances
are an upper bound.

Usage: python refine_distance.py RUN_DIR [--two self_C_two] [--direct self_C_direct]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS = float(np.finfo(np.float32).eps)
CLOSE_FVU = 1e-3
THRESHOLDS = (0.1, 0.2, 0.3)


def skeleton(prefix: list) -> list:
    out = []
    for t in prefix:
        if t == "<constant>":
            out.append("<c>")
            continue
        try:
            float(t)
            out.append("<c>")
        except ValueError:
            out.append(t)
    return out


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def load(path: Path) -> dict:
    return {r["pid"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--two", default="self_C_two")
    ap.add_argument("--direct", default="self_C_direct")
    args = ap.parse_args()
    two = load(args.run_dir / "final" / f"{args.two}.jsonl")
    direct = load(args.run_dir / "final" / f"{args.direct}.jsonl")

    rows, close = [], []
    for pid, r in two.items():
        h = r.get("history")
        if not r.get("phase0") or not r.get("phase1") or not h:
            continue
        top0, top1 = r["phase0"]["top1_val"], r["phase1"]["top1_val"]
        dist = None
        if r["phase1"]["top1"] is not None:
            hs, ss = skeleton(h["expression_prefix"]), skeleton(r["phase1"]["top1"]["skeleton_prefix"])
            dist = edit_distance(hs, ss) / max(len(hs), len(ss))
        if not top0 and h["fvu"] is not None and h["fvu"] < CLOSE_FVU:
            close.append((pid, top1, dist))
        if top1 and not top0:
            rows.append(dict(pid=pid, history_fvu=h["fvu"], distance=dist,
                             direct_solved=direct[pid]["merged"]["top1_val"]))

    print(f"stage 2 newly solved: {len(rows)}")
    print("pid | history support FVU | distance | direct also solved")
    for x in rows:
        print(f"{x['pid']} | {x['history_fvu']:.1e} | {x['distance']:.0%} | {x['direct_solved']}")
    if rows:
        print(f"median distance: {np.median([x['distance'] for x in rows]):.0%}")
    print(f"\ndraft 修正成功率: denominator (stage-1 wrong, history FVU < {CLOSE_FVU:g}) = {len(close)}")
    for t in THRESHOLDS:
        k = sum(1 for _, ok, d in close if ok and d is not None and d <= t)
        print(f"  distance <= {t:.0%}: {k}/{len(close)}")
    solved_close = sum(1 for _, ok, _ in close if ok)
    print(f"  stage 2 solved at any distance: {solved_close}/{len(close)}")
    print(f"  direct generation solved: {sum(direct[p]['merged']['top1_val'] for p, _, _ in close)}/{len(close)}")


if __name__ == "__main__":
    main()
