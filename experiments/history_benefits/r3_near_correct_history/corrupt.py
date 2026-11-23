"""Near-correct histories: the true formula, slightly broken (round 3, 0-实验规则.md §12).

Levels (the requested level is drawn by the caller; what was actually done is recorded):

    exact            the true expression unchanged
    constants        every numeric literal x (1 ± U(0.1, 0.5)), except pow/rootn exponents
    structure_1      one structural edit
    structure_2_3    two or three structural edits

Structural edits on the realized prefix expression:

    variable   one variable leaf -> another variable of the problem
    operator   one operator -> another of the same arity
    drop       one non-leaf subtree -> a constant (its mean on the support points)
    wrap       one subtree -> unary(subtree)

A result must evaluate to finite values on the support points and pass ``check`` (e.g. the
history tokenizer accepts it); a non-exact result must differ from the truth (FVU > 1e-10 on
the clean support values), so the history is genuinely wrong.
"""

from typing import Any, Callable

import numpy as np
from flash_ansr.inference import evaluate_prefix

LEVELS = ("exact", "constants", "structure_1", "structure_2_3")
EDITS = ("variable", "operator", "drop", "wrap")
BINARY = ("+", "-", "*", "/", "pow")
WRONG_FVU = 1e-10


def is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def subtree_end(expr: list, start: int, arity: dict) -> int:
    """Index just past the subtree that starts at ``start``."""
    need, i = 1, start
    while need:
        need += arity.get(expr[i], 0) - 1
        i += 1
    return i


def evaluate(engine: Any, expr: list, x: np.ndarray) -> np.ndarray | None:
    try:
        pred = np.asarray(evaluate_prefix(engine, expr, x, n_variables=x.shape[1]), dtype=float)
    except Exception:  # noqa: BLE001 -- an unevaluable expression is simply rejected
        return None
    pred = np.broadcast_to(pred.reshape(-1) if pred.size > 1 else pred.reshape(()), (len(x),))
    return pred.copy() if np.isfinite(pred).all() else None


def fvu(y: np.ndarray, pred: np.ndarray) -> float:
    var = float(np.var(y))
    err = float(np.mean((np.asarray(y).reshape(-1) - pred) ** 2))
    return err / var if var else (0.0 if err == 0 else float("inf"))


def exponent_positions(expr: list, arity: dict) -> set:
    """Positions of numeric exponents (second argument of pow / rootn): scaling those turns
    x^2 into x^2.6, which is NaN for negative x -- a different kind of change."""
    out = set()
    for i, t in enumerate(expr):
        if t in ("pow", "rootn"):
            second = subtree_end(expr, i + 1, arity)
            if second < len(expr) and is_number(expr[second]):
                out.add(second)
    return out


def scale_constants(expr: list, rng: np.random.Generator, arity: dict) -> list | None:
    skip = exponent_positions(expr, arity)
    out, changed = [], False
    for i, t in enumerate(expr):
        if is_number(t) and i not in skip:
            factor = 1.0 + float(rng.choice([-1, 1])) * float(rng.uniform(0.1, 0.5))
            out.append(repr(float(t) * factor))
            changed = True
        else:
            out.append(t)
    return out if changed else None


def structural_edit(expr: list, kind: str, rng: np.random.Generator, engine: Any,
                    variables: list, x: np.ndarray) -> list | None:
    arity = engine.operator_arity
    ops = [i for i, t in enumerate(expr) if t in arity]
    if kind == "variable":
        leaves = [i for i, t in enumerate(expr) if t in variables]
        if not leaves:
            return None
        i = int(rng.choice(leaves))
        others = [v for v in variables if v != expr[i]]
        if not others:
            return None
        return expr[:i] + [str(rng.choice(others))] + expr[i + 1:]
    if kind == "operator":
        if not ops:
            return None
        i = int(rng.choice(ops))
        if arity[expr[i]] == 2:
            choices = [o for o in BINARY if o != expr[i] and o in arity]
        else:
            choices = [o for o, n in arity.items() if n == 1 and o != expr[i]]
        return expr[:i] + [str(rng.choice(choices))] + expr[i + 1:]
    if kind == "drop":
        inner = [i for i in ops if i != 0]  # never the whole expression
        if not inner:
            return None
        i = int(rng.choice(inner))
        end = subtree_end(expr, i, arity)
        part = evaluate(engine, expr[i:end], x)
        value = float(np.mean(part)) if part is not None else 1.0
        return expr[:i] + [repr(float(f"{value:.4g}"))] + expr[end:]
    if kind == "wrap":
        i = int(rng.integers(len(expr)))
        unary = [o for o, n in arity.items() if n == 1]
        return expr[:i] + [str(rng.choice(unary))] + expr[i:]
    raise ValueError(kind)


def near_correct(problem: Any, level: str, rng: np.random.Generator, engine: Any,
                 check: Callable[[list], bool] = lambda e: True, tries: int = 30) -> dict | None:
    """A near-correct history of ``problem`` at ``level``, or None if none could be made.

    Returns ``dict(expression, predictions, level, actual, edits)``; ``actual`` differs from
    ``level`` only when a problem has no numeric literal to change, or every scaling attempt
    failed ("constants" then falls back to one structural edit).
    """
    truth = list(problem.expression)
    x, y = problem.x_support, problem.y_support
    if level not in LEVELS:
        raise ValueError(level)
    for attempt in range(tries):
        actual, edits, expr = level, [], truth
        if level == "constants" and attempt < tries // 2:
            scaled = scale_constants(truth, rng, engine.operator_arity)
            if scaled is None:
                actual = "structure_1"
            else:
                expr, edits = scaled, ["constants"]
        elif level == "constants":
            actual = "structure_1"
        if actual.startswith("structure"):
            n = 1 if actual == "structure_1" else int(rng.integers(2, 4))
            expr = truth
            for _ in range(n):
                kind = str(rng.choice(EDITS))
                edited = structural_edit(expr, kind, rng, engine, list(problem.variables), x)
                if edited is None:
                    break
                expr = edited
                edits.append(kind)
            if len(edits) != n:
                continue
        pred = evaluate(engine, expr, x)
        if pred is None or not check(expr):
            continue
        if level != "exact" and (expr == truth or fvu(y, pred) <= WRONG_FVU):
            continue
        return dict(expression=list(expr), predictions=pred, level=level, actual=actual,
                    edits=edits)
    return None
