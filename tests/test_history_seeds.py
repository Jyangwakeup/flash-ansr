"""Per-split data seeds make history-experiment problems reproducible and prefix-stable."""

import sys
from pathlib import Path

import numpy as np
import pytest

HB = Path(__file__).resolve().parents[1] / "experiments" / "history_benefits"
sys.path.insert(0, str(HB / "core"))
sys.path.insert(0, str(HB / "r3_near_correct_history"))  # r3_samples, corrupt

from common import eval_seed, fit_seed, problem_ids, split_seeds, train_seed  # noqa: E402
from pool import draw_problems, split_of  # noqa: E402

CFG = {
    "recipe": "configs/v25.0-T8.1-3M/dataset_train.yaml",
    "seed": 20261120,
    "data_seed_train": 20261121,
    "data_seed_validation": 20261122,
    "fit_seed": 20261124,
    "train_problems": 1000,
}


def same(a, b):
    return (
        list(a.expression) == list(b.expression)
        and np.array_equal(a.x_support, b.x_support, equal_nan=True)
        and np.array_equal(a.y_support_noisy, b.y_support_noisy, equal_nan=True)
    )


def test_same_seed_draws_identical_problems():
    first = draw_problems(CFG, "train", 5)
    second = draw_problems(CFG, "train", 5)
    assert all(same(a, b) for a, b in zip(first, second))


def test_longer_draw_starts_with_the_shorter_one():
    short = draw_problems(CFG, "validation", 3)
    long = draw_problems(CFG, "validation", 12)
    assert all(same(a, b) for a, b in zip(short, long[:3]))


def test_train_and_validation_streams_differ():
    train = {tuple(p.expression) for p in draw_problems(CFG, "train", 30)}
    validation = {tuple(p.expression) for p in draw_problems(CFG, "validation", 30)}
    assert not train & validation


def test_ids_and_seeds():
    assert split_of(CFG, 999) == ("train", 999)
    assert split_of(CFG, 100_000) == ("validation", 0)
    assert split_of(dict(CFG, train_problems=10_000), 100_005) == ("validation", 5)
    with pytest.raises(ValueError):
        split_of(CFG, 1000)
    assert problem_ids(CFG, "validation", 2) == [100_000, 100_001]
    assert problem_ids({"seed": 1, "train_problems": 160}, "validation", 1) == [160]
    assert split_of({"seed": 1, "train_problems": 160}, 160) == ("validation", 0)
    assert fit_seed(CFG, 7) == 20261131
    assert fit_seed({"seed": 5}, 7) == 12
    assert train_seed(dict(CFG, train_seed=20261123)) == 20261123
    assert train_seed({"seed": 5}) == 5
    assert eval_seed(dict(CFG, eval_seed=20261125), 3) == 20261128
    assert eval_seed({"seed": 5}, 3) == 8
    assert split_seeds(CFG) and not split_seeds({"seed": 1})
    with pytest.raises(ValueError):
        split_seeds({"data_seed_train": 1})


# --- round 3: selection / fresh splits and near-correct histories --------------------------

from common import SELECTION_ID_OFFSET  # noqa: E402

MIX = dict(CFG, data_seed_fresh=20261126, data_seed_selection=20261127, train_problems=5000,
           validation_sampling={"n_support": 512, "n_validation": 512, "noise": 0.0})


def test_selection_split_ids_and_stream():
    assert problem_ids(MIX, "selection", 2) == [SELECTION_ID_OFFSET, SELECTION_ID_OFFSET + 1]
    assert split_of(MIX, SELECTION_ID_OFFSET + 3) == ("selection", 3)
    assert split_of(MIX, 4999) == ("train", 4999)
    selection = {tuple(p.expression) for p in draw_problems(MIX, "selection", 30)}
    validation = {tuple(p.expression) for p in draw_problems(MIX, "validation", 30)}
    train = {tuple(p.expression) for p in draw_problems(MIX, "train", 30)}
    assert not selection & validation and not selection & train


def test_extra_seed_keys_do_not_change_old_splits():
    old = draw_problems(CFG, "train", 3)
    new = draw_problems(MIX, "train", 3)
    assert all(same(a, b) for a, b in zip(old, new))


def test_kind_counts():
    from r3_samples import kind_counts

    assert kind_counts({"none": 0.3, "near": 0.4, "real": 0.3}, 128) == {"none": 39, "near": 51, "real": 38}
    assert kind_counts({"none": 1.0}, 16) == {"none": 16, "near": 0, "real": 0}
    with pytest.raises(ValueError):
        kind_counts({"none": 0.5}, 8)


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("acj-5-4-llm")


def test_near_correct_levels(engine):
    from types import SimpleNamespace
    from corrupt import LEVELS, near_correct, subtree_end

    x = np.random.default_rng(0).uniform(0.5, 2.0, size=(64, 3))
    truth = ["+", "*", "1.7", "x1", "sin", "*", "x2", "pow", "x3", "2"]
    problem = SimpleNamespace(expression=truth, variables=["x1", "x2", "x3"], x_support=x,
                              y_support=1.7 * x[:, 0] + np.sin(x[:, 1] * x[:, 2] ** 2))
    assert subtree_end(truth, 1, engine.operator_arity) == 4
    for level in LEVELS:
        a = near_correct(problem, level, np.random.default_rng(1), engine)
        b = near_correct(problem, level, np.random.default_rng(1), engine)
        assert a is not None and a["expression"] == b["expression"]  # reproducible
        assert np.isfinite(a["predictions"]).all()
        if level == "exact":
            assert a["expression"] == truth and a["edits"] == []
        else:
            assert a["expression"] != truth
        if level == "constants":
            changed = [i for i, (s, t) in enumerate(zip(truth, a["expression"])) if s != t]
            assert changed == [2]  # the exponent 2 of pow stays
        if level == "structure_1":
            assert len(a["edits"]) == 1
        if level == "structure_2_3":
            assert len(a["edits"]) in (2, 3)


def test_constants_fall_back_without_literals(engine):
    from types import SimpleNamespace
    from corrupt import near_correct

    x = np.random.default_rng(0).uniform(0.5, 2.0, size=(32, 2))
    problem = SimpleNamespace(expression=["+", "x1", "x2"], variables=["x1", "x2"], x_support=x,
                              y_support=x[:, 0] + x[:, 1])
    r = near_correct(problem, "constants", np.random.default_rng(0), engine)
    assert r["actual"] == "structure_1" and len(r["edits"]) == 1
