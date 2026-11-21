"""Per-split data seeds make history-experiment problems reproducible and prefix-stable."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "history_benefits"))

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
