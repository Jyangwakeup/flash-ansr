"""History absence is a feature-space mask, not a numeric value."""

import pytest
import torch
from flash_ansr import FlashANSRModel, get_path
from flash_ansr.utils.config_io import load_config


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(19)
    cfg = load_config(get_path("configs", "v25.0-T8.1-3M", "model.yaml"))
    cfg["history_slots"] = 1
    return FlashANSRModel.from_config(cfg).eval()


def test_absent_history_ignores_nonfinite_and_arbitrary_values(model):
    data = torch.randn(2, 6, 19, dtype=torch.float64)
    absent = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        expected = model._create_memory(data, history_mask=absent)
        data[0, :, -1] = float("nan")
        data[1, :, -1] = 1e300
        actual = model._create_memory(data, history_mask=absent)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_zero_prediction_is_different_from_absence(model):
    data = torch.zeros(1, 6, 19, dtype=torch.float64)
    with torch.no_grad():
        absent = model._create_memory(data, history_mask=torch.tensor([False]))
        valid_zero = model._create_memory(data, history_mask=torch.tensor([True]))
    assert not torch.allclose(absent, valid_zero)


def test_expanded_data_requires_explicit_validity(model):
    with pytest.raises(ValueError, match="history_mask"):
        model._create_memory(torch.zeros(1, 6, 19, dtype=torch.float64))
    with torch.no_grad():
        legacy = model._create_memory(torch.zeros(1, 6, 18, dtype=torch.float64))
        explicit = model._create_memory(
            torch.zeros(1, 6, 19, dtype=torch.float64),
            history_mask=torch.tensor([False]),
        )
    torch.testing.assert_close(legacy, explicit, rtol=0, atol=0)
