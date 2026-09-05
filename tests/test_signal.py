import logging
import numpy as np
import pandas as pd
import pytest
from run import compute_rolling_mean, compute_signal


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger


def test_identical_prices(dummy_logger):
    """When all prices are identical, close > rolling_mean is always False (signal = 0)."""
    df = pd.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]})
    window = 3
    rolling_mean = compute_rolling_mean(df, window=window, logger=dummy_logger)
    signal = compute_signal(df["close"], rolling_mean, logger=dummy_logger)

    # First window-1 rows must be NaN
    assert np.isnan(signal.iloc[0])
    assert np.isnan(signal.iloc[1])

    # Remaining rows must be 0.0
    valid_signals = signal.dropna()
    assert len(valid_signals) == 3
    assert (valid_signals == 0.0).all()
    assert valid_signals.mean() == 0.0


def test_alternating_values(dummy_logger):
    """When prices alternate around the rolling average, signals alternate deterministically."""
    # [10, 20, 10, 20, 10, 20], window=2 -> rolling means are [NaN, 15, 15, 15, 15, 15]
    df = pd.DataFrame({"close": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]})
    window = 2
    rolling_mean = compute_rolling_mean(df, window=window, logger=dummy_logger)
    signal = compute_signal(df["close"], rolling_mean, logger=dummy_logger)

    assert np.isnan(signal.iloc[0])
    expected_signals = [1.0, 0.0, 1.0, 0.0, 1.0]
    np.testing.assert_array_equal(signal.iloc[1:].to_numpy(), expected_signals)
    assert signal.dropna().mean() == pytest.approx(0.6)


def test_window_one(dummy_logger):
    """Window=1 means rolling mean equals close price; close > close is False everywhere."""
    df = pd.DataFrame({"close": [10.0, 25.0, 5.0, 30.0]})
    window = 1
    rolling_mean = compute_rolling_mean(df, window=window, logger=dummy_logger)
    signal = compute_signal(df["close"], rolling_mean, logger=dummy_logger)

    # 0 NaNs when window=1
    assert signal.isna().sum() == 0
    assert len(signal) == 4
    assert (signal == 0.0).all()
    assert signal.mean() == 0.0


def test_window_larger_than_dataset(dummy_logger):
    """When window > len(df), all rolling mean and signal values must be NaN."""
    df = pd.DataFrame({"close": [10.0, 20.0, 30.0]})
    window = 5
    rolling_mean = compute_rolling_mean(df, window=window, logger=dummy_logger)
    signal = compute_signal(df["close"], rolling_mean, logger=dummy_logger)

    assert rolling_mean.isna().all()
    assert signal.isna().all()
    assert len(signal.dropna()) == 0
