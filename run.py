#!/usr/bin/env python3
"""
MLOps Batch Job: Rolling Mean Signal Pipeline
Loads OHLCV data, computes rolling mean on close, generates binary signals,
writes structured metrics JSON and detailed logs.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (stderr so stdout stays clean for JSON)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config YAML must be a mapping (dict).")

    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(config['seed'])}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str):
        raise ValueError(f"Config 'version' must be a string, got: {type(config['version'])}")

    logger.info(
        f"Config loaded and validated - seed={config['seed']}, "
        f"window={config['window']}, version={config['version']}"
    )
    return config


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise ValueError("Column 'close' must contain numeric values.")

    if df["close"].isnull().any():
        null_count = df["close"].isnull().sum()
        logger.warning(f"Column 'close' has {null_count} null value(s); they will produce NaN signals.")

    logger.info(f"Dataset loaded - {len(df)} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """
    Compute rolling mean on 'close' with the given window.
    The first (window-1) rows will be NaN; those rows are excluded from signal computation.
    """
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logger.info(
        f"Rolling mean computed - window={window}, "
        f"rows with NaN (excluded from signal): {nan_count}"
    )
    return rolling_mean


def compute_signal(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN there).
    """
    signal = (close > rolling_mean).astype(float)
    signal[rolling_mean.isna()] = np.nan
    valid_signals = signal.dropna()
    logger.info(
        f"Signal generated - {len(valid_signals)} valid rows, "
        f"signal_rate={valid_signals.mean():.6f}"
    )
    return signal


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Metrics written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps Rolling Mean Signal Pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path for log file")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)

    start_time = time.time()
    logger.info("=== Job started ===")

    version = "unknown"  # fallback for error metrics before config is parsed

    try:
        # 1. Load & validate config
        config = load_config(args.config, logger)
        version = config["version"]
        seed = config["seed"]
        window = config["window"]

        # 2. Set seed for reproducibility
        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # 3. Load & validate dataset
        df = load_dataset(args.input, logger)

        # 4. Compute rolling mean
        logger.info("Computing rolling mean ...")
        rolling_mean = compute_rolling_mean(df, window, logger)

        # 5. Compute signal
        logger.info("Computing binary signal ...")
        signal = compute_signal(df["close"], rolling_mean, logger)

        # 6. Metrics
        valid_signal = signal.dropna()
        rows_processed = len(valid_signal)
        signal_rate = float(valid_signal.mean())
        latency_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 6),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        logger.info(
            f"Metrics summary - rows_processed={rows_processed}, "
            f"signal_rate={signal_rate:.6f}, latency_ms={latency_ms}"
        )

        write_metrics(args.output, metrics, logger)
        logger.info("=== Job completed successfully ===")

        # Print final metrics to stdout (required by Docker spec)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Job failed: {exc}", exc_info=True)

        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
            "latency_ms": latency_ms,
        }
        write_metrics(args.output, error_metrics, logger)
        logger.info("=== Job ended with error ===")

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
