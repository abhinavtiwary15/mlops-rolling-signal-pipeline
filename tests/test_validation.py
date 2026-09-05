import json
import logging
import subprocess
import sys
from pathlib import Path
import pytest
import yaml

from run import load_config, load_dataset


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test_validation_logger")
    logger.setLevel(logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Unit tests on validation functions
# ---------------------------------------------------------------------------

def test_missing_config_file(dummy_logger, tmp_path):
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(str(non_existent), dummy_logger)


def test_missing_dataset_file(dummy_logger, tmp_path):
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_dataset(str(non_existent), dummy_logger)


def test_missing_close_column(dummy_logger, tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("timestamp,open,high,low\n2024-01-01,1,2,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required column 'close' not found"):
        load_dataset(str(csv_file), dummy_logger)


def test_non_numeric_close_column(dummy_logger, tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("close\nabc\ndef\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain numeric values"):
        load_dataset(str(csv_file), dummy_logger)


def test_empty_dataset_file(dummy_logger, tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse CSV|Input CSV is empty"):
        load_dataset(str(csv_file), dummy_logger)


@pytest.mark.parametrize(
    "config_content,match_pattern",
    [
        ("seed: 'not_an_int'\nwindow: 5\nversion: 'v1'", "Config 'seed' must be an integer"),
        ("seed: 42\nwindow: 'five'\nversion: 'v1'", "Config 'window' must be a positive integer"),
        ("seed: 42\nwindow: -1\nversion: 'v1'", "Config 'window' must be a positive integer"),
        ("seed: 42\nwindow: 0\nversion: 'v1'", "Config 'window' must be a positive integer"),
        ("seed: 42\nwindow: 5\nversion: 123", "Config 'version' must be a string"),
        ("window: 5\nversion: 'v1'", "Config missing required keys"),
        ("seed: 42\nversion: 'v1'", "Config missing required keys"),
        ("seed: 42\nwindow: 5", "Config missing required keys"),
    ],
)
def test_bad_config_values(dummy_logger, tmp_path, config_content, match_pattern):
    cfg_file = tmp_path / "bad_config.yaml"
    cfg_file.write_text(config_content, encoding="utf-8")
    with pytest.raises(ValueError, match=match_pattern):
        load_config(str(cfg_file), dummy_logger)


# ---------------------------------------------------------------------------
# CLI end-to-end failure mode tests (asserting exit code 1)
# ---------------------------------------------------------------------------

def run_cli(args):
    """Run run.py CLI via subprocess and return CompletedProcess."""
    cmd = [sys.executable, "run.py"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_missing_input_file(tmp_path):
    output_json = tmp_path / "metrics.json"
    log_file = tmp_path / "run.log"
    res = run_cli([
        "--input", str(tmp_path / "nonexistent.csv"),
        "--config", "config.yaml",
        "--output", str(output_json),
        "--log-file", str(log_file),
    ])

    assert res.returncode == 1
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "Input file not found" in payload["error_message"]


def test_cli_missing_close_column(tmp_path):
    bad_csv = tmp_path / "no_close.csv"
    bad_csv.write_text("timestamp,volume\n2024-01-01,100\n", encoding="utf-8")
    output_json = tmp_path / "metrics.json"
    log_file = tmp_path / "run.log"

    res = run_cli([
        "--input", str(bad_csv),
        "--config", "config.yaml",
        "--output", str(output_json),
        "--log-file", str(log_file),
    ])

    assert res.returncode == 1
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "Required column 'close' not found" in payload["error_message"]


def test_cli_bad_config_type(tmp_path):
    bad_cfg = tmp_path / "bad_config.yaml"
    bad_cfg.write_text("seed: 42\nwindow: 'invalid_window'\nversion: 'v1'\n", encoding="utf-8")
    output_json = tmp_path / "metrics.json"
    log_file = tmp_path / "run.log"

    res = run_cli([
        "--input", "data.csv",
        "--config", str(bad_cfg),
        "--output", str(output_json),
        "--log-file", str(log_file),
    ])

    assert res.returncode == 1
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "Config 'window' must be a positive integer" in payload["error_message"]
