# MLOps Batch Job - Rolling Mean Signal Pipeline

A minimal, reproducible MLOps-style batch job that loads OHLCV data, computes a rolling mean on `close`, generates binary trading signals, and writes structured metrics + detailed logs.

---

## Project Structure

```
mlops-task/
├── run.py            # Main pipeline script
├── config.yaml       # Job configuration (seed, window, version)
├── data.csv          # 10 000-row OHLCV input dataset
├── requirements.txt  # Pinned Python dependencies
├── Dockerfile        # Container definition
├── .dockerignore     # Docker build context exclusions
├── tests/            # Automated test suite (pytest)
│   ├── test_signal.py
│   └── test_validation.py
├── metrics.json      # Sample output from a successful run
├── run.log           # Sample log from a successful run
└── README.md         # Documentation
```

---

## Local Run

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

The final metrics JSON is printed to **stdout** and written to `metrics.json`.  
Detailed logs are written to `run.log` (and `stderr`).

### Run the tests

```bash
python -m pytest -v
```

---

## Docker Build & Run

### Build

```bash
docker build -t mlops-task .
```

### Run (Default container run)

```bash
docker run --rm mlops-task
```

Exit code `0` = success, non-zero = failure.

### Run with Volume Mount (Persisting Output Files)

By default, files generated inside the container are ephemeral. To persist `metrics.json` and `run.log` to the host filesystem, mount a host directory (e.g. `./output`):

**Linux / macOS / PowerShell:**
```bash
mkdir -p output
docker run --rm \
  -v "${PWD}/output:/app/output" \
  mlops-task \
  python run.py \
    --input data.csv \
    --config config.yaml \
    --output /app/output/metrics.json \
    --log-file /app/output/run.log
```

**Windows CMD:**
```cmd
mkdir output
docker run --rm ^
  -v "%cd%/output:/app/output" ^
  mlops-task ^
  python run.py ^
    --input data.csv ^
    --config config.yaml ^
    --output /app/output/metrics.json ^
    --log-file /app/output/run.log
```

---

## Configuration (`config.yaml`)

| Key       | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility   |
| `window`  | int     | Rolling mean window size (rows)         |
| `version` | string  | Pipeline version tag in output metrics  |

---

## Signal Logic

1. Compute a rolling mean on `close` with the configured `window`.
2. The first `window - 1` rows produce `NaN` and are **excluded** from signal computation and metrics.
3. For each valid row: `signal = 1` if `close > rolling_mean`, else `signal = 0`.

> **Note on Indicator Design:**  
> The rolling mean is computed as a **contemporaneous indicator** (evaluating $\text{Close}_t > \text{SMA}_k(\text{Close})_t$, where $\text{Close}_t$ is included in the rolling window). It is designed to categorize the current state against its immediate historical baseline, rather than acting as a lagged predictive signal (which would compare against $\text{SMA}_{t-1}$ to forecast forward returns).

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 23,
  "seed": 42,
  "status": "success"
}
```

`rows_processed` = 9 996 because the first 4 rows (window - 1 = 4) are excluded from signal computation.

### Error output shape

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input file not found: data.csv",
  "latency_ms": 3
}
```

---

## Validation & Error Handling

The pipeline catches and reports errors for:
- Missing input file or config file
- Invalid / non-parseable CSV
- Empty CSV
- Missing `close` column
- Missing or wrong-typed config keys (`seed`, `window`, `version`)

In all error cases `metrics.json` is written with `"status": "error"` and the script exits with code `1`.

---

## Reproducibility

Results are fully deterministic given the same `data.csv` and `config.yaml`:  
`numpy.random.seed(seed)` is set before any processing, and rolling-mean computation is purely deterministic.
