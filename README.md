# Anomaly Detector

This project aims to create a Production-Grade HTM-Based Anomaly Detection System.

## Dataset Structure

The dataset consists of JSON Lines with the following format:

```json
{"timestamp": 1717000123, "entity_id": "machine_abc91", "signal_type": "network_event", "metric": "sendmsg", "value": 3}
```

## DataGenerator Package

The DataGenerator package is structured as follows:

- **NormalBehaviorSimulator**: Simulates normal behavior.
  - EntityProfiles: Defines profiles for entities like machines and servers.
  - MetricGenerators: Generates metrics for network, system, and application signals.
  - TemporalPatterns: Handles circadian, weekly, and seasonal patterns.
  - CorrelationEngine: Manages cross-metric dependencies.

- **AnomalyInjector**: Injects various types of anomalies.
  - PointAnomalies: Single data point anomalies (spikes or drops).
  - ContextualAnomalies: Anomalies that are abnormal given the local context.
  - CollectiveAnomalies: Anomalies affecting a group of points (clusters, variance changes, trend reversals, distortions).
  - TimingAnomalies: Anomalies related to timing (delays, lags, missed beats).

- **RealismEngine**: Adds realism to the data.
  - NoiseGenerator: Adds Gaussian or Poisson noise.
  - MissingDataSimulator: Simulates missing data points.
  - DriftSimulator: Simulates concept drift.
  - SeasonalityModeler: Models seasonal variations.

- **OutputFormatter**: Formats and outputs the data.
  - JSONLWriter: Writes data in JSON Lines format.
  - MetadataGenerator: Generates ground truth labels.
  - StatisticsReporter: Reports statistics on the generated data.

## Anomaly Types and Generation

The DataGenerator automatically injects various types of anomalies into the synthetic data to create comprehensive test datasets for anomaly detection systems. Anomalies are injected randomly based on configurable probabilities.

### Point Anomalies
- **Spike**: Sudden increase in value (multiplied by a factor, default 3x).
- **Drop**: Sudden decrease in value (reduced by a percentage, 20-50%).

### Contextual Anomalies
- **Contextual Anomaly**: A point that deviates significantly from its local moving average but remains within global bounds.

### Collective Anomalies
- **Spike Cluster**: A cluster of spikes across consecutive points.
- **Variance Explosion**: Drastic increase in data variance/noise.
- **Trend Reversal**: Reversal of an existing trend (e.g., increasing becomes decreasing).
- **Square Wave Distortion**: Replacement of smooth curves with square-wave patterns.

### Timing Anomalies
- **Timestamp Shift**: Random delay added to a timestamp.
- **Lag**: Shifting of values forward by several steps.
- **Missed Beat**: Setting a value to zero or holding the previous value, simulating missed periodic events.

### Configuration
Anomalies are injected with the following default probabilities:
- Point: 50%
- Collective: 20%
- Contextual: 15%
- Timing: 15%

Each data segment receives approximately 5% anomalous points on average. The specific anomaly subtype is chosen randomly within each type.

## Anomaly Detection (Current Implementation)

The current detector is an in-memory, per-stream anomaly service:
- One model per `(entity_id, signal_type, metric)` stream.
- Input normalization and deduplication by `(timestamp, entity_id, signal_type, metric)`.
- API-first detection flow (`/train`, `/detect`, `/detect_batch`, `/status`).
- Metadata is treated as **reference/weak labels** for UI comparison (not strict ground truth).

`htm.core` is used when available; the runtime falls back to deterministic statistical scoring if HTM initialization/import is unavailable.

## Setup

### 1) Use Python 3.11 (recommended)

`htm.core` is most reliable with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) (Optional) Generate sample dataset

```bash
python main.py --points 90000 --anomaly-percent 10
```

For a taxi-demand-like long-horizon shape (rise then decline):

```bash
python main.py --points 90000 --anomaly-percent 5 --series-style taxi_like --smoothness 0.75
```

Arguments:
- `--points`: exact number of points to generate across all streams.
- `--anomaly-percent`: anomaly injection rate, must be between `5` and `15`.
- `--smoothness`: continuity factor for smoother time series (`0.0` to `<1.0`, default `0.9`).
- `--series-style`: baseline style (`default` or `taxi_like`).
- `--output-format`: `jsonl` (default) or `csv`.
- `--duration-hours`: used only when `--points` is not provided.

This creates/updates `dataset.jsonl` and `metadata.json`.

## How To Run

### Terminal 1: Start API server

```bash
python api_server.py
```

Server runs on `http://localhost:8001`.

Available endpoints:
- `POST /train` with `{ "data": [...] }`
- `POST /detect` with `{ "point": {...} }`
- `POST /detect_batch` with `{ "points": [...], "return_scores": true }`
- `GET /status`

### Terminal 2: Start frontend static server

```bash
python generate_html.py
python -m http.server 8000
```

Open `http://localhost:8000/index.html`.

### UI workflow

1. Upload `dataset.jsonl` (or `.csv`).
2. Optionally upload `metadata.json` as **reference anomalies**.
3. Click **Train Model**.
4. Click **Detect Anomalies** (batch detection via `/detect_batch`).
5. Use chart layer toggles (Detected / Reference / Overlap).
6. Review TP/FP/FN, precision/recall/F1, and split anomaly lists.

## Tests

```bash
python -m unittest discover -s tests -q
```

## Dependencies

- numpy
- pandas
- scipy
- htm.core
