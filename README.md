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
  - PointAnomalies: Single spikes or drops.
  - ContextualAnomalies: Anomalies unexpected in context.
  - CollectiveAnomalies: Unusual patterns across data.
  - TimingAnomalies: Issues like race conditions or delays.

- **RealismEngine**: Adds realism to the data.
  - NoiseGenerator: Adds Gaussian or Poisson noise.
  - MissingDataSimulator: Simulates missing data points.
  - DriftSimulator: Simulates concept drift.
  - SeasonalityModeler: Models seasonal variations.

- **OutputFormatter**: Formats and outputs the data.
  - JSONLWriter: Writes data in JSON Lines format.
  - MetadataGenerator: Generates ground truth labels.
  - StatisticsReporter: Reports statistics on the generated data.

## Anomaly Detection Algorithm

The system includes a core anomaly detection algorithm based on Hierarchical Temporal Memory (HTM) and Liquid State Machines (LSM):

### Architecture

```
Input Event
    ↓
Feature Extraction Layer
    ↓
Preprocessing (Normalization/Scaling)
    ↓
    ├─ LSM Pathway (Reservoir: 500 neurons)
    │   - Liquid State computation
    │   - Readout for temporal anomalies
    └─ HTM Pathway (SDR Encoding + Spatial Pooler + Temporal Memory)
        - RandomDistributedScalarEncoder for SDR creation
        - Sequence anomaly detection
    ↓
Fusion Layer (Weighted average, confidence, voting)
    ↓
Output Layer (Anomaly flag, score, explanation)
```

### Components

- **Feature Extraction**: Value encoding, temporal features, entity context
- **LSM Pathway**: Reservoir computing for timing-based anomalies
- **HTM Pathway**: Uses NuPIC library with SDR (Sparse Distributed Representations) encoding via RandomDistributedScalarEncoder, Spatial Pooler, Temporal Memory, and Anomaly detector for pattern-based anomalies
- **Fusion Layer**: Combines scores from both pathways
- **Output Layer**: Provides final anomaly decision with explanation

The algorithm is implemented in `anomaly_detector.py` with simplified versions of HTM and LSM due to dependency constraints. A client-side version is integrated into the web UI that implements actual HTM-inspired training (learning patterns, transitions, and sequence predictions) for interactive anomaly detection.

## Setup

1. Install dependencies: `pip install -r requirements.txt` (Note: Dependencies may need to be installed in a proper environment with internet access.)
2. The package is written in Python.

## Usage

- Run `python3 main.py` to generate a sample dataset with anomalies.
- Run `python3 ui.py` to view the anomalous data points in text format.
- Run `python3 generate_html.py` to generate the beautiful HTML frontend with upload functionality.
- Run `python3 api_server.py` to start the anomaly detection API server on port 8001.
- Run `python3 -m http.server 8000` and open `http://localhost:8000/index.html` in a web browser to:
  - Upload your own `.jsonl` dataset file.
  - Upload the corresponding `metadata.json` file for anomalies (optional).
  - View the interactive time series chart with anomalies highlighted in red.
  - See statistics and a list of detected anomalies.
  - Use granularity controls (Minute/Hour wise) and time range slider.
  - Click "Train Model" to train HTM/LSM models via API.
  - Click "Detect Anomalies" to run anomaly detection on the uploaded data.
  - Clear metadata with the clear button.
- Run `python3 anomaly_detector.py` to test the HTM-LSM based anomaly detection algorithm.

## Dependencies

- numpy
- pandas
- scipy

Note: For full HTM implementation, install NuPIC:
```
pip install nupic
```

For LSM, ReservoirPy:
```
pip install reservoirpy
```

The code is designed to use these libraries when available, falling back to simplified implementations otherwise.
