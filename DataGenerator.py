from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricConfig:
    baseline: float
    noise_std: float
    min_value: float
    max_value: float
    anomaly_spike_multiplier: float
    anomaly_drop_multiplier: float
    weekly_factor: float = 1.0


DEFAULT_METRIC_CONFIG = MetricConfig(
    baseline=100.0,
    noise_std=10.0,
    min_value=0.0,
    max_value=10_000.0,
    anomaly_spike_multiplier=5.0,
    anomaly_drop_multiplier=0.1,
    weekly_factor=1.0,
)


METRIC_CONFIGS: dict[str, MetricConfig] = {
    "sendmsg": MetricConfig(
        baseline=850,
        noise_std=35,
        min_value=100,
        max_value=2500,
        anomaly_spike_multiplier=5.0,
        anomaly_drop_multiplier=0.1,
        weekly_factor=1.3,
    ),
    "recvmsg": MetricConfig(
        baseline=820,
        noise_std=32,
        min_value=90,
        max_value=2400,
        anomaly_spike_multiplier=5.0,
        anomaly_drop_multiplier=0.1,
        weekly_factor=1.25,
    ),
    "http_requests": MetricConfig(
        baseline=1200,
        noise_std=80,
        min_value=150,
        max_value=3500,
        anomaly_spike_multiplier=6.0,
        anomaly_drop_multiplier=0.05,
        weekly_factor=1.4,
    ),
    "cpu_usage": MetricConfig(
        baseline=45,
        noise_std=5,
        min_value=0,
        max_value=100,
        anomaly_spike_multiplier=2.5,
        anomaly_drop_multiplier=0.3,
        weekly_factor=1.1,
    ),
    "memory_usage": MetricConfig(
        baseline=60,
        noise_std=4,
        min_value=0,
        max_value=100,
        anomaly_spike_multiplier=1.8,
        anomaly_drop_multiplier=0.5,
        weekly_factor=1.05,
    ),
    "disk_io": MetricConfig(
        baseline=300,
        noise_std=40,
        min_value=0,
        max_value=5000,
        anomaly_spike_multiplier=4.5,
        anomaly_drop_multiplier=0.2,
        weekly_factor=1.1,
    ),
    "query_count": MetricConfig(
        baseline=800,
        noise_std=90,
        min_value=0,
        max_value=10_000,
        anomaly_spike_multiplier=4.0,
        anomaly_drop_multiplier=0.3,
        weekly_factor=1.2,
    ),
    "api_latency": MetricConfig(
        baseline=120,
        noise_std=15,
        min_value=1,
        max_value=5_000,
        anomaly_spike_multiplier=8.0,
        anomaly_drop_multiplier=0.4,
        weekly_factor=1.05,
    ),
    "temperature": MetricConfig(
        baseline=40,
        noise_std=1.2,
        min_value=-20,
        max_value=120,
        anomaly_spike_multiplier=1.6,
        anomaly_drop_multiplier=0.6,
        weekly_factor=1.0,
    ),
}


def _daily_pattern(dt: datetime) -> float:
    hour = dt.hour + dt.minute / 60.0
    if 9 <= hour <= 17:
        if hour < 13:
            progress = (hour - 9) / 4
            return 0.6 + 0.4 * progress
        progress = (hour - 13) / 4
        return 1.0 - 0.4 * progress
    if 0 <= hour <= 6:
        return 0.3 + 0.1 * math.sin(hour * math.pi / 3)
    return 0.5 + 0.2 * math.sin((hour - 18) * math.pi / 6)


def _weekly_pattern(dt: datetime, weekly_factor: float) -> float:
    weekday = dt.weekday()  # 0=Mon, 6=Sun
    if weekday < 5:
        if weekday in (1, 2):
            return 1.0 * weekly_factor
        if weekday == 0:
            return 0.8 * weekly_factor
        if weekday == 3:
            return 0.9 * weekly_factor
        return 0.7 * weekly_factor
    return (0.4 if weekday == 5 else 0.3) * weekly_factor


def _taxi_like_factor(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return 0.55 + 0.65 * math.sin(math.pi * progress)


class DataGenerator:
    """
    Minimal dataset generator used by `main.py`.

    Writes:
      - `dataset.jsonl` or `dataset.csv` (based on `output_format`)
      - `metadata.json` (array of anomaly records for UI/reference comparison)
    """

    def __init__(
        self,
        *,
        output_format: str = "jsonl",
        dataset_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        step_seconds: int = 60,
        seed: int | None = 42,
    ) -> None:
        if output_format not in ("jsonl", "csv"):
            raise ValueError("output_format must be 'jsonl' or 'csv'")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")

        self.output_format = output_format
        self.dataset_path = Path(dataset_path) if dataset_path else Path(
            "dataset.jsonl" if output_format == "jsonl" else "dataset.csv"
        )
        self.metadata_path = Path(metadata_path) if metadata_path else Path("metadata.json")
        self.step_seconds = step_seconds
        self.seed = seed

        self.continuity_alpha: float = 0.9
        self.series_style: str = "default"

    def generate_dataset(
        self,
        entities: list[dict[str, Any]],
        start_time: int,
        *,
        end_time: int | None = None,
        target_points: int | None = None,
        anomaly_percent: float = 10.0,
    ) -> list[dict[str, Any]]:
        if (end_time is None) == (target_points is None):
            raise ValueError("Provide exactly one of end_time or target_points")
        if anomaly_percent < 0 or anomaly_percent > 100:
            raise ValueError("anomaly_percent must be in [0, 100]")
        if target_points is not None and target_points <= 0:
            raise ValueError("target_points must be > 0")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be > start_time")

        if self.seed is not None:
            random.seed(self.seed)

        streams = self._expand_streams(entities)
        if not streams:
            raise ValueError("No streams found in entities configuration")

        counts_by_stream = self._counts_by_stream(
            len(streams), start_time, end_time=end_time, target_points=target_points
        )

        all_points: list[dict[str, Any]] = []
        anomaly_map: dict[tuple[int, str, str, str], set[str]] = {}

        for (entity_id, entity_type, signal_type, metric), count in zip(streams, counts_by_stream, strict=True):
            metric_cfg = METRIC_CONFIGS.get(metric, DEFAULT_METRIC_CONFIG)
            values, labels = self._generate_stream_values(
                metric_cfg,
                count=count,
                start_time=start_time,
                anomaly_percent=anomaly_percent,
            )

            for idx in range(count):
                ts = int(start_time + idx * self.step_seconds)
                value = float(values[idx])

                point = {
                    "timestamp": ts,
                    "entity_id": entity_id,
                    "signal_type": signal_type,
                    "metric": metric,
                    "value": value,
                }
                all_points.append(point)

                if labels[idx]:
                    key = (ts, entity_id, signal_type, metric)
                    existing = anomaly_map.setdefault(key, set())
                    existing.update(labels[idx])

        # Stable ordering helps debugging and downstream tools.
        all_points.sort(key=lambda p: (p["entity_id"], p["signal_type"], p["metric"], p["timestamp"]))

        reference_anomalies: list[dict[str, Any]] = []
        for (ts, entity_id, signal_type, metric), types in anomaly_map.items():
            entity_type = self._entity_type_for_entity(entities, entity_id)
            reference_anomalies.append(
                {
                    "timestamp": ts,
                    "entity_id": entity_id,
                    "signal_type": signal_type,
                    "metric": metric,
                    "entity_type": entity_type,
                    "anomaly_types": sorted(types),
                }
            )
        reference_anomalies.sort(key=lambda r: (r["entity_id"], r["signal_type"], r["metric"], r["timestamp"]))

        self._write_dataset(all_points)
        self._write_metadata(reference_anomalies)
        return all_points

    def _expand_streams(self, entities: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
        streams: list[tuple[str, str, str, str]] = []
        for ent in entities:
            entity_id = str(ent.get("id", "unknown"))
            entity_type = str(ent.get("type", "Unknown"))
            signal_types = ent.get("signal_types") or []
            for sig in signal_types:
                signal_type = str(sig.get("type", "default"))
                for metric in sig.get("metrics") or []:
                    streams.append((entity_id, entity_type, signal_type, str(metric)))
        return streams

    def _entity_type_for_entity(self, entities: list[dict[str, Any]], entity_id: str) -> str:
        for ent in entities:
            if str(ent.get("id")) == entity_id:
                return str(ent.get("type", "Unknown"))
        return "Unknown"

    def _counts_by_stream(
        self,
        stream_count: int,
        start_time: int,
        *,
        end_time: int | None,
        target_points: int | None,
    ) -> list[int]:
        if target_points is not None:
            base = target_points // stream_count
            remainder = target_points % stream_count
            return [base + (1 if i < remainder else 0) for i in range(stream_count)]

        assert end_time is not None
        points_per_stream = max(1, (end_time - start_time) // self.step_seconds)
        return [points_per_stream for _ in range(stream_count)]

    def _generate_stream_values(
        self,
        cfg: MetricConfig,
        *,
        count: int,
        start_time: int,
        anomaly_percent: float,
    ) -> tuple[list[float], list[set[str]]]:
        if count <= 0:
            return [], []

        alpha = min(0.999, max(0.0, float(self.continuity_alpha)))
        labels: list[set[str]] = [set() for _ in range(count)]

        values: list[float] = []
        prev = cfg.baseline
        for i in range(count):
            ts = int(start_time + i * self.step_seconds)
            dt = datetime.fromtimestamp(ts)
            daily = _daily_pattern(dt)
            weekly = _weekly_pattern(dt, cfg.weekly_factor)
            if self.series_style == "taxi_like":
                progress = i / (count - 1) if count > 1 else 0.0
                shape = _taxi_like_factor(progress)
            else:
                shape = 1.0

            target = cfg.baseline * daily * weekly * shape
            smoothed = alpha * prev + (1.0 - alpha) * target
            noise = random.gauss(0.0, cfg.noise_std)
            val = smoothed + noise
            val = max(cfg.min_value, min(cfg.max_value, val))

            values.append(float(val))
            prev = smoothed

        self._inject_anomalies(values, labels, cfg, anomaly_percent=anomaly_percent)
        return values, labels

    def _inject_anomalies(
        self,
        values: list[float],
        labels: list[set[str]],
        cfg: MetricConfig,
        *,
        anomaly_percent: float,
    ) -> None:
        n = len(values)
        if n == 0 or anomaly_percent <= 0:
            return

        target = int(round(n * (anomaly_percent / 100.0)))
        target = max(1, min(target, n))

        used = 0
        safety = 0
        while used < target and safety < target * 20:
            safety += 1
            r = random.random()
            if r < 0.50:
                # Spike (single point)
                idx = random.randrange(0, n)
                values[idx] = min(cfg.max_value, values[idx] + cfg.baseline * cfg.anomaly_spike_multiplier)
                labels[idx].add("spike")
                used += 1
            elif r < 0.65:
                # Drop (short segment)
                seg_len = random.randint(10, 60)
                start = random.randrange(0, n)
                end = min(n, start + seg_len)
                for i in range(start, end):
                    values[i] = max(cfg.min_value, values[i] * cfg.anomaly_drop_multiplier)
                    labels[i].add("drop")
                used += (end - start)
            elif r < 0.85:
                # Collective (medium segment)
                seg_len = random.randint(30, 150)
                start = random.randrange(0, n)
                end = min(n, start + seg_len)
                for i in range(start, end):
                    bump = cfg.baseline * 2.0 * random.uniform(0.6, 1.4)
                    values[i] = min(cfg.max_value, values[i] + bump)
                    labels[i].add("collective")
                used += (end - start)
            else:
                # Drift (long segment)
                seg_len = random.randint(80, 300)
                start = random.randrange(0, n)
                end = min(n, start + seg_len)
                for j, i in enumerate(range(start, end)):
                    drift = cfg.baseline * 0.6 * (j / max(1, (end - start - 1)))
                    values[i] = min(cfg.max_value, values[i] + drift)
                    labels[i].add("drift")
                used += (end - start)

        # Clamp after injection
        for i, v in enumerate(values):
            values[i] = float(max(cfg.min_value, min(cfg.max_value, v)))

    def _write_dataset(self, points: Iterable[dict[str, Any]]) -> None:
        if self.output_format == "jsonl":
            with self.dataset_path.open("w", encoding="utf-8") as f:
                for point in points:
                    f.write(json.dumps(point) + "\n")
            return

        # csv
        with self.dataset_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "entity_id", "signal_type", "metric", "value"],
            )
            writer.writeheader()
            for point in points:
                writer.writerow(
                    {
                        "timestamp": int(point["timestamp"]),
                        "entity_id": str(point["entity_id"]),
                        "signal_type": str(point["signal_type"]),
                        "metric": str(point["metric"]),
                        "value": float(point["value"]),
                    }
                )

    def _write_metadata(self, records: list[dict[str, Any]]) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

