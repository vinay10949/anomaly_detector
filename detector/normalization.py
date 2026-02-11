from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .types import DataPoint

PointKey = Tuple[int, str, str, str]
StreamKey = Tuple[str, str, str]


def normalize_point(raw: dict) -> DataPoint:
    point: DataPoint = {
        "timestamp": int(raw["timestamp"]),
        "entity_id": str(raw["entity_id"]),
        "signal_type": str(raw["signal_type"]),
        "metric": str(raw["metric"]),
        "value": float(raw["value"]),
    }
    if "label" in raw:
        try:
            point["label"] = int(raw["label"])
        except Exception:
            pass
    if math.isnan(point["value"]) or math.isinf(point["value"]):
        raise ValueError("value must be finite")
    return point


def point_key(point: DataPoint) -> PointKey:
    return (
        int(point["timestamp"]),
        str(point["entity_id"]),
        str(point["signal_type"]),
        str(point["metric"]),
    )


def stream_key(point: DataPoint) -> StreamKey:
    return (
        str(point["entity_id"]),
        str(point["signal_type"]),
        str(point["metric"]),
    )


def stream_key_str(stream: StreamKey) -> str:
    return f"{stream[0]}|{stream[1]}|{stream[2]}"


def normalize_points(raw_points: Sequence[dict]) -> List[DataPoint]:
    dedup: dict[PointKey, DataPoint] = {}
    for raw in raw_points:
        try:
            point = normalize_point(raw)
        except Exception:
            continue
        dedup[point_key(point)] = point

    points = list(dedup.values())
    points.sort(
        key=lambda p: (
            p["entity_id"],
            p["signal_type"],
            p["metric"],
            p["timestamp"],
        )
    )
    return points


def normalize_reference_metadata(raw_records: Iterable[dict]) -> list[dict]:
    by_key: dict[PointKey, dict] = {}
    for raw in raw_records:
        try:
            timestamp = int(raw["timestamp"])
            entity_id = str(raw["entity_id"])
            signal_type = str(raw["signal_type"])
            metric = str(raw["metric"])
        except Exception:
            continue
        key: PointKey = (timestamp, entity_id, signal_type, metric)
        entry = by_key.setdefault(
            key,
            {
                "timestamp": timestamp,
                "entity_id": entity_id,
                "signal_type": signal_type,
                "metric": metric,
                "entity_type": raw.get("entity_type"),
                "anomaly_types": set(),
            },
        )
        anomaly_type = raw.get("anomaly_type")
        if anomaly_type:
            entry["anomaly_types"].add(str(anomaly_type))

    normalized = []
    for item in by_key.values():
        anomaly_types = sorted(item["anomaly_types"])
        normalized.append(
            {
                "timestamp": item["timestamp"],
                "entity_id": item["entity_id"],
                "signal_type": item["signal_type"],
                "metric": item["metric"],
                "entity_type": item["entity_type"],
                "anomaly_types": anomaly_types,
            }
        )

    normalized.sort(
        key=lambda x: (
            x["entity_id"],
            x["signal_type"],
            x["metric"],
            x["timestamp"],
        )
    )
    return normalized
