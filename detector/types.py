from typing import NotRequired, TypedDict


class DataPoint(TypedDict):
    timestamp: int
    entity_id: str
    signal_type: str
    metric: str
    value: float
    label: NotRequired[int]


class DetectionResult(TypedDict):
    timestamp: int
    entity_id: str
    signal_type: str
    metric: str
    anomaly_flag: bool
    score: float
    score_likelihood: float
    confidence: float
    explanation: str
    stream_key: str
    value: NotRequired[float]
    learn_applied: NotRequired[bool]
    scores: NotRequired[dict]
    episode_event: NotRequired[dict]
