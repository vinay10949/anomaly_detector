from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


class StreamModel:
    """Per-stream model with optional htm.core pathway and deterministic fallback."""
    _DEFAULT_COLLECTIVE_WINDOW = 20
    _DEFAULT_EWMA_ALPHA = 0.05
    _DEFAULT_COLLECTIVE_ENTRY_SCORE = 0.45
    _DEFAULT_COLLECTIVE_TRIGGER_POINTS = 4
    _DEFAULT_STRONG_SPIKE_Z = 6.0
    _DEFAULT_CONTEXTUAL_Z_DIVISOR = 3.0
    _DEFAULT_SCORE_COMBINE = "max"
    _DEFAULT_SCORE_WEIGHTS = {
        "tm": 1.0,
        "point": 0.6,
        "collective": 0.85,
        "contextual": 0.75,
    }
    _DEFAULT_CONTEXTUAL_MIN_RECORDS = 50
    _DEFAULT_LEARN_POLICY = "gated"
    _DEFAULT_LEARN_MAX_TM_SCORE = 0.80
    _DEFAULT_LEARN_MAX_POINT_Z = 3.5
    _DEFAULT_LEARN_MAX_CONTEXTUAL_SCORE = 0.70
    _DEFAULT_EPISODE_CONTINUE_SCORE = 0.30
    _DEFAULT_EPISODE_EXIT_POINTS = 3

    @staticmethod
    def default_htm_params() -> dict:
        return {
            "enc": {
                "value": {"size": 600, "sparsity": 0.02, "resolution": 0.1},
                "time": {"timeOfDay": (200, 10), "dayOfWeek": (100, 5)},
            },
            "sp": {"columnCount": 2048},
            "tm": {
                "cellsPerColumn": 8,
                "activationThreshold": 13,
                "initialPerm": 0.21,
                "permanenceInc": 0.1,
                "permanenceDec": 0.1,
            },
            "predictor": {"steps": [1, 5], "sdrc_alpha": 0.1, "resolution": 1.0},
            "anomaly": {"period": 256},
        }

    @staticmethod
    def normalize_htm_params(params: dict | None) -> dict:
        if not isinstance(params, dict) or not params:
            return StreamModel.default_htm_params()

        nested = StreamModel.default_htm_params()
        # If the caller already uses the community-style blocks, start from that.
        if any(k in params for k in ("enc", "sp", "tm", "predictor", "anomaly")):
            nested.update({k: params.get(k) for k in ("enc", "sp", "tm", "predictor", "anomaly") if k in params})

        # Backwards compat: allow flat keys and merge into the nested blocks.
        enc_value = ((nested.get("enc") or {}).get("value") or {})
        enc_value.update(
            {
                "size": params.get("enc_size", enc_value.get("size")),
                "sparsity": params.get("enc_sparsity", enc_value.get("sparsity")),
                "resolution": params.get("enc_resolution", enc_value.get("resolution")),
            }
        )
        nested.setdefault("enc", {})["value"] = enc_value

        sp = nested.get("sp") or {}
        if "sp_columnDimensions" in params and "columnCount" not in sp:
            sp["columnCount"] = params.get("sp_columnDimensions")
        nested["sp"] = sp

        tm = nested.get("tm") or {}
        tm_map = {
            "tm_cellsPerColumn": "cellsPerColumn",
            "tm_activationThreshold": "activationThreshold",
            "tm_initialPerm": "initialPerm",
            "tm_permanenceInc": "permanenceInc",
            "tm_permanenceDec": "permanenceDec",
        }
        for flat_key, nested_key in tm_map.items():
            if flat_key in params and nested_key not in tm:
                tm[nested_key] = params.get(flat_key)
        nested["tm"] = tm

        if "predictor" in params and isinstance(params.get("predictor"), dict):
            nested["predictor"] = params.get("predictor")
        if "anomaly" in params and isinstance(params.get("anomaly"), dict):
            nested["anomaly"] = params.get("anomaly")

        return nested

    def __init__(
        self,
        *,
        warmup_points: int = 120,
        raw_threshold: float = 0.80,
        likelihood_threshold: float = 0.995,
        likelihood_window: int = 256,
        htm_params: dict | None = None,
        use_temporal_features: bool = True,
        scoring: dict | None = None,
        learning: dict | None = None,
        episode: dict | None = None,
    ) -> None:
        self.warmup_points = warmup_points
        self.raw_threshold = raw_threshold
        self.likelihood_threshold = likelihood_threshold
        self.use_temporal_features = use_temporal_features
        self._scoring = scoring or {}
        self._learning = learning or {}
        self._episode_cfg = episode or {}
        self.stats = RunningStats()
        self.observations = 0
        self.raw_history: deque[float] = deque(maxlen=max(likelihood_window, 8))
        self.value_history: deque[float] = deque(
            maxlen=max(self._DEFAULT_COLLECTIVE_WINDOW * 4, 32)
        )
        self._collective_window = max(8, int(self._DEFAULT_COLLECTIVE_WINDOW))
        self._ewma_alpha = float(self._DEFAULT_EWMA_ALPHA)
        self._ewma_value: float | None = None
        self._collective_run_length = 0
        self._collective_entry_score = float(self._episode_cfg.get("entry_score", self._DEFAULT_COLLECTIVE_ENTRY_SCORE))
        self._collective_continue_score = float(
            self._episode_cfg.get("continue_score", self._DEFAULT_EPISODE_CONTINUE_SCORE)
        )
        self._collective_trigger_points = int(
            self._episode_cfg.get("trigger_points", self._DEFAULT_COLLECTIVE_TRIGGER_POINTS)
        )
        self._episode_exit_points = int(self._episode_cfg.get("exit_points", self._DEFAULT_EPISODE_EXIT_POINTS))
        self._strong_spike_z = float(self._DEFAULT_STRONG_SPIKE_Z)
        self.htm_params = StreamModel.normalize_htm_params(htm_params)
        self._score_combine = str(self._scoring.get("combine") or self._DEFAULT_SCORE_COMBINE).strip().lower()
        self._score_weights = self._load_weights(self._scoring.get("weights") or {})
        self._contextual_z_divisor = float(
            (self._scoring.get("contextual_z_divisor") or self._DEFAULT_CONTEXTUAL_Z_DIVISOR)
        )
        self._contextual_min_records = int(
            self._scoring.get("contextual_min_records") or self._DEFAULT_CONTEXTUAL_MIN_RECORDS
        )
        self._learn_policy = str(self._learning.get("policy") or self._DEFAULT_LEARN_POLICY)
        self._learn_max_tm_score = float(
            self._learning.get("max_tm_score") or self._DEFAULT_LEARN_MAX_TM_SCORE
        )
        self._learn_max_point_z = float(self._learning.get("max_point_z") or self._DEFAULT_LEARN_MAX_POINT_Z)
        self._learn_max_contextual_score = float(
            self._learning.get("max_contextual_score") or self._DEFAULT_LEARN_MAX_CONTEXTUAL_SCORE
        )

        self._htm_enabled = False
        self._htm_error: str | None = None
        self._last_htm_likelihood: float | None = None
        self._last_prediction: dict[int, float] | None = None
        self._predicted_next_value: float | None = None
        self._anomaly_likelihood = None
        self._anomaly_likelihood_period = 0
        self._predictor = None
        self._predictor_resolution = 1.0
        self._record_num = 0
        self._recent_episode_candidates: deque[tuple[int, float, str]] = deque(
            maxlen=max(8, int(self._collective_trigger_points) * 2)
        )
        self._episode_active: dict | None = None
        self._episode_cooldown = 0
        self._episode_id_seq = 0
        self._init_htm_core()

    def _extract_temporal_features(self, timestamp: int):
        """Extract datetime from timestamp for DateEncoder"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp)

    @property
    def uses_htm_core(self) -> bool:
        return self._htm_enabled

    @property
    def htm_error(self) -> str | None:
        return self._htm_error

    def _init_htm_core(self) -> None:
        # htm.core is optional at runtime; fallback scoring remains active.
        try:
            from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
            from htm.bindings.algorithms import Predictor, SpatialPooler, TemporalMemory
            from htm.bindings.sdr import SDR
            from htm.encoders.rdse import RDSE, RDSE_Parameters
            try:
                from htm.encoders.date import DateEncoder
            except Exception:  # pragma: no cover
                from htm.encoders.date_encoder import DateEncoder

            defaults = self.default_htm_params()
            params_dict = self.htm_params or {}

            default_enc_value = ((defaults.get("enc") or {}).get("value") or {})
            enc_value = (params_dict.get("enc", {}) or {}).get("value", {}) if "enc" in params_dict else {}
            p_size = int(enc_value.get("size", params_dict.get("enc_size", default_enc_value.get("size", 400))))
            p_sparsity = float(
                enc_value.get("sparsity", params_dict.get("enc_sparsity", default_enc_value.get("sparsity", 0.02)))
            )
            p_resolution = float(
                enc_value.get(
                    "resolution",
                    params_dict.get("enc_resolution", default_enc_value.get("resolution", 0.1)),
                )
            )

            sp_params = params_dict.get("sp", {}) or {}
            default_sp = defaults.get("sp") or {}
            p_col_dim = int(
                sp_params.get(
                    "columnCount",
                    params_dict.get("sp_columnDimensions", default_sp.get("columnCount", 2048)),
                )
            )

            tm_params = params_dict.get("tm", {}) or {}
            default_tm = defaults.get("tm") or {}
            p_cells_per_col = int(
                tm_params.get(
                    "cellsPerColumn",
                    params_dict.get("tm_cellsPerColumn", default_tm.get("cellsPerColumn", 8)),
                )
            )
            p_activationThreshold = int(
                tm_params.get(
                    "activationThreshold",
                    params_dict.get("tm_activationThreshold", default_tm.get("activationThreshold", 13)),
                )
            )
            p_initialPerm = float(
                tm_params.get("initialPerm", params_dict.get("tm_initialPerm", default_tm.get("initialPerm", 0.21)))
            )
            p_permanenceInc = float(
                tm_params.get(
                    "permanenceInc",
                    params_dict.get("tm_permanenceInc", default_tm.get("permanenceInc", 0.1)),
                )
            )
            p_permanenceDec = float(
                tm_params.get(
                    "permanenceDec",
                    params_dict.get("tm_permanenceDec", default_tm.get("permanenceDec", 0.1)),
                )
            )
            p_minThreshold = int(tm_params.get("minThreshold", 10))
            p_newSynapseCount = int(tm_params.get("newSynapseCount", 32))
            p_maxSegmentsPerCell = int(tm_params.get("maxSegmentsPerCell", 128))
            p_maxSynapsesPerSegment = int(tm_params.get("maxSynapsesPerSegment", 64))

            p_connectedPerm = float(sp_params.get("synPermConnected", 0.14))

            params = RDSE_Parameters()
            params.size = p_size
            params.sparsity = p_sparsity
            params.resolution = p_resolution

            self._value_encoder = RDSE(params)
            self._sdr_cls = SDR

            # Initialize DateEncoder if enabled
            if self.use_temporal_features:
                default_time = ((defaults.get("enc") or {}).get("time") or {})
                time_params = (params_dict.get("enc", {}) or {}).get("time", {}) if "enc" in params_dict else {}
                date_kwargs: dict = {"timeOfDay": time_params.get("timeOfDay", default_time.get("timeOfDay", (200, 10)))}
                if "weekend" in time_params:
                    date_kwargs["weekend"] = time_params.get("weekend", 21)
                else:
                    date_kwargs["dayOfWeek"] = time_params.get("dayOfWeek", default_time.get("dayOfWeek", (100, 5)))
                self._date_encoder = DateEncoder(**date_kwargs)
                self._input_size = int(self._value_encoder.size + self._date_encoder.size)
            else:
                self._date_encoder = None
                self._input_size = int(self._value_encoder.size)

            self._sp = SpatialPooler(
                inputDimensions=(self._input_size,),
                columnDimensions=(p_col_dim,),
                seed=42,
                globalInhibition=True,
                potentialPct=float(sp_params.get("potentialPct", 0.85)),
                potentialRadius=int(sp_params.get("potentialRadius", self._input_size)),
                localAreaDensity=float(sp_params.get("localAreaDensity", 0.02)),
                synPermInactiveDec=float(sp_params.get("synPermInactiveDec", 0.006)),
                synPermActiveInc=float(sp_params.get("synPermActiveInc", 0.04)),
                synPermConnected=p_connectedPerm,
                boostStrength=float(sp_params.get("boostStrength", 3.0)),
                wrapAround=bool(sp_params.get("wrapAround", True)),
            )
            self._active_columns = SDR((p_col_dim,))
            self._tm = TemporalMemory(
                columnDimensions=(p_col_dim,),
                cellsPerColumn=p_cells_per_col,
                activationThreshold=p_activationThreshold,
                initialPermanence=p_initialPerm,
                connectedPermanence=p_connectedPerm,
                minThreshold=p_minThreshold,
                maxNewSynapseCount=p_newSynapseCount,
                permanenceIncrement=p_permanenceInc,
                permanenceDecrement=p_permanenceDec,
                predictedSegmentDecrement=0.0,
                maxSegmentsPerCell=p_maxSegmentsPerCell,
                maxSynapsesPerSegment=p_maxSynapsesPerSegment,
                seed=42
            )

            anomaly_params = params_dict.get("anomaly", {}) or {}
            default_period = int(getattr(self.raw_history, "maxlen", 256) or 256)
            default_period = max(64, default_period)
            period = int(anomaly_params.get("period", default_period))
            self._anomaly_likelihood_period = period
            self._anomaly_likelihood = AnomalyLikelihood(period)

            predictor_params = params_dict.get("predictor", {}) or {}
            default_pred = defaults.get("predictor") or {}
            steps = predictor_params.get("steps", default_pred.get("steps", [1, 5]))
            alpha = float(predictor_params.get("sdrc_alpha", default_pred.get("sdrc_alpha", 0.1)))
            self._predictor_resolution = float(predictor_params.get("resolution", default_pred.get("resolution", 1.0)))
            self._predictor = Predictor(steps=steps, alpha=alpha)
            self._record_num = 0
            self._htm_enabled = True
        except Exception as exc:
            self._htm_error = str(exc)
            self._htm_enabled = False

    def reset_sequence_state(self) -> None:
        """Reset short-term temporal state while preserving learned HTM synapses."""
        if not self._htm_enabled:
            return
        try:
            if hasattr(self._tm, "reset"):
                self._tm.reset()
        except Exception:
            pass
        self._last_htm_likelihood = None
        self._last_prediction = None
        self.raw_history.clear()
        self._collective_run_length = 0

        # Likelihood is sequence/history-dependent; reset it to avoid replay bias.
        if self._anomaly_likelihood is not None and self._anomaly_likelihood_period > 0:
            try:
                self._anomaly_likelihood = type(self._anomaly_likelihood)(self._anomaly_likelihood_period)
            except Exception:
                self._anomaly_likelihood = None
        self._predicted_next_value = None
        self._recent_episode_candidates.clear()
        self._episode_active = None
        self._episode_cooldown = 0

    def _load_weights(self, override: dict) -> dict[str, float]:
        weights = dict(self._DEFAULT_SCORE_WEIGHTS)
        for k, v in (override or {}).items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue
        # For sum-mode, normalize to avoid score inflation.
        if self._score_combine == "sum":
            total = sum(max(0.0, float(v)) for v in weights.values())
            if total <= 0.0:
                return dict(self._DEFAULT_SCORE_WEIGHTS)
            return {k: max(0.0, float(v)) / total for k, v in weights.items()}
        # For max-mode, treat weights as multipliers (no normalization).
        return {k: max(0.0, float(v)) for k, v in weights.items()}

    def _contextual_score(self, value: float) -> tuple[float, float | None, float | None]:
        if (
            self._predicted_next_value is None
            or self.stats.n < 10
            or int(self._record_num) < int(self._contextual_min_records)
        ):
            return 0.0, None, None
        std = max(self.stats.std, 1e-6)
        expected = float(self._predicted_next_value)
        residual = float(value) - expected
        z = abs(residual) / std
        divisor = max(0.5, float(self._contextual_z_divisor))
        return min(1.0, z / divisor), expected, residual

    def _value_to_bucket(self, value: float) -> int:
        enc = getattr(self, "_value_encoder", None)
        for name in ("getBucketIndex", "getBucketIdx"):
            fn = getattr(enc, name, None)
            if callable(fn):
                try:
                    return int(fn(float(value)))
                except Exception:
                    pass
        for name in ("getBucketIndices", "getBucketIdxs"):
            fn = getattr(enc, name, None)
            if callable(fn):
                try:
                    out = fn(float(value))
                    if isinstance(out, (list, tuple)) and out:
                        return int(out[0])
                except Exception:
                    pass
        denom = float(self._predictor_resolution) if float(self._predictor_resolution) != 0.0 else 1.0
        return int(float(value) / denom)

    def _bucket_to_value(self, bucket_idx: int) -> float:
        enc = getattr(self, "_value_encoder", None)
        for name in ("getBucketValue", "getBucketValues"):
            fn = getattr(enc, name, None)
            if callable(fn):
                try:
                    val = fn(int(bucket_idx))
                    if isinstance(val, (list, tuple)):
                        if val:
                            return float(val[0])
                    return float(val)
                except Exception:
                    pass
        return float(bucket_idx) * float(self._predictor_resolution)

    def _should_learn(
        self,
        *,
        mode: str,
        tm_score: float,
        point_z: float,
        contextual_score: float,
        episode_active: bool,
        label: int | None,
    ) -> bool:
        """
        Smart online learning gate for production-style HTM operation.

        Modes:
        - 'offline': force learning (training / warm-up)
        - 'predict_only': disable learning
        - 'online': apply policy-driven anomaly gating
        """
        # Never learn on labeled anomalies
        if label is not None and int(label) != 0:
            return False

        mode_key = str(mode).strip().lower()
        if mode_key == "offline":
            return True
        if mode_key == "predict_only":
            return False
        if mode_key != "online":
            return False

        policy = str(self._learn_policy).strip().lower()
        if policy in {"never", "off", "disabled"}:
            return False
        if policy in {"always", "on"}:
            return True

        # Default policy: gated learning
        if episode_active:
            return False
        if float(tm_score) > float(self._learn_max_tm_score):
            return False
        if not math.isfinite(float(point_z)) or abs(float(point_z)) > float(self._learn_max_point_z):
            return False
        if float(contextual_score) > float(self._learn_max_contextual_score):
            return False
        return True

    def _update_short_term(self, value: float) -> None:
        self.value_history.append(float(value))
        if self._ewma_value is None:
            self._ewma_value = float(value)
            return
        a = min(0.5, max(0.0, self._ewma_alpha))
        self._ewma_value = (1.0 - a) * self._ewma_value + a * float(value)

    def train(self, value: float, timestamp: int) -> None:
        # Training is also warmup: advance internal counters so post-train detection
        # doesn't incorrectly treat the stream as "cold".
        htm_raw = self._step_htm(value, learn=True, timestamp=timestamp)
        if htm_raw is None:
            raise RuntimeError(f"HTM step failed: {self._htm_error or 'unknown error'}")
        raw_score = float(htm_raw)
        self.raw_history.append(raw_score)
        self.observations += 1
        self.stats.update(float(value))
        self._update_short_term(float(value))

    def detect(self, value: float, timestamp: int, *, mode: str = 'online', label: int | None = None) -> dict:
        """
        Detect anomalies in a data point.
        
        Args:
            value: The value to analyze
            timestamp: Unix timestamp
            mode: Learning mode - 'offline' (always learn), 'online' (smart learning), 'predict_only' (never learn)
            label: Optional ground truth label (1 = anomaly, 0 = normal)
        
        Returns:
            Dictionary with anomaly_flag, scores, and metadata
        """
        point_score, point_z = self._point_deviation_score(float(value))
        contextual_score, expected_value, residual = self._contextual_score(float(value))
        
        # First, get TM prediction (before learning)
        tm_raw = self._step_htm(value, learn=False, timestamp=timestamp)
        if tm_raw is None:
            raise RuntimeError(f"HTM step failed: {self._htm_error or 'unknown error'}")
        tm_score = float(tm_raw)
        
        # Decide whether to learn based on mode and prediction quality
        learn_applied = self._should_learn(
            mode=str(mode),
            tm_score=float(tm_score),
            point_z=float(point_z),
            contextual_score=float(contextual_score),
            episode_active=bool(self._episode_active),
            label=label,
        )
        
        # If learning, step HTM again with learn=True
        if learn_applied:
            tm_raw_learn = self._step_htm(value, learn=True, timestamp=timestamp)
            if tm_raw_learn is None:
                raise RuntimeError(f"HTM step failed: {self._htm_error or 'unknown error'}")

        collective_score = self._collective_shift_score(float(value))
        episode_event, episode_flag = self._update_episode_state(
            timestamp=timestamp,
            collective_score=float(collective_score),
            contextual_score=float(contextual_score),
        )

        weights = self._score_weights
        if self._score_combine == "sum":
            combined = (
                weights.get("tm", 0.0) * tm_score
                + weights.get("point", 0.0) * float(point_score)
                + weights.get("collective", 0.0) * float(collective_score)
                + weights.get("contextual", 0.0) * float(contextual_score)
            )
            raw_score = float(max(0.0, min(1.0, combined)))
        else:
            raw_score = float(
                max(
                    weights.get("tm", 0.0) * tm_score,
                    weights.get("point", 0.0) * float(point_score),
                    weights.get("collective", 0.0) * float(collective_score),
                    weights.get("contextual", 0.0) * float(contextual_score),
                )
            )
            raw_score = float(max(0.0, min(1.0, raw_score)))

        htm_likelihood = float(self._last_htm_likelihood) if self._last_htm_likelihood is not None else 0.0
        fallback_likelihood = self._likelihood(raw_score)
        likelihood = htm_likelihood if self._last_htm_likelihood is not None else fallback_likelihood

        self.raw_history.append(raw_score)
        self.observations += 1
        self.stats.update(float(value))
        self._update_short_term(float(value))

        is_warm = self.observations >= self.warmup_points
        strong_spike = float(point_z) >= float(self._strong_spike_z)
        # Episodes are tracked and emitted separately (start/end). Point-level anomaly_flag
        # is driven by likelihood/spike to avoid turning an entire sustained episode into
        # thousands of "anomalous points" in batch UX.
        # Allow clear point spikes to surface even during warmup; keep likelihood-gated
        # anomalies behind warmup to avoid noisy cold-start alerts.
        is_anomaly = bool(strong_spike) or (
            is_warm and float(likelihood) >= float(self.likelihood_threshold)
        )

        scores = {
            "tm": tm_score,
            "point": float(point_score),
            "collective": float(collective_score),
            "contextual": float(contextual_score),
            "combined": raw_score,
            "combine_mode": self._score_combine,
            "likelihood_htm": htm_likelihood,
            "likelihood_fallback": fallback_likelihood,
            "expected_value": expected_value,
            "residual": residual,
            "episode_active": bool(episode_flag),
        }

        if not is_warm:
            explanation = "Warmup phase; scoring only."
        elif strong_spike:
            explanation = "Strong point spike detected."
        elif episode_flag:
            explanation = "Collective anomaly episode active."
        elif contextual_score >= 0.6:
            explanation = "Contextual anomaly (unexpected vs predictor)."
        elif raw_score >= float(self.raw_threshold):
            explanation = "High raw anomaly score (below likelihood threshold)."
        elif is_anomaly:
            explanation = "Anomaly thresholds exceeded."
        else:
            explanation = "Below anomaly thresholds."

        confidence = max(raw_score, float(likelihood))
        return {
            "anomaly_flag": bool(is_anomaly),
            "score": float(raw_score),
            "score_likelihood": float(likelihood),
            "confidence": float(confidence),
            "explanation": str(explanation),
            "scores": scores,
            "episode_event": episode_event,
            "learn_applied": bool(learn_applied),
        }

    def _update_episode_state(
        self,
        *,
        timestamp: int,
        collective_score: float,
        contextual_score: float,
    ) -> tuple[dict | None, bool]:
        kind = "collective" if collective_score >= contextual_score else "contextual"
        signal = float(max(collective_score, contextual_score))
        if signal >= float(self._collective_entry_score):
            self._collective_run_length += 1
            self._recent_episode_candidates.append((int(timestamp), signal, kind))
        else:
            self._collective_run_length = 0
            self._recent_episode_candidates.clear()

        event: dict | None = None
        if self._episode_active is None:
            if self._collective_run_length >= int(self._collective_trigger_points):
                self._episode_id_seq += 1
                start_ts = int(self._recent_episode_candidates[0][0]) if self._recent_episode_candidates else int(timestamp)
                self._episode_active = {
                    "id": int(self._episode_id_seq),
                    "kind": str(kind),
                    "start_timestamp": int(start_ts),
                    "last_timestamp": int(timestamp),
                    "points": int(self._collective_run_length),
                    "max_signal": float(signal),
                }
                self._episode_cooldown = 0
                event = {"event": "start", **self._episode_active}
        else:
            self._episode_active["last_timestamp"] = int(timestamp)
            self._episode_active["points"] = int(self._episode_active.get("points", 0)) + 1
            self._episode_active["max_signal"] = float(max(float(self._episode_active.get("max_signal", 0.0)), signal))

            if signal >= float(self._collective_continue_score):
                self._episode_cooldown = 0
            else:
                self._episode_cooldown += 1
                if self._episode_cooldown >= int(self._episode_exit_points):
                    ended = dict(self._episode_active)
                    ended["end_timestamp"] = int(timestamp)
                    event = {"event": "end", **ended}
                    self._episode_active = None
                    self._episode_cooldown = 0
                    self._recent_episode_candidates.clear()

        return event, self._episode_active is not None

    def flush_episode(self, timestamp: int) -> dict | None:
        if self._episode_active is None:
            return None
        ended = dict(self._episode_active)
        ended["end_timestamp"] = int(timestamp)
        event = {"event": "end", **ended}
        self._episode_active = None
        self._episode_cooldown = 0
        self._recent_episode_candidates.clear()
        return event

    def _point_deviation_score(self, value: float) -> tuple[float, float]:
        if self.stats.n < 5:
            return 0.0, 0.0
        std = max(self.stats.std, 1e-6)
        mean = self.stats.mean
        z = abs((float(value) - mean) / std)
        return min(1.0, z / 6.0), z

    def _collective_shift_score(self, value: float) -> float:
        if self.stats.n < max(10, self._collective_window):
            return 0.0
        std = max(self.stats.std, 1e-6)
        mean = self.stats.mean

        window = list(self.value_history)[-(self._collective_window - 1) :]
        window.append(float(value))
        if len(window) < 4:
            return 0.0

        window_mean = sum(window) / len(window)
        window_var = sum((x - window_mean) ** 2 for x in window) / len(window)
        window_std = math.sqrt(max(window_var, 1e-6))

        z_mean = abs((window_mean - mean) / std)
        mean_shift_score = min(1.0, z_mean / 3.0)

        std_ratio = max(window_std / std, 1e-6)
        var_shift = abs(math.log(std_ratio))
        variance_shift_score = min(1.0, var_shift / math.log(4.0))

        slope = (float(window[-1]) - float(window[0])) / max(1.0, float(len(window)))
        slope_z = abs(slope) / max(std, 1e-6)
        slope_score = min(1.0, slope_z / 0.25)

        return max(mean_shift_score, variance_shift_score, slope_score)

    def _likelihood(self, raw_score: float) -> float:
        if len(self.raw_history) < 5:
            return raw_score

        mean = sum(self.raw_history) / len(self.raw_history)
        variance = sum((x - mean) ** 2 for x in self.raw_history) / len(self.raw_history)
        std = math.sqrt(max(variance, 1e-6))
        z = (raw_score - mean) / std
        
        # Use a proper anomaly likelihood calculation
        # Higher z-scores should result in higher likelihood
        # We want likelihood to be high only when the score is significantly above the mean
        if z <= 0:
            # Score is at or below mean, not anomalous
            return 0.0
        
        # For positive z, use a sigmoid-like transformation
        # This maps z-scores to [0, 1] range
        try:
            # Scale z so that z=3 gives ~0.95 likelihood
            scaled_z = z / 3.0
            likelihood = min(1.0, scaled_z)
        except OverflowError:
            likelihood = 1.0
        
        return likelihood

    def _step_htm(self, value: float, *, learn: bool, timestamp: int) -> float | None:
        if not self._htm_enabled:
            return None

        try:
            self._last_htm_likelihood = None
            self._last_prediction = None

            # Encode value
            value_encoded = self._value_encoder.encode(value)
            
            # Convert to sparse list
            if isinstance(value_encoded, self._sdr_cls):
                value_indices = list(value_encoded.sparse)
            elif hasattr(value_encoded, "sparse"):
                value_indices = list(value_encoded.sparse)
            else:
                value_indices = [idx for idx, bit in enumerate(value_encoded) if bit]

            # Concatenate with temporal encoding if enabled
            if self.use_temporal_features and self._date_encoder:
                dt = self._extract_temporal_features(timestamp)
                temporal_encoded = self._date_encoder.encode(dt)
                
                # Convert temporal encoding to sparse list
                if isinstance(temporal_encoded, self._sdr_cls):
                    temporal_indices = list(temporal_encoded.sparse)
                elif hasattr(temporal_encoded, "sparse"):
                    temporal_indices = list(temporal_encoded.sparse)
                else:
                    temporal_indices = [idx for idx, bit in enumerate(temporal_encoded) if bit]

                # Offset temporal bits by value encoder size
                value_width = int(self._value_encoder.size)
                temporal_indices = [idx + value_width for idx in temporal_indices]
                
                # Combine indices
                combined_indices = value_indices + temporal_indices
            else:
                combined_indices = value_indices

            # Create input SDR
            input_sdr = self._sdr_cls((self._input_size,))
            input_sdr.sparse = combined_indices

            self._sp.compute(input_sdr, learn, self._active_columns)
            self._tm.compute(self._active_columns, learn=learn)
            anomaly = getattr(self._tm, "anomaly", None)
            if anomaly is None:
                return None

            if self._anomaly_likelihood is not None:
                try:
                    val = float(self._anomaly_likelihood.compute(float(anomaly)))
                    # AnomalyLikelihood returns 0.0 until it has enough history (period).
                    # Treat that as "not available" and fall back to local likelihood.
                    if val == 0.0 and getattr(self._anomaly_likelihood, "n_records", 0) < getattr(
                        self._anomaly_likelihood, "period", 0
                    ):
                        self._last_htm_likelihood = None
                    else:
                        self._last_htm_likelihood = val
                except Exception:
                    self._last_htm_likelihood = None

            if self._predictor is not None:
                try:
                    active_cells = self._tm.getActiveCells()
                    pdf = self._predictor.infer(active_cells) or {}
                    preds: dict[int, float] = {}
                    for step, arr in pdf.items():
                        if arr is None:
                            continue
                        try:
                            n = len(arr)
                        except Exception:
                            continue
                        if n <= 0:
                            continue
                        bucket = max(range(n), key=lambda i: arr[i])
                        preds[int(step)] = float(self._bucket_to_value(int(bucket)))
                    self._last_prediction = preds or None
                    if self._last_prediction is not None and 1 in self._last_prediction:
                        self._predicted_next_value = float(self._last_prediction[1])

                    if learn:
                        bucket_idx = self._value_to_bucket(float(value))
                        self._predictor.learn(self._record_num, active_cells, int(bucket_idx))
                except Exception:
                    self._last_prediction = None
                    self._predicted_next_value = None

            self._record_num += 1
            return float(max(0.0, min(1.0, anomaly)))
        except Exception as exc:
            self._htm_error = f"htm.core runtime disabled: {exc}"
            self._htm_enabled = False
            return None
