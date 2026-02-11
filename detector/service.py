from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

from .model import StreamModel
from .normalization import normalize_points, normalize_reference_metadata, point_key, stream_key, stream_key_str
from .types import DataPoint, DetectionResult


@dataclass
class DetectorConfig:
    warmup_points: int = 50
    raw_threshold: float = 0.80
    likelihood_threshold: float = 0.99
    likelihood_window: int = 256
    htm_params: dict | None = None
    use_temporal_features: bool = True
    require_htm: bool = True
    scoring: dict | None = None
    learning: dict | None = None
    episode: dict | None = None


class AnomalyDetectorService:
    STATE_FORMAT = "anomaly-detector-service-state-v1"
    FULL_FORMAT = "anomaly-detector-service-full-pickle-v1"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._refresh_runtime_flags()
        if self.config.require_htm and not self._htm_core_available:
            raise RuntimeError(
                f"HTM is required but not available in this Python ({self._htm_core_error or 'unknown error'}). "
                "Start the server with: .venv/bin/python api_server.py"
            )
        self.config.htm_params = StreamModel.normalize_htm_params(self.config.htm_params)
        self.reset()

    def _refresh_runtime_flags(self) -> None:
        try:
            import htm  # noqa: F401
            self._htm_core_available = True
            self._htm_core_error = None
        except Exception as exc:  # pragma: no cover
            self._htm_core_available = False
            self._htm_core_error = str(exc)

    def _ensure_runtime_flags(self) -> None:
        if not hasattr(self, "_htm_core_available"):
            self._refresh_runtime_flags()
        if not hasattr(self, "_htm_core_error"):
            self._htm_core_error = None

    def reset(self) -> None:
        self._models: dict[tuple[str, str, str], StreamModel] = {}
        self._trained = False
        self._points_seen = 0
        self._points_used = 0
        self._htm_core_streams = 0
        self._training_active = False
        self._training_total = 0
        self._training_processed = 0
        self._training_started_at: float | None = None
        self._training_last_update_at: float | None = None

    def training_status(self) -> dict:
        if not self._training_active or self._training_started_at is None:
            return {"active": False}

        now = time.monotonic()
        elapsed = max(0.0, now - self._training_started_at)
        processed = int(self._training_processed)
        total = int(self._training_total)
        rate = (processed / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total - processed)
        eta = (remaining / rate) if rate > 0 else None
        return {
            "active": True,
            "processed": processed,
            "total": total,
            "elapsed_sec": elapsed,
            "rate_points_per_sec": rate,
            "eta_sec": eta,
        }

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        raw_points: list[dict],
        *,
        reference_metadata: list[dict] | None = None,
        log_every: int | None = None,
        log_fn: Optional[Callable[[str], None]] = None,
        checkpoint_every: int | None = None,
        checkpoint_keep: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str = "model.checkpoint",
    ) -> dict:
        points = normalize_points(raw_points)
        self.reset()
        self._points_seen = len(raw_points)

        excluded_reference = 0
        if reference_metadata:
            normalized_ref = normalize_reference_metadata(reference_metadata)
            ref_keys = {
                (int(r["timestamp"]), str(r["entity_id"]), str(r["signal_type"]), str(r["metric"]))
                for r in normalized_ref
            }
            filtered: list[DataPoint] = []
            for p in points:
                if point_key(p) in ref_keys:
                    excluded_reference += 1
                else:
                    filtered.append(p)
            points = filtered

        self._points_used = len(points)
        self._training_active = True
        self._training_total = len(points)
        self._training_processed = 0
        self._training_started_at = time.monotonic()
        self._training_last_update_at = self._training_started_at

        total = len(points)
        last_stream = None
        checkpoint_paths: list[str] = []
        checkpoint_keep = max(1, int(checkpoint_keep))
        if checkpoint_every is not None:
            try:
                checkpoint_every = int(checkpoint_every)
            except Exception:
                checkpoint_every = None
            if checkpoint_every is not None and checkpoint_every <= 0:
                checkpoint_every = None

        if log_fn and log_every and log_every > 0:
            log_fn(
                f"[TRAIN] starting points_seen={self._points_seen} points_used={self._points_used} "
                f"excluded_reference={excluded_reference} streams_estimate={len({stream_key(p) for p in points})}"
            )
        for idx, point in enumerate(points, start=1):
            # Update this before the potentially-expensive HTM step so /status can show progress
            # and logs can appear even if the first compute takes a long time.
            self._training_processed = idx - 1
            model = self._model_for_point(point)
            if log_fn and log_every and log_every > 0:
                if idx == 1 or idx % log_every == 0 or idx == total:
                    key = stream_key(point)
                    log_fn(f"[TRAIN] processing {idx}/{total} stream={stream_key_str(key)}")

            model.train(point["value"], point["timestamp"])
            self._training_processed = idx

            if checkpoint_every and checkpoint_dir and idx % checkpoint_every == 0:
                try:
                    import os

                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ck_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}.{idx}.full.pkl")
                    self.save_full(ck_path)
                    checkpoint_paths.append(ck_path)
                    while len(checkpoint_paths) > checkpoint_keep:
                        old = checkpoint_paths.pop(0)
                        try:
                            os.remove(old)
                        except Exception:
                            pass
                    if log_fn:
                        log_fn(f"[TRAIN] checkpoint saved: {ck_path}")
                except Exception as exc:  # pragma: no cover
                    if log_fn:
                        log_fn(f"[TRAIN] checkpoint failed at {idx}: {exc}")

            if log_fn and log_every and log_every > 0:
                key = stream_key(point)
                if key != last_stream:
                    last_stream = key
                    log_fn(f"[TRAIN] stream={stream_key_str(key)} uses_htm={model.uses_htm_core}")
                if idx == 1 or idx % log_every == 0 or idx == total:
                    status = self.training_status()
                    pct = (idx / total * 100.0) if total else 100.0
                    eta = status.get("eta_sec")
                    eta_str = f"{eta:.1f}s" if isinstance(eta, (int, float)) else "n/a"
                    rate = float(status.get("rate_points_per_sec") or 0.0)
                    log_fn(
                        f"[TRAIN] {idx}/{total} ({pct:.1f}%) "
                        f"rate={rate:.1f} pts/s elapsed={status.get('elapsed_sec', 0.0):.1f}s eta={eta_str}"
                    )

        self._trained = True
        self._htm_core_streams = sum(1 for model in self._models.values() if model.uses_htm_core)
        self._training_active = False
        return {
            "trained_streams": len(self._models),
            "points_seen": self._points_seen,
            "points_used": self._points_used,
            "warmup_points": self.config.warmup_points,
            "raw_threshold": self.config.raw_threshold,
            "likelihood_threshold": self.config.likelihood_threshold,
            "htm_core_streams": self._htm_core_streams,
            "excluded_reference": excluded_reference,
            "checkpoints_saved": len(checkpoint_paths),
        }

    def detect(self, raw_point: dict, mode: str = 'online') -> DetectionResult:
        # Backward compatibility: full pickles from older versions may not have
        # runtime flag attributes. Rebuild them lazily on first detect call.
        self._ensure_runtime_flags()
        if self.config.require_htm and not self._htm_core_available:
            raise RuntimeError(
                f"HTM runtime unavailable ({self._htm_core_error or 'unknown error'}). "
                "Start the server with: .venv/bin/python api_server.py"
            )

        points = normalize_points([raw_point])
        if not points:
            raise ValueError("No valid point supplied")

        if not self._trained:
            # Online initialization path (used for streaming simulation).
            self._points_seen += 1
            self._points_used += 1
            result = self._detect_point(points[0], mode='online')
            self._trained = True
            self._htm_core_streams = sum(1 for model in self._models.values() if model.uses_htm_core)
            return result

        return self._detect_point(points[0], mode=mode)

    def _detect_point(
        self,
        point: DataPoint,
        mode: str = 'online',
        *,
        model: StreamModel | None = None,
    ) -> DetectionResult:
        model = model or self._model_for_point(point)
        details = model.detect(
            point["value"],
            point["timestamp"],
            mode=str(mode),
            label=point.get("label"),
        )
        result: DetectionResult = {
            "timestamp": point["timestamp"],
            "entity_id": point["entity_id"],
            "signal_type": point["signal_type"],
            "metric": point["metric"],
            "value": point["value"],
            "anomaly_flag": bool(details["anomaly_flag"]),
            "score": float(details["score"]),
            "score_likelihood": float(details["score_likelihood"]),
            "confidence": float(details["confidence"]),
            "explanation": str(details["explanation"]),
            "stream_key": stream_key_str(stream_key(point)),
        }
        if "scores" in details and isinstance(details["scores"], dict):
            result["scores"] = details["scores"]
        if details.get("episode_event") is not None:
            result["episode_event"] = details["episode_event"]
        result["learn_applied"] = bool(details.get("learn_applied", False))
        return result

    def detect_batch(
        self,
        raw_points: list[dict],
        *,
        return_scores: bool = True,
        mode: str = 'predict_only',
        reset_sequence: bool = True,
        batch_warmup_points: int | None = None,
        finalize_episodes: bool = True,
    ) -> dict:
        if not self._trained:
            raise RuntimeError("Model not trained")

        points = normalize_points(raw_points)
        mode_str = str(mode)
        learn_enabled = mode_str != "predict_only"
        results: list[dict] = []
        detected_count = 0
        learned_count = 0
        episodes: list[dict] = []
        if batch_warmup_points is None:
            reset_warmup = max(0, min(8, int(self.config.warmup_points)))
        else:
            reset_warmup = max(0, int(batch_warmup_points))
        seen_since_reset: dict[tuple[str, str, str], int] = {}
        last_ts_by_stream: dict[tuple[str, str, str], int] = {}
        models_used: dict[tuple[str, str, str], StreamModel] = {}

        for point in points:
            key = stream_key(point)
            model = self._model_for_point(point)
            models_used[key] = model
            last_ts_by_stream[key] = int(point["timestamp"])
            if reset_sequence and key not in seen_since_reset:
                model.reset_sequence_state()
                seen_since_reset[key] = 0

            result = self._detect_point(point, mode=mode, model=model)
            if reset_sequence:
                seen_since_reset[key] = seen_since_reset.get(key, 0) + 1
                if reset_warmup > 0 and seen_since_reset[key] <= reset_warmup:
                    result["anomaly_flag"] = False
                    result["explanation"] = (
                        f"Batch warmup after sequence reset "
                        f"({seen_since_reset[key]}/{reset_warmup}); anomaly suppressed."
                    )
            if result["anomaly_flag"]:
                detected_count += 1
            if result.get("learn_applied"):
                learned_count += 1
            episode_event = result.get("episode_event")
            if isinstance(episode_event, dict) and episode_event.get("event") == "end":
                ended = dict(episode_event)
                ended.pop("event", None)
                episodes.append(ended)

            if return_scores:
                results.append(result)
            else:
                results.append(
                    {
                        "timestamp": result["timestamp"],
                        "entity_id": result["entity_id"],
                        "signal_type": result["signal_type"],
                        "metric": result["metric"],
                        "anomaly_flag": result["anomaly_flag"],
                    }
                )

        if finalize_episodes and models_used:
            for key, model in models_used.items():
                last_ts = last_ts_by_stream.get(key)
                if last_ts is None:
                    continue
                ev = model.flush_episode(int(last_ts))
                if isinstance(ev, dict) and ev.get("event") == "end":
                    ended = dict(ev)
                    ended.pop("event", None)
                    episodes.append(ended)

        return {
            "results": results,
            "summary": {
                "detected_count": detected_count,
                "total": len(points),
                "learn": bool(learn_enabled),
                "mode": mode_str,
                "learned_points": int(learned_count),
                "reset_sequence": bool(reset_sequence),
                "batch_warmup_points": int(reset_warmup),
            },
            "episodes": episodes,
            "episode_summary": {
                "total": len(episodes),
                "by_kind": {
                    kind: sum(1 for e in episodes if e.get("kind") == kind)
                    for kind in sorted({str(e.get("kind")) for e in episodes if e.get("kind") is not None})
                },
            },
        }

    def status(self) -> dict:
        import sys
        try:
            import htm  # noqa: F401
            htm_available = True
        except Exception as exc:
            htm_available = False
            htm_exc = str(exc)

        stream_items = list(self._models.items())
        stream_limit = 25
        sample = stream_items[:stream_limit]
        stream_diag = []
        for key, model in sample:
            try:
                n_records = int(getattr(model._anomaly_likelihood, "n_records", 0)) if model._anomaly_likelihood else 0
                period = int(getattr(model._anomaly_likelihood, "period", 0)) if model._anomaly_likelihood else 0
            except Exception:
                n_records, period = 0, 0
            stream_diag.append(
                {
                    "stream_key": stream_key_str(key),
                    "uses_htm_core": bool(model.uses_htm_core),
                    "htm_error": model.htm_error,
                    "predictor_enabled": bool(model._predictor is not None),
                    "prediction_available": bool(getattr(model, "_predicted_next_value", None) is not None),
                    "likelihood_ready": bool(period > 0 and n_records >= period),
                    "episode_active": bool(getattr(model, "_episode_active", None) is not None),
                }
            )

        return {
            "status": "trained" if self._trained else "not_trained",
            "trained_streams": len(self._models),
            "points_seen": self._points_seen,
            "points_used": self._points_used,
            "htm_core_streams": self._htm_core_streams,
            "training": self.training_status(),
            "runtime": {
                "python": sys.executable,
                "python_version": sys.version.split()[0],
                "htm_available": htm_available,
                **({"htm_error": htm_exc} if not htm_available else {}),
            },
            "config": {
                "warmup_points": self.config.warmup_points,
                "raw_threshold": self.config.raw_threshold,
                "likelihood_threshold": self.config.likelihood_threshold,
                "likelihood_window": self.config.likelihood_window,
                "use_temporal_features": self.config.use_temporal_features,
                "require_htm": self.config.require_htm,
                "htm_params": self.config.htm_params,
                "scoring": self.config.scoring,
                "learning": self.config.learning,
                "episode": self.config.episode,
            },
            "streams_sample": stream_diag,
            "streams_truncated": len(stream_items) > stream_limit,
        }

    def _model_for_point(self, point: DataPoint) -> StreamModel:
        key = stream_key(point)
        if key not in self._models:
            model = StreamModel(
                warmup_points=self.config.warmup_points,
                raw_threshold=self.config.raw_threshold,
                likelihood_threshold=self.config.likelihood_threshold,
                likelihood_window=self.config.likelihood_window,
                htm_params=self.config.htm_params,
                use_temporal_features=self.config.use_temporal_features,
                scoring=self.config.scoring,
                learning=self.config.learning,
                episode=self.config.episode,
            )
            if self.config.require_htm and not model.uses_htm_core:
                raise RuntimeError(
                    "HTM initialization failed for stream "
                    f"{stream_key_str(key)}: {model.htm_error or 'unknown error'}"
                )
            self._models[key] = model
        return self._models[key]

    def _to_state(self) -> dict:
        return {
            "format": self.STATE_FORMAT,
            "config": {
                "warmup_points": self.config.warmup_points,
                "raw_threshold": self.config.raw_threshold,
                "likelihood_threshold": self.config.likelihood_threshold,
                "likelihood_window": self.config.likelihood_window,
                "htm_params": self.config.htm_params,
                "use_temporal_features": self.config.use_temporal_features,
                "require_htm": self.config.require_htm,
                "scoring": self.config.scoring,
                "learning": self.config.learning,
                "episode": self.config.episode,
            },
            "trained": self._trained,
            "points_seen": self._points_seen,
            "points_used": self._points_used,
            "htm_core_streams": self._htm_core_streams,
            "models": [
                {
                    "stream": [k[0], k[1], k[2]],
                    "stats": {
                        "n": m.stats.n,
                        "mean": m.stats.mean,
                        "m2": m.stats.m2,
                    },
                    "observations": m.observations,
                    "raw_history": list(m.raw_history),
                    "value_history": list(m.value_history),
                    "ewma_value": m._ewma_value,
                    "predicted_next_value": getattr(m, "_predicted_next_value", None),
                    "episode_active": getattr(m, "_episode_active", None),
                    "episode_cooldown": getattr(m, "_episode_cooldown", 0),
                }
                for k, m in self._models.items()
            ],
        }

    @classmethod
    def _from_state(cls, state: dict) -> "AnomalyDetectorService":
        if state.get("format") != cls.STATE_FORMAT:
            raise ValueError("Unsupported model state format")

        cfg = state.get("config") or {}
        service = cls(
            DetectorConfig(
                warmup_points=int(cfg.get("warmup_points", DetectorConfig.warmup_points)),
                raw_threshold=float(cfg.get("raw_threshold", DetectorConfig.raw_threshold)),
                likelihood_threshold=float(cfg.get("likelihood_threshold", DetectorConfig.likelihood_threshold)),
                likelihood_window=int(cfg.get("likelihood_window", DetectorConfig.likelihood_window)),
                htm_params=cfg.get("htm_params"),
                use_temporal_features=bool(cfg.get("use_temporal_features", True)),
                require_htm=bool(cfg.get("require_htm", True)),
                scoring=cfg.get("scoring"),
                learning=cfg.get("learning"),
                episode=cfg.get("episode"),
            )
        )
        service.reset()

        service._trained = bool(state.get("trained", False))
        service._points_seen = int(state.get("points_seen", 0))
        service._points_used = int(state.get("points_used", 0))
        service._htm_core_streams = int(state.get("htm_core_streams", 0))

        models = state.get("models") or []
        for entry in models:
            stream = entry.get("stream") or []
            if not (isinstance(stream, list) and len(stream) == 3):
                continue
            key = (str(stream[0]), str(stream[1]), str(stream[2]))
            model = StreamModel(
                warmup_points=service.config.warmup_points,
                raw_threshold=service.config.raw_threshold,
                likelihood_threshold=service.config.likelihood_threshold,
                likelihood_window=service.config.likelihood_window,
                htm_params=service.config.htm_params,
                use_temporal_features=service.config.use_temporal_features,
                scoring=service.config.scoring,
                learning=service.config.learning,
                episode=service.config.episode,
            )

            stats = entry.get("stats") or {}
            try:
                model.stats.n = int(stats.get("n", 0))
                model.stats.mean = float(stats.get("mean", 0.0))
                model.stats.m2 = float(stats.get("m2", 0.0))
                model.observations = int(entry.get("observations", 0))
                raw_history = entry.get("raw_history") or []
                model.raw_history.clear()
                for x in raw_history:
                    try:
                        model.raw_history.append(float(x))
                    except Exception:
                        continue

                value_history = entry.get("value_history") or []
                model.value_history.clear()
                for x in value_history:
                    try:
                        model.value_history.append(float(x))
                    except Exception:
                        continue
                ewma_value = entry.get("ewma_value")
                if ewma_value is not None:
                    model._ewma_value = float(ewma_value)
                predicted_next_value = entry.get("predicted_next_value")
                if predicted_next_value is not None:
                    try:
                        model._predicted_next_value = float(predicted_next_value)
                    except Exception:
                        pass
                episode_active = entry.get("episode_active")
                if isinstance(episode_active, dict):
                    model._episode_active = episode_active
                try:
                    model._episode_cooldown = int(entry.get("episode_cooldown", 0))
                except Exception:
                    model._episode_cooldown = 0
            except Exception:
                continue

            service._models[key] = model

        if service._models:
            service._trained = True
        service._htm_core_streams = sum(1 for model in service._models.values() if model.uses_htm_core)
        return service

    def save(self, path: str) -> None:
        import pickle

        state = self._to_state()
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def save_full(self, path: str) -> None:
        """
        Save the full service object using pickle (includes HTM/SP/TM internal state).

        Notes:
        - This format is not guaranteed to be portable across Python/htm.core versions.
        - Only load files you trust (pickle executes code during load).
        """
        import pickle

        import sys
        runtime = {
            "python_version": sys.version.split()[0],
        }
        try:
            import htm  # noqa: F401

            runtime["htm_version"] = getattr(htm, "__version__", None)
        except Exception:
            runtime["htm_version"] = None

        payload = {"format": self.FULL_FORMAT, "service": self, "runtime": runtime}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> AnomalyDetectorService:
        import pickle

        class _HTMPlaceholder:
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
                pass

            def __setstate__(self, state) -> None:  # noqa: ANN001
                # Keep placeholder lightweight; state is intentionally ignored.
                self._state = state

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):  # noqa: ANN001
                if module == "htm" or module.startswith("htm."):
                    return _HTMPlaceholder
                return super().find_class(module, name)

        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except ModuleNotFoundError as exc:
                # Backwards-compat: allow loading older pickles saved with htm.core enabled
                # even if `htm` isn't installed. HTM internals are dropped.
                if (exc.name or "").split(".")[0] != "htm":
                    raise
                f.seek(0)
                obj = _CompatUnpickler(f).load()

        if isinstance(obj, dict) and obj.get("format") == AnomalyDetectorService.STATE_FORMAT:
            return AnomalyDetectorService._from_state(obj)

        if isinstance(obj, dict) and obj.get("format") == AnomalyDetectorService.FULL_FORMAT:
            service = obj.get("service")
            if not isinstance(service, AnomalyDetectorService):
                raise ValueError("Invalid full model payload")
            # If loaded under a placeholder HTM environment, normalize to portable state.
            return AnomalyDetectorService._from_state(service._to_state())

        # Legacy: older pickles may have stored the full service object.
        if isinstance(obj, AnomalyDetectorService):
            # Convert to portable state so the loaded instance doesn't retain placeholder HTM internals.
            return AnomalyDetectorService._from_state(obj._to_state())

        raise ValueError("Unrecognized saved model format")

    @staticmethod
    def load_full(path: str) -> AnomalyDetectorService:
        """
        Load a model saved by `save_full`.

        Requires the same/similar runtime environment (including htm.core availability)
        used when the model was saved. If HTM modules are missing, this will fail.
        """
        import pickle

        try:
            import htm  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                f"htm.core runtime is required to load full models ({exc}). "
                "Start the server with: .venv/bin/python api_server.py"
            ) from exc

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and obj.get("format") == AnomalyDetectorService.FULL_FORMAT:
            service = obj.get("service")
            if not isinstance(service, AnomalyDetectorService):
                raise ValueError("Invalid full model payload")
            return service

        if isinstance(obj, AnomalyDetectorService):
            return obj

        raise ValueError("Unrecognized full model format")
