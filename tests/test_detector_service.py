import unittest
import tempfile

try:
    import htm  # noqa: F401
    _HTM_AVAILABLE = True
except Exception:
    _HTM_AVAILABLE = False

from detector.normalization import normalize_points
from detector.service import AnomalyDetectorService, DetectorConfig


@unittest.skipUnless(_HTM_AVAILABLE, "htm.core is required for these tests")
class TestDetectorNormalization(unittest.TestCase):
    def test_normalize_points_dedupes_by_key(self):
        raw = [
            {
                "timestamp": 10,
                "entity_id": "a",
                "signal_type": "sys",
                "metric": "cpu",
                "value": 1.0,
            },
            {
                "timestamp": 10,
                "entity_id": "a",
                "signal_type": "sys",
                "metric": "cpu",
                "value": 9.0,
            },
            {
                "timestamp": 20,
                "entity_id": "a",
                "signal_type": "sys",
                "metric": "cpu",
                "value": 2.0,
            },
        ]

        points = normalize_points(raw)
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["value"], 9.0)
        self.assertEqual(points[1]["value"], 2.0)


@unittest.skipUnless(_HTM_AVAILABLE, "htm.core is required for these tests")
class TestDetectorService(unittest.TestCase):
    def setUp(self):
        self.config = DetectorConfig(
            warmup_points=1,
            raw_threshold=0.6,
            likelihood_threshold=0.95,
            use_temporal_features=False,
            # Keep HTM objects small so save_full/load_full stays fast.
            htm_params={
                "enc_size": 700,
                "enc_sparsity": 0.02,
                "enc_resolution": 0.88,
                "sp_columnDimensions": 512,
                "tm_cellsPerColumn": 4,
                "tm_activationThreshold": 10,
                "tm_initialPerm": 0.21,
                "tm_permanenceInc": 0.1,
                "tm_permanenceDec": 0.1,
                "predictor": {"steps": [1], "sdrc_alpha": 0.1, "resolution": 1.0},
                "anomaly": {"period": 128},
            },
        )
        self.data = [
            {
                "timestamp": i,
                "entity_id": "machine-1",
                "signal_type": "system",
                "metric": "cpu",
                "value": 10.0,
            }
            for i in range(20)
        ]
        self.data.append(
            {
                "timestamp": 21,
                "entity_id": "machine-1",
                "signal_type": "system",
                "metric": "cpu",
                "value": 200.0,
            }
        )

    def test_train_and_status(self):
        svc = AnomalyDetectorService(self.config)
        summary = svc.train(self.data)
        self.assertTrue(svc.is_trained)
        self.assertEqual(summary["trained_streams"], 1)
        self.assertGreater(summary["points_used"], 0)

        status = svc.status()
        self.assertEqual(status["status"], "trained")
        self.assertEqual(status["trained_streams"], 1)

    def test_detect_batch_is_deterministic_after_retrain(self):
        svc_a = AnomalyDetectorService(self.config)
        svc_b = AnomalyDetectorService(self.config)
        svc_a.train(self.data)
        svc_b.train(self.data)

        out_a = svc_a.detect_batch(self.data, return_scores=True)
        out_b = svc_b.detect_batch(self.data, return_scores=True)

        self.assertEqual(out_a["summary"]["total"], len(self.data))
        self.assertEqual(out_a["summary"]["detected_count"], out_b["summary"]["detected_count"])
        self.assertEqual(out_a["results"], out_b["results"])

    def test_save_full_and_load_full_roundtrip(self):
        svc = AnomalyDetectorService(self.config)
        svc.train(self.data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.full.pkl"
            svc.save_full(path)
            loaded = AnomalyDetectorService.load_full(path)

        self.assertTrue(loaded.is_trained)
        out_a = svc.detect_batch(self.data, return_scores=True)
        out_b = loaded.detect_batch(self.data, return_scores=True)
        self.assertEqual(out_a["summary"]["detected_count"], out_b["summary"]["detected_count"])

    def test_collective_episode_detection_emits_episode(self):
        cfg = DetectorConfig(
            warmup_points=1,
            raw_threshold=0.95,
            likelihood_threshold=0.999,
            use_temporal_features=False,
            episode={"entry_score": 0.45, "trigger_points": 3, "continue_score": 0.25, "exit_points": 2},
            htm_params={
                "enc_size": 700,
                "enc_sparsity": 0.02,
                "enc_resolution": 1.0,
                "sp_columnDimensions": 512,
                "tm_cellsPerColumn": 4,
                "tm_activationThreshold": 10,
                "tm_initialPerm": 0.21,
                "tm_permanenceInc": 0.1,
                "tm_permanenceDec": 0.1,
                "predictor": {"steps": [1], "sdrc_alpha": 0.1, "resolution": 1.0},
                "anomaly": {"period": 64},
            },
        )
        svc = AnomalyDetectorService(cfg)
        baseline = [
            {
                "timestamp": i,
                "entity_id": "machine-1",
                "signal_type": "system",
                "metric": "cpu",
                "value": 100.0,
            }
            for i in range(120)
        ]
        svc.train(baseline)

        shifted = [
            {
                "timestamp": 1000 + i,
                "entity_id": "machine-1",
                "signal_type": "system",
                "metric": "cpu",
                "value": 160.0 if 10 <= i < 40 else 100.0,
            }
            for i in range(60)
        ]

        out = svc.detect_batch(
            shifted,
            return_scores=True,
            mode='predict_only',
            reset_sequence=True,
            batch_warmup_points=0,
            finalize_episodes=True,
        )
        episodes = out.get("episodes") or []
        self.assertTrue(len(episodes) >= 1)
        ep = episodes[0]
        self.assertIn("start_timestamp", ep)
        self.assertIn("end_timestamp", ep)
        self.assertGreaterEqual(int(ep.get("end_timestamp", 0)), int(ep.get("start_timestamp", 0)))

    def test_contextual_anomaly_triggers_via_predictor_residual(self):
        cfg = DetectorConfig(
            warmup_points=1,
            raw_threshold=0.6,
            likelihood_threshold=0.999,
            use_temporal_features=False,
            scoring={"weights": {"contextual": 1.0, "tm": 0.0, "point": 0.0, "collective": 0.0}},
            htm_params={
                "enc_size": 700,
                "enc_sparsity": 0.02,
                "enc_resolution": 1.0,
                "sp_columnDimensions": 512,
                "tm_cellsPerColumn": 4,
                "tm_activationThreshold": 10,
                "tm_initialPerm": 0.21,
                "tm_permanenceInc": 0.1,
                "tm_permanenceDec": 0.1,
                "predictor": {"steps": [1], "sdrc_alpha": 0.1, "resolution": 1.0},
                "anomaly": {"period": 64},
            },
        )
        svc = AnomalyDetectorService(cfg)
        training = [
            {
                "timestamp": i,
                "entity_id": "machine-1",
                "signal_type": "system",
                "metric": "cpu",
                "value": 90.0 if i % 2 == 0 else 110.0,
            }
            for i in range(200)
        ]
        svc.train(training)

        # After ending on 110, the learned pattern should predict 90 next; provide 110 instead.
        point = {
            "timestamp": 200,
            "entity_id": "machine-1",
            "signal_type": "system",
            "metric": "cpu",
            "value": 110.0,
        }
        res = svc.detect(point, mode='predict_only')
        self.assertTrue(res["anomaly_flag"])
        scores = res.get("scores") or {}
        self.assertGreater(float(scores.get("contextual") or 0.0), 0.5)


if __name__ == "__main__":
    unittest.main()
