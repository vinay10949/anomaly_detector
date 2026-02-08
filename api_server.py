import http.server
import socketserver
import json
from detector.service import AnomalyDetectorService, DetectorConfig

# Global detector instance and configuration
current_config = None
detector = None
generator_instance = None
MODEL_PATH = None

def _model_path():
    global MODEL_PATH
    if MODEL_PATH is not None:
        return MODEL_PATH
    import os
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
    return MODEL_PATH

def initialize_detector(config_dict=None):
    """Initialize or reinitialize the detector with given configuration."""
    global detector, current_config
    
    if config_dict:
        config = DetectorConfig(
            warmup_points=config_dict.get('warmup_points', 100),
            raw_threshold=config_dict.get('raw_threshold', 0.80),
            likelihood_threshold=config_dict.get('likelihood_threshold', 0.995),
            likelihood_window=config_dict.get('likelihood_window', 256),
            htm_params=config_dict.get('htm_params'),
            use_temporal_features=bool(config_dict.get('use_temporal_features', True)),
            require_htm=bool(config_dict.get('require_htm', True)),
            scoring=config_dict.get("scoring"),
            learning=config_dict.get("learning"),
            episode=config_dict.get("episode"),
        )
        detector = AnomalyDetectorService(config)
        current_config = {
            "warmup_points": detector.config.warmup_points,
            "raw_threshold": detector.config.raw_threshold,
            "likelihood_threshold": detector.config.likelihood_threshold,
            "likelihood_window": detector.config.likelihood_window,
            "htm_params": detector.config.htm_params,
            "use_temporal_features": detector.config.use_temporal_features,
            "require_htm": detector.config.require_htm,
            "scoring": detector.config.scoring,
            "learning": detector.config.learning,
            "episode": detector.config.episode,
        }
    else:
        current_config = None
        detector = AnomalyDetectorService(DetectorConfig(require_htm=True))

# Initialize with defaults
initialize_detector()

class APIHandler(http.server.BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _read_json(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            return {}
        post_data = self.rfile.read(content_length)
        return json.loads(post_data.decode('utf-8'))

    def _json_response(self, status_code, payload):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def _text_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/plain')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(message.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        global detector
        try:
            data = self._read_json()
        except Exception as exc:
            self._text_response(400, f"Invalid JSON: {exc}")
            return

        try:
            if self.path == '/configure':
                # Configure detector parameters
                htm_params = data.get("htm_params")
                if htm_params is None and isinstance(data.get("htm"), dict):
                    htm_params = data.get("htm")
                if htm_params is None:
                    blocks = {}
                    for k in ("enc", "sp", "tm", "predictor", "anomaly"):
                        if k in data and isinstance(data.get(k), dict):
                            blocks[k] = data.get(k)
                    htm_params = blocks or {}

                config_dict = {
                    'warmup_points': data.get('warmup_points', 100),
                    'raw_threshold': data.get('raw_threshold', 0.80),
                    'likelihood_threshold': data.get('likelihood_threshold', 0.995),
                    'likelihood_window': data.get('likelihood_window', 256),
                    'htm_params': htm_params,
                    'use_temporal_features': data.get('use_temporal_features', True),
                    'require_htm': data.get('require_htm', True),
                    'scoring': data.get("scoring"),
                    'learning': data.get("learning"),
                    'episode': data.get("episode"),
                }
                
                initialize_detector(config_dict)
                
                response = {
                    'status': 'success',
                    'message': 'Detector configured successfully',
                    'config': current_config
                }
                self._json_response(200, response)
                return

            if self.path == '/train':
                training_data = data.get('data', [])
                if not training_data:
                    self._text_response(400, "No training data provided")
                    return

                reference_metadata = data.get('reference_metadata')
                log_progress = bool(data.get("log_progress", True))
                log_every = int(data.get("log_every", 1000))
                log_fn = (lambda msg: print(msg, flush=True)) if log_progress else None
                checkpoint = data.get("checkpoint") or {}
                checkpoint_every = checkpoint.get("every") if isinstance(checkpoint, dict) else None
                checkpoint_keep = checkpoint.get("keep", 3) if isinstance(checkpoint, dict) else 3
                checkpoint_dir = checkpoint.get("dir") if isinstance(checkpoint, dict) else None
                checkpoint_prefix = checkpoint.get("prefix", "model.checkpoint") if isinstance(checkpoint, dict) else "model.checkpoint"
                summary = detector.train(
                    training_data,
                    reference_metadata=reference_metadata,
                    log_every=log_every,
                    log_fn=log_fn,
                    checkpoint_every=checkpoint_every,
                    checkpoint_keep=checkpoint_keep,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=checkpoint_prefix,
                )
                response = {
                    'status': 'success',
                    'message': 'Model trained successfully',
                    **summary,
                }
                self._json_response(200, response)
                return

            if self.path == '/detect':
                if not detector.is_trained:
                    self._text_response(400, "Model not trained")
                    return

                data_point = data.get('point')
                if not data_point:
                    self._text_response(400, "No data point provided")
                    return
                
                learn_requested = bool(data.get('learn', False))
                result = detector.detect(data_point, learn=learn_requested)
                print("Result ", result)
                response = {
                    'timestamp': result['timestamp'],
                    'entity_id': result['entity_id'],
                    'signal_type': result['signal_type'],
                    'metric': result['metric'],
                    'anomaly_flag': result['anomaly_flag'],
                    'score': result['score'],
                    'score_likelihood': result['score_likelihood'],
                    'confidence': result['confidence'],
                    'explanation': result['explanation'],
                    'stream_key': result['stream_key'],
                    'learn_applied': result.get('learn_applied', False),
                    'scores': result.get('scores'),
                    'episode_event': result.get('episode_event'),
                }
                
                # Log detection result
                print(f"[DETECT] Time: {result['timestamp']} | Value: {data_point.get('value')} | Anomaly: {result['anomaly_flag']} | Score: {result['score']:.4f}")
                
                self._json_response(200, response)
                return

            if self.path == '/detect_batch':
                if not detector.is_trained:
                    self._text_response(400, "Model not trained")
                    return

                points = data.get('points', [])
                if not points:
                    self._text_response(400, "No points provided")
                    return

                return_scores = bool(data.get('return_scores', True))
                learn = bool(data.get('learn', False))
                reset_sequence = bool(data.get('reset_sequence', True))
                finalize_episodes = bool(data.get("finalize_episodes", True))
                batch_warmup_points = data.get('batch_warmup_points')
                if batch_warmup_points is not None:
                    try:
                        batch_warmup_points = int(batch_warmup_points)
                    except Exception:
                        self._text_response(400, "batch_warmup_points must be an integer")
                        return
                response = detector.detect_batch(
                    points,
                    return_scores=return_scores,
                    learn=learn,
                    reset_sequence=reset_sequence,
                    batch_warmup_points=batch_warmup_points,
                    finalize_episodes=finalize_episodes,
                )
                self._json_response(200, response)
                return

            if self.path == '/reset':
                # Reset detector to untrained state
                initialize_detector(current_config)
                response = {
                    'status': 'success',
                    'message': 'Detector reset successfully'
                }
                self._json_response(200, response)
                return

            if self.path == '/simulation/start':
                # Initialize simulation generator
                global generator_instance
                try:
                    from simulation.generator import RealTimeGenerator
                    metric = data.get('metric', 'sendmsg')
                    config = data.get('config', {})
                    
                    generator_instance = RealTimeGenerator(
                        metric=metric,
                        entity_id=data.get('entity_id', 'simulated_server'),
                        signal_type=data.get('signal_type', 'network_event'),
                        config_overrides=config
                    )
                    
                    response = {
                        'status': 'success',
                        'message': 'Simulation started',
                        'config': generator_instance.config
                    }
                    self._json_response(200, response)
                except Exception as e:
                    self._text_response(500, f"Failed to start simulation: {e}")
                return

            if self.path == '/simulation/stop':
                # Clear generator instance
                generator_instance = None
                response = {
                    'status': 'success',
                    'message': 'Simulation stopped'
                }
                self._json_response(200, response)
                return

            if self.path == '/model/save':
                try:
                    detector.save(_model_path())
                    print(f"[SAVE] Model saved. Params: {detector.config}")
                    self._json_response(200, {"status": "success", "message": f"Model saved to {_model_path()}"})
                except Exception as e:
                    self._json_response(
                        500,
                        {
                            "status": "error",
                            "message": f"Failed to save model: {str(e)}",
                            "error": {"type": type(e).__name__, "detail": str(e)},
                        },
                    )
                return

            if self.path == '/model/save_full':
                try:
                    full_path = _model_path().replace("model.pkl", "model.full.pkl")
                    detector.save_full(full_path)
                    print(f"[SAVE_FULL] Model saved. Params: {detector.config}")
                    self._json_response(200, {"status": "success", "message": f"Model saved to {full_path}"})
                except Exception as e:
                    self._json_response(
                        500,
                        {
                            "status": "error",
                            "message": f"Failed to save full model: {str(e)}",
                            "error": {"type": type(e).__name__, "detail": str(e)},
                        },
                    )
                return

            if self.path == '/model/load':
                try:
                    import os
                    if not os.path.exists(_model_path()):
                        self._json_response(404, {"status": "error", "message": f"No saved model found ({_model_path()})"})
                        return
                         
                    from detector.service import AnomalyDetectorService
                    detector = AnomalyDetectorService.load(_model_path())
                    
                    print(f"[LOAD] Model loaded. Params: {detector.config}")
                    
                    self._json_response(200, {
                        "status": "success", 
                        "message": "Model loaded from model.pkl",
                        "details": detector.status()
                    })
                except Exception as e:
                    self._json_response(
                        500,
                        {
                            "status": "error",
                            "message": f"Failed to load model: {str(e)}",
                            "error": {"type": type(e).__name__, "detail": str(e)},
                        },
                    )
                return

            if self.path == '/model/load_full':
                try:
                    import os
                    full_path = _model_path().replace("model.pkl", "model.full.pkl")
                    if not os.path.exists(full_path):
                        self._json_response(404, {"status": "error", "message": f"No saved full model found ({full_path})"})
                        return

                    from detector.service import AnomalyDetectorService
                    detector = AnomalyDetectorService.load_full(full_path)

                    print(f"[LOAD_FULL] Model loaded. Params: {detector.config}")

                    self._json_response(200, {
                        "status": "success",
                        "message": f"Model loaded from {full_path}",
                        "details": detector.status()
                    })
                except Exception as e:
                    self._json_response(
                        500,
                        {
                            "status": "error",
                            "message": f"Failed to load full model: {str(e)}",
                            "error": {"type": type(e).__name__, "detail": str(e)},
                        },
                    )
                return

            self._text_response(404, "Endpoint not found")
        except RuntimeError as exc:
            self._text_response(400, str(exc))
        except ValueError as exc:
            self._text_response(400, str(exc))
        except Exception as exc:
            self._text_response(500, str(exc))

    def do_GET(self):
        global detector, generator_instance
        if self.path == '/status':
            status_data = detector.status()
            # Add current configuration to status
            status_data['current_config'] = current_config
            status_data['simulation_active'] = generator_instance is not None
            self._json_response(200, status_data)
            return

        if self.path in ('/simulation/next', '/simulation/generate'):
            if generator_instance is None:
                self._text_response(400, "Simulation not active")
                return
            
            # Generate next point
            import time
            timestamp = int(time.time())
            point = generator_instance.generate_next(timestamp)
            
            # Return point only (client handles detection)
            response = {
                'point': point
            }
            self._json_response(200, response)
            return

        self._text_response(404, "Endpoint not found")

def run_server(port=8001):
    class ThreadingServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with ThreadingServer(("", port), APIHandler) as httpd:
        print(f"API Server running on port {port}")
        try:
            import htm  # noqa: F401
        except Exception as exc:
            print(f"[WARN] HTM not available in this Python ({exc}).")
            print("[WARN] For HTM learning/prediction, run with: .venv/bin/python api_server.py")
        print(f"Available endpoints:")
        print(f"  POST /configure - Configure detector parameters")
        print(f"  POST /train - Train the detector")
        print(f"  POST /detect - Detect single point")
        print(f"  POST /detect_batch - Detect batch of points")
        print(f"  POST /reset - Reset detector to untrained state")
        print(f"  GET /status - Get detector status")
        print(f"  POST /model/save - Save portable model (no HTM state)")
        print(f"  POST /model/load - Load portable model (best-effort)")
        print(f"  POST /model/save_full - Save full model (includes HTM state)")
        print(f"  POST /model/load_full - Load full model (requires HTM runtime)")
        print(f"  POST /simulation/start - Start live simulation")
        print(f"  POST /simulation/stop - Stop live simulation")
        print(f"  GET /simulation/next - Get next simulation point")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
