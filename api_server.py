import http.server
import socketserver
import json
import urllib.parse
from anomaly_detector import AnomalyDetector

# Global detector instance
detector = None

class APIHandler(http.server.BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8000')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        global detector

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if self.path == '/train':
            try:
                # Expect data as list of points
                training_data = data.get('data', [])
                if not training_data:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/plain')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"No training data provided")
                    return

                detector = AnomalyDetector()
                detector.train(training_data)

                response = {'status': 'success', 'message': 'Model trained successfully'}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_error(500, str(e))

        elif self.path == '/detect':
            try:
                if not detector:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/plain')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"Model not trained")
                    return

                # Expect single data point
                data_point = data.get('point')
                if not data_point:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/plain')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"No data point provided")
                    return

                result = detector.detect_anomaly(data_point)

                response = {
                    'anomaly_flag': result['anomaly_flag'],
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'explanation': result['explanation']
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_error(500, str(e))

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    def do_GET(self):
        if self.path == '/status':
            status = 'trained' if detector and detector.trained else 'not_trained'
            response = {'status': status}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

def run_server(port=5000):
    with socketserver.TCPServer(("", port), APIHandler) as httpd:
        print(f"API Server running on port {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server(8001)