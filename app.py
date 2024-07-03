from flask import Flask, jsonify
from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
import time

app = Flask(__name__)

# Define Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
ACCURACY = Gauge('model_accuracy', 'Model accuracy')
LOSS = Gauge('model_loss', 'Model loss')

# Simulate model accuracy and loss
current_accuracy = 0.9
current_loss = 0.1

@app.route('/')
def index():
    """Main endpoint to process requests."""
    with REQUEST_TIME.time():
        # Simulate processing time
        time.sleep(1)
        REQUEST_COUNT.inc()

        # Update accuracy and loss
        global current_accuracy, current_loss
        current_accuracy += 0.01
        current_loss -= 0.01

        ACCURACY.set(current_accuracy)
        LOSS.set(current_loss)

        return jsonify({"status": "success"})

@app.route('/metrics')
def metrics():
    """Expose metrics to Prometheus."""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    # Start Prometheus client server on a different port
    start_http_server(8000)
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
