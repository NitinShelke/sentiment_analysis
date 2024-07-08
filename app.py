from flask import Flask, Response
from prometheus_client import Counter, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Create a registry to register metrics
registry = CollectorRegistry()

# Create a counter metric
REQUEST_COUNT = Counter('request_count', 'App Request Count', registry=registry)

@app.route('/')
def index():
    REQUEST_COUNT.inc()
    return "Hello, World!"

@app.route('/metrics')
def metrics():
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
