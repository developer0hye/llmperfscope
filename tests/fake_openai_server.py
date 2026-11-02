"""An OpenAI-compatible server with exactly known timing. usage: fake_openai_server.py PORT READY_DELAY TTFT INTER_TOKEN"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
READY_DELAY = float(sys.argv[2])
TTFT = float(sys.argv[3])
INTER_TOKEN = float(sys.argv[4])
STARTED = time.monotonic()
MODELS = os.environ.get("FAKE_SERVER_MODELS", "fake-model").split(",")
BODIES = os.environ.get("FAKE_SERVER_BODIES")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() - STARTED < READY_DELAY:
                self._json(503, {"status": "loading"})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": model} for model in MODELS]})
        elif self.path == "/metrics":
            data = b"# HELP fake gauge\nfake_kv_cache_usage_perc 0.1\nfake_requests_total 3\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if BODIES:
            with open(BODIES, "a") as f:
                f.write(json.dumps(body) + "\n")
        count = body.get("max_tokens", 16)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload):
            data = f"data: {payload}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        time.sleep(TTFT)
        for i in range(count):
            if i:
                time.sleep(INTER_TOKEN)
            if "messages" in body:
                choice = {"delta": {"content": "x"}, "index": 0}
            else:
                choice = {"text": "x", "index": 0}
            chunk(json.dumps({"choices": [choice]}))
        chunk(json.dumps({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": count}}))
        chunk("[DONE]")
        self.wfile.write(b"0\r\n\r\n")


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
