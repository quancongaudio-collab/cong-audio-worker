"""
Công Audio — Python Worker Flask API
Chạy trên Render.com Standard ($25/tháng · 2GB RAM)
"""
import os
import json as json_lib
import traceback
import logging
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

WORKER_SECRET = os.environ.get("WORKER_SECRET", "")


def verify_secret(req):
    auth = req.headers.get("X-Worker-Secret", "")
    if not WORKER_SECRET:
        return True
    return auth == WORKER_SECRET


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/process", methods=["POST"])
def process():
    try:
        app.logger.info("=== /process called ===")
        if not verify_secret(request):
