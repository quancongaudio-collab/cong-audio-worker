"""
Công Audio — Python Worker Flask API
Chạy trên Render.com Standard ($25/tháng · 2GB RAM)
"""

import os
import traceback
from flask import Flask, request, jsonify
from worker import process_video, get_drive_service, get_db

app = Flask(__name__)

import logging
logging.basicConfig(level=logging.DEBUG)

WORKER_SECRET = os.environ.get("WORKER_SECRET", "")


def verify_secret(req):
    auth = req.headers.get("X-Worker-Secret", "")
    if not WORKER_SECRET:
        return True  # Nếu không set secret thì bỏ qua verify
    return auth == WORKER_SECRET


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/process", methods=["POST"])
def process():
    if not verify_secret(request):
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json()
    if not data or "file_id" not in data:
        return jsonify({"error": "missing file_id"}), 400

    video_info = {
        "file_id":      data["file_id"],
        "file_name":    data.get("file_name", ""),
        "folder_name":  data.get("folder_name", ""),
        "file_path":    data.get("file_path", ""),
        "created_time": data.get("created_time", ""),
        "size_bytes":   data.get("size_bytes", 0),
    }

    try:
        drive_service = get_drive_service()
        conn = get_db()
        result = process_video(drive_service, conn, video_info)
        conn.close()
        return jsonify(result), 200
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[ERROR DETAIL]\n{error_detail}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "detail": error_detail
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
