"""
Python Worker — Flask API wrapper
Chạy trên Render.com Starter ($7/tháng · 2GB RAM)
n8n gọi vào POST /process với file_id và thông tin video
"""

import os
import hmac
import hashlib
from flask import Flask, request, jsonify
from worker import process_video, get_drive_service, get_db

app = Flask(__name__)

WORKER_SECRET = os.environ["WORKER_SECRET"]


def verify_secret(req):
    """Xác thực request từ n8n — tránh ai khác gọi vào Worker."""
    auth = req.headers.get("X-Worker-Secret", "")
    return hmac.compare_digest(auth, WORKER_SECRET)


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
        "file_id":     data["file_id"],
        "file_name":   data["file_name"],
        "folder_name": data["folder_name"],
        "file_path":   data["file_path"],
        "created_time": data.get("created_time", ""),
        "size_bytes":  data.get("size_bytes", 0),
    }

    try:
        drive_service = get_drive_service()
        conn          = get_db()
        result        = process_video(drive_service, conn, video_info)
        conn.close()
        return jsonify(result), 200
    except Exception as e:
    import traceback
    error_detail = traceback.format_exc()
    print(f"[ERROR DETAIL] {error_detail}")
    return jsonify({"status": "error", "error": str(e), "detail": error_detail}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
