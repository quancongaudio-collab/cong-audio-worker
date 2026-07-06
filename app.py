"""
Công Audio — Python Worker Flask API
Chạy trên Render.com Standard ($25/tháng · 2GB RAM)
"""

import os
import traceback
import logging
from flask import Flask, request, jsonify

# Setup logging đầy đủ
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
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
            app.logger.warning("Unauthorized request")
            return jsonify({"error": "unauthorized"}), 401

        data = request.get_json()
        app.logger.info(f"Request data: {data}")

        if not data or "file_id" not in data:
            return jsonify({"error": "missing file_id"}), 400

        app.logger.info("Importing worker modules...")
        from worker import process_video, get_drive_service, get_db
        app.logger.info("Worker modules imported OK")

        video_info = {
            "file_id":      data["file_id"],
            "file_name":    data.get("file_name", ""),
            "folder_name":  data.get("folder_name", ""),
            "file_path":    data.get("file_path", ""),
            "created_time": data.get("created_time", ""),
            "size_bytes":   data.get("size_bytes", 0),
        }

        app.logger.info(f"Processing video: {video_info['file_name']}")
        drive_service = get_drive_service()
        app.logger.info("Drive service OK")

        conn = get_db()
        app.logger.info("DB connection OK")

        result = process_video(drive_service, conn, video_info)
        conn.close()

        app.logger.info(f"Done: {result}")
        return jsonify(result), 200

    except Exception as e:
        error_detail = traceback.format_exc()
        app.logger.error(f"[ERROR DETAIL]\n{error_detail}")
        print(f"[ERROR DETAIL]\n{error_detail}", flush=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "detail": error_detail
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
