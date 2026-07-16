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
            return jsonify({"error": "unauthorized"}), 401

        data = request.get_json(force=True, silent=True)
        if data is None:
            raw = request.data.decode("utf-8")
            app.logger.info(f"Raw body: {raw[:200]}")
            data = json_lib.loads(raw)
        if isinstance(data, str):
            data = json_lib.loads(data)

        app.logger.info(f"Parsed data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        if not isinstance(data, dict) or "file_id" not in data:
            return jsonify({"error": "missing file_id", "received": str(data)[:200]}), 400

        from worker import process_video, get_drive_service, get_db

        video_info = {
            "file_id":      data["file_id"],
            "file_name":    data.get("file_name", ""),
            "folder_name":  data.get("folder_name", ""),
            "file_path":    data.get("file_path", ""),
            "created_time": data.get("created_time", ""),
            "size_bytes":   data.get("size_bytes", 0),
        }

        app.logger.info(f"Processing: {video_info['file_name']}")

        drive_service = get_drive_service()
        conn = get_db()
        result = process_video(drive_service, conn, video_info)
        conn.close()

        app.logger.info(f"Done: {result}")
        return jsonify(result), 200

    except Exception as e:
        error_detail = traceback.format_exc()
        app.logger.error(f"[ERROR]\n{error_detail}")
        print(f"[ERROR]\n{error_detail}", flush=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "detail": error_detail
        }), 500


@app.route("/rescore", methods=["POST"])
def rescore():
    try:
        app.logger.info("=== /rescore called ===")
        if not verify_secret(request):
            return jsonify({"error": "unauthorized"}), 401

        data = request.get_json(force=True, silent=True)
        if data is None:
            raw = request.data.decode("utf-8")
            app.logger.info(f"Raw body: {raw[:200]}")
            data = json_lib.loads(raw) if raw else {}
        if isinstance(data, str):
            data = json_lib.loads(data) if data else {}
        if not isinstance(data, dict):
            data = {}

        limit = data.get("limit")
        app.logger.info(f"Rescore batch, limit={limit}")

        from worker import run_rescore_batch

        results = run_rescore_batch(limit=limit)

        app.logger.info(f"Rescore batch done: {results}")
        return jsonify(results), 200

    except Exception as e:
        error_detail = traceback.format_exc()
        app.logger.error(f"[ERROR /rescore]\n{error_detail}")
        print(f"[ERROR /rescore]\n{error_detail}", flush=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "detail": error_detail
        }), 500


@app.route("/render-story", methods=["POST"])
def render_story_route():
    app.logger.info("=== /render-story called ===")
    if not verify_secret(request):
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True, silent=True)
    if data is None:
        raw = request.data.decode("utf-8")
        app.logger.info(f"Raw body: {raw[:200]}")
        data = json_lib.loads(raw)
    if isinstance(data, str):
        data = json_lib.loads(data)

    if not isinstance(data, dict) or "drive_file_id" not in data:
        return jsonify({"error": "missing drive_file_id", "received": str(data)[:200]}), 400

    try:
        from render_story import render_story_video
        from worker import get_drive_service

        app.logger.info(f"Rendering story_id={data.get('story_id')}")

        drive_service = get_drive_service()
        result = render_story_video(drive_service, data)

        app.logger.info(f"Render done: {result}")
        return jsonify(result), 200

    except Exception as e:
        error_detail = traceback.format_exc()
        app.logger.error(f"[ERROR /render-story]\n{error_detail}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "detail": error_detail
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
