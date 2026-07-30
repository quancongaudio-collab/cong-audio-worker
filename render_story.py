"""
reindex_transcripts.py — Xử lý lại transcript + điểm cắt cho video CŨ
(Quân CôngAudio)

Mục đích: video cũ đã có transcript (văn bản thô, không mốc giờ) và
start_time/end_time (do GPT-4o Vision đoán gần đúng khi index lần đầu).
Script này CHỈ chạy lại Whisper (lấy mốc giờ thật theo câu) rồi dùng
cut_point.py để chốt lại start_time/end_time chính xác — KHÔNG gọi lại
GPT-4o Vision (đã có đủ summary/reason/... từ lần index đầu, theo đúng
nguyên tắc "Analyze Once – Reuse Everywhere" đã áp dụng trong worker.py).
Nhờ vậy chi phí xử lý lại rẻ hơn nhiều so với index lần đầu.

CÁCH DÙNG:
  python3 reindex_transcripts.py                 # chạy tối đa BATCH_SIZE video 1 lượt
  python3 reindex_transcripts.py --limit 50       # tuỳ chỉnh số video mỗi lượt
  python3 reindex_transcripts.py --dry-run        # chỉ in ra, không ghi DB

AN TOÀN KHI DỪNG GIỮA CHỪNG:
  - Script luôn lọc theo `WHERE transcript_segments IS NULL` — video nào đã
    xử lý xong (dù ở lượt chạy trước) sẽ tự động được bỏ qua ở lượt sau.
  - Chạy lại nhiều lần cho tới khi không còn video nào cần xử lý.
"""

import argparse
import os
import sys
import tempfile
import time

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from worker import (
    get_drive_service, get_db, download_video, extract_audio,
    transcribe_audio, _safe_remove, cleanup_stale_tmp_files,
)
from cut_point import select_cut_window, seconds_to_timestamp, timestamp_to_seconds

BATCH_SIZE_DEFAULT = 30
SLEEP_BETWEEN_FILES_SEC = 2  # tránh dồn dập gọi Groq API


def fetch_files_to_process(conn, limit: int):
    """Lấy danh sách các drive_file_id còn thiếu transcript_segments (video cũ chưa xử lý lại)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT drive_file_id
            FROM videos
            WHERE transcript_segments IS NULL
              AND status = 'Indexed'
            ORDER BY drive_file_id
            LIMIT %s;
        """, (limit,))
        return [row["drive_file_id"] for row in cur.fetchall()]


def fetch_rows_for_file(conn, drive_file_id: str):
    """Lấy toàn bộ dòng (mỗi dòng = 1 đoạn giá trị) hiện có của 1 video gốc."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, start_time, end_time
            FROM videos
            WHERE drive_file_id = %s;
        """, (drive_file_id,))
        return cur.fetchall()


def update_row(conn, video_id: str, start_time: str, end_time: str,
               transcript_segments: list, dry_run: bool):
    if dry_run:
        print(f"        [DRY-RUN] id={video_id} -> start_time={start_time} end_time={end_time}")
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE videos
            SET start_time = %s, end_time = %s, transcript_segments = %s
            WHERE id = %s;
        """, (start_time, end_time, Json(transcript_segments), video_id))
    conn.commit()


def exclude_row(conn, video_id: str, transcript_segments: list, dry_run: bool):
    """Video/đoạn này không tìm được điểm cắt hợp lệ nào -> loại khỏi kho chọn video.
    Dùng selection_status (cơ chế loại trừ có sẵn, n8n đang lọc WHERE
    selection_status IS NULL) thay vì đụng vào status, để không tác động tới
    ràng buộc nào chưa biết trên cột đó. Vẫn lưu transcript_segments để không
    phải chạy lại Whisper cho video này lần nữa."""
    if dry_run:
        print(f"        [DRY-RUN] id={video_id} -> LOẠI BỎ (không cắt được câu trọn vẹn)")
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE videos
            SET selection_status = 'loai_cut_cut_nghia', transcript_segments = %s
            WHERE id = %s;
        """, (Json(transcript_segments), video_id))
    conn.commit()


def process_one_file(drive_service, conn, drive_file_id: str, dry_run: bool) -> str:
    cleanup_stale_tmp_files()
    rows = fetch_rows_for_file(conn, drive_file_id)
    if not rows:
        return "no_rows"

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "video.mp4")
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        try:
            download_video(drive_service, drive_file_id, video_path)
            extract_audio(video_path, audio_path)
        except Exception as e:
            print(f"        [ERROR] Không tải/tách audio được: {e}")
            return "error"
        finally:
            _safe_remove(video_path)

        whisper_result = transcribe_audio(audio_path)
        _safe_remove(audio_path)
        segments = whisper_result["segments"]

        if not segments:
            print(f"        [WARN] Whisper không trả về segment nào (có thể video câm) — bỏ qua.")
            for row in rows:
                exclude_row(conn, row["id"], [], dry_run)
            return "no_speech"

        for row in rows:
            requested_start_sec = timestamp_to_seconds(row["start_time"])
            window = select_cut_window(segments, requested_start_sec)
            if window is None:
                exclude_row(conn, row["id"], segments, dry_run)
                print(f"        [LOẠI] id={row['id']} — không tìm được điểm cắt không cụt câu.")
                continue
            start_sec, end_sec = window
            update_row(
                conn, row["id"],
                seconds_to_timestamp(start_sec),
                seconds_to_timestamp(end_sec),
                segments, dry_run,
            )
        return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=BATCH_SIZE_DEFAULT,
                         help="Số video (theo drive_file_id) xử lý trong lượt chạy này")
    parser.add_argument("--dry-run", action="store_true",
                         help="Chỉ in ra, không ghi vào DB")
    args = parser.parse_args()

    drive_service = get_drive_service()
    conn = get_db()

    files = fetch_files_to_process(conn, args.limit)
    print(f"Tìm thấy {len(files)} video cần xử lý lại (giới hạn {args.limit}/lượt).")
    if not files:
        print("Không còn video nào cần xử lý lại. Xong.")
        conn.close()
        return

    done, errors = 0, 0
    for i, drive_file_id in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {drive_file_id}")
        try:
            result = process_one_file(drive_service, conn, drive_file_id, args.dry_run)
            if result == "error":
                errors += 1
            else:
                done += 1
        except Exception as e:
            import traceback
            print(f"        [ERROR] {e}")
            print(traceback.format_exc())
            errors += 1
        time.sleep(SLEEP_BETWEEN_FILES_SEC)

    print(f"\nHoàn tất lượt này: {done} video xử lý xong, {errors} lỗi.")
    print("Chạy lại script này (không cần tham số gì thêm) để xử lý lô tiếp theo.")
    conn.close()


if __name__ == "__main__":
    main()
