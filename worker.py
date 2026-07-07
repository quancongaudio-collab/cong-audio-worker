"""
Công Audio — Video Indexing Worker V1.0
Phương án B: Gemini 2.0 Flash + Whisper (Groq) + FFmpeg
Chi phí ước tính: ~$0.04 / video
"""

import os
import json
import uuid
import subprocess
import tempfile
import base64
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import psycopg2
from psycopg2.extras import Json, execute_values
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io

# ─────────────────────────────────────────────
# CONFIG — đọc từ environment variables
# ─────────────────────────────────────────────
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DATABASE_URL     = os.environ["DATABASE_URL"]
DRIVE_CREDS_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT"]
DRIVE_ROOT_ID    = os.environ.get("DRIVE_ROOT_FOLDER_ID", "")

FRAMES_PER_SEC   = 1
MAX_FRAMES_BATCH = 5
DAILY_VIDEO_LIMIT = 50

GEMINI_VISION_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY
)
EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

VALID_SERIES = [
    "Sai lầm Audiophile", "Một phút Hi-end", "Bắt bệnh hệ thống",
    "Case Study", "Setup thực tế", "Điện Audio", "Phòng nghe",
    "Dây dẫn", "Nguồn phát", "Loa", "Ampli", "Digital", "Analog",
    "Streaming", "Vinyl", "CD", "Luxury Lifestyle", "Behind the scenes",
    "Một ngày tại Công Audio", "Đập tan hiểu lầm", "Hỏi đáp",
    "Nhật ký kỹ thuật", "Một sản phẩm – Một câu chuyện",
    "Triển lãm Audio", "Lịch sử Audio", "Thương hiệu", "So sánh",
    "Tư duy Audiophile", "Triết lý sống", "Khoảnh khắc đáng nhớ",
]

# ─────────────────────────────────────────────
# 1. GOOGLE DRIVE
# ─────────────────────────────────────────────

def get_drive_service():
    creds_dict = json.loads(DRIVE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def download_video(service, file_id: str, dest_path: str):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


# ─────────────────────────────────────────────
# 2. FFMPEG
# ─────────────────────────────────────────────

def extract_tech_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    duration_sec = float(info.get("format", {}).get("duration", 0))
    h = int(duration_sec // 3600)
    m = int((duration_sec % 3600) // 60)
    s = int(duration_sec % 60)
    duration_str = f"{h:02d}:{m:02d}:{s:02d}"

    orientation = "Horizontal (16:9)"
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            w = stream.get("width", 1920)
            h_px = stream.get("height", 1080)
            ratio = w / h_px if h_px else 1
            if ratio < 0.8:
                orientation = "Vertical (9:16)"
            elif 0.8 <= ratio <= 1.2:
                orientation = "Square (1:1)"
            else:
                orientation = "Horizontal (16:9)"
            break

    tags = info.get("format", {}).get("tags", {})
    rec_date = (
        tags.get("creation_time", "")
        or tags.get("date", "")
        or tags.get("com.apple.quicktime.creationdate", "")
    )
    if rec_date:
        rec_date = rec_date[:10]
    else:
        rec_date = ""

    return {
        "duration":       duration_str,
        "duration_sec":   duration_sec,
        "orientation":    orientation,
        "recording_date": rec_date,
    }


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list:
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",
        "-s", "1280x720",
        frame_pattern,
        "-y", "-loglevel", "error"
    ], check=True)
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def extract_audio(video_path: str, output_path: str):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1",
        output_path,
        "-y", "-loglevel", "error"
    ], check=True)


# ─────────────────────────────────────────────
# 3. WHISPER VIA GROQ
# ─────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={
                    "model":    "whisper-large-v3",
                    "language": "vi",
                    "response_format": "text",
                },
                timeout=120,
            )
        if resp.status_code == 200:
            return resp.text.strip()
        print(f"[WARN] Whisper error: {resp.status_code} {resp.text[:200]}")
        return ""
    except Exception as e:
        print(f"[WARN] Whisper exception: {e}")
        return ""


# ─────────────────────────────────────────────
# 4. GEMINI 2.0 FLASH — có retry khi rate limit
# ─────────────────────────────────────────────

GEMINI_PROMPT_TEMPLATE = """
Bạn là chuyên gia phân tích nội dung video cho Công Audio — cửa hàng audio hi-end tại Việt Nam.

=== THÔNG TIN KỸ THUẬT ĐÃ BIẾT ===
- File name: {file_name}
- Thư mục: {folder_name}
- Thời lượng: {duration}
- Ngày quay: {recording_date}

=== TRANSCRIPT ÂM THANH (Whisper STT) ===
{transcript}

=== NHIỆM VỤ ===
Phân tích toàn bộ {frame_count} frame bên dưới và transcript, rồi trả về DUY NHẤT một JSON hợp lệ.
Không thêm bất kỳ text nào ngoài JSON. Không dùng markdown code block.

=== DANH SÁCH 30 SERIES HỢP LỆ ===
{series_list}

=== CẤU TRÚC JSON CẦN TRẢ VỀ ===
{{
  "summary": "Tóm tắt nội dung video trong 1-2 câu tiếng Việt.",
  "scenes": [
    {{
      "start": "00:00",
      "end":   "00:18",
      "description": "Mô tả ngắn cảnh này tiếng Việt."
    }}
  ],
  "keywords": ["từ khóa 1", "từ khóa 2", "từ khóa 3"],
  "series": "Chọn ĐÚNG MỘT tên series trong danh sách 30 series ở trên.",
  "topic": "Chủ đề chính (Loa / Ampli / DAC / Phòng nghe / Streaming / Vinyl / Điện Audio / Dây dẫn / Lifestyle...)",
  "core_message": "Thông điệp quan trọng nhất của video. Chỉ một câu.",
  "knowledge": "Tri thức kỹ thuật hoặc audiophile rút ra từ video. Một câu cụ thể.",
  "hook": "Câu mở đầu hấp dẫn nếu có, hoặc chuỗi rỗng nếu không có.",
  "highlight_scenes": [
    {{
      "timestamp": "00:00–00:18",
      "reason": "Lý do cảnh này đáng tái sử dụng."
    }}
  ],
  "brands": ["Tên thương hiệu xuất hiện rõ ràng trong video"],
  "products": ["Loại thiết bị (Loa, DAC, Power Amp, Pre Amp, Streamer, Turntable, Cable...)"],
  "people": ["Quân / Khách hàng / Kỹ thuật viên / Không có người — chỉ ghi những người thấy rõ"],
  "target_audience": "Audiophile mới / Người đang nâng cấp / Chủ biệt thự / Người làm phòng nghe / Audiophile kỳ cựu",
  "copyright_status": "Công Audio sở hữu / Được khách cho phép sử dụng / Video hãng cung cấp / Chỉ dùng nội bộ / Không được đăng công khai",
  "shoot_location": "Showroom 1 / Showroom 2 / Nhà khách / hoặc địa điểm nhận ra từ video"
}}

=== QUY TẮC QUAN TRỌNG ===
1. Chỉ trả về JSON. Không giải thích, không markdown.
2. Chỉ ghi nhận những gì THẤY RÕ trong frame hoặc NGHE RÕ trong transcript.
3. Series phải là MỘT trong 30 tên chính xác ở trên.
4. copyright_status mặc định là "Công Audio sở hữu" nếu không có dấu hiệu khác.
5. Timestamp scene theo định dạng MM:SS hoặc HH:MM:SS tùy thời lượng video.
"""


def encode_frame_to_base64(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini_vision(frames: list, prompt: str) -> dict:
    """
    Gọi Gemini 2.0 Flash với retry khi bị rate limit 429.
    Thử tối đa 5 lần, chờ 60 giây mỗi lần.
    """
    if len(frames) > MAX_FRAMES_BATCH:
        step = len(frames) / MAX_FRAMES_BATCH
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES_BATCH)]

    parts = [{"text": prompt}]
    for frame_path in frames:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encode_frame_to_base64(frame_path),
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature":      0.2,
            "maxOutputTokens":  2048,
            "responseMimeType": "application/json",
        },
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_VISION_URL, json=payload, timeout=120)

            if resp.status_code == 429:
                wait_sec = 60 * (attempt + 1)  # 60, 120, 180, 240, 300 giây
                print(f"[WARN] Gemini rate limit 429 — chờ {wait_sec}s (lần {attempt+1}/{max_retries})")
                time.sleep(wait_sec)
                continue

            resp.raise_for_status()
            raw = resp.json()
            text = raw["candidates"][0]["content"]["parts"][0]["text"]

            tokens_used   = raw.get("usageMetadata", {})
            input_tokens  = tokens_used.get("promptTokenCount", 0)
            output_tokens = tokens_used.get("candidatesTokenCount", 0)
            cost_usd = (input_tokens * 0.10 + output_tokens * 0.40) / 1_000_000

            return {"text": text, "cost_usd": cost_usd}

        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Gemini HTTP error: {e} — thử lại sau 60s")
                time.sleep(60)
            else:
                raise

    raise Exception("Gemini API vẫn lỗi sau 5 lần thử")


def parse_gemini_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error: {e}. Raw: {text[:500]}")
        data = {}

    if data.get("series") not in VALID_SERIES:
        data["series"] = "Khoảnh khắc đáng nhớ"
        data["series_needs_review"] = True

    return data


# ─────────────────────────────────────────────
# 5. EMBEDDING
# ─────────────────────────────────────────────

def create_embedding(text: str) -> Optional[list]:
    if not OPENAI_API_KEY or not text.strip():
        return None
    try:
        resp = requests.post(
            EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "text-embedding-3-small", "input": text[:8000]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] Embedding error: {e}")
        return None


# ─────────────────────────────────────────────
# 6. DATABASE
# ─────────────────────────────────────────────

def get_db():
    return psycopg2.connect(DATABASE_URL)


def is_already_processed(conn, drive_file_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM processing_log WHERE drive_file_id = %s AND status = 'done'",
            (drive_file_id,)
        )
        return cur.fetchone() is not None


def mark_processing(conn, drive_file_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO processing_log (drive_file_id, status, started_at)
            VALUES (%s, 'processing', NOW())
            ON CONFLICT (drive_file_id) DO UPDATE SET status='processing', started_at=NOW()
        """, (drive_file_id,))
    conn.commit()


def mark_done(conn, drive_file_id: str, video_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE processing_log
            SET status='done', video_id=%s, done_at=NOW()
            WHERE drive_file_id=%s
        """, (video_id, drive_file_id))
    conn.commit()


def mark_error(conn, drive_file_id: str, error: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO processing_log (drive_file_id, status, error_message, started_at)
            VALUES (%s, 'error', %s, NOW())
            ON CONFLICT (drive_file_id) DO UPDATE
            SET status='error', error_message=%s, done_at=NOW()
        """, (drive_file_id, error[:1000], error[:1000]))
    conn.commit()


def save_video_metadata(conn, meta: dict) -> str:
    video_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO videos (
                id, drive_file_id, file_name, folder, file_path,
                status, duration, orientation, recording_date, shoot_location,
                summary, keywords, series, topic, core_message,
                knowledge, hook, brands, products, people,
                target_audience, copyright_status, transcript,
                processing_cost_usd, extra_meta, created_at
            ) VALUES (
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,NOW()
            )
        """, (
            video_id,
            meta["drive_file_id"],
            meta["file_name"],
            meta["folder"],
            meta["file_path"],
            "Indexed",
            meta["duration"],
            meta["orientation"],
            meta["recording_date"],
            meta.get("shoot_location", ""),
            meta.get("summary", ""),
            meta.get("keywords", []),
            meta.get("series", ""),
            meta.get("topic", ""),
            meta.get("core_message", ""),
            meta.get("knowledge", ""),
            meta.get("hook", ""),
            meta.get("brands", []),
            meta.get("products", []),
            meta.get("people", []),
            meta.get("target_audience", ""),
            meta.get("copyright_status", "Công Audio sở hữu"),
            meta.get("transcript", ""),
            meta.get("processing_cost_usd", 0),
            Json(meta.get("extra_meta", {})),
        ))
    conn.commit()
    return video_id


def save_scenes(conn, video_id: str, scenes: list, highlight_scenes: list):
    if not scenes:
        return

    rows = []
    for i, scene in enumerate(scenes):
        embed_text = scene.get("description", "")
        embedding  = create_embedding(embed_text)

        highlight_timestamps = {h.get("timestamp", "") for h in highlight_scenes}
        ts = f"{scene.get('start', '')}–{scene.get('end', '')}"
        is_highlight = ts in highlight_timestamps

        rows.append((
            str(uuid.uuid4()),
            video_id,
            i + 1,
            scene.get("start", ""),
            scene.get("end", ""),
            scene.get("description", ""),
            is_highlight,
            embedding,
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO scenes (id, video_id, scene_index, start_time, end_time,
                                description, is_highlight, embedding)
            VALUES %s
        """, rows)
    conn.commit()


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def process_video(drive_service, conn, video_info: dict) -> dict:
    file_id   = video_info["file_id"]
    file_name = video_info["file_name"]

    print(f"\n[START] {file_name}")

    if is_already_processed(conn, file_id):
        print(f"[SKIP] Đã xử lý trước đó: {file_name}")
        return {"status": "skipped"}

    mark_processing(conn, file_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Bước 1: Download
            video_path = os.path.join(tmp_dir, file_name)
            print(f"  [1/5] Downloading...")
            download_video(drive_service, file_id, video_path)

            # Bước 2: FFmpeg tech info
            print(f"  [2/5] Extracting tech info...")
            tech = extract_tech_info(video_path)

            # Bước 3: Whisper STT
            print(f"  [3/5] Transcribing audio...")
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(video_path, audio_path)
            transcript = transcribe_audio(audio_path)

            # Bước 4: Extract frames
            print(f"  [4/5] Extracting frames...")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir)
            frames = extract_frames(video_path, frames_dir, fps=FRAMES_PER_SEC)
            print(f"        {len(frames)} frames trích xuất")

            # Bước 5: Gemini Vision (có retry rate limit)
            print(f"  [5/5] Calling Gemini 2.0 Flash...")
            prompt = GEMINI_PROMPT_TEMPLATE.format(
                file_name      = file_name,
                folder_name    = video_info["folder_name"],
                duration       = tech["duration"],
                recording_date = tech["recording_date"] or video_info.get("created_time", "")[:10],
                transcript     = transcript[:3000] if transcript else "(không có audio rõ)",
                frame_count    = min(len(frames), MAX_FRAMES_BATCH),
                series_list    = "\n".join(f"- {s}" for s in VALID_SERIES),
            )

            gemini_result = call_gemini_vision(frames, prompt)
            ai_data       = parse_gemini_response(gemini_result["text"])
            cost_usd      = gemini_result["cost_usd"]

            print(f"        Cost: ${cost_usd:.4f} | Series: {ai_data.get('series', '?')}")

            full_meta = {
                "drive_file_id":      file_id,
                "folder":             video_info["folder_name"],
                "file_path":          video_info["file_path"],
                "file_name":          file_name,
                "duration":           tech["duration"],
                "orientation":        tech["orientation"],
                "recording_date":     tech["recording_date"],
                "summary":            ai_data.get("summary", ""),
                "keywords":           ai_data.get("keywords", []),
                "series":             ai_data.get("series", ""),
                "topic":              ai_data.get("topic", ""),
                "core_message":       ai_data.get("core_message", ""),
                "knowledge":          ai_data.get("knowledge", ""),
                "hook":               ai_data.get("hook", ""),
                "transcript":         transcript,
                "brands":             ai_data.get("brands", []),
                "products":           ai_data.get("products", []),
                "people":             ai_data.get("people", []),
                "target_audience":    ai_data.get("target_audience", ""),
                "shoot_location":     ai_data.get("shoot_location", ""),
                "copyright_status":   ai_data.get("copyright_status", "Công Audio sở hữu"),
                "processing_cost_usd": cost_usd,
                "extra_meta": {
                    "series_needs_review": ai_data.get("series_needs_review", False),
                    "frame_count":         len(frames),
                    "duration_sec":        tech["duration_sec"],
                    "file_size_bytes":     video_info.get("size_bytes", 0),
                },
            }

            video_id = save_video_metadata(conn, full_meta)
            save_scenes(conn, video_id, ai_data.get("scenes", []), ai_data.get("highlight_scenes", []))
            mark_done(conn, file_id, video_id)

            print(f"  [DONE] video_id={video_id} | ${cost_usd:.4f}")
            return {"status": "done", "video_id": video_id, "cost_usd": cost_usd}

        except Exception as e:
            error_msg = str(e)
            print(f"  [ERROR] {error_msg}")
            mark_error(conn, file_id, error_msg)
            return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    pass
