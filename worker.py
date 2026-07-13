"""
Công Audio — Video Indexing Worker V1.0
Phương án B: GPT-4o mini Vision + Whisper (Groq) + FFmpeg
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
from typing import Optional

import requests
import psycopg2
from psycopg2.extras import Json, execute_values
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
DATABASE_URL     = os.environ["DATABASE_URL"]
DRIVE_CREDS_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT"]

FRAMES_PER_SEC   = 1
MAX_FRAMES_BATCH = 10

OPENAI_VISION_URL = "https://api.openai.com/v1/chat/completions"
EMBEDDING_URL     = "https://api.openai.com/v1/embeddings"

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
        "-q:v", "5",
        "-s", "640x360",
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
    if not GROQ_API_KEY:
        return ""
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={
                    "model":         "whisper-large-v3",
                    "language":      "vi",
                    "response_format": "text",
                },
                timeout=120,
            )
        if resp.status_code == 200:
            return resp.text.strip()
        print(f"[WARN] Whisper error: {resp.status_code}")
        return ""
    except Exception as e:
        print(f"[WARN] Whisper exception: {e}")
        return ""


# ─────────────────────────────────────────────
# 4. GPT-4o mini VISION
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """
Bạn là chuyên gia phân tích nội dung video cho Công Audio — cửa hàng audio hi-end tại Việt Nam.
Nhiệm vụ của bạn là xây dựng KHO TRI THỨC VIDEO để các AI Agent sau này tự động sản xuất nội dung
cho Facebook Story, Facebook Reel, Video Facebook, TikTok, YouTube Shorts, Website — MÀ KHÔNG PHẢI
xem lại video gốc.

=== THÔNG TIN KỸ THUẬT ===
- File name: {file_name}
- Thư mục: {folder_name}
- Thời lượng gốc: {duration} (tổng {duration_sec:.0f} giây)
- Ngày quay: {recording_date}

=== TRANSCRIPT ===
{transcript}

=== NHIỆM VỤ ===
Phân tích {frame_count} frame bên dưới (trải đều từ 00:00 đến hết video) và transcript.
Tìm trong video các ĐOẠN CÓ GIÁ TRỊ TÁI SỬ DỤNG — không phải mọi đoạn, không phải toàn bộ video.

=== QUY TẮC CHỌN ĐOẠN — RẤT QUAN TRỌNG ===
TUYỆT ĐỐI KHÔNG chia video theo thời gian đều nhau. KHÔNG bắt buộc phải bao phủ hết video.
KHÔNG chọn một đoạn chỉ vì camera đổi góc quay.

Chỉ chọn đoạn khi nó có ít nhất một trong các giá trị sau:
- Kể được một câu chuyện nhỏ
- Truyền được cảm xúc
- Có ý đồ truyền thông rõ ràng
- Có giá trị tri thức (dạy/giải thích điều gì đó)
- Có khả năng tái sử dụng nhiều lần cho nội dung marketing

Ưu tiên các loại khoảnh khắc sau (nếu có xuất hiện trong video):
- Quá trình setup / lắp đặt thiết bị
- Phản ứng thật của khách hàng
- Lời giải thích giá trị sản phẩm/dịch vụ
- Khoảnh khắc đẹp về hình ảnh/âm thanh
- Không gian showroom
- Thao tác kỹ thuật
- Trải nghiệm thực tế của người dùng

Số lượng đoạn: KHÔNG giới hạn, do BẠN tự quyết định dựa trên nội dung thực tế.
Một video 15 phút có thể cho ra 5, 8, hay 12 đoạn — tuỳ nội dung, không tuỳ công thức.
Nếu video ({duration_sec:.0f} giây) THỰC SỰ không có đoạn nào đáng tái sử dụng, trả về mảng rỗng []
— điều này hoàn toàn chấp nhận được, không cố gò ép tạo ra đoạn giá trị thấp.

=== ĐỘ DÀI ĐOẠN — CHỈ LÀ GỢI Ý, KHÔNG PHẢI QUY ĐỊNH CỨNG ===
Tham khảo (được phép linh hoạt nếu nội dung thực sự có giá trị):
- Facebook Story: 15-30 giây (tối đa ~40 giây nếu thực sự cuốn)
- Facebook Reel: 30-90 giây
- TikTok: 30-60 giây
- YouTube Shorts: 45-90 giây
- Facebook Video / Website: 2-5 phút

=== 30 SERIES HỢP LỆ (áp dụng cho cả video, không phải riêng từng đoạn) ===
{series_list}

=== JSON CẦN TRẢ VỀ ===
{{
  "series": "Chọn ĐÚNG MỘT trong 30 series, áp dụng cho cả video.",
  "topic": "Chủ đề chính của cả video.",
  "brands": ["thương hiệu xuất hiện rõ"],
  "products": ["loại thiết bị"],
  "people": ["người xuất hiện"],
  "target_audience": "Đối tượng mục tiêu.",
  "shoot_location": "Địa điểm quay.",
  "copyright_status": "Công Audio sở hữu",
  "valuable_segments": [
    {{
      "start_time": "00:12",
      "end_time": "00:47",
      "summary": "Mô tả cụ thể câu chuyện/nội dung của đoạn này.",
      "reason": "Vì sao đoạn này có giá trị tái sử dụng (câu chuyện/cảm xúc/tri thức/...).",
      "core_message": "Thông điệp quan trọng nhất của đoạn này. Một câu.",
      "knowledge": "Tri thức rút ra từ đoạn này, nếu có. Một câu hoặc để trống.",
      "hook": "Câu mở đầu hấp dẫn nếu dùng đoạn này làm clip riêng, hoặc chuỗi rỗng.",
      "keywords": ["từ khóa 1", "từ khóa 2", "từ khóa 3"],
      "segment_type": "Chọn 1 trong: qua_trinh_setup, phan_ung_khach_hang, loi_giai_thich_gia_tri, khoanh_khac_dep, khong_gian_showroom, thao_tac_ky_thuat, trai_nghiem_thuc_te, khac",
      "platform_usage": "Liệt kê các nền tảng phù hợp, cách nhau bởi ' | '. Ví dụ: 'Story' hoặc 'Reel | TikTok | Shorts'. Được phép chọn nhiều, không cần xếp hạng."
    }}
  ]
}}

QUY TẮC: Chỉ trả về JSON, không markdown. Series phải đúng 1 trong 30 tên.
Chỉ ghi những gì thấy/nghe rõ. Không cần viết caption, không cần nội dung đăng bài —
chỉ mô tả và phân loại để phục vụ tra cứu sau này.
"""


def encode_frame_to_base64(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_api(frames: list, prompt: str) -> dict:
    """Gọi GPT-4o mini Vision với retry khi rate limit."""
    if len(frames) > MAX_FRAMES_BATCH:
        step = len(frames) / MAX_FRAMES_BATCH
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES_BATCH)]

    content = [{"type": "text", "text": prompt}]
    for frame_path in frames:
        b64 = encode_frame_to_base64(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"
            }
        })

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                OPENAI_VISION_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429:
                wait_sec = 30 * (attempt + 1)
                print(f"[WARN] OpenAI rate limit — chờ {wait_sec}s (lần {attempt+1}/3)")
                time.sleep(wait_sec)
                continue
            resp.raise_for_status()
            raw = resp.json()
            text = raw["choices"][0]["message"]["content"]
            usage = raw.get("usage", {})
            input_tokens  = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
            return {"text": text, "cost_usd": cost_usd}
        except requests.exceptions.HTTPError as e:
            if attempt < 2:
                print(f"[WARN] OpenAI error: {e} — thử lại sau 30s")
                time.sleep(30)
            else:
                raise
    raise Exception("OpenAI API vẫn lỗi sau 3 lần thử")


def parse_ai_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error: {e}")
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
            ON CONFLICT (drive_file_id) DO UPDATE
            SET status='processing', started_at=NOW()
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


def save_video_records(conn, common_meta: dict, segments: list) -> list:
    """
    Lưu N dòng vào bảng `videos` cho 1 file gốc — mỗi dòng là 1 đoạn có giá trị
    tái sử dụng (không còn bảng `scenes` riêng). Metadata cấp file (drive_file_id,
    file_name, series, brands...) được lặp lại trên mọi dòng; metadata cấp đoạn
    (start_time, end_time, summary, reason, segment_type, platform_usage, embedding)
    là riêng cho từng dòng.
    """
    if not segments:
        # Không có đoạn nào đáng tái sử dụng — vẫn lưu 1 dòng fallback để không mất
        # hoàn toàn dữ liệu / video vẫn tra cứu được, đánh dấu rõ lý do.
        segments = [{
            "start_time":     "00:00",
            "end_time":       common_meta.get("duration", ""),
            "summary":        common_meta.get("video_overview", "Không phát hiện đoạn nổi bật."),
            "reason":         "Video không có đoạn nào đủ giá trị tái sử dụng theo tiêu chí; lưu toàn video để tham khảo.",
            "core_message":   "",
            "knowledge":      "",
            "hook":           "",
            "keywords":       [],
            "segment_type":   "khac",
            "platform_usage": "",
        }]

    rows = []
    ids  = []
    for i, seg in enumerate(segments):
        video_id   = str(uuid.uuid4())
        embed_text = seg.get("summary", "")
        embedding  = create_embedding(embed_text)
        ids.append(video_id)
        rows.append((
            video_id,
            common_meta["drive_file_id"],
            common_meta["file_name"],
            common_meta["folder"],
            common_meta["file_path"],
            "Indexed",
            common_meta["duration"],
            common_meta["orientation"],
            common_meta["recording_date"] or None,
            common_meta.get("shoot_location", ""),
            seg.get("summary", ""),
            seg.get("keywords", []),
            common_meta.get("series", ""),
            common_meta.get("topic", ""),
            seg.get("core_message", ""),
            seg.get("knowledge", ""),
            seg.get("hook", ""),
            common_meta.get("brands", []),
            common_meta.get("products", []),
            common_meta.get("people", []),
            common_meta.get("target_audience", ""),
            common_meta.get("copyright_status", "Công Audio sở hữu"),
            common_meta.get("transcript", ""),
            common_meta.get("processing_cost_usd", 0),
            Json({**common_meta.get("extra_meta", {}), "segment_index": i + 1, "total_segments": len(segments)}),
            i + 1,
            seg.get("start_time", ""),
            seg.get("end_time", ""),
            seg.get("reason", ""),
            seg.get("segment_type", ""),
            seg.get("platform_usage", ""),
            embedding,
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO videos (
                id, drive_file_id, file_name, folder, file_path,
                status, duration, orientation, recording_date, shoot_location,
                summary, keywords, series, topic, core_message,
                knowledge, hook, brands, products, people,
                target_audience, copyright_status, transcript,
                processing_cost_usd, extra_meta, created_at,
                segment_index, start_time, end_time, reason, segment_type,
                platform_usage, embedding
            ) VALUES %s
        """, rows, template="""(
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,NOW(),
            %s,%s,%s,%s,%s,
            %s,%s
        )""")
    conn.commit()
    return ids


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def process_video(drive_service, conn, video_info: dict) -> dict:
    file_id   = video_info["file_id"]
    file_name = video_info["file_name"]

    print(f"\n[START] {file_name}")

    if is_already_processed(conn, file_id):
        print(f"[SKIP] Đã xử lý: {file_name}")
        return {"status": "skipped"}

    mark_processing(conn, file_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Bước 1: Download
            video_path = os.path.join(tmp_dir, "video.mp4")
            print(f"  [1/5] Downloading...")
            download_video(drive_service, file_id, video_path)

            # Bước 2: Tech info
            print(f"  [2/5] Tech info...")
            tech = extract_tech_info(video_path)

            # Bước 3: Whisper
            print(f"  [3/5] Transcribing...")
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(video_path, audio_path)
            transcript = transcribe_audio(audio_path)

            # Bước 4: Frames
            print(f"  [4/5] Extracting frames...")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir)
            frames = extract_frames(video_path, frames_dir, fps=FRAMES_PER_SEC)
            print(f"        {len(frames)} frames")

            # Bước 5: GPT-4o mini Vision
            print(f"  [5/5] Calling GPT-4o mini Vision...")
            prompt = PROMPT_TEMPLATE.format(
                file_name      = file_name,
                folder_name    = video_info.get("folder_name", ""),
                duration       = tech["duration"],
                duration_sec   = tech["duration_sec"],
                recording_date = tech["recording_date"] or video_info.get("created_time", "")[:10],
                transcript     = transcript[:3000] if transcript else "(không có audio)",
                frame_count    = min(len(frames), MAX_FRAMES_BATCH),
                series_list    = "\n".join(f"- {s}" for s in VALID_SERIES),
            )

            result   = call_vision_api(frames, prompt)
            ai_data  = parse_ai_response(result["text"])
            cost_usd = result["cost_usd"]
            n_segs   = len(ai_data.get("valuable_segments", []))
            print(f"        Cost: ${cost_usd:.4f} | Series: {ai_data.get('series', '?')} | Đoạn giá trị: {n_segs}")

            common_meta = {
                "drive_file_id":       file_id,
                "folder":              video_info.get("folder_name", ""),
                "file_path":           video_info.get("file_path", ""),
                "file_name":           file_name,
                "duration":            tech["duration"],
                "orientation":         tech["orientation"],
                "recording_date":      tech["recording_date"],
                "series":              ai_data.get("series", ""),
                "topic":               ai_data.get("topic", ""),
                "transcript":          transcript,
                "brands":              ai_data.get("brands", []),
                "products":            ai_data.get("products", []),
                "people":              ai_data.get("people", []),
                "target_audience":     ai_data.get("target_audience", ""),
                "shoot_location":      ai_data.get("shoot_location", ""),
                "copyright_status":    ai_data.get("copyright_status", "Công Audio sở hữu"),
                "processing_cost_usd": cost_usd,
                "extra_meta": {
                    "series_needs_review": ai_data.get("series_needs_review", False),
                    "frame_count":         len(frames),
                    "duration_sec":        tech["duration_sec"],
                },
            }

            video_ids = save_video_records(conn, common_meta, ai_data.get("valuable_segments", []))
            mark_done(conn, file_id, video_ids[0])

            print(f"  [DONE] {len(video_ids)} record(s) | ${cost_usd:.4f}")
            return {"status": "done", "video_ids": video_ids, "cost_usd": cost_usd}

        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"  [ERROR] {error_msg}")
            print(traceback.format_exc())
            mark_error(conn, file_id, error_msg)
            return {"status": "error", "error": error_msg}
