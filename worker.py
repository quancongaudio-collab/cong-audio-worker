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
# Số frame gửi cho Vision API giờ co giãn theo thời lượng video (xem hàm
# compute_max_frames bên dưới). Các hằng số này là biên trên/dưới của việc co giãn đó:
MIN_FRAMES_BATCH = 12   # video ngắn vẫn có tối thiểu chừng này frame
MAX_FRAMES_BATCH = 40   # video dài không vượt quá chừng này (tránh phình chi phí quá mức)
SECONDS_PER_FRAME = 12  # cứ ~12 giây video thì lấy thêm 1 frame
# Timeout (giây) cho các lệnh ffmpeg/ffprobe và tải file — để chúng KHÔNG BAO GIỜ
# treo vô hạn khi gặp file hỏng/tải dở dang. Nếu vượt quá, subprocess sẽ tự raise
# TimeoutExpired, được bắt bởi except Exception trong process_video() và ghi
# thành 'error' trong processing_log thay vì chiếm giữ worker mãi mãi.
FFPROBE_TIMEOUT_SEC   = 60
FFMPEG_AUDIO_TIMEOUT  = 180
FFMPEG_FRAMES_TIMEOUT = 600  # 10 phút
DOWNLOAD_TIMEOUT_SEC  = 600  # tối đa 10 phút để tải xong 1 video
# ─────────────────────────────────────────────
# NÉN VIDEO TRƯỚC KHI XỬ LÝ (nếu bitrate quá cao)
# Một số video (đặc biệt quay 4K/HEVC từ iPhone) có bitrate rất cao dù thời
# lượng ngắn (vd: 1.29GB cho video 2 phút ~ 77 Mbps). Với server chỉ 1 CPU,
# việc giải mã trực tiếp các video này khiến ffmpeg cực chậm, dễ vượt
# FFMPEG_FRAMES_TIMEOUT dù video không hề hỏng. Bước nén này downscale +
# chuyển sang H.264 (giải mã nhẹ hơn HEVC nhiều) để các bước sau nhanh hơn.
# ─────────────────────────────────────────────
FFMPEG_COMPRESS_TIMEOUT      = 500   # tối đa ~8.3 phút cho riêng bước nén (đủ cho file rất nặng)
COMPRESS_BITRATE_THRESHOLD_MBPS = 8  # nếu bitrate gốc > ngưỡng này thì mới nén
COMPRESS_MAX_HEIGHT          = 720   # nén xuống tối đa cao 720px
COMPRESS_CRF                 = 23    # chất lượng: 18=rất cao, 23=tốt/cân bằng, 28=thấp hơn nhưng nhẹ hơn
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
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)
def download_video(service, file_id: str, dest_path: str):
    """
    Tải video từ Drive. Có giới hạn thời gian tổng thể (DOWNLOAD_TIMEOUT_SEC) —
    nếu quá trình tải bị treo/quá chậm (mạng lỗi, file lớn bất thường), hàm sẽ
    tự raise TimeoutError thay vì để vòng lặp next_chunk() chạy vô thời hạn.
    """
    request = service.files().get_media(fileId=file_id)
    started = time.monotonic()
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            if time.monotonic() - started > DOWNLOAD_TIMEOUT_SEC:
                raise TimeoutError(
                    f"Tải file vượt quá {DOWNLOAD_TIMEOUT_SEC}s — có thể mạng lỗi hoặc file quá lớn"
                )
            _, done = downloader.next_chunk()
# ─────────────────────────────────────────────
# 2. FFMPEG
# ─────────────────────────────────────────────
def extract_tech_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SEC
    )
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
def compute_max_frames(duration_sec: float) -> int:
    """
    Tính số frame tối đa nên gửi cho Vision API, co giãn theo thời lượng video.
    Video càng dài thì càng cần nhiều frame để AI có đủ dữ liệu tách ra nhiều
    đoạn giá trị khác nhau, thay vì luôn luôn cố định 10 frame như trước đây.
    """
    estimated = int(duration_sec / SECONDS_PER_FRAME)
    return max(MIN_FRAMES_BATCH, min(MAX_FRAMES_BATCH, estimated))
def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list:
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "5",
        "-s", "640x360",
        frame_pattern,
        "-y", "-loglevel", "error"
    ], check=True, timeout=FFMPEG_FRAMES_TIMEOUT)
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]
def compress_video_if_needed(video_path: str, output_path: str, duration_sec: float) -> str:
    """
    Kiểm tra bitrate thực tế của video (size_bytes / duration). Nếu bitrate
    vượt ngưỡng COMPRESS_BITRATE_THRESHOLD_MBPS, nén lại xuống H.264 +
    COMPRESS_MAX_HEIGHT trước khi dùng cho các bước sau (audio/frame
    extraction) — giúp giảm tải CPU đáng kể cho các video nặng/HEVC/4K.

    Trả về đường dẫn video nên dùng tiếp theo: bản nén nếu nén thành công,
    hoặc bản gốc nếu video đã nhẹ sẵn, hoặc nếu bước nén thất bại/quá lâu
    (trường hợp đó các bước sau sẽ tự xử lý bản gốc như trước đây, có thể
    vẫn timeout nhưng không làm mất dữ liệu hay crash worker).
    """
    try:
        size_bytes = os.path.getsize(video_path)
        if duration_sec <= 0:
            return video_path
        bitrate_mbps = (size_bytes * 8) / duration_sec / 1_000_000
        if bitrate_mbps <= COMPRESS_BITRATE_THRESHOLD_MBPS:
            return video_path  # video đã nhẹ, không cần nén thêm
        print(f"        Bitrate cao ({bitrate_mbps:.1f} Mbps, {size_bytes/1e6:.0f}MB) -> đang nén lại...")
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vf", f"scale=-2:{COMPRESS_MAX_HEIGHT}",
            "-c:v", "libx264", "-crf", str(COMPRESS_CRF), "-preset", "ultrafast",
            "-c:a", "copy",  # không re-encode audio — bước transcribe tự tách audio riêng sau này
            "-threads", "1",
            output_path,
            "-y", "-loglevel", "error"
        ], check=True, timeout=FFMPEG_COMPRESS_TIMEOUT)
        new_size = os.path.getsize(output_path)
        print(f"        Nén xong: {size_bytes/1e6:.0f}MB -> {new_size/1e6:.0f}MB")
        return output_path
    except subprocess.TimeoutExpired:
        print(f"        [WARN] Nén video timeout sau {FFMPEG_COMPRESS_TIMEOUT}s, dùng video gốc")
        return video_path
    except Exception as e:
        print(f"        [WARN] Nén video lỗi ({e}), dùng video gốc")
        return video_path
def extract_audio(video_path: str, output_path: str):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1",
        output_path,
        "-y", "-loglevel", "error"
    ], check=True, timeout=FFMPEG_AUDIO_TIMEOUT)
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
=== AUDIO METADATA — BẮT BUỘC, PHỤC VỤ "ANALYZE ONCE – REUSE EVERYWHERE" ===
Các workflow sản xuất nội dung sau này (Facebook Story, Facebook Reel, TikTok, YouTube Shorts,
Website) sẽ CHỈ ĐỌC các trường audio dưới đây để quyết định giữ âm thanh gốc hay ghép nhạc nền —
KHÔNG được phép gọi AI phân tích lại âm thanh lần thứ hai. Vì vậy bạn phải đánh giá kỹ và chính xác
cho TỪNG đoạn (dựa vào transcript tương ứng với khoảng start_time-end_time của đoạn đó):
- Lời thoại trong đoạn có rõ ràng, nghe được, có giá trị thông tin không?
- Chất lượng âm thanh nền có tốt không (không bị gió, ồn, rè, tiếng nhạc lấn át giọng nói)?
- Nếu lời thoại rõ và có giá trị → nên GIỮ ÂM THANH GỐC.
- Nếu không có lời thoại giá trị, hoặc âm thanh nền kém/ồn/nhiễu → nên GHÉP NHẠC NỀN.
- Nếu lời thoại có giá trị nhưng nền hơi ồn → GIỮ GỐC + HẠ ÂM LƯỢNG, THÊM NHẠC NHẸ.
=== CHẤM ĐIỂM CHẤT LƯỢNG HÌNH ẢNH — BẮT BUỘC, THANG ĐIỂM 0-100 ===
Giống nguyên tắc "Analyze Once – Reuse Everywhere" ở trên, các workflow sản xuất nội dung sau này
sẽ CHỈ ĐỌC các điểm số dưới đây để quyết định đoạn nào đủ chất lượng để dùng ngay, đoạn nào cần
chỉnh sửa hậu kỳ, đoạn nào nên loại bỏ — KHÔNG được phép chấm điểm lại lần thứ hai. Vì vậy bạn phải
đánh giá kỹ và chính xác cho TỪNG đoạn, dựa trên các frame tương ứng với khoảng start_time-end_time
của đoạn đó:
- stability_score (0-100): Độ ổn định hình ảnh — rung tay, lắc máy càng nhiều thì điểm càng thấp.
  90-100: cầm vững/dùng gimbal, gần như không rung. 50-70: rung nhẹ, vẫn xem được.
  Dưới 40: rung mạnh, khó chịu khi xem.
- focus_score (0-100): Độ nét — chủ thể (sản phẩm/người) có bị mờ, out nét, chưa lấy nét kịp không.
  90-100: sắc nét hoàn toàn. Dưới 40: mờ nhòe rõ rệt, mất chi tiết.
- lighting_score (0-100): Ánh sáng — đủ sáng, không cháy sáng (overexposed) hoặc thiếu sáng
  (underexposed), có tôn được chi tiết sản phẩm không.
- composition_score (0-100): Bố cục khung hình — cân đối, chủ thể không bị cắt cụt/che khuất,
  không rối mắt.
- luxury_score (0-100): Cảm giác sang trọng, cao cấp phù hợp thương hiệu audio hi-end — ánh sáng/
  không gian/góc quay có tôn được giá trị sản phẩm không.
- visual_quality_score (0-100): Điểm TỔNG HỢP phản ánh chất lượng hình ảnh chung của đoạn này —
  tự cân nhắc dựa trên 4 điểm trên theo đánh giá tổng thể của bạn, KHÔNG phải phép tính trung bình
  cộng máy móc.
- visual_issues: mảng liệt kê ngắn gọn các vấn đề hình ảnh cụ thể quan sát được trong đoạn này
  (ví dụ: "rung nhẹ ở đầu đoạn", "thiếu sáng góc trái", "mất nét lúc zoom"). Để mảng rỗng [] nếu
  không có vấn đề gì đáng kể.
- asset_quality_status: chọn ĐÚNG MỘT trong 3 giá trị:
  "đạt" — không có vấn đề rung/mất nét/thiếu sáng nghiêm trọng nào, dùng được ngay.
  "cần chỉnh sửa" — có vấn đề nhưng có thể xử lý hậu kỳ (hơi thiếu sáng, rung nhẹ...).
  "không đạt" — rung mạnh, mất nét nghiêm trọng, hoặc thiếu sáng/cháy sáng nặng, không nên dùng.
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
      "platform_usage": "Liệt kê các nền tảng phù hợp, cách nhau bởi ' | '. Ví dụ: 'Story' hoặc 'Reel | TikTok | Shorts'. Được phép chọn nhiều, không cần xếp hạng.",
      "audio_type": "Chọn 1 trong: loi_noi_ro_rang, am_thanh_hien_truong, co_nhac_nen_san, im_lang_hoac_nhieu",
      "has_clear_speech": true,
      "audio_quality": "Chọn 1 trong: tot, trung_binh, kem",
      "audio_recommendation": "Chọn 1 trong: giữ âm thanh gốc, ghép nhạc nền, giữ gốc + hạ âm lượng thêm nhạc nhẹ",
      "audio_reason": "Giải thích ngắn gọn 1 câu vì sao đề xuất audio_recommendation ở trên.",
      "visual_quality_score": 85,
      "luxury_score": 80,
      "composition_score": 82,
      "lighting_score": 78,
      "stability_score": 90,
      "focus_score": 88,
      "visual_issues": ["hơi thiếu sáng ở góc phải"],
      "asset_quality_status": "đạt"
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
def call_vision_api(frames: list, prompt: str, max_frames: int) -> dict:
    """Gọi GPT-4o mini Vision với retry khi rate limit.
    max_frames được tính theo thời lượng video (xem compute_max_frames) thay vì
    dùng một hằng số cố định — video càng dài càng được gửi nhiều frame hơn, giúp
    AI có đủ dữ liệu để tách ra nhiều đoạn giá trị khác nhau thay vì chỉ ra 1 đoạn.
    """
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
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
        "max_tokens": 4096,  # tăng từ 2048 → 4096 để JSON không bị cắt cụt khi có nhiều segment
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
    text = re.sub(r"^(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error: {e}")
        print(f"[WARN] Raw text length: {len(text)} chars — có thể bị cắt cụt do max_tokens")
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
            INSERT INTO processing_log (drive_file_id, status, started_at, updated_at)
            VALUES (%s, 'processing', NOW(), NOW())
            ON CONFLICT (drive_file_id) DO UPDATE
            SET status='processing', started_at=NOW(), updated_at=NOW()
        """, (drive_file_id,))
    conn.commit()
def update_step(conn, drive_file_id: str, step: str):
    """
    Ghi lại bước hiện tại worker đang xử lý cho file này, kèm thời điểm cập nhật.
    Dùng để theo dõi tiến độ/chẩn đoán khi 1 video bị "kẹt" lâu ở trạng thái
    processing — chỉ cần xem cột current_step + updated_at là biết đang đứng ở
    bước nào, không cần mò log Render.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE processing_log
            SET current_step = %s, updated_at = NOW()
            WHERE drive_file_id = %s
        """, (step, drive_file_id))
    conn.commit()
def mark_done(conn, drive_file_id: str, video_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE processing_log
            SET status='done', video_id=%s, done_at=NOW(), current_step='done', updated_at=NOW()
            WHERE drive_file_id=%s
        """, (video_id, drive_file_id))
    conn.commit()
def mark_error(conn, drive_file_id: str, error: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO processing_log (drive_file_id, status, error_message, started_at, current_step, updated_at)
            VALUES (%s, 'error', %s, NOW(), 'error', NOW())
            ON CONFLICT (drive_file_id) DO UPDATE
            SET status='error', error_message=%s, done_at=NOW(), current_step='error', updated_at=NOW()
        """, (drive_file_id, error[:1000], error[:1000]))
    conn.commit()
def save_video_records(conn, common_meta: dict, segments: list) -> list:
    """
    Lưu N dòng vào bảng videos cho 1 file gốc — mỗi dòng là 1 đoạn có giá trị
    tái sử dụng (không còn bảng scenes riêng). Metadata cấp file (drive_file_id,
    file_name, series, brands...) được lặp lại trên mọi dòng; metadata cấp đoạn
    (start_time, end_time, summary, reason, segment_type, platform_usage,
    audio_type, has_clear_speech, audio_quality, audio_recommendation, audio_reason,
    visual_quality_score, luxury_score, composition_score, lighting_score,
    stability_score, focus_score, visual_issues, asset_quality_status, embedding)
    là riêng cho từng dòng.
    Audio metadata (audio_type, has_clear_speech, audio_quality, audio_recommendation,
    audio_reason) và điểm chất lượng hình ảnh (visual_quality_score, luxury_score,
    composition_score, lighting_score, stability_score, focus_score, visual_issues,
    asset_quality_status) đều được AI phân tích DUY NHẤT 1 LẦN ở đây. Các workflow
    sản xuất nội dung sau này (Story/Reel/TikTok/Shorts/Website) chỉ đọc lại các cột
    này để quyết định giữ âm thanh gốc hay ghép nhạc, đoạn nào đủ chất lượng dùng ngay
    hay cần chỉnh sửa hậu kỳ — tuyệt đối không gọi AI phân tích lại lần 2
    (nguyên tắc "Analyze Once – Reuse Everywhere").
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
            "audio_type":            "im_lang_hoac_nhieu",
            "has_clear_speech":      False,
            "audio_quality":         "trung_binh",
            "audio_recommendation":  "ghép nhạc nền",
            "audio_reason":          "Không có đoạn giá trị được xác định nên không đánh giá âm thanh chi tiết; mặc định đề xuất ghép nhạc nền.",
            "visual_quality_score":  None,
            "luxury_score":          None,
            "composition_score":     None,
            "lighting_score":        None,
            "stability_score":       None,
            "focus_score":           None,
            "visual_issues":         [],
            "asset_quality_status":  "cần chỉnh sửa",
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
            seg.get("audio_type", ""),
            bool(seg.get("has_clear_speech", False)),
            seg.get("audio_quality", ""),
            seg.get("audio_recommendation", ""),
            seg.get("audio_reason", ""),
            seg.get("visual_quality_score"),
            seg.get("luxury_score"),
            seg.get("composition_score"),
            seg.get("lighting_score"),
            seg.get("stability_score"),
            seg.get("focus_score"),
            seg.get("visual_issues", []),
            seg.get("asset_quality_status", ""),
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
                platform_usage, audio_type, has_clear_speech, audio_quality,
                audio_recommendation, audio_reason,
                visual_quality_score, luxury_score, composition_score, lighting_score,
                stability_score, focus_score, visual_issues, asset_quality_status,
                embedding
            ) VALUES %s
        """, rows, template="""(
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,NOW(),
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s
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
            update_step(conn, file_id, "downloading")
            download_video(drive_service, file_id, video_path)
            # Bước 2: Tech info
            print(f"  [2/5] Tech info...")
            update_step(conn, file_id, "extracting_tech_info")
            tech = extract_tech_info(video_path)
            # Số frame tối đa sẽ gửi cho Vision API, co giãn theo thời lượng video
            max_frames = compute_max_frames(tech["duration_sec"])
            print(f"        duration={tech['duration_sec']:.0f}s -> max_frames={max_frames}")
            # Bước 2.5: Nén video nếu bitrate quá cao (giúp các bước sau nhanh hơn nhiều)
            update_step(conn, file_id, "compressing")
            compressed_path = os.path.join(tmp_dir, "video_compressed.mp4")
            video_path = compress_video_if_needed(video_path, compressed_path, tech["duration_sec"])
            # Bước 3: Whisper
            print(f"  [3/5] Transcribing...")
            update_step(conn, file_id, "extracting_audio")
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(video_path, audio_path)
            update_step(conn, file_id, "transcribing")
            transcript = transcribe_audio(audio_path)
            # Bước 4: Frames
            print(f"  [4/5] Extracting frames...")
            update_step(conn, file_id, "extracting_frames")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir)
            frames = extract_frames(video_path, frames_dir, fps=FRAMES_PER_SEC)
            print(f"        {len(frames)} frames")
            # Bước 5: GPT-4o mini Vision
            print(f"  [5/5] Calling GPT-4o mini Vision...")
            update_step(conn, file_id, "calling_vision_api")
            prompt = PROMPT_TEMPLATE.format(
                file_name      = file_name,
                folder_name    = video_info.get("folder_name", ""),
                duration       = tech["duration"],
                duration_sec   = tech["duration_sec"],
                recording_date = tech["recording_date"] or video_info.get("created_time", "")[:10],
                transcript     = transcript[:3000] if transcript else "(không có audio)",
                frame_count    = min(len(frames), max_frames),
                series_list    = "\n".join(f"- {s}" for s in VALID_SERIES),
            )
            result   = call_vision_api(frames, prompt, max_frames)
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
            update_step(conn, file_id, "saving_to_db")
            video_ids = save_video_records(conn, common_meta, ai_data.get("valuable_segments", []))
            mark_done(conn, file_id, video_ids[0])
            print(f"  [DONE] {len(video_ids)} record(s) | ${cost_usd:.4f}")
            return {"status": "done", "video_ids": video_ids, "cost_usd": cost_usd}
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout khi chạy ffmpeg/ffprobe: {e}"
            print(f"  [ERROR] {error_msg}")
            mark_error(conn, file_id, error_msg)
            return {"status": "error", "error": error_msg}
        except TimeoutError as e:
            error_msg = f"Timeout khi tải file từ Drive: {e}"
            print(f"  [ERROR] {error_msg}")
            mark_error(conn, file_id, error_msg)
            return {"status": "error", "error": error_msg}
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"  [ERROR] {error_msg}")
            print(traceback.format_exc())
            mark_error(conn, file_id, error_msg)
            return {"status": "error", "error": error_msg}
