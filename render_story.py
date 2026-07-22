"""
render_story.py — Logic render Video Story (Quân CôngAudio)

File này KHÔNG đụng tới worker.py hay logic lập chỉ mục video hiện có.
Chỉ import và dùng trong route mới /render-story ở app.py.
"""

import os
import re
import time
import random
import shutil
import subprocess


# Thư mục nhạc nền — cần bạn tự tạo và tải file nhạc lên đây trong repo:
#   music/ambient/*.mp3
#   music/piano/*.mp3
#   music/jazz/*.mp3
#   music/classical/*.mp3
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")


def pick_music_file(mood):
    """Chọn ngẫu nhiên 1 file nhạc trong thư mục đúng mood. Nếu không có, fallback về 'ambient'."""
    folder = os.path.join(MUSIC_DIR, mood)
    if not os.path.isdir(folder):
        folder = os.path.join(MUSIC_DIR, "ambient")
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))


def slugify_for_filename(text, max_len=60):
    """
    Chuyển overlay_text thành đoạn an toàn để chèn vào tên file trên Drive.
    Giữ nguyên tiếng Việt có dấu (Drive/Windows/macOS đều hỗ trợ Unicode trong tên file),
    chỉ loại bỏ các ký tự không hợp lệ trong tên file và đổi khoảng trắng thành dấu gạch ngang.
    """
    if not text:
        return ""
    # Bỏ ký tự không hợp lệ trong tên file: \ / : * ? " < > | và xuống dòng
    cleaned = re.sub(r'[\\/:*?"<>|\n\r]', '', text)
    # Gộp khoảng trắng thừa, đổi thành dấu gạch ngang
    cleaned = re.sub(r'\s+', '-', cleaned.strip())
    # Bỏ dấu gạch ngang/chấm thừa ở đầu/cuối
    cleaned = cleaned.strip('-.')
    return cleaned[:max_len]


def download_drive_file(drive_service, file_id, dest_path):
    """Tải 1 file từ Google Drive về đường dẫn local."""
    from googleapiclient.http import MediaIoBaseDownload
    request = drive_service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_drive_file(drive_service, local_path, file_name, parent_folder_id=None):
    """Upload 1 file lên Google Drive, trả về id + link xem."""
    from googleapiclient.http import MediaFileUpload
    file_metadata = {"name": file_name}
    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]
    media = MediaFileUpload(local_path, mimetype="video/mp4", resumable=True)
    uploaded = drive_service.files().create(
        body=file_metadata, media_body=media, fields="id, webViewLink",
        supportsAllDrives=True
    ).execute()
    return uploaded


def get_user_drive_service():
    """Tạo Drive service dùng OAuth2 tài khoản Gmail thật — dùng riêng cho upload, tránh lỗi storage quota của Service Account."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def render_story_video(drive_service, data):
    """
    Hàm chính: tải video gốc -> FFmpeg cắt/ghép nhạc -> upload kết quả -> dọn dẹp.

    data (dict) cần có các khoá:
      story_id (str)               - mã Story, dùng đặt tên file
      drive_file_id (str)          - id video gốc trên Drive
      overlay_text (str)           - câu mô tả (KHÔNG còn burn vào video — chỉ dùng để đặt tên file Drive)
      overlay_time_sec (float)     - không còn dùng để vẽ chữ, giữ lại tham số cho tương thích ngược
      duration (float)             - thời lượng đoạn cắt (giây), sẽ tự giới hạn 20-40s
      highlight_start_sec (float)  - giây bắt đầu cắt trong video gốc
      keep_original_audio (bool)   - True: giữ tiếng gốc, False: ghép nhạc nền
      music_mood (str)             - 'ambient' | 'piano' | 'jazz' | 'classical'
      output_folder_id (str, optional) - id thư mục Drive để lưu video kết quả
    """
    story_id = str(data.get("story_id", f"tmp{int(time.time())}"))
    drive_file_id = data["drive_file_id"]
    overlay_text = (data.get("overlay_text") or "").replace("\n", " ").strip()
    duration = min(max(float(data.get("duration") or 30), 20), 40)
    start = float(data.get("highlight_start_sec") or 0)
    keep_original_audio = bool(data.get("keep_original_audio") or False)
    music_mood = data.get("music_mood") or "ambient"
    output_folder_id = data.get("output_folder_id")

    tmp_dir = f"/tmp/qca_render_{story_id}_{int(time.time())}"
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, "src.mp4")
    output_path = os.path.join(tmp_dir, f"qca_story_{story_id}.mp4")

    try:
        # 1. Tải video gốc về
        download_drive_file(drive_service, drive_file_id, input_path)

        # 2. Filter video: crop dọc + fade — KHÔNG còn drawtext, video xuất ra sạch không chữ
        vf = (
            f"crop=ih*9/16:ih,scale=1080:1920,"
            f"fade=t=in:st=0:d=0.6,fade=t=out:st={duration - 0.6}:d=0.6"
        )

        # 3. Ghép lệnh FFmpeg tuỳ theo có giữ âm thanh gốc hay không
        if keep_original_audio:
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
                "-t", str(duration), "-vf", vf,
                "-map", "0:v:0", "-map", "0:a:0?",
                "-threads", "1", "-preset", "ultrafast",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", output_path,
            ]
        else:
            music_path = pick_music_file(music_mood)
            if not music_path:
                raise RuntimeError(f"Không tìm thấy file nhạc cho mood '{music_mood}' trong thư mục music/.")
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
                "-t", str(duration), "-i", music_path,
                "-vf", vf, "-filter_complex", "[1:a]volume=0.18[bgm]",
                "-map", "0:v", "-map", "[bgm]",
                "-threads", "1", "-preset", "ultrafast",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", output_path,
            ]

        # 4. Chạy FFmpeg
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg lỗi (mã {proc.returncode}): {proc.stderr[-2000:]}")

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg chạy xong nhưng không tạo được file output.")

        # 5. Đặt tên file Drive kèm câu mô tả (overlay_text) để người đăng thấy ngay trên Drive,
        #    thay vì burn chữ vào video. Ví dụ: qca_story_1_Am-thanh-xua-tan-moi-uu-phien.mp4
        filename_slug = slugify_for_filename(overlay_text)
        drive_file_name = (
            f"qca_story_{story_id}.mp4"
            if not filename_slug
            else f"qca_story_{story_id}_{filename_slug}.mp4"
        )

        # 6. Upload kết quả lên Drive (dùng tài khoản Gmail thật qua OAuth, không dùng Service Account)
        user_drive_service = get_user_drive_service()
        uploaded = upload_drive_file(
            user_drive_service, output_path, drive_file_name, output_folder_id
        )

        return {
            "status": "ok",
            "story_id": story_id,
            "output_drive_file_id": uploaded.get("id"),
            "output_drive_link": uploaded.get("webViewLink"),
            "used_music": None if keep_original_audio else os.path.basename(music_path),
        }

    finally:
        # 7. Dọn dẹp file tạm — LUÔN chạy dù thành công hay lỗi
        shutil.rmtree(tmp_dir, ignore_errors=True)
