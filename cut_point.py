"""
cut_point.py — Chọn điểm cắt video theo câu nói trọn vẹn (Quân CôngAudio)

Áp dụng 7 quy tắc:
  1. Thời lượng mục tiêu 20-40 giây, linh hoạt +/-7 giây (tức chấp nhận 13-47s).
  2. Tuyệt đối không kết thúc khi người nói đang nói dở 1 từ/câu/ý.
  3. Điểm kết thúc phải nằm sau khi hoàn thành trọn câu/ý.
  4. Ưu tiên cắt tại khoảng lặng tự nhiên >= 0.3s sau câu nói.
  5. Có thể co giãn thời lượng so với mục tiêu để giữ câu trọn vẹn.
  6. Nếu câu vượt quá thời lượng tối đa -> không cắt ngang, chọn 1 câu/đoạn
     khác ngắn hơn trong cùng video thay thế.
  7. Chừa thêm 0.2-0.5 giây sau từ cuối cùng trước khi kết thúc.

Nếu không tìm được điểm cắt nào hợp lệ trong toàn bộ video (mọi lựa chọn đều
cụt nghĩa) -> trả về None, video đó sẽ bị loại khỏi kho chọn video.
"""

TARGET_MIN = 20.0
TARGET_MAX = 40.0
FLEX_SEC = 7.0
ABS_MIN = TARGET_MIN - FLEX_SEC   # 13s — sàn tuyệt đối
ABS_MAX = TARGET_MAX + FLEX_SEC   # 47s — trần tuyệt đối

END_BUFFER_SEC = 0.35             # nằm giữa khoảng 0.2-0.5s yêu cầu
NATURAL_PAUSE_MIN_SEC = 0.3       # khoảng lặng tự nhiên tối thiểu để được ưu tiên


def seconds_to_timestamp(sec: float) -> str:
    """Đổi số giây sang chuỗi 'MM:SS' hoặc 'HH:MM:SS' (khớp định dạng đang lưu trong DB)."""
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def timestamp_to_seconds(ts) -> float:
    """Đổi chuỗi 'MM:SS' hoặc 'HH:MM:SS' (hoặc số) sang giây (float). Trả 0 nếu không hợp lệ."""
    if ts is None or ts == "":
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return 0.0


def _find_start_index(segments, requested_start_sec):
    """Tìm segment đầu tiên mà mốc requested_start_sec đang nằm trong đó hoặc ngay trước đó."""
    for i, seg in enumerate(segments):
        if seg["end"] > requested_start_sec:
            return i
    return None


def _ends_with_question(text: str) -> bool:
    """Kiểm tra thô: câu này có phải câu hỏi không (kết thúc bằng dấu '?').
    Dùng làm lớp phòng vệ thêm — ưu tiên không dừng ngay sau câu hỏi nếu còn
    lựa chọn khác, vì hỏi xong mà không có câu trả lời là cụt ý, dù câu đó
    ngữ pháp vẫn trọn vẹn. Đây chỉ là tín hiệu cơ học (dấu câu), không hiểu
    được ngữ nghĩa thật — lớp chính vẫn là hướng dẫn cho AI ở worker.py."""
    return text.strip().endswith("?")


def _try_window(segments, start_idx):
    """
    Từ segment start_idx (dùng làm điểm bắt đầu thật), dò tới các segment sau để
    tìm điểm kết thúc hợp lệ tốt nhất: nằm trong [ABS_MIN, ABS_MAX], ưu tiên có
    khoảng lặng tự nhiên trước câu kế tiếp, ưu tiên gần dải mục tiêu 20-40s, và
    ưu tiên KHÔNG dừng ngay sau 1 câu hỏi (tránh cụt ý kiểu hỏi xong không có
    trả lời).
    Trả về (start_sec, end_sec) hoặc None nếu không có ứng viên hợp lệ.
    """
    actual_start = segments[start_idx]["start"]
    best_score = None
    best_window = None
    for j in range(start_idx, len(segments)):
        end_candidate = segments[j]["end"] + END_BUFFER_SEC
        dur = end_candidate - actual_start
        if dur > ABS_MAX:
            break  # càng đi xa càng dài hơn nữa, không cần dò tiếp
        if dur < ABS_MIN:
            continue
        dist_to_target = 0.0 if TARGET_MIN <= dur <= TARGET_MAX else min(
            abs(dur - TARGET_MIN), abs(dur - TARGET_MAX)
        )
        gap_to_next = (
            segments[j + 1]["start"] - segments[j]["end"]
            if j + 1 < len(segments) else 999.0
        )
        has_natural_pause = gap_to_next >= NATURAL_PAUSE_MIN_SEC
        is_question_end = _ends_with_question(segments[j]["text"])
        score = (1 if is_question_end else 0, 0 if has_natural_pause else 1, dist_to_target)
        if best_score is None or score < best_score:
            best_score = score
            best_window = (actual_start, end_candidate)
    return best_window


def select_cut_window(segments, requested_start_sec):
    """
    segments: list các dict {"start": float, "end": float, "text": str} lấy từ
              Whisper verbose_json (đã có mốc giờ thật theo câu).
    requested_start_sec: điểm bắt đầu do AI-Vision đề xuất (chỉ mang tính gần đúng).

    Trả về (start_sec, end_sec) đã chốt đúng 7 quy tắc, hoặc None nếu không tìm
    được điểm cắt hợp lệ nào trong toàn bộ video (video này nên bị loại bỏ).
    """
    if not segments:
        return None
    segments = sorted(segments, key=lambda s: s["start"])

    # 1. Thử bắt đầu đúng tại điểm AI-Vision đề xuất trước
    start_idx = _find_start_index(segments, requested_start_sec)
    if start_idx is not None:
        window = _try_window(segments, start_idx)
        if window:
            return window

    # 2. Không cắt được từ điểm đề xuất (vd câu đó tự nó đã dài hơn ABS_MAX)
    #    -> quy tắc 6: chọn một câu/đoạn khác ngắn hơn, ưu tiên gần điểm đề xuất nhất
    order = sorted(
        range(len(segments)),
        key=lambda i: abs(segments[i]["start"] - requested_start_sec),
    )
    for i in order:
        window = _try_window(segments, i)
        if window:
            return window

    # 3. Không có cách nào cắt mà không cụt câu trong toàn bộ video -> bỏ
    return None
