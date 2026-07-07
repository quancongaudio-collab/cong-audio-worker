OPENAI_VISION_URL = "https://api.openai.com/v1/chat/completions"

def call_gemini_vision(frames: list, prompt: str) -> dict:
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

    max_retries = 3
    for attempt in range(max_retries):
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
                print(f"[WARN] OpenAI rate limit — chờ {wait_sec}s")
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
            if attempt < max_retries - 1:
                time.sleep(30)
            else:
                raise
    raise Exception("OpenAI API vẫn lỗi sau 3 lần thử")
