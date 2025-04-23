import os
import requests
import cv2
import base64
import json
import shutil

API_KEY = "AIzaSyCaWfMhLhTmBuf7wWEfrNOeNTKKkErstzQ"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def extract_frames_per_second(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"📸 Extracting {video_path} ({duration:.2f}s, {fps}fps)")

    frame_count = 0
    second = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % fps == 0:
            frame_path = os.path.join(frames_dir, f"{second}.jpg")
            cv2.imwrite(frame_path, frame)
            second += 1
        frame_count += 1
    cap.release()

def encode_video_to_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_to_bullet_points(text, duration):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    bullets = []
    for line in lines:
        import re
        match = re.search(r'\[(\d+(\.\d+)?)\s*sec\]', line)
        if match:
            timecode = float(match.group(1))
            line = line.replace(match.group(0), "").strip()
        else:
            timecode = None
        bullets.append({"bullet_point": line, "timecode": timecode})
    # auto-fill missing timecodes
    no_ts = [b for b in bullets if b["timecode"] is None]
    if no_ts:
        step = duration / len(no_ts)
        t = 0
        for b in bullets:
            if b["timecode"] is None:
                b["timecode"] = round(t, 2)
                t += step
    return bullets

def detect_accident_and_copy_frames(bullet_points, source_dir, target_dir):
    import re

    # 保險起見先建立目標資料夾
    os.makedirs(target_dir, exist_ok=True)

    active_keywords = {
        "crash", "crashing", "crashes", "collision", "collides", "collide", "colliding", "fall",
        "falls", "falling", "roll", "hit", "hits", "hitting", "flip", "flips", "flipping",
        "overturn", "lose", "lost", "accident", "topple", "topples", "debris"
    }
    negation_keywords = {"no", "not", "never", "none", "without"}

    accident_frames = set()
    last_accident_time = -float('inf')
    COOLDOWN_PERIOD = 3

    for point in bullet_points:
        text = point["bullet_point"].lower()
        words = re.findall(r'\b\w+\b', text)

        if any(w in active_keywords for w in words) and not any(n in words for n in negation_keywords):
            timestamp = int(point["timecode"])
            if (timestamp - last_accident_time) > COOLDOWN_PERIOD:
                last_accident_time = timestamp
                for ts in [timestamp - 1, timestamp, timestamp + 1]:
                    if ts < 0:
                        continue
                    src = os.path.join(source_dir, f"{ts}.jpg")
                    dst = os.path.join(target_dir, f"{ts}.jpg")
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        accident_frames.add(dst)

    if accident_frames:
        print(f"✅ Copied {len(accident_frames)} accident frame(s) to {target_dir}")
        return list(accident_frames)
    else:
        print("⚠️ No accident frames found or copied.")
        return None

def run_video_vlm(video_path, output_dir):
    frames_dir = os.path.join(output_dir, "video_frames")
    accident_dir = os.path.join(output_dir, "accident_frames")
    json_output_path = os.path.join(output_dir, "video_vlm_analysis.json")

    extract_frames_per_second(video_path, frames_dir)

    video_b64 = encode_video_to_base64(video_path)
    payload = {
        "contents": [{
            "parts": [
                {"text": (
                            "Analyze the traffic condition in this video frame-by-frame. "
                            "For each observation, provide a timestamp in seconds (e.g., [5 sec]) "
                            "indicating when it occurs. "
                            "Return the analysis as a list of observations, one per line."
                            "I need a fresh, frame-by-frame analysis without prior context."
                        )},
                {"inlineData": {"mimeType": "video/mp4", "data": video_b64}}
            ]
        }]
    }
    res = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, json=payload)
    if res.status_code != 200:
        print("❌ Gemini API error:", res.text)
        return

    response_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    bullets = parse_to_bullet_points(response_text, duration)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(bullets, f, indent=2, ensure_ascii=False)

    detect_accident_and_copy_frames(bullets, frames_dir, accident_dir)
