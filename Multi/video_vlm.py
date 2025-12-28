import os
import requests
import cv2
import base64
import json
import shutil

# ✅ Your Gemini API Key (replace with your actual API Key)
API_KEY = ""
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ✅ File paths)

def load_camera_heading_from_json(video_path, json_path="data.json"):
    if not os.path.exists(json_path):
        print(f"❌ JSON metadata not found: {json_path}")
        return None
    
    video_name = os.path.basename(video_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item.get("檔案名稱") == video_name:
            heading = item.get("鏡頭拍攝方向")
            if heading:
                return heading.lower()
    print(f"⚠️ No matching metadata for video: {video_path}")
    return None



def extract_frames_per_second(video_path):
    """Extract one frame per second from the video and save as images"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Unable to open video file for frame extraction")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Video's frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Duration in seconds

    print(f"📸 Extracting frames from {video_path} (Duration: {duration} seconds, FPS: {fps})...")

    frame_count = 0
    second = 0  # Start naming from 0.jpg

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0 and second <= duration:
            frame_path = os.path.join(FRAMES_DIR, f"{second}.jpg")
            cv2.imwrite(frame_path, frame)
            second += 1

        frame_count += 1

    cap.release()

def extract_frames_every_0_1s(video_path, output_dir):
    """⭐ 新增：從影片中每 0.1 秒提取一張影格並儲存為圖片 (output_dir)"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 無法開啟影片檔案進行影格擷取（0.1 秒）")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("❌ FPS 讀取失敗，無法進行每 0.1 秒擷取")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    #print(f"📸 正在從 {video_path} 擷取【每 0.1 秒 1 張】影格（持續時間: {duration} 秒, FPS: {fps}）...")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    # 每 0.1 秒的 frame 步長
    step = max(1, int(round(fps * 0.1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            frame_path = os.path.join(output_dir, f"{t:.1f}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_idx += 1

    cap.release()


def encode_video_to_base64(video_path):
    """Convert video to Base64 encoding"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def parse_to_bullet_points(text, video_duration):
    """Parse text into bullet points with explicit timestamps from the model"""
    sentences = [s.strip() for s in text.split("\n") if s.strip()]  # Split by newline for structured output
    bullet_points = []

    for sentence in sentences:
        import re
        timestamp_match = re.search(r'\[(\d+(\.\d+)?)\s*sec\]', sentence)
        if timestamp_match:
            timecode = float(timestamp_match.group(1))
            bullet_text = sentence.replace(timestamp_match.group(0), "").strip()
        else:
            timecode = None
            bullet_text = sentence

        if bullet_text.startswith("- "):
            bullet_text = bullet_text[2:].strip()

        bullet_points.append({
            "bullet_point": f"- {bullet_text}",
            "timecode": timecode if timecode is not None else None
        })

    no_timestamp_count = sum(1 for bp in bullet_points if bp["timecode"] is None)
    if no_timestamp_count > 0:
        time_increment = video_duration / no_timestamp_count
        current_time = 0
        for bp in bullet_points:
            if bp["timecode"] is None:
                bp["timecode"] = round(current_time, 2)
                current_time += time_increment

    return bullet_points

def set_timecodes(bullet_points):
    """Prepare the bullet points with timecodes for JSON output"""
    return bullet_points

def detect_accident_and_copy_frames(bullet_points, source_dir, target_dir):
    """Detect accident keywords and copy frames at, before, and after the timestamp"""
    # Active keywords indicate the moment of the accident
    active_keywords = {"crash", "crashing","crashes", "collision", "collides", "collide", "colliding","fall",
                        "falls", "falling","roll over", "rolls over", "hit","hits", "hitting", "tips over", 
                        "tip over","topple onto", "topples onto", "overturned", "flips over", "flip over",
                        "lose control", "lost control", "accident", "debris", "out of control", "crashed",
                        "Debris", "accidents", "fell"}
    accident_frames = set()  # Use set to avoid duplicates
    last_accident_time = -float('inf')  # Track the last accident time to avoid overlapping events


    for point in bullet_points:
        bullet_text = point["bullet_point"].lower()
        words = bullet_text.split()
        
        # Check for active accident keywords (moment of accident)
        active_matched = [keyword for keyword in active_keywords if keyword in words]
        # Check for aftermath keywords (post-accident state)

        timestamp = int(point["timecode"])
        
        # Detect if this is an active accident event
        if active_matched :
            last_accident_time = timestamp  # Update the last accident time
            timestamps_to_copy = [timestamp, timestamp+1]

            for ts in timestamps_to_copy:
                if ts >= 0:
                    source_frame = os.path.join(source_dir, f"{ts}.jpg")
                    target_frame = os.path.join(target_dir, f"{ts}.jpg")
                    
                    if os.path.exists(source_frame):
                        shutil.copy(source_frame, target_frame)
                        accident_frames.add(target_frame)
                    
        
    if accident_frames:
        #print(f"✅ {len(accident_frames)} unique accident frames copied to {target_dir}")
        return list(accident_frames)
    else:
        print("✅ No accident detected in the video analysis.")
        return None

def analyze_video_with_gemini(video_path, CAMERA_HEADING):
    """Upload video to Google Gemini API and analyze"""
    video_base64 = encode_video_to_base64(video_path)
    if not video_base64:
        return

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Analyze the traffic condition in this video frame-by-frame. "
                            "For each observation, provide a timestamp in seconds (e.g., [5 sec]) "
                            "indicating when it occurs. "
                            "Return the analysis as a list of observations, one per line."
                            "I need a fresh, frame-by-frame analysis without prior context."
                        )
                    },
                    {"inlineData": {"mimeType": "video/mp4", "data": video_base64}}
                ]
            }
        ]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    #print("📡 Gemini status:", response.status_code)  # 👈 add this
    #print("📜 Raw Gemini reply:", response.text[:500])  # 👈 add this (partial)

    if response.status_code == 200:
        response_json = response.json()
        text_output = response_json["candidates"][0]["content"]["parts"][0]["text"]

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_duration = total_frames / fps
        cap.release()

        bullet_points = parse_to_bullet_points(text_output, video_duration=actual_duration)
        formatted_data = set_timecodes(bullet_points)

        with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump(formatted_data, file, ensure_ascii=False, indent=4)


        accident_frames = detect_accident_and_copy_frames(formatted_data, FRAMES_DIR, ACCIDENT_FRAMES_DIR)
        # === 新增：第二次問 Gemini「事故車道方向」 ===

        # === 第二次問 Gemini：取 motion_relative_to_camera ===

        direction_prompt = f"""
            You are analyzing a traffic accident video.

            Your ONLY task:
            Determine the motion direction of the involved vehicle(s) in the moments just before the crash,
            relative to the camera, as exactly one of:

            - "toward"
            - "away"
            - "left"
            - "right"

            Definitions:
            - "toward": the vehicle becomes larger / approaches the camera.
            - "away": the vehicle moves toward the horizon / becomes smaller.
            - "left": the vehicle moves left across the screen.
            - "right": the vehicle moves right across the screen.

            Rules:
            - DO NOT infer real-world road direction.
            - DO NOT output compass directions.
            - DO NOT judge lane direction.
            - DO NOT mention camera orientation.
            - DO NOT provide additional text.

            Output strictly in JSON:

            {{
            "motion_relative_to_camera": "toward | away | left | right"
            }}
        """

        direction_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": direction_prompt},
                        {"inlineData": {"mimeType": "video/mp4", "data": video_base64}}
                    ]
                }
            ]
        }

        # ==== 1. 送出第二次請求 ====
        try:
            direction_response = requests.post(GEMINI_API_URL, headers=headers, json=direction_payload)
            if direction_response.status_code == 200:
                direction_json_raw = direction_response.json()
                direction_text = direction_json_raw["candidates"][0]["content"]["parts"][0]["text"]

                # ==== 2. 清洗掉 ```json 包裝 ====
                direction_text_clean = direction_text.strip()
                if direction_text_clean.startswith("```"):
                    lines = direction_text_clean.splitlines()
                    if len(lines) >= 3:
                        direction_text_clean = "\n".join(lines[1:-1]).strip()

                # ==== 3. Parse JSON ====
                try:
                    motion_data = json.loads(direction_text_clean)
                except json.JSONDecodeError:
                    motion_data = {
                        "motion_relative_to_camera": None,
                        "raw_response": direction_text
                    }

                motion = motion_data.get("motion_relative_to_camera")

                # ==== 4. Python 做 mapping ====
                mapping = {
                    "north": {
                        "toward": "southbound",
                        "away": "northbound",
                        "left": "westbound",
                        "right": "eastbound"
                    },
                    "south": {
                        "toward": "northbound",
                        "away": "southbound",
                        "left": "eastbound",
                        "right": "westbound"
                    },
                    "east": {
                        "toward": "westbound",
                        "away": "eastbound",
                        "left": "northbound",
                        "right": "southbound"
                    },
                    "west": {
                        "toward": "eastbound",
                        "away": "westbound",
                        "left": "southbound",
                        "right": "northbound"
                    }
                }

                accident_lane_direction = None
                if motion in mapping[CAMERA_HEADING]:
                    accident_lane_direction = mapping[CAMERA_HEADING][motion]

                # ==== 5. 寫入最終 JSON ====
                final_output = {
                    "motion_relative_to_camera": motion,
                    "accident_lane_direction": accident_lane_direction,
                    "camera_heading": CAMERA_HEADING,
                    "raw_model_output": direction_text
                }

                with open(ACCIDENT_DIRECTION_JSON, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, ensure_ascii=False, indent=4)

                #print("🚧 Accident lane direction saved to", ACCIDENT_DIRECTION_JSON)
            else:
                print("⚠️ Gemini direction API error:", direction_response.status_code, direction_response.text)

        except Exception as e:
            print("⚠️ Error while requesting accident lane direction from Gemini:", e)


def run_video_vlm(video_path, output_dir):
    global FRAMES_DIR, FRAMES_DIR_01, ACCIDENT_FRAMES_DIR, JSON_OUTPUT_PATH, ACCIDENT_DIRECTION_JSON

    # 動態換 output 資料夾
    FRAMES_DIR = os.path.join(output_dir, "frames")
    FRAMES_DIR_01 = os.path.join(output_dir, "frames_01")
    ACCIDENT_FRAMES_DIR = os.path.join(output_dir, "accident_frames")
    JSON_OUTPUT_PATH = os.path.join(output_dir, "video_vlm_analysis.json")
    ACCIDENT_DIRECTION_JSON = os.path.join(output_dir, "accident_direction.json")

    # 目錄確保存在
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR_01, exist_ok=True)
    os.makedirs(ACCIDENT_FRAMES_DIR, exist_ok=True)

    load_camera_heading_from_json(video_path, json_path="data.json")
    CAMERA_HEADING = load_camera_heading_from_json(video_path)

    if CAMERA_HEADING is None:
        CAMERA_HEADING = "north" 

    normalize_map = {
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west"
    }

    CAMERA_HEADING = normalize_map.get(CAMERA_HEADING.lower(), CAMERA_HEADING)

    extract_frames_per_second(video_path)
    extract_frames_every_0_1s(video_path, FRAMES_DIR_01)
    analyze_video_with_gemini(video_path, CAMERA_HEADING)
