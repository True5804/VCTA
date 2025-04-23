import os
import requests
import cv2
import base64
import json
import shutil

# ✅ Your Gemini API Key (replace with your actual API Key)
API_KEY = "AIzaSyCaWfMhLhTmBuf7wWEfrNOeNTKKkErstzQ"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ✅ File paths
VIDEO_PATH = "2.mp4"  # The existing video file to analyze
JSON_OUTPUT_PATH = "video_vlm_analysis.json"
FRAMES_DIR = "video_frames"  # Directory to save all frames
ACCIDENT_FRAMES_DIR = "accident_frames"  # Directory to save accident frames

# Write the video name to a file so line.py can read it
with open("current_video.txt", "w") as f:
    f.write(VIDEO_PATH)

# Ensure directories exist
if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)
if not os.path.exists(ACCIDENT_FRAMES_DIR):
    os.makedirs(ACCIDENT_FRAMES_DIR)

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
                        "lose control", "lost control", "accident", "debris", "out of control", "crashed"}
    # Aftermath keywords indicate post-accident state (we'll use these to filter out aftermath)
    aftermath_keywords = {"accident", "incident", "debris", "damage", "damages", "damaged"}
    accident_frames = set()  # Use set to avoid duplicates
    last_accident_time = -float('inf')  # Track the last accident time to avoid overlapping events
    COOLDOWN_PERIOD = 3  # Seconds to wait before considering a new accident

    for point in bullet_points:
        bullet_text = point["bullet_point"].lower()
        words = bullet_text.split()
        
        # Check for active accident keywords (moment of accident)
        active_matched = [keyword for keyword in active_keywords if keyword in words]
        # Check for aftermath keywords (post-accident state)
        aftermath_matched = [keyword for keyword in aftermath_keywords if keyword in words]

        timestamp = int(point["timecode"])
        
        # Detect if this is an active accident event
        if active_matched and (timestamp - last_accident_time) > COOLDOWN_PERIOD:
            last_accident_time = timestamp  # Update the last accident time
            timestamps_to_copy = [timestamp]

            for ts in timestamps_to_copy:
                if ts >= 0:
                    source_frame = os.path.join(source_dir, f"{ts}.jpg")
                    target_frame = os.path.join(target_dir, f"{ts}.jpg")
                    
                    if os.path.exists(source_frame):
                        shutil.copy(source_frame, target_frame)
                        accident_frames.add(target_frame)
                    
        
    if accident_frames:
        print(f"✅ {len(accident_frames)} unique accident frames copied to {target_dir}")
        return list(accident_frames)
    else:
        print("✅ No accident detected in the video analysis.")
        return None

def analyze_video_with_gemini(video_path):
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
        

if __name__ == "__main__":
    extract_frames_per_second(VIDEO_PATH)
    analyze_video_with_gemini(VIDEO_PATH)