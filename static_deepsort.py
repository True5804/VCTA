import os
import cv2
import json
import numpy as np
from inference_sdk import InferenceHTTPClient
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageDraw, ImageFont

# 初始化 Roboflow client
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="2wVWTbD6xxHqAe1v8EIC")
model_id = "crash-car-detection/3"

# 初始化 DeepSORT
tracker = DeepSort(max_age=20, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)

input_dir = "accident_frames"
output_img_dir = "bbox/images"
output_json_dir = "bbox/json"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# 依檔名排序確保時間序
frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

track_data = {}  # {id: [(frame_idx, (x, y))]}

for frame_idx, image_name in enumerate(frame_files):
    image_path = os.path.join(input_dir, image_name)
    frame = cv2.imread(image_path)
    H, W, _ = frame.shape

    # --- Roboflow 偵測 ---
    result = CLIENT.infer(image_path, model_id=model_id)
    preds = result.get("predictions", [])

    detections = []
    for pred in preds:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        conf = pred["confidence"]
        detections.append(([x1, y1, x2, y2], conf, pred["class"]))

    # --- DeepSORT 追蹤 ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- 繪製結果 ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for t in tracks:
        if not t.is_confirmed():
            continue
        track_id = t.track_id
        l, t_, r, b = t.to_ltrb()
        cx, cy = int((l+r)/2), int((t_+b)/2)

        # 儲存軌跡
        track_data.setdefault(track_id, []).append((frame_idx, (cx, cy)))

        # 繪圖
        draw.rectangle([l, t_, r, b], outline="red", width=4)
        draw.text((l, t_-25), f"ID {track_id}", fill="yellow", font=font)
        draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill="red")

    # 儲存結果圖
    out_img_path = os.path.join(output_img_dir, f"{image_name}")
    pil_img.convert("RGB").save(out_img_path)

    # 儲存每幀的 JSON
    json_output = {
        "frame": image_name,
        "detections": preds,
        "tracks": {k: v[-1] for k, v in track_data.items() if v[-1][0] == frame_idx}
    }
    with open(os.path.join(output_json_dir, f"{os.path.splitext(image_name)[0]}.json"), "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"✅ Frame {frame_idx}: {len(preds)} objects, {len(tracks)} active tracks.")

# --- 儲存完整軌跡 ---
with open(os.path.join(output_json_dir, "tracks_all.json"), "w") as f:
    json.dump(track_data, f, indent=2)
print("💾 Saved all tracks to bbox/json/tracks_all.json")
