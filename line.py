import os
import cv2
import numpy as np
import json

def read_y_coordinates(txt_path, scale_factor):
    with open(txt_path, "r") as file:
        return sorted([int(int(line.strip()) * scale_factor) for line in file if line.strip().isdigit()], reverse=True)

def get_center_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "predictions" not in data or not data["predictions"]:
        return None
    return [(int(pred["x"]), int(pred["y"])) for pred in data["predictions"]]

def find_interval(y, lines):
    for i in range(len(lines) - 1):
        if lines[i] >= y > lines[i + 1]:
            return (lines[i], lines[i + 1])
    return None

# ✅ JSON output file
output_json_path = "accident_distances.json"

# ✅ Append result to JSON
def append_to_json(video_filename, min_dist, max_dist, output_path):
    entry = {
        "name": video_filename,
        "distance": f"{min_dist}-{max_dist}m"
    }

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_images(input_dir, mask_path, y_coords_path, json_dir, output_dir, video_filename):
    os.makedirs(output_dir, exist_ok=True)
    mask = cv2.imread(mask_path)

    sample_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not sample_images:
        raise FileNotFoundError("❌ No image found in input directory.")

    sample_image = cv2.imread(os.path.join(input_dir, sample_images[0]))
    scale_factor = sample_image.shape[0] / mask.shape[0]
    y_coords = read_y_coordinates(y_coords_path, scale_factor)

    distances = set()

    for image_name in sample_images:
        image_path = os.path.join(input_dir, image_name)
        json_path = os.path.join(json_dir, os.path.splitext(image_name)[0] + ".json")
        frame = cv2.imread(image_path)

        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_binary = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        mask_indices = np.where(mask_binary > 128)
        line_overlay = np.zeros_like(frame)

        for y in y_coords:
            if 0 <= y < frame.shape[0] and np.any(mask_indices[0] == y):
                cv2.line(line_overlay, (0, y), (frame.shape[1], y), (0, 255, 0), 2)

        frame[mask_indices] = cv2.addWeighted(frame, 1, line_overlay, 1, 0)[mask_indices]

        centers = get_center_from_json(json_path)
        if not centers:
            print(f"🚫 No prediction for {image_name}")
            continue

        sorted_lines = sorted(y_coords, reverse=True)

        for cx, cy in centers:
            interval = find_interval(cy, sorted_lines)
            if interval:
                idx_top = sorted_lines.index(interval[0])
                idx_bottom = sorted_lines.index(interval[1])
                line_index = min(idx_top, idx_bottom)
                min_dist = 20 + line_index * 10
                max_dist = min_dist + 10
                distances.add((min_dist, max_dist))
                print(f"📍 {image_name}: Location of center dot : {min_dist}m–{max_dist}m from camera.")
            else:
                print(f"📍 {image_name}: Location of center dot is outside the defined intervals.")

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, frame)

    if distances:
        min_dist, max_dist = min(distances, key=lambda x: x[0])
        append_to_json(video_filename, min_dist, max_dist, output_json_path)

# === Read video filename from current_video.txt ===
if not os.path.exists("current_video.txt"):
    raise FileNotFoundError("❌ current_video.txt not found. Make sure video_vlm.py has been run.")

with open("current_video.txt", "r") as f:
    video_filename = f.read().strip()

video_name = os.path.splitext(video_filename)[0]

# Paths based on video name
input_dir = "bbox/images"
json_dir = "bbox/json"
mask_path = f"{video_name}_mask.png"
y_coords_path = f"{video_name}.txt"
output_dir = f"bbox/line_output_{video_name}"

process_images(input_dir, mask_path, y_coords_path, json_dir, output_dir, video_filename)
