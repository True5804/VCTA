import os
import cv2
import json
import numpy as np

def read_y_coords(txt_path, scale):
    with open(txt_path, "r") as f:
        return sorted([int(int(line.strip()) * scale) for line in f if line.strip().isdigit()], reverse=True)

def get_centers(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [(int(p["x"]), int(p["y"])) for p in data.get("predictions", [])]

def find_interval(y, lines):
    for i in range(len(lines) - 1):
        if lines[i] >= y > lines[i + 1]:
            return (lines[i], lines[i + 1])
    return None

def append_distance(video_name, min_d, max_d, out_path="accident_distances.json"):
    entry = {"name": video_name, "distance": f"{min_d}-{max_d}m"}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []
    data.append(entry)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_line(video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    img_dir = os.path.join(output_dir, "bbox/images")
    json_dir = os.path.join(output_dir, "bbox/json")
    mask_path = f"{video_name}_mask.png"
    y_txt = f"{video_name}.txt"
    out_dir = os.path.join(output_dir, f"bbox/line_output_{video_name}")
    os.makedirs(out_dir, exist_ok=True)

    sample = cv2.imread(os.path.join(img_dir, os.listdir(img_dir)[0]))
    mask = cv2.imread(mask_path)
    scale = sample.shape[0] / mask.shape[0]
    y_coords = read_y_coords(y_txt, scale)

    dists = set()
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        mask_rs = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_bin = cv2.cvtColor(mask_rs, cv2.COLOR_BGR2GRAY)
        mask_idx = np.where(mask_bin > 128)
        overlay = np.zeros_like(frame)
        for y in y_coords:
            if np.any(mask_idx[0] == y):
                cv2.line(overlay, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
        frame[mask_idx] = cv2.addWeighted(frame, 1, overlay, 1, 0)[mask_idx]

        centers = get_centers(os.path.join(json_dir, os.path.splitext(img_name)[0] + ".json"))
        for cx, cy in centers:
            interval = find_interval(cy, y_coords)
            if interval:
                idx = y_coords.index(interval[0])
                min_d = 20 + idx * 10
                max_d = min_d + 10
                dists.add((min_d, max_d))
        cv2.imwrite(os.path.join(out_dir, img_name), frame)

    if dists:
        d1, d2 = min(dists), max(dists)
        append_distance(video_name, d1[0], d2[1])
