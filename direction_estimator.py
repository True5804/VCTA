import os
import cv2
import json
import numpy as np

# ========== 可調參數 ==========
TRACK_JSON = "bbox/json/tracks_all.json"
ACC_FRAME_DIR = "accident_frames"
ROAD_AXIS_FILE = "camera_config.json"  # 檔內需含 {"road_axis": [x, y], "cctv_heading": "north"}
OUT_JSON = "bbox/json/accident_directions.json"

# ========== 工具函式 ==========

def direction_from_tracks(track_data, acc_center, radius=30):
    """在追蹤資料中尋找與事故點最近的軌跡，並求移動方向。"""
    best_id, best_dist = None, 1e9
    direction_vec = None
    for tid, pts in track_data.items():
        pts = np.array([p[1] for p in pts])
        if len(pts) < 2:
            continue
        dist = np.linalg.norm(pts[-1] - np.array(acc_center))
        if dist < best_dist and dist < radius:
            best_id, best_dist = tid, dist
            direction_vec = np.mean(np.diff(pts[-5:], axis=0), axis=0)
    if direction_vec is None:
        return None
    norm = np.linalg.norm(direction_vec)
    return direction_vec / (norm + 1e-6)


def direction_from_optical_flow(prev_frame, curr_frame, acc_center, roi=40):
    """以光流法估算事故中心附近的平均移動方向。"""
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    x, y = int(acc_center[0]), int(acc_center[1])
    roi_flow = flow[max(0, y - roi): y + roi, max(0, x - roi): x + roi, :]
    if roi_flow.size == 0:
        return None
    mean_flow = np.mean(roi_flow.reshape(-1, 2), axis=0)
    norm = np.linalg.norm(mean_flow)
    return mean_flow / (norm + 1e-6)


def classify_direction(vec, road_axis, cctv_heading):
    proj = np.dot(vec, road_axis)
    if abs(proj) < 1e-3:
        return "未知"
    same_dir = proj > 0

    # 根據攝影機朝向轉換真實方向
    if cctv_heading == "south":
        return "南下" if same_dir else "北上"
    elif cctv_heading == "north":
        return "北上" if same_dir else "南下"
    elif cctv_heading == "east":
        return "東行" if same_dir else "西行"
    elif cctv_heading == "west":
        return "西行" if same_dir else "東行"
    else:
        return "未知"


# ========== 主程式 ==========

def main():
    if not os.path.exists(TRACK_JSON):
        print(f"❌ 找不到追蹤檔案：{TRACK_JSON}")
        return

    with open(TRACK_JSON, "r") as f:
        track_data = json.load(f)
        # 轉換成 ndarray 格式
        for k in track_data:
            track_data[k] = [(p[0], np.array(p[1])) for p in track_data[k]]

    # 讀取道路方向與鏡頭朝向
    if os.path.exists(ROAD_AXIS_FILE):
        with open(ROAD_AXIS_FILE, "r") as f:
            cam_cfg = json.load(f)
            road_axis = np.array(cam_cfg.get("road_axis", [1, 0]), dtype=float)
            road_axis /= np.linalg.norm(road_axis)
            cctv_heading = cam_cfg.get("cctv_heading", "north")
    else:
        print("⚠️ 找不到 camera_config.json，使用預設 road_axis=[1,0], heading=north")
        road_axis = np.array([1, 0])
        cctv_heading = "north"

    # 讀取事故影格
    frames = sorted([f for f in os.listdir(ACC_FRAME_DIR) if f.lower().endswith(('.jpg', '.png'))])
    if len(frames) < 2:
        print("⚠️ 影格不足，無法分析方向。")
        return

    directions = []
    for idx, frame_name in enumerate(frames):
        frame_path = os.path.join(ACC_FRAME_DIR, frame_name)
        curr_frame = cv2.imread(frame_path)
        if curr_frame is None:
            continue

        # 估計事故中心（目前取整張畫面中心，也可讀 YOLO 輸出中心）
        h, w, _ = curr_frame.shape
        acc_center = (w / 2, h / 2)

        # 若有軌跡匹配 → 使用追蹤方向；否則用光流
        vec = direction_from_tracks(track_data, acc_center)
        if vec is None and idx > 0:
            prev_frame = cv2.imread(os.path.join(ACC_FRAME_DIR, frames[idx - 1]))
            vec = direction_from_optical_flow(prev_frame, curr_frame, acc_center)
        if vec is None:
            continue

        direction = classify_direction(vec, road_axis, cctv_heading)
        print(f"[Frame {frame_name}] → {direction}")
        directions.append({"frame": frame_name, "center": acc_center, "direction": direction})

    with open(OUT_JSON, "w") as f:
        json.dump(directions, f, indent=2, ensure_ascii=False)
    print(f"💾 已輸出方向結果至 {OUT_JSON}")


if __name__ == "__main__":
    main()
