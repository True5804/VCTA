import subprocess
import os
import cv2
import time
import json

def display_images_from_folder(folder_path, window_title):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"📭 No images to show in {folder_path}")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            continue
        cv2.imshow(f"{window_title} - {image_file}", img)
        cv2.waitKey(1000)
        cv2.destroyWindow(f"{window_title} - {image_file}")

def show_json_content(json_path, title="JSON Output"):
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n🧾 {title} ({json_path})\n" + "="*60)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("="*60 + "\n")


# === STEP 1: Run video analysis and extract accident frames ===
print("▶️ Step 1: Running video_vlm.py to analyze video and extract accident frames...")
subprocess.run(["python", "video_vlm.py"], check=True)
print("✅ video_vlm.py completed.\n")

show_json_content("video_vlm_analysis.json", "Step 1: AI Video Analysis")

# === STEP 2: Roboflow detection and drawing bounding boxes ===
print("▶️ Step 2 : Running static_deepsort.py to draw bounding boxes on frames...")
subprocess.run(["python", "static.py"], check=True)
print("✅ static_deepsort.py completed.\n")

display_images_from_folder("accident_frames", "Step 2: Accident Frames")
display_images_from_folder("bbox/images", "Step 3: Localization")

# print("▶️ Step 2.5 : Running direction_estimator.py to analysis direction...")
# subprocess.run(["python", "direction_estimator.py"], check=True)
# print("✅ direction_estimator.py completed.\n")

# === STEP 3: Camera height + MiDaS distance (run_height_then_midas.py) ===
print("▶️ Step 3 : Running run_height_then_midas.py for camera height & MiDaS distance...")
subprocess.run(["python", "run_height_then_midas.py"], check=True)
print("✅ run_height_then_midas.py completed.\n")

# 如果你之後想看高度估計的 JSON，可以順便秀出來（可留可刪）
height_summary_path = os.path.join("track_vis", "height_est_summary.json")
show_json_content("final_results.json", "Step 3: Distance Estimation")

print("🎉 All tasks complete!")
