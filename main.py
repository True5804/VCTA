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
print("▶️ Step 2 : Running static.py to draw bounding boxes on frames...")
subprocess.run(["python", "static.py"], check=True)
print("✅ static.py completed.\n")

display_images_from_folder("accident_frames", "Step 2: Accident Frames")

# === STEP 3: Draw distance lines and determine location ===
print("▶️ Step 3 : Running line.py to overlay lines and locate object distance...")
subprocess.run(["python", "line.py"], check=True)
print("✅ line.py completed.\n")

if os.path.exists("current_video.txt"):
    with open("current_video.txt", "r") as f:
        video_filename = f.read().strip()
    video_name = os.path.splitext(video_filename)[0]
    line_output_dir = f"bbox/line_output_{video_name}"
    
    display_images_from_folder("bbox/images", "Step 3: Roboflow BBox Images")
    display_images_from_folder(line_output_dir, "Step 3: Line Overlay Results")

# 顯示 accident_distances.json 新增的結果
show_json_content("accident_distances.json", "Step 3: Estimated Distance")

print("🎉 All tasks complete!")
