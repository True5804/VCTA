import os
import json
import multiprocessing
from video_vlm import run_video_vlm
from static import run_static
from line import run_line
import cv2

DISTANCE_OUTPUT_FILE = "accident_distances.json"
VIDEO_DIR = "videos"

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
        cv2.waitKey(0)  # 顯示每張圖片 1 秒


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_output_dir = f"output_{video_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"\n🚀 Start processing {video_name}\n{'='*60}")

    try:
        # Step 1: Run video_vlm.py
        from video_vlm import run_video_vlm
        run_video_vlm(video_path, base_output_dir)

        # ➕ 顯示每支影片的 video_vlm_analysis.json 分析結果
        analysis_path = os.path.join(base_output_dir, "video_vlm_analysis.json")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
            print(f"\n🧠 Gemini Analysis Result for {video_name} ({analysis_path})")
            print("=" * 60)
            print(json.dumps(analysis_data, indent=2, ensure_ascii=False))
            print("=" * 60 + "\n")
        else:
            print(f"⚠️ Analysis result not found for {video_name}")

        # Step 2: Run static.py
        from static import run_static
        run_static(base_output_dir)

        # Step 3: Run line.py
        from line import run_line
        run_line(video_path, base_output_dir)

        # Step 4: Show line overlay images
        line_output_dir = os.path.join(base_output_dir, f"bbox/line_output_{video_name}")
        display_images_from_folder(line_output_dir, f"{video_name} - Line Output")

        print(f"\n✅ Completed: {video_name}\n{'='*60}")
    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")


def main():
    if os.path.exists(DISTANCE_OUTPUT_FILE):
        os.remove(DISTANCE_OUTPUT_FILE)

    video_files = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    if not video_files:
        print("❌ No videos found in 'videos/'")
        return

    # Use multiprocessing to handle videos concurrently
    processes = []
    for video in video_files:
        p = multiprocessing.Process(target=process_video, args=(video,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Show final accident distances
    if os.path.exists(DISTANCE_OUTPUT_FILE):
        print(f"\n📊 Final accident distances in {DISTANCE_OUTPUT_FILE}")
        with open(DISTANCE_OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("⚠️ No accident_distances.json generated.")

if __name__ == "__main__":
    main()
