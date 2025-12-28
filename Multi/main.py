import os
import json
import multiprocessing
from datetime import datetime

from video_vlm import run_video_vlm
from static import run_static
from run_height_then_midas import run_height   

VIDEOS_DIR = "videos"
FINAL_RESULTS = "final_results.json"


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_output = f"output_{video_name}"
    os.makedirs(base_output, exist_ok=True)

    print(f"\n🚀 Start processing {video_name}\n{'='*60}")

    try:
        # === 1️⃣ Video VLM (frame extraction + Gemini AI analysis + accident frames copy)
        run_video_vlm(video_path, base_output)

        # === 2️⃣ Roboflow static detection
        run_static(base_output)

        # === 3️⃣ Camera height + MiDaS distance
        distance = run_height(video_path, base_output)

        # === 4️⃣ Load direction JSON
        direction_path = os.path.join(base_output, "accident_direction.json")
        if os.path.exists(direction_path):
            with open(direction_path, "r", encoding="utf-8") as f:
                direction_data = json.load(f)
            road_direction = direction_data.get("accident_lane_direction")
            camera_heading = direction_data.get("camera_heading")
        else:
            road_direction = None
            camera_heading = None

        # === 5️⃣ Write per-video summary
        entry = {
            "video": f"{video_name}.mp4",
            "distance_m": float(distance) if distance is not None else None,
            "road_direction": road_direction,
            "camera_heading": camera_heading,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return entry

    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")
        return None


def main():
    # 清除舊 final_results
    if os.path.exists(FINAL_RESULTS):
        os.remove(FINAL_RESULTS)

    # 找所有影片
    videos = [
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    ]

    if not videos:
        print("❌ No videos found")
        return

    # multiprocessing
    pool = multiprocessing.Pool(processes=min(len(videos), multiprocessing.cpu_count()))
    results = pool.map(process_video, videos)

    # remove None
    results = [r for r in results if r]

    # 寫 final_results.json
    with open(FINAL_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n📊 Final results saved → {FINAL_RESULTS}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    print("\n🎉 All videos processed")
