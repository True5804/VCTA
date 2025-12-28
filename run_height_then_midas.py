# -*- coding: utf-8 -*-
"""
run_height_then_midas.py

啟動順序：
1. 先執行 estimate_height_ransac.py 產生 height_est_summary.json
2. 再執行 midas_click_measure.py 讀那個 JSON + MiDaS 互動量測

三支檔案需放在同一個資料夾：
- estimate_height_ransac.py
- midas_click_measure.py
- run_height_then_midas.py
"""

import argparse
import os
import sys
import subprocess


def main():
    ap = argparse.ArgumentParser(
        description="Run estimate_height_ransac.py first, then midas_click_measure.py."
    )

    # 共同參數：影像來源（現在變成可以不填，會自動用預設）
    ap.add_argument(
        "--glob",
        type=str,
        default=None,
        help=r'影像 glob，例如 "C:\Users\USER\Desktop\accident\Tainan_frames\*.jpg"',
    )

    # ---- 第一支（高度估計）的主要參數 ----
    ap.add_argument("--yolo", type=str, default="yolov8n.pt",
                    help="Ultralytics YOLO 權重檔")
    ap.add_argument("--classes", type=int, nargs="*",
                    default=[2],
                    help="COCO 類別 id，預設 [2(car)]")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="YOLO 信心門檻")
    ap.add_argument("--carH", type=float, default=1.5,
                    help="車高 prior H (m)")
    ap.add_argument("--ultra-track", type=str,
                    choices=["bytetrack", "botsort"],
                    default="bytetrack",
                    help="Ultralytics tracker 選擇")
    ap.add_argument("--device", type=str, default="cpu",
                    help="device：'cpu'、'0' 等")
    ap.add_argument("--min-box", type=int, default=12,
                    help="忽略太小的 bbox")
    ap.add_argument("--no-micro-adjust", action="store_true",
                    help="關掉 focal micro-adjust")
    ap.add_argument("--vis-dir", type=str, default="track_vis",
                    help="視覺化與 JSON 存放資料夾（第一支會寫 height_est_summary.json 在這裡）")

    # ---- 第二支（MiDaS 量測）的參數 ----
    ap.add_argument("--scale-mode", type=str, default="anchor",
                    choices=["median", "anchor", "blend"],
                    help="MiDaS 尺度模式：median / anchor / blend")
    ap.add_argument("--anchor-kind", type=str, default="three",
                    choices=["center", "three"],
                    help="anchor 像素集合（center or three）")
    ap.add_argument("--anchor-target-mult", type=float, default=1.15,
                    help="anchor 目標幾何距離倍數 k")
    ap.add_argument("--anchor-gain", type=float, default=1.10,
                    help="anchor raw s 之後再乘上的 gain")
    ap.add_argument("--anchor-auto", action="store_true",
                    help="對 (k, gain) 做小範圍 grid search 自動微調")
    ap.add_argument("--blend-alpha", type=float, default=0.7,
                    help="blend 模式：alpha * s_med + (1-alpha)*s_anchor")
    ap.add_argument("--vp-relax", action="store_true",
                    help="如果沒有 summary，又要從影像估 VP 時，可以放寬閥值")
    ap.add_argument("--summary-name", type=str, default="height_est_summary.json",
                    help="第一支輸出的 JSON 檔名（預設 height_est_summary.json）")

    args = ap.parse_args()

    # 這支檔案所在資料夾
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 如果沒有指定 --glob，就用預設的 .\video_frames_0.1s\*.jpg
    if args.glob is None:
        args.glob = os.path.join(base_dir, "video_frames_0.1s", "*.jpg")
        # print(f"[INFO] 未指定 --glob，使用預設影像路徑：{args.glob}", flush=True)

    # ---- 找到兩個子腳本路徑 ----
    height_script = os.path.join(base_dir, "estimate_height_ransac.py")
    midas_script = os.path.join(base_dir, "midas_click_measure.py")

    if not os.path.isfile(height_script):
        print(f"[ERROR] 找不到 {height_script}", flush=True)
        sys.exit(1)
    if not os.path.isfile(midas_script):
        print(f"[ERROR] 找不到 {midas_script}", flush=True)
        sys.exit(1)

    # =====================================================
    # STEP 1：先執行 estimate_height_ransac.py
    # =====================================================
    cmd1 = [
        sys.executable, height_script,
        "--glob", args.glob,
        "--yolo", args.yolo,
        "--conf", str(args.conf),
        "--carH", str(args.carH),
        "--ultra-track", args.ultra_track,
        "--device", args.device,
        "--min-box", str(args.min_box),
        "--vis-dir", args.vis_dir,
        "--save-vis",   # 一定開啟，這樣 JSON 會寫在 vis-dir
    ]

    if args.no_micro_adjust:
        cmd1.append("--no-micro-adjust")
    if args.classes:
        cmd1 += ["--classes"] + [str(c) for c in args.classes]

    print("\n[STEP 1] 執行 estimate_height_ransac.py：", flush=True)
    print(" ".join(cmd1), flush=True)

    r1 = subprocess.run(cmd1)
    if r1.returncode != 0:
        print(f"[ERROR] 第一支程式執行失敗 (returncode={r1.returncode})", flush=True)
        sys.exit(r1.returncode)

    # 預期 JSON 位置：vis-dir/height_est_summary.json
    summary_path = os.path.join(args.vis_dir, args.summary_name)
    if not os.path.isfile(summary_path):
        print(f"[WARN] 找不到 {summary_path}", flush=True)
        print("       第二支會照樣執行，但會改用 --h 或自行算 VP（若有給）。", flush=True)

    # =====================================================
    # STEP 2：再執行 midas_click_measure.py
    # =====================================================
    cmd2 = [
        sys.executable, midas_script,
        "--glob", args.glob,
    ]

    if os.path.isfile(summary_path):
        cmd2 += ["--summary", summary_path]

    # MiDaS 的尺度 & anchor 參數
    cmd2 += [
        "--scale-mode", args.scale_mode,
        "--anchor-kind", args.anchor_kind,
        "--anchor-target-mult", str(args.anchor_target_mult),
        "--anchor-gain", str(args.anchor_gain),
        "--blend-alpha", str(args.blend_alpha),
    ]
    if args.anchor_auto:
        cmd2.append("--anchor-auto")
    if args.vp_relax:
        cmd2.append("--vp-relax")

    print("\n[STEP 2] 執行 midas_click_measure.py：", flush=True)
    print(" ".join(cmd2), flush=True)

    r2 = subprocess.run(cmd2)
    if r2.returncode != 0:
        print(f"[ERROR] 第二支程式執行失敗 (returncode={r2.returncode})", flush=True)
        sys.exit(r2.returncode)

    #print("\n✅ 完成：已先跑 estimate_height_ransac，再開 MiDaS 互動量距。", flush=True)


if __name__ == "__main__":
    main()
