# -*- coding: utf-8 -*-
"""
video_compressed.py
批次壓縮多資料集 (TAD, SO-TAD ...) 下的事故與正常影片
保持原資料夾結構，輸出至 datasets_compressed/
"""

import os
import subprocess

# === 設定根目錄 ===
SRC_ROOT = "datasets"
DST_ROOT = "datasets_compressed384"

# === 壓縮參數設定 ===
SCALE = "384:216"    # 解析度 (可改為 320:240、480:270 ...)
FRAMERATE = "3"      # 降幀率 (每秒取幾幀)

# === 自動掃描所有資料集 ===
datasets = [d for d in os.listdir(SRC_ROOT)
             if os.path.isdir(os.path.join(SRC_ROOT, d))]

print(f"📁 偵測到 {len(datasets)} 個資料集: {datasets}")

# === 批次處理 ===
for dataset in datasets:
    src_dataset = os.path.join(SRC_ROOT, dataset)
    dst_dataset = os.path.join(DST_ROOT, dataset)

    # 遞迴搜尋 test/accident 與 test/normal
    for category in ["accident", "normal"]:
        src_dir = os.path.join(src_dataset, "test", category)
        if not os.path.exists(src_dir):
            continue  # 若沒有該類別則跳過

        dst_dir = os.path.join(dst_dataset, "test", category)
        os.makedirs(dst_dir, exist_ok=True)
        print(f"\n🎞️ 開始壓縮 {dataset}/{category} 影片...")

        # 遍歷所有 mp4 影片
        for filename in os.listdir(src_dir):
            if not filename.lower().endswith(".mp4"):
                continue
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # ffmpeg 壓縮命令
            cmd = [
                "ffmpeg", "-i", src_path,
                "-vf", f"scale={SCALE}",
                "-r", FRAMERATE,
                "-preset", "fast",
                "-y", dst_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {dataset}/{category}/{filename} 壓縮完成")

print("\n🎉 所有資料集影片壓縮完成！")
print(f"➡️ 結果已儲存於：{DST_ROOT}/")
