# -*- coding: utf-8 -*-
"""
VideoLLaMA3 事故偵測效能量化腳本（單一資料集版）
用途：
  - 針對指定資料集（TAD 或 SOTAD）進行模型推論與效能評估
  - 僅評估 VideoLLaMA3 零樣本辨識能力
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from scipy.stats import bootstrap

# === 1. 手動選擇要測的資料集 ===
# ⚠️ 請改成你要執行的資料夾名稱（TAD 或 SOTAD）
TARGET_DATASET = "SOTAD"

# === 2. 路徑設定 ===
DATASET_ROOT = "datasets_compressed384"
RESULTS_ROOT = "results_videollama3"
os.makedirs(RESULTS_ROOT, exist_ok=True)

dataset_dir = os.path.join(DATASET_ROOT, TARGET_DATASET)
results_dir = os.path.join(RESULTS_ROOT, TARGET_DATASET)
os.makedirs(results_dir, exist_ok=True)

# === 3. 其他設定 ===
CATEGORIES = [("accident", 1), ("normal", 0)]
MAX_FRAMES = 30
N_BOOTSTRAP = 2000

# === 4. 匯入模型 ===
from llama import TrafficAccidentAnalyzer, model, processor
analyzer = TrafficAccidentAnalyzer(model, processor)

# === 5. 收集影片清單 ===
videos, y_true = [], []
for cat, label in CATEGORIES:
    cat_dir = os.path.join(dataset_dir, "test", cat)
    if not os.path.exists(cat_dir):
        print(f"⚠️ 找不到資料夾 {cat_dir}，略過此類別。")
        continue
    for file in os.listdir(cat_dir):
        if file.lower().endswith(".mp4"):
            videos.append(os.path.join(cat_dir, file))
            y_true.append(label)

if not videos:
    raise SystemExit(f"🚫 資料集 {TARGET_DATASET} 無影片可測。")

# === 6. 模型推論 ===
y_pred, y_score = [], []
for video_path in tqdm(videos, desc=f"Analyzing {TARGET_DATASET}"):
    base_name = os.path.basename(video_path).replace(".mp4", ".json")
    result_path = os.path.join(results_dir, base_name)

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = analyzer.analyze_traffic_video(video_path, max_frames=MAX_FRAMES)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    score = float(result.get("summary", {}).get("analysis_confidence", 0.0))
    pred = 1 if score >= 0.5 else 0
    y_score.append(score)
    y_pred.append(pred)

# === 7. 計算效能 ===
y_true_arr = np.array(y_true)
y_pred_arr = np.array(y_pred)
y_score_arr = np.array(y_score)

# === 基本指標 ===
f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
auc = roc_auc_score(y_true_arr, y_score_arr)

# AUROC / AUPRC 若只有單一類別會報錯，需防護
try:
    auc = roc_auc_score(y_true_arr, y_score_arr)
except Exception:
    auc = float("nan")

# === Bootstrap 信賴區間（改為索引抽樣版本）===
def bootstrap_ci(metric_fn, y_true, y_pred_or_score, n_resamples=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)  # 有放回抽樣索引
        yt = y_true[idx]
        yp = y_pred_or_score[idx]
        try:
            val = metric_fn(yt, yp)
            if np.isfinite(val):
                samples.append(val)
        except Exception:
            continue
    if len(samples) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))

# 計算 F1 與 AUROC 的 95% 信賴區間
f1_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp, zero_division=0),
                y_true_arr, y_pred_arr)
auc_ci = bootstrap_ci(lambda yt, ys: roc_auc_score(yt, ys),
                y_true_arr, y_score_arr)

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n🔍 Threshold sweep (0.1 → 0.9)")
for t in np.linspace(0.1, 0.9, 9):
    preds = [1 if s >= t else 0 for s in y_score_arr]
    p = precision_score(y_true_arr, preds, zero_division=0)
    r = recall_score(y_true_arr, preds, zero_division=0)
    f = f1_score(y_true_arr, preds, zero_division=0)
    print(f"t={t:.1f} → Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

# === 8. 顯示結果 ===
print("\n📊 Accident Detection Metrics (AUC + F1 only)")
print("=" * 50)
print(f"Dataset : {TARGET_DATASET}")
print(f"Videos : {len(videos)} (accident={sum(y_true)}, normal={len(videos)-sum(y_true)})")
print(f"F1 Score: {f1:.4f}")
print(f"AUC : {auc:.4f}")
print("=" * 50)

# === 9. 儲存統計結果 ===
# Save
metrics = {
    "dataset": TARGET_DATASET,
    "num_videos": len(videos),
    "f1": float(f1),
    "auc": float(auc),
}
with open(os.path.join(results_dir, "metrics_summary_auc_f1.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n✅ {TARGET_DATASET} 測試完成，結果已輸出至 {results_dir}/metrics_summary.json")
