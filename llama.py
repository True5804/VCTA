import os
import cv2
import json
import shutil
import torch
import re
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Dict, Tuple, Optional

# ✅ VideoLLaMA3 本地模型設定
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-2B"

# Initialize model and processor directly
print("🔄 正在載入 VideoLLaMA3 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ 模型載入完成")

# ✅ 檔案路徑設定
VIDEO_PATH = "2.mp4"
JSON_OUTPUT_PATH = "video_vlm_analysis.json"
FRAMES_DIR = "video_frames"              # 每秒 1 張
FRAMES_DIR_01 = "video_frames_0.1s"      # ⭐ 新增：每 0.1 秒 1 張
ACCIDENT_FRAMES_DIR = "accident_frames"
STATIC_ANALYSIS_LOG = "static_image_analysis.json"

# 將影片名稱寫入檔案，供 line.py 讀取
with open("current_video.txt", "w") as f:
    f.write(VIDEO_PATH)

# 確保目錄存在
for dir_path in [FRAMES_DIR, FRAMES_DIR_01, ACCIDENT_FRAMES_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class TrafficAccidentAnalyzer:
    """完整三步驟交通事故分析器 - 符合原始要求"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    @torch.inference_mode()
    def infer(self, conversation: List[Dict], generation_params: Dict = None) -> str:
        """基礎推理函數"""
        
        default_params = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        if generation_params:
            default_params.update(generation_params)
        
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        output_ids = self.model.generate(**inputs, **default_params)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return response
    
    def step1_analyze_video_natural_language(self, video_path: str, max_frames: int = 30) -> str:
        """步驟 1: 分析視頻並以自然語言輸出描述"""
        print("🎥 步驟 1: 以自然語言分析視頻...")
        
        conversation = [
            {
                "role": "system", 
                "content": (
                    "You are a professional traffic accident analyst. "
                    "Analyze this video frame by frame and describe what happens in natural language. "
                    "For each observation, state at which second it happens in the video. "
                    "Focus on vehicles, movements, traffic conditions, and any potential accidents or dangerous situations."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": {
                            "video_path": video_path, 
                            "fps": 1, 
                            "max_frames": max_frames
                        }
                    },
                    {
                        "type": "text", 
                        "text": (
                            f"Please analyze this {max_frames}-second traffic video frame by frame. "
                            "For each second, describe what you observe. "
                            "Pay special attention to:\n"
                            "- Vehicle movements and positions\n"
                            "- Traffic signals and signs\n"
                            "- Any collisions, near-misses, or dangerous situations\n"
                            "- Pedestrians or other road users\n"
                            "- Weather and road conditions\n\n"
                            "Format: Start each observation with the second number (e.g., 'At 5 seconds:...')"
                        )
                    }
                ]
            }
        ]
        
        description = self.infer(conversation)
        
        print(f"✅ 步驟 1 完成: 生成了 {len(description)} 字符的描述")
        print(f"\n📝 自然語言描述:")
        print("-" * 60)
        print(description)
        print("-" * 60)
        
        return description
    
    def identify_potential_accident_timestamps(self, description: str) -> List[float]:
        """從描述中識別潛在事故時間戳 - 改進版本"""
        
        accident_keywords = [
            'collision', 'crash', 'accident', 'impact', 'hit', 'struck',
            'emergency', 'brake', 'swerve', 'skid', 'damage', 'debris',
            'stopped suddenly', 'near miss', 'close call', 'dangerous',
            'unsafe', 'violation', 'speed', 'reckless', 'collides', 'colliding',
            'fall over', 'falls', 'falling', 'crashes', 'crashed', 'fallen', 'fell',
            'crashing', 'hits', 'damaged', 'damages'
        ]
        
        potential_timestamps = []
        
        print(f"🔍 正在分析描述中的事故關鍵字...")
        print(f"描述內容: {description}")
        
        # 改進的時間戳提取 - 支持範圍格式如 "3.4 - 6.9seconds"
        timestamp_patterns = [
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*second',  # "3.4 - 6.9seconds"
            r'(\d+(?:\.\d+)?)\s*second',                        # "5 seconds"
            r'at\s+(\d+(?:\.\d+)?)',                            # "at 5"
            r'^(\d+(?:\.\d+)?)',                                # 行開頭的數字
        ]
        
        # 檢查整個描述中的事故關鍵字
        description_lower = description.lower()
        has_accident_keywords = any(keyword in description_lower for keyword in accident_keywords)
        
        if has_accident_keywords:
            print(f"✅ 在描述中發現事故關鍵字")
            
            # 提取所有時間戳
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, description_lower)
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            # 範圍格式 (start, end)
                            start_time = float(match[0])
                            end_time = float(match[1])
                            # 添加範圍內的整數時間戳
                            for t in range(int(start_time), int(end_time) + 2):  # +2 確保包含結束時間
                                potential_timestamps.append(float(t))
                                print(f"🚨 從範圍 {start_time}-{end_time} 提取時間戳: {t}s")
                        else:
                            # 單一時間戳
                            timestamp = float(match)
                            potential_timestamps.append(timestamp)
                            potential_timestamps.append(timestamp + 1)  # 也添加下一秒
                            print(f"🚨 發現潛在事故時間戳: {timestamp}s")
                    except (ValueError, TypeError):
                        continue
        
        # 如果沒有找到特定時間戳但有事故關鍵字，使用預設時間範圍
        if has_accident_keywords and not potential_timestamps:
            print(f"⚠️ 發現事故關鍵字但無法提取時間戳，使用預設範圍")
            potential_timestamps = [3.0, 4.0, 5.0, 6.0, 7.0]  # 預設範圍
        
        # 移除重複並排序
        potential_timestamps = sorted(list(set(potential_timestamps)))
        
        print(f"📊 最終提取的時間戳: {potential_timestamps}")
        
        return potential_timestamps
    
    def extract_frames_from_video(self, video_path: str, timestamps: List[float], output_dir: str) -> List[str]:
        """從視頻中提取特定時間戳的影格"""
        
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 錯誤: 無法打開視頻 {video_path}")
            return frame_paths
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for timestamp in timestamps:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                frame_filename = f"frame_{timestamp:.1f}s.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                print(f"📸 提取影格 {timestamp}s -> {frame_filename}")
            else:
                print(f"⚠️ 提取影格失敗 {timestamp}s")
        
        cap.release()
        return frame_paths
    
    def step2_verify_accident_frames(self, frame_paths: List[str]) -> List[Dict]:
        """步驟 2: 分析靜態交通事故圖像以驗證是否真的顯示事故"""
        print(f"🖼️ 步驟 2: 驗證 {len(frame_paths)} 張潛在事故影格...")
        
        if not frame_paths:
            print("❌ 沒有影格需要驗證")
            return []
        
        verified_accidents = []
        
        for i, frame_path in enumerate(frame_paths):
            if not os.path.exists(frame_path):
                print(f"⚠️ 影格未找到: {frame_path}")
                continue
            
            print(f"\n🔍 分析影格 {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"image_path": frame_path}},
                        {
                            "type": "text", 
                            "text": (
                                "Analyze this traffic image carefully. Is there a traffic accident happening in this picture?\n\n"
                                "Look for:\n"
                                "- Vehicle collisions or damage\n"
                                "- Vehicles in abnormal positions\n"
                                "- Debris on the road\n"
                                "- Emergency vehicles or personnel\n"
                                "- Vehicles that appear to have crashed\n\n"
                                "Answer with:\n"
                                "1. YES or NO - Is there definitely a traffic accident?\n"
                                "2. Confidence level (1-10)\n"
                                "3. Detailed description of what you see\n"
                                "4. Specific evidence of accident (if any)"
                            )
                        }
                    ]
                }
            ]
            
            # 生成詳細的文本分析輸出
            print(f"🔄 正在分析圖像...")
            analysis = self.infer(conversation)
            
            # 顯示靜態圖像分析的文本輸出
            print(f"📋 靜態圖像分析結果:")
            print("=" * 80)
            print(analysis)
            print("=" * 80)
            
            # 解析響應以確定是否真的是事故
            is_accident = self.parse_accident_verification(analysis)
            
            # 從檔案名稱提取時間戳
            timestamp_match = re.search(r'frame_(\d+(?:\.\d+)?)s', frame_path)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else 0.0
            
            result = {
                "timestamp": timestamp,
                "frame_path": frame_path,
                "is_accident": is_accident["confirmed"],
                "confidence": is_accident["confidence"],
                "analysis": analysis,
                "evidence": is_accident["evidence"]
            }
            
            if is_accident["confirmed"]:
                print(f"✅ 事故確認 在 {timestamp}s (信心度: {is_accident['confidence']}/10)")
                verified_accidents.append(result)
                
                # 將確認的事故影格複製到 accident_frames 目錄
                final_frame_path = os.path.join(ACCIDENT_FRAMES_DIR, os.path.basename(frame_path))
                if frame_path != final_frame_path:
                    shutil.copy(frame_path, final_frame_path)
                    result["frame_path"] = final_frame_path
                    print(f"📁 影格已複製到: {final_frame_path}")
            else:
                print(f"❌ 未檢測到事故 在 {timestamp}s (信心度: {is_accident['confidence']}/10)")
        
        print(f"\n🎯 步驟 2 完成: {len(verified_accidents)} 張確認事故影格，共 {len(frame_paths)} 張")
        return verified_accidents
    
    def parse_accident_verification(self, analysis: str) -> Dict:
        """解析事故驗證響應"""
        
        analysis_lower = analysis.lower()
        
        # 尋找是/否響應
        is_confirmed = False
        if 'yes' in analysis_lower[:200]:  # 檢查前200個字符
            is_confirmed = True
        elif 'no' in analysis_lower[:200]:
            is_confirmed = False
        
        # 提取信心等級 (1-10)
        confidence_patterns = [r'confidence[:\s]*(\d+)', r'confidence[:\s]+level[:\s]*(\d+)']
        confidence = 5  # 預設值
        for pattern in confidence_patterns:
            match = re.search(pattern, analysis_lower)
            if match:
                confidence = int(match.group(1))
                break
        
        # 提取證據
        evidence_keywords = ['collision', 'damage', 'crash', 'debris', 'emergency', 'abnormal', 'impact']
        evidence = [keyword for keyword in evidence_keywords if keyword in analysis_lower]
        
        result = {
            "confirmed": "yes" in analysis_lower[:200],
            "confidence": confidence,
            "evidence": evidence
        }

        
        print(f"📊 解析結果: confirmed={result['confirmed']}, confidence={confidence}, evidence={evidence}")
        
        return result
    
    def step3_parse_to_json(self, description: str, verified_accidents: List[Dict]) -> Dict:
        """步驟 3: 將描述解析為 JSON 格式"""
        print("📄 步驟 3: 將描述解析為 JSON 格式...")
        
        # 從自然語言描述中解析時間戳
        parsed_timeline = self.parse_timestamped_description(description)
        
        # 創建綜合 JSON 結構
        result = {
            "video_analysis": {
                "raw_description": description,
                "parsed_timeline": parsed_timeline,
                "total_frames_analyzed": len(parsed_timeline)
            },
            "accident_detection": {
                "verified_accidents": verified_accidents,
                "accident_count": len(verified_accidents),
                "accident_timestamps": [acc["timestamp"] for acc in verified_accidents]
            },
            "summary": {
                "has_accidents": len(verified_accidents) > 0,
                "accident_severity": self.assess_accident_severity(verified_accidents),
                "analysis_confidence": self.calculate_overall_confidence(verified_accidents)
            }
        }
        
        print(f"✅ 步驟 3 完成: 創建了包含 {len(parsed_timeline)} 個時間線條目的 JSON 結構")
        return result
    
    def parse_timestamped_description(self, description: str) -> List[Dict]:
        """將自然語言描述解析為帶時間戳的條目"""
        
        timeline = []
        lines = description.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 提取時間戳和描述
            timestamp_patterns = [
                r'at\s+(\d+(?:\.\d+)?)\s*second[s]?[:\s]*(.+)',
                r'(\d+(?:\.\d+)?)\s*second[s]?[:\s]*(.+)',
                r'(\d+(?:\.\d+)?)[:\s]+(.+)',
            ]
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, line.lower())
                if match:
                    try:
                        timestamp = float(match.group(1))
                        description_text = match.group(2).strip()
                        
                        # 清理描述
                        description_text = re.sub(r'^[:\-\s]+', '', description_text)
                        
                        if description_text:
                            timeline.append({
                                "timestamp": timestamp,
                                "description": description_text,
                                "raw_line": line
                            })
                            break
                    except (ValueError, IndexError):
                        continue
        
        return sorted(timeline, key=lambda x: x["timestamp"])
    
    def assess_accident_severity(self, verified_accidents: List[Dict]) -> str:
        """評估檢測到的事故的整體嚴重程度"""
        
        if not verified_accidents:
            return "none"
        
        high_severity_keywords = ['collision', 'crash', 'severe', 'major', 'emergency']
        medium_severity_keywords = ['impact', 'damage', 'minor', 'fender']
        
        severity_scores = []
        
        for accident in verified_accidents:
            analysis_lower = accident["analysis"].lower()
            score = accident["confidence"]
            
            if any(keyword in analysis_lower for keyword in high_severity_keywords):
                score += 3
            elif any(keyword in analysis_lower for keyword in medium_severity_keywords):
                score += 1
            
            severity_scores.append(score)
        
        if not severity_scores:
            return "low"
        
        avg_score = sum(severity_scores) / len(severity_scores)
        
        if avg_score >= 8:
            return "high"
        elif avg_score >= 6:
            return "medium"
        else:
            return "low"
    
    def calculate_overall_confidence(self, verified_accidents: List[Dict]) -> float:
        """改良版：連續化的事故置信度"""
        if not verified_accidents:
            # 若無事故但有檢測到可疑文字，可以加入輕微分數
            return 0.1  # 模型完全沒看到事故時為 0.1
        confidences = [acc["confidence"] for acc in verified_accidents]
        avg_conf = sum(confidences) / len(confidences)
        # 平滑映射：允許低信心也有區分度
        normalized = min(max(avg_conf / 12.0, 0.0), 1.0)
        return normalized
    
    def analyze_traffic_video(self, video_path: str, max_frames: int = 30) -> Dict:
        """實施三步驟分析過程的主函數"""
        print(f"🚀 開始三步驟交通事故分析: {video_path}")
        print("="*60)
        
        try:
            # 步驟 1: 以自然語言分析視頻
            description = self.step1_analyze_video_natural_language(video_path, max_frames)
            
            # 識別潛在事故時間戳
            potential_timestamps = self.identify_potential_accident_timestamps(description)
            
            if not potential_timestamps:
                print("ℹ️ 在視頻描述中未檢測到潛在事故")
                result = self.step3_parse_to_json(description, [])
            else:
                # 為潛在事故提取影格
                temp_dir = "temp_frames"
                os.makedirs(temp_dir, exist_ok=True)
                
                print(f"📸 正在提取 {len(potential_timestamps)} 個時間戳的影格...")
                frame_paths = self.extract_frames_from_video(video_path, potential_timestamps, temp_dir)
                
                if frame_paths:
                    # 步驟 2: 驗證事故影格
                    verified_accidents = self.step2_verify_accident_frames(frame_paths)
                    
                    # 清理臨時目錄
                    for frame_path in frame_paths:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    print("❌ 未能提取任何影格")
                    verified_accidents = []
                
                # 步驟 3: 解析為 JSON
                result = self.step3_parse_to_json(description, verified_accidents)
            
            print("="*60)
            print("🎉 分析完成!")
            print(f"📊 檢測到的事故總數: {result['accident_detection']['accident_count']}")
            print(f"📈 整體信心度: {result['summary']['analysis_confidence']:.2f}")
            print(f"⚠️ 嚴重程度: {result['summary']['accident_severity']}")
            
            return result
            
        except Exception as e:
            print(f"❌ 分析過程中出錯: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "video_path": video_path,
                "success": False
            }


def extract_frames_per_second(video_path):
    """從影片中每秒提取一張影格並儲存為圖片 (FRAMES_DIR)"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ 無法開啟影片檔案進行影格擷取")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📸 正在從 {video_path} 擷取【每秒 1 張】影格（持續時間: {duration} 秒, FPS: {fps}）...")

    frame_count = 0
    second = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if fps > 0 and frame_count % fps == 0 and second <= duration:
            frame_path = os.path.join(FRAMES_DIR, f"{second}.jpg")
            cv2.imwrite(frame_path, frame)
            second += 1

        frame_count += 1

    cap.release()


def extract_frames_every_0_1s(video_path, output_dir):
    """⭐ 新增：從影片中每 0.1 秒提取一張影格並儲存為圖片 (output_dir)"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 無法開啟影片檔案進行影格擷取（0.1 秒）")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("❌ FPS 讀取失敗，無法進行每 0.1 秒擷取")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"📸 正在從 {video_path} 擷取【每 0.1 秒 1 張】影格（持續時間: {duration} 秒, FPS: {fps}）...")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    # 每 0.1 秒的 frame 步長
    step = max(1, int(round(fps * 0.1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            frame_path = os.path.join(output_dir, f"{t:.1f}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_idx += 1

    cap.release()


def main():
    """主函數 - 實施完整的三步驟過程"""
    
    # 檢查視頻文件是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 找不到視頻文件: {VIDEO_PATH}")
        print("請確保視頻文件存在於當前目錄中")
        return
    
    # 初始化分析器
    analyzer = TrafficAccidentAnalyzer(model, processor)
    
    # 先提取所有影格（用於後續步驟）
    print("📸 提取所有影格...")
    extract_frames_per_second(VIDEO_PATH)                 # 原本就有：每秒一張，存到 FRAMES_DIR
    extract_frames_every_0_1s(VIDEO_PATH, FRAMES_DIR_01) # ⭐ 新增：每 0.1 秒一張，存到 FRAMES_DIR_01
    
    # 運行三步驟分析
    result = analyzer.analyze_traffic_video(VIDEO_PATH, max_frames=30)
    
    # 保存結果到 JSON 文件
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    with open(STATIC_ANALYSIS_LOG, "w", encoding="utf-8") as f:
        json.dump({
            "static_analysis_results": result.get("accident_detection", {}).get("verified_accidents", []),
            "total_verified": result.get("accident_detection", {}).get("accident_count", 0)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📄 結果已保存至: {JSON_OUTPUT_PATH}")
    print(f"📄 靜態分析日誌已保存至: {STATIC_ANALYSIS_LOG}")
    
    # 顯示摘要
    if "error" not in result:
        print("\n" + "="*50)
        print("📋 分析摘要")
        print("="*50)
        print(f"視頻: {VIDEO_PATH}")
        print(f"檢測到的事故: {result['accident_detection']['accident_count']}")
        
        if result['accident_detection']['verified_accidents']:
            print("\n🚨 驗證的事故:")
            for i, accident in enumerate(result['accident_detection']['verified_accidents'], 1):
                print(f"  {i}. 時間: {accident['timestamp']}s | 信心度: {accident['confidence']}/10")
                print(f"     影格: {accident['frame_path']}")
        
        print(f"\n嚴重程度: {result['summary']['accident_severity'].upper()}")
        print(f"信心度: {result['summary']['analysis_confidence']:.1%}")


if __name__ == "__main__":
    main()
