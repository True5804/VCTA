import torch
import json
import re
import os
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Dict, Tuple, Optional
import numpy as np

# Initialize model (keep your original configuration)
model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

class TrafficAccidentAnalyzer:
    """Traffic Accident Video Analyzer with Three-Step Process"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.accident_frames_dir = "accident_frames"
        
        # Create accident_frames directory if it doesn't exist
        os.makedirs(self.accident_frames_dir, exist_ok=True)
    
    @torch.inference_mode()
    def infer(self, conversation: List[Dict], generation_params: Dict = None) -> str:
        """Basic inference function similar to example_videollama3.py"""
        
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
        """
        Step 1: Analyze the video and output description in natural language
        """
        print("🎥 Step 1: Analyzing video in natural language...")
        
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
        print(f"✅ Step 1 Complete: Generated {len(description)} characters of description")
        return description
    
    def extract_frames_from_video(self, video_path: str, timestamps: List[float], output_dir: str) -> List[str]:
        """Extract specific frames from video at given timestamps"""
        
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
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
                print(f"📸 Extracted frame at {timestamp}s -> {frame_filename}")
            else:
                print(f"⚠️ Failed to extract frame at {timestamp}s")
        
        cap.release()
        return frame_paths
    
    def identify_potential_accident_timestamps(self, description: str) -> List[float]:
        """Identify timestamps that might contain accidents from the description"""
        
        accident_keywords = [
            'collision', 'crash', 'accident', 'impact', 'hit', 'struck',
            'emergency', 'brake', 'swerve', 'skid', 'damage', 'debris',
            'stopped suddenly', 'near miss', 'close call', 'dangerous',
            'unsafe', 'violation', 'speed', 'reckless'
        ]
        
        # Extract timestamps with potential accidents
        potential_timestamps = []
        lines = description.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains accident-related keywords
            if any(keyword in line_lower for keyword in accident_keywords):
                # Extract timestamp from the line
                timestamp_patterns = [
                    r'(\d+(?:\.\d+)?)\s*second',
                    r'at\s+(\d+(?:\.\d+)?)',
                    r'^(\d+(?:\.\d+)?)',  # Line starting with number
                ]
                
                for pattern in timestamp_patterns:
                    matches = re.findall(pattern, line_lower)
                    if matches:
                        try:
                            timestamp = float(matches[0])
                            potential_timestamps.append(timestamp)
                            print(f"🚨 Potential accident at {timestamp}s: {line.strip()}")
                            break
                        except ValueError:
                            continue
        
        return sorted(list(set(potential_timestamps)))
    
    def step2_verify_accident_frames(self, frame_paths: List[str]) -> List[Dict]:
        """
        Step 2: Analyze static traffic accident images to verify if they truly show accidents
        """
        print(f"🖼️ Step 2: Verifying {len(frame_paths)} potential accident frames...")
        
        verified_accidents = []
        
        for i, frame_path in enumerate(frame_paths):
            if not os.path.exists(frame_path):
                print(f"⚠️ Frame not found: {frame_path}")
                continue
            
            print(f"🔍 Analyzing frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
            
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
            
            analysis = self.infer(conversation)
            
            # Parse the response to determine if it's truly an accident
            is_accident = self.parse_accident_verification(analysis)
            
            # Extract timestamp from filename
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
                print(f"✅ ACCIDENT CONFIRMED at {timestamp}s (confidence: {is_accident['confidence']}/10)")
                verified_accidents.append(result)
                
                # Move verified accident frame to accident_frames directory
                final_frame_path = os.path.join(self.accident_frames_dir, os.path.basename(frame_path))
                if frame_path != final_frame_path:
                    os.rename(frame_path, final_frame_path)
                    result["frame_path"] = final_frame_path
            else:
                print(f"❌ No accident detected at {timestamp}s (confidence: {is_accident['confidence']}/10)")
        
        print(f"🎯 Step 2 Complete: {len(verified_accidents)} confirmed accidents out of {len(frame_paths)} frames")
        return verified_accidents
    
    def parse_accident_verification(self, analysis: str) -> Dict:
        """Parse the accident verification response"""
        
        analysis_lower = analysis.lower()
        
        # Look for YES/NO response
        is_confirmed = False
        if 'yes' in analysis_lower[:100]:  # Check in first 100 characters
            is_confirmed = True
        
        # Extract confidence level (1-10)
        confidence_match = re.search(r'confidence[:\s]*(\d+)', analysis_lower)
        confidence = int(confidence_match.group(1)) if confidence_match else 5
        
        # Extract evidence
        evidence_keywords = ['collision', 'damage', 'crash', 'debris', 'emergency', 'abnormal position']
        evidence = [keyword for keyword in evidence_keywords if keyword in analysis_lower]
        
        return {
            "confirmed": is_confirmed and confidence >= 6,  # Require confidence >= 6
            "confidence": confidence,
            "evidence": evidence
        }
    
    def step3_parse_to_json(self, description: str, verified_accidents: List[Dict]) -> Dict:
        """
        Step 3: Parse the description into JSON format
        """
        print("📄 Step 3: Parsing description into JSON format...")
        
        # Parse timestamps from natural language description
        parsed_timeline = self.parse_timestamped_description(description)
        
        # Create comprehensive JSON structure
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
        
        print(f"✅ Step 3 Complete: JSON structure created with {len(parsed_timeline)} timeline entries")
        return result
    
    def parse_timestamped_description(self, description: str) -> List[Dict]:
        """Parse natural language description into timestamped entries"""
        
        timeline = []
        lines = description.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract timestamp and description
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
                        
                        # Clean up the description
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
        """Assess the overall severity of detected accidents"""
        
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
        """Calculate overall confidence in accident detection"""
        
        if not verified_accidents:
            return 0.0
        
        confidences = [acc["confidence"] for acc in verified_accidents]
        return sum(confidences) / len(confidences) / 10.0  # Normalize to 0-1
    
    def analyze_traffic_video(self, video_path: str, max_frames: int = 30) -> Dict:
        """
        Main function implementing the three-step analysis process
        """
        print(f"🚀 Starting three-step traffic accident analysis for: {video_path}")
        print("="*60)
        
        try:
            # Step 1: Analyze video in natural language
            description = self.step1_analyze_video_natural_language(video_path, max_frames)
            
            # Identify potential accident timestamps
            potential_timestamps = self.identify_potential_accident_timestamps(description)
            
            if not potential_timestamps:
                print("ℹ️ No potential accidents detected in video description")
                # Still parse to JSON for completeness
                result = self.step3_parse_to_json(description, [])
            else:
                # Extract frames for potential accidents
                temp_dir = "temp_frames"
                os.makedirs(temp_dir, exist_ok=True)
                
                frame_paths = self.extract_frames_from_video(video_path, potential_timestamps, temp_dir)
                
                if frame_paths:
                    # Step 2: Verify accident frames
                    verified_accidents = self.step2_verify_accident_frames(frame_paths)
                    
                    # Clean up temp directory
                    for frame_path in frame_paths:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    os.rmdir(temp_dir)
                else:
                    verified_accidents = []
                
                # Step 3: Parse to JSON
                result = self.step3_parse_to_json(description, verified_accidents)
            
            print("="*60)
            print("🎉 Analysis Complete!")
            print(f"📊 Total accidents detected: {result['accident_detection']['accident_count']}")
            print(f"📈 Overall confidence: {result['summary']['analysis_confidence']:.2f}")
            print(f"⚠️ Severity level: {result['summary']['accident_severity']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return {
                "error": str(e),
                "video_path": video_path,
                "success": False
            }

# Main execution function
def main():
    """Main function demonstrating the three-step process"""
    
    # Initialize analyzer
    analyzer = TrafficAccidentAnalyzer(model, processor)
    
    # Video path (update this to your video file)
    video_path = "night_balanced_h264.mp4"  # Change this to your video file
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable to point to your video file.")
        return
    
    # Run the three-step analysis
    result = analyzer.analyze_traffic_video(video_path, max_frames=30)
    
    # Save results to JSON file
    output_file = "traffic_accident_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Results saved to: {output_file}")
    
    # Display summary
    if "error" not in result:
        print("\n" + "="*50)
        print("📋 ANALYSIS SUMMARY")
        print("="*50)
        print(f"Video: {video_path}")
        print(f"Accidents detected: {result['accident_detection']['accident_count']}")
        
        if result['accident_detection']['verified_accidents']:
            print("\n🚨 Verified Accidents:")
            for i, accident in enumerate(result['accident_detection']['verified_accidents'], 1):
                print(f"  {i}. Time: {accident['timestamp']}s | Confidence: {accident['confidence']}/10")
                print(f"     Frame: {accident['frame_path']}")
        
        print(f"\nSeverity: {result['summary']['accident_severity'].upper()}")
        print(f"Confidence: {result['summary']['analysis_confidence']:.1%}")

if __name__ == "__main__":
    main()