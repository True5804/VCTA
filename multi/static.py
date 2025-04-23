import os
import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2wVWTbD6xxHqAe1v8EIC"
)
model_id = "crash-car-detection/3"

def run_static(output_dir):
    input_dir = os.path.join(output_dir, "accident_frames")
    bbox_json = os.path.join(output_dir, "bbox/json")
    bbox_img = os.path.join(output_dir, "bbox/images")
    os.makedirs(bbox_json, exist_ok=True)
    os.makedirs(bbox_img, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(input_dir, img_name)
        print(f"🔍 Analyzing {img_name}")

        result = CLIENT.infer(img_path, model_id=model_id)
        if not result["predictions"]:
            print(f"❌ No accident in {img_name}")
            continue

        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for pred in result["predictions"]:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=5)
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="red")
        image.save(os.path.join(bbox_img, img_name))

        with open(os.path.join(bbox_json, f"{os.path.splitext(img_name)[0]}.json"), "w") as f:
            json.dump(result, f, indent=2)
