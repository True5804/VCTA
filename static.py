import os
import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient

# === STEP 2: Analyze images in accident_frames using Roboflow ===

# Roboflow settings
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2wVWTbD6xxHqAe1v8EIC"
)
model_id = "crash-car-detection/3"

# Directories
input_dir = "accident_frames"
output_json_dir = "bbox/json"
output_img_dir = "bbox/images"

os.makedirs(output_json_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Analyze each image
for image_name in os.listdir(input_dir):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_dir, image_name)
    print(f"🔍 Analyzing: {image_path}")

    result = CLIENT.infer(image_path, model_id=model_id)

    if not result["predictions"]:
        print(f"❌ No accident detected in {image_name}, skipping drawing.\n")
        continue

    # Draw bounding boxes
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for pred in result["predictions"]:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        label = f'{pred["class"]} ({pred["confidence"]:.2f})'
        draw.text((left, top - 30), label, fill="red", font=font)

        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="black")

        coord_text = f"({int(x)}, {int(y)})"
        draw.text((x + 10, y), coord_text, fill="red", font=font)


    # Save result image
    output_image_path = os.path.join(output_img_dir, image_name)
    image.convert("RGB").save(output_image_path)
    print(f"✅ Saved image with bounding box to: {output_image_path}")

    # Save JSON
    json_output_path = os.path.join(output_json_dir, f"{os.path.splitext(image_name)[0]}.json")
    with open(json_output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"📝 Saved JSON result to: {json_output_path}\n")