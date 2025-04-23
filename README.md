# VCTA
video captioning via Gemini and YOLO

# There are 3 steps for this project :

1. video_vlm.py : Connect to Gemini by API key and generate the caption of input video in JSON format. Filter the key words and copy the specific images into directory called "accident_frames"
2. ststic.py :  Catch the images in "accident_frames". Draw the bounding box of the accident by connecting to API key of Roboflow in YOLOv5 model. Draw a red dot in the center of the bounding box as the accident dot.
3. line.py : Calculate the distance range between CCTV camera and the accident dot by the pre-drawn masks of the road in the input video file.Store the output data into "accident_distances.json"

# Frontend
accident_distances.json : Records the calculated output in queue, which means the data can be added continuously.
data.json : Location, hyperlink, and name of CCTV (the names are fake but locations and hyperlinks are real)
