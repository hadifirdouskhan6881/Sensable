"""
Visual Navigation Assistant for Visually Impaired Users
Flask-based web application with YOLOv8 object detection and audio guidance
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from gtts import gTTS
import os
import time
import torch

app = Flask(__name__)
CORS(app)

# Load YOLOv8 small model
model = YOLO('yolov8s.pt')

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MiDaS depth model
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Audio cache directory
AUDIO_DIR = 'audio_cache'
os.makedirs(AUDIO_DIR, exist_ok=True)

# State tracking for debouncing audio
last_message = ""
last_message_time = 0
MESSAGE_COOLDOWN = 3.0


def decode_image(base64_string):
    """Decode base64 image to numpy array"""
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_position(x_center, frame_width):
    """Determine object position: left, center, or right"""
    left_threshold = frame_width * 0.33
    right_threshold = frame_width * 0.67

    if x_center < left_threshold:
        return "left"
    elif x_center > right_threshold:
        return "right"
    else:
        return "center"


def generate_audio(text):
    """Generate audio file from text using gTTS"""
    try:
        filename = f"{hash(text) % 10000000}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(filepath):
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)

        return filename
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None


@app.route('/')
def landing():
    """Serve landing page"""
    return render_template('landing.html')


@app.route('/app')
def index():
    """Serve main app page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_objects():
    """Process frame and return detected objects with positions"""
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        img = decode_image(image_data)
        height, width = img.shape[:2]

        # 1. YOLO detection
        results = model(img, conf=0.5, verbose=False, device=device.type)

        detections = []
        detected_objects = {}

        # 2. MiDaS depth estimation
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()

        center_left = int(width * 0.40)
        center_right = int(width * 0.60)
        center_top = int(height * 0.3)
        center_bottom = int(height * 0.7)

        center_depth_slice = depth_map[center_top:center_bottom, center_left:center_right]
        average_center_proximity = float(np.mean(center_depth_slice))

        wall_warning = False
        if average_center_proximity > 450:
            wall_warning = True

        left_slice = depth_map[center_top:center_bottom, 0:int(width * 0.25)]
        right_slice = depth_map[center_top:center_bottom, int(width * 0.75):width]
        avg_left_prox = float(np.mean(left_slice))
        avg_right_prox = float(np.mean(right_slice))

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = float((x1 + x2) / 2)

                position = get_position(x_center, width)

                box_area = float((x2 - x1) * (y2 - y1))
                frame_area = float(width * height)
                size_ratio = float(box_area / frame_area)

                if size_ratio > 0.15:
                    distance = "very close"
                elif size_ratio > 0.08:
                    distance = "close"
                else:
                    distance = "far"

                detection = {
                    'object': class_name,
                    'position': position,
                    'confidence': round(conf, 2),
                    'distance': distance,
                    'size_ratio': round(size_ratio, 3)
                }

                detections.append(detection)

                key = f"{class_name}_{position}"
                if key not in detected_objects or detected_objects[key]['size_ratio'] < size_ratio:
                    detected_objects[key] = detection

        global last_message, last_message_time
        audio_message = generate_guidance_message(
            list(detected_objects.values()),
            wall_warning,
            avg_left_prox,
            avg_right_prox
        )
        audio_file = None

        current_time = time.time()

        if audio_message:
            if audio_message == last_message and (current_time - last_message_time) < MESSAGE_COOLDOWN:
                audio_file = None
            else:
                last_message = audio_message
                last_message_time = current_time
                audio_file = generate_audio(audio_message)

        return jsonify({
            'detections': list(detected_objects.values()),
            'message': audio_message,
            'audio_file': audio_file,
            'total_objects': len(detected_objects),
            'wall_warning': wall_warning,
            'depth_proximity': average_center_proximity,
            'left_prox': avg_left_prox,
            'right_prox': avg_right_prox
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


def generate_guidance_message(detections, wall_warning=False, left_prox=0, right_prox=0):
    """Generate concise audio guidance prioritizing the front path"""
    if wall_warning:
        if left_prox < 400 and right_prox > 400:
            return "Solid object ahead. Move Left."
        elif right_prox < 400 and left_prox > 400:
            return "Solid object ahead. Move Right."
        else:
            return "Stop. Solid object directly ahead."

    if not detections:
        return "Path clear."

    relevant_detections = [d for d in detections if d['distance'] in ["very close", "close"]]

    if not relevant_detections:
        return ""

    relevant_detections.sort(key=lambda x: x['size_ratio'], reverse=True)

    center_dangers = [d for d in relevant_detections if d['position'] == 'center']
    side_dangers = [d for d in relevant_detections if d['position'] != 'center']

    if center_dangers:
        primary_threat = center_dangers[0]
        obj = primary_threat['object']

        if primary_threat['distance'] == "very close":
            return f"Stop. {obj} ahead."
        else:
            return f"Caution. {obj} ahead."

    if side_dangers:
        primary_threat = side_dangers[0]
        obj = primary_threat['object']
        pos = primary_threat['position']
        return f"{obj} on {pos}, straight path clear."

    return ""


@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='audio/mpeg')
    return "Audio not found", 404


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'running', 'model': 'YOLOv8s'})


if __name__ == '__main__':
    print("Starting Visual Navigation Assistant...")
    print("Make sure YOLOv8 small model is downloaded (yolov8s.pt)")
    print("Installing required packages: flask, flask-cors, ultralytics, gtts, opencv-python")

    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("Running with HTTPS (SSL enabled)")
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
    else:
        print("Running with HTTP (no SSL certificates found)")
        app.run(debug=True, host='0.0.0.0', port=5000)