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
from io import BytesIO
from gtts import gTTS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load YOLOv8 small model for improved detection
model = YOLO('yolov8s.pt')

# Audio cache directory
AUDIO_DIR = 'audio_cache'
os.makedirs(AUDIO_DIR, exist_ok=True)

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
        # Create unique filename based on text hash
        filename = f"{hash(text) % 10000000}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # Check if audio already cached
        if not os.path.exists(filepath):
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)
        
        return filename
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Process frame and return detected objects with positions"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode image
        img = decode_image(image_data)
        height, width = img.shape[:2]
        
        # Run YOLOv8 detection
        results = model(img, conf=0.5, verbose=False)
        
        detections = []
        detected_objects = {}
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Detect all objects YOLO is capable of recognizing
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = float((x1 + x2) / 2)
                y_center = float((y1 + y2) / 2)
                
                # Determine position
                position = get_position(x_center, width)
                
                # Calculate distance approximation (based on box size)
                box_area = float((x2 - x1) * (y2 - y1))
                frame_area = float(width * height)
                size_ratio = float(box_area / frame_area)
                
                # Categorize distance
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
                
                # Group by object type and position for cleaner feedback
                key = f"{class_name}_{position}"
                if key not in detected_objects or detected_objects[key]['size_ratio'] < size_ratio:
                    detected_objects[key] = detection
        
        # Generate audio message
        audio_message = generate_guidance_message(list(detected_objects.values()))
        audio_file = None
        
        if audio_message:
            audio_file = generate_audio(audio_message)
        
        return jsonify({
            'detections': list(detected_objects.values()),
            'message': audio_message,
            'audio_file': audio_file,
            'total_objects': len(detected_objects)
        })
    
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_guidance_message(detections):
    """Generate concise audio guidance message with path routing"""
    if not detections:
        return "Path clear ahead."
    
    # Sort by size (closest first)
    detections.sort(key=lambda x: x['size_ratio'], reverse=True)
    
    blocked_areas = set()
    for det in detections:
        if det['distance'] in ["very close", "close"]:
            blocked_areas.add(det['position'])
            
    # Determine path recommendation
    if "center" in blocked_areas:
        if "left" not in blocked_areas and "right" not in blocked_areas:
            action = "Obstacles ahead, move left or right."
        elif "left" not in blocked_areas:
            action = "Obstacle ahead, move left."
        elif "right" not in blocked_areas:
            action = "Obstacle ahead, move right."
        else:
            action = "Stop! Path blocked."
    else:
        action = "Path clear ahead."
        
    messages = [action]
    
    # Tell them about the top 2 closest objects
    limit = 2
    for det in detections[:limit]:
        obj = det['object']
        pos = det['position']
        dist = det['distance']
        
        if dist == "very close":
            messages.append(f"{obj} {pos}, very close.")
        elif dist == "close":
            messages.append(f"{obj} {pos}.")
            
    return " ".join(messages)

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
    
    # Check if SSL certificates exist
    import os
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("Running with HTTPS (SSL enabled)")
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
    else:
        print("Running with HTTP (no SSL certificates found)")
        print("To enable HTTPS, generate certificates with:")
        print("openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
        app.run(debug=True, host='0.0.0.0', port=5000)