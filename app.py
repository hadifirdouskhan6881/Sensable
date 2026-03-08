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

# Load YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Priority objects for navigation
PRIORITY_OBJECTS = {
    'person', 'chair', 'couch', 'bed', 'dining table', 'door', 
    'stairs', 'bicycle', 'car', 'motorcycle', 'bench', 'backpack',
    'handbag', 'suitcase', 'bottle', 'cup'
}

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
        return "ahead"

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
                
                # Only process priority objects
                if class_name not in PRIORITY_OBJECTS:
                    continue
                
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
                    distance = "ahead"
                
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
    """Generate concise audio guidance message"""
    if not detections:
        return "Path clear"
    
    # Sort by size (closest first)
    detections.sort(key=lambda x: x['size_ratio'], reverse=True)
    
    # Take top 3 most relevant objects
    top_detections = detections[:3]
    
    messages = []
    for det in top_detections:
        obj = det['object']
        pos = det['position']
        dist = det['distance']
        
        if dist == "very close":
            messages.append(f"{obj.capitalize()} {pos}, very close")
        else:
            messages.append(f"{obj.capitalize()} {pos}")
    
    return ". ".join(messages)

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
    return jsonify({'status': 'running', 'model': 'YOLOv8n'})

if __name__ == '__main__':
    print("Starting Visual Navigation Assistant...")
    print("Make sure YOLOv8 nano model is downloaded (yolov8n.pt)")
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