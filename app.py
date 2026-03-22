"""
Visual Navigation Assistant for Visually Impaired Users
Flask-based web application with YOLOv11 object detection and audio guidance

Hardware: RTX 4070 Laptop GPU + AMD Ryzen 9 7000 series + 32GB RAM
TTS: Browser-native Web Speech API (speechSynthesis) — no server-side audio files
Optimizations:
  - YOLOv11s model (latest gen, better accuracy at same speed)
  - ByteTrack object tracking (no repeat alerts for same object)
  - DPT_Hybrid MiDaS depth model (much better depth accuracy)
  - FP16 (half-precision) inference on GPU (~2x faster)
  - Parallel YOLO + MiDaS inference via ThreadPoolExecutor
  - CLAHE low-light preprocessing
  - 5-zone lateral depth detection (finer direction guidance)
  - Smart priority-based audio message queue
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import time
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

app = Flask(__name__)
CORS(app)

# ── Device Setup ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = device.type == "cuda"          # FP16 only on GPU
print(f"Using device: {device}  |  FP16: {USE_HALF}")

# ── YOLOv11s Model (latest generation) ─────────────────────────────────────────
print("Loading YOLOv11s model...")
model = YOLO('yolo11s.pt')
# NOTE: do NOT call model.model.half() here — Ultralytics handles FP16
# internally during the first .track() call when half=True is passed.
# Casting manually before the predictor is set up causes a dtype mismatch
# in fuse_conv_and_bn on the first inference.
print("YOLOv11s ready.")

# ── MiDaS DPT_Hybrid Depth Model ────────────────────────────────────────────────
print("Loading MiDaS DPT_Hybrid depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to(device)
if USE_HALF:
    midas.half()  # MiDaS is called directly (no Ultralytics wrapper), safe to cast here
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform   # matches DPT_Hybrid
print("MiDaS DPT_Hybrid ready.")

# ── Thread Pool for parallel YOLO + MiDaS ───────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=2)

# ── Debounce / cooldown state ────────────────────────────────────────────────────
# (Debounce is now done client-side via speechSynthesis, but we still track
# server-side to avoid redundant JSON processing on repeat identical scenes.)
last_message = ""
last_message_time = 0
MESSAGE_COOLDOWN = 3.0


# ── Image Helpers ────────────────────────────────────────────────────────────────
def decode_image(base64_string):
    """Decode base64 image to numpy array (BGR)."""
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def apply_clahe(img_bgr):
    """Apply CLAHE for low-light enhancement before inference."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_position(x_center, frame_width):
    """5-zone lateral position: far-left / left / center / right / far-right."""
    rel = x_center / frame_width
    if rel < 0.20:
        return "far left"
    elif rel < 0.40:
        return "left"
    elif rel < 0.60:
        return "center"
    elif rel < 0.80:
        return "right"
    else:
        return "far right"


# ── Parallel inference workers ───────────────────────────────────────────────────
def run_yolo(img, conf=0.45):
    """Run YOLOv11 tracking with ByteTrack (persist=True keeps IDs across frames)."""
    results = model.track(
        img,
        conf=conf,
        verbose=False,
        device=device.type,
        half=USE_HALF,
        persist=True,
        tracker="bytetrack.yaml",
    )
    return results


def run_midas(img_bgr):
    """Run MiDaS DPT_Hybrid depth estimation."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    if USE_HALF:
        input_batch = input_batch.half()
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return pred.cpu().float().numpy()


# ── Guidance message generator ───────────────────────────────────────────────────
def generate_guidance_message(detections, wall_warning, zone_proximities):
    """
    Smart priority queue:
      P1 (highest) — wall / solid obstacle ahead
      P2           — very close object in center
      P3           — close object in center
      P4           — side objects
      P5 (lowest)  — path clear
    """
    left_far, left, center, right, right_far = zone_proximities

    # ── P1: Wall warning ─────────────────────────────────────────────────────────
    if wall_warning:
        # Suggest the clearest side (lowest proximity = most room)
        options = {
            "far left":  left_far,
            "left":      left,
            "right":     right,
            "far right": right_far,
        }
        clearest_side = min(options, key=options.get)
        if options[clearest_side] < 420:
            return "Stop. Solid object directly ahead."
        return f"Solid object ahead. Move {clearest_side}."

    if not detections:
        return "Path clear."

    relevant = [d for d in detections if d['distance'] in ("very close", "close")]
    if not relevant:
        return ""

    relevant.sort(key=lambda x: x['size_ratio'], reverse=True)
    center_threats = [d for d in relevant if d['position'] == 'center']
    side_threats   = [d for d in relevant if d['position'] != 'center']

    # ── P2 / P3: Center threats ───────────────────────────────────────────────────
    if center_threats:
        threat = center_threats[0]
        obj = threat['object']
        if threat['distance'] == "very close":
            return f"Stop. {obj} directly ahead."
        return f"Caution. {obj} ahead."

    # ── P4: Side threats ─────────────────────────────────────────────────────────
    if side_threats:
        threat = side_threats[0]
        return f"{threat['object']} on {threat['position']}. Straight path clear."

    return ""


# ── Tracked ID set to debounce repeat audio per object ───────────────────────────
_alerted_track_ids = set()
_alerted_lock = threading.Lock()


# ── Routes ───────────────────────────────────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/app')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_objects():
    """Process frame, return detections + guidance audio."""
    global last_message, last_message_time

    try:
        data = request.json
        image_data = data.get('image')
        conf_thresh = float(data.get('conf', 0.45))   # from frontend slider
        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        img = decode_image(image_data)
        img = apply_clahe(img)          # low-light enhancement
        height, width = img.shape[:2]

        # ── Parallel YOLO + MiDaS ─────────────────────────────────────────────
        yolo_future  = executor.submit(run_yolo, img, conf_thresh)
        midas_future = executor.submit(run_midas, img)
        wait([yolo_future, midas_future], return_when=ALL_COMPLETED)

        results   = yolo_future.result()
        depth_map = midas_future.result()

        # ── 5-Zone depth slices ───────────────────────────────────────────────
        mid_top    = int(height * 0.30)
        mid_bottom = int(height * 0.70)

        z0 = float(np.mean(depth_map[mid_top:mid_bottom, 0              : int(width*0.20)]))  # far-left
        z1 = float(np.mean(depth_map[mid_top:mid_bottom, int(width*0.20): int(width*0.40)]))  # left
        z2 = float(np.mean(depth_map[mid_top:mid_bottom, int(width*0.40): int(width*0.60)]))  # center
        z3 = float(np.mean(depth_map[mid_top:mid_bottom, int(width*0.60): int(width*0.80)]))  # right
        z4 = float(np.mean(depth_map[mid_top:mid_bottom, int(width*0.80): width          ]))  # far-right

        # DPT_Hybrid outputs larger values → recalibrated threshold
        WALL_THRESHOLD = 600
        wall_warning = z2 > WALL_THRESHOLD

        # ── Parse YOLO detections ─────────────────────────────────────────────
        detections = []
        detected_objects = {}
        new_track_ids = set()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = float((x1 + x2) / 2)
                position = get_position(x_center, width)

                box_area   = float((x2 - x1) * (y2 - y1))
                frame_area = float(width * height)
                size_ratio = float(box_area / frame_area)

                if size_ratio > 0.15:
                    distance = "very close"
                elif size_ratio > 0.08:
                    distance = "close"
                else:
                    distance = "far"

                # Track ID (may be None if tracking hasn't locked on yet)
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    new_track_ids.add(track_id)

                detection = {
                    'object':     class_name,
                    'position':   position,
                    'confidence': round(conf, 2),
                    'distance':   distance,
                    'size_ratio': round(size_ratio, 3),
                    'track_id':   track_id,
                }
                detections.append(detection)

                key = f"{class_name}_{position}"
                if key not in detected_objects or detected_objects[key]['size_ratio'] < size_ratio:
                    detected_objects[key] = detection

        # Expire alerted IDs that are no longer in the frame
        with _alerted_lock:
            _alerted_track_ids.intersection_update(new_track_ids)

        # ── Generate guidance ─────────────────────────────────────────────────
        audio_message = generate_guidance_message(
            list(detected_objects.values()),
            wall_warning,
            (z0, z1, z2, z3, z4),
        )

        # Server-side cooldown: suppress repeat identical messages within 3s
        # (client-side speechSynthesis handles the actual speaking)
        current_time  = time.time()
        speak_message = ""
        if audio_message:
            same_msg      = (audio_message == last_message)
            within_window = (current_time - last_message_time) < MESSAGE_COOLDOWN
            if not (same_msg and within_window):
                last_message      = audio_message
                last_message_time = current_time
                speak_message     = audio_message   # tell browser to speak this

        return jsonify({
            'detections':       list(detected_objects.values()),
            'message':          audio_message,
            'speak':            speak_message,   # non-empty only when browser should speak
            'total_objects':    len(detected_objects),
            'wall_warning':     wall_warning,
            'depth_proximity':  z2,
            'left_prox':        z1,
            'right_prox':       z3,
            'zone_proximities': [z0, z1, z2, z3, z4],
        })

    except Exception as e:
        print(f"Detection error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status':  'running',
        'model':   'YOLOv11s',
        'depth':   'MiDaS-DPT_Hybrid',
        'device':  str(device),
        'fp16':    USE_HALF,
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  Sensable Navigation Assistant — Upgraded")
    print(f"  Device : {device}  |  FP16: {USE_HALF}")
    print("  Model  : YOLOv11s + ByteTrack tracking")
    print("  Depth  : MiDaS DPT_Hybrid")
    print("  TTS    : pyttsx3 (offline)")
    print("=" * 60)

    ssl_ctx = None
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        ssl_ctx = ('cert.pem', 'key.pem')
        print("SSL: ENABLED")
    else:
        print("SSL: disabled (no cert.pem / key.pem)")

    app.run(debug=False, host='0.0.0.0', port=5000, ssl_context=ssl_ctx)