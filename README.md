# 🦮 Visual Navigation Assistant for Visually Impaired Users

A Python-based web application that provides real-time audio guidance for visually impaired users using computer vision and YOLOv8 object detection. This educational prototype demonstrates how AI can enhance accessibility through spatial awareness and text-to-speech feedback.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Testing & Evaluation](#testing--evaluation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project is an educational prototype developed as a university assignment to demonstrate assistive technology for visually impaired users. The system uses a device's camera to detect obstacles in real-time and provides spatial audio guidance, helping users navigate safely through their environment.

**Key Objectives:**
- Provide real-time object detection and spatial awareness
- Generate contextual audio feedback for safe navigation
- Create an accessible, browser-based solution requiring no special hardware
- Demonstrate practical application of computer vision for accessibility

---

## ✨ Features

### Core Functionality
- ✅ **Real-time Object Detection** - Uses YOLOv8 nano model for fast, accurate detection
- ✅ **Spatial Positioning** - Identifies object locations (left, ahead, right)
- ✅ **Audio Guidance** - Text-to-speech feedback for navigation
- ✅ **Browser-based** - Accessible via web browser on any device
- ✅ **Mobile Support** - Works on smartphones and tablets
- ✅ **HTTPS Security** - SSL certificate support for secure camera access
- ✅ **Low Latency** - Optimized for quick response (~2 second intervals)

### Accessibility Features
- 🎯 **High-contrast UI** - Easy to see for users with partial vision
- ⌨️ **Keyboard Shortcuts** - Space to start/stop, M to mute
- 🔊 **Clear Audio Feedback** - Concise, actionable guidance
- 📱 **Touch-friendly Controls** - Large buttons for easy interaction

---

## 🎥 Demo

### Example Scenarios

**Indoor Navigation:**
```
User approaches doorway with chair nearby
→ Audio: "Chair on your left. Door ahead"

Person walks toward user
→ Audio: "Person ahead, very close"

User navigates around table
→ Audio: "Dining table on your right"
```

### Supported Objects
- 👤 People
- 🪑 Furniture (chairs, tables, couches, beds)
- 🚪 Doors
- 🚲 Vehicles (bicycles, cars, motorcycles)
- 🎒 Common items (backpacks, bottles, bags)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User's Device                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Web Browser (Chrome/Safari)             │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │         Camera Feed (Live Video)          │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │     JavaScript (Capture & Send Frames)    │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓ ↑
                    HTTP/HTTPS
                         ↓ ↑
┌─────────────────────────────────────────────────────────┐
│                   Flask Backend Server                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         YOLOv8 Object Detection Model           │  │
│  │    (Detects objects + calculates positions)     │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Position Analysis Algorithm             │  │
│  │       (Left / Ahead / Right detection)          │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Text-to-Speech Engine (gTTS)              │  │
│  │      (Generates audio guidance)                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
                  Audio Feedback
                         ↓
                    👤 User
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **Webcam or smartphone camera**
- **Modern web browser** (Chrome recommended)
- **Internet connection** (for initial setup and TTS)

### Step 1: Clone Repository

```bash
git clone https://github.com/hadifirdouskhan881/visual-navigation-assistant.git
cd visual-navigation-assistant
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- Flask 3.0.0
- Flask-CORS 4.0.0
- Ultralytics 8.1.0 (YOLOv8)
- OpenCV-Python 4.8.1.78
- gTTS 2.5.0
- NumPy 1.24.3
- PyTorch 2.1.0
- PyOpenSSL (for HTTPS)

### Step 4: Generate SSL Certificates (Optional but Recommended)

```bash
pip install pyopenssl
python generate_cert.py
```

This creates `cert.pem` and `key.pem` for HTTPS support.

### Step 5: Download YOLOv8 Model

The YOLOv8 nano model will download automatically on first run. Alternatively:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 💻 Usage

### Running on Local Computer

1. **Start the Flask server:**

```bash
python app.py
```

2. **Open browser and navigate to:**

```
http://localhost:5000
```

Or with HTTPS:

```
https://localhost:5000
```

3. **Grant camera permissions** when prompted

4. **Click "Start Navigation"** or press `Space`

5. **Move objects in front of camera** to test detection

### Running on Mobile Phone

#### Via WiFi Hotspot (Recommended for Demos):

1. **Enable hotspot** on your phone

2. **Connect laptop** to phone's hotspot

3. **Find laptop's IP address:**

**Windows:**
```bash
ipconfig
```
Look for IPv4 Address (e.g., `192.168.43.156`)

**Mac/Linux:**
```bash
ifconfig
```

4. **Run Flask app** on laptop:
```bash
python app.py
```

5. **On phone browser**, go to:
```
https://192.168.43.156:5000
```
(Replace with your actual IP)

6. **Accept security warning:**
   - Tap "Advanced"
   - Tap "Proceed to site (unsafe)"

7. **Grant camera permission** and start navigation!

### Keyboard Shortcuts

- **Space** - Start/Stop navigation
- **M** - Mute/Unmute audio

---

## 🔧 How It Works

### 1. Camera Capture
- Browser accesses device camera via WebRTC
- JavaScript captures frames every 2 seconds
- Frames converted to base64 and sent to backend

### 2. Object Detection
- Flask backend receives image data
- YOLOv8 nano model processes frame
- Detects objects with >50% confidence
- Filters for navigation-relevant objects

### 3. Position Analysis
Frame divided into three zones:
- **Left third** (0-33%) → "on your left"
- **Center third** (33-67%) → "ahead"
- **Right third** (67-100%) → "on your right"

### 4. Distance Estimation
Based on bounding box size relative to frame:
- **Large box (>15% of frame)** → "very close"
- **Medium box (8-15%)** → "close"
- **Small box (<8%)** → detected but not emphasized

### 5. Audio Generation
- Text-to-speech converts detections to audio
- Messages prioritize closest/largest objects
- Audio cached for performance
- Prevents overlapping speech

### 6. User Feedback
- Audio played through browser
- Visual display shows detections
- Real-time status updates

---

## 📁 Project Structure

```
visual-navigation-assistant/
│
├── app.py                      # Main Flask application
├── generate_cert.py            # SSL certificate generator
├── requirements.txt            # Python dependencies
│
├── templates/
│   └── index.html             # Frontend interface
│
├── audio_cache/               # Generated audio files (auto-created)
│
├── cert.pem                   # SSL certificate (generated)
├── key.pem                    # SSL private key (generated)
│
├── venv/                      # Virtual environment (not in repo)
│
└── README.md                  # This file
```

---

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **YOLOv8 (Ultralytics)** - Object detection model
- **OpenCV** - Image processing
- **gTTS** - Text-to-speech synthesis
- **NumPy** - Numerical operations
- **PyTorch** - Deep learning framework

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **JavaScript (ES6)** - Camera access and API calls
- **WebRTC** - Camera stream handling

### APIs & Libraries
- **getUserMedia API** - Camera access
- **Web Audio API** - Audio playback
- **Fetch API** - Backend communication
- **Google Text-to-Speech** - Audio generation

---

## 📊 Testing & Evaluation

### Test Scenarios

#### 1. Object Detection Accuracy
- Test with various objects (chairs, people, cups, etc.)
- Measure detection confidence scores
- Evaluate false positive/negative rates

#### 2. Position Detection
- Test left/center/right classification
- Verify accuracy at different distances
- Test with multiple simultaneous objects

#### 3. Audio Guidance Quality
- Clarity of speech
- Timeliness of feedback
- Relevance of information

#### 4. Performance Metrics
- Detection latency (target: <2 seconds)
- Frame processing time
- Audio generation speed

#### 5. User Experience
- Interface usability
- Control responsiveness
- Overall accessibility

### Expected Results

| Metric | Target | Actual |
|--------|--------|--------|
| Detection Accuracy | >85% | Test in controlled environment |
| Position Accuracy | >90% | Verify with ground truth |
| Response Latency | <2s | Measure end-to-end |
| Audio Clarity | High | User feedback |

---

## ⚠️ Limitations

### Technical Limitations
- **No depth sensing** - Distance estimated from object size, not actual depth
- **2D detection only** - Cannot detect stairs, holes, or height changes
- **Indoor lighting required** - Poor performance in low light
- **Processing delay** - 2-second interval between detections
- **Single camera view** - Limited field of view

### Environmental Constraints
- **Good lighting needed** - Requires adequate illumination
- **Clear line of sight** - Obstacles may hide other objects
- **Internet dependency** - TTS requires internet connection (first time)
- **Device limitations** - Performance varies by hardware

### Safety Considerations
- **NOT a replacement** for traditional mobility aids (white cane, guide dog)
- **Prototype only** - Not certified as medical device
- **Supervised use** - Requires oversight during testing
- **Controlled environments** - Tested in known, safe spaces only

---

## 🚀 Future Enhancements

### Planned Improvements
1. **Depth Sensing** - Integration with LiDAR or stereo cameras
2. **Offline Mode** - Local TTS for no internet dependency
3. **Path Planning** - Suggested navigation routes
4. **Obstacle Avoidance** - Predictive guidance
5. **Wearable Integration** - Smart glasses or headset support
6. **Multi-language Support** - TTS in multiple languages
7. **Custom Object Training** - User-specific object recognition
8. **Cloud Deployment** - Web-hosted version for wider access

### Advanced Features
- **3D Mapping** - Environment reconstruction
- **Indoor Navigation** - GPS-denied positioning
- **Social Features** - Share safe routes
- **Voice Commands** - Hands-free control
- **Haptic Feedback** - Vibration alerts for obstacles

---

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and feedback are welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Performance optimization
- Additional object classes
- UI/UX improvements
- Documentation enhancements
- Bug fixes
- Accessibility features

---

## 📄 License

This project is developed for educational purposes as a university assignment. 

**Usage Guidelines:**
- ✅ Educational use
- ✅ Research purposes
- ✅ Learning and experimentation
- ❌ Commercial use without permission
- ❌ Medical device claims
- ❌ Production deployment without proper testing

---

## 🙏 Acknowledgments

### Technologies & Libraries
- **Ultralytics** - YOLOv8 object detection framework
- **Flask** - Web framework by Pallets
- **Google TTS** - Text-to-speech service
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning framework

### Resources
- YOLOv8 Documentation: https://docs.ultralytics.com/
- Flask Documentation: https://flask.palletsprojects.com/
- Web Accessibility Guidelines: https://www.w3.org/WAI/

### Inspiration
- Assistive technology research community
- Accessibility advocacy organizations
- Open-source computer vision projects

---

## 📞 Contact & Support

**Developer:** Hadi Firdous Khan  
**GitHub:** [@hadifirdouskhan881](https://github.com/hadifirdouskhan881)  
**Repository:** [visual-navigation-assistant](https://github.com/hadifirdouskhan881/visual-navigation-assistant)

### Issues & Bug Reports
Please use GitHub Issues for:
- Bug reports
- Feature requests
- Technical questions
- Documentation improvements

---

## 📚 References

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.

2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub repository.

3. World Health Organization. (2023). Blindness and vision impairment. Retrieved from https://www.who.int/

4. Web Content Accessibility Guidelines (WCAG) 2.1. (2018). W3C Recommendation.

---

## 🎓 Educational Context

**Project Type:** University Assignment - Assistive Technology  
**Course:** Computer Vision / AI for Accessibility  
**Objective:** Demonstrate practical application of AI for social good  
**Status:** Educational Prototype - Not for Production Use  

**Learning Outcomes:**
- Computer vision implementation
- Real-time object detection
- Web application development
- Accessibility-focused design
- User-centered prototyping

---

## ⚡ Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/hadifirdouskhan881/visual-navigation-assistant.git
cd visual-navigation-assistant
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Generate SSL certificates
python generate_cert.py

# Run application
python app.py

# Access in browser
# https://localhost:5000
```

---

## 🎯 Project Goals Achieved

✅ Real-time object detection using YOLOv8  
✅ Spatial position analysis (left/ahead/right)  
✅ Text-to-speech audio guidance  
✅ Browser-based accessible interface  
✅ Mobile device compatibility  
✅ Low-latency performance (<2s)  
✅ HTTPS security implementation  
✅ Comprehensive documentation  

---

**Made with ❤️ for accessibility and inclusion**

*This project demonstrates how technology can enhance independence and quality of life for visually impaired individuals.*