# Sensable: Visual Navigation Assistant

Sensable is an intelligent navigation assistant designed to help visually impaired individuals navigate their surroundings safely. By combining real-time object detection with advanced depth estimation, the application identifies obstacles and provides immediate audio feedback to guide the user.

## 🌟 Key Features

- **Real-time Object Detection:** Uses **YOLOv8** to identify common objects like chairs, people, doors, and tables.
- **Smart Obstacle Detection:** Leverages **MiDaS (Monocular Depth Estimation)** to detect generic barriers like walls and solid objects, even if they aren't recognized by the object detection model.
- **Audio Guidance:** Converts detection results into clear, concise voice instructions using **Google Text-to-Speech (gTTS)**.
- **Prioritized Feedback:** Intelligently filters audio alerts to focus on immediate threats in the user's path, reducing "audio spam."
- **Low Hardware Barrier:** Runs on a standard laptop (GPU recommended) and can be accessed via a mobile browser over a local network.

---

## 🚀 Getting Started (Beginner Friendly)

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

Make sure you have the following installed:
- **Python (3.8 or higher):** [Download Python](https://www.python.org/downloads/)
- **Git:** [Download Git](https://git-scm.com/downloads)

### 2. Clone the Repository

Open your terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
git clone https://github.com/hadifirdouskhan6881/Sensable.git
cd Sensable
```

### 3. Create a Virtual Environment (Recommended)

This keeps the project's libraries organized and separate from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the necessary AI models and web libraries with one command:

```bash
pip intall requirements.txt

#or

pip install flask flask-cors ultralytics gtts opencv-python numpy torch torchvision timm
```

### 5. (Optional) Setup SSL for Camera Access

Browsers require **HTTPS** to allow camera access on devices other than `localhost`. To enable this on your local network:

1. Run the certificate generator (if you have one) or generate them manually:
   ```bash
   openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
   ```
2. The app will automatically detect `cert.pem` and `key.pem` and switch to HTTPS mode.

---

## 🏃 Running the App

1. **Start the server:**
   ```bash
   python app.py
   ```
   *Note: On the first run, the app will download the YOLOv8 (`yolov8s.pt`) and MiDaS models. This might take a few minutes depending on your internet speed.*

2. **Access the interface:**
   - Look at the terminal output for the address (usually `https://127.0.0.1:5000`).
   - Open that link in your browser.
   - If you are on the same Wi-Fi, you can open the **Network IP** (e.g., `https://192.168.1.X:5000`) on your smartphone.

3. **Grant camera permissions** when prompted, and you're ready to go!

---

## 🛠 Tech Stack

- **Backend:** Flask (Python)
- **Computer Vision:** Ultralytics YOLOv8, Intel MiDaS
- **Deep Learning:** PyTorch
- **Audio:** Google Text-to-Speech (gTTS)
- **Frontend:** HTML5, CSS3, JavaScript (Webcam streaming)

---

## ⚠️ Troubleshooting

- **"Camera not found"**: Ensure you are using `https://` if accessing from a phone.
- **Slow Performance**: The app performs best with a dedicated GPU (e.g., RTX 30/40 series). If running on a CPU, expect a slight delay in audio feedback.
- **Model Download Errors**: Ensure you have a stable internet connection for the first startup.
