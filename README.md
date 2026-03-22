# Sensable: Visual Navigation Assistant

Sensable is an intelligent navigation assistant designed to help visually impaired individuals navigate their surroundings safely. Using real-time object detection combined with monocular depth estimation, it identifies obstacles and provides immediate audio + haptic guidance.

---

## 🌟 Key Features

- **Real-time Object Detection** — [YOLOv11s](https://docs.ultralytics.com/) with ByteTrack object tracking (no repeat alerts for the same object)
- **Monocular Depth Estimation** — Intel MiDaS DPT_Hybrid detects walls and solid obstacles that YOLO might miss
- **Offline Audio Guidance** — Instant, zero-latency voice instructions via `pyttsx3` (no internet required)
- **GPU Accelerated** — Full FP16 (half-precision) inference on CUDA GPUs; falls back to CPU automatically
- **Parallel Processing** — YOLO and MiDaS run simultaneously for lower latency
- **5-Zone Depth Map** — Far-Left / Left / Center / Right / Far-Right coverage for precise directional guidance
- **Haptic Feedback** — Phone vibration on wall warnings and close obstacles (mobile browsers)
- **Smart Priority Queue** — `Stop` → `Very Close` → `Close` → `Path Clear` — no audio spam
- **CLAHE Low-light Enhancement** — Works better in dim conditions
- **Live Confidence Slider** — Tune detection sensitivity without restarting

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- **NVIDIA GPU** (optional but strongly recommended) — RTX 30/40 series for full performance

### 2. Clone the Repository

```bash
git clone https://github.com/hadifirdouskhan6881/Sensable.git
cd Sensable
```

### 3. Create a Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Users (NVIDIA):** Install the CUDA-enabled PyTorch for maximum performance:
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```
> Replace `cu124` with your CUDA version (check with `nvidia-smi`).

### 5. Generate SSL Certificates (Required for Phone Camera Access)

Browsers only allow camera access over HTTPS when not on `localhost`. Run once to generate a self-signed cert:

```bash
python generate_cert.py
```

The app auto-detects `cert.pem` / `key.pem` and enables HTTPS.

---

## 🏃 Running the App

```bash
python app.py
```

> **First run:** The app will automatically download `yolo11s.pt` (~22MB) and the MiDaS DPT_Hybrid weights (~400MB). This takes a few minutes once.

**Access the app:**
- Local browser: `https://127.0.0.1:5000`
- From your phone (same Wi-Fi): `https://<your-PC-IP>:5000`
  - Accept the self-signed certificate warning in your browser
  - Go to **App** from the landing page, tap to start

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask + Flask-CORS |
| Object Detection | Ultralytics YOLOv11s + ByteTrack |
| Depth Estimation | Intel MiDaS DPT_Hybrid |
| Deep Learning | PyTorch (CUDA 12.4) |
| Audio TTS | pyttsx3 (offline) |
| Image Processing | OpenCV + CLAHE |
| Frontend | HTML5 / CSS3 / Vanilla JS |

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| Camera not accessible from phone | Make sure you're using `https://` and accepted the cert warning |
| `CUDA: False` in logs | Run the `pip install --force-reinstall torch ...` GPU command in step 4 |
| Slow / laggy on CPU | Expected — GPU strongly recommended. Lower confidence slider helps |
| First run takes forever | MiDaS DPT_Hybrid is ~400MB, download only happens once |
| pyttsx3 no audio | On Linux, install `espeak`: `sudo apt install espeak` |
| Wall warnings too sensitive | Increase `WALL_THRESHOLD` on line ~228 of `app.py` (default: 600) |

---

## 📁 Project Structure

```
Sensable/
├── app.py                # Main Flask server + AI pipeline
├── generate_cert.py      # One-time SSL cert generator
├── requirements.txt      # Python dependencies
├── templates/
│   ├── landing.html      # Landing / intro page
│   └── index.html        # Main app interface
└── audio_cache/          # Auto-generated TTS audio files (gitignored)
```

---

## 🔄 Git Workflow

- **Before coding:** Pull latest changes in VS Code Source Control (`↓ Pull`)
- **After changes:** Save → Source Control → write a commit message → Commit → Sync Changes (`↑`)