# Sensable: Visual Navigation Assistant

This is a Flask-based web application designed to act as a visual navigation assistant for visually impaired users. It uses the YOLOv8 object detection model to identify objects in a video stream and provides audio guidance using text-to-speech.

## Prerequisites

Before running the application, make sure you have Python installed on your computer.

## Setup Instructions

1. **Open your terminal or command prompt** and navigate to the project folder (`c:\Users\hadif\OneDrive\Desktop\Upload to Git` or wherever you cloned the repository).

2. **(Optional but recommended) Create a virtual environment:**
   This keeps the project's dependencies separate from your main Python installation.
   ```bash
   python -m venv venv
   ```
   Activate it:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. **Install the required packages:**
   Run the following command to install all the necessary libraries (Flask, YOLO, OpenCV, etc.):
   ```bash
   pip install -r requirements.txt
   ```
   *(If you don't have a `requirements.txt` file, you can install them manually by running: `pip install flask flask-cors ultralytics gtts opencv-python numpy`)*

4. **SSL Certificates (For HTTPS - Optional but needed for camera access on some browsers):**
   If you want to run the app with HTTPS, you can use the included `generate_cert.py` to create local certificates:
   ```bash
   python generate_cert.py
   ```
   This will generate `cert.pem` and `key.pem` files.

## How to Run

1. Start the Flask server by running the main application script:
   ```bash
   python app.py
   ```

2. When the server starts, it will print some text in the console. It will also automatically download the YOLOv8 nano model (`yolov8n.pt`) the first time you run it.

3. Open your web browser and go to the address shown in the terminal.
   - If running with HTTP: `http://localhost:5000` or `http://127.0.0.1:5000`
   - If running with HTTPS (SSL certificates): `https://localhost:5000` or `https://127.0.0.1:5000`

4. Grant your browser permission to access your webcam when prompted. The application will start analyzing the feed and providing audio guidance for objects like people, chairs, tables, doors, etc.

## Troubleshooting
- **No camera access?** Browsers often restrict camera access strictly to `https://` sites (unless you are using exactly `localhost`). If you are accessing the app from another device on the network, make sure to generate the SSL certificates and use `https://`.
- **Missing YOLO model?** The `ultralytics` library should download `yolov8n.pt` automatically, but if it fails, ensure you have an active internet connection on the first run.