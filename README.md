# Real-Time Multi-Modal Surveillance & Alert System

![Surveillance](https://img.shields.io/badge/System-Surveillance%20%26%20Alert-blue) ![Python](https://img.shields.io/badge/Python-3.9+-yellowgreen) ![Framework](https://img.shields.io/badge/Framework-Flask-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project is a comprehensive, real-time surveillance and alert system that leverages computer vision, sound event detection, and face recognition to provide advanced monitoring and instant notifications. It is designed to enhance traditional surveillance setups by adding intelligent detection and automated alerting capabilities, making it ideal for security, safety, and monitoring applications.

**Key Features:**
- 🔥 **Real-Time Fire Detection:** Uses YOLO-based object detection to identify fire and smoke in video streams.
- 🗣️ **Distress Sound Detection:** Detects distress sounds (e.g., screams, threats) using a deep learning audio classifier (YAMNet).
- 🕵️ **Unknown Person Detection:** Recognizes faces and flags unknown individuals, allowing for dynamic updating of the known persons list.
- 📲 **Instant Notifications:** Sends real-time alerts (text, images, audio) to user’s Telegram and WhatsApp accounts.
- 🌐 **Live Video Feed:** Provides a secure, globally accessible live video stream via a hosted link using Cloudflare Tunnel.
- 🛠️ **Extensible & Portable:** Can be deployed on Raspberry Pi or similar edge devices for on-premise surveillance upgrades.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Extending the System](#extending-the-system)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Features

### 1. Real-Time Fire Detection
- Utilizes YOLOv8 model to detect fire, smoke, and related hazards in video frames.
- Triggers alerts and saves incident snapshots.

### 2. Distress Sound Detection
- Employs YAMNet (pre-trained on AudioSet) to classify audio events.
- Detects screams, threats, and other critical sounds.
- Captures and sends last 7 seconds of detected audio as an MP3.

### 3. Unknown Person Detection
- Uses `face_recognition` to identify known and unknown faces.
- Unknown faces are saved and can be added to the known list via Telegram bot interaction.

### 4. Real-Time Notifications
- Sends alerts to Telegram (with images/audio) and WhatsApp (via local API).
- Includes incident details, location, and a live feed link.

### 5. Live Video Feed
- Streams video using Flask and OpenCV.
- Exposes the feed securely to the internet using Cloudflare Tunnel.

---

## Architecture

- **Flask Web Server:** Hosts the web interface and video stream.
- **YOLOv8:** For fire/smoke detection in video frames.
- **YAMNet:** For real-time audio event classification.
- **Face Recognition:** For identifying and managing known/unknown persons.
- **Telegram & WhatsApp Integration:** For instant notifications.
- **Cloudflare Tunnel:** For secure, public access to the local server.

Processes are managed using Python’s `multiprocessing` to ensure real-time performance and separation of concerns.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Install Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Required Packages:**
- Flask
- OpenCV (`opencv-python`)
- telepot
- pyaudio
- sounddevice
- numpy
- pandas
- keras, tensorflow
- face_recognition
- ultralytics
- pydub
- cloudflared (for tunnel)

### 3. Download Model Weights

- Place YOLO weights (e.g., `fire49mb.pt`) in the `yolo-Weights/` directory.
- Place YAMNet weights (`yamnet.h5`) and class map in `keras_yamnet/`.

### 4. Configure WhatsApp API

- Ensure your WhatsApp API server is running locally (see [send_message](app%20copy.py) for endpoint details).

### 5. Set Up Cloudflare Tunnel

Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/) and run:

```bash
cloudflared tunnel --url http://localhost:5000
```

Copy the generated public URL and update your configuration if needed.

---

## Usage

### 1. Start the Application 

```bash
python app\ copy.py
```

- The Flask server will start, and all detection/notification processes will run in parallel.
- Access the web interface at `http://localhost:5000` or via your Cloudflare Tunnel URL.

### 2. Login

- Default credentials are set in the code (see `valid_credentials` in [app copy.py](app%20copy.py)).
- Update as needed for your deployment.

### 3. Telegram & WhatsApp Alerts

- Ensure your Telegram bot token and chat IDs are set correctly.
- WhatsApp notifications require the local API server to be running.

### 4. Adding Unknown Faces

- When an unknown person is detected, you’ll receive a Telegram message with an image and an inline button.
- Click the button to add the person to the known list.

---

## Configuration

- **Detection Thresholds:** Adjust fire/sound detection thresholds in [app copy.py](app%20copy.py) as needed.
- **Notification Recipients:** Update chat IDs and phone numbers in the code.
- **Model Classes:** To change which sound events are detected, edit the `plt_classes` list in the sound detection section.

---

## Extending the System

- **Raspberry Pi Deployment:** The system can run on a Raspberry Pi with a camera and microphone for edge deployments.
- **Custom Models:** Swap in your own YOLO or audio models for different detection tasks.
- **Additional Integrations:** Add SMS, email, or other notification channels as needed.

---

## File Structure

```
├── app copy.py                # Main application entry point
├── sound_detection_pack/
│   ├── sound_event_detection.py
│   ├── plot.py
│   └── ...
├── keras_yamnet/
│   ├── yamnet.h5
│   ├── yamnet_class_map.csv
│   └── ...
├── yolo-Weights/
│   └── fire49mb.pt
├── images/                    # Saved unknown face images
├── images2/                   # Annotated images for Telegram
├── kang.jpg                   # Fire incident snapshot
├── recorded_audio.mp3         # Last detected distress audio
├── templates/
│   ├── index.html
│   └── login.html
└── ...
```

---

## Troubleshooting

- **No WhatsApp Messages:** Ensure your local WhatsApp API server is running and accessible.
- **No Telegram Alerts:** Double-check your bot token and chat IDs.
- **Cloudflare Tunnel Not Working:** Restart `cloudflared` and verify the URL.
- **Camera/Microphone Not Detected:** Check device permissions and drivers.

---

## Acknowledgements

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) for audio event detection.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection.
- [face_recognition](https://github.com/ageitgey/face_recognition) for face identification.
- [telepot](https://github.com/nickoala/telepot) for Telegram bot integration.
- [cloudflared](https://github.com/cloudflare/cloudflared) for secure tunneling.

---

## License

This project is for educational and research purposes. Please check the licenses of the included models and libraries before deploying commercially.

---

**For questions or contributions, please open an issue or pull request!**
```
