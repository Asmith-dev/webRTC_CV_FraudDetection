#  Real-Time WebRTC Video Analysis

A plug-and-play server & client that streams webcam video over WebRTC, runs on-the-fly face detection / head-pose / identity checks via OpenCV, MediaPipe, and `face_recognition`, and sends live analysis back to the browser.

<p align="center">
  <img src="docs/demo.gif" width="650" alt="Demo recording">
</p>

---

## ⚡️ Features
- **Low-latency WebRTC** using `aiortc`
- **Face detection & bounding boxes** (OpenCV/MediaPipe)
- **Head-pose & attention estimation**
- **Identity change alert** (face re-ID)
- **Modern web dashboard** with live overlays & metrics
- Pluggable ML pipeline – add your own models in one place
- Runs anywhere - Windows, macOS, Linux, Raspberry Pi

## 🗂️ Project Layout
```
webRTC/
├── backend_server.py     # Backend control
├── ml_detector.py          # Classic ML pipeline
├── en_ml.py                # Enhanced MediaPipe/Face-ID pipeline
├── en_webrtc_client.html   # Polished front-end
├── requirements.txt        # Python deps
└── README.md               # You're here
```

## 🚀 Quick Start
1. Clone & enter the repo  
   ```bash
   git clone https://github.com/Asmith-dev/webRTC_CV_FraudDetection.git
   ```

2. Create a virtual environment (recommended)  
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Start backend server (WebSocket , webRTC and ML)  
   ```bash
   python backend_server.py       # default ws://0.0.0.0:8765
   ```

6. Open the client  
   * **Option A:** double-click `client_browser.html`  
   * **Option B:** serve it: `python -m http.server 8000` → <http://localhost:8000/client_browser.html>

7. Click **"Start"**, allow camera access, enjoy live metrics.

## 🔧 Configuration
All knobs are centralized in `config.py` / `ml_detector.py` / `en_ml.py`.

```python
ML_CONFIG = {
    "analysis_interval": 10,     # analyse every 5th frame
    "face_detection": {"model": "mediapipe", "min_detection_confidence": 0.5},
    "head_pose": {"enabled": True, "model": "mediapipe_mesh"},
    "identity_tracking": {"enabled": True, "interval": 20, "tolerance": 0.6},
}
```

Change values → restart server → client auto-adapts.

## 📡 Protocol
| Client → Server | Purpose |
| --------------- | -------- |
| `offer`         | WebRTC SDP offer |
| `ice-candidate` | NAT traversal data |

| Server → Client | Purpose |
| --------------- | -------- |
| `answer`        | WebRTC SDP answer |
| `analysis`      | JSON analysis payload (faces, pose, etc.) |

See [`docs/message_schema.md`](docs/message_schema.md) for full schema.

## 🐳 Docker (optional)
```bash
docker build -t webrtc-ml .
docker run -p 8765:8765 webrtc-ml
```
Then open the HTML client as usual.

## 🏗️ Roadmap
- [ ] Emotion recognition
- [ ] Eye-tracking & gaze estimation
- [ ] gRPC API option
- [ ] Docker-compose for STUN/TURN integration
- [ ] CI pipeline with GitHub Actions

## 🤝 Contributing
1. Fork → 🍴  
2. `git checkout -b feature/thing`  
3. Commit + test  
4. Open a Pull Request – we ❤️ PRs!

## 🐛 Troubleshooting
| Problem | Fix |
| ------- | ---- |
| "Camera not available" | Use HTTPS or `localhost`, check browser permissions |
| WebRTC connection fails | Verify firewall and STUN reachability |
| `ImportError: cv2` | `pip install opencv-python` (ensure Python ≥3.8 for wheels) |

## 📜 License
[MIT](LICENSE)

---

> Built with ❤️ and caffeine.  Pull requests welcome! 
