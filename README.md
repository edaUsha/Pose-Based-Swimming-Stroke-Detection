# 🏊 Swimming Stroke Detector

Pose-based swimming stroke detection using **YOLOv8 + MediaPipe Lite**, served as a **Streamlit web app** inside Docker.

---

## 📁 Required folder structure before building

```
📁 your_project/
  ├── app.py
  ├── Dockerfile
  ├── .dockerignore
  ├── requirements.txt
  └── best(4classes).pt      ← your YOLO weights (must be present)
```

---

## 🐳 Build the Docker image

```bash
docker build -t swimming-stroke-detector .
```

> First build takes ~5–10 minutes (downloads Python packages + PyTorch).  
> Subsequent builds are fast due to layer caching.

---

## ▶️ Run the container

```bash
docker run -p 8501:8501 swimming-stroke-detector
```

Then open your browser at:  
**http://localhost:8501**

---

## 🔄 Share the image with others

### Option A — Save to a file and share
```bash
# On your machine: save
docker save swimming-stroke-detector | gzip > swimming-stroke-detector.tar.gz

# On their machine: load and run
docker load < swimming-stroke-detector.tar.gz
docker run -p 8501:8501 swimming-stroke-detector
```

### Option B — Push to Docker Hub
```bash
docker tag swimming-stroke-detector YOUR_DOCKERHUB_USERNAME/swimming-stroke-detector
docker push YOUR_DOCKERHUB_USERNAME/swimming-stroke-detector

# Others pull and run with:
docker run -p 8501:8501 YOUR_DOCKERHUB_USERNAME/swimming-stroke-detector
```

---

## 🛑 Stop the container

```bash
# Find the running container ID
docker ps

# Stop it
docker stop <CONTAINER_ID>
```

---

## ⚙️ Notes

- The **MediaPipe Lite model** (`pose_landmarker_lite.task`) is auto-downloaded on first run inside the container and cached for the session.
- The app runs on port **8501** by default.
- No GPU required — runs on CPU.
- Tested on Python 3.10 / Ubuntu slim base.