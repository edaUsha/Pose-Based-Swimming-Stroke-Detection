# 🏊 Pose-Based Swimming Stroke Detection

**Real-time swimming stroke classification system using YOLOv8 Pose estimation — end-to-end ML project from dataset collection to production deployment.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue?style=flat-square)](https://huggingface.co/spaces/edaUsha/swimming-stroke-detector)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-green?style=flat-square)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=flat-square)](https://streamlit.io)

---

## 💼 Executive Summary

This project demonstrates **end-to-end machine learning expertise** by building a production-ready system that automatically identifies swimming strokes from video using computer vision. The system combines pose estimation, deep learning, and real-time inference to achieve practical results on a challenging visual recognition task.

**Try the live demo:** Upload a swimming video → model identifies the stroke (Freestyle, Backstroke, Breaststroke, or Butterfly) with frame-level annotations and confidence scores.

---

## 🎯 Problem Statement & Solution

### Challenge
Swimming stroke classification requires understanding complex body dynamics and movements. Manual frame-by-frame analysis is time-consuming; automated detection enables applications in:
- **Sports analytics** — technique analysis and performance tracking
- **Swimming education** — real-time feedback for coaches and athletes
- **Data collection** — at-scale labeling of swimming videos

### Solution Approach
A single-stage pose detection pipeline optimized for real-time inference:
1. **Pose Extraction** — YOLOv8 Pose detects swimmer and extracts 17 body keypoints per frame
2. **Stroke Classification** — Temporal pattern matching maps pose sequences to stroke types
3. **Frame Caching** — Efficient detection every N frames to optimize speed on CPU
4. **Annotation** — Output video with bounding boxes, skeleton overlay, and predictions

---

## 🧠 Technical Architecture

```
Video Input (mp4/avi)
    ↓
[YOLOv8n-pose] — Swimmer detection + 17 keypoint extraction
                  (inference every N frames for speed optimization)
    ↓
[Stroke Pattern Matching] — Map pose keypoints to stroke classification
    ↓
[Frame Caching Strategy] — Reuse predictions for N frames to maintain efficiency
    ↓
[Temporal Aggregation] — Dominant stroke across all frames
    ↓
Annotated Output Video
  ├─ Bounding box with stroke label
  ├─ Skeleton overlay (17 keypoints)
  ├─ Confidence score per detection
  └─ Responsive text layout (portrait/landscape aware)
```

---

## 📊 Model Performance & Results

| Metric | Value |
|---|---|
| **Precision** | 70.1% |
| **Recall** | 62.3% |
| **mAP@50** | 62.3% |
| **Model Size** | 6 MB |
| **Inference Speed** | ~8-12 FPS on CPU (inference every 4 frames) |
| **Max Processing** | 150 frames (~5 seconds at 30fps) per video |

**Base Model:** YOLOv8n-pose (pretrained on COCO dataset)  
**Fine-tuned on:** Custom swimming stroke dataset (698 annotated images)

---

## 🗂️ Dataset & Training

| Aspect | Details |
|---|---|
| **Source** | Self-collected competitive swimmer footage |
| **Annotation** | Manual labeling using Roboflow annotation tool |
| **Classes** | 4 swimming strokes (Freestyle, Backstroke, Breaststroke, Butterfly) |
| **Training Images** | 698 images with bounding box + pose keypoint annotations |
| **Transfer Learning** | COCO-pretrained YOLOv8n-pose |

**Key insight:** Custom dataset collection ensures domain-specific training data, enabling the model to learn stroke-specific pose patterns effectively.

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Pose Detection** | YOLOv8n-pose | Lightweight, real-time capable; 'n' variant optimized for edge deployment |
| **Video Processing** | OpenCV | Frame extraction, resizing, encoding; industry standard |
| **Web Interface** | Streamlit | Rapid prototyping, interactive UI, live frame preview streaming |
| **Video Re-encoding** | FFmpeg | H.264 encoding for browser compatibility |
| **Deployment** | Hugging Face Spaces | Free GPU tier, automated CI/CD, shareable public link |
| **Language** | Python 3.10 | ML ecosystem, rapid iteration |

---

## 🚀 Usage & Deployment

### Local Development
```bash
# Clone repository
git clone https://github.com/edaUsha/Pose-Based-Swimming-Stroke-Detection.git
cd Pose-Based-Swimming-Stroke-Detection

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

Access at `http://localhost:8501`

### Production Deployment
- **Hosted on:** Hugging Face Spaces (GPU-backed or CPU)
- **Interface:** Streamlit web app with drag-and-drop video upload
- **Output:** Annotated video download + live frame preview during processing
- **Video formats:** MP4, AVI, MOV, MKV

---

## 📁 Project Structure

```
Pose-Based-Swimming-Stroke-Detection/
│
├── app.py                      # Streamlit application (video processing, UI, inference pipeline)
├── best_4classes.pt            # Fine-tuned YOLOv8 model weights (~6MB)
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for deployment)
└── README.md                   # Documentation
```

---

## ✨ Key Features & Implementation Highlights

### 1. **CPU-Optimized Inference Pipeline**
- **Adaptive frame skipping** — Process every N frames (configurable via UI: 1, 2, 3, or 4)
- **Scalable resolution** — Inference scale slider (0.4x to 1.0x) for speed/accuracy tradeoff
- **Frame caching** — Reuse predictions across frames to maximize throughput on CPU

### 2. **Robust Pose Detection**
- 17-keypoint skeleton extraction per frame
- Confidence-based filtering for low-quality detections
- Lightweight model (6MB) suitable for edge deployment

### 3. **Responsive UI Design**
- Portrait/landscape-aware layout (different annotation styles for each)
- Real-time frame preview during processing (~4 previews/second)
- Live status updates (FPS, frame count, detected stroke)
- Color-coded stroke legend with semantic highlighting

### 4. **Production-Ready Video Output**
- OpenCV MP4v codec initial encoding
- Optional FFmpeg H.264 re-encoding for universal browser compatibility
- Annotated frames with bounding boxes, stroke labels, confidence scores
- Download-ready output files

### 5. **Streamlit-Native Architecture**
- Session state management for video persistence
- Cached model loading (load only once per session)
- Interactive settings panel with preset options
- Zero-friction file upload and download

---

## ⚠️ Limitations & Scope

| Limitation | Impact | Rationale |
|---|---|---|
| **Limited dataset size** | 698 images; small for general deep learning | Transfer learning from COCO reduces data requirements; project scope |
| **Side-view camera dependency** | Top/front angles underperform | Competitive swimming videos are typically side-view; documented limitation |
| **CPU inference speed** | 8-12 FPS limits real-time applications | Free tier deployment; GPU deployment available but requires paid hosting |
| **Competitive strokes only** | Non-standard strokes not recognized | Model scope is intentional; extensible architecture for future classes |
| **Fixed max frame count** | 150 frames (~5 seconds) per video | Prevents memory overflow on free tier; can be adjusted for paid deployments |

---

## 🔮 Future Enhancements

**Immediate (High Impact):**
- [ ] Expand dataset with diverse swimmers, pool environments, lighting conditions
- [ ] Add stroke technique scoring (arm entry angle, kick frequency, body alignment)
- [ ] Real-time webcam inference for coaching applications

**Medium-term:**
- [ ] Multi-swimmer detection in single frame
- [ ] Transition-phase detection (between strokes)
- [ ] Batch API endpoint for high-throughput inference

**Long-term:**
- [ ] Mobile app deployment (TFLite or CoreML)
- [ ] Integration with sports analytics platforms
- [ ] Swimmer re-identification and tracking across videos

---

## 💡 Skills Demonstrated

✅ **Computer Vision & Deep Learning**
- Transfer learning from foundation models (COCO-pretrained YOLOv8)
- Real-time pose estimation optimization (frame skipping, resolution scaling)
- Model performance evaluation (Precision, Recall, mAP metrics)

✅ **ML Engineering & MLOps**
- End-to-end ML lifecycle (problem definition → data collection → annotation → training → deployment)
- Model fine-tuning and validation on custom datasets
- Production deployment on cloud infrastructure

✅ **Software Engineering**
- Clean, modular architecture (separate concerns: video processing, inference, rendering)
- Full-stack application (backend ML pipeline + interactive frontend)
- Video codec handling and browser compatibility optimization

✅ **Product & UX Thinking**
- User-centric design (intuitive settings, live feedback, responsive layouts)
- Performance optimization for constrained environments (CPU inference)
- Clear documentation of model limitations and use cases

---

## 📚 References & Resources

- **YOLOv8 Documentation:** [Ultralytics](https://docs.ultralytics.com/tasks/pose/)
- **Streamlit Docs:** [Streamlit.io](https://docs.streamlit.io/)
- **Dataset Annotation:** [Roboflow](https://roboflow.com/)
- **OpenCV Video Processing:** [OpenCV Docs](https://docs.opencv.org/)

---

**Built as a portfolio project showcasing Computer Vision, Deep Learning, and Production ML capabilities.**
