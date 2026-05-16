# 🏊 Pose Based Swimming Stroke Detection

> Real-time swimming stroke classification using YOLOv8 Pose + MediaPipe — trained on a self-collected and annotated dataset.

[![Hugging Face](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/edaUsha/swimming-stroke-detector)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)

---

## 🎯 Live Demo

👉 **([https://huggingface.co/spaces/edaUsha/swimming-stroke-detector](https://huggingface.co/spaces/edaUsha/Pose-Based-Swimming-Stroke-Detection))**

Upload a swimming video → model detects stroke type frame by frame → annotated video with skeleton overlay returned.

---

## 📌 What It Does

This app automatically identifies which swimming stroke a swimmer is performing by analysing body pose landmarks extracted from video footage.

**Detects 4 strokes:**

| Stroke | Description |
|---|---|
| 🏃 Freestyle | Face-down, alternating overhead pull — fastest competitive stroke |
| 🔄 Backstroke | Face-up, alternating arm pull with flutter kick |
| 🐸 Breaststroke | Simultaneous arm sweep with frog kick — slowest competitive stroke |
| 🦋 Butterfly | Simultaneous arm recovery with dolphin kick — most technically demanding |

---

## 🧠 How It Works

```
Video Input
    ↓
YOLOv8 Pose — detects swimmer and extracts 17 body keypoints per frame
    ↓
MediaPipe Pose Landmarker — refines skeleton landmarks for accuracy
    ↓
Stroke Classification — maps pose patterns to stroke type per frame
    ↓
Aggregation — dominant stroke across all frames = Final Answer
    ↓
Annotated Video — bounding box + skeleton + stroke name + confidence baked in
```

---

## 🗂️ Dataset

- **Collected by:** Self-collected video footage of competitive swimmers
- **Annotated by:** Manually labelled using Roboflow annotation tool
- **Classes:** 4 (Freestyle, Backstroke, Breaststroke, Butterfly)
- **Train images:** 698
- **Annotation type:** Bounding box + pose keypoints

This was not a public dataset — footage was collected and annotated from scratch, making this a genuine end-to-end ML project.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Precision | 0.701 |
| Recall | 0.623 |
| mAP@50 | 0.623 |
| Model size | ~6MB |

**Base model:** YOLOv8n-pose (pretrained on COCO)
**Fine-tuned on:** Custom swimming stroke dataset

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Stroke Detection | YOLOv8n-pose (fine-tuned) |
| Pose Estimation | MediaPipe Pose Landmarker Lite |
| Video Processing | OpenCV |
| Web App | Streamlit |
| Deployment | Hugging Face Spaces |
| Language | Python 3.10 |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/edaUsha/swimming-stroke-detector
cd swimming-stroke-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Requirements:**
```
streamlit
ultralytics
opencv-python-headless
numpy
mediapipe==0.10.9
```

---

## 📁 Project Structure

```
swimming-stroke-detector/
│
├── app.py                    ← Streamlit web app
├── best(4classes).pt         ← fine-tuned YOLOv8 model weights
├── requirements.txt          ← Python dependencies
├── packages.txt              ← system dependencies for deployment
└── README.md                 ← this file
```

---

## 💡 Key Features

- **Live frame preview** — watch inference happen in real time as video processes
- **Skeleton overlay** — 17 body keypoints connected on swimmer's body
- **Bounding box** — swimmer localisation with stroke label and confidence score
- **Final Answer card** — dominant stroke across all frames with percentage agreement
- **Annotated video download** — take away the processed output
- **Adjustable settings** — confidence threshold, inference frequency, scale

---

## ⚠️ Limitations

- Works best with side-view or overhead camera angles
- Optimised for competitive swimming strokes only
- Free tier deployment runs on CPU — videos under 15 seconds recommended
- Model trained on limited dataset — accuracy improves with more training data

---

## 🔮 Future Work

- Expand dataset with more diverse swimmers and pool environments
- Add stroke technique scoring (e.g. arm entry angle, kick frequency)
- Real-time webcam inference
- GPU deployment for faster processing
- Multi-swimmer detection in the same frame

---


---

*Built as part of an AI/ML portfolio — Computer Vision · Deep Learning · Pose Estimation · Real-time Inference*
