import cv2
import numpy as np
import os
import time
import urllib.request
import tempfile
import streamlit as st
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Swimming Stroke Detector",
    page_icon="🏊",
    layout="wide",
)

st.title("🏊 Swimming Stroke Detection")
st.markdown("Upload a swimming video to detect and annotate stroke types using YOLOv8 Pose.")
st.info("⚠️ For best performance upload videos under 30 seconds. Processing takes 2-3 minutes on CPU.")
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

YOLO_WEIGHTS       = "best(4classes).pt"      # Must be in the same folder as app.py
CONFIDENCE_THRESH  = 0.3
YOLO_EVERY_N_FRAMES = 4
#POSE_EVERY_N_FRAMES = 4
INFER_SCALE         = 0.5
#POSE_MODEL_PATH     = "pose_landmarker_lite.task"

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING  (cached so they load only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    yolo= YOLO(YOLO_WEIGHTS)
    return yolo
    

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

STROKE_COLORS = {
    "backstroke":   (0,   255, 0),
    "breaststroke": (255, 0,   255),
    "freestyle":    (0,   165, 255),
    "butterfly":    (255, 255, 0),
}
DEFAULT_COLOR = (0, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_color(label):
    return STROKE_COLORS.get(label.lower(), DEFAULT_COLOR)

def is_portrait(w, h):
    return h > w

def draw_transparent_box(frame, x1, y1, x2, y2, color, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def draw_yolo_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text      = f"{label.capitalize()}  {conf:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    label_y   = max(y1 - 10, 25)
    cv2.rectangle(frame,
                  (x1, label_y - text_size[1] - 6),
                  (x1 + text_size[0] + 8, label_y + 4),
                  color, -1)
    cv2.putText(frame, text, (x1 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)



def draw_info_panel(frame, stroke_label, yolo_conf):
    h, w  = frame.shape[:2]
    color = get_color(stroke_label)
    if is_portrait(w, h):
        draw_transparent_box(frame, 0, 0, w, 75, color, alpha=0.55)
        cv2.putText(frame, stroke_label.upper(),
                    (12,38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        det_c = (0,255,0) if yolo_conf>=0.7 else (0,200,255) if yolo_conf>=0.5 else (0,80,255)
        cv2.putText(frame, f"Conf: {yolo_conf:.2f}",
                    (12,65), cv2.FONT_HERSHEY_SIMPLEX, 0.72, det_c, 2, cv2.LINE_AA)
    else:
        panel_w = min(380, w-20)
        draw_transparent_box(frame, 10, h-100, panel_w, h-10, color, alpha=0.45)
        cv2.putText(frame, f"Stroke: {stroke_label.capitalize()}",
                    (20,h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        det_c = (0,255,0) if yolo_conf>=0.7 else (0,200,255) if yolo_conf>=0.5 else (0,80,255)
        cv2.putText(frame, f"Confidence: {yolo_conf:.2f}",
                    (20,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, det_c, 2, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  CORE PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def process_video(input_path: str, output_path: str, yolo_model,yolo_every, infer_scale,
                  live_placeholder, progress_bar, status_text):
    """
    Process the video frame by frame.
    - live_placeholder : st.empty() slot for the live preview image
    - progress_bar     : st.progress() widget
    - status_text      : st.empty() slot for status messages
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    MAX_FRAMES = 150   # ~5 seconds at 30fps
    total = min(total, MAX_FRAMES)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    infer_w = int(width  * infer_scale)
    infer_h = int(height * infer_scale)

    # Cached state
    cached_stroke_label = "Unknown"
    cached_yolo_conf    = 0.0
    cached_yolo_box     = None

    frame_count = 0
    fps_start   = time.time()
    fps_counter = 0
    current_fps = 0.0

    # How often to refresh the live preview (every N frames) to avoid slowdown
    PREVIEW_EVERY = max(1, int(fps // 4))   # ~4 previews per second of video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_counter += 1
        h, w, _ = frame.shape

        small_frame = cv2.resize(frame, (infer_w, infer_h))
        scale_x     = w / infer_w
        scale_y     = h / infer_h

        # ── YOLO ──────────────────────────────────────────────────────────────
        #if frame_count % YOLO_EVERY_N_FRAMES == 0:
        if frame_count % yolo_every == 0:    
            results = yolo_model(small_frame, conf=CONFIDENCE_THRESH, verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes    = results[0].boxes
                best_idx = int(boxes.conf.argmax())
                best     = boxes[best_idx]
                sx1, sy1, sx2, sy2 = map(float, best.xyxy[0])
                cached_yolo_box     = (int(sx1*scale_x), int(sy1*scale_y),
                                       int(sx2*scale_x), int(sy2*scale_y))
                cached_yolo_conf    = float(best.conf[0])
                cached_stroke_label = yolo_model.names[int(best.cls[0])]
            else:
                cached_yolo_box     = None
                cached_stroke_label = "Unknown"
                cached_yolo_conf    = 0.0

        

        # ── Draw ──────────────────────────────────────────────────────────────
        if cached_yolo_box:
            draw_yolo_box(frame, *cached_yolo_box,
                          cached_stroke_label, cached_yolo_conf,
                          get_color(cached_stroke_label))



        if cached_yolo_box:
            draw_info_panel(frame, cached_stroke_label,
                            cached_yolo_conf)
        else:
            if is_portrait(w, h):
                draw_transparent_box(frame, 0, 0, w, 55, (0,0,255), alpha=0.5)
                cv2.putText(frame, "No swimmer detected",
                            (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                draw_transparent_box(frame, 10, 10, 380, 60, (0,0,255), alpha=0.4)
                cv2.putText(frame, "No swimmer detected",
                            (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # ── FPS ───────────────────────────────────────────────────────────────
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        out.write(frame)

        # ── Live preview + progress update ────────────────────────────────────
        if frame_count % PREVIEW_EVERY == 0:
            pct = (frame_count / total) if total > 0 else 0
            progress_bar.progress(min(pct, 1.0))
            status_text.text(
                f"Processing frame {frame_count}/{total}  |  "
                f"FPS: {current_fps:.1f}  |  "
                f"Stroke: {cached_stroke_label}"
            )
            # Convert BGR → RGB for Streamlit display
            preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_placeholder.image(preview, channels="RGB", width='stretch')

    cap.release()
    out.release()
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  RE-ENCODE WITH H.264 FOR BROWSER PLAYBACK
#  mp4v written by OpenCV is not always browser-playable;
#  re-encode with ffmpeg if available, else serve as-is.
# ─────────────────────────────────────────────────────────────────────────────

def reencode_for_web(src: str, dst: str) -> str:
    """Try to re-encode to H.264/AAC MP4. Returns path to playable file."""
    import shutil
    if shutil.which("ffmpeg"):
        ret = os.system(
            f'ffmpeg -y -i "{src}" -vcodec libx264 -crf 23 -preset fast '
            f'-movflags +faststart -an "{dst}" -loglevel quiet'
        )
        return dst if ret == 0 else src
    return src   # ffmpeg not found — serve raw mp4v (may not play in all browsers)


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

# Load models once
yolo_model= load_models()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    #confidence = st.slider("Confidence Threshold", 0.1, 0.9, CONFIDENCE_THRESH, 0.05)
    #yolo_skip  = st.slider("Run YOLO every N frames", 1, 6, YOLO_EVERY_N_FRAMES)
    #pose_skip  = st.slider("Run Pose every N frames", 1, 8, POSE_EVERY_N_FRAMES)
    #infer_scale = st.slider("Inference scale", 0.25, 1.0, INFER_SCALE, 0.05,help="Lower = faster but less accurate")
    
    # Change defaults to be more CPU-friendly
    yolo_every  = st.select_slider("YOLO Frequency",  options=[1,2,3,4], value=4)
    #pose_every  = st.select_slider("Pose Frequency",  options=[1,2,3,4,5,6], value=6)
    infer_scale = st.select_slider("Inference Scale", options=[0.4,0.5,0.6,0.75,1.0], value=0.4)
    st.markdown("---")
    st.markdown("**Stroke colour legend**")
    for stroke, bgr in STROKE_COLORS.items():
        r, g, b = bgr[2], bgr[1], bgr[0]
        st.markdown(
            f'<span style="color:rgb({r},{g},{b}); font-weight:600;">■</span> '
            f'{stroke.capitalize()}',
            unsafe_allow_html=True
        )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload a Swimming Video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported formats: MP4, AVI, MOV, MKV"
)


if uploaded_file is not None:
    st.success(f"✅ Uploaded: **{uploaded_file.name}**  ({uploaded_file.size / 1e6:.1f} MB)")

    if st.button("🚀 Start Processing", type="primary", width='stretch'):

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
        tmp_raw.close()
        output_raw = tmp_raw.name

        tmp_web = tempfile.NamedTemporaryFile(delete=False, suffix="_web.mp4")
        tmp_web.close()
        output_web = tmp_web.name

        st.markdown("---")
        st.subheader("🎬 Live Preview")

        live_col, _ = st.columns([3, 1])
        with live_col:
            live_placeholder = st.empty()

        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        # Run processing

        
        success = process_video(
            input_path, output_raw,
            yolo_model,
            yolo_every, infer_scale,
            live_placeholder, progress_bar, status_text,
        )

        
        if success:
            output_web = reencode_for_web(output_raw, output_web)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # DEBUG — add these 3 lines temporarily
            import os
            st.write(f"Output file exists: {os.path.exists(output_raw)}")
            st.write(f"Output file size: {os.path.getsize(output_raw) if os.path.exists(output_raw) else 'FILE MISSING'} bytes")

            #st.markdown("---")
            #st.subheader("✅ Annotated Output Video")

            with open(output_web, "rb") as f:
                video_bytes = f.read()

            st.session_state["video_bytes"]    = video_bytes
            st.session_state["video_filename"] = f"annotated_{uploaded_file.name}"

# Cleanup temp files
        for p in [input_path, output_raw, output_web]:
            try:
                os.remove(p)
            except Exception:
                pass

        # Same indent as 'if success:' and cleanup block
        if "video_bytes" in st.session_state:
            st.markdown("---")
            st.subheader("✅ Annotated Output Video")
            st.video(st.session_state["video_bytes"])
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=st.session_state["video_bytes"],
                file_name=st.session_state["video_filename"],
                mime="video/mp4",
                width='stretch',
            )
