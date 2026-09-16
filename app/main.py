import time
import os
import sys
import json
import cv2
import pandas as pd
import numpy as np
import streamlit as st

# Add app directory to path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_tracking import HandDetector
from inference import SignLanguageClassifier
from landmark_classifier import LandmarkGestureClassifier

# Streamlit Page Config
st.set_page_config(
    page_title="Sign Language Translator AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme & UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #f0f6fc;
    }
    
    .card-box {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .prediction-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b949e;
        margin-bottom: 8px;
    }
    
    .prediction-letter {
        font-size: 64px;
        font-weight: 800;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        margin: 0;
        line-height: 1.1;
    }

    .no-hand-badge {
        font-size: 20px;
        font-weight: 600;
        color: #8b949e;
        padding: 12px;
        border-radius: 8px;
        background: rgba(110, 118, 129, 0.1);
        display: inline-block;
    }

    .fps-badge {
        font-size: 14px;
        font-weight: 700;
        color: #00ff88;
        background: rgba(0, 255, 136, 0.15);
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(0, 255, 136, 0.3);
        display: inline-block;
        margin-bottom: 10px;
    }

    .warning-tip {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid rgba(210, 153, 34, 0.4);
        color: #e3b341;
        padding: 12px;
        border-radius: 8px;
        font-size: 14px;
        margin-top: 10px;
    }
    
    .sentence-display {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        background: #0d1117;
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        min-height: 64px;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_letter" not in st.session_state:
    st.session_state.last_letter = None
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "current_landmarks" not in st.session_state:
    st.session_state.current_landmarks = None


# Cache Components
@st.cache_resource
def load_components():
    detector = HandDetector(max_hands=1, detection_con=0.6, track_con=0.6)
    tflite_classifier = SignLanguageClassifier()
    landmark_classifier = LandmarkGestureClassifier()
    return detector, tflite_classifier, landmark_classifier

model_error_msg = None
try:
    detector, tflite_classifier, landmark_classifier = load_components()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error_msg = str(e)


# Header & Onboarding Help Panel
st.title("Real-Time Sign Language Translator")
st.caption("AI-Powered American Sign Language Translation with 3D Landmark Tracking & Gesture Memory")

with st.expander("How to Use & Supported ASL Sign Reference"):
    st.markdown("""
    ### User Guide
    1. **Connect & Start Camera**: Select your camera index in the sidebar and click **Start Camera**.
    2. **Position Your Hand**: Hold your hand clearly in front of the camera frame.
    3. **Hold Gesture**: When confidence exceeds threshold for consecutive frames, the letter is committed.
    4. **Remember Gestures**: Use the **Gesture Calibration & Memory** section to record and train custom signs!
    
    ---
    ### Supported ASL Alphabet Gestures (24 Classes)
    *Note: Letters **J** and **Z** are excluded as they involve motion gestures.*
    """)
    
    chart_img_path = os.path.join(os.path.dirname(__file__), "asl_chart.png")
    if os.path.exists(chart_img_path):
        st.image(chart_img_path, caption="ASL Gesture Class Chart (Sign Language MNIST)", use_container_width=True)


# Custom Gesture Recording & Memory Section
with st.expander("Gesture Calibration & Memory (Teach AI Your Signs)", expanded=False):
    st.markdown("Record custom hand gesture samples directly from your webcam so the model remembers your exact signs!")
    col_rec1, col_rec2, col_rec3 = st.columns([1.5, 1.5, 1.5])
    
    with col_rec1:
        target_sign = st.text_input("Sign Name / Letter to Remember", value="A").strip().upper()
    with col_rec2:
        if st.button("Record Current Gesture Sample", use_container_width=True):
            if st.session_state.current_landmarks is not None:
                success = landmark_classifier.add_sample(target_sign, st.session_state.current_landmarks)
                if success:
                    st.success(f"Successfully recorded sample for sign '{target_sign}'! Total samples: {len(landmark_classifier.samples)}")
                else:
                    st.error("Failed to extract landmark features.")
            else:
                st.warning("No hand currently detected in camera view. Hold your hand in camera view and try again.")
    with col_rec3:
        if st.button("Clear All Memory Samples", use_container_width=True):
            landmark_classifier.clear_samples()
            st.info("Cleared all custom gesture samples from memory.")

    st.caption(f"Memory Status: {len(landmark_classifier.samples)} recorded gesture samples in memory.")


# Check Model Error Status
if not model_loaded:
    st.error(f"Model Initialization Error: {model_error_msg}")
    st.warning("Please execute notebooks/02_train_cnn_model.ipynb to generate models/sign_cnn.tflite.")


# Sidebar Controls
with st.sidebar:
    st.header("Settings & Controls")
    
    camera_id = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)
    
    model_mode = st.radio(
        "Inference Engine Mode",
        options=["Landmark Memory Engine (3D Keypoints & Custom Memory)", "TFLite CNN (Image-based)"],
        index=0,
        help="Landmark Memory Engine uses 21 3D MediaPipe hand joint angles and is immune to background lighting."
    )
    
    confidence_thresh = st.slider(
        "Confidence Threshold", 
        min_value=0.50, max_value=0.99, value=0.75, step=0.05,
        help="Minimum confidence required to classify a gesture."
    )
    
    hold_frames = st.slider(
        "Stability Hold Frames", 
        min_value=3, max_value=15, value=5, step=1,
        help="Number of consecutive stable frames required to commit a letter."
    )
    
    show_landmarks = st.toggle("Show MediaPipe Landmarks", value=True)
    
    st.divider()
    st.subheader("Session Export")
    
    if st.session_state.session_log:
        df_log = pd.DataFrame(st.session_state.session_log)
        csv_bytes = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Session Log CSV",
            data=csv_bytes,
            file_name=f"asl_session_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("No session events logged yet.")


# Layout Columns: Video Feed (Left) & Translation Dashboard (Right)
col_video, col_dash = st.columns([1.5, 1.0], gap="medium")

with col_video:
    st.subheader("Live Camera Feed")
    fps_placeholder = st.empty()
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Start Camera", use_container_width=True, type="primary"):
            st.session_state.camera_active = True
    with col_btn2:
        if st.button("Stop Camera", use_container_width=True):
            st.session_state.camera_active = False

with col_dash:
    st.subheader("Real-Time Inference")
    pred_placeholder = st.empty()
    
    st.subheader("Sentence Builder")
    sentence_placeholder = st.empty()
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("Space", use_container_width=True):
            st.session_state.sentence += " "
    with col_s2:
        if st.button("Backspace", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
    with col_s3:
        if st.button("Clear", use_container_width=True):
            st.session_state.sentence = ""
            st.session_state.history = []

    st.subheader("Recent Detections")
    history_placeholder = st.empty()


def render_prediction_card(letter=None, confidence=0.0, top_probs=None, hand_detected=False):
    """Renders prediction display card."""
    if not hand_detected or letter is None:
        html = """
        <div class="card-box">
            <div class="prediction-title">Current Gesture</div>
            <div class="no-hand-badge">Waiting for hand gesture...</div>
        </div>
        """
    else:
        conf_pct = int(confidence * 100)
        html = f"""
        <div class="card-box">
            <div class="prediction-title">Detected Sign</div>
            <div class="prediction-letter">{letter}</div>
            <div style="margin-top: 12px; font-weight: 600; color: #8b949e;">
                Confidence: <span style="color: #00ff88;">{conf_pct}%</span>
            </div>
        </div>
        """
    pred_placeholder.markdown(html, unsafe_allow_html=True)


def update_sentence_display():
    txt = st.session_state.sentence if st.session_state.sentence else "..."
    sentence_placeholder.markdown(f'<div class="sentence-display">{txt}</div>', unsafe_allow_html=True)


def update_history_display():
    if not st.session_state.history:
        history_placeholder.caption("No signs committed yet.")
    else:
        recent = st.session_state.history[-5:][::-1]
        history_text = " • ".join([f"**{item['letter']}** ({int(item['conf']*100)}%)" for item in recent])
        history_placeholder.markdown(history_text)


# Initial UI Renders
fps_placeholder.markdown('<div class="fps-badge">FPS: 0.0</div>', unsafe_allow_html=True)
render_prediction_card(hand_detected=False)
update_sentence_display()
update_history_display()


# Camera & Inference Loop
if st.session_state.camera_active and model_loaded:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error(f"Unable to access webcam at device index {camera_id}. Verify camera is connected and not used by another app.")
        st.session_state.camera_active = False

    fps_smooth = 0.0
    prev_time = time.time()
    last_hand_time = time.time()

    while st.session_state.camera_active and cap.isOpened():
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time

        if dt > 0:
            instant_fps = 1.0 / dt
            fps_smooth = 0.85 * fps_smooth + 0.15 * instant_fps if fps_smooth > 0 else instant_fps

        ret, frame = cap.read()
        if not ret:
            st.warning("Stream interrupted or frame capture failed.")
            break

        # Flip horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Process frame with MediaPipe Hand Detector
        annotated_frame, hand_crop, bbox, raw_landmarks = detector.process_frame(frame, draw_landmarks=show_landmarks)

        # Update landmarks in session state for custom gesture recorder
        st.session_state.current_landmarks = raw_landmarks

        # Run inference if hand detected
        if hand_crop is not None or raw_landmarks is not None:
            last_hand_time = time.time()
            alert_placeholder.empty()

            letter, confidence = None, 0.0
            top_probs = {}

            # Use selected inference engine mode
            if "Landmark Memory Engine" in model_mode:
                letter, confidence = landmark_classifier.predict(raw_landmarks)
            else:
                letter, confidence, top_probs = tflite_classifier.predict(hand_crop)

            if letter is not None and confidence >= confidence_thresh:
                if letter == st.session_state.last_letter:
                    st.session_state.frame_count += 1
                else:
                    st.session_state.last_letter = letter
                    st.session_state.frame_count = 1

                # Commit letter when stable for N frames
                if st.session_state.frame_count >= hold_frames:
                    timestamp_str = time.strftime("%H:%M:%S")
                    st.session_state.sentence += letter
                    
                    st.session_state.history.append({
                        "letter": letter, 
                        "conf": confidence, 
                        "time": timestamp_str
                    })
                    
                    st.session_state.session_log.append({
                        "timestamp": timestamp_str,
                        "detected_letter": letter,
                        "confidence": round(confidence, 4),
                        "current_sentence": st.session_state.sentence
                    })
                    
                    st.session_state.frame_count = 0
            else:
                st.session_state.frame_count = 0

            render_prediction_card(letter, confidence, top_probs, hand_detected=True)
        else:
            st.session_state.frame_count = 0
            st.session_state.last_letter = None
            render_prediction_card(hand_detected=False)

            # Check > 5 seconds no hand timeout alert
            if time.time() - last_hand_time > 5.0:
                alert_placeholder.markdown(
                    '<div class="warning-tip">Tip: No hand detected for over 5 seconds. '
                    'Ensure your hand is clearly visible and well-lit inside the camera view.</div>',
                    unsafe_allow_html=True
                )

        # Update FPS display
        fps_placeholder.markdown(f'<div class="fps-badge">FPS: {fps_smooth:.1f}</div>', unsafe_allow_html=True)

        # Render RGB frame in Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Refresh dashboard displays
        update_sentence_display()
        update_history_display()

        # Yield execution to maintain UI responsiveness
        time.sleep(0.01)

    cap.release()
    video_placeholder.info("Camera stream stopped.")
