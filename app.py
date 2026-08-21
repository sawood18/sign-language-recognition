# =============================================
# app.py  —  SIGNIFY
# Works on Streamlit Cloud + all devices
# Run: streamlit run app.py
# =============================================

import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import time
import os
import threading
from collections import deque, Counter
from PIL import Image

# =============================================
# PAGE CONFIG
# =============================================

st.set_page_config(
    page_title="SIGNIFY",
    page_icon="🤟",
    layout="wide",
)

# =============================================
# CONSTANTS
# =============================================

MIN_CONFIDENCE  = 70.0   # model must be this % sure
STABLE_FRAMES   = 8      # frames that must agree
STABLE_MAJORITY = 6      # how many must match
LETTER_DELAY    = 2.5    # seconds between letters
MOTION_LEN      = 30     # frames for J/Z (must match training)
MIN_HAND_DIST   = 0.04   # ignore tiny false-positive hands

# =============================================
# PATHS
# =============================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL = os.path.join(BASE_DIR, "models", "sign_model.pkl")
MOTION_MODEL = os.path.join(BASE_DIR, "models", "motion_model.pkl")

# =============================================
# LOAD MODELS
# =============================================

@st.cache_resource
def load_models():
    errors = []
    if not os.path.exists(STATIC_MODEL):
        errors.append(f"❌ sign_model.pkl not found → run `python train_model.py`")
    if not os.path.exists(MOTION_MODEL):
        errors.append(f"❌ motion_model.pkl not found → run `python train_model.py`")
    if errors:
        return None, None, errors
    try:
        sm = joblib.load(STATIC_MODEL)
        mm = joblib.load(MOTION_MODEL)
        return sm, mm, []
    except Exception as e:
        return None, None, [str(e)]

sign_model, motion_model, model_errors = load_models()

# =============================================
# MEDIAPIPE  (mp.solutions — needs mediapipe==0.10.9)
# =============================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# =============================================
# SESSION STATE  — persistent across reruns
# =============================================

def init_state():
    defaults = {
        "sentence":         "",
        "last_sign":        "-",
        "last_conf":        0.0,
        "last_letter_time": 0.0,
        "sign_locked":      False,
        "history":          deque(maxlen=STABLE_FRAMES),
        "motion_seq":       [],
        "no_hand_count":    0,
        "mode":             "camera",   # "camera" or "upload"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================
# FEATURE EXTRACTION
# Must be IDENTICAL to collect_data.py
# =============================================

@st.cache_resource
def get_hands_detector():
    return mp_hands.Hands(
        static_image_mode=True,    # True for per-image processing
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

def extract_static_features(hand_landmarks):
    """63 normalised floats — same pipeline as collect_data.py"""
    wrist = hand_landmarks.landmark[0]
    points = []
    for lm in hand_landmarks.landmark:
        points.append([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    max_dist = max(
        math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        for p in points
    )
    if max_dist < MIN_HAND_DIST:
        return None
    row = []
    for p in points:
        row.append(p[0] / max_dist)
        row.append(p[1] / max_dist)
        row.append(p[2] / max_dist)
    return row   # 63 values

def extract_motion_row(hand_landmarks):
    """63 wrist-relative floats — same pipeline as collect_motion.py"""
    wrist = hand_landmarks.landmark[0]
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return row   # 63 values

def process_frame(img_bgr, hands_detector):
    """
    Process one BGR frame.
    Returns (annotated_img, sign, confidence, hand_found)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    prediction = ""
    confidence = 0.0
    hand_found = False

    if results.multi_hand_landmarks:
        hand_found = True
        hand = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_draw.draw_landmarks(
            img_bgr, hand, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        # Static features
        row = extract_static_features(hand)
        if row and sign_model:
            try:
                proba      = sign_model.predict_proba([row])[0]
                confidence = float(max(proba)) * 100.0
                label      = str(sign_model.predict([row])[0]).upper()
                if confidence >= MIN_CONFIDENCE:
                    prediction = label
            except Exception:
                pass

        # Motion features (J/Z)
        motion_row = extract_motion_row(hand)
        st.session_state["motion_seq"].append(motion_row)
        if len(st.session_state["motion_seq"]) > MOTION_LEN:
            st.session_state["motion_seq"].pop(0)

        if (
            motion_model
            and len(st.session_state["motion_seq"]) == MOTION_LEN
        ):
            try:
                feat    = np.array(
                    st.session_state["motion_seq"],
                    dtype=np.float32
                ).reshape(1, -1)
                m_proba = motion_model.predict_proba(feat)[0]
                m_conf  = float(max(m_proba)) * 100.0
                m_label = str(motion_model.predict(feat)[0]).upper()
                if m_label in ("J", "Z") and m_conf >= 85.0:
                    prediction = m_label
                    confidence = m_conf
            except Exception:
                pass

        # Stability buffer
        if prediction:
            st.session_state["history"].append(prediction)

        stable_sign = None
        hist = st.session_state["history"]
        if len(hist) >= STABLE_FRAMES:
            top_sign, top_count = Counter(hist).most_common(1)[0]
            if top_count >= STABLE_MAJORITY:
                stable_sign = top_sign

        if stable_sign:
            st.session_state["last_sign"] = stable_sign
            st.session_state["last_conf"] = confidence

            # Commit letter
            now = time.time()
            if (
                not st.session_state["sign_locked"]
                and (now - st.session_state["last_letter_time"]) >= LETTER_DELAY
                and confidence >= MIN_CONFIDENCE
            ):
                st.session_state["sentence"]         += stable_sign
                st.session_state["last_letter_time"]  = now
                st.session_state["sign_locked"]       = True
                st.session_state["history"]           = deque(maxlen=STABLE_FRAMES)
                if stable_sign in ("J", "Z"):
                    st.session_state["motion_seq"] = []

        prediction = st.session_state["last_sign"]
        confidence = st.session_state["last_conf"]

    else:
        # No hand
        st.session_state["no_hand_count"] += 1
        if st.session_state["no_hand_count"] >= 3:
            st.session_state["history"]      = deque(maxlen=STABLE_FRAMES)
            st.session_state["motion_seq"]   = []
            st.session_state["sign_locked"]  = False
            st.session_state["last_sign"]    = "-"
            st.session_state["last_conf"]    = 0.0
            st.session_state["no_hand_count"] = 0

    # Overlay text on frame
    if hand_found and prediction not in ("-", ""):
        text  = f"SIGN: {prediction}  ({confidence:.0f}%)"
        color = (0, 220, 80)
    elif hand_found:
        text  = "Detecting..."
        color = (0, 200, 255)
    else:
        text  = "No hand detected"
        color = (0, 80, 255)

    cv2.putText(img_bgr, text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img_bgr, text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return img_bgr, prediction, confidence, hand_found

# =============================================
# CSS
# =============================================

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1400px;
    padding-top: 12px;
    padding-bottom: 5px;
}
.app-title {
    font-size: 36px;
    font-weight: 900;
    letter-spacing: -1px;
}
.app-sub {
    color: #94a3b8;
    font-size: 14px;
    margin-top: -4px;
}
.sign-box {
    text-align: center;
    padding: 8px 0 4px 0;
}
.sign-letter {
    font-size: 120px;
    font-weight: 900;
    line-height: 1.0;
    display: block;
}
.sign-letter.active { color: #22c55e; }
.sign-letter.wait   { color: #f59e0b; }
.sign-letter.nohand { color: #475569; }
.sentence-area {
    background: #0f172a;
    border: 1.5px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
    min-height: 68px;
    font-size: 22px;
    color: #f1f5f9;
    font-family: monospace;
    letter-spacing: 2px;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================

h1, h2 = st.columns([7, 1])
with h1:
    st.markdown('<div class="app-title">🤟 SIGNIFY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Real-Time ASL Sign Language Recognition</div>',
        unsafe_allow_html=True,
    )

st.divider()

# =============================================
# MODEL ERROR
# =============================================

if model_errors:
    for e in model_errors:
        st.error(e)
    st.info("Run `python train_model.py` first, then restart the app.")
    st.stop()

# =============================================
# MAIN LAYOUT
# =============================================

cam_col, panel_col = st.columns([1.1, 0.9], gap="large")

# =============================================
# LEFT: CAMERA via st.camera_input
# Works on ALL devices and Streamlit Cloud
# No WebRTC, no STUN/TURN needed
# =============================================

with cam_col:
    st.subheader("📷 Live Camera")
    st.caption("Click **Take Photo** → hold your sign → it will be recognised instantly.")

    # st.camera_input works on every browser and Streamlit Cloud
    camera_image = st.camera_input(
        label="camera",
        label_visibility="collapsed",
    )

    hands_detector = get_hands_detector()

    if camera_image is not None:
        # Convert uploaded snapshot to numpy BGR
        pil_img  = Image.open(camera_image)
        img_rgb  = np.array(pil_img)
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Process frame
        annotated, sign, conf, hand = process_frame(img_bgr, hands_detector)

        # Show annotated frame
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    else:
        st.info("👆 Click the camera button above to take a photo of your sign.")

# =============================================
# RIGHT: Info panel
# =============================================

with panel_col:
    st.subheader("🔎 Detected Sign")

    sign = st.session_state["last_sign"]
    conf = st.session_state["last_conf"]
    sent = st.session_state["sentence"]

    # Big letter display
    if sign not in ("-", "...","") and conf >= MIN_CONFIDENCE:
        css_class = "active"
    elif camera_image is not None:
        css_class = "wait"
    else:
        css_class = "nohand"

    st.markdown(
        f'<div class="sign-box">'
        f'<span class="sign-letter {css_class}">{sign}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Status
    if sign not in ("-", "...", "") and conf >= MIN_CONFIDENCE:
        st.success(f"✅ Recognised  •  Confidence: {conf:.1f}%")
    elif camera_image is not None:
        st.warning("👋 Hand detected — hold sign steady and retake")
    else:
        st.info("📸 Take a photo to begin")

    st.markdown("---")

    # Sentence
    st.subheader("📝 Recognised Text")
    display = sent if sent.strip() else "Your text will appear here..."
    st.markdown(
        f'<div class="sentence-area">{display}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["sentence"]      = ""
            st.session_state["history"]       = deque(maxlen=STABLE_FRAMES)
            st.session_state["motion_seq"]    = []
            st.session_state["sign_locked"]   = False
            st.session_state["last_sign"]     = "-"
            st.session_state["last_conf"]     = 0.0
            st.rerun()

    with c2:
        if st.button("⌫ Backspace", use_container_width=True):
            if st.session_state["sentence"]:
                st.session_state["sentence"] = st.session_state["sentence"][:-1]
            st.session_state["sign_locked"] = False
            st.rerun()

    with c3:
        if st.button("␣  Space", use_container_width=True):
            s = st.session_state["sentence"]
            if s and not s.endswith(" "):
                st.session_state["sentence"] += " "
            st.session_state["sign_locked"] = False
            st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ How to use"):
        st.markdown("""
**How to sign a letter:**
1. Position your hand clearly in the camera frame
2. Make the ASL sign
3. Click **Take Photo** (the camera button)
4. The letter appears automatically if confidence ≥ 70%
5. Remove your hand, then repeat for the next letter

**Tips:**
- Good lighting = better accuracy
- Keep hand fully in frame, avoid busy backgrounds
- Hold sign steady when taking the photo
- Use **Backspace** to fix mistakes

**Buttons:**
- 🗑️ **Clear** — erase everything
- ⌫ **Backspace** — delete last letter
- ␣ **Space** — add space between words
        """)

    st.caption(
        f"Min confidence: {MIN_CONFIDENCE}% • "
        f"Stability: {STABLE_FRAMES} frames • "
        f"Letter delay: {LETTER_DELAY}s"
    )

# =============================================
# FOOTER
# =============================================

st.divider()
st.caption("Kolkar Osman • Sawood Salha • MJCET  🤟")