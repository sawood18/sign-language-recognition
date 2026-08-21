# =============================================
# app.py  —  SIGNIFY
# Live camera with WebRTC + free TURN servers
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

MIN_CONFIDENCE  = 70.0
STABLE_FRAMES   = 10
STABLE_MAJORITY = 7
LETTER_DELAY    = 2.5
MOTION_LEN      = 30
MIN_HAND_DIST   = 0.04
NO_HAND_RESET   = 5

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
        errors.append(f"sign_model.pkl not found — run `python train_model.py`")
    if not os.path.exists(MOTION_MODEL):
        errors.append(f"motion_model.pkl not found — run `python train_model.py`")
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
# MEDIAPIPE  (requires mediapipe==0.10.9)
# =============================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# =============================================
# FREE TURN SERVERS  — fixes blank camera on
# Streamlit Cloud and all devices behind NAT
# These are from open-relay.metered.ca (free)
# =============================================

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        # Google STUN
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        # Metered free STUN
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        # Metered free TURN (UDP port 80 — works through most firewalls)
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        # Metered free TURN (TCP port 443 — bypasses strict firewalls)
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        # Metered free TURN over TLS (most strict networks)
        {
            "urls": ["turns:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

# =============================================
# FEATURE EXTRACTION
# Identical to collect_data.py and collect_motion.py
# =============================================

def extract_static_features(hand_landmarks):
    """Returns 63 normalised floats or None."""
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
    return row

def extract_motion_row(hand_landmarks):
    """Returns 63 wrist-relative floats."""
    wrist = hand_landmarks.landmark[0]
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return row

# =============================================
# VIDEO PROCESSOR
# =============================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()

        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

        self._fc            = 0
        self._history       = deque(maxlen=STABLE_FRAMES)
        self._current_sign  = "-"
        self._confidence    = 0.0
        self._hand_visible  = False
        self._sentence      = ""
        self._last_time     = 0.0
        self._sign_locked   = False
        self._motion_seq    = []
        self._no_hand_count = 0

    # ---- public API ----

    def get_state(self):
        with self._lock:
            return {
                "sign":       self._current_sign,
                "confidence": self._confidence,
                "sentence":   self._sentence,
                "hand":       self._hand_visible,
            }

    def clear_text(self):
        with self._lock:
            self._sentence = ""
        self._reset_detection()

    def delete_last(self):
        with self._lock:
            if self._sentence:
                self._sentence = self._sentence[:-1]
        self._sign_locked = False

    def add_space(self):
        with self._lock:
            if self._sentence and not self._sentence.endswith(" "):
                self._sentence += " "
        self._sign_locked = False
        self._history.clear()

    def _reset_detection(self):
        self._history.clear()
        self._motion_seq.clear()
        self._sign_locked = False

    # ---- frame processing ----

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        self._fc += 1

        # Process every 2nd frame for performance
        if self._fc % 2 == 0:
            self._process(img)

        self._draw_overlay(img)
        return frame.from_ndarray(img, format="bgr24")

    def _process(self, img):
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        # ---- NO HAND ----
        if not results.multi_hand_landmarks:
            self._no_hand_count += 1
            if self._no_hand_count >= NO_HAND_RESET:
                self._history.clear()
                self._motion_seq.clear()
                self._sign_locked = False
                with self._lock:
                    self._hand_visible = False
                    self._current_sign = "-"
                    self._confidence   = 0.0
            return

        # ---- HAND DETECTED ----
        self._no_hand_count = 0
        hand = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_draw.draw_landmarks(
            img, hand, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        with self._lock:
            self._hand_visible = True

        # Static features
        row = extract_static_features(hand)
        if row is None:
            return   # hand too small / false positive

        # Motion features
        motion_row = extract_motion_row(hand)
        self._motion_seq.append(motion_row)
        if len(self._motion_seq) > MOTION_LEN:
            self._motion_seq.pop(0)

        # ---- STATIC PREDICTION (A-Y) ----
        prediction = ""
        confidence = 0.0

        if sign_model:
            try:
                proba      = sign_model.predict_proba([row])[0]
                confidence = float(max(proba)) * 100.0
                label      = str(sign_model.predict([row])[0]).upper()
                if confidence >= MIN_CONFIDENCE:
                    prediction = label
            except Exception:
                pass

        # ---- MOTION PREDICTION (J/Z) ----
        if (
            motion_model
            and len(self._motion_seq) == MOTION_LEN
            and self._fc % 6 == 0
        ):
            try:
                feat    = np.array(self._motion_seq, dtype=np.float32).reshape(1, -1)
                m_proba = motion_model.predict_proba(feat)[0]
                m_conf  = float(max(m_proba)) * 100.0
                m_label = str(motion_model.predict(feat)[0]).upper()
                if m_label in ("J", "Z") and m_conf >= 85.0:
                    prediction = m_label
                    confidence = m_conf
            except Exception:
                pass

        # ---- STABILITY BUFFER ----
        if prediction:
            self._history.append(prediction)

        stable_sign = None
        if len(self._history) == STABLE_FRAMES:
            top_sign, top_count = Counter(self._history).most_common(1)[0]
            if top_count >= STABLE_MAJORITY:
                stable_sign = top_sign

        with self._lock:
            if stable_sign:
                self._current_sign = stable_sign
                self._confidence   = confidence
            elif not prediction:
                self._current_sign = "..."
                self._confidence   = 0.0

        # ---- COMMIT LETTER ----
        now = time.time()
        if (
            stable_sign is not None
            and not self._sign_locked
            and (now - self._last_time) >= LETTER_DELAY
            and confidence >= MIN_CONFIDENCE
        ):
            with self._lock:
                self._sentence += stable_sign
            self._last_time   = now
            self._sign_locked = True
            self._history.clear()
            if stable_sign in ("J", "Z"):
                self._motion_seq.clear()

    def _draw_overlay(self, img):
        with self._lock:
            sign = self._current_sign
            conf = self._confidence
            hand = self._hand_visible

        if hand and sign not in ("-", "..."):
            text  = f"SIGN: {sign}  ({conf:.0f}%)"
            color = (0, 220, 80)
        elif hand:
            text  = "Detecting..."
            color = (0, 200, 255)
        else:
            text  = "No hand detected"
            color = (0, 80, 255)

        # Shadow + text for readability
        cv2.putText(img, text, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(img, text, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

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
.sign-box { text-align: center; padding: 8px 0 4px 0; }
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
# MODEL ERRORS
# =============================================

if model_errors:
    for e in model_errors:
        st.error(f"⚠️ {e}")
    st.info("Run `python train_model.py` first, then restart the app.")
    st.stop()

# =============================================
# MAIN LAYOUT
# =============================================

cam_col, panel_col = st.columns([1.1, 0.9], gap="large")

# ---- LEFT: Live Camera ----
with cam_col:
    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="signify-live",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 640},
                "height":    {"ideal": 480},
                "frameRate": {"ideal": 24},
            },
            "audio": False,
        },
        async_processing=True,
        translations={
            "start": "▶ Start Camera",
            "stop":  "⏹ Stop Camera",
            "select_device": "Select Camera",
        },
    )

# ---- RIGHT: Info panel ----
with panel_col:
    st.subheader("🔎 Detected Sign")

    processor = ctx.video_processor if ctx else None

    if processor:
        state = processor.get_state()
        sign  = state["sign"]
        conf  = state["confidence"]
        sent  = state["sentence"]
        hand  = state["hand"]
    else:
        sign, conf, sent, hand = "-", 0.0, "", False

    # Big letter
    if hand and sign not in ("-", "..."):
        css_class = "active"
    elif hand:
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
    if hand and sign not in ("-", "..."):
        st.success(f"✅ Recognised  •  Confidence: {conf:.1f}%")
    elif hand:
        st.warning("👋 Hand visible — hold sign steady...")
    else:
        st.info("👋 Click START CAMERA then show your hand")

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
            if processor:
                processor.clear_text()
    with c2:
        if st.button("⌫ Backspace", use_container_width=True):
            if processor:
                processor.delete_last()
    with c3:
        if st.button("␣  Space", use_container_width=True):
            if processor:
                processor.add_space()

    st.markdown("---")

    with st.expander("ℹ️ How to use"):
        st.markdown(f"""
**Steps:**
1. Click **▶ Start Camera**
2. Allow browser camera permission
3. Hold an ASL sign clearly in frame
4. Letter appears when confidence ≥ {MIN_CONFIDENCE}%
5. Remove hand between letters to unlock next letter

**Controls:**
- 🗑️ **Clear** — erase all text
- ⌫ **Backspace** — delete last letter
- ␣ **Space** — add space between words

**Tips for accuracy:**
- Good lighting is important
- Keep hand fully in frame
- Avoid busy/dark backgrounds
- Hold sign still for ~{LETTER_DELAY} seconds
        """)

    st.caption(
        f"Min confidence: {MIN_CONFIDENCE}% • "
        f"Stability: {STABLE_FRAMES} frames • "
        f"Letter delay: {LETTER_DELAY}s"
    )

    # Auto-refresh
    if processor:
        time.sleep(0.35)
        st.rerun()

# =============================================
# FOOTER
# =============================================

st.divider()
st.caption("Kolkar Osman • Sawood Salha • MJCET  🤟")