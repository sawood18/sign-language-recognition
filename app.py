# =============================================
# app.py  —  SIGNIFY
# Optimised for speed + accuracy
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

st.set_page_config(page_title="SIGNIFY", page_icon="🤟", layout="wide")

# =============================================
# TUNING  — every value explained
# =============================================

MIN_CONFIDENCE  = 60.0  # lower = faster response, raise if wrong letters appear
STABLE_FRAMES   = 6     # fewer frames = faster detection (was 10 = too slow)
STABLE_MAJORITY = 5     # must agree out of STABLE_FRAMES
LETTER_DELAY    = 2.0   # seconds before next letter allowed
MOTION_LEN      = 30    # must match collect_motion.py
NO_HAND_RESET   = 4     # empty frames before reset (was 5)
MIN_HAND_DIST   = 0.03  # minimum hand size to accept

# =============================================
# MODELS
# =============================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL = os.path.join(BASE_DIR, "models", "sign_model.pkl")
MOTION_MODEL = os.path.join(BASE_DIR, "models", "motion_model.pkl")

@st.cache_resource
def load_models():
    errors = []
    if not os.path.exists(STATIC_MODEL):
        errors.append("sign_model.pkl not found — run: python train_model.py")
    if not os.path.exists(MOTION_MODEL):
        errors.append("motion_model.pkl not found — run: python train_model.py")
    if errors:
        return None, None, errors
    try:
        return joblib.load(STATIC_MODEL), joblib.load(MOTION_MODEL), []
    except Exception as e:
        return None, None, [str(e)]

sign_model, motion_model, model_errors = load_models()

# =============================================
# MEDIAPIPE  — requires mediapipe==0.10.9
# =============================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# =============================================
# TURN SERVERS  — needed for cloud + all devices
# =============================================

RTC_CONFIG = RTCConfiguration({"iceServers": [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun.relay.metered.ca:80"]},
    {"urls": ["turn:openrelay.metered.ca:80"],
     "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": ["turn:openrelay.metered.ca:443"],
     "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
     "username": "openrelayproject", "credential": "openrelayproject"},
]})

# =============================================
# FEATURE EXTRACTION — identical to training
# =============================================

def extract_static_features(hand_landmarks):
    wrist  = hand_landmarks.landmark[0]
    points = [[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
              for lm in hand_landmarks.landmark]
    max_dist = max(math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in points)
    if max_dist < MIN_HAND_DIST:
        return None
    row = []
    for p in points:
        row += [p[0]/max_dist, p[1]/max_dist, p[2]/max_dist]
    return row   # 63 values

def extract_motion_row(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    row   = []
    for lm in hand_landmarks.landmark:
        row += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
    return row   # 63 values

# =============================================
# VIDEO PROCESSOR
# =============================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()

        # FIX 1: model_complexity=0 = much faster, same accuracy for hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
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

        # FIX 2: cache last prediction so overlay never shows stale ""
        self._last_prediction = "-"
        self._last_conf       = 0.0

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
        self._history.clear()
        self._motion_seq.clear()
        self._sign_locked = False

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

    # ------------------------------------------
    # RECV — called for every frame from browser
    # ------------------------------------------

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # FIX 3: process at 320x240 (4x fewer pixels = 4x faster)
        # then upscale back to display at 640x480
        small = cv2.resize(img, (320, 240))

        self._fc += 1

        # FIX 4: process every 3rd frame — reduces CPU load by 66%
        # At 20fps input → ~7 detections/sec, plenty for sign language
        if self._fc % 3 == 0:
            self._process(small, img)

        self._draw_overlay(img)
        return frame.from_ndarray(img, format="bgr24")

    def _process(self, small, img):
        # FIX 5: set writeable=False before mediapipe — skips memory copy
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        # ---- NO HAND ----
        if not results.multi_hand_landmarks:
            self._no_hand_count += 1
            if self._no_hand_count >= NO_HAND_RESET:
                self._history.clear()
                self._motion_seq.clear()
                self._sign_locked = False
                with self._lock:
                    self._hand_visible  = False
                    self._current_sign  = "-"
                    self._confidence    = 0.0
            return

        # ---- HAND FOUND ----
        self._no_hand_count = 0
        hand = results.multi_hand_landmarks[0]

        # FIX 6: draw on FULL img not small — landmarks are normalised 0-1
        # so they scale correctly regardless of image size
        mp_draw.draw_landmarks(
            img, hand, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        with self._lock:
            self._hand_visible = True

        row = extract_static_features(hand)
        if row is None:
            return

        # Motion buffer
        self._motion_seq.append(extract_motion_row(hand))
        if len(self._motion_seq) > MOTION_LEN:
            self._motion_seq.pop(0)

        # ---- STATIC PREDICTION (A-Y) ----
        prediction = ""
        confidence = 0.0

        if sign_model:
            try:
                # FIX 7: run predict and predict_proba in ONE call
                # predict_proba gives us both label index and confidence
                proba      = sign_model.predict_proba([row])[0]
                idx        = int(np.argmax(proba))
                confidence = float(proba[idx]) * 100.0
                label      = str(sign_model.classes_[idx]).upper()
                if confidence >= MIN_CONFIDENCE:
                    prediction = label
            except Exception:
                pass

        # ---- MOTION (J/Z) — every 9th frame (3rd processed) ----
        if (
            motion_model
            and len(self._motion_seq) == MOTION_LEN
            and self._fc % 9 == 0
        ):
            try:
                feat    = np.array(self._motion_seq, dtype=np.float32).reshape(1, -1)
                m_proba = motion_model.predict_proba(feat)[0]
                m_idx   = int(np.argmax(m_proba))
                m_conf  = float(m_proba[m_idx]) * 100.0
                m_label = str(motion_model.classes_[m_idx]).upper()
                if m_label in ("J", "Z") and m_conf >= 80.0:
                    prediction = m_label
                    confidence = m_conf
            except Exception:
                pass

        # ---- STABILITY ----
        if prediction:
            self._history.append(prediction)

        stable_sign = None
        if len(self._history) >= STABLE_FRAMES:
            top_sign, top_count = Counter(self._history).most_common(1)[0]
            if top_count >= STABLE_MAJORITY:
                stable_sign = top_sign

        with self._lock:
            if stable_sign:
                self._current_sign = stable_sign
                self._confidence   = confidence
            elif prediction:
                # show live prediction even before stable
                self._current_sign = prediction
                self._confidence   = confidence
            else:
                self._current_sign = "..."
                self._confidence   = 0.0

        # ---- COMMIT LETTER ----
        now = time.time()
        if (
            stable_sign
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

        # Background bar for readability
        cv2.rectangle(img, (0, 0), (640, 55), (0, 0, 0), -1)

        if hand and sign not in ("-", "..."):
            text  = f"SIGN: {sign}   {conf:.0f}%"
            color = (0, 230, 80)
        elif hand:
            text  = "Detecting..."
            color = (0, 200, 255)
        else:
            text  = "Show your hand"
            color = (80, 80, 255)

        cv2.putText(img, text, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

# =============================================
# CSS
# =============================================

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1400px; padding-top: 10px; padding-bottom: 0; }
.app-title { font-size: 34px; font-weight: 900; letter-spacing: -1px; }
.app-sub   { color: #94a3b8; font-size: 13px; margin-top: -4px; }
.sign-box  { text-align: center; padding: 4px 0; }
.sign-letter {
    font-size: 110px; font-weight: 900;
    line-height: 1.0; display: block;
}
.sign-letter.active { color: #22c55e; }
.sign-letter.wait   { color: #f59e0b; }
.sign-letter.nohand { color: #475569; }
.sentence-area {
    background: #0f172a;
    border: 1.5px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    min-height: 64px;
    font-size: 22px;
    color: #f1f5f9;
    font-family: monospace;
    letter-spacing: 2px;
    word-break: break-all;
}
/* confidence progress bar */
.conf-bar-wrap {
    background: #1e293b;
    border-radius: 6px;
    height: 10px;
    margin: 6px 0 10px 0;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #f59e0b, #22c55e);
    transition: width 0.3s;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================

h1, _ = st.columns([7, 1])
with h1:
    st.markdown('<div class="app-title">🤟 SIGNIFY</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Real-Time ASL Sign Language Recognition</div>',
                unsafe_allow_html=True)
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

cam_col, panel_col = st.columns([1.15, 0.85], gap="large")

# ---- CAMERA ----
with cam_col:
    st.subheader("📷 Live Camera")
    ctx = webrtc_streamer(
        key="signify-v5",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                # FIX 8: lower resolution from browser = less data to transmit
                # = less lag over WebRTC
                "width":     {"ideal": 640, "max": 640},
                "height":    {"ideal": 480, "max": 480},
                "frameRate": {"ideal": 20, "max": 20},
            },
            "audio": False,
        },
        async_processing=True,
    )

# ---- PANEL ----
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
    css = "active" if (hand and sign not in ("-","...")) else ("wait" if hand else "nohand")
    st.markdown(
        f'<div class="sign-box"><span class="sign-letter {css}">{sign}</span></div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    bar_w = int(conf) if hand else 0
    bar_color = "#22c55e" if conf >= MIN_CONFIDENCE else "#f59e0b"
    st.markdown(
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{bar_w}%;background:{bar_color}"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Status
    if hand and sign not in ("-", "..."):
        st.success(f"✅ Recognised  •  {conf:.1f}% confidence")
    elif hand:
        st.warning("👋 Hand visible — hold sign steady")
    else:
        st.info("👋 Click START then show your hand")

    st.markdown("---")

    # Sentence
    st.subheader("📝 Recognised Text")
    display = sent if sent.strip() else "Your text will appear here..."
    st.markdown(f'<div class="sentence-area">{display}</div>', unsafe_allow_html=True)
    st.write("")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            if processor: processor.clear_text()
    with c2:
        if st.button("⌫ Backspace", use_container_width=True):
            if processor: processor.delete_last()
    with c3:
        if st.button("␣ Space", use_container_width=True):
            if processor: processor.add_space()

    st.markdown("---")

    with st.expander("ℹ️ How to use"):
        st.markdown(f"""
**Steps:**
1. Click **START** button inside the camera box
2. Allow browser camera permission
3. Hold your ASL sign in frame — letter appears at {MIN_CONFIDENCE:.0f}%+ confidence
4. Remove hand fully between letters (unlocks next letter)

**Controls:** Clear / Backspace / Space buttons above

**Tips:**
- Good lighting = biggest accuracy boost
- Plain wall behind hand helps
- Hold sign still for {LETTER_DELAY:.0f} seconds
- Hand should fill roughly half the frame
        """)

    st.caption(f"Threshold: {MIN_CONFIDENCE}% • Stability: {STABLE_FRAMES} frames • Delay: {LETTER_DELAY}s")

    # FIX 9: rerun only when camera is active, shorter sleep = snappier UI
    if processor:
        time.sleep(0.15)
        st.rerun()

st.divider()
st.caption("Kolkar Osman • Sawood Salha • MJCET  🤟")