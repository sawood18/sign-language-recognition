# =============================================
# app.py  —  SIGNIFY
# Real-Time ASL Sign Language Recognition
# Run: streamlit run app.py
# =============================================

import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import time
import threading
import os

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
# CONSTANTS  — tune here if needed
# =============================================

MIN_CONFIDENCE  = 70.0   # % — model must be this sure before showing a sign
STABLE_FRAMES   = 10     # number of frames that must agree on same sign
STABLE_MAJORITY = 7      # how many of those frames must match
LETTER_DELAY    = 2.2    # seconds between auto-adding letters
NO_HAND_RESET   = 5      # consecutive empty frames before full reset
MOTION_LEN      = 30     # frames per motion sequence (must match training)
MIN_HAND_DIST   = 0.04   # ignore tiny "hands" (false positives)

# =============================================
# PATHS
# =============================================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL  = os.path.join(BASE_DIR, "models", "sign_model.pkl")
MOTION_MODEL  = os.path.join(BASE_DIR, "models", "motion_model.pkl")

# =============================================
# LOAD MODELS  (only once)
# =============================================

@st.cache_resource
def load_models():
    errors = []

    if not os.path.exists(STATIC_MODEL):
        errors.append(f"sign_model.pkl not found at: {STATIC_MODEL}")
    if not os.path.exists(MOTION_MODEL):
        errors.append(f"motion_model.pkl not found at: {MOTION_MODEL}")

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
# MEDIAPIPE  —  mp.solutions (requires mediapipe==0.10.9)
# =============================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# =============================================
# FEATURE EXTRACTION
# Identical pipeline to collect_data.py
# =============================================

def extract_static_features(hand_landmarks):
    """
    Returns list of 63 normalised floats, or None if hand too small.
    Pipeline:
        1. Make all landmarks relative to wrist (landmark[0])
        2. Divide by max distance from wrist  (scale invariant)
    """
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
        return None, 0.0   # hand too small — likely false positive

    row = []
    for p in points:
        row.append(p[0] / max_dist)
        row.append(p[1] / max_dist)
        row.append(p[2] / max_dist)

    return row, max_dist


def extract_motion_row(hand_landmarks):
    """
    Returns 63-float wrist-relative row (NOT normalised by max_dist).
    Identical to collect_motion.py.
    """
    wrist = hand_landmarks.landmark[0]
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return row   # 63 values


# =============================================
# VIDEO PROCESSOR
# =============================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()

        # MediaPipe
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

        # Frame counter
        self._fc = 0

        # ----- prediction state -----
        self._history       = deque(maxlen=STABLE_FRAMES)
        self._current_sign  = "-"
        self._confidence    = 0.0
        self._hand_visible  = False

        # ----- sentence state -----
        self._sentence         = ""
        self._last_letter_time = 0.0
        self._sign_locked      = False

        # ----- motion (J / Z) -----
        self._motion_seq = []   # list of 63-float rows

        # ----- no-hand tracking -----
        self._no_hand_count = 0

    # ------------------------------------------
    # Public API (called from Streamlit thread)
    # ------------------------------------------

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
    # Frame processing
    # ------------------------------------------

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        self._fc += 1

        # Process every 2nd frame (performance)
        if self._fc % 2 == 0:
            self._process(img)

        self._draw_overlay(img)
        return frame.from_ndarray(img, format="bgr24")

    def _process(self, img):
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        # ==========================================
        # NO HAND
        # ==========================================
        if not results.multi_hand_landmarks:
            self._no_hand_count += 1

            if self._no_hand_count >= NO_HAND_RESET:
                # Full reset — hand is truly gone
                self._history.clear()
                self._motion_seq.clear()
                self._sign_locked = False
                with self._lock:
                    self._hand_visible = False
                    self._current_sign = "-"
                    self._confidence   = 0.0
            return

        # ==========================================
        # HAND DETECTED
        # ==========================================
        self._no_hand_count = 0
        hand = results.multi_hand_landmarks[0]

        # Draw hand skeleton on the frame
        mp_draw.draw_landmarks(
            img, hand, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        with self._lock:
            self._hand_visible = True

        # ---- static features ----
        row, max_dist = extract_static_features(hand)
        if row is None:
            # Hand too small / false positive — ignore
            return

        # ---- motion row (always collect) ----
        motion_row = extract_motion_row(hand)
        self._motion_seq.append(motion_row)
        if len(self._motion_seq) > MOTION_LEN:
            self._motion_seq.pop(0)

        # ==========================================
        # STATIC PREDICTION  (A-Y)
        # ==========================================
        prediction = ""
        confidence = 0.0

        if sign_model is not None:
            try:
                proba      = sign_model.predict_proba([row])[0]
                confidence = float(max(proba)) * 100.0
                label      = str(sign_model.predict([row])[0]).upper()

                # Only accept if model is confident enough
                if confidence >= MIN_CONFIDENCE:
                    prediction = label

            except Exception:
                pass

        # ==========================================
        # MOTION PREDICTION  (J / Z)
        # Run every 6 processed frames once buffer full
        # ==========================================
        if (
            motion_model is not None
            and len(self._motion_seq) == MOTION_LEN
            and self._fc % 6 == 0
        ):
            try:
                feat = np.array(
                    self._motion_seq, dtype=np.float32
                ).reshape(1, -1)

                m_proba = motion_model.predict_proba(feat)[0]
                m_conf  = float(max(m_proba)) * 100.0
                m_label = str(motion_model.predict(feat)[0]).upper()

                # Override static prediction ONLY if very confident
                if m_label in ("J", "Z") and m_conf >= 85.0:
                    prediction = m_label
                    confidence = m_conf

            except Exception:
                pass

        # ==========================================
        # STABILITY BUFFER
        # Require majority agreement over N frames
        # ==========================================
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

        # ==========================================
        # COMMIT LETTER
        # Only when:
        #   1. hand IS visible  (not a false positive)
        #   2. sign is stable across N frames
        #   3. confidence is high enough
        #   4. sign is not locked (user hasn't removed hand yet)
        #   5. enough time has passed since last letter
        # ==========================================
        now = time.time()

        if (
            stable_sign is not None
            and not self._sign_locked
            and (now - self._last_letter_time) >= LETTER_DELAY
            and confidence >= MIN_CONFIDENCE
        ):
            with self._lock:
                self._sentence += stable_sign

            self._last_letter_time = now
            self._sign_locked      = True   # unlock when hand removed
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
            color = (0, 230, 80)
        elif hand:
            text  = "Detecting..."
            color = (0, 200, 255)
        else:
            text  = "No hand detected"
            color = (0, 80, 255)

        cv2.putText(img, text, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)   # shadow
        cv2.putText(img, text, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


# =============================================
# CSS
# =============================================

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1400px; padding-top: 12px; padding-bottom: 5px; }

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

/* Big sign letter */
.sign-box {
    text-align: center;
    padding: 10px 0 5px 0;
}
.sign-letter {
    font-size: 130px;
    font-weight: 900;
    line-height: 1.0;
    display: block;
}
.sign-letter.active  { color: #22c55e; }
.sign-letter.wait    { color: #f59e0b; }
.sign-letter.nohand  { color: #475569; }

/* Sentence box */
.sentence-area {
    background: #0f172a;
    border: 1.5px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
    min-height: 70px;
    font-size: 22px;
    color: #f1f5f9;
    font-family: monospace;
    letter-spacing: 2px;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# SHOW MODEL ERRORS
# =============================================

if model_errors:
    st.error("⚠️ Model Error\n\n" + "\n".join(model_errors))
    st.info("Run `python train_model.py` first to create the models.")
    st.stop()

# =============================================
# HEADER
# =============================================

h1, h2 = st.columns([7, 1])
with h1:
    st.markdown('<div class="app-title">🤟 SIGNIFY</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Real-Time ASL Sign Language Recognition</div>',
                unsafe_allow_html=True)
with h2:
    st.write("")

st.divider()

# =============================================
# MAIN LAYOUT
# =============================================

cam_col, panel_col = st.columns([1.1, 0.9], gap="large")

# ---- LEFT: Camera --------------------------------
with cam_col:
    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="signify-v3",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 640},
                "height":    {"ideal": 480},
                "frameRate": {"ideal": 24},
            },
            "audio": False,
        },
        async_processing=True,
    )

# ---- RIGHT: Info panel ---------------------------
with panel_col:
    st.subheader("🔎 Detected Sign")

    processor = ctx.video_processor if ctx else None

    if processor:
        state      = processor.get_state()
        sign       = state["sign"]
        confidence = state["confidence"]
        sentence   = state["sentence"]
        hand       = state["hand"]
    else:
        sign, confidence, sentence, hand = "-", 0.0, "", False

    # -- big letter --
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

    # -- status bar --
    if hand and sign not in ("-", "..."):
        st.success(f"✅ Recognised  •  Confidence: {confidence:.1f}%")
    elif hand:
        st.warning("👋 Hand visible — stabilising...")
    else:
        st.info("👋 Show your hand to the camera")

    st.markdown("---")

    # -- sentence --
    st.subheader("📝 Recognised Text")

    display_text = sentence if sentence.strip() else "Your text will appear here..."
    st.markdown(
        f'<div class="sentence-area">{display_text}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # -- controls --
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

    # -- tips --
    with st.expander("ℹ️ How to use"):
        st.markdown("""
**Static signs (A–Y):**
- Hold the sign steady until the letter appears
- Remove your hand fully between letters

**Motion signs (J, Z):**
- Perform the J or Z stroke naturally
- The motion model detects it automatically

**Controls:**
- **Clear** — erase all text
- **Backspace** — delete last letter
- **Space** — add a space between words

**Tips for accuracy:**
- Good lighting helps a lot
- Keep your hand fully in frame
- Avoid busy backgrounds
""")

    st.caption(
        f"Min confidence: {MIN_CONFIDENCE}% • "
        f"Stability: {STABLE_FRAMES} frames • "
        f"Delay: {LETTER_DELAY}s"
    )

    # -- auto-refresh so UI stays live --
    if processor:
        time.sleep(0.35)
        st.rerun()

# =============================================
# FOOTER
# =============================================

st.divider()
st.caption("Kolkar Osman • Sawood Salha • MJCET  🤟")