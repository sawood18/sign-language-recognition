# ============================================================
# SIGNIFY
# Real-Time ASL Sign Language Recognition
# ============================================================

import os
import math
import time
import threading
from collections import deque, Counter

import av
import cv2
import joblib
import numpy as np
import streamlit as st
import mediapipe as mp

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SIGNIFY",
    page_icon="🤟",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MIN_CONFIDENCE = 70.0
STABLE_FRAMES = 10
STABLE_MAJORITY = 7

LETTER_DELAY = 2.5

MOTION_LEN = 30
MIN_HAND_DIST = 0.04
NO_HAND_RESET = 5


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "sign_model.pkl"
)

MOTION_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "motion_model.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    errors = []

    if not os.path.exists(STATIC_MODEL):
        errors.append(
            "sign_model.pkl not found in models folder."
        )

    if not os.path.exists(MOTION_MODEL):
        errors.append(
            "motion_model.pkl not found in models folder."
        )

    if errors:
        return None, None, errors

    try:

        static_model = joblib.load(STATIC_MODEL)
        motion_model = joblib.load(MOTION_MODEL)

        return static_model, motion_model, []

    except Exception as e:

        return None, None, [str(e)]


sign_model, motion_model, model_errors = load_models()


# ============================================================
# MEDIAPIPE
# ============================================================

# IMPORTANT:
# requirements.txt MUST use mediapipe==0.10.21

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

# Start with STUN.
#
# Streamlit Community Cloud can require TURN depending on
# network conditions. The official streamlit-webrtc docs
# recommend a proper TURN service when STUN isn't enough.
#
# We intentionally do NOT use the unstable Open Relay server.

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        },
        {
            "urls": [
                "stun:stun1.l.google.com:19302"
            ]
        }
    ]
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_static_features(hand_landmarks):

    wrist = hand_landmarks.landmark[0]

    points = []

    for lm in hand_landmarks.landmark:

        points.append([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    max_dist = max(
        math.sqrt(
            p[0] ** 2 +
            p[1] ** 2 +
            p[2] ** 2
        )
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

    wrist = hand_landmarks.landmark[0]

    row = []

    for lm in hand_landmarks.landmark:

        row.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    return row


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.frame_count = 0

        self.history = deque(
            maxlen=STABLE_FRAMES
        )

        self.current_sign = "-"

        self.confidence = 0.0

        self.hand_visible = False

        self.sentence = ""

        self.last_letter_time = 0.0

        self.sign_locked = False

        self.motion_sequence = []

        self.no_hand_count = 0


    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        with self.lock:

            return {
                "sign": self.current_sign,
                "confidence": self.confidence,
                "sentence": self.sentence,
                "hand": self.hand_visible
            }


    # ========================================================
    # CLEAR
    # ========================================================

    def clear_text(self):

        with self.lock:
            self.sentence = ""

        self.reset_detection()


    # ========================================================
    # DELETE
    # ========================================================

    def delete_last(self):

        with self.lock:

            if self.sentence:

                self.sentence = self.sentence[:-1]

        self.sign_locked = False


    # ========================================================
    # SPACE
    # ========================================================

    def add_space(self):

        with self.lock:

            if (
                self.sentence
                and not self.sentence.endswith(" ")
            ):
                self.sentence += " "

        self.sign_locked = False

        self.history.clear()


    # ========================================================
    # RESET
    # ========================================================

    def reset_detection(self):

        self.history.clear()

        self.motion_sequence.clear()

        self.sign_locked = False


    # ========================================================
    # RECEIVE FRAME
    # ========================================================

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        img = cv2.flip(img, 1)

        # Resize
        img = cv2.resize(
            img,
            (640, 480)
        )

        self.frame_count += 1


        # Process every second frame
        if self.frame_count % 2 == 0:

            self.process_frame(img)


        # Draw overlay
        self.draw_overlay(img)


        # IMPORTANT:
        # Return the processed image
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, img):

        try:

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(rgb)

        except Exception:

            return


        # ====================================================
        # NO HAND
        # ====================================================

        if not results.multi_hand_landmarks:

            self.no_hand_count += 1

            if self.no_hand_count >= NO_HAND_RESET:

                self.history.clear()

                self.motion_sequence.clear()

                self.sign_locked = False

                with self.lock:

                    self.hand_visible = False

                    self.current_sign = "-"

                    self.confidence = 0.0

            return


        # ====================================================
        # HAND FOUND
        # ====================================================

        self.no_hand_count = 0

        hand = results.multi_hand_landmarks[0]


        with self.lock:

            self.hand_visible = True


        # ====================================================
        # DRAW HAND
        # ====================================================

        mp_draw.draw_landmarks(

            img,

            hand,

            mp_hands.HAND_CONNECTIONS,

            mp_style.get_default_hand_landmarks_style(),

            mp_style.get_default_hand_connections_style()
        )


        # ====================================================
        # STATIC FEATURES
        # ====================================================

        row = extract_static_features(hand)

        if row is None:

            return


        # ====================================================
        # MOTION FEATURES
        # ====================================================

        motion_row = extract_motion_row(hand)

        self.motion_sequence.append(
            motion_row
        )

        if len(self.motion_sequence) > MOTION_LEN:

            self.motion_sequence.pop(0)


        # ====================================================
        # STATIC MODEL
        # ====================================================

        prediction = ""

        confidence = 0.0


        if sign_model is not None:

            try:

                probabilities = (
                    sign_model.predict_proba(
                        [row]
                    )[0]
                )

                confidence = (
                    float(max(probabilities))
                    * 100.0
                )

                label = str(
                    sign_model.predict(
                        [row]
                    )[0]
                ).upper()

                if confidence >= MIN_CONFIDENCE:

                    prediction = label

            except Exception:

                pass


        # ====================================================
        # MOTION MODEL
        # ====================================================

        if (

            motion_model is not None

            and len(self.motion_sequence)
            == MOTION_LEN

            and self.frame_count % 6 == 0

        ):

            try:

                features = np.array(
                    self.motion_sequence,
                    dtype=np.float32
                ).reshape(1, -1)


                probabilities = (
                    motion_model.predict_proba(
                        features
                    )[0]
                )

                motion_confidence = (
                    float(max(probabilities))
                    * 100.0
                )

                motion_label = str(
                    motion_model.predict(
                        features
                    )[0]
                ).upper()


                if (

                    motion_label in ("J", "Z")

                    and motion_confidence >= 85.0

                ):

                    prediction = motion_label

                    confidence = motion_confidence


            except Exception:

                pass


        # ====================================================
        # STABILITY BUFFER
        # ====================================================

        if prediction:

            self.history.append(
                prediction
            )


        stable_sign = None


        if len(self.history) == STABLE_FRAMES:

            top_sign, top_count = (
                Counter(
                    self.history
                ).most_common(1)[0]
            )

            if top_count >= STABLE_MAJORITY:

                stable_sign = top_sign


        # ====================================================
        # UPDATE DISPLAY
        # ====================================================

        with self.lock:

            if stable_sign:

                self.current_sign = stable_sign

                self.confidence = confidence

            elif not prediction:

                self.current_sign = "..."

                self.confidence = 0.0


        # ====================================================
        # COMMIT LETTER
        # ====================================================

        now = time.time()


        if (

            stable_sign is not None

            and not self.sign_locked

            and (
                now - self.last_letter_time
            ) >= LETTER_DELAY

            and confidence >= MIN_CONFIDENCE

        ):

            with self.lock:

                self.sentence += stable_sign


            self.last_letter_time = now

            self.sign_locked = True

            self.history.clear()


            if stable_sign in ("J", "Z"):

                self.motion_sequence.clear()


    # ========================================================
    # OVERLAY
    # ========================================================

    def draw_overlay(self, img):

        with self.lock:

            sign = self.current_sign

            confidence = self.confidence

            hand = self.hand_visible


        if (

            hand

            and sign not in ("-", "...")

        ):

            text = (
                f"SIGN: {sign} "
                f"({confidence:.0f}%)"
            )

            color = (
                0,
                220,
                80
            )

        elif hand:

            text = "Detecting..."

            color = (
                0,
                200,
                255
            )

        else:

            text = "No hand detected"

            color = (
                0,
                80,
                255
            )


        # Shadow

        cv2.putText(

            img,

            text,

            (15, 42),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 0),

            4
        )


        # Main text

        cv2.putText(

            img,

            text,

            (15, 42),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            color,

            2
        )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

#MainMenu,
footer,
header {
    visibility: hidden;
}

.block-container {

    max-width: 1400px;

    padding-top: 12px;

    padding-bottom: 5px;
}

.app-title {

    font-size: 36px;

    font-weight: 900;
}

.app-sub {

    color: #94a3b8;

    font-size: 14px;
}

.sign-box {

    text-align: center;

    padding: 8px 0;
}

.sign-letter {

    font-size: 120px;

    font-weight: 900;

    line-height: 1;
}

.active {

    color: #22c55e;
}

.wait {

    color: #f59e0b;
}

.nohand {

    color: #475569;
}

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
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

h1, h2 = st.columns([7, 1])


with h1:

    st.markdown(
        '<div class="app-title">'
        '🤟 SIGNIFY'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-sub">'
        'Real-Time ASL Sign Language Recognition'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# MODEL ERROR
# ============================================================

if model_errors:

    for error in model_errors:

        st.error(
            f"⚠️ {error}"
        )

    st.stop()


# ============================================================
# MAIN COLUMNS
# ============================================================

camera_col, panel_col = st.columns(
    [1.1, 0.9],
    gap="large"
)


# ============================================================
# CAMERA
# ============================================================

with camera_col:

    st.subheader(
        "📷 Live Camera"
    )


    ctx = webrtc_streamer(

        key="signify-camera-v4",

        video_processor_factory=SignProcessor,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 640
                },

                "height": {
                    "ideal": 480
                },

                "frameRate": {
                    "ideal": 24
                },

            },

            "audio": False,

        },

        async_processing=True,

    )


# ============================================================
# RIGHT PANEL
# ============================================================

with panel_col:

    st.subheader(
        "🔎 Detected Sign"
    )


    processor = None


    if ctx is not None:

        processor = (
            ctx.video_processor
        )


    if processor is not None:

        state = (
            processor.get_state()
        )

        sign = state["sign"]

        confidence = (
            state["confidence"]
        )

        sentence = (
            state["sentence"]
        )

        hand = state["hand"]


    else:

        sign = "-"

        confidence = 0.0

        sentence = ""

        hand = False


    # ========================================================
    # BIG LETTER
    # ========================================================

    if (

        hand

        and sign not in ("-", "...")

    ):

        css_class = "active"

    elif hand:

        css_class = "wait"

    else:

        css_class = "nohand"


    st.markdown(

        f"""
        <div class="sign-box">
            <span class="sign-letter {css_class}">
                {sign}
            </span>
        </div>
        """,

        unsafe_allow_html=True
    )


    # ========================================================
    # STATUS
    # ========================================================

    if (

        hand

        and sign not in ("-", "...")

    ):

        st.success(
            f"✅ Recognised • "
            f"Confidence: {confidence:.1f}%"
        )

    elif hand:

        st.warning(
            "👋 Hand visible — "
            "hold sign steady..."
        )

    else:

        st.info(
            "👋 Click START CAMERA "
            "then show your hand"
        )


    st.markdown("---")


    # ========================================================
    # TEXT
    # ========================================================

    st.subheader(
        "📝 Recognised Text"
    )


    display_text = (
        sentence
        if sentence.strip()
        else "Your text will appear here..."
    )


    st.markdown(

        f"""
        <div class="sentence-area">
            {display_text}
        </div>
        """,

        unsafe_allow_html=True
    )


    st.write("")


    # ========================================================
    # BUTTONS
    # ========================================================

    c1, c2, c3 = st.columns(3)


    with c1:
        if st.button(
            "🗑️ Clear",
            use_container_width=True
        ):

            if processor:

                processor.clear_text()


    with c2:

        if st.button(
            "⌫ Backspace",
            use_container_width=True
        ):

            if processor:

                processor.delete_last()


    with c3:

        if st.button(
            "␣ Space",
            use_container_width=True
        ):

            if processor:

                processor.add_space()


    st.markdown("---")


    # ========================================================
    # HOW TO USE
    # ========================================================

    with st.expander(
        "ℹ️ How to use"
    ):

        st.markdown(
            f"""
**Steps**

1. Click **▶ START CAMERA**
2. Allow camera permission
3. Show one hand clearly
4. Hold the ASL sign steady
5. The letter is added automatically
6. Remove your hand before the next letter

**Accuracy**

- Minimum confidence: {MIN_CONFIDENCE}%
- Stability: {STABLE_FRAMES} frames
- Letter delay: {LETTER_DELAY} seconds

**Tips**

- Use good lighting
- Keep your complete hand visible
- Avoid very dark backgrounds
- Keep the camera stable
"""
        )


    st.caption(
        f"Confidence ≥ {MIN_CONFIDENCE}% • "
        f"Stability {STABLE_FRAMES} frames • "
        f"Delay {LETTER_DELAY}s"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Kolkar Osman • Sawood Salha • MJCET 🤟"
)