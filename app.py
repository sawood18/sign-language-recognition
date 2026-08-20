import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import time
import threading

from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SIGNIFY",
    page_icon="🤟",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    sign_model = joblib.load(
        "models/sign_model.pkl"
    )

    motion_model = joblib.load(
        "models/motion_model.pkl"
    )

    return sign_model, motion_model


sign_model, motion_model = load_models()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# PROCESSOR
# ============================================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame_count = 0

        # Stable prediction
        self.history = deque(maxlen=5)

        # FIX 2: motion_length must match training (30 frames)
        self.motion_sequence = []
        self.motion_length = 30

        # Current sign
        self.current_sign = "-"
        self.confidence = 0.0
        self.hand_detected = False

        # Sentence
        self.sentence = ""

        # Timing
        self.last_letter_time = 0
        self.letter_delay = 1.0

        # Prevent one held sign from repeating
        self.sign_locked = False

    # ========================================================
    # GET CURRENT STATE
    # ========================================================

    def get_state(self):

        with self.lock:

            return {
                "sign": self.current_sign,
                "confidence": self.confidence,
                "sentence": self.sentence,
                "hand": self.hand_detected
            }

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_text(self):

        with self.lock:

            self.sentence = ""

        self.history.clear()
        self.sign_locked = False

    # ========================================================
    # BACKSPACE
    # ========================================================

    def delete_last(self):

        with self.lock:

            self.sentence = self.sentence[:-1]

        self.sign_locked = False

    # ========================================================
    # SPACE
    # ========================================================

    def add_space(self):

        with self.lock:

            # Don't add multiple spaces
            if self.sentence and not self.sentence.endswith(" "):

                self.sentence += " "

        self.sign_locked = False

        self.history.clear()

    # ========================================================
    # VIDEO
    # ========================================================

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror
        img = cv2.flip(
            img,
            1
        )

        # Smaller processing size
        img = cv2.resize(
            img,
            (640, 480),
            interpolation=cv2.INTER_AREA
        )

        self.frame_count += 1

        # ====================================================
        # PROCESS EVERY 2ND FRAME
        # ====================================================

        if self.frame_count % 2 == 0:

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(
                rgb
            )

            prediction = ""

            # =================================================
            # HAND DETECTED
            # =================================================

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                self.hand_detected = True

                # Draw landmarks
                mp_draw.draw_landmarks(
                    img,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                # =============================================
                # EXACT FEATURES USED DURING TRAINING
                # =============================================

                wrist = hand.landmark[0]

                points = []

                for landmark in hand.landmark:

                    x = landmark.x - wrist.x
                    y = landmark.y - wrist.y
                    z = landmark.z - wrist.z

                    points.append(
                        [x, y, z]
                    )

                # =============================================
                # MAX DISTANCE
                # =============================================

                max_distance = 0.0

                for point in points:

                    distance = math.sqrt(
                        point[0] ** 2 +
                        point[1] ** 2 +
                        point[2] ** 2
                    )

                    if distance > max_distance:

                        max_distance = distance

                # =============================================
                # NORMALIZE
                # =============================================

                if max_distance > 0:

                    row = []

                    for point in points:

                        row.append(
                            point[0] / max_distance
                        )

                        row.append(
                            point[1] / max_distance
                        )

                        row.append(
                            point[2] / max_distance
                        )

                    # =========================================
                    # STATIC MODEL
                    # =========================================

                    try:

                        prediction = str(
                            sign_model.predict(
                                [row]
                            )[0]
                        ).upper()

                        # Confidence
                        try:

                            probabilities = (
                                sign_model.predict_proba(
                                    [row]
                                )[0]
                            )

                            self.confidence = (
                                max(probabilities) * 100
                            )

                        except Exception:

                            self.confidence = 0.0

                    except Exception:

                        prediction = ""

                # =============================================
                # FIX 1: MOTION SEQUENCE — wrist-relative coords
                # (must match collect_motion.py exactly)
                # =============================================

                motion_points = []

                for landmark in hand.landmark:

                    motion_points.extend([
                        landmark.x - wrist.x,   # relative x
                        landmark.y - wrist.y,   # relative y
                        landmark.z - wrist.z    # relative z
                    ])

                self.motion_sequence.append(
                    motion_points
                )

                if len(self.motion_sequence) > self.motion_length:

                    self.motion_sequence.pop(0)

                # =============================================
                # J / Z  — check every 6 processed frames
                # =============================================

                if (
                    len(self.motion_sequence)
                    == self.motion_length
                    and self.frame_count % 6 == 0
                ):

                    try:

                        motion_features = np.array(
                            self.motion_sequence,
                            dtype=np.float32
                        ).reshape(
                            1,
                            -1
                        )

                        motion_prediction = str(
                            motion_model.predict(
                                motion_features
                            )[0]
                        ).upper()

                        if motion_prediction in ["J", "Z"]:

                            prediction = motion_prediction

                            self.confidence = 100.0

                    except Exception:

                        pass

                # =============================================
                # STABLE PREDICTION
                # =============================================

                if prediction:

                    self.history.append(
                        prediction
                    )

                if len(self.history) >= 4:

                    stable_sign, count = Counter(
                        self.history
                    ).most_common(1)[0]

                    with self.lock:

                        self.current_sign = stable_sign

                    # =========================================
                    # ADD LETTER
                    # =========================================

                    now = time.time()

                    if (
                        count >= 3
                        and not self.sign_locked
                        and now - self.last_letter_time
                        >= self.letter_delay
                    ):

                        with self.lock:

                            self.sentence += stable_sign

                        self.last_letter_time = now

                        self.sign_locked = True

                        self.history.clear()

            # =================================================
            # NO HAND
            # =================================================

            else:

                self.hand_detected = False

                self.motion_sequence.clear()

                self.history.clear()

                with self.lock:

                    self.current_sign = "-"
                    self.confidence = 0.0

                self.sign_locked = False

        # ====================================================
        # CAMERA ONLY SHOWS SIGN
        # ====================================================

        with self.lock:

            display_sign = self.current_sign

        cv2.putText(
            img,
            "SIGN: " + display_sign,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 100),
            2
        )

        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# SIMPLE UI
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 15px;
        padding-bottom: 10px;
    }

    .title {
        font-size: 32px;
        font-weight: 800;
    }

    .subtitle {
        color: #718096;
        font-size: 13px;
    }

    .big-sign {
        font-size: 100px;
        font-weight: 800;
        text-align: center;
        margin: 10px;
    }

    .sentence-box {
        min-height: 80px;
        padding: 15px;
        border: 1px solid #475569;
        border-radius: 10px;
        font-size: 24px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

head1, head2 = st.columns(
    [5, 1]
)

with head1:

    st.markdown(
        '<div class="title">🤟 SIGNIFY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Real-Time Sign Language Recognition'
        '</div>',
        unsafe_allow_html=True
    )

with head2:

    st.button(
        "ⓘ About",
        use_container_width=True
    )


st.divider()


# ============================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================

camera_col, right_col = st.columns(
    [1, 1],
    gap="medium"
)


# ============================================================
# CAMERA — LEFT HALF
# ============================================================

with camera_col:

    st.subheader("🟢 LIVE CAMERA")

    ctx = webrtc_streamer(
        key="signify-camera",

        video_processor_factory=SignProcessor,

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
                }
            },
            "audio": False
        },

        async_processing=True
    )


# ============================================================
# RIGHT HALF
# ============================================================

with right_col:

    st.subheader("🔎 DETECTED SIGN")

    # --------------------------------------------------------
    # Get processor
    # --------------------------------------------------------

    processor = ctx.video_processor

    if processor:

        state = processor.get_state()

        sign = state["sign"]
        confidence = state["confidence"]
        sentence = state["sentence"]
        hand = state["hand"]

    else:

        sign = "-"
        confidence = 0.0
        sentence = ""
        hand = False

    # --------------------------------------------------------
    # Detected sign
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="big-sign">
            {sign}
        </div>
        """,
        unsafe_allow_html=True
    )

    if hand:

        st.success(
            f"Recognition Active   •   "
            f"Confidence: {confidence:.1f}%"
        )

    else:

        st.info(
            "Show your hand to the camera"
        )

    # --------------------------------------------------------
    # Recognized text
    # --------------------------------------------------------

    st.subheader("📝 RECOGNIZED TEXT")

    if sentence:

        st.text_area(
            "Sentence",
            value=sentence,
            height=100,
            label_visibility="collapsed",
            disabled=True
        )

    else:

        st.text_area(
            "Sentence",
            value="Your sentence will appear here...",
            height=100,
            label_visibility="collapsed",
            disabled=True
        )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    b1, b2 = st.columns(2)

    with b1:

        if st.button(
            "🗑️ Clear",
            use_container_width=True
        ):

            if processor:

                processor.clear_text()

    with b2:

        if st.button(
            "⌫ Backspace",
            use_container_width=True
        ):

            if processor:

                processor.delete_last()

    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if st.button(
        "SPACE",
        use_container_width=True
    ):

        if processor:

            processor.add_space()

    # --------------------------------------------------------
    # FIX 4: Auto-refresh so sign/sentence updates live
    # --------------------------------------------------------

    if processor:

        time.sleep(0.5)
        st.rerun()

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    st.caption(
        "A–Y Static Recognition  •  J/Z Motion Recognition"
    )

    st.caption(
        "Remove your hand between repeated letters."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Kolkar Osman • Sawood Salha • MJCET"
)