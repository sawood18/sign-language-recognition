# =============================================
# collect_motion.py
# Collects motion sign data for J and Z
# Run: python collect_motion.py
# =============================================

import cv2
import mediapipe as mp
import csv
import os
import time

# ----- paths -----
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

# ----- input -----
sign = input("Enter motion sign (J / Z): ").upper().strip()
if sign not in ["J", "Z"]:
    print("Only J or Z accepted.")
    exit()

filename = os.path.join(DATASET_DIR, sign + "_motion.csv")
print(f"\nCollecting MOTION data for: {sign}")
print("Each recording = 30 frames of movement.")
print("Press Q inside the camera window to stop.\n")

# ----- mediapipe -----
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

SEQUENCE_LEN  = 30          # must match app.py and train_motion_model.py
FEATURES_PER_FRAME = 63     # 21 landmarks × 3 (x,y,z)
EXPECTED_LEN  = SEQUENCE_LEN * FEATURES_PER_FRAME   # 1890

time.sleep(2)
print("Starting...")

saved_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        running = True
        while running:
            print(f"\nPerform the '{sign}' movement now!")
            sequence  = []
            quit_flag = False

            for frame_idx in range(SEQUENCE_LEN):
                ok, frame = cap.read()
                if not ok:
                    running = False
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                if res.multi_hand_landmarks:
                    hand  = res.multi_hand_landmarks[0]
                    wrist = hand.landmark[0]

                    # --- wrist-relative (NO normalisation for motion) ---
                    frame_row = []
                    for lm in hand.landmark:
                        frame_row.extend([
                            lm.x - wrist.x,
                            lm.y - wrist.y,
                            lm.z - wrist.z,
                        ])
                    sequence.extend(frame_row)   # accumulate 63 values per frame

                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                cv2.putText(frame, f"Sign: {sign}  frame {frame_idx+1}/{SEQUENCE_LEN}",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Saved sequences: {saved_count}",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Motion Collection - " + sign, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_flag = True
                    running   = False
                    break

            # --- save only if complete and hand was in ALL frames ---
            if len(sequence) == EXPECTED_LEN:
                writer.writerow(sequence)
                saved_count += 1
                print(f"  Saved! ({saved_count} total)")
            else:
                missing = SEQUENCE_LEN - (len(sequence) // FEATURES_PER_FRAME)
                print(f"  Incomplete ({missing} frames missing hand) — not saved.")

            if not quit_flag:
                time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
print(f"\nDone!  {saved_count} sequences saved to: {filename}")