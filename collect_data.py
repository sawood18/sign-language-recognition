import cv2
import mediapipe as mp
import csv
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

sign = input("Enter sign name (A/B/C/D/E): ").upper().strip()
filename = os.path.join(DATASET_DIR, sign + ".csv")

print("Collecting:", sign)
print("Show the sign and move your hand around.")
print("Press Q to stop.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                # Wrist is the reference point
                wrist = hand.landmark[0]

                points = []

                # Make coordinates relative to wrist
                for landmark in hand.landmark:
                    x = landmark.x - wrist.x
                    y = landmark.y - wrist.y
                    z = landmark.z - wrist.z

                    points.append([x, y, z])

                # Find maximum distance from wrist
                max_distance = 0

                for point in points:
                    distance = math.sqrt(
                        point[0] ** 2 +
                        point[1] ** 2 +
                        point[2] ** 2
                    )

                    max_distance = max(
                        max_distance,
                        distance
                    )

                # Normalize hand size
                if max_distance > 0:

                    row = []

                    for point in points:
                        row.extend([
                            point[0] / max_distance,
                            point[1] / max_distance,
                            point[2] / max_distance
                        ])

                    writer.writerow(row)

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    "Collecting: " + sign,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Show your hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Improved Sign Data Collection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()

print()
print("Data collection completed!")
print("Saved to:", filename)
