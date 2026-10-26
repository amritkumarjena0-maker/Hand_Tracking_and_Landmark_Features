
import csv
import math
import time
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "hand_gesture_dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

DATASET_FILE = DATASET_DIR / "landmark_samples.csv"
LOG_FILE = OUTPUT_DIR / "tracking_log.csv"

for folder in (DATASET_DIR, MODELS_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

LANDMARK_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

DATASET_COLUMNS = [
    "timestamp",
    "gesture",
    "handedness",
    *LANDMARK_COLUMNS,
]

LOG_COLUMNS = [
    "timestamp",
    "handedness",
    "gesture",
    "landmark_count",
    "fps",
]


# --------------------------------------------------
# CSV HELPERS
# --------------------------------------------------

def prepare_csv(path, columns):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)


def append_csv(path, row):
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


prepare_csv(DATASET_FILE, DATASET_COLUMNS)
prepare_csv(LOG_FILE, LOG_COLUMNS)


# --------------------------------------------------
# LANDMARK FEATURE EXTRACTION
# --------------------------------------------------

def extract_landmarks(hand_landmarks):
    """
    Return normalized x, y, z coordinates for 21 landmarks.
    Each hand produces 63 numerical features.
    """
    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # Make coordinates relative to the wrist.
    points -= points[0]

    # Normalize by the largest absolute coordinate.
    scale = np.max(np.abs(points))

    if scale > 1e-8:
        points /= scale

    return points.flatten().tolist()


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def get_finger_states(hand_landmarks):
    """
    Estimate whether each finger is extended.
    This uses landmark geometry and is a heuristic.
    """
    lm = hand_landmarks.landmark

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    states = []

    # Fingers other than the thumb:
    # a lower image y-coordinate generally means a higher fingertip.
    for tip, pip in zip(finger_tips, finger_pips):
        states.append(lm[tip].y < lm[pip].y)

    # Thumb extension is estimated from distance to the index MCP.
    thumb_tip = np.array([lm[4].x, lm[4].y])
    thumb_joint = np.array([lm[2].x, lm[2].y])
    index_mcp = np.array([lm[5].x, lm[5].y])

    thumb_extended = (
        np.linalg.norm(thumb_tip - index_mcp)
        > np.linalg.norm(thumb_joint - index_mcp) * 1.15
    )

    return [thumb_extended] + states


def recognize_gesture(hand_landmarks):
    """Recognize basic gestures using rule-based finger states."""
    states = get_finger_states(hand_landmarks)
    thumb, index, middle, ring, little = states

    if not any(states):
        return "FIST"

    if all(states):
        return "OPEN PALM"

    if index and not middle and not ring and not little:
        return "POINTING"

    if index and middle and not ring and not little:
        return "PEACE"

    if thumb and not index and not middle and not ring and not little:
        return "THUMBS UP"

    return "CUSTOM"


# --------------------------------------------------
# WEBCAM APPLICATION
# --------------------------------------------------

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check that your webcam is connected and not being used")
        print("by another application.")
        return

    previous_time = time.perf_counter()
    last_log_time = 0.0

    print("\nHAND TRACKING AND LANDMARK FEATURES")
    print("-----------------------------------")
    print("S : Save the current hand sample")
    print("Q : Quit the application")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Log:     {LOG_FILE}\n")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:

            while True:
                success, frame = camera.read()

                if not success:
                    print("WARNING: Could not read a webcam frame.")
                    break

                # Mirror the frame for a natural webcam experience.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                results = hands.process(rgb_frame)

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                detected_hands = []

                if results.multi_hand_landmarks:
                    for hand_index, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        features = extract_landmarks(hand_landmarks)
                        gesture = recognize_gesture(hand_landmarks)

                        handedness = "Unknown"
                        if (
                            results.multi_handedness
                            and hand_index < len(results.multi_handedness)
                        ):
                            handedness = (
                                results.multi_handedness[hand_index]
                                .classification[0].label
                            )

                        detected_hands.append(
                            {
                                "features": features,
                                "gesture": gesture,
                                "handedness": handedness,
                            }
                        )

                        # Display label and gesture.
                        wrist = hand_landmarks.landmark[0]
                        height, width = frame.shape[:2]
                        x = int(wrist.x * width)
                        y = int(wrist.y * height)

                        cv2.putText(
                            frame,
                            f"{handedness}: {gesture}",
                            (max(10, x - 40), max(30, y - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                # Write a tracking record at most once per second.
                if current_time - last_log_time >= 1.0:
                    if detected_hands:
                        for hand in detected_hands:
                            append_csv(
                                LOG_FILE,
                                [
                                    datetime.now().isoformat(
                                        timespec="seconds"
                                    ),
                                    hand["handedness"],
                                    hand["gesture"],
                                    21,
                                    round(fps, 2),
                                ],
                            )
                    else:
                        append_csv(
                            LOG_FILE,
                            [
                                datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "None",
                                "NO HAND",
                                0,
                                round(fps, 2),
                            ],
                        )

                    last_log_time = current_time

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "S: Save Sample | Q: Quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Hand Tracking and Landmark Features", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    if detected_hands:
                        for hand in detected_hands:
                            append_csv(
                                DATASET_FILE,
                                [
                                    datetime.now().isoformat(
                                        timespec="seconds"
                                    ),
                                    hand["gesture"],
                                    hand["handedness"],
                                    *hand["features"],
                                ],
                            )

                        print(
                            f"Saved {len(detected_hands)} hand sample(s)."
                        )
                    else:
                        print("No hand detected. Nothing was saved.")

                elif key == ord("q"):
                    break

    except Exception as error:
        print(f"Application error: {error}")
        raise

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nHand tracking application closed.")


if __name__ == "__main__":
    main()
