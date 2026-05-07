import cv2
import numpy as np
import mediapipe as mp
from .utils import extract_landmarks

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def collect_data(label, gesture_name, save_prefix, target_samples=300):
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    sequence_length = 30
    data, labels = [], []
    sequence = []
    sample_count = 0

    cap = cv2.VideoCapture(0)

    print(f"Recording {gesture_name}")

    while sample_count < target_samples:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

        landmarks = extract_landmarks(results)
        sequence.append(landmarks)

        if len(sequence) == sequence_length:
            data.append(sequence)
            labels.append(label)
            sequence = []
            sample_count += 1

        cv2.imshow("Dataset Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    np.save(f"data/raw/{save_prefix}_data.npy", data)
    np.save(f"data/raw/{save_prefix}_labels.npy", labels)