import numpy as np

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        data = []
        for lm in hand.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return np.array(data)

    return np.zeros(63)