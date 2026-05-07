import cv2
import numpy as np
import time
import math
import mediapipe as mp
from tensorflow.keras.models import load_model

try:
    from pymycobot.mycobot import MyCobot
except ImportError:
    MyCobot = None

from .utils import extract_landmarks
from .game_logic import get_winner

# model = load_model("models/gesture_model.h5", compile=False)

def run_game():
    model = load_model("models/gesture_model.h5",compile=False)

    # Initialize MyCobot
    mc = None
    if MyCobot is not None:
        try:
            mc = MyCobot('/dev/ttyUSB0', 115200)
            print("MyCobot initialized.")
        except Exception as e:
            print(f"MyCobot not found or error: {e}")

    # MyCobot angle configurations for [Rock, Paper, Scissors]
    mc_angles = {
        0: [32.12, -7.47, -52.38, -20.74, 4.81, 75.43],    # Rock (Stone)
        1: [-17.88, -7.47, -52.38, -20.74, 4.81, 180],     # Paper
        2: [32.12, -7.47, -52.38, -20.74, -90.19, 135]     # Scissors
    }

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands()

    sequence = []
    round_duration = 3
    result_duration = 3
    round_start = time.time()
    state = "countdown"
    
    player_score = 0
    robot_score = 0
    rounds_played = 0
    
    player_move_text = ""
    robot_move_text = ""
    result_text = ""
    locked = False
    robotMove = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        landmarks = extract_landmarks(results)
        sequence.append(landmarks)
        sequence = sequence[-30:]

        current_time = time.time()
        time_elapsed = current_time - round_start

        if state == "countdown":
            phase = time_elapsed / 0.75
            phase_idx = int(phase)
            texts = ["Rock...", "Paper...", "Scissors...", "Shoot!"]
            current_text = texts[min(phase_idx, 3)]
            
            bounce = math.sin(math.pi * (time_elapsed % 0.75) / 0.75)
            y_offset = int(50 * bounce)
            
            cv2.putText(frame, current_text, (50, 100 + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
            if time_elapsed >= round_duration:
                import random
                robotMove = random.randint(0, 2)
                robot_move_text = ["Rock", "Paper", "Scissors"][robotMove]
                
                if mc:
                    try:
                        mc.send_angles(mc_angles[robotMove], 50)
                    except Exception as e:
                        print(f"Error moving MyCobot: {e}")

                player_move_text = "Detecting..."
                result_text = "Waiting..."
                locked = False
                
                state = "result"
                round_start = current_time

        elif state == "result":
            if not locked:
                instant_sequence = [landmarks] * 30
                is_hand_present = np.any(np.array(landmarks) != 0)
                
                if is_hand_present:
                    prediction = model.predict(np.expand_dims(instant_sequence, axis=0), verbose=0)
                    confidence = np.max(prediction)
                    
                    if confidence >= 0.7:
                        playerMove = np.argmax(prediction)
                        moves = ["Rock", "Paper", "Scissors"]
                        player_move_text = moves[playerMove]
                        result_text = get_winner(playerMove, robotMove)
                        locked = True
                        
                        if result_text == "Player Wins":
                            player_score += 1
                        elif result_text == "Robot Wins":
                            robot_score += 1
                        rounds_played += 1
                    else:
                        player_move_text = "Invalid"
                        result_text = "Waiting for valid gesture..."
                else:
                    player_move_text = "No Hand"
                    result_text = "Show Hand"

            cv2.putText(frame, f"Player: {player_move_text}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Robot: {robot_move_text}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, result_text, (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            
            if time_elapsed >= result_duration:
                if player_score >= 3 or robot_score >= 3:
                    state = "game_over"
                else:
                    state = "countdown"
                round_start = current_time
                sequence = []

        elif state == "game_over":
            if player_score > robot_score:
                final_text = "Overall Winner: Player!"
                color = (0, 255, 0)
            elif robot_score > player_score:
                final_text = "Overall Winner: Robot!"
                color = (0, 0, 255)
            else:
                final_text = "Overall Result: Draw!"
                color = (255, 0, 0)
                
            cv2.putText(frame, final_text, (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            if time_elapsed >= 5:
                state = "countdown"
                round_start = current_time
                sequence = []
                player_score = 0
                robot_score = 0
                rounds_played = 0

        display_round = rounds_played
        if state == "countdown":
            display_round += 1
        elif state == "result" and not locked:
            display_round += 1

        cv2.putText(frame, f"Score: Player {player_score} - {robot_score} Robot", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Round {display_round} (First to 3)", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("Game", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()