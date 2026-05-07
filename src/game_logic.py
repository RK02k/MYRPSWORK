import random
from collections import Counter

def predict_player(history):
    if len(history) < 3:
        return random.choice([0, 1, 2])

    most_common = Counter(history).most_common(1)[0][0]

    return (most_common + 1) % 3


def get_winner(player, robot):
    if player == robot:
        return "Draw"

    if (player == 0 and robot == 2) or \
       (player == 1 and robot == 0) or \
       (player == 2 and robot == 1):
        return "Player Wins"

    return "Robot Wins"