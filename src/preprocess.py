import numpy as np

def combine_datasets():
    rock_data = np.load("data/raw/rock_data.npy")
    rock_labels = np.load("data/raw/rock_labels.npy")

    paper_data = np.load("data/raw/paper_data.npy")
    paper_labels = np.load("data/raw/paper_labels.npy")

    scissors_data = np.load("data/raw/scissors_data.npy")
    scissors_labels = np.load("data/raw/scissors_labels.npy")

    data = np.concatenate((rock_data, paper_data, scissors_data))
    labels = np.concatenate((rock_labels, paper_labels, scissors_labels))

    np.save("data/processed/gesture_data.npy", data)
    np.save("data/processed/gesture_labels.npy", labels)