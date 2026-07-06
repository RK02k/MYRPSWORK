# 🎮 Real-Time Gesture-Based Rock–Paper–Scissors using Computer Vision & LSTM

A real-time AI-powered Rock–Paper–Scissors system built using **Computer Vision**, **MediaPipe**, and **Long Short-Term Memory (LSTM)** networks. The system recognizes hand gestures from a webcam, predicts the player's move, and competes against the user in real time.

> This project demonstrates how temporal deep learning models can be combined with computer vision for Human–Computer Interaction (HCI), intelligent gaming, and future robotic control applications.

---

## 📌 Features

- ✋ Real-time hand tracking using MediaPipe
- 🎥 Live webcam gesture recognition
- 🧠 LSTM-based temporal sequence classification
- 🎮 AI opponent with game decision logic
- 📊 High gesture recognition accuracy (~96.67%)
- 🏗 Modular architecture for future expansion
- 🤖 Ready for robotic integration (myCobot / Arduino)

---

## 🏛 System Architecture

```
Webcam
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
21 Hand Landmarks
(x, y, z)
   │
   ▼
63 Features per Frame
   │
   ▼
30 Frame Sequence
   │
   ▼
LSTM Neural Network
   │
   ▼
Gesture Prediction
(Rock / Paper / Scissors)
   │
   ▼
Game AI Decision
   │
   ▼
Display Result
```

---

## 🧠 Technologies Used

| Category | Technology |
|-----------|------------|
| Programming | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Hardware (Optional) | Arduino / myCobot |

---

## 📂 Project Structure

```
MYRPSWORK/
│
├── data/                 # Collected gesture sequences
├── models/               # Trained LSTM model
├── notebooks/            # Training notebooks
├── utils/                # Helper functions
├── train.py              # Model training
├── collect_data.py       # Dataset collection
├── realtime.py           # Live prediction
├── game.py               # Rock-Paper-Scissors AI
├── requirements.txt
└── README.md
```

---
## 📊 Dataset

Three gesture classes are used:

- ✊ Rock
- ✋ Paper
- ✌️ Scissors

Each sample contains:

- 21 hand landmarks
- 3 coordinates (x,y,z)
- 63 features/frame
- 30 frames/sequence

Input Shape

```
(30,63)
```

Dataset Summary

| Property | Value |
|-----------|--------|
| Classes | 3 |
| Samples/Class | 300 |
| Total Samples | 900 |
| Train/Test Split | 80/20 |

---

## 🏗 Model Architecture

```
Input (30 × 63)

        │
        ▼

LSTM (64 Units)
(return_sequences=True)

        │
        ▼

LSTM (128 Units)

        │
        ▼

Dense (64)

        │
        ▼

Dense (3)
Softmax

        │
        ▼

Rock / Paper / Scissors

```

Training Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 50
- Framework: TensorFlow / Keras

---

## 📈 Performance

### Overall Performance

| Metric | Value |
|---------|--------|
| Accuracy | **96.67%** |
| Precision | **96.61%** |
| Recall | **96.90%** |
| Macro F1 Score | **96.62%** |
| Weighted F1 Score | **96.71%** |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Rock | 100% | 96.67% | 98.31% |
| Paper | 100% | 94.03% | 96.92% |
| Scissors | 89.83% | 100% | 94.64% |

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/RK02k/MYRPSWORK.git

cd MYRPSWORK
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Collect Dataset

```bash
python collect_data.py
```

### Train Model

```bash
python train.py
```

### Run Real-Time Prediction

```bash
python realtime.py
```

### Play Against AI

```bash
python game.py
```

---

## 🤖 Robotics Extension

The project is designed with future robotic integration in mind.

Predicted gestures can be translated into robotic commands for platforms such as:

- myCobot
- Arduino robotic hand
- ROS-based manipulators

This enables applications in:

- Human-Robot Interaction
- Gesture-controlled robots
- Intelligent automation
- Vision-Language-Action systems

---

**Raj Kashyap**

GitHub:
https://github.com/RK02k

---
