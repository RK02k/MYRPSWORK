import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data():
    X = np.load("data/processed/gesture_data.npy")
    y = np.load("data/processed/gesture_labels.npy")

    return X, y


def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():
    X, y = load_data()

    # convert labels → one-hot
    y = to_categorical(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model((X.shape[1], X.shape[2]))

    model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_test, y_test)
    )

    model.save("models/gesture_model.h5")

    print("✅ Model trained & saved!")


if __name__ == "__main__":
    train_model()