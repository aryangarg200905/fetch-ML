import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

def create_sequences(data, n_steps):
    sequences = []
    for i in range(len(data) - n_steps):
        sequences.append((data[i:i + n_steps], data[i + n_steps]))
    return np.array(sequences)

def build_and_train_model(data, n_steps=3, epochs=200):
    # Create sequences for time series model
    sequences = create_sequences(data, n_steps)

    # Split into training and target data
    X, y = zip(*sequences)
    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, verbose=0)

    return model
